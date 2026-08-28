"""向量索引服务模块"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, Optional

from loguru import logger

from app.config import config
from app.services.document_splitter_service import document_splitter_service
from app.services.es_store_manager import es_store_manager
from app.services.index_task_store import IndexTaskStore
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """索引结果类"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        """增加成功计数"""
        self.success_count += 1

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """添加失败文件"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """获取耗时（毫秒）"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(self):
        """初始化向量索引服务"""
        self.upload_path = "./uploads"
        self.task_store = IndexTaskStore()
        self._retry_stop = Event()
        self._retry_thread: Thread | None = None
        logger.info("向量索引服务初始化完成")

    def start_retry_worker(self) -> None:
        """启动持久化索引失败队列的后台自动重试 worker。"""
        if not config.index_retry_enabled or self._retry_thread is not None:
            return
        self._retry_stop.clear()
        self._retry_thread = Thread(target=self._retry_loop, name="index-retry-worker", daemon=True)
        self._retry_thread.start()
        logger.info("索引失败队列自动重试 worker 已启动")

    def stop_retry_worker(self) -> None:
        """停止后台重试 worker。"""
        self._retry_stop.set()
        if self._retry_thread is not None:
            self._retry_thread.join(timeout=5)
            self._retry_thread = None

    def _retry_loop(self) -> None:
        while not self._retry_stop.wait(config.index_retry_poll_seconds):
            for task in self.task_store.list({"failed", "partial_success"}):
                if self._retry_stop.is_set():
                    return
                if int(task.get("retry_count", 0)) >= config.index_retry_max_attempts:
                    continue
                next_retry_at = task.get("next_retry_at")
                if next_retry_at and datetime.fromisoformat(next_retry_at).timestamp() > datetime.now().timestamp():
                    continue
                try:
                    logger.info(f"自动重试索引任务: {task['task_id']}")
                    self.retry_task(task["task_id"])
                except Exception as exc:
                    logger.warning(f"自动重试索引任务失败: {task['task_id']}, 错误: {exc}")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """
        索引指定目录下的所有文件

        Args:
            directory_path: 目录路径（可选，默认使用配置的上传目录）

        Returns:
            IndexingResult: 索引结果
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 获取所有支持的文件
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历并索引每个文件
            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(
        self,
        file_path: str,
        staged_file_path: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        索引单个文件 (使用新的 LangChain 分割器)

        Args:
            file_path: 文件路径

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引失败时抛出
        """
        path = Path(file_path).resolve()
        read_path = Path(staged_file_path).resolve() if staged_file_path else path

        if not read_path.exists() or not read_path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")
        normalized_path = path.as_posix()
        stable_task_id = task_id or str(uuid.uuid4())
        index_version = str(uuid.uuid4())
        task = self.task_store.get(stable_task_id)
        if task is None:
            task = self.task_store.create(
                file_path=normalized_path,
                staged_file_path=str(read_path) if staged_file_path else None,
                version=index_version,
                task_id=stable_task_id,
            )
        else:
            self.task_store.update(
                stable_task_id,
                index_version=index_version,
                staged_file_path=str(read_path) if staged_file_path else task.get("staged_file_path"),
            )
        self.task_store.update(stable_task_id, status="indexing", error=None, next_retry_at=None)
        vector_written = False
        es_written = False
        chunk_ids: list[str] = []

        try:
            # 1. 读取文件内容
            content = read_path.read_text(encoding="utf-8")
            logger.info(f"读取文件: {read_path}, 内容长度: {len(content)} 字符")

            # 2. 先生成带版本号的候选数据，旧版本保持可用
            documents = document_splitter_service.split_document(content, normalized_path)
            for document in documents:
                document.metadata["_index_version"] = index_version
                document.metadata["_index_task_id"] = stable_task_id
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 3. 双写候选版本；任一侧失败时保留旧版本
            if documents:
                chunk_ids = [str(uuid.uuid4()) for _ in documents]
                # add_documents 可能在底层部分写入后抛错，因此调用前就标记为需清理。
                vector_written = True
                vector_store_manager.add_documents(documents, chunk_ids)
                self.task_store.update(stable_task_id, status="partial_success", vector_chunk_ids=chunk_ids)
                es_store_manager.add_documents(documents, chunk_ids)
                es_written = True
                self.task_store.update(stable_task_id, status="partial_success", es_chunk_ids=chunk_ids)
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

            # 4. 双写成功后再删除旧版本（空文件也要清理旧版本）
            vector_store_manager.delete_old_versions(normalized_path, index_version)
            es_store_manager.delete_old_versions(normalized_path, index_version)

            # 5. 文件索引成功后才替换正式文件
            if staged_file_path:
                Path(read_path).parent.mkdir(parents=True, exist_ok=True)
                os.replace(read_path, path)

            self.task_store.update(stable_task_id, status="success", vector_chunk_ids=chunk_ids,
                                   es_chunk_ids=chunk_ids)
            logger.info(f"文件索引完成: {file_path}, 共 {len(documents)} 个分片, 版本={index_version}")
            return self.task_store.get(stable_task_id) or {}

        except Exception as e:
            self.task_store.update(stable_task_id, status="partial_success" if vector_written else "failed",
                                   error=str(e), vector_chunk_ids=chunk_ids)
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            # 记录每个清理动作的结果；清理失败必须进入持久化补偿队列。
            if vector_written and not es_written:
                try:
                    vector_store_manager.delete_by_ids(chunk_ids)
                except Exception as cleanup_error:
                    self.task_store.add_compensation(stable_task_id, {
                        "store": "milvus", "operation": "delete_by_ids", "ids": chunk_ids,
                        "error": str(cleanup_error),
                    })
                try:
                    es_store_manager.delete_by_ids(chunk_ids)
                except Exception as cleanup_error:
                    self.task_store.add_compensation(stable_task_id, {
                        "store": "elasticsearch", "operation": "delete_by_ids", "ids": chunk_ids,
                        "error": str(cleanup_error),
                    })
            task = self.task_store.get(stable_task_id) or {}
            if es_written:
                self.task_store.add_compensation(stable_task_id, {
                    "store": "both", "operation": "delete_old_versions",
                    "source": normalized_path, "keep_version": index_version,
                    "error": str(e),
                })
            retry_count = int(task.get("retry_count", 0)) + 1
            delay = config.index_retry_base_delay_seconds * (2 ** max(0, retry_count - 1))
            next_retry_at = None if retry_count >= config.index_retry_max_attempts else datetime.now().timestamp() + delay
            self.task_store.update(
                stable_task_id,
                status="failed",
                error=str(e),
                retry_count=retry_count,
                next_retry_at=datetime.fromtimestamp(next_retry_at).isoformat() if next_retry_at else None,
            )
            raise RuntimeError(f"索引文件失败: {e}") from e

    def retry_task(self, task_id: str) -> dict[str, Any]:
        """重试持久化任务队列中的失败任务。"""
        task = self.task_store.get(task_id)
        if task is None:
            raise ValueError(f"索引任务不存在: {task_id}")
        if task.get("status") not in {"failed", "partial_success"}:
            raise ValueError(f"任务当前不可重试: {task.get('status')}")
        staged = task.get("staged_file_path")
        source = staged if staged and Path(staged).exists() else task["file_path"]
        return self.index_single_file(task["file_path"], source if staged else None, task_id=task_id)


# 全局单例
vector_index_service = VectorIndexService()
