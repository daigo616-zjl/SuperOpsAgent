"""持久化文件索引任务及补偿队列。"""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from loguru import logger


INDEX_TASK_STATUSES = {"pending", "indexing", "partial_success", "success", "failed"}


class IndexTaskStore:
    """使用同目录临时文件 + 原子替换保存任务，避免失败状态只存在日志中。"""

    def __init__(self, path: str | Path = "./uploads/.index_tasks.json") -> None:
        self.path = Path(path)
        self._lock = RLock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"读取索引任务队列失败: {self.path}, 错误: {exc}")
            return []

    def _write(self, tasks: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(tasks, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create(
        self, *, file_path: str, staged_file_path: str | None, version: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        task = {
            "task_id": task_id or version,
            "file_path": file_path,
            "staged_file_path": staged_file_path,
            "index_version": version,
            "status": "pending",
            "retry_count": 0,
            "next_retry_at": None,
            "error": None,
            "compensations": [],
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        with self._lock:
            tasks = self._read()
            tasks.append(task)
            self._write(tasks)
        return task

    def update(self, task_id: str, **changes: Any) -> dict[str, Any]:
        status = changes.get("status")
        if status is not None and status not in INDEX_TASK_STATUSES:
            raise ValueError(f"非法索引任务状态: {status}")
        with self._lock:
            tasks = self._read()
            for task in tasks:
                if task.get("task_id") == task_id:
                    task.update(changes)
                    task["updated_at"] = self._now()
                    self._write(tasks)
                    return task
        raise KeyError(f"索引任务不存在: {task_id}")

    def add_compensation(self, task_id: str, operation: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            tasks = self._read()
            for task in tasks:
                if task.get("task_id") == task_id:
                    task.setdefault("compensations", []).append(
                        {**operation, "created_at": self._now()}
                    )
                    task["updated_at"] = self._now()
                    self._write(tasks)
                    return task
        raise KeyError(f"索引任务不存在: {task_id}")

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            return next((task for task in self._read() if task.get("task_id") == task_id), None)

    def list(self, statuses: set[str] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            tasks = self._read()
        if statuses is None:
            return tasks
        return [task for task in tasks if task.get("status") in statuses]

    def latest_for_file(self, file_path: str) -> dict[str, Any] | None:
        """返回指定文件最近一次任务，用于把失败任务 ID 返回给上传方。"""
        with self._lock:
            tasks = self._read()
        for task in reversed(tasks):
            if task.get("file_path") == file_path:
                return task
        return None
