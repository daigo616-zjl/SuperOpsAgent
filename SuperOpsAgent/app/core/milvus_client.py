"""Milvus Lite 客户端工厂模块。"""

from pathlib import Path
from typing import Any

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    connections,
    utility,
)

from app.config import config

_milvus_client_alias_patch_done = False


def _patch_pymilvus_milvus_client_orm_alias() -> None:
    """
    langchain_milvus 内部创建的 MilvusClient 会将 _using 设为 ``cm-{id}``，
    该别名未在 pymilvus.orm.connections 中注册；随后 ORM ``Collection(..., using=...)``
    会抛出 ConnectionNotExistException: should create connection first.

    在已通过 ``connections.connect(alias="default", ...)`` 建立连接后，
    强制让 MilvusClient 使用 ``default`` 别名，与 ORM 一致。
    """
    global _milvus_client_alias_patch_done

    if _milvus_client_alias_patch_done:
        return
    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient
    except ImportError:
        return

    _orig_init = MilvusClient.__init__

    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        self._using = "default"

    MilvusClient.__init__ = _wrapped_init
    _milvus_client_alias_patch_done = True


class MilvusClientManager:
    """管理进程内 Milvus Lite 数据库和 ``biz`` collection。"""

    # 常量定义
    COLLECTION_NAME: str = "biz"
    VECTOR_DIM: int = 1024  # 统一使用 1024 维
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None
        self._uri: str | None = None

    @property
    def uri(self) -> str:
        """返回规范化后的 Milvus Lite 数据库文件路径。"""
        if self._uri is None:
            path = Path(config.milvus_lite_path).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._uri = path.resolve().as_posix()
        return self._uri

    @property
    def grpc_options(self) -> dict[str, int | bool]:
        """Return conservative keepalive settings for the embedded server."""
        return {
            "grpc.keepalive_time_ms": config.milvus_grpc_keepalive_time_ms,
            "grpc.keepalive_timeout_ms": config.milvus_grpc_keepalive_timeout_ms,
            "grpc.keepalive_permit_without_calls": (
                config.milvus_grpc_keepalive_permit_without_calls
            ),
        }

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        # 幂等：导入阶段可能已由 VectorStoreManager 等提前连接，避免重复初始化
        if self._collection is not None and self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            _patch_pymilvus_milvus_client_orm_alias()

            logger.info(f"正在打开 Milvus Lite: {self.uri}")

            # Windows 路径会被 Milvus Lite 解析为数据库上下文，因此显式指定
            # default db_name，保证相对路径和绝对路径行为一致。
            connections.connect(
                alias="default",
                uri=self.uri,
                db_name=config.milvus_lite_db_name,
                timeout=config.milvus_timeout / 1000,  # 转换为秒
                grpc_options=self.grpc_options,
            )

            # 创建客户端
            self._client = MilvusClient(
                uri=self.uri,
                db_name=config.milvus_lite_db_name,
                timeout=config.milvus_timeout / 1000,
                grpc_options=self.grpc_options,
            )

            logger.info("成功打开 Milvus Lite")

            # 检查并创建 collection
            if not self._collection_exists():
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)

                # 检查向量维度是否匹配
                schema = self._collection.schema
                vector_field = None
                existing_dim = None
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break

                if (
                    vector_field
                    and hasattr(vector_field, "params")
                    and "dim" in vector_field.params
                ):
                    existing_dim = vector_field.params["dim"]
                    if existing_dim != self.VECTOR_DIM:
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        logger.info(f"正在删除旧 collection '{self.COLLECTION_NAME}'...")
                        _ = utility.drop_collection(self.COLLECTION_NAME)
                        logger.info(f"正在重新创建 collection '{self.COLLECTION_NAME}'...")
                        self._create_collection()
                        logger.info(f"成功重新创建 collection，维度: {self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 加载 collection
            self._load_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        # pymilvus 的类型标注可能不准确，实际返回 bool
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)

    def _create_collection(self) -> None:
        """创建 biz collection"""
        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        # 创建 schema
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        # 创建 collection
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
        )

        # 创建索引
        self._create_index()

    def _create_index(self) -> None:
        """为 vector 字段创建索引"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        index_params = {
            "metric_type": "L2",  # 欧氏距离
            "index_type": "FLAT",
            "params": {},
        }

        _ = self._collection.create_index(
            field_name="vector",
            index_params=index_params,
        )

        logger.info("成功为 vector 字段创建索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        # 检查 collection 是否已加载（兼容多版本）
        try:
            # 方法 1: 尝试使用 utility.load_state（新版本）
            load_state = utility.load_state(self.COLLECTION_NAME)
            # load_state 返回字符串或枚举，如 "Loaded" 或 "NotLoad"
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方法 2: 直接尝试加载，捕获 "already loaded" 异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        Returns:
            Collection: collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False

            _ = self._client.list_collections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        errors = []

        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if self._client is not None:
                self._client.close()
        except Exception as e:
            errors.append(f"关闭 Milvus Lite 客户端失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None

        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口"""
        _ = self.connect()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """上下文管理器退出"""
        self.close()


# 全局单例
milvus_manager = MilvusClientManager()
