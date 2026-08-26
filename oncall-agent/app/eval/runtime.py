from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.core.es_client import es_client_manager
from app.core.milvus_client import milvus_manager


class EvaluationRuntimeError(RuntimeError):
    """评测运行环境初始化失败。"""


@asynccontextmanager
async def evaluation_runtime() -> AsyncIterator[None]:
    """为离线评测初始化并可靠释放检索存储连接。"""
    try:
        milvus_manager.connect()
    except Exception as exc:
        raise EvaluationRuntimeError(
            "无法打开 Milvus Lite。请先停止 FastAPI 服务，确保没有其他进程占用 "
            "MILVUS_LITE_PATH，然后重新运行评测。"
        ) from exc

    try:
        await es_client_manager.connect()
    except Exception as exc:
        await es_client_manager.close()
        milvus_manager.close()
        raise EvaluationRuntimeError(
            "无法连接 Elasticsearch，请检查 ES_SCHEME、ES_HOST、ES_PORT 和 ES_INDEX 配置。"
        ) from exc

    try:
        yield
    finally:
        await es_client_manager.close()
        milvus_manager.close()
