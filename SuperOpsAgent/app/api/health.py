"""健康检查接口"""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import config
from app.core.es_client import es_client_manager
from app.core.milvus_client import milvus_manager
from app.core.postgres import postgres_manager

router = APIRouter()


@router.get("/health")
async def health_check():

    """健康检查接口
    检查服务状态和数据库连接状态

    Returns:
        JSONResponse: 健康检查结果
    """
    # 检查服务基本状态
    health_data: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }

    # PostgreSQL 是权威数据源。
    try:
        postgres_healthy = postgres_manager.health_check()
        health_data["postgresql"] = {
            "status": "connected" if postgres_healthy else "disconnected",
            "message": "PostgreSQL 连接正常" if postgres_healthy else "PostgreSQL 连接异常",
        }
    except Exception as e:
        logger.warning(f"PostgreSQL 健康检查失败: {e}")
        health_data["postgresql"] = {
            "status": "error",
            "message": f"PostgreSQL 检查失败: {str(e)}",
        }

    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = "connected" if milvus_healthy else "disconnected"
        milvus_message: str = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }

    # 检查 Elasticsearch 连接状态
    try:
        es_healthy = es_client_manager.health_check()
        es_status: str = "connected" if es_healthy else "disconnected"
        es_message: str = "Elasticsearch 连接正常" if es_healthy else "Elasticsearch 连接异常"
        health_data["elasticsearch"] = {
            "status": es_status,
            "message": es_message
        }
    except Exception as e:
        logger.warning(f"Elasticsearch 健康检查失败: {e}")
        health_data["elasticsearch"] = {
            "status": "error",
            "message": f"Elasticsearch 检查失败: {str(e)}"
        }

    # 判断整体健康状态
    overall_status = "healthy"
    status_code = 200

    # 如果数据库不可用，服务不可用
    if (
        health_data["postgresql"]["status"] != "connected"
        or health_data["milvus"]["status"] != "connected"
        or health_data["elasticsearch"]["status"] != "connected"
    ):
        overall_status = "unhealthy"
        status_code = 503
        health_data["error"] = "数据库不可用"

    health_data["status"] = overall_status

    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": "服务运行正常" if overall_status == "healthy" else "服务不可用",
            "data": health_data
        }
    )
