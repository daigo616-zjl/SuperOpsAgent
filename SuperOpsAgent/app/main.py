"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import aiops, chat, file, health, users
from app.config import config
from app.core.es_client import es_client_manager
from app.core.milvus_client import milvus_manager
from app.core.postgres import postgres_manager
from app.memory.memory_writer import memory_write_worker
from app.memory.redis_client import redis_client_manager
from app.services.aiops_service import aiops_service
from app.services.postgres_index_worker import postgres_index_worker
from app.services.rerank_service import rerank_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # PostgreSQL 是文档、注册表和 Outbox 的唯一权威来源，连接失败时拒绝启动。
    logger.info("🔌 正在连接 PostgreSQL...")
    postgres_manager.connect()

    # Milvus 和 Elasticsearch 是可重建的派生索引。
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")

    logger.info("🔌 正在连接 Elasticsearch...")
    try:
        await es_client_manager.connect()
        logger.info("✅ Elasticsearch 连接成功")
    except Exception as exc:
        if config.es_required:
            raise
        logger.warning(f"⚠️ Elasticsearch 不可用，继续启动（ES_REQUIRED=false）: {exc}")

    logger.info("🔥 正在预热 Rerank 模型...")
    await rerank_service.warmup_async()

    # Redis 短期记忆不可用时由 ShortTermMemory 熔断降级，不阻断启动
    if config.memory_enabled:
        logger.info("🔌 正在连接 Redis（短期记忆）...")
        try:
            await redis_client_manager.connect()
        except Exception as exc:
            logger.warning(f"⚠️ Redis 不可用，短期记忆将降级到 checkpoint 路径: {exc}")

    postgres_index_worker.start()
    memory_write_worker.start()

    logger.info("=" * 60)

    yield

    # 关闭时执行
    postgres_index_worker.stop()
    memory_write_worker.stop()
    logger.info("🔌 正在关闭 AIOps checkpoint 连接池...")
    await aiops_service.aclose()
    logger.info("🔌 正在关闭 Redis 连接...")
    await redis_client_manager.close()
    logger.info("🔌 正在关闭 Elasticsearch 连接...")
    await es_client_manager.close()
    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info("🔌 正在关闭 PostgreSQL 连接池...")
    postgres_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_frontend_asset_cache(request: Request, call_next):
    """避免浏览器混用不同版本的首页、脚本和样式。"""
    response = await call_next(request)
    if request.url.path in {"/", "/static/app.js", "/static/styles.css"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# 注册路由
app.include_router(health.router, prefix="/api", tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(users.router, prefix="/api", tags=["用户"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
