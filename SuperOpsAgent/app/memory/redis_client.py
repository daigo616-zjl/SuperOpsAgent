"""Redis 客户端管理器（短期记忆存储）"""

import redis.asyncio as aioredis
from loguru import logger

from app.config import config


class RedisClientManager:
    """Redis 客户端管理器

    Redis 仅承载短期记忆（可重建、可降级），连接失败不阻断应用启动，
    运行时由 ShortTermMemory 熔断降级到 checkpoint 路径。
    """

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        if self._client is not None:
            logger.debug("Redis 已连接，跳过重复 connect")
            return

        self._client = aioredis.from_url(
            config.redis_url,
            socket_timeout=config.redis_timeout,
            socket_connect_timeout=config.redis_timeout,
            decode_responses=True,
        )
        if not await self._client.ping():
            raise RuntimeError(f"连接 Redis 失败: {config.redis_url}")
        logger.info(f"✅ Redis 连接成功: {config.redis_url}")

    def get_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis 客户端未初始化，请先调用 connect()")
        return self._client

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            return bool(await self._client.ping())
        except Exception as e:
            logger.error(f"Redis 健康检查失败: {e}")
            return False

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()
            logger.info("已关闭 Redis 连接")


redis_client_manager = RedisClientManager()
