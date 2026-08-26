"""Elasticsearch 客户端工厂模块"""

from elasticsearch import AsyncElasticsearch, Elasticsearch
from loguru import logger

from app.config import config


class EsClientManager:
    """Elasticsearch 客户端管理器"""

    def __init__(self) -> None:
        self._sync_client: Elasticsearch | None = None
        self._async_client: AsyncElasticsearch | None = None

    def _hosts(self) -> list[str]:
        return [f"{config.es_scheme}://{config.es_host}:{config.es_port}"]

    def _index_body(self) -> dict:
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25"}},
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": config.es_analyzer,
                        "search_analyzer": config.es_search_analyzer,
                    },
                    "source": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "extension": {"type": "keyword"},
                    "h1": {"type": "keyword"},
                    "h2": {"type": "keyword"},
                    "h3": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                    "indexed_at": {"type": "date"},
                }
            },
        }

    async def connect(self) -> None:
        if self._sync_client is not None and self._async_client is not None:
            logger.debug("Elasticsearch 已连接，跳过重复 connect")
            return

        self._sync_client = Elasticsearch(
            hosts=self._hosts(),
            request_timeout=config.es_timeout,
        )
        self._async_client = AsyncElasticsearch(
            hosts=self._hosts(),
            request_timeout=config.es_timeout,
        )

        if not self._sync_client.ping():
            raise RuntimeError(f"连接 Elasticsearch 失败: {config.es_host}:{config.es_port}")

        index_exists = self._sync_client.indices.exists(index=config.es_index)
        if not bool(index_exists):
            self._sync_client.indices.create(
                index=config.es_index,
                body=self._index_body(),
            )
            logger.info(f"创建 Elasticsearch index 成功: {config.es_index}")
        else:
            logger.info(f"Elasticsearch index 已存在: {config.es_index}")

    def get_sync_client(self) -> Elasticsearch:
        if self._sync_client is None:
            raise RuntimeError("Elasticsearch 同步客户端未初始化，请先调用 connect()")
        return self._sync_client

    def get_async_client(self) -> AsyncElasticsearch:
        if self._async_client is None:
            raise RuntimeError("Elasticsearch 异步客户端未初始化，请先调用 connect()")
        return self._async_client

    def health_check(self) -> bool:
        try:
            if self._sync_client is None:
                return False
            return bool(self._sync_client.ping())
        except Exception as e:
            logger.error(f"Elasticsearch 健康检查失败: {e}")
            return False

    async def close(self) -> None:
        async_client = self._async_client
        sync_client = self._sync_client
        self._async_client = None
        self._sync_client = None

        if async_client is not None:
            await async_client.close()

        if sync_client is not None:
            sync_client.close()

        logger.info("已关闭 Elasticsearch 连接")


es_client_manager = EsClientManager()
