"""长期记忆：Elasticsearch 半结构化长文本存储（BM25 精确召回）"""

from datetime import UTC, datetime

from loguru import logger

from app.config import config
from app.core.es_client import es_client_manager
from app.memory.models import MemoryHit, content_hash


class EsMemoryStore:
    """独立于业务文档索引 biz 的记忆索引（doc _id = content_hash 天然幂等）"""

    _ensured = False

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
                    "user_id": {"type": "keyword"},
                    "subject": {"type": "keyword"},
                    "content_hash": {"type": "keyword"},
                    "created_at": {"type": "date"},
                }
            },
        }

    def ensure_index(self) -> None:
        if self._ensured:
            return
        client = es_client_manager.get_sync_client()
        if not bool(client.indices.exists(index=config.es_memory_index)):
            client.indices.create(index=config.es_memory_index, body=self._index_body())
            logger.info(f"创建长期记忆 ES 索引: {config.es_memory_index}")
        self._ensured = True

    def index(self, user_id: str, content: str, subject: str = "") -> str:
        c_hash = content_hash(user_id, content)
        self.ensure_index()
        client = es_client_manager.get_sync_client()
        client.index(
            index=config.es_memory_index,
            id=c_hash,
            document={
                "content": content,
                "user_id": user_id,
                "subject": subject,
                "content_hash": c_hash,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return c_hash

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryHit]:
        if not query.strip():
            return []
        self.ensure_index()
        client = es_client_manager.get_sync_client()
        response = client.search(
            index=config.es_memory_index,
            query={
                "bool": {
                    "must": [{"match": {"content": query}}],
                    "filter": [{"term": {"user_id": user_id}}],
                }
            },
            size=top_k,
        )
        hits = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            hits.append(
                MemoryHit(
                    content=source.get("content", ""),
                    subject=source.get("subject", "") or "",
                    score=float(hit.get("_score", 0.0)),
                    content_hash=source.get("content_hash", "") or hit.get("_id", ""),
                )
            )
        return hits


es_memory_store = EsMemoryStore()
