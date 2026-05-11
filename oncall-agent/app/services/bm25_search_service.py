"""BM25 检索服务模块"""

from typing import Any, Dict, List

from loguru import logger

from app.config import config
from app.core.es_client import es_client_manager


class BM25SearchResult:
    """BM25 搜索结果类"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata


class BM25SearchService:
    """BM25 检索服务"""

    async def search(self, query: str, top_k: int) -> List[BM25SearchResult]:
        if not query.strip():
            return []

        client = es_client_manager.get_async_client()
        body = {
            "size": top_k,
            "query": {"match": {"content": query}},
            "_source": ["content", "source", "file_name", "extension", "h1", "h2", "h3", "metadata"],
        }

        response = await client.search(index=config.es_index, body=body)
        hits = response.get("hits", {}).get("hits", [])

        results: List[BM25SearchResult] = []
        for hit in hits:
            source = hit.get("_source", {})
            raw_metadata = source.get("metadata") or {}
            metadata = dict(raw_metadata)
            metadata["_source"] = source.get("source")
            metadata["_file_name"] = source.get("file_name")
            metadata["_extension"] = source.get("extension")
            metadata["h1"] = source.get("h1")
            metadata["h2"] = source.get("h2")
            metadata["h3"] = source.get("h3")

            results.append(
                BM25SearchResult(
                    id=hit.get("_id", ""),
                    content=source.get("content", ""),
                    score=float(hit.get("_score") or 0.0),
                    metadata=metadata,
                )
            )

        logger.info(f"BM25 检索完成, 找到 {len(results)} 个文档")
        return results


bm25_search_service = BM25SearchService()
