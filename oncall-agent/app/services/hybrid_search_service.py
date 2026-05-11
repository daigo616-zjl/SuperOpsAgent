"""Hybrid 检索服务模块"""

import asyncio
from typing import Dict, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.bm25_search_service import BM25SearchResult, bm25_search_service
from app.services.vector_search_service import SearchResult, vector_search_service


class HybridSearchService:
    """双路召回 + RRF 融合服务"""

    def _vector_to_document(self, result: SearchResult) -> Document:
        metadata = dict(result.metadata or {})
        metadata["id"] = result.id
        return Document(page_content=result.content, metadata=metadata)

    def _bm25_to_document(self, result: BM25SearchResult) -> Document:
        metadata = dict(result.metadata or {})
        metadata["id"] = result.id
        return Document(page_content=result.content, metadata=metadata)

    def _rrf_fuse(
        self,
        vector_hits: List[SearchResult],
        bm25_hits: List[BM25SearchResult],
        top_k: int,
    ) -> List[Document]:
        scores: Dict[str, float] = {}
        payload: Dict[str, Document] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (config.rag_rrf_k + rank)
            payload[hit.id] = self._vector_to_document(hit)

        for rank, hit in enumerate(bm25_hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1 / (config.rag_rrf_k + rank)
            payload.setdefault(hit.id, self._bm25_to_document(hit))

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [payload[chunk_id] for chunk_id, _ in ranked]

    async def search(self, query: str, top_k: int) -> List[Document]:
        if not query.strip():
            return []

        vector_task = asyncio.to_thread(
            vector_search_service.search_similar_documents,
            query,
            config.rag_recall_size,
        )
        bm25_task = bm25_search_service.search(query, config.rag_recall_size)
        vector_result, bm25_result = await asyncio.gather(
            vector_task,
            bm25_task,
            return_exceptions=True,
        )

        vector_hits: List[SearchResult] = []
        bm25_hits: List[BM25SearchResult] = []

        if isinstance(vector_result, Exception):
            logger.warning(f"向量检索降级: {vector_result}")
        else:
            vector_hits = vector_result

        if isinstance(bm25_result, Exception):
            logger.warning(f"BM25 检索降级: {bm25_result}")
        else:
            bm25_hits = bm25_result

        if not vector_hits and not bm25_hits:
            return []

        if vector_hits and bm25_hits:
            docs = self._rrf_fuse(vector_hits, bm25_hits, top_k)
            logger.info(
                f"Hybrid 检索完成: BM25={len(bm25_hits)}, 向量={len(vector_hits)}, RRF={len(docs)}"
            )
            return docs

        if vector_hits:
            docs = [self._vector_to_document(hit) for hit in vector_hits[:top_k]]
            logger.info(f"Hybrid 检索降级为纯向量: {len(docs)}")
            return docs

        docs = [self._bm25_to_document(hit) for hit in bm25_hits[:top_k]]
        logger.info(f"Hybrid 检索降级为纯 BM25: {len(docs)}")
        return docs

    def search_sync(self, query: str, top_k: int) -> List[Document]:
        return asyncio.run(self.search(query, top_k))


hybrid_search_service = HybridSearchService()
