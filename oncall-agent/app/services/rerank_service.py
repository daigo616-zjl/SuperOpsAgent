from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from app.config import config


class RerankService:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _load_model(self) -> Any:
        if not config.rag_rerank_model:
            raise ValueError("rag_rerank_model is not configured")

        from sentence_transformers import CrossEncoder

        return CrossEncoder(config.rag_rerank_model)

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _predict_scores(self, query: str, docs: list[Document]) -> list[float]:
        model = self._get_model()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = model.predict(pairs)
        return [float(score) for score in scores]

    def _score_pairs(self, query: str, docs: list[Document]) -> list[float]:
        future = self._executor.submit(self._predict_scores, query, docs)
        try:
            return future.result(timeout=config.rag_rerank_timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError("rerank timeout") from exc

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs or top_k <= 0:
            return []
        if not query.strip():
            logger.warning("Rerank 降级: empty query")
            return docs[:top_k]

        started_at = perf_counter()
        try:
            scores = self._score_pairs(query, docs)
            if len(scores) != len(docs):
                raise ValueError("score length mismatch")
            ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
            result = [doc for doc, _ in ranked[:top_k]]
            duration_ms = int((perf_counter() - started_at) * 1000)
            logger.info(
                f"Rerank 完成: candidates={len(docs)}, returned={len(result)}, duration_ms={duration_ms}"
            )
            return result
        except Exception as exc:
            logger.warning(f"Rerank 降级: {exc}, fallback=hybrid_top_k")
            return docs[:top_k]


rerank_service = RerankService()
