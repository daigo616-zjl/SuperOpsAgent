from __future__ import annotations

import asyncio
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

    def warmup(self) -> bool:
        """加载重排模型并执行一次最小推理，避免首个请求承担冷启动耗时。"""
        if not config.rag_rerank_enabled or not config.rag_rerank_warmup_enabled:
            logger.info("Rerank 预热已跳过")
            return False

        started_at = perf_counter()
        warmup_doc = Document(page_content="用于初始化重排模型的预热文本。")
        future = self._executor.submit(
            self._predict_scores,
            "重排模型预热",
            [warmup_doc],
        )
        try:
            scores = future.result(timeout=config.rag_rerank_warmup_timeout)
            if len(scores) != 1:
                raise ValueError("warmup score length mismatch")
        except FuturesTimeoutError:
            future.cancel()
            logger.warning(
                f"Rerank 预热超时: timeout={config.rag_rerank_warmup_timeout}s，"
                "后续请求将按原有降级策略执行"
            )
            return False
        except Exception as exc:
            logger.warning(f"Rerank 预热失败: {exc}，后续请求将按原有降级策略执行")
            return False

        duration_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            f"Rerank 预热完成: model={config.rag_rerank_model}, duration_ms={duration_ms}"
        )
        return True

    async def warmup_async(self) -> bool:
        """在线程中执行预热，避免阻塞异步事件循环。"""
        return await asyncio.to_thread(self.warmup)

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
            ranked = sorted(
                zip(docs, scores, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
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
