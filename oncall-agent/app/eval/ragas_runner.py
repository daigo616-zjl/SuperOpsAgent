from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any
from uuid import uuid4

from loguru import logger
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory as ragas_llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextRelevance,
    Faithfulness,
)

from app.config import config
from app.eval.answer_generator import EvalAnswerGenerationError, generate_answer_with_context
from app.eval.dataset import EvalSample


@dataclass(slots=True)
class EvalDetail:
    id: str
    question: str
    ground_truth: str
    answer: str | None = None
    retrieved_contexts: list[str] = field(default_factory=list)
    scores: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class EvalSummary:
    total: int
    success: int
    failed: int
    metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class EvalReport:
    summary: EvalSummary
    details: list[EvalDetail]
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": asdict(self.summary),
            "details": [asdict(detail) for detail in self.details],
            "errors": self.errors,
        }


def _build_evaluator_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_api_base,
        timeout=config.eval_metric_timeout,
        max_retries=1,
    )


def _build_evaluator_llm(client: AsyncOpenAI) -> Any:
    return ragas_llm_factory(
        model=config.eval_model,
        provider="openai",
        client=client,
        temperature=0.0,
        max_tokens=4096,
    )


def _build_metrics() -> tuple[AsyncOpenAI, list[Any]]:
    client = _build_evaluator_client()
    evaluator_llm = _build_evaluator_llm(client)
    evaluator_embeddings = embedding_factory(
        provider="openai",
        model=config.dashscope_embedding_model,
        client=client,
    )
    return (
        client,
        [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
            ContextRelevance(llm=evaluator_llm),
        ],
    )


async def _score_details(
    details: list[EvalDetail],
) -> tuple[dict[str, float | None], list[dict[str, str]]]:
    client, metrics = _build_metrics()
    faithfulness, answer_relevancy, answer_correctness, context_relevance = metrics
    errors: list[dict[str, str]] = []

    async def score_metric(
        detail: EvalDetail,
        name: str,
        operation: Awaitable[Any],
    ) -> float | None:
        try:
            result = await asyncio.wait_for(operation, timeout=config.eval_metric_timeout)
            return float(result.value) if isinstance(result.value, (int, float)) else None
        except Exception as exc:
            message = f"{name} 评分失败: {exc}"
            logger.error(f"评测样本 {detail.id}: {message}")
            errors.append({"id": detail.id, "metric": name, "error": str(exc)})
            return None

    try:
        for detail in details:
            if not detail.answer or detail.error:
                continue

            operations = [
                (
                    "answer_relevancy",
                    answer_relevancy.ascore(
                        user_input=detail.question,
                        response=detail.answer,
                    ),
                ),
                (
                    "answer_correctness",
                    answer_correctness.ascore(
                        user_input=detail.question,
                        response=detail.answer,
                        reference=detail.ground_truth,
                    ),
                ),
            ]
            if detail.retrieved_contexts:
                operations.extend(
                    [
                        (
                            "faithfulness",
                            faithfulness.ascore(
                                user_input=detail.question,
                                response=detail.answer,
                                retrieved_contexts=detail.retrieved_contexts,
                            ),
                        ),
                        (
                            "context_relevance",
                            context_relevance.ascore(
                                user_input=detail.question,
                                retrieved_contexts=detail.retrieved_contexts,
                            ),
                        ),
                    ]
                )

            values = await asyncio.gather(
                *[score_metric(detail, name, operation) for name, operation in operations]
            )
            detail.scores.update(
                {name: value for (name, _), value in zip(operations, values, strict=True)}
            )
    finally:
        await client.close()

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_relevance",
    ]
    summary: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            value
            for detail in details
            if isinstance((value := detail.scores.get(name)), (int, float))
        ]
        summary[name] = mean(values) if values else None
    return summary, errors


async def run_ragas_evaluation(samples: list[EvalSample]) -> EvalReport:
    details: list[EvalDetail] = []
    errors: list[dict[str, str]] = []

    for sample in samples:
        detail = EvalDetail(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
        )
        details.append(detail)

        try:
            result = await generate_answer_with_context(
                sample.question,
                session_id=f"eval-{sample.id}-{uuid4().hex}",
            )
            detail.answer = result.answer
            detail.retrieved_contexts = result.retrieved_contexts
        except EvalAnswerGenerationError as exc:
            detail.error = str(exc)
            errors.append({"id": sample.id, "error": str(exc)})

    successful_details = [detail for detail in details if detail.answer and not detail.error]
    if not successful_details:
        raise ValueError("All evaluation samples failed to generate answers")

    metric_summary, metric_errors = await _score_details(successful_details)
    errors.extend(metric_errors)

    summary = EvalSummary(
        total=len(samples),
        success=len(successful_details),
        failed=len(samples) - len(successful_details),
        metrics=metric_summary,
    )
    return EvalReport(summary=summary, details=details, errors=errors)
