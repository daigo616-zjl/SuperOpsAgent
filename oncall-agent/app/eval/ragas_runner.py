from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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


class BatchedFaithfulness(Faithfulness):
    """Run NLI in bounded batches to keep structured responses reliable."""

    def __init__(self, *args: Any, statement_batch_size: int = 10, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.statement_batch_size = max(1, statement_batch_size)

    async def _create_verdicts(self, statements: list[str], context: str) -> Any:
        if not statements:
            return await super()._create_verdicts(statements, context)

        all_verdicts: list[Any] = []
        last_result: Any = None
        for start in range(0, len(statements), self.statement_batch_size):
            batch = statements[start : start + self.statement_batch_size]
            last_result = await super()._create_verdicts(batch, context)
            all_verdicts.extend(last_result.statements)

        return last_result.model_copy(update={"statements": all_verdicts})


def _build_evaluator_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_api_base,
        timeout=config.eval_metric_timeout,
        max_retries=config.eval_client_max_retries,
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
            BatchedFaithfulness(
                llm=evaluator_llm,
                statement_batch_size=config.eval_faithfulness_statement_batch_size,
            ),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
            ContextRelevance(llm=evaluator_llm),
        ],
    )


def _metric_timeout(name: str) -> int:
    if name == "faithfulness":
        return config.eval_faithfulness_timeout
    if name == "answer_correctness":
        return config.eval_answer_correctness_timeout
    return config.eval_metric_timeout


def _metric_error_message(exc: Exception, timeout: int) -> str:
    if isinstance(exc, TimeoutError):
        return f"TimeoutError: metric exceeded {timeout}s"
    detail = str(exc).strip() or repr(exc)
    return f"{type(exc).__name__}: {detail}"


async def _score_details(
    details: list[EvalDetail],
) -> tuple[dict[str, float | None], list[dict[str, str]]]:
    client, metrics = _build_metrics()
    faithfulness, answer_relevancy, answer_correctness, context_relevance = metrics
    errors: list[dict[str, str]] = []
    metric_semaphore = asyncio.Semaphore(max(1, config.eval_metric_max_concurrency))

    async def score_metric(
        detail: EvalDetail,
        name: str,
        operation_factory: Callable[[], Awaitable[Any]],
    ) -> float | None:
        timeout = _metric_timeout(name)
        try:
            async with metric_semaphore:
                result = await asyncio.wait_for(operation_factory(), timeout=timeout)
            return float(result.value) if isinstance(result.value, (int, float)) else None
        except Exception as exc:
            error = _metric_error_message(exc, timeout)
            message = f"{name} 评分失败: {error}"
            logger.error(f"评测样本 {detail.id}: {message}")
            errors.append({"id": detail.id, "metric": name, "error": error})
            return None

    try:
        for detail in details:
            if not detail.answer or detail.error:
                continue

            operations = [
                (
                    "answer_relevancy",
                    lambda detail=detail: answer_relevancy.ascore(
                        user_input=detail.question,
                        response=detail.answer or "",
                    ),
                ),
                (
                    "answer_correctness",
                    lambda detail=detail: answer_correctness.ascore(
                        user_input=detail.question,
                        response=detail.answer or "",
                        reference=detail.ground_truth,
                    ),
                ),
            ]
            if detail.retrieved_contexts:
                operations.extend(
                    [
                        (
                            "faithfulness",
                            lambda detail=detail: faithfulness.ascore(
                                user_input=detail.question,
                                response=detail.answer or "",
                                retrieved_contexts=detail.retrieved_contexts,
                            ),
                        ),
                        (
                            "context_relevance",
                            lambda detail=detail: context_relevance.ascore(
                                user_input=detail.question,
                                retrieved_contexts=detail.retrieved_contexts,
                            ),
                        ),
                    ]
                )

            values = await asyncio.gather(
                *[
                    score_metric(detail, name, operation_factory)
                    for name, operation_factory in operations
                ]
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
