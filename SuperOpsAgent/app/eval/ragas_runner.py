from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
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
from app.eval.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    relevant_sources_from_metadata,
)


@dataclass(slots=True)
class EvalDetail:
    id: str
    question: str
    ground_truth: str
    answer: str | None = None
    retrieved_contexts: list[str] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)
    retrieval_attempted: bool = False
    retrieval_candidate_sources: list[str] = field(default_factory=list)
    reranked_sources: list[str] = field(default_factory=list)
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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalReport":
        """Restore a report so already-generated answers can be rescored."""
        summary_payload = payload.get("summary") or {}
        details = [
            EvalDetail(
                id=str(item["id"]),
                question=str(item["question"]),
                ground_truth=str(item["ground_truth"]),
                answer=item.get("answer"),
                retrieved_contexts=list(item.get("retrieved_contexts") or []),
                relevant_sources=list(item.get("relevant_sources") or []),
                retrieval_attempted=bool(item.get("retrieval_attempted", False)),
                retrieval_candidate_sources=list(item.get("retrieval_candidate_sources") or []),
                reranked_sources=list(item.get("reranked_sources") or []),
                scores=dict(item.get("scores") or {}),
                error=item.get("error"),
            )
            for item in payload.get("details", [])
        ]
        return cls(
            summary=EvalSummary(
                total=int(summary_payload.get("total", len(details))),
                success=int(summary_payload.get("success", 0)),
                failed=int(summary_payload.get("failed", 0)),
                metrics=dict(summary_payload.get("metrics") or {}),
            ),
            details=details,
            errors=list(payload.get("errors") or []),
        )


SUPPORTED_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_relevance",
)


def normalize_metric_names(metric_names: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and normalize the metric selection for one evaluation run."""
    if metric_names is None:
        return SUPPORTED_METRICS

    normalized = tuple(dict.fromkeys(name.strip().lower() for name in metric_names if name.strip()))
    unsupported = sorted(set(normalized) - set(SUPPORTED_METRICS))
    if unsupported:
        supported = ", ".join(SUPPORTED_METRICS)
        raise ValueError(f"Unsupported metrics: {', '.join(unsupported)}. Supported: {supported}")
    if not normalized:
        raise ValueError("At least one evaluation metric must be selected")
    return normalized


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


def _retrieval_metric_names() -> tuple[str, str]:
    if config.eval_recall_k <= 0 or config.eval_hit_k <= 0:
        raise ValueError("EVAL_RECALL_K and EVAL_HIT_K must be positive")
    return f"recall_at_{config.eval_recall_k}", f"hit_at_{config.eval_hit_k}"


def _score_retrieval(detail: EvalDetail) -> dict[str, float | None]:
    recall_name, hit_name = _retrieval_metric_names()
    return {
        recall_name: recall_at_k(
            detail.relevant_sources,
            detail.retrieval_candidate_sources,
            config.eval_recall_k,
        ),
        hit_name: hit_at_k(
            detail.relevant_sources,
            detail.reranked_sources,
            config.eval_hit_k,
        ),
    }


def _summarize_metrics(
    details: list[EvalDetail],
    metric_names: list[str],
) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            value
            for detail in details
            if isinstance((value := detail.scores.get(name)), (int, float))
        ]
        summary[name] = mean(values) if values else None
    return summary


async def _score_details(
    details: list[EvalDetail],
    metric_names: Sequence[str] | None = None,
) -> tuple[dict[str, float | None], list[dict[str, str]]]:
    selected_metrics = normalize_metric_names(metric_names)
    client, metrics = _build_metrics()
    metric_by_name = {metric.name: metric for metric in metrics}
    faithfulness = metric_by_name["faithfulness"]
    answer_relevancy = metric_by_name["answer_relevancy"]
    answer_correctness = metric_by_name["answer_correctness"]
    context_relevance = metric_by_name["context_relevance"]
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

            operations: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
            if "answer_relevancy" in selected_metrics:
                operations.append(
                    (
                        "answer_relevancy",
                        lambda detail=detail: answer_relevancy.ascore(
                            user_input=detail.question,
                            response=detail.answer or "",
                        ),
                    )
                )
            if "answer_correctness" in selected_metrics:
                operations.append(
                    (
                        "answer_correctness",
                        lambda detail=detail: answer_correctness.ascore(
                            user_input=detail.question,
                            response=detail.answer or "",
                            reference=detail.ground_truth,
                        ),
                    )
                )
            if detail.retrieved_contexts and "faithfulness" in selected_metrics:
                operations.append(
                    (
                        "faithfulness",
                        lambda detail=detail: faithfulness.ascore(
                            user_input=detail.question,
                            response=detail.answer or "",
                            retrieved_contexts=detail.retrieved_contexts,
                        ),
                    )
                )
            if detail.retrieved_contexts and "context_relevance" in selected_metrics:
                operations.append(
                    (
                        "context_relevance",
                        lambda detail=detail: context_relevance.ascore(
                            user_input=detail.question,
                            retrieved_contexts=detail.retrieved_contexts,
                        ),
                    )
                )

            if not operations:
                continue

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

    return _summarize_metrics(details, list(selected_metrics)), errors


async def score_existing_report(
    report: EvalReport,
    metric_names: Sequence[str],
) -> EvalReport:
    """Score selected metrics using answers and contexts already in a report."""
    selected_metrics = normalize_metric_names(metric_names)
    for detail in report.details:
        for name in selected_metrics:
            detail.scores.pop(name, None)
    report.errors = [
        error for error in report.errors if error.get("metric") not in selected_metrics
    ]

    scoreable_details = [detail for detail in report.details if detail.answer and not detail.error]
    metric_summary, metric_errors = await _score_details(scoreable_details, selected_metrics)
    report.summary.metrics.update(metric_summary)
    report.errors.extend(metric_errors)
    return report


async def run_ragas_evaluation(
    samples: list[EvalSample],
    metric_names: Sequence[str] | None = None,
) -> EvalReport:
    details: list[EvalDetail] = []
    errors: list[dict[str, str]] = []

    for sample in samples:
        detail = EvalDetail(
            id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            relevant_sources=relevant_sources_from_metadata(sample.metadata),
        )
        details.append(detail)

        try:
            result = await generate_answer_with_context(
                sample.question,
                session_id=f"eval-{sample.id}-{uuid4().hex}",
            )
            detail.answer = result.answer
            detail.retrieved_contexts = result.retrieved_contexts
            detail.retrieval_attempted = result.retrieval_attempted
            detail.retrieval_candidate_sources = result.retrieval_candidate_sources
            detail.reranked_sources = result.reranked_sources
        except EvalAnswerGenerationError as exc:
            detail.error = str(exc)
            errors.append({"id": sample.id, "error": str(exc)})
        finally:
            detail.scores.update(_score_retrieval(detail))

    successful_details = [detail for detail in details if detail.answer and not detail.error]
    if not successful_details:
        raise ValueError("All evaluation samples failed to generate answers")

    selected_metrics = normalize_metric_names(metric_names)
    if metric_names is None:
        metric_summary, metric_errors = await _score_details(successful_details)
    else:
        metric_summary, metric_errors = await _score_details(
            successful_details,
            selected_metrics,
        )
    retrieval_metric_names = list(_retrieval_metric_names())
    metric_summary.update(_summarize_metrics(details, retrieval_metric_names))
    errors.extend(metric_errors)

    summary = EvalSummary(
        total=len(samples),
        success=len(successful_details),
        failed=len(samples) - len(successful_details),
        metrics=metric_summary,
    )
    return EvalReport(summary=summary, details=details, errors=errors)
