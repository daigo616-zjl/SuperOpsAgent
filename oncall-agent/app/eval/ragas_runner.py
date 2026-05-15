from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness

from app.config import config
from app.core.llm_factory import llm_factory
from app.eval.answer_generator import EvalAnswerGenerationError, generate_answer
from app.eval.dataset import EvalSample


@dataclass(slots=True)
class EvalDetail:
    id: str
    question: str
    ground_truth: str
    answer: str | None = None
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


def _build_evaluator_llm() -> LangchainLLMWrapper:
    llm = llm_factory.create_chat_model(
        model=config.eval_model,
        temperature=0.0,
        streaming=False,
    )
    return LangchainLLMWrapper(llm)


def _with_llm(metric: Any, evaluator_llm: LangchainLLMWrapper) -> Any:
    if callable(metric):
        try:
            return metric(llm=evaluator_llm)
        except TypeError:
            try:
                return metric()
            except TypeError:
                return metric
    return metric


def _build_metrics() -> list[Any]:
    evaluator_llm = _build_evaluator_llm()
    metrics: list[Any] = [
        _with_llm(faithfulness, evaluator_llm),
        _with_llm(answer_relevancy, evaluator_llm),
    ]

    try:
        metrics.append(_with_llm(answer_correctness, evaluator_llm))
    except Exception:
        pass

    return metrics


def _build_dataset_payload(details: list[EvalDetail]) -> Dataset:
    successful_details = [detail for detail in details if detail.answer and not detail.error]
    return Dataset.from_dict(
        {
            "question": [detail.question for detail in successful_details],
            "answer": [detail.answer for detail in successful_details],
            "ground_truth": [detail.ground_truth for detail in successful_details],
        }
    )


def _extract_metric_scores(result: Any, successful_details: list[EvalDetail]) -> dict[str, float | None]:
    scores = getattr(result, "scores", None) or []
    if not scores:
        return {}

    metric_names = list(scores[0].keys())
    detail_scores: dict[str, list[float | None]] = {name: [] for name in metric_names}

    for detail, row in zip(successful_details, scores, strict=False):
        detail.scores = {name: row.get(name) for name in metric_names}
        for name in metric_names:
            detail_scores[name].append(row.get(name))

    return {
        name: mean([value for value in values if isinstance(value, (int, float))]) if any(isinstance(value, (int, float)) for value in values) else None
        for name, values in detail_scores.items()
    }


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
            detail.answer = await generate_answer(sample.question, session_id=f"eval-{sample.id}")
        except EvalAnswerGenerationError as exc:
            detail.error = str(exc)
            errors.append({"id": sample.id, "error": str(exc)})

    successful_details = [detail for detail in details if detail.answer and not detail.error]
    if not successful_details:
        raise ValueError("All evaluation samples failed to generate answers")

    dataset = _build_dataset_payload(details)
    result = evaluate(dataset=dataset, metrics=_build_metrics())
    metric_summary = _extract_metric_scores(result, successful_details)

    summary = EvalSummary(
        total=len(samples),
        success=len(successful_details),
        failed=len(samples) - len(successful_details),
        metrics=metric_summary,
    )
    return EvalReport(summary=summary, details=details, errors=errors)
