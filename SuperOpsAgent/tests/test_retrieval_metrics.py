import asyncio

import app.eval.ragas_runner as runner_module
from app.eval.answer_generator import EvalAnswerResult
from app.eval.dataset import EvalSample
from app.eval.ragas_runner import EvalDetail, _score_retrieval, normalize_metric_names
from app.eval.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    relevant_sources_from_metadata,
)


def test_relevant_sources_supports_single_and_multiple_documents() -> None:
    assert relevant_sources_from_metadata({"source": "docs/cpu_high_usage.md"}) == [
        "cpu_high_usage.md"
    ]
    assert relevant_sources_from_metadata(
        {"relevant_sources": ["CPU_HIGH_USAGE.md", "docs/disk_high_usage.md"]}
    ) == ["cpu_high_usage.md", "disk_high_usage.md"]


def test_recall_and_hit_use_ranked_source_labels() -> None:
    relevant = ["cpu.md", "disk.md"]
    retrieved = ["memory.md", "C:\\docs\\CPU.md", "network.md", "disk.md"]

    assert recall_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 4) == 1.0
    assert hit_at_k(relevant, retrieved, 1) == 0.0
    assert hit_at_k(relevant, retrieved, 2) == 1.0


def test_retrieval_metrics_are_none_without_relevance_labels() -> None:
    assert recall_at_k([], ["cpu.md"], 20) is None
    assert hit_at_k([], ["cpu.md"], 5) is None


def test_metric_selection_is_normalized_and_validated() -> None:
    assert normalize_metric_names([" Faithfulness ", "faithfulness", "answer_correctness"]) == (
        "faithfulness",
        "answer_correctness",
    )

    try:
        normalize_metric_names(["unknown"])
    except ValueError as exc:
        assert "Unsupported metrics" in str(exc)
    else:
        raise AssertionError("unsupported metric should fail")


def test_eval_detail_scores_recall_candidates_and_reranked_hit(monkeypatch) -> None:
    monkeypatch.setattr("app.eval.ragas_runner.config.eval_recall_k", 20)
    monkeypatch.setattr("app.eval.ragas_runner.config.eval_hit_k", 5)
    detail = EvalDetail(
        id="case-001",
        question="question",
        ground_truth="answer",
        relevant_sources=["cpu.md", "disk.md"],
        retrieval_candidate_sources=["cpu.md", "memory.md"],
        reranked_sources=["memory.md", "cpu.md"],
    )

    assert _score_retrieval(detail) == {
        "recall_at_20": 0.5,
        "hit_at_5": 1.0,
    }


def test_evaluation_report_includes_retrieval_metric_summary(monkeypatch) -> None:
    async def fake_generate_answer_with_context(
        _question: str,
        session_id: str,
    ) -> EvalAnswerResult:
        assert session_id.startswith("eval-case-001-")
        return EvalAnswerResult(
            answer="answer",
            retrieved_contexts=["context"],
            retrieval_attempted=True,
            retrieval_candidate_sources=["cpu.md", "memory.md"],
            reranked_sources=["memory.md", "cpu.md"],
        )

    async def fake_score_details(
        _details: list[EvalDetail],
    ) -> tuple[dict[str, float | None], list[dict[str, str]]]:
        return {"faithfulness": 1.0}, []

    monkeypatch.setattr(
        runner_module,
        "generate_answer_with_context",
        fake_generate_answer_with_context,
    )
    monkeypatch.setattr(runner_module, "_score_details", fake_score_details)
    sample = EvalSample(
        id="case-001",
        question="question",
        ground_truth="answer",
        metadata={"source": "cpu.md"},
    )

    report = asyncio.run(runner_module.run_ragas_evaluation([sample]))

    assert report.summary.metrics == {
        "faithfulness": 1.0,
        "recall_at_20": 1.0,
        "hit_at_5": 1.0,
    }
    assert report.details[0].retrieval_attempted is True
    assert report.details[0].relevant_sources == ["cpu.md"]
