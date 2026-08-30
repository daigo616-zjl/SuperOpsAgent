import asyncio
from dataclasses import dataclass

from ragas.metrics.collections import Faithfulness

from app.eval.ragas_runner import (
    BatchedFaithfulness,
    EvalReport,
    _build_metrics,
    _metric_error_message,
    _metric_timeout,
    score_existing_report,
)


def test_ragas_metrics_have_required_models() -> None:
    client, metrics = _build_metrics()
    try:
        by_name = {metric.name: metric for metric in metrics}

        assert by_name["faithfulness"].llm is not None
        assert by_name["answer_relevancy"].llm is not None
        assert by_name["answer_relevancy"].embeddings is not None
        assert by_name["answer_correctness"].llm is not None
        assert by_name["answer_correctness"].embeddings is not None
        assert by_name["context_relevance"].llm is not None
        assert by_name["faithfulness"].statement_batch_size == 10
    finally:
        asyncio.run(client.close())


def test_complex_metrics_use_dedicated_timeouts(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.eval.ragas_runner.config.eval_metric_timeout",
        90,
    )
    monkeypatch.setattr(
        "app.eval.ragas_runner.config.eval_faithfulness_timeout",
        360,
    )
    monkeypatch.setattr(
        "app.eval.ragas_runner.config.eval_answer_correctness_timeout",
        300,
    )

    assert _metric_timeout("answer_relevancy") == 90
    assert _metric_timeout("context_relevance") == 90
    assert _metric_timeout("faithfulness") == 360
    assert _metric_timeout("answer_correctness") == 300


def test_timeout_error_message_is_actionable() -> None:
    assert _metric_error_message(TimeoutError(), 300) == (
        "TimeoutError: metric exceeded 300s"
    )


def test_faithfulness_verdicts_are_batched_and_merged(monkeypatch) -> None:
    batch_sizes: list[int] = []

    @dataclass
    class FakeResult:
        statements: list[str]

        def model_copy(self, *, update: dict[str, list[str]]) -> "FakeResult":
            return FakeResult(statements=update["statements"])

    async def fake_create_verdicts(
        _metric: Faithfulness,
        statements: list[str],
        _context: str,
    ) -> FakeResult:
        batch_sizes.append(len(statements))
        return FakeResult(statements=statements)

    monkeypatch.setattr(Faithfulness, "_create_verdicts", fake_create_verdicts)
    metric = object.__new__(BatchedFaithfulness)
    metric.statement_batch_size = 10
    statements = [f"statement-{index}" for index in range(23)]

    result = asyncio.run(metric._create_verdicts(statements, "context"))

    assert batch_sizes == [10, 10, 3]
    assert result.statements == statements


def test_existing_report_can_be_rescored_without_regenerating_answers(monkeypatch) -> None:
    report = EvalReport.from_dict(
        {
            "summary": {
                "total": 1,
                "success": 1,
                "failed": 0,
                "metrics": {"answer_relevancy": 0.9},
            },
            "details": [
                {
                    "id": "case-001",
                    "question": "question",
                    "ground_truth": "truth",
                    "answer": "answer",
                    "retrieved_contexts": ["context"],
                    "scores": {
                        "faithfulness": 0.1,
                        "answer_correctness": 0.2,
                    },
                }
            ],
            "errors": [
                {"id": "case-001", "metric": "faithfulness", "error": "old"}
            ],
        }
    )
    captured: list[tuple[str, ...]] = []

    async def fake_score_details(details, metric_names):
        assert len(details) == 1
        captured.append(tuple(metric_names))
        details[0].scores.update({"faithfulness": 0.8, "answer_correctness": 0.7})
        return {"faithfulness": 0.8, "answer_correctness": 0.7}, []

    monkeypatch.setattr("app.eval.ragas_runner._score_details", fake_score_details)

    result = asyncio.run(
        score_existing_report(report, ("faithfulness", "answer_correctness"))
    )

    assert captured == [("faithfulness", "answer_correctness")]
    assert result.summary.metrics == {
        "answer_relevancy": 0.9,
        "faithfulness": 0.8,
        "answer_correctness": 0.7,
    }
    assert result.details[0].scores["faithfulness"] == 0.8
    assert result.details[0].scores["answer_correctness"] == 0.7
