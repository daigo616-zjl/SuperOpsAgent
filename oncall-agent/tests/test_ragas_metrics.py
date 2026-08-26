import asyncio
from dataclasses import dataclass

from ragas.metrics.collections import Faithfulness

from app.eval.ragas_runner import (
    BatchedFaithfulness,
    _build_metrics,
    _metric_error_message,
    _metric_timeout,
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
        300,
    )
    monkeypatch.setattr(
        "app.eval.ragas_runner.config.eval_answer_correctness_timeout",
        240,
    )

    assert _metric_timeout("answer_relevancy") == 90
    assert _metric_timeout("context_relevance") == 90
    assert _metric_timeout("faithfulness") == 300
    assert _metric_timeout("answer_correctness") == 240


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
