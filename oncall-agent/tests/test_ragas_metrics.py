import asyncio

from app.eval.ragas_runner import _build_metrics


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
    finally:
        asyncio.run(client.close())
