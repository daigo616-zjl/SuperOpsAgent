from app.services.rerank_service import RerankService


def test_rerank_warmup_runs_minimal_prediction(monkeypatch) -> None:
    service = RerankService()
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        "app.services.rerank_service.config.rag_rerank_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.services.rerank_service.config.rag_rerank_warmup_enabled",
        True,
    )
    monkeypatch.setattr(
        service,
        "_predict_scores",
        lambda query, docs: calls.append((query, len(docs))) or [0.5],
    )

    assert service.warmup() is True
    assert calls == [("重排模型预热", 1)]


def test_rerank_warmup_skips_when_disabled(monkeypatch) -> None:
    service = RerankService()

    monkeypatch.setattr(
        "app.services.rerank_service.config.rag_rerank_enabled",
        True,
    )
    monkeypatch.setattr(
        "app.services.rerank_service.config.rag_rerank_warmup_enabled",
        False,
    )

    assert service.warmup() is False
