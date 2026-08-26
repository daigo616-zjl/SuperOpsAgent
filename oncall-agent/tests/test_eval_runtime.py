import asyncio
from typing import Any

import pytest

import app.eval.runtime as runtime_module
from app.eval.runtime import EvaluationRuntimeError, evaluation_runtime


def test_evaluation_runtime_connects_and_closes_stores(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        runtime_module.milvus_manager,
        "connect",
        lambda: events.append("milvus_connect"),
    )
    monkeypatch.setattr(
        runtime_module.milvus_manager,
        "close",
        lambda: events.append("milvus_close"),
    )

    async def es_connect() -> None:
        events.append("es_connect")

    async def es_close() -> None:
        events.append("es_close")

    monkeypatch.setattr(runtime_module.es_client_manager, "connect", es_connect)
    monkeypatch.setattr(runtime_module.es_client_manager, "close", es_close)

    async def run() -> None:
        async with evaluation_runtime():
            events.append("evaluate")

    asyncio.run(run())

    assert events == [
        "milvus_connect",
        "es_connect",
        "evaluate",
        "es_close",
        "milvus_close",
    ]


def test_evaluation_runtime_reports_milvus_lock(monkeypatch) -> None:
    def fail_connect() -> Any:
        raise RuntimeError("data directory is locked")

    monkeypatch.setattr(runtime_module.milvus_manager, "connect", fail_connect)

    async def run() -> None:
        async with evaluation_runtime():
            pass

    with pytest.raises(EvaluationRuntimeError, match="停止 FastAPI"):
        asyncio.run(run())


def test_evaluation_runtime_cleans_up_when_es_connect_fails(monkeypatch) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        runtime_module.milvus_manager,
        "connect",
        lambda: events.append("milvus_connect"),
    )
    monkeypatch.setattr(
        runtime_module.milvus_manager,
        "close",
        lambda: events.append("milvus_close"),
    )

    async def es_connect() -> None:
        events.append("es_connect")
        raise RuntimeError("connection refused")

    async def es_close() -> None:
        events.append("es_close")

    monkeypatch.setattr(runtime_module.es_client_manager, "connect", es_connect)
    monkeypatch.setattr(runtime_module.es_client_manager, "close", es_close)

    async def run() -> None:
        async with evaluation_runtime():
            pass

    with pytest.raises(EvaluationRuntimeError, match="Elasticsearch"):
        asyncio.run(run())

    assert events == ["milvus_connect", "es_connect", "es_close", "milvus_close"]
