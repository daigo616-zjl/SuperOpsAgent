import asyncio

import pytest

from app.core.llm_resilience import (
    CircuitOpenError,
    ResilientChatModel,
    _CircuitBreaker,
    _RateLimiter,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def ainvoke(self, *_args, **_kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def make_model(model, fallback=None, retries=2, threshold=3):
    return ResilientChatModel(
        model,
        model_name="primary",
        timeout=0.05,
        max_retries=retries,
        rate_limiter=_RateLimiter(2, 0),
        breaker=_CircuitBreaker(threshold, 60),
        fallback=fallback,
        retry_backoff=0,
    )


@pytest.mark.asyncio
async def test_retries_transient_error_then_succeeds():
    primary = FakeModel([TimeoutError("timeout"), RuntimeError("500 server error"), "ok"])
    result = await make_model(primary).ainvoke("prompt")
    assert result == "ok"
    assert primary.calls == 3


@pytest.mark.asyncio
async def test_uses_fallback_after_primary_exhausted():
    primary = FakeModel([RuntimeError("429 rate limited")])
    fallback = FakeModel(["fallback answer"])
    result = await make_model(primary, make_model(fallback, retries=0)).ainvoke("prompt")
    assert result == "fallback answer"
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_calls_after_threshold():
    primary = FakeModel([RuntimeError("500 server error")] * 3)
    model = make_model(primary, retries=0, threshold=2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await model.ainvoke("prompt")
    with pytest.raises(CircuitOpenError):
        await model.ainvoke("prompt")
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_timeout_is_applied_to_llm_call():
    async def slow(*_args, **_kwargs):
        await asyncio.sleep(1)

    model = make_model(FakeModel([]), retries=0)
    model.model.ainvoke = slow
    with pytest.raises(TimeoutError):
        await model.ainvoke("prompt")
