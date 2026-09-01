import asyncio

import pytest
from langchain_core.messages import AIMessageChunk

from app.core.llm_resilience import (
    CircuitOpenError,
    ResilientChatModel,
    StallTimeoutError,
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


def make_model(model, fallback=None, retries=2, threshold=3, timeout=0.05, **kwargs):
    return ResilientChatModel(
        model,
        model_name="primary",
        timeout=timeout,
        max_retries=retries,
        rate_limiter=_RateLimiter(2, 0),
        breaker=_CircuitBreaker(threshold, 60),
        fallback=fallback,
        retry_backoff=0,
        **kwargs,
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


class FakeStreamingModel:
    """脚本化流式模型：float 项为睡眠秒数，异常项直接抛出，其余项为 chunk 内容。"""

    def __init__(self, scripts):
        self.scripts = iter(scripts)
        self.calls = 0

    async def astream(self, *_args, **_kwargs):
        self.calls += 1
        for action in next(self.scripts):
            if isinstance(action, float):
                await asyncio.sleep(action)
            elif isinstance(action, BaseException):
                raise action
            else:
                yield AIMessageChunk(content=action)


@pytest.mark.asyncio
async def test_streaming_ainvoke_aggregates_chunks_into_message():
    model = make_model(
        FakeStreamingModel([["he", "llo", ""]]), retries=0, timeout=5.0, stall_timeout=1.0
    )
    result = await model.ainvoke("prompt")
    assert result.content == "hello"
    assert type(result).__name__ == "AIMessage"


@pytest.mark.asyncio
async def test_streaming_stall_before_first_chunk_aborts_quickly():
    model = make_model(
        FakeStreamingModel([[5.0, "late"]]), retries=0, timeout=30.0, stall_timeout=0.05
    )
    with pytest.raises(StallTimeoutError):
        await model.ainvoke("prompt")


@pytest.mark.asyncio
async def test_streaming_midstream_stall_is_retried_and_recovers():
    primary = FakeStreamingModel(
        [
            ["a", 1.0, "b"],
            ["he", "llo"],
        ]
    )
    model = make_model(primary, retries=1, timeout=30.0, stall_timeout=0.05)
    result = await model.ainvoke("prompt")
    assert result.content == "hello"
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_streaming_empty_stream_raises_stall():
    model = make_model(FakeStreamingModel([[]]), retries=0, timeout=5.0, stall_timeout=1.0)
    with pytest.raises(StallTimeoutError):
        await model.ainvoke("prompt")


@pytest.mark.asyncio
async def test_streaming_non_chunk_objects_returns_last():
    class DictStream:
        async def astream(self, *_args, **_kwargs):
            yield {"partial": 1}
            yield {"final": 2}

    model = make_model(DictStream(), retries=0, timeout=5.0, stall_timeout=1.0)
    assert await model.ainvoke("prompt") == {"final": 2}
