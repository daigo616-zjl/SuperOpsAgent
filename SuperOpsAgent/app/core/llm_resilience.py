"""统一的 LLM 限流、超时、重试、熔断和备用模型策略。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.runnables import Runnable
from loguru import logger


class CircuitOpenError(RuntimeError):
    """模型熔断期间拒绝请求。"""


class _CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: float) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def before_call(self) -> None:
        async with self._lock:
            if self.opened_at is None:
                return
            if time.monotonic() - self.opened_at < self.recovery_timeout:
                raise CircuitOpenError("LLM circuit breaker is open")
            self.opened_at = None
            self.failures = 0

    async def success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.opened_at = None

    async def failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
                logger.warning("LLM circuit opened after {} failures", self.failures)


class _RateLimiter:
    def __init__(self, max_concurrency: int, min_interval: float) -> None:
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self.min_interval = max(0.0, min_interval)
        self.last_started = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> _RateLimiter:
        await self.semaphore.acquire()
        async with self._lock:
            delay = self.min_interval - (time.monotonic() - self.last_started)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last_started = time.monotonic()
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.semaphore.release()


class ResilientChatModel(Runnable[Any, Any]):
    """Runnable-compatible代理，确保每个模型调用走统一保护策略。"""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        timeout: float,
        max_retries: int,
        rate_limiter: _RateLimiter,
        breaker: _CircuitBreaker,
        fallback: ResilientChatModel | None = None,
        retry_backoff: float = 0.25,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.rate_limiter = rate_limiter
        self.breaker = breaker
        self.fallback = fallback
        self.retry_backoff = max(0.0, retry_backoff)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    @staticmethod
    def _is_retryable(error: BaseException) -> bool:
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if isinstance(status, str) and status.isdigit():
            status = int(status)
        if isinstance(status, int):
            return status == 429 or status >= 500
        return isinstance(error, (TimeoutError, asyncio.TimeoutError, OSError)) or any(
            marker in str(error).lower()
            for marker in ("timeout", "timed out", "connection", "connection reset", "429")
        )

    async def _call(self, operation: Callable[[], Any]) -> Any:
        await self.breaker.before_call()
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                async with self.rate_limiter:
                    result = await asyncio.wait_for(operation(), timeout=self.timeout)
                await self.breaker.success()
                return result
            except Exception as error:
                retryable = self._is_retryable(error)
                await self.breaker.failure()
                if not retryable or attempt == attempts - 1:
                    raise
                delay = self.retry_backoff * (2**attempt)
                logger.warning(
                    "LLM call retry: model={}, attempt={}/{}, delay={}s, error={}",
                    self.model_name,
                    attempt + 1,
                    attempts - 1,
                    round(delay, 3),
                    error,
                )
                await asyncio.sleep(delay)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return await self._call(lambda: self.model.ainvoke(input, config=config, **kwargs))
        except Exception as primary_error:
            if self.fallback is None:
                raise
            logger.warning("Primary LLM failed; using fallback model {}: {}", self.fallback.model_name, primary_error)
            return await self.fallback.ainvoke(input, config=config, **kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return asyncio.run(self.ainvoke(input, config=config, **kwargs))

    async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> AsyncIterator[Any]:
        await self.breaker.before_call()
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            emitted = False
            try:
                async with self.rate_limiter, asyncio.timeout(self.timeout):
                    async for chunk in self.model.astream(input, config=config, **kwargs):
                        emitted = True
                        yield chunk
                await self.breaker.success()
                return
            except Exception as error:
                await self.breaker.failure()
                if emitted or not self._is_retryable(error) or attempt == attempts - 1:
                    break
                await asyncio.sleep(self.retry_backoff * (2**attempt))

        if self.fallback is None:
            raise error
        logger.warning("Primary streaming LLM failed; using fallback model {}: {}", self.fallback.model_name, error)
        async for chunk in self.fallback.astream(input, config=config, **kwargs):
            yield chunk

    def bind_tools(self, tools: Any, **kwargs: Any) -> ResilientChatModel:
        fallback = self.fallback.model.bind_tools(tools, **kwargs) if self.fallback else None
        return self._wrap(self.model.bind_tools(tools, **kwargs), fallback)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> ResilientChatModel:
        fallback = self.fallback.model.with_structured_output(schema, **kwargs) if self.fallback else None
        return self._wrap(self.model.with_structured_output(schema, **kwargs), fallback)

    def _wrap(self, bound_model: Any, bound_fallback: Any = None) -> ResilientChatModel:
        return ResilientChatModel(
            bound_model,
            model_name=self.model_name,
            timeout=self.timeout,
            max_retries=self.max_retries,
            rate_limiter=self.rate_limiter,
            breaker=self.breaker,
            fallback=self.fallback._wrap(bound_fallback) if self.fallback and bound_fallback else None,
            retry_backoff=self.retry_backoff,
        )


def build_resilient_model(
    model: Any,
    *,
    model_name: str,
    timeout: float,
    max_retries: int,
    max_concurrency: int,
    min_interval: float,
    failure_threshold: int,
    recovery_timeout: float,
    retry_backoff: float,
    fallback: Any = None,
) -> ResilientChatModel:
    limiter = _RateLimiter(max_concurrency, min_interval)
    breaker = _CircuitBreaker(failure_threshold, recovery_timeout)
    fallback_proxy = None
    if fallback is not None:
        fallback_proxy = ResilientChatModel(
            fallback,
            model_name=getattr(fallback, "model", "fallback"),
            timeout=timeout,
            max_retries=max_retries,
            rate_limiter=limiter,
            breaker=_CircuitBreaker(failure_threshold, recovery_timeout),
            retry_backoff=retry_backoff,
        )
    return ResilientChatModel(
        model,
        model_name=model_name,
        timeout=timeout,
        max_retries=max_retries,
        rate_limiter=limiter,
        breaker=breaker,
        fallback=fallback_proxy,
        retry_backoff=retry_backoff,
    )
