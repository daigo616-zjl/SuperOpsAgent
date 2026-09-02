"""短期记忆单元测试：滑动窗口压缩、熔断降级、轮次上下文组装"""

from types import SimpleNamespace

import pytest

from app.memory import short_term as st_module
from app.memory.short_term import ShortTermMemory


class FakePipeline:
    def __init__(self, store):
        self._store = store
        self._ops = []

    def rpush(self, key, value):
        self._ops.append(("rpush", key, (value,)))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", key, (ttl,)))
        return self

    def set(self, key, value, ex=None):
        self._ops.append(("set", key, (value,)))
        return self

    def incr(self, key):
        self._ops.append(("incr", key, ()))
        return self

    def ltrim(self, key, start, end):
        self._ops.append(("ltrim", key, (start, end)))
        return self

    async def execute(self):
        for op, key, args in self._ops:
            if op == "rpush":
                self._store.setdefault(key, []).append(args[0])
            elif op == "expire":
                self._store[f"ttl:{key}"] = args[0]
            elif op == "set":
                self._store[key] = args[0]
            elif op == "incr":
                self._store[key] = self._store.get(key, 0) + 1
            elif op == "ltrim":
                start, end = args
                values = self._store.get(key, [])
                self._store[key] = values[start:] if end == -1 else values[start : end + 1]
        return [True] * len(self._ops)


class FakeRedis:
    """仅实现 ShortTermMemory 用到的子集"""

    def __init__(self):
        self.data: dict = {}

    def pipeline(self):
        return FakePipeline(self.data)

    async def incr(self, key):
        self.data[key] = self.data.get(key, 0) + 1
        return self.data[key]

    async def rpush(self, key, value):
        self.data.setdefault(key, []).append(value)

    async def lrange(self, key, start, end):
        values = self.data.get(key, [])
        if end == -1:
            return values[start:]
        return values[start : end + 1]

    async def llen(self, key):
        return len(self.data.get(key, []))

    async def ltrim(self, key, start, end):
        values = self.data.get(key, [])
        self.data[key] = values[start:] if end == -1 else values[start : end + 1]

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, nx=False, px=None):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
        return count


class FakeSummarizer:
    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    async def summarize(self, old_summary, messages):
        self.calls.append((old_summary, list(messages)))
        return f"merged({old_summary!r}+{len(messages)}条)"


@pytest.fixture
def memory():
    stm = ShortTermMemory(client=FakeRedis(), summarizer=FakeSummarizer(), enabled=True)
    stm._reset()
    return stm


async def test_append_and_build_roundtrip(memory):
    assert await memory.available() is True
    assert await memory.append_turn("s1", "问题1", "答案1") is True

    summary, window = await memory.build_context("s1")
    assert summary == ""
    assert [m["role"] for m in window] == ["user", "assistant"]
    assert window[0]["content"] == "问题1"

    seq = await memory.next_seq("s1")
    assert seq == 1


async def test_compression_triggers_at_window(memory, monkeypatch):
    monkeypatch.setattr(st_module.config, "memory_window_messages", 4)
    monkeypatch.setattr(st_module.config, "memory_compress_keep", 2)

    await memory.append_turn("s1", "q1", "a1")
    _, window = await memory.build_context("s1")
    assert len(window) == 2  # 未达阈值不压缩

    await memory.append_turn("s1", "q2", "a2")
    summary, window = await memory.build_context("s1")
    # 4 条达到阈值，压缩最旧 2 条，保留尾部 2 条
    assert len(window) == 2
    assert window[0]["content"] == "q2"
    assert summary == "merged(''+2条)"
    assert len(memory._summarizer.calls) == 1


async def test_history_limit(memory):
    await memory.append_turn("s1", "q1", "a1")
    await memory.append_turn("s1", "q2", "a2")
    history = await memory.history("s1", limit=2)
    assert len(history) == 2
    assert history[-1]["content"] == "a2"


async def test_clear(memory):
    await memory.append_turn("s1", "q1", "a1")
    assert await memory.clear("s1") is True
    summary, window = await memory.build_context("s1")
    assert summary == ""
    assert window == []


async def test_breaker_trips_and_recovers(memory, monkeypatch):
    import app.memory.redis_client as rc_module

    async def fail_connect():
        raise ConnectionError("cannot connect")

    monkeypatch.setattr(
        rc_module,
        "redis_client_manager",
        SimpleNamespace(connected=False, connect=fail_connect, get_client=lambda: None),
    )
    memory._client = None

    # 触发故障 → 熔断
    ok = await memory.append_turn("s1", "q", "a")
    assert ok is False
    assert memory.tripped is True
    # 冷却期内不可用（不触碰 Redis）
    assert await memory.available() is False


async def test_breaker_half_open_recovery(memory):
    class FlakyRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.fail = True

        def pipeline(self):
            if self.fail:
                raise ConnectionError("still down")
            return super().pipeline()

    flaky = FlakyRedis()
    memory._client = flaky
    ok = await memory.append_turn("s1", "q", "a")
    assert ok is False and memory.tripped is True
    assert await memory.available() is False

    # 冷却期满（模拟时间流逝）且 Redis 恢复 → 半开探测成功
    memory._open_until = 0.0
    flaky.fail = False
    assert await memory.available() is True


async def test_disabled_memory_unavailable():
    stm = ShortTermMemory(client=FakeRedis(), enabled=False)
    assert await stm.available() is False


async def test_start_turn_memory_mode(monkeypatch):
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)

    async def available():
        return True

    async def next_seq(sid):
        return 7

    async def build_context(sid):
        return "用户之前咨询过 CPU 告警。", [
            {"role": "user", "content": "q1", "ts": "t"},
            {"role": "assistant", "content": "a1", "ts": "t"},
        ]

    monkeypatch.setattr(st_module.short_term_memory, "available", available)
    monkeypatch.setattr(st_module.short_term_memory, "next_seq", next_seq)
    monkeypatch.setattr(st_module.short_term_memory, "build_context", build_context)

    turn = await service._start_turn("sess1", "当前问题")
    assert turn.memory_mode is True
    assert turn.thread_id == "sess1:7"
    # [system, 摘要 system, 窗口 q1/a1, 当前问题]
    assert len(turn.messages) == 5
    assert "历史对话摘要" in turn.messages[1].content
    assert turn.messages[-1].content == "当前问题"


async def test_start_turn_degraded_mode(monkeypatch):
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)

    async def unavailable():
        return False

    monkeypatch.setattr(st_module.short_term_memory, "available", unavailable)

    turn = await service._start_turn("sess1", "当前问题")
    assert turn.memory_mode is False
    assert turn.thread_id == "sess1"
    assert len(turn.messages) == 2
