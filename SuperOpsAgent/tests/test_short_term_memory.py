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
    import app.services.rag_agent_service as ras_module
    from app.memory.models import MemoryContext
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

    async def recall(question, user_id):
        return MemoryContext()

    monkeypatch.setattr(st_module.short_term_memory, "available", available)
    monkeypatch.setattr(st_module.short_term_memory, "next_seq", next_seq)
    monkeypatch.setattr(st_module.short_term_memory, "build_context", build_context)
    monkeypatch.setattr(ras_module.memory_recall_service, "recall", recall)

    turn = await service._start_turn("sess1", "当前问题", "sess1")
    assert turn.memory_mode is True
    assert turn.thread_id == "sess1:7"
    # [system, 摘要 system, 窗口 q1/a1, 当前问题]
    assert len(turn.messages) == 5
    assert "历史对话摘要" in turn.messages[1].content
    assert turn.messages[-1].content == "当前问题"


async def test_start_turn_degraded_mode(monkeypatch):
    import app.services.rag_agent_service as ras_module
    from app.memory.models import MemoryContext, MemoryHit
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)

    async def unavailable():
        return False

    recalled_with: dict[str, str] = {}

    async def recall(question, user_id):
        recalled_with["question"] = question
        recalled_with["user_id"] = user_id
        return MemoryContext(facts=[MemoryHit(content="服务 A 部署在华南1", subject="服务-A-部署")])

    monkeypatch.setattr(st_module.short_term_memory, "available", unavailable)
    monkeypatch.setattr(ras_module.memory_recall_service, "recall", recall)

    turn = await service._start_turn("sess1", "当前问题", "u1")
    assert turn.memory_mode is False
    assert turn.thread_id == "sess1"
    assert len(turn.messages) == 2
    # 降级路径虽不走 Redis，但长期记忆召回不依赖 Redis，应照常注入
    assert "长期记忆-强事实" in turn.messages[0].content
    assert "服务 A 部署在华南1" in turn.messages[0].content
    assert recalled_with == {"question": "当前问题", "user_id": "u1"}


class _FakeUsersEngine:
    """只服务 users 存在性查询的假 engine；db_fail 时模拟基础设施故障"""

    def __init__(self, exists: bool, db_fail: Exception | None = None):
        self._exists = exists
        self._db_fail = db_fail
        self.queries = 0

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, stmt, params):
        self.queries += 1
        if self._db_fail is not None:
            raise self._db_fail
        return SimpleNamespace(scalar=lambda: 1 if self._exists else None)


def test_resolve_user_id_fallback_without_user(monkeypatch):
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)
    # 缺省 / 与 session 相同时直接回退，不触碰数据库
    assert service._resolve_user_id("sess1", None) == "sess1"
    assert service._resolve_user_id("sess1", "") == "sess1"
    assert service._resolve_user_id("sess1", "sess1") == "sess1"


def test_resolve_user_id_existing(monkeypatch):
    import uuid as uuid_mod

    from app.core import postgres as pg_module
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)
    user_id = str(uuid_mod.uuid4())
    engine = _FakeUsersEngine(exists=True)
    monkeypatch.setattr(pg_module.postgres_manager, "_engine", engine, raising=False)
    assert service._resolve_user_id("sess1", user_id) == user_id
    assert engine.queries == 1


def test_resolve_user_id_missing_user(monkeypatch):
    import uuid as uuid_mod

    import pytest

    from app.core import postgres as pg_module
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)
    user_id = str(uuid_mod.uuid4())
    monkeypatch.setattr(
        pg_module.postgres_manager, "_engine", _FakeUsersEngine(exists=False), raising=False
    )
    with pytest.raises(ValueError, match="用户不存在"):
        service._resolve_user_id("sess1", user_id)


def test_resolve_user_id_invalid_format(monkeypatch):
    import pytest

    from app.core import postgres as pg_module
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)
    engine = _FakeUsersEngine(exists=True)
    monkeypatch.setattr(pg_module.postgres_manager, "_engine", engine, raising=False)
    with pytest.raises(ValueError, match="格式非法"):
        service._resolve_user_id("sess1", "not-a-uuid")
    assert engine.queries == 0  # 格式错误不应打数据库


def test_resolve_user_id_db_failure_not_valueerror(monkeypatch):
    import uuid as uuid_mod

    import pytest

    from app.core import postgres as pg_module
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)
    user_id = str(uuid_mod.uuid4())
    boom = RuntimeError("connection refused")
    monkeypatch.setattr(
        pg_module.postgres_manager, "_engine", _FakeUsersEngine(True, db_fail=boom), raising=False
    )
    with pytest.raises(RuntimeError):
        service._resolve_user_id("sess1", user_id)


async def test_query_stream_valueerror_surfaces_as_error_event(monkeypatch):
    from app.services.rag_agent_service import RagAgentService

    service = RagAgentService(streaming=False)

    async def no_init():
        return None

    def resolve(session_id, user_id):
        raise ValueError("用户不存在: u1")

    monkeypatch.setattr(service, "_initialize_agent", no_init)
    monkeypatch.setattr(service, "_resolve_user_id", resolve)

    chunks = [c async for c in service.query_stream("q", "sess1", user_id="u1")]
    types = [c["type"] for c in chunks]
    assert "error" in types, f"应产生 error 事件, 实际 {types}"
    assert all(t != "content" for t in types), "请求侧错误不应伪装成模型不可用文案"
    assert types[-1] == "complete"
