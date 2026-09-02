"""长期记忆存储层测试：迁移登记、强事实仓储行为、幂等锚点

不依赖真实 PG/ES/Milvus：仓储测试用 FakeEngine 捕获 SQL 与参数。
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.core.postgres import MIGRATIONS
from app.memory import long_term_repository as ltr_module
from app.memory.long_term_repository import LongTermRepository
from app.memory.models import MemoryContext, content_hash

ROOT = Path(__file__).resolve().parents[1]


class FakeRow:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping


class FakeConnection:
    def __init__(self, calls: list, rows: list[FakeRow] | None = None) -> None:
        self._calls = calls
        self._rows = rows or []

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(str(sql).split()).lower()
        self._calls.append((normalized, params))
        if normalized.lstrip().startswith("select"):
            return SimpleNamespace(all=lambda: self._rows)
        return SimpleNamespace(rowcount=1)


class _Transaction:
    def __init__(self, calls: list) -> None:
        self._connection = FakeConnection(calls)

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, *args) -> bool:
        return False


class FakeEngine:
    def __init__(self, calls: list) -> None:
        self._calls = calls
        self.transactions = 0

    def begin(self) -> _Transaction:
        self.transactions += 1
        return _Transaction(self._calls)

    def connect(self) -> FakeConnection:
        return FakeConnection(self._calls)


# ------------------------------------------------------------
# 迁移登记与 schema
# ------------------------------------------------------------

def test_migration_003_registered_and_covers_tables() -> None:
    sql_files = sorted(p.name for p in (ROOT / "migrations").glob("*.sql"))
    assert MIGRATIONS == sql_files

    schema = (ROOT / "migrations" / "003_rag_memory.sql").read_text(encoding="utf-8")
    for table in ("rag_memory_facts", "rag_memory_jobs"):
        assert f"create table if not exists {table}" in schema
    assert "status in ('active', 'superseded')" in schema
    assert "where status = 'active'" in schema  # 部分唯一索引
    assert "using gin (keywords)" in schema
    assert "status in ('pending', 'processing', 'done', 'dead')" in schema


# ------------------------------------------------------------
# 幂等锚点
# ------------------------------------------------------------

def test_content_hash_is_stable_and_normalized() -> None:
    assert content_hash("u1", "用户服务是 data-sync-service") == content_hash(
        "u1", "用户服务是  Data-Sync-Service "
    )
    assert content_hash("u1", "a") != content_hash("u2", "a")


# ------------------------------------------------------------
# 强事实仓储（FakeEngine）
# ------------------------------------------------------------

def _patch_engine(monkeypatch, rows: list[FakeRow] | None = None):
    calls: list = []
    engine = FakeEngine(calls)
    if rows:
        engine._rows = rows

        def connect_with_rows():
            return FakeConnection(calls, rows)

        engine.connect = connect_with_rows  # type: ignore[method-assign]
    monkeypatch.setattr(ltr_module.postgres_manager, "_engine", engine)
    return LongTermRepository(), calls


def test_upsert_fact_supersedes_then_inserts(monkeypatch) -> None:
    repository, calls = _patch_engine(monkeypatch)

    c_hash = repository.upsert_fact(
        "user-1", "生产环境 MySQL 主库在华东1", subject="数据库", keywords=["mysql", "华东1"],
    )

    assert len(c_hash) == 64
    supersede_sql, supersede_params = calls[0]
    assert supersede_sql.startswith("update rag_memory_facts")
    assert "status = 'superseded'" in supersede_sql
    assert supersede_params["subject"] == "数据库"

    insert_sql, insert_params = calls[1]
    assert insert_sql.startswith("insert into rag_memory_facts")
    assert "on conflict (user_id, content_hash) where status = 'active' do nothing" in insert_sql
    assert insert_params["keywords"] == ["mysql", "华东1"]


def test_match_facts_requires_terms(monkeypatch) -> None:
    repository, calls = _patch_engine(monkeypatch)
    assert repository.match_facts("user-1", []) == []
    assert calls == []


def test_match_facts_builds_ilike_and_array_overlap(monkeypatch) -> None:
    rows = [
        FakeRow({"content": "用户服务是 data-sync-service", "subject": "服务", "content_hash": "h1"}),
    ]
    repository, calls = _patch_engine(monkeypatch, rows=rows)

    hits = repository.match_facts("user-1", ["data-sync"], limit=5)

    assert len(hits) == 1
    assert hits[0].content == "用户服务是 data-sync-service"
    sql, params = calls[0]
    assert "ilike any" in sql
    assert "keywords &&" in sql
    assert params["patterns"] == ["%data-sync%"]
    assert params["limit"] == 5


def test_recent_facts_returns_ordered_hits(monkeypatch) -> None:
    rows = [
        FakeRow({"content": "事实B", "subject": "", "content_hash": "h2"}),
        FakeRow({"content": "事实A", "subject": "s", "content_hash": "h1"}),
    ]
    repository, calls = _patch_engine(monkeypatch, rows=rows)

    hits = repository.recent_facts("user-1", limit=2)

    assert [h.content for h in hits] == ["事实B", "事实A"]
    sql, _ = calls[0]
    assert "order by updated_at desc" in sql


def test_memory_context_priority_fields() -> None:
    from app.memory.models import MemoryHit

    ctx = MemoryContext()
    assert ctx.is_empty() is True
    ctx.facts.append(MemoryHit(content="f"))
    assert ctx.is_empty() is False
