"""P2 Evidence Store 测试：模型约束、append-only 仓储行为、迁移登记与 schema 内容。

不依赖真实 PostgreSQL：仓储测试用 FakeEngine/FakeConnection 捕获 SQL 与参数。
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.aiops.diagnosis_models import (
    AdjudicationDecision,
    BudgetLedger,
    ClaimProvenance,
    Directive,
    Elimination,
    EvidenceCard,
    EvidenceClaim,
    Hypothesis,
)
from app.core import postgres as postgres_module
from app.core.postgres import MIGRATIONS
from app.services import evidence_repository as evidence_module
from app.services.evidence_repository import EvidenceRepository

ROOT = Path(__file__).resolve().parents[1]


def _claim(claim_id: str = "ev-test-1") -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        statement="内存曲线在 10:04 达到 95% 后骤降",
        confidence=0.9,
        polarity="supports",
        hypothesis_ids=["hyp-oom"],
        provenance=ClaimProvenance(
            tool_name="query_memory_metrics",
            args_digest="sha256:abc123",
            output_path="statistics.max",
            excerpt="max=95.2",
        ),
    )


def _card() -> EvidenceCard:
    return EvidenceCard(
        card_id="card-1",
        domain="metrics",
        directive_id="dir-1",
        round=1,
        claims=[_claim("ev-test-1"), _claim("ev-test-2")],
        summary="内存存在 OOMKilled 重启锯齿",
    )


# ------------------------------------------------------------
# 模型约束
# ------------------------------------------------------------

def test_hypothesis_defaults() -> None:
    hypothesis = Hypothesis(id="hyp-1", statement="GC 压力导致超时")
    assert hypothesis.status == "active"
    assert hypothesis.prior == 0.5
    assert hypothesis.ruled_out_by == []


def test_card_rejects_duplicate_claim_ids() -> None:
    with pytest.raises(ValueError, match="claim_id 重复"):
        EvidenceCard(
            card_id="card-1", domain="metrics", directive_id="dir-1", round=1,
            claims=[_claim("ev-dup"), _claim("ev-dup")], summary="s",
        )


def test_claim_id_must_use_ev_prefix() -> None:
    with pytest.raises(ValueError):
        _claim(claim_id="not-a-claim-id")


def test_elimination_requires_claim_reference() -> None:
    with pytest.raises(ValueError):
        Elimination(hypothesis_id="hyp-1", ruled_out_by=[], reason="无证据")
    Elimination(hypothesis_id="hyp-1", ruled_out_by=["ev-1"], reason="指标反驳")


def test_converged_decision_requires_hypothesis_id() -> None:
    with pytest.raises(ValueError, match="converged_hypothesis_id"):
        AdjudicationDecision(converged=True)
    decision = AdjudicationDecision(
        eliminations=[Elimination(hypothesis_id="hyp-1", ruled_out_by=["ev-1"], reason="r")],
        new_directives=[Directive(id="dir-9", target_domain="logs", objective="查 GC 日志")],
        converged=True,
        converged_hypothesis_id="hyp-2",
    )
    assert decision.new_directives[0].target_domain == "logs"


def test_budget_ledger_exhaustion() -> None:
    from datetime import datetime, timedelta

    ledger = BudgetLedger(max_rounds=3, max_wall_seconds=60.0)
    assert not ledger.exhausted()

    ledger.round = 3
    assert ledger.exhausted() and ledger.remaining_rounds() == 0

    clock = {"now": datetime.now() + timedelta(seconds=61)}
    fresh = BudgetLedger(max_rounds=10, max_wall_seconds=60.0)
    assert fresh.wall_exhausted(now=clock["now"])
    assert fresh.exhausted(now=clock["now"])


# ------------------------------------------------------------
# 迁移登记与 schema 内容
# ------------------------------------------------------------

def test_migrations_list_covers_all_sql_files_in_order() -> None:
    sql_files = sorted(p.name for p in (ROOT / "migrations").glob("*.sql"))
    assert MIGRATIONS == sql_files
    assert "002_aiops_evidence.sql" in MIGRATIONS


def test_postgres_initialize_schema_executes_all_migrations(monkeypatch) -> None:
    executed: list[str] = []

    class FakeEngine:
        def begin(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def exec_driver_sql(self, sql: str) -> None:
            executed.append(sql)

    manager = postgres_module.PostgresManager()
    monkeypatch.setattr(manager, "_engine", FakeEngine())
    manager._initialize_schema()

    assert len(executed) == len(MIGRATIONS)
    assert "aiops_evidence_claims" in executed[1]
    assert "rag_memory_facts" in executed[2]
    assert "create table if not exists users" in executed[-1]


def test_evidence_schema_defines_append_only_tables() -> None:
    schema = (ROOT / "migrations" / "002_aiops_evidence.sql").read_text(encoding="utf-8")
    for table in ("aiops_diagnosis_sessions", "aiops_evidence_cards", "aiops_evidence_claims"):
        assert f"create table if not exists {table}" in schema
    assert "domain in ('metrics', 'logs', 'knowledge')" in schema
    assert "numeric(4, 3)" in schema
    assert "hypothesis_ids text[]" in schema
    assert "gin (hypothesis_ids)" in schema


# ------------------------------------------------------------
# 仓储行为（FakeEngine，无真实数据库）
# ------------------------------------------------------------

class FakeRow:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping


class FakeConnection:
    def __init__(self, calls: list, rows: list[FakeRow] | None = None) -> None:
        self._calls = calls
        self._rows = rows or []

    def execute(self, sql, params=None):
        normalized = " ".join(str(sql).split()).lower()
        self._calls.append((normalized, params))
        if "returning" in normalized or normalized.lstrip().startswith("select"):
            return SimpleNamespace(
                one=lambda: self._rows[0],
                one_or_none=lambda: self._rows[0] if self._rows else None,
                all=lambda: self._rows,
            )
        return SimpleNamespace(rowcount=1)


class _Transaction:
    def __init__(self, calls: list, rows: list[FakeRow] | None = None) -> None:
        self._connection = FakeConnection(calls, rows)

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, *args) -> bool:
        return False


class FakeEngine:
    def __init__(self, calls: list, rows: list[FakeRow] | None = None) -> None:
        self._calls = calls
        self._rows = rows or []
        self.transactions = 0

    def begin(self) -> _Transaction:
        self.transactions += 1
        return _Transaction(self._calls, self._rows)


def _patch_engine(monkeypatch) -> tuple[EvidenceRepository, list, FakeEngine]:
    calls: list = []
    engine = FakeEngine(calls)
    # engine 是只读 property（未连接时 getattr 会抛 RuntimeError），改 patch 私有字段
    monkeypatch.setattr(evidence_module.postgres_manager, "_engine", engine)
    return EvidenceRepository(), calls, engine


def test_append_evidence_card_is_transactional_and_insert_only(monkeypatch) -> None:
    repository, calls, engine = _patch_engine(monkeypatch)

    repository.append_evidence_card("session-1", _card(), directive={"id": "dir-1"})

    assert engine.transactions == 1  # 卡片与 claims 同一事务
    card_sql, card_params = calls[0]
    assert card_sql.startswith("insert into aiops_evidence_cards")
    assert "on conflict (card_id) do nothing" in card_sql
    assert json.loads(card_params["directive"]) == {"id": "dir-1"}

    claim_sqls = [sql for sql, _ in calls[1:]]
    assert len(claim_sqls) == 2
    for sql in claim_sqls:
        assert sql.startswith("insert into aiops_evidence_claims")
        assert "on conflict (claim_id) do nothing" in sql
        assert "update" not in sql and "delete" not in sql
    first_claim = calls[1][1]
    assert first_claim["hypothesis_ids"] == ["hyp-oom"]
    assert json.loads(first_claim["provenance"])["tool_name"] == "query_memory_metrics"


def test_finish_session_only_finalizes_once(monkeypatch) -> None:
    repository, calls, _ = _patch_engine(monkeypatch)
    repository.finish_session("session-1", "completed", final_hypothesis_id="hyp-2")

    sql, params = calls[0]
    assert "finished_at is null" in sql
    assert params["status"] == "completed"
    assert params["final_hypothesis_id"] == "hyp-2"


def test_start_session_returns_row_mapping(monkeypatch) -> None:
    calls: list = []
    row = FakeRow({"session_id": "session-1", "status": "running"})
    engine = FakeEngine(calls, rows=[row])
    monkeypatch.setattr(evidence_module.postgres_manager, "_engine", engine)

    result = EvidenceRepository().start_session("session-1", "data-sync-service", scenario_id="gc-pressure")

    assert result["session_id"] == "session-1"
    sql, params = calls[0]
    assert sql.startswith("insert into aiops_diagnosis_sessions")
    assert params["scenario_id"] == "gc-pressure"
