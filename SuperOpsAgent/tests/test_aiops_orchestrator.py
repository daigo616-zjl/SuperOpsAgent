"""星型编排器测试：拓扑铁律、确定性路由、扇出、淘汰与白名单剥离。"""

import importlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.agent.aiops.diagnosis_models import (
    AdjudicationDecision,
    BudgetLedger,
    Directive,
    Elimination,
    EvidenceCard,
    EvidenceClaim,
    ClaimProvenance,
    Hypothesis,
    SupervisorDecision,
)
from app.agent.aiops.diagnosis_models import DiagnosisContext
from app.agent.aiops.orchestrator import supervisor as supervisor_module
from app.agent.aiops.orchestrator.graph import build_orchestrator_graph
from app.agent.aiops.orchestrator.reporter import strip_unresolved_claims
from app.agent.aiops.orchestrator.state import OrchestratorState

hypothesizer_module = importlib.import_module("app.agent.aiops.orchestrator.hypothesizer")
adjudicator_module = importlib.import_module("app.agent.aiops.orchestrator.adjudicator")
reporter_module = importlib.import_module("app.agent.aiops.orchestrator.reporter")
graph_module = importlib.import_module("app.agent.aiops.orchestrator.graph")

SPOKE_NODES = {"hypothesizer", "investigate", "adjudicator", "reporter"}


def make_hypothesis(id: str = "hyp-gc", statement: str = "GC 压力高") -> Hypothesis:
    return Hypothesis(
        id=id,
        statement=statement,
        expected_support=["GC pause 上升"],
        expected_refuting=["内存平稳"],
    )


def make_card(directive_id: str, domain: str) -> EvidenceCard:
    return EvidenceCard(
        card_id=f"card-{directive_id}",
        domain=domain,
        directive_id=directive_id,
        round=1,
        claims=[
            EvidenceClaim(
                claim_id=f"ev-{directive_id}-1",
                statement="GC pause 4200 ms 超阈值",
                confidence=0.9,
                polarity="supports",
                hypothesis_ids=["hyp-gc"],
                provenance=ClaimProvenance(
                    tool_name="search_log",
                    args_digest="sha256:abc",
                    excerpt="GC pause 4200 ms",
                ),
            )
        ],
        summary="日志显示持续 GC 停顿",
    )


def make_state(**overrides: Any) -> OrchestratorState:
    state: OrchestratorState = {
        "input": "诊断告警",
        "session_id": "s1",
        "context": DiagnosisContext(service_name="data-sync-service"),
        "hypotheses": [],
        "directives": [],
        "dispatched": [],
        "evidence": [],
        "investigation_errors": [],
        "adjudications": [],
        "pending_decision": None,
        "decision": SupervisorDecision(action="hypothesize", reason="初始状态"),
        "adjudicated_evidence_count": 0,
        "converged_hypothesis_id": None,
        "report_violations": [],
        "response": "",
        "budget": BudgetLedger(),
    }
    state.update(overrides)
    return state


def test_star_topology_no_spoke_to_spoke_edges() -> None:
    graph = build_orchestrator_graph()
    edges = graph.get_graph().edges
    for edge in edges:
        source = edge.source
        target = edge.target
        if source in SPOKE_NODES:
            assert target in {"supervisor", "__end__"}, f"星型违规边: {source} -> {target}"
        if target in SPOKE_NODES:
            assert source == "supervisor", f"星型违规边: {source} -> {target}"


def test_supervisor_routes_hypothesize_when_no_hypotheses() -> None:
    update = supervisor_module.supervisor(make_state())
    assert update["decision"].action == "hypothesize"
    assert update["budget"].round == 1


def test_supervisor_dispatches_default_directives_for_all_domains() -> None:
    state = make_state(hypotheses=[make_hypothesis(), make_hypothesis("hyp-oom", "OOM")])
    update = supervisor_module.supervisor(state)
    decision = update["decision"]
    assert decision.action == "dispatch"
    assert {d.target_domain for d in decision.directives} == {"metrics", "logs", "knowledge"}
    assert update["budget"].invocations > 0


def test_supervisor_converges_when_wall_time_insufficient_for_dispatch() -> None:
    started = datetime.now() - timedelta(seconds=20)
    budget = BudgetLedger(
        max_wall_seconds=100.0, min_dispatch_wall_seconds=90.0, started_at=started
    )
    state = make_state(hypotheses=[make_hypothesis()], budget=budget)
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "converge"
    assert "剩余墙钟" in update["decision"].reason


def test_supervisor_dispatches_adjudicator_new_directives() -> None:
    directive = Directive(
        id="drill-gc", target_domain="logs", objective="查 GC 日志", hypothesis_ids=["hyp-gc"]
    )
    state = make_state(
        hypotheses=[make_hypothesis()],
        directives=[],
        pending_decision=AdjudicationDecision(
            new_directives=[directive],
        ),
    )
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "dispatch"
    assert update["decision"].directives[0].id == "drill-gc"
    assert update["directives"][0].id == "drill-gc"


def test_supervisor_applies_eliminations() -> None:
    hypothesis = make_hypothesis()
    other = make_hypothesis("hyp-oom", "OOM")
    state = make_state(
        hypotheses=[hypothesis, other],
        evidence=[make_card("r1-logs", "logs")],
        pending_decision=AdjudicationDecision(
            eliminations=[
                Elimination(
                    hypothesis_id="hyp-oom",
                    ruled_out_by=["ev-r1-logs-1"],
                    reason="内存平稳",
                )
            ],
        ),
    )
    update = supervisor_module.supervisor(state)
    by_id = {h.id: h for h in update["hypotheses"]}
    assert by_id["hyp-oom"].status == "ruled_out"
    assert by_id["hyp-oom"].ruled_out_by == ["ev-r1-logs-1"]
    assert by_id["hyp-gc"].status == "active"


def test_supervisor_adjudicates_on_new_evidence() -> None:
    state = make_state(
        hypotheses=[make_hypothesis()],
        directives=[Directive(id="r1-logs", target_domain="logs", objective="查日志")],
        dispatched=["r1-logs"],
        evidence=[make_card("r1-logs", "logs")],
        adjudicated_evidence_count=0,
    )
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "adjudicate"
    assert update["adjudicated_evidence_count"] == 1


def test_supervisor_converges_when_nothing_left() -> None:
    state = make_state(
        hypotheses=[make_hypothesis()],
        directives=[Directive(id="r1-logs", target_domain="logs", objective="查日志")],
        dispatched=["r1-logs"],
        evidence=[make_card("r1-logs", "logs")],
        adjudicated_evidence_count=1,
    )
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "converge"


def test_supervisor_converges_on_round_budget_exhaustion() -> None:
    budget = BudgetLedger(max_rounds=1)
    state = make_state(
        hypotheses=[make_hypothesis()],
        budget=budget,
    )
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "converge"
    assert "预算" in update["decision"].reason


def test_supervisor_converges_when_hypothesizer_failed() -> None:
    state = make_state(investigation_errors=["hypothesizer:boom"])
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "converge"
    assert "假设生成失败" in update["decision"].reason


@pytest.mark.asyncio
async def test_investigate_times_out_on_task_wall_limit(monkeypatch) -> None:
    """单域卡住时独立任务上限生效，不再拖到全局墙钟尽头。"""
    import asyncio as _asyncio

    from app.agent.aiops.orchestrator.graph import investigate

    async def _hang(*_args, **_kwargs):
        await _asyncio.sleep(5)

    monkeypatch.setattr(graph_module, "run_domain_investigation", _hang)
    directive = Directive(
        id="d1-metrics",
        target_domain="metrics",
        objective="验证 GC 压力假设",
        hypothesis_ids=["hyp-gc"],
    )
    budget = BudgetLedger(investigation_wall_seconds=0.5)
    task = {
        "directive": directive.model_dump(mode="json"),
        "hypotheses": [],
        "round": 2,
        "session_id": "s1",
        "context": DiagnosisContext(service_name="data-sync-service"),
        "budget": budget.model_dump(mode="json"),
    }
    result = await investigate(task)
    assert result["investigation_errors"][0].startswith("d1-metrics: 取证超时")
    assert "任务上限" in result["investigation_errors"][0]


def test_supervisor_converges_when_all_hypotheses_ruled_out() -> None:
    ruled_out = make_hypothesis().model_copy(update={"status": "ruled_out"})
    state = make_state(hypotheses=[ruled_out])
    update = supervisor_module.supervisor(state)
    assert update["decision"].action == "converge"


def test_strip_unresolved_claims() -> None:
    report = "根因 [ev-a-1] 与 [ev-bogus-9]，另见 [ev-a-1] 和 [ev-c-2]"
    cleaned, violations = strip_unresolved_claims(report, {"ev-a-1", "ev-c-2"})
    assert cleaned == "根因 [ev-a-1] 与 ，另见 [ev-a-1] 和 [ev-c-2]"
    assert violations == ["ev-bogus-9"]
