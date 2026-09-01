"""多 Agent 诊断端到端（Fake 模型）与 SSE 契约测试。

覆盖：全流程（假设→扇出取证→评审收敛→报告）、事件契约
（type ⊆ 旧契约且以 complete 收尾）、report_chunk 与 report 去重。
"""

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.agent.aiops.diagnosis_models import (
    AdjudicationDecision,
    Elimination,
    EvidenceCard,
    EvidenceClaim,
    ClaimProvenance,
)
from app.agent.aiops.orchestrator.hypothesizer import HypothesisSet
from app.services.aiops_service import AIOpsService

hypothesizer_module = importlib.import_module("app.agent.aiops.orchestrator.hypothesizer")
adjudicator_module = importlib.import_module("app.agent.aiops.orchestrator.adjudicator")
reporter_module = importlib.import_module("app.agent.aiops.orchestrator.reporter")
graph_module = importlib.import_module("app.agent.aiops.orchestrator.graph")
aiops_service_module = importlib.import_module("app.services.aiops_service")

SSE_CONTRACT_TYPES = {"status", "plan", "step_complete", "report", "report_chunk", "complete", "error"}


class FakeStructuredChain:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.calls: list[Any] = []

    async def ainvoke(self, inputs: Any) -> Any:
        self.calls.append(inputs)
        return self.outputs.pop(0)

    def as_runnable(self) -> RunnableLambda:
        chain = self

        async def _fn(inputs: Any) -> Any:
            return await chain.ainvoke(inputs)

        return RunnableLambda(_fn)


class FakeLlm:
    """Fake role model: scripted structured outputs; report path uses a streamable FakeChatModel."""

    def __init__(self, structured: list[Any] | None = None, report: str | None = None):
        self.chain = FakeStructuredChain(structured or [])
        self.chat = (
            GenericFakeChatModel(messages=iter([AIMessage(content=report)]))
            if report is not None
            else None
        )

    def with_structured_output(self, _schema):
        return self.chain.as_runnable()


def make_card(directive_id: str, domain: str, round_number: int = 2) -> EvidenceCard:
    return EvidenceCard(
        card_id=f"card-{directive_id}",
        domain=domain,
        directive_id=directive_id,
        round=round_number,
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
        summary=f"{domain} 域证据",
    )


def patch_multiagent_world(monkeypatch, *, reporter_report: str | None = None) -> dict[str, Any]:
    """替换全部 LLM 与外部依赖：假设→取证→评审→报告全 Fake。"""
    recorded: dict[str, Any] = {"sessions": [], "cards": []}

    hypothesizer_llm = FakeLlm(
        structured=[
            HypothesisSet(
                hypotheses=[
                    {
                        "id": "hyp-gc",
                        "statement": "JVM GC 压力过高",
                        "prior": 0.6,
                        "expected_support": ["GC pause 超阈值"],
                        "expected_refuting": ["GC 平稳"],
                    },
                    {
                        "id": "hyp-oom",
                        "statement": "内存泄漏导致 OOM",
                        "prior": 0.3,
                        "expected_support": ["内存锯齿"],
                        "expected_refuting": ["内存平稳"],
                    },
                ]
            )
        ]
    )
    adjudicator_llm = FakeLlm(
        structured=[
            AdjudicationDecision(
                eliminations=[
                    Elimination(
                        hypothesis_id="hyp-oom",
                        ruled_out_by=["ev-r2-logs-1"],
                        reason="日志无 OOM 痕迹，内存平稳",
                    )
                ],
                converged=True,
                converged_hypothesis_id="hyp-gc",
            )
        ]
    )
    reporter_llm = FakeLlm(
        report=reporter_report
        or "## 结论\nGC 压力过高 [ev-r2-logs-1] 与 [ev-bogus-9]\n\n## 处置建议\n扩容堆内存"
    )

    async def fake_investigation(domain, directive, context, *, hypotheses=None, round_number=0):
        return make_card(directive.id, domain, round_number)

    class FakeRepository:
        def start_session(self, session_id, service_name, scenario_id, budget_snapshot):
            recorded["sessions"].append(("start", session_id, service_name, scenario_id))

        def finish_session(self, session_id, status, final_hypothesis_id=None, budget_snapshot=None):
            recorded["sessions"].append(("finish", session_id, status))

        def append_evidence_card(self, session_id, card, directive=None):
            recorded["cards"].append((session_id, card.card_id))

    monkeypatch.setattr(
        hypothesizer_module,
        "LLMFactory",
        SimpleNamespace(create_qwen_chat_model=lambda **kwargs: hypothesizer_llm),
    )
    monkeypatch.setattr(hypothesizer_module, "_fetch_alerts", async_returns("[]"))
    monkeypatch.setattr(hypothesizer_module, "_fetch_experience", async_returns("无"))
    monkeypatch.setattr(
        adjudicator_module,
        "LLMFactory",
        SimpleNamespace(create_qwen_chat_model=lambda **kwargs: adjudicator_llm),
    )
    monkeypatch.setattr(
        reporter_module,
        "LLMFactory",
        SimpleNamespace(create_qwen_chat_model=lambda **kwargs: reporter_llm.chat),
    )
    monkeypatch.setattr(graph_module, "run_domain_investigation", fake_investigation)
    fake_repo = FakeRepository()
    monkeypatch.setattr(aiops_service_module, "evidence_repository", fake_repo)
    monkeypatch.setattr(graph_module, "evidence_repository", fake_repo)
    recorded["hypothesizer_chain"] = hypothesizer_llm.chain
    recorded["adjudicator_chain"] = adjudicator_llm.chain
    return recorded


def async_returns(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


async def collect_events(service: AIOpsService, **kwargs) -> list[dict[str, Any]]:
    return [event async for event in service.execute(**kwargs)]


@pytest.mark.asyncio
async def test_multiagent_full_flow_contract(monkeypatch) -> None:
    recorded = patch_multiagent_world(monkeypatch)
    service = AIOpsService()

    events = await collect_events(
        service, user_input="诊断 data-sync-service 告警", session_id="s-flow"
    )

    types = [event["type"] for event in events]
    assert set(types) <= SSE_CONTRACT_TYPES
    assert types[-1] == "complete"

    plan_events = [event for event in events if event["type"] == "plan"]
    assert len(plan_events) == 1
    assert len(plan_events[0]["plan"]["hypotheses"]) == 2

    step_events = [event for event in events if event["type"] == "step_complete"]
    assert {event["result"]["domain"] for event in step_events} == {"metrics", "logs", "knowledge"}

    # reporter 走流式 chunk：完整 report 事件被去重，最终文本由 complete 携带
    assert [event for event in events if event["type"] == "report"] == []
    assert len([event for event in events if event["type"] == "report_chunk"]) >= 1

    complete = events[-1]
    assert complete["response"].startswith("## 结论")
    # 白名单内的引用保留，未定义引用被剥离
    assert "[ev-r2-logs-1]" in complete["response"]
    assert "ev-bogus-9" not in complete["response"]
    assert ("start", "s-flow", "data-sync-service", "no-fault") in recorded["sessions"]
    assert ("finish", "s-flow", "completed") in recorded["sessions"]
    assert len(recorded["cards"]) == 3


@pytest.mark.asyncio
async def test_report_event_without_stream_writer(monkeypatch) -> None:
    """非流式环境（无 stream writer）时降级为单个 report 事件。"""
    def _raise():
        raise RuntimeError("not in stream context")

    patch_multiagent_world(monkeypatch)
    monkeypatch.setattr(reporter_module, "get_stream_writer", _raise)
    service = AIOpsService()

    events = await collect_events(service, user_input="诊断告警", session_id="s-nostream")

    assert [event for event in events if event["type"] == "report_chunk"] == []
    report_events = [event for event in events if event["type"] == "report"]
    assert len(report_events) == 1
    assert "ev-bogus-9" not in report_events[0]["report"]


@pytest.mark.asyncio
async def test_multiagent_budget_exhaustion_still_completes(monkeypatch) -> None:
    recorded = patch_multiagent_world(monkeypatch)
    monkeypatch.setattr(aiops_service_module.config, "aiops_max_rounds", 1)
    service = AIOpsService()

    events = await collect_events(service, user_input="诊断告警", session_id="s-budget")

    types = [event["type"] for event in events]
    assert types[-1] == "complete"
    # 只跑了假设一轮就被预算收敛，不应有任何取证完成事件
    assert not [event for event in events if event["type"] == "step_complete"]


@pytest.mark.asyncio
async def test_diagnose_wrapper_maps_complete(monkeypatch) -> None:
    patch_multiagent_world(monkeypatch)
    service = AIOpsService()

    events = [
        event
        async for event in service.diagnose(session_id="s-wrap", service_name="data-sync-service")
    ]

    assert events[-1]["type"] == "complete"
    assert events[-1]["stage"] == "diagnosis_complete"
    assert events[-1]["diagnosis"]["status"] == "completed"
    assert events[-1]["diagnosis"]["report"].startswith("## 结论")


def test_hypothesizer_invalid_output_degrades() -> None:
    llm = FakeLlm(structured=[{"hypotheses": []}])
    result = asyncio.run(_run_hypothesizer(llm))
    assert result["hypotheses"] == []
    assert result["investigation_errors"]
    assert len(llm.chain.calls) == 1


async def _run_hypothesizer(llm: FakeLlm) -> dict[str, Any]:
    from app.agent.aiops.diagnosis_models import DiagnosisContext

    state = {"input": "诊断", "context": DiagnosisContext(service_name="svc")}
    original = hypothesizer_module.LLMFactory
    original_alerts, original_experience = (
        hypothesizer_module._fetch_alerts,
        hypothesizer_module._fetch_experience,
    )
    hypothesizer_module.LLMFactory = SimpleNamespace(
        create_qwen_chat_model=lambda **kwargs: llm
    )
    hypothesizer_module._fetch_alerts = async_returns("[]")
    hypothesizer_module._fetch_experience = async_returns("")
    try:
        return await hypothesizer_module.hypothesizer(state)
    finally:
        hypothesizer_module.LLMFactory = original
        hypothesizer_module._fetch_alerts, hypothesizer_module._fetch_experience = (
            original_alerts,
            original_experience,
        )
