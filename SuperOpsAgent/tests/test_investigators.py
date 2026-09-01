"""Investigator 域测试：域切分注册表、ReAct 子图与机械 provenance。"""

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.aiops import investigators
from app.agent.aiops.diagnosis_models import Directive
from app.agent.aiops.investigators import base as base_module
from app.agent.aiops.investigators.base import (
    EvidenceDraft,
    ToolCallRecord,
    extract_tool_call_records,
)
from app.agent.aiops.diagnosis_models import DiagnosisContext
from app.agent.aiops.tool_registry import ToolDescriptor, ToolRegistry
from app.agent.aiops.tool_runtime import args_digest
from app.config import config

tool_registry_module = importlib.import_module("app.agent.aiops.tool_registry")


def make_tool(name: str, description: str = "测试工具"):
    return SimpleNamespace(name=name, description=description)


ALL_TOOLS = {
    "query_cpu_metrics": make_tool("query_cpu_metrics", "查询 CPU"),
    "query_memory_metrics": make_tool("query_memory_metrics", "查询内存"),
    "query_active_alerts": make_tool("query_active_alerts", "查询告警"),
    "search_topic_by_service_name": make_tool("search_topic_by_service_name", "查主题"),
    "get_topic_info_by_name": make_tool("get_topic_info_by_name", "主题详情"),
    "get_current_timestamp": make_tool("get_current_timestamp", "当前时间戳"),
    "search_log": make_tool("search_log", "搜索日志"),
    "retrieve_knowledge": make_tool("retrieve_knowledge", "检索知识"),
}


def make_full_registry() -> ToolRegistry:
    descriptors = {
        name: ToolDescriptor(name=name, description=tool.description, source="mcp")
        for name, tool in ALL_TOOLS.items()
    }
    return ToolRegistry(descriptors=descriptors, handlers=dict(ALL_TOOLS))


def make_directive(domain: str = "metrics", hypothesis_ids: list[str] | None = None):
    return Directive(
        id=f"d1-{domain}",
        target_domain=domain,
        objective="验证 GC 压力假设",
        hypothesis_ids=hypothesis_ids or ["hyp-gc"],
        max_iterations=3,
    )


class FakeStructuredChain:
    def __init__(self, drafts: list[Any]) -> None:
        self.drafts = list(drafts)
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        return self.drafts.pop(0)


class FakeLlm:
    def __init__(self, chain: FakeStructuredChain) -> None:
        self._chain = chain

    def with_structured_output(self, _schema):
        return self._chain


class FakeAgent:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.configs: list[dict[str, Any] | None] = []

    async def ainvoke(self, _input, config=None):
        self.configs.append(config)
        return {"messages": self.messages}


def patch_run_environment(
    monkeypatch,
    *,
    agent: FakeAgent,
    chain: FakeStructuredChain,
) -> None:
    registry = ToolRegistry(
        descriptors={
            name: ToolDescriptor(name=name, description="测试工具", source="mcp")
            for name in ("query_cpu_metrics", "retrieve_knowledge")
        },
        handlers={
            name: make_tool(name) for name in ("query_cpu_metrics", "retrieve_knowledge")
        },
    )
    monkeypatch.setattr(base_module, "get_domain_registry", _async_returning(registry))
    monkeypatch.setattr(
        base_module,
        "LLMFactory",
        SimpleNamespace(
            create_qwen_chat_model=lambda model, temperature, **_kwargs: FakeLlm(chain)
        ),
    )
    monkeypatch.setattr(base_module, "create_react_agent", lambda llm, tools, prompt: agent)


def _async_returning(value):
    async def _factory(*_args) -> ToolRegistry:
        return value

    return _factory


def make_draft(call_index: int = 0, hypothesis_ids: list[str] | None = None) -> EvidenceDraft:
    return EvidenceDraft(
        summary="CPU 尖峰与告警吻合",
        claims=[
            {
                "call_index": call_index,
                "statement": "CPU 使用率达到 91.2",
                "confidence": 0.9,
                "polarity": "supports",
                "hypothesis_ids": hypothesis_ids if hypothesis_ids is not None else ["hyp-gc"],
                "excerpt": "value=91.2",
            }
        ],
    )


def make_agent_messages() -> list[Any]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-1", "name": "query_cpu_metrics", "args": {"service_name": "svc"}}
            ],
        ),
        ToolMessage(content='{"data":[{"value":91.2}]}', tool_call_id="call-1"),
    ]


def test_get_investigator_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="未知取证域"):
        investigators.get_investigator("traces")


@pytest.mark.parametrize("domain", ["metrics", "logs", "knowledge"])
def test_get_investigator_returns_callable(domain: str) -> None:
    assert callable(investigators.get_investigator(domain))


def test_domain_registry_isolates_tools(monkeypatch) -> None:
    monkeypatch.setattr(tool_registry_module, "get_tool_registry", _async_returning(make_full_registry()))

    import asyncio

    logs_registry = asyncio.run(tool_registry_module.get_domain_registry("logs"))
    metrics_registry = asyncio.run(tool_registry_module.get_domain_registry("metrics"))
    knowledge_registry = asyncio.run(tool_registry_module.get_domain_registry("knowledge"))

    assert "query_cpu_metrics" not in logs_registry.handlers
    assert set(logs_registry.handlers) == set(config.aiops_domain_tools["logs"])
    assert "search_log" not in metrics_registry.handlers
    assert set(metrics_registry.handlers) == set(config.aiops_domain_tools["metrics"])
    assert set(knowledge_registry.handlers) == {"retrieve_knowledge"}


def test_domain_registry_missing_configured_tool_raises(monkeypatch) -> None:
    monkeypatch.setattr(tool_registry_module, "get_tool_registry", _async_returning(make_full_registry()))
    monkeypatch.setattr(
        config, "aiops_domain_tools", {**config.aiops_domain_tools, "metrics": ["nonexistent_tool"]}
    )

    import asyncio

    with pytest.raises(LookupError, match="取证域 metrics 缺少已配置工具"):
        asyncio.run(tool_registry_module.get_domain_registry("metrics"))


def test_extract_tool_call_records_pairs_by_call_id() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-a", "name": "search_log", "args": {"topic_id": "topic-001"}},
                {"id": "call-b", "name": "get_current_timestamp", "args": {}},
            ],
        ),
        ToolMessage(content="1789", tool_call_id="call-b"),
        ToolMessage(content="log lines", tool_call_id="call-a"),
        ToolMessage(content="orphan", tool_call_id="call-x"),
    ]

    records = extract_tool_call_records(messages)

    assert [record.tool_name for record in records] == [
        "get_current_timestamp",
        "search_log",
    ]
    assert [record.call_index for record in records] == [0, 1]
    assert records[1].arguments == {"topic_id": "topic-001"}


def test_args_digest_is_deterministic() -> None:
    first = args_digest("search_log", {"topic_id": "t"}, "content")
    second = args_digest("search_log", {"topic_id": "t"}, "content")
    third = args_digest("search_log", {"topic_id": "other"}, "content")

    assert first == second
    assert first != third
    assert first.startswith("sha256:")


@pytest.mark.asyncio
async def test_run_investigation_builds_card_with_mechanical_provenance(monkeypatch) -> None:
    agent = FakeAgent(make_agent_messages())
    chain = FakeStructuredChain([make_draft()])
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    card = await base_module.run_investigation(
        make_directive("metrics"),
        DiagnosisContext(service_name="data-sync-service"),
        domain="metrics",
        system_prompt="你是指标取证 Agent",
        hypotheses=[{"id": "hyp-gc", "statement": "GC 压力高", "expected_support": []}],
        round_number=1,
    )

    assert card.card_id == "card-d1-metrics-r1"
    assert card.domain == "metrics"
    assert card.directive_id == "d1-metrics"
    assert card.round == 1
    assert len(card.claims) == 1
    claim = card.claims[0]
    assert claim.claim_id == "ev-d1-metrics-1"
    assert claim.provenance.tool_name == "query_cpu_metrics"
    assert claim.provenance.args_digest == args_digest(
        "query_cpu_metrics", {"service_name": "svc"}, '{"data":[{"value":91.2}]}'
    )
    assert claim.provenance.output_path is None
    assert claim.provenance.excerpt == "value=91.2"
    assert agent.configs == [{"recursion_limit": make_directive().max_iterations * 2 + 6}]


@pytest.mark.asyncio
async def test_run_investigation_retries_after_draft_validation_error(monkeypatch) -> None:
    agent = FakeAgent(make_agent_messages())
    chain = FakeStructuredChain(
        [make_draft(call_index=5), make_draft(call_index=0, hypothesis_ids=["hyp-other", "hyp-gc"])]
    )
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    card = await base_module.run_investigation(
        make_directive("metrics"),
        DiagnosisContext(service_name="data-sync-service"),
        domain="metrics",
        system_prompt="prompt",
    )

    assert len(chain.calls) == 2
    correction = chain.calls[1][-1][1]
    assert "call_index 只能取 0 到 0" in correction
    # 未知假设 ID 被清洗剔除，已知 ID 保留，证据卡不再作废
    assert card.claims[0].hypothesis_ids == ["hyp-gc"]


@pytest.mark.asyncio
async def test_run_investigation_strips_invented_hypothesis_ids(monkeypatch) -> None:
    """模型编造候选列表之外的假设 ID 时剔除 ID 保留证据卡。"""
    agent = FakeAgent(make_agent_messages())
    chain = FakeStructuredChain([make_draft(call_index=0, hypothesis_ids=["hyp_memory_pressure"])])
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    card = await base_module.run_investigation(
        make_directive("metrics", hypothesis_ids=[]),
        DiagnosisContext(service_name="data-sync-service"),
        domain="metrics",
        system_prompt="prompt",
    )

    assert card.claims[0].hypothesis_ids == []


@pytest.mark.asyncio
async def test_run_investigation_accepts_corrected_second_draft(monkeypatch) -> None:
    agent = FakeAgent(make_agent_messages())
    chain = FakeStructuredChain([make_draft(call_index=9), make_draft(call_index=0)])
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    card = await base_module.run_investigation(
        make_directive("metrics"),
        DiagnosisContext(service_name="data-sync-service"),
        domain="metrics",
        system_prompt="prompt",
    )

    assert card.claims[0].claim_id == "ev-d1-metrics-1"
    assert len(chain.calls) == 2


@pytest.mark.asyncio
async def test_run_investigation_accepts_global_hypothesis_reference(monkeypatch) -> None:
    """模型引用 directive 子集之外、但存在于全局候选假设中的 ID 不应作废整卡。"""
    agent = FakeAgent(make_agent_messages())
    chain = FakeStructuredChain([make_draft(call_index=0, hypothesis_ids=["hyp-oom"])])
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    card = await base_module.run_investigation(
        make_directive("metrics", hypothesis_ids=["hyp-gc"]),
        DiagnosisContext(service_name="data-sync-service"),
        domain="metrics",
        system_prompt="prompt",
        hypotheses=[{"id": "hyp-oom", "statement": "内存耗尽"}],
    )

    assert card.claims[0].hypothesis_ids == ["hyp-oom"]


@pytest.mark.asyncio
async def test_run_investigation_raises_without_tool_calls(monkeypatch) -> None:
    agent = FakeAgent([AIMessage(content="无法收集")])
    chain = FakeStructuredChain([])
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    with pytest.raises(RuntimeError, match="未产生任何工具调用"):
        await base_module.run_investigation(
            make_directive("logs"),
            DiagnosisContext(service_name="data-sync-service"),
            domain="logs",
            system_prompt="prompt",
        )


@pytest.mark.asyncio
async def test_run_investigation_truncates_long_excerpt(monkeypatch) -> None:
    agent = FakeAgent(make_agent_messages())
    long_excerpt = "x" * 5000
    draft = EvidenceDraft(
        summary="超长摘录",
        claims=[
            {
                "call_index": 0,
                "statement": "原始输出",
                "confidence": 0.5,
                "polarity": "neutral",
                "hypothesis_ids": [],
                "excerpt": long_excerpt,
            }
        ],
    )
    chain = FakeStructuredChain([draft])
    patch_run_environment(monkeypatch, agent=agent, chain=chain)

    card = await base_module.run_investigation(
        make_directive("logs", hypothesis_ids=[]),
        DiagnosisContext(service_name="data-sync-service"),
        domain="logs",
        system_prompt="prompt",
    )

    assert len(card.claims[0].provenance.excerpt) == 2000


def test_tool_call_record_content_list_normalization() -> None:
    record = ToolCallRecord(call_index=0, tool_name="t", arguments={}, content="kept")
    assert record.content == "kept"
