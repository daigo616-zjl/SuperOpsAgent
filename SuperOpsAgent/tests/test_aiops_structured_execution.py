import importlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from app.agent.aiops.executor import _get_path, _resolve_value, executor
from app.agent.aiops.models import (
    DiagnosisContext,
    DiagnosticPlan,
    DiagnosticStep,
    StepExecutionResult,
    SuccessCriterion,
    ToolCallSpec,
)
from app.agent.aiops.planner import planner
from app.agent.aiops.replanner import _apply_replan, _chunk_text, _generate_response
from app.agent.aiops.tool_registry import ToolDescriptor, ToolRegistry
from app.services.aiops_service import AIOpsService

executor_module = importlib.import_module("app.agent.aiops.executor")
replanner_module = importlib.import_module("app.agent.aiops.replanner")
planner_module = importlib.import_module("app.agent.aiops.planner")
aiops_service_module = importlib.import_module("app.services.aiops_service")


class QueryArgs(BaseModel):
    service_name: str


class FakeTool:
    name = "query_cpu_metrics"
    description = "查询 CPU"
    args_schema = QueryArgs

    def __init__(self) -> None:
        self.arguments: dict[str, Any] | None = None

    async def ainvoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.arguments = arguments
        return {"data": [{"value": 91.2}]}


class FakeMcpTool:
    name = "get_current_timestamp"
    description = "获取当前时间戳"
    args_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def ainvoke(self, arguments: dict[str, Any]) -> int:
        assert arguments == {}
        return 1_787_972_400_000


def make_step(tool_name: str = "query_cpu_metrics") -> DiagnosticStep:
    return DiagnosticStep(
        id="query_cpu",
        title="查询 CPU 指标",
        purpose="获取目标服务 CPU 数据",
        tool_call=ToolCallSpec(
            tool_name=tool_name,
            arguments={"service_name": {"source": "context", "path": "service_name"}},
        ),
        success_criteria=[SuccessCriterion(path="data", operator="not_empty")],
    )


def make_registry(tool: FakeTool) -> ToolRegistry:
    return ToolRegistry(
        descriptors={
            tool.name: ToolDescriptor(
                name=tool.name,
                description=tool.description,
                source="mcp",
                input_schema=QueryArgs.model_json_schema(),
            )
        },
        handlers={tool.name: tool},
    )


def test_plan_rejects_forward_dependency() -> None:
    with pytest.raises(ValueError, match="依赖不存在"):
        DiagnosticPlan(
            goal="诊断 CPU",
            steps=[
                DiagnosticStep(
                    id="first",
                    title="第一步",
                    purpose="测试依赖校验",
                    tool_call=ToolCallSpec(tool_name="query_cpu_metrics"),
                    depends_on=["later"],
                ),
                DiagnosticStep(
                    id="later",
                    title="后续步骤",
                    purpose="测试依赖校验",
                    tool_call=ToolCallSpec(tool_name="query_cpu_metrics"),
                ),
            ],
        )


@pytest.mark.asyncio
async def test_planner_retries_once_after_structured_validation_error(monkeypatch) -> None:
    valid_plan = DiagnosticPlan(goal="诊断 CPU", steps=[make_step()])
    invocations: list[dict[str, Any]] = []

    class FakeKnowledgeTool:
        async def ainvoke(self, _arguments):
            return ""

    class FakeRegistry:
        def prompt_description(self):
            return "query_cpu_metrics"

        def validate_plan(self, plan):
            assert plan is valid_plan

    class FakeChain:
        async def ainvoke(self, chain_input):
            invocations.append(chain_input)
            if len(invocations) == 1:
                raise ValueError("步骤 query_logs 引用了 get_time，但未声明对应依赖")
            return valid_plan

    class FakePrompt:
        def __or__(self, _model):
            return FakeChain()

    class FakeLlm:
        def with_structured_output(self, _schema):
            return object()

    async def fake_registry():
        return FakeRegistry()

    monkeypatch.setattr(planner_module, "retrieve_knowledge", FakeKnowledgeTool())
    monkeypatch.setattr(planner_module, "get_tool_registry", fake_registry)
    monkeypatch.setattr(planner_module, "planner_prompt", FakePrompt())
    monkeypatch.setattr(
        planner_module.LLMFactory,
        "create_qwen_chat_model",
        lambda **_kwargs: FakeLlm(),
    )

    update = await planner(
        {
            "input": "诊断 CPU",
            "context": DiagnosisContext(service_name="orders-service"),
        }
    )

    assert update == {"plan": valid_plan}
    assert len(invocations) == 2
    assert len(invocations[1]["messages"]) == 2
    assert "未声明对应依赖" in invocations[1]["messages"][1][1]


def test_step_reference_supports_numeric_offset() -> None:
    now = datetime.now(UTC)
    prior_result = StepExecutionResult(
        step_id="current_time",
        step_title="获取当前时间",
        tool_name="get_current_timestamp",
        status="succeeded",
        output=2_000_000,
        started_at=now,
        finished_at=now,
        duration_ms=0,
    )

    resolved = _resolve_value(
        {
            "end_time": {"source": "step", "step_id": "current_time"},
            "start_time": {
                "source": "step",
                "step_id": "current_time",
                "offset": -900_000,
            },
        },
        {"service_name": "orders-service"},
        {"current_time": prior_result},
    )

    assert resolved == {"end_time": 2_000_000, "start_time": 1_100_000}


def test_get_path_supports_dot_and_bracket_list_indexes() -> None:
    output = {"topics": [{"topic_id": "topic-001"}]}

    assert _get_path(output, "topics.0.topic_id") == "topic-001"
    assert _get_path(output, "topics[0].topic_id") == "topic-001"


@pytest.mark.asyncio
async def test_executor_resolves_context_and_records_structured_result(monkeypatch) -> None:
    tool = FakeTool()
    registry = make_registry(tool)

    async def fake_registry() -> ToolRegistry:
        return registry

    monkeypatch.setattr(executor_module, "get_tool_registry", fake_registry)
    plan = DiagnosticPlan(goal="诊断 CPU", steps=[make_step()])
    state = {
        "input": "诊断 CPU",
        "context": DiagnosisContext(service_name="orders-service"),
        "plan": plan,
        "execution_results": [],
        "response": "",
        "replan_count": 0,
    }

    update = await executor(state)
    result = update["execution_results"][0]

    assert tool.arguments == {"service_name": "orders-service"}
    assert result.status == "succeeded"
    assert result.output == {"data": [{"value": 91.2}]}
    assert result.criteria_results[0].passed is True


@pytest.mark.asyncio
async def test_executor_records_unknown_tool_without_keyword_fallback(monkeypatch) -> None:
    tool = FakeTool()
    registry = make_registry(tool)

    async def fake_registry() -> ToolRegistry:
        return registry

    monkeypatch.setattr(executor_module, "get_tool_registry", fake_registry)
    state = {
        "input": "诊断日志",
        "context": DiagnosisContext(service_name="orders-service"),
        "plan": DiagnosticPlan(goal="诊断日志", steps=[make_step("missing_tool")]),
        "execution_results": [],
        "response": "",
        "replan_count": 0,
    }

    update = await executor(state)
    result = update["execution_results"][0]

    assert result.status == "invalid_tool"
    assert result.error is not None
    assert result.error.type == "UnknownToolError"


def test_replan_keeps_completed_steps_and_replaces_pending_steps() -> None:
    tool = FakeTool()
    registry = make_registry(tool)
    completed_step = make_step()
    pending_step = completed_step.model_copy(
        update={"id": "old_pending", "depends_on": ["query_cpu"]}
    )
    replacement_step = completed_step.model_copy(
        update={"id": "replacement", "depends_on": ["query_cpu"]}
    )
    now = datetime.now(UTC)
    completed_result = StepExecutionResult(
        step_id="query_cpu",
        step_title=completed_step.title,
        tool_name=completed_step.tool_call.tool_name,
        status="succeeded",
        output={"data": [{"value": 91.2}]},
        started_at=now,
        finished_at=now,
        duration_ms=0,
    )
    state = {
        "input": "诊断 CPU",
        "context": DiagnosisContext(service_name="orders-service"),
        "plan": DiagnosticPlan(goal="诊断 CPU", steps=[completed_step, pending_step]),
        "execution_results": [completed_result],
        "response": "",
        "replan_count": 0,
    }

    update = _apply_replan(state, [replacement_step], registry)

    assert [step.id for step in update["plan"].steps] == ["query_cpu", "replacement"]
    assert update["replan_count"] == 1


def test_registry_accepts_mcp_json_schema_arguments() -> None:
    tool = FakeMcpTool()
    registry = ToolRegistry(
        descriptors={
            tool.name: ToolDescriptor(
                name=tool.name,
                description=tool.description,
                source="mcp",
                input_schema=tool.args_schema,
            )
        },
        handlers={tool.name: tool},
    )

    assert registry.validate_arguments(tool.name, {}) == {}


@pytest.mark.asyncio
async def test_executor_invokes_tool_with_mcp_json_schema(monkeypatch) -> None:
    tool = FakeMcpTool()
    registry = ToolRegistry(
        descriptors={
            tool.name: ToolDescriptor(
                name=tool.name,
                description=tool.description,
                source="mcp",
                input_schema=tool.args_schema,
            )
        },
        handlers={tool.name: tool},
    )

    async def fake_registry() -> ToolRegistry:
        return registry

    monkeypatch.setattr(executor_module, "get_tool_registry", fake_registry)
    step = DiagnosticStep(
        id="get_timestamp",
        title="获取当前时间戳",
        purpose="用于查询时间范围",
        tool_call=ToolCallSpec(tool_name=tool.name),
        success_criteria=[SuccessCriterion(operator="exists")],
    )
    state = {
        "input": "诊断 CPU",
        "context": DiagnosisContext(service_name="orders-service"),
        "plan": DiagnosticPlan(goal="诊断 CPU", steps=[step]),
        "execution_results": [],
        "response": "",
        "replan_count": 0,
    }

    update = await executor(state)

    assert update["execution_results"][0].status == "succeeded"
    assert update["execution_results"][0].output == 1_787_972_400_000


def test_chunk_text_extracts_string_and_text_blocks() -> None:
    assert _chunk_text(SimpleNamespace(content="诊断")) == "诊断"
    assert (
        _chunk_text(
            SimpleNamespace(
                content_blocks=[
                    {"type": "reasoning", "text": "hidden"},
                    {"type": "text", "text": "报告"},
                ]
            )
        )
        == "报告"
    )


@pytest.mark.asyncio
async def test_generate_response_forwards_each_model_chunk(monkeypatch) -> None:
    class FakeChain:
        async def astream(self, _input):
            yield SimpleNamespace(content="第一段")
            yield SimpleNamespace(content="第二段")

    class FakePrompt:
        def __or__(self, _llm):
            return FakeChain()

    streamed: list[dict[str, Any]] = []
    monkeypatch.setattr(replanner_module, "response_prompt", FakePrompt())
    monkeypatch.setattr(replanner_module, "get_stream_writer", lambda: streamed.append)
    state = {
        "input": "诊断 CPU",
        "context": DiagnosisContext(service_name="orders-service"),
        "plan": None,
        "execution_results": [],
        "response": "",
        "replan_count": 0,
    }

    update = await _generate_response(state, object())

    assert update == {"response": "第一段第二段"}
    assert [event["data"] for event in streamed] == ["第一段", "第二段"]
    assert all(event["type"] == "report_chunk" for event in streamed)


@pytest.mark.asyncio
async def test_aiops_service_forwards_chunks_without_duplicate_report() -> None:
    plan = DiagnosticPlan(goal="诊断 CPU", steps=[make_step()])

    class FakeGraph:
        async def astream(self, **kwargs):
            assert kwargs["stream_mode"] == ["updates", "custom"]
            yield "updates", {"planner": {"plan": plan}}
            yield "custom", {"type": "report_chunk", "data": "第一段"}
            yield "custom", {"type": "report_chunk", "data": "第二段"}
            yield "updates", {"replanner": {"response": "第一段第二段"}}

        def get_state(self, _config):
            return SimpleNamespace(values={"response": "第一段第二段"})

    service = AIOpsService.__new__(AIOpsService)
    service.graph = FakeGraph()

    events = [event async for event in service.execute("诊断 CPU", session_id="test")]

    assert [event["type"] for event in events] == [
        "plan",
        "report_chunk",
        "report_chunk",
        "complete",
    ]
    assert "report" not in [event["type"] for event in events]


@pytest.mark.asyncio
async def test_graph_returns_to_executor_after_replan(monkeypatch) -> None:
    first_step = make_step()
    retry_step = first_step.model_copy(update={"id": "query_cpu_retry"})
    initial_plan = DiagnosticPlan(goal="诊断 CPU", steps=[first_step])
    replanned = DiagnosticPlan(goal="诊断 CPU", steps=[first_step, retry_step])
    executor_calls = 0

    async def fake_planner(_state):
        return {"plan": initial_plan}

    async def fake_executor(state):
        nonlocal executor_calls
        executor_calls += 1
        step = first_step if executor_calls == 1 else retry_step
        now = datetime.now(UTC)
        result = StepExecutionResult(
            step_id=step.id,
            step_title=step.title,
            tool_name=step.tool_call.tool_name,
            status="failed" if executor_calls == 1 else "succeeded",
            output=None if executor_calls == 1 else {"data": [{"value": 42}]},
            error=(
                {"type": "ToolError", "message": "temporary failure"}
                if executor_calls == 1
                else None
            ),
            started_at=now,
            finished_at=now,
            duration_ms=0,
        )
        return {
            "plan": state["plan"],
            "execution_results": state.get("execution_results", []) + [result],
        }

    async def fake_replanner(state):
        if len(state.get("execution_results", [])) == 1:
            return {"plan": replanned, "replan_count": 1}
        return {"response": "diagnosis complete"}

    monkeypatch.setattr(aiops_service_module, "planner", fake_planner)
    monkeypatch.setattr(aiops_service_module, "executor", fake_executor)
    monkeypatch.setattr(aiops_service_module, "replanner", fake_replanner)
    service = AIOpsService()

    events = [event async for event in service.execute("诊断 CPU", session_id="replan-test")]

    assert executor_calls == 2
    assert events[-1]["type"] == "complete"
    assert events[-1]["response"] == "diagnosis complete"
