import importlib
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from app.agent.aiops.executor import _resolve_value, executor
from app.agent.aiops.models import (
    DiagnosisContext,
    DiagnosticPlan,
    DiagnosticStep,
    StepExecutionResult,
    SuccessCriterion,
    ToolCallSpec,
)
from app.agent.aiops.replanner import _apply_replan
from app.agent.aiops.tool_registry import ToolDescriptor, ToolRegistry

executor_module = importlib.import_module("app.agent.aiops.executor")


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
