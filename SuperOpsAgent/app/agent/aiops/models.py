"""AIOps 结构化计划、上下文和执行结果模型。"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DiagnosisContext(BaseModel):
    """贯穿诊断流程且由服务层统一解析的上下文。"""

    service_name: str = Field(min_length=1, description="本次诊断的目标服务")


class ValueReference(BaseModel):
    """工具参数对诊断上下文或既有步骤输出的引用。"""

    source: Literal["context", "step"]
    path: str | None = None
    step_id: str | None = None
    offset: int | float | None = Field(
        default=None,
        description="对解析出的数值执行加法偏移，例如当前时间戳减 15 分钟为 -900000",
    )

    @model_validator(mode="after")
    def validate_step_reference(self) -> "ValueReference":
        if self.source == "step" and not self.step_id:
            raise ValueError("step 引用必须提供 step_id")
        if self.source == "context" and self.step_id:
            raise ValueError("context 引用不能提供 step_id")
        if self.source == "context" and not self.path:
            raise ValueError("context 引用必须提供 path")
        return self


class ToolCallSpec(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class SuccessCriterion(BaseModel):
    path: str | None = None
    operator: Literal["exists", "not_empty", "eq", "ne", "gt", "gte", "lt", "lte"]
    expected: Any | None = None

    @model_validator(mode="after")
    def validate_expected_value(self) -> "SuccessCriterion":
        if self.operator not in {"exists", "not_empty"} and self.expected is None:
            raise ValueError(f"{self.operator} 条件必须提供 expected")
        return self


class DiagnosticStep(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    tool_call: ToolCallSpec
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    failure_policy: Literal["stop", "continue", "replan"] = "replan"


class DiagnosticPlan(BaseModel):
    goal: str = Field(min_length=1)
    steps: list[DiagnosticStep] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "DiagnosticPlan":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("计划步骤 id 必须唯一")

        known: set[str] = set()
        for step in self.steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise ValueError(f"步骤 {step.id} 依赖不存在或尚未声明的步骤: {sorted(unknown)}")
            for reference in _find_references(step.tool_call.arguments):
                if reference.source == "context" and reference.path != "service_name":
                    raise ValueError(f"步骤 {step.id} 引用了未知上下文字段: {reference.path}")
                if reference.source == "step" and reference.step_id not in step.depends_on:
                    raise ValueError(f"步骤 {step.id} 引用了 {reference.step_id}，但未声明对应依赖")
            known.add(step.id)
        return self


def _find_references(value: Any) -> list[ValueReference]:
    references: list[ValueReference] = []
    if isinstance(value, dict) and value.get("source") in {"context", "step"}:
        references.append(ValueReference.model_validate(value))
    elif isinstance(value, dict):
        for item in value.values():
            references.extend(_find_references(item))
    elif isinstance(value, list):
        for item in value:
            references.extend(_find_references(item))
    return references


class ExecutionError(BaseModel):
    type: str
    message: str


class CriterionResult(BaseModel):
    criterion: SuccessCriterion
    passed: bool
    actual: Any | None = None
    message: str = ""


class StepExecutionResult(BaseModel):
    step_id: str
    step_title: str
    tool_name: str
    status: Literal["succeeded", "failed", "blocked", "invalid_tool", "invalid_arguments"]
    resolved_arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any | None = None
    error: ExecutionError | None = None
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(ge=0)
    attempt: int = Field(default=1, ge=1)


class ReplanDecision(BaseModel):
    action: Literal["continue", "replan", "respond"]
    reason: str
    updated_steps: list[DiagnosticStep] = Field(default_factory=list)
