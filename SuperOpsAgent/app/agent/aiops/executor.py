"""Executor 节点：校验并执行 Planner 指定的单个工具调用。"""

import json
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from loguru import logger

from .models import (
    CriterionResult,
    DiagnosticStep,
    ExecutionError,
    StepExecutionResult,
    SuccessCriterion,
    ValueReference,
)
from .state import PlanExecuteState
from .tool_registry import InvalidToolArgumentsError, UnknownToolError, get_tool_registry


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    logger.info("=== Executor：执行结构化步骤 ===")
    if state.get("response") or state.get("plan") is None:
        return {}

    plan = state["plan"]
    results = state.get("execution_results", [])
    result_by_id = {result.step_id: result for result in results}
    pending = [step for step in plan.steps if step.id not in result_by_id]
    if not pending:
        return {}

    step = pending[0]
    failed_dependencies = [
        dep
        for dep in step.depends_on
        if dep in result_by_id and result_by_id[dep].status != "succeeded"
    ]
    unresolved_dependencies = [dep for dep in step.depends_on if dep not in result_by_id]
    if failed_dependencies or unresolved_dependencies:
        dependencies = failed_dependencies + unresolved_dependencies
        return {
            "plan": plan,
            "execution_results": results
            + [
                _instant_result(
                    step,
                    "blocked",
                    ExecutionError(
                        type="DependencyError",
                        message=f"依赖步骤未成功完成: {dependencies}",
                    ),
                )
            ],
        }

    started_at = datetime.now(UTC)
    started_counter = perf_counter()
    resolved_arguments: dict[str, Any] = {}
    try:
        registry = await get_tool_registry()
        registry.get_descriptor(step.tool_call.tool_name)
        resolved_arguments = _resolve_value(
            step.tool_call.arguments,
            state["context"].model_dump(),
            result_by_id,
        )
        resolved_arguments = registry.validate_arguments(
            step.tool_call.tool_name, resolved_arguments
        )
        output = await registry.invoke(step.tool_call.tool_name, resolved_arguments)
        normalized_output = _normalize_output(output)
        criteria_results = [
            _evaluate_criterion(criterion, normalized_output) for criterion in step.success_criteria
        ]
        status = "succeeded" if all(item.passed for item in criteria_results) else "failed"
        error = None
        if status == "failed":
            error = ExecutionError(
                type="SuccessCriteriaNotMet",
                message="工具调用成功，但输出未满足步骤成功标准",
            )
        result = _completed_result(
            step,
            status,
            started_at,
            started_counter,
            resolved_arguments,
            output=normalized_output,
            error=error,
            criteria_results=criteria_results,
        )
    except UnknownToolError as exc:
        result = _completed_result(
            step,
            "invalid_tool",
            started_at,
            started_counter,
            resolved_arguments,
            error=ExecutionError(type=type(exc).__name__, message=str(exc)),
        )
    except (InvalidToolArgumentsError, ValueError, KeyError) as exc:
        result = _completed_result(
            step,
            "invalid_arguments",
            started_at,
            started_counter,
            resolved_arguments,
            error=ExecutionError(type=type(exc).__name__, message=str(exc)),
        )
    except Exception as exc:
        logger.error(f"工具执行失败: {exc}", exc_info=True)
        result = _completed_result(
            step,
            "failed",
            started_at,
            started_counter,
            resolved_arguments,
            error=ExecutionError(type=type(exc).__name__, message=str(exc)),
        )

    logger.info(f"步骤 {step.id} 执行结束，状态={result.status}")
    return {"plan": plan, "execution_results": results + [result]}


def _resolve_value(
    value: Any,
    context: dict[str, Any],
    results: dict[str, StepExecutionResult],
) -> Any:
    if isinstance(value, dict) and value.get("source") in {"context", "step"}:
        reference = ValueReference.model_validate(value)
        if reference.source == "context":
            resolved = _get_path(context, reference.path)
        else:
            result = results.get(reference.step_id or "")
            if result is None:
                raise KeyError(f"引用的步骤结果不存在: {reference.step_id}")
            resolved = _get_path(result.output, reference.path)
        if reference.offset is not None:
            if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
                raise ValueError("只有数值参数引用可以使用 offset")
            resolved += reference.offset
        return resolved
    if isinstance(value, dict):
        return {key: _resolve_value(item, context, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, context, results) for item in value]
    return value


def _get_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(f"字段路径不存在: {path}")
    return current


def _normalize_output(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1 and _is_text_content(value[0]):
            return _normalize_output(value[0])
        return [_normalize_output(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return _normalize_output(value["text"])
        if "content" in value and len(value) == 1:
            return _normalize_output(value["content"])
        return {key: _normalize_output(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            return _normalize_output(json.loads(value))
        except json.JSONDecodeError:
            return value
    if hasattr(value, "model_dump"):
        return _normalize_output(value.model_dump())
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _is_text_content(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("type") == "text" and "text" in value
    return getattr(value, "type", None) == "text" and hasattr(value, "text")


def _evaluate_criterion(criterion: SuccessCriterion, output: Any) -> CriterionResult:
    try:
        actual = _get_path(output, criterion.path)
        operations = {
            "exists": lambda: actual is not None,
            "not_empty": lambda: actual not in (None, "", [], {}),
            "eq": lambda: actual == criterion.expected,
            "ne": lambda: actual != criterion.expected,
            "gt": lambda: actual > criterion.expected,
            "gte": lambda: actual >= criterion.expected,
            "lt": lambda: actual < criterion.expected,
            "lte": lambda: actual <= criterion.expected,
        }
        passed = bool(operations[criterion.operator]())
        return CriterionResult(criterion=criterion, passed=passed, actual=actual)
    except (KeyError, TypeError, ValueError) as exc:
        return CriterionResult(criterion=criterion, passed=False, message=str(exc))


def _completed_result(
    step: DiagnosticStep,
    status: Any,
    started_at: datetime,
    started_counter: float,
    resolved_arguments: dict[str, Any],
    output: Any = None,
    error: ExecutionError | None = None,
    criteria_results: list[CriterionResult] | None = None,
) -> StepExecutionResult:
    finished_at = datetime.now(UTC)
    return StepExecutionResult(
        step_id=step.id,
        step_title=step.title,
        tool_name=step.tool_call.tool_name,
        status=status,
        resolved_arguments=resolved_arguments,
        output=output,
        error=error,
        criteria_results=criteria_results or [],
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=max(0, round((perf_counter() - started_counter) * 1000)),
    )


def _instant_result(
    step: DiagnosticStep, status: Any, error: ExecutionError
) -> StepExecutionResult:
    now = datetime.now(UTC)
    return StepExecutionResult(
        step_id=step.id,
        step_title=step.title,
        tool_name=step.tool_call.tool_name,
        status=status,
        error=error,
        started_at=now,
        finished_at=now,
        duration_ms=0,
    )
