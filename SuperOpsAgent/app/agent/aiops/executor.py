"""Executor 节点：校验并执行 Planner 指定的单个工具调用。"""

from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from loguru import logger

from .models import (
    CriterionResult,
    DiagnosticStep,
    ExecutionError,
    StepExecutionResult,
)
from .state import PlanExecuteState
from .tool_registry import InvalidToolArgumentsError, UnknownToolError, get_tool_registry
from .tool_runtime import (
    evaluate_criterion as _evaluate_criterion,
    get_path as _get_path,
    normalize_output as _normalize_output,
    resolve_value as _resolve_value,
)


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
        logger.exception("工具执行失败: {}", exc)
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
