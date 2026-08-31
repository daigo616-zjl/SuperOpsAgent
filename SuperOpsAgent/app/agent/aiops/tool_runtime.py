"""工具运行时：legacy executor 与取证 Agent 共用的机械内核。

这里只做参数解析、校验转发、输出归一化与出处摘要，绝不调 LLM。
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from loguru import logger

from .models import (
    CriterionResult,
    SuccessCriterion,
    ValueReference,
)
from .tool_registry import ToolRegistry


def resolve_value(
    value: Any,
    context: dict[str, Any],
    results: dict[str, Any],
) -> Any:
    if isinstance(value, dict) and value.get("source") in {"context", "step"}:
        reference = ValueReference.model_validate(value)
        if reference.source == "context":
            resolved = get_path(context, reference.path)
        else:
            result = results.get(reference.step_id or "")
            if result is None:
                raise KeyError(f"引用的步骤结果不存在: {reference.step_id}")
            resolved = get_path(result.output, reference.path)
        if reference.offset is not None:
            if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
                raise ValueError("只有数值参数引用可以使用 offset")
            resolved += reference.offset
        return resolved
    if isinstance(value, dict):
        return {key: resolve_value(item, context, results) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_value(item, context, results) for item in value]
    return value


def get_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    # Accept both JSON-style list indexes (topics[0].topic_id) and the dot
    # notation used by the planner contract (topics.0.topic_id).
    normalized_path = re.sub(r"\[(\d+)\]", r".\1", path).strip(".")
    for part in normalized_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(f"字段路径不存在: {path}")
    return current


def normalize_output(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 1 and _is_text_content(value[0]):
            return normalize_output(value[0])
        return [normalize_output(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return normalize_output(value["text"])
        if "content" in value and len(value) == 1:
            return normalize_output(value["content"])
        return {key: normalize_output(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            return normalize_output(json.loads(value))
        except json.JSONDecodeError:
            return value
    if hasattr(value, "model_dump"):
        return normalize_output(value.model_dump())
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _is_text_content(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("type") == "text" and "text" in value
    return getattr(value, "type", None) == "text" and hasattr(value, "text")


def evaluate_criterion(criterion: SuccessCriterion, output: Any) -> CriterionResult:
    try:
        actual = get_path(output, criterion.path)
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


def args_digest(tool_name: str, arguments: dict[str, Any], content: str = "") -> str:
    """工具调用的确定性摘要，作为证据 provenance 的出处指纹。"""
    payload = json.dumps(
        {"tool": tool_name, "args": arguments, "content": content[:500]},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def invoke_tool_normalized(
    registry: ToolRegistry, tool_name: str, arguments: dict[str, Any]
) -> tuple[Any, str]:
    """校验并调用工具，返回（归一化输出, 参数摘要）。"""
    registry.get_descriptor(tool_name)
    validated = registry.validate_arguments(tool_name, arguments)
    raw_output = await registry.invoke(tool_name, validated)
    return normalize_output(raw_output), args_digest(tool_name, validated)


def utc_now() -> datetime:
    return datetime.now(UTC)


def elapsed_ms(started_counter: float) -> int:
    return max(0, round((perf_counter() - started_counter) * 1000))
