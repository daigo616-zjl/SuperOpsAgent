"""Replanner 节点：依据结构化执行结果继续、重规划或生成报告。"""

import json
from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_qwq import ChatQwen
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.core.llm_factory import LLMFactory

from .models import DiagnosticPlan, DiagnosticStep, ReplanDecision
from .state import PlanExecuteState
from .tool_registry import get_tool_registry


class Response(BaseModel):
    response: str = Field(description="基于实际工具结果生成的最终 Markdown 响应")


replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是运维诊断 Replanner。你只能依据结构化计划与执行结果决定：
                continue、replan 或 respond，不执行工具。

                诊断上下文：
                {diagnosis_context}

                可用工具注册表：
                {tool_registry}

                决策规则：
                - 信息足以回答原始任务时 respond。
                - 剩余步骤仍然必要且可执行时 continue。
                - 工具无效、参数无效、关键步骤失败，或现有计划不能继续时 replan。
                - replan 时 updated_steps 只包含新的未执行步骤，每步一次工具调用。
                - updated_steps 的 id 不得与已执行步骤 id 重复；重试需使用新的 id。
                - 服务名必须通过 context 引用，不得写死。
                - 步骤 id 使用英文字母开头，只包含英文字母、数字、下划线或连字符。
                - 数值引用需要偏移时使用 offset，例如时间戳减 15 分钟为 -900000。
                - 不得把失败结果描述成成功，也不得编造工具输出。
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务、诊断上下文、结构化计划和结构化执行结果生成最终响应。
                使用 Markdown；只引用真实工具输出；明确标注失败、阻塞和缺失证据。
                诊断结论要区分已证实事实、合理推断和无法确认的事项。
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def replanner(state: PlanExecuteState) -> dict[str, Any]:
    logger.info("=== Replanner：评估结构化执行结果 ===")
    if state.get("response"):
        return {}

    plan = state.get("plan")
    results = state.get("execution_results", [])
    if plan is None:
        return {"response": "诊断未能生成有效执行计划。"}

    result_by_id = {result.step_id: result for result in results}
    pending = [step for step in plan.steps if step.id not in result_by_id]
    latest = results[-1] if results else None
    step_by_id = {step.id: step for step in plan.steps}

    llm = LLMFactory.create_qwen_chat_model(model=config.rag_model, temperature=0)

    if not pending:
        return await _generate_response(state, llm)

    if latest is not None:
        latest_step = step_by_id.get(latest.step_id)
        if latest_step and latest.status != "succeeded":
            if latest_step.failure_policy == "stop":
                return await _generate_response(state, llm)
            if latest_step.failure_policy == "continue":
                return {}

    if len(results) >= 8 or state.get("replan_count", 0) >= 2:
        return await _generate_response(state, llm)

    try:
        registry = await get_tool_registry()
        chain = replanner_prompt | llm.with_structured_output(ReplanDecision)
        decision = await chain.ainvoke(
            {
                "messages": [
                    ("user", f"原始任务：{state.get('input', '')}"),
                    ("user", f"完整计划：{plan.model_dump_json(indent=2)}"),
                    ("user", f"执行结果：{_results_json(results)}"),
                    ("user", f"剩余步骤：{_steps_json(pending)}"),
                ],
                "diagnosis_context": state["context"].model_dump_json(indent=2),
                "tool_registry": registry.prompt_description(),
            }
        )
        if isinstance(decision, dict):
            decision = ReplanDecision.model_validate(decision)
        if not isinstance(decision, ReplanDecision):
            raise ValueError("Replanner 未返回有效决策")

        logger.info(f"Replanner 决策={decision.action}, 原因={decision.reason}")
        if decision.action == "respond":
            return await _generate_response(state, llm)
        if decision.action == "continue":
            return {}
        return _apply_replan(state, decision.updated_steps, registry)
    except Exception as exc:
        logger.error(f"Replanner 决策失败: {exc}", exc_info=True)
        if latest is not None and latest.status != "succeeded":
            return await _generate_response(state, llm)
        return {}


def _apply_replan(
    state: PlanExecuteState,
    updated_steps: list[DiagnosticStep],
    registry: Any,
) -> dict:
    if not updated_steps:
        raise ValueError("replan 决策没有提供替代步骤")

    plan = state["plan"]
    assert plan is not None
    completed_ids = {result.step_id for result in state.get("execution_results", [])}
    completed_steps = [step for step in plan.steps if step.id in completed_ids]
    if len(updated_steps) > len(plan.steps):
        raise ValueError("重新规划的步骤数量超过原计划上限")

    new_plan = DiagnosticPlan(goal=plan.goal, steps=completed_steps + updated_steps)
    registry.validate_plan(new_plan)
    return {
        "plan": new_plan,
        "replan_count": state.get("replan_count", 0) + 1,
    }


async def _generate_response(state: PlanExecuteState, llm: ChatQwen) -> dict[str, Any]:
    plan = state.get("plan")
    messages = [
        ("user", f"原始任务：{state.get('input', '')}"),
        ("user", f"诊断上下文：{state['context'].model_dump_json(indent=2)}"),
        ("user", f"执行计划：{plan.model_dump_json(indent=2) if plan else '无'}"),
        ("user", f"执行结果：{_results_json(state.get('execution_results', []))}"),
        ("user", "请生成最终诊断报告。"),
    ]
    try:
        response_gen = response_prompt | llm.with_structured_output(Response)
        response_obj = await response_gen.ainvoke({"messages": messages})
        if isinstance(response_obj, dict):
            response_obj = Response.model_validate(response_obj)
        if not isinstance(response_obj, Response) or not response_obj.response.strip():
            raise ValueError("结构化最终响应为空")
        return {"response": response_obj.response}
    except Exception as exc:
        logger.error(f"结构化最终响应生成失败: {exc}")
        try:
            plain_response = await llm.ainvoke(messages)
            content = getattr(plain_response, "content", plain_response)
            if isinstance(content, str) and content.strip():
                return {"response": content}
        except Exception as fallback_exc:
            logger.error(f"普通文本响应降级失败: {fallback_exc}")
        return {"response": _fallback_response(state)}


def _results_json(results: list) -> str:
    return json.dumps(
        [result.model_dump(mode="json") for result in results],
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _steps_json(steps: list[DiagnosticStep]) -> str:
    return json.dumps(
        [step.model_dump(mode="json") for step in steps],
        ensure_ascii=False,
        indent=2,
    )


def _fallback_response(state: PlanExecuteState) -> str:
    results = state.get("execution_results", [])
    lines = ["# 诊断执行结果", "", f"目标服务：{state['context'].service_name}", ""]
    for result in results:
        lines.extend(
            [
                f"## {result.step_title}",
                f"- 状态：{result.status}",
                f"- 工具：{result.tool_name}",
                f"- 输出：`{json.dumps(result.output, ensure_ascii=False, default=str)}`",
                f"- 错误：{result.error.message if result.error else '无'}",
                "",
            ]
        )
    return "\n".join(lines)
