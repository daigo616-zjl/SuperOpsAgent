"""Hypothesizer 节点：基于观测生成鉴别性候选假设。"""

import json
from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.core.llm_factory import LLMFactory
from app.tools import retrieve_knowledge

from .state import OrchestratorState

hypothesizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是运维诊断 Hypothesizer。根据观测生成 2-4 个候选根因假设，
                用于后续鉴别性取证，不执行工具。

                规划规则：
                - 假设之间必须互斥或可鉴别：每个假设写出
                  expected_support（若为真应观察到什么）和
                  expected_refuting（若为真不应观察到什么）。
                - 假设要覆盖常见根因类别（资源压力、依赖故障、容量、配置变更等），
                  不要只罗列症状。
                - id 使用 hyp- 前缀加短横线小写词，如 hyp-gc-pressure。
                - prior 是 0-1 的先验概率，总和不必为 1。
                - 只依据观测内容提出假设，不得编造观测中不存在的现象。
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


class HypothesisDraft(BaseModel):
    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    prior: float = Field(ge=0.0, le=1.0, default=0.5)
    expected_support: list[str] = Field(default_factory=list)
    expected_refuting: list[str] = Field(default_factory=list)


class HypothesisSet(BaseModel):
    hypotheses: list[HypothesisDraft] = Field(min_length=1)


async def _fetch_alerts(service_name: str) -> str:
    from app.agent.aiops.tool_registry import get_tool_registry

    registry = await get_tool_registry()
    if "query_active_alerts" not in registry.handlers:
        return "（无告警查询工具）"
    output = await registry.invoke(
        "query_active_alerts", {"service_name": service_name}
    )
    return json.dumps(output, ensure_ascii=False, default=str)[:4000]


async def _fetch_experience(user_input: str) -> str:
    try:
        result = await retrieve_knowledge.ainvoke({"query": user_input})
        if isinstance(result, str) and result.strip():
            return result
    except Exception as exc:
        logger.warning(f"查询内部经验失败: {exc}")
    return ""


async def hypothesizer(state: OrchestratorState) -> dict[str, Any]:
    logger.info("=== Hypothesizer：生成鉴别性候选假设 ===")
    context = state["context"]
    try:
        alerts, experience = await _gather_observations(
            context.service_name, state.get("input", "")
        )
        llm = LLMFactory.create_qwen_chat_model(
            model=config.aiops_hypothesizer_model or config.rag_model, temperature=0
        )
        chain = hypothesizer_prompt | llm.with_structured_output(HypothesisSet)
        messages = [("user", state.get("input", ""))]
        chain_input = {
            "messages": messages,
            "observation": f"当前告警：\n{alerts}\n\n相关运维经验：\n{experience or '无'}",
        }
        # 空集或不合法输出会在结构化校验层抛错，由外层统一降级处理。
        raw = await chain.ainvoke(chain_input)
        draft = raw if isinstance(raw, HypothesisSet) else HypothesisSet.model_validate(raw)

        from app.agent.aiops.diagnosis_models import Hypothesis

        hypotheses = [
            Hypothesis(
                id=item.id,
                statement=item.statement,
                prior=item.prior,
                expected_support=item.expected_support,
                expected_refuting=item.expected_refuting,
            )
            for item in draft.hypotheses
        ]
        logger.info(f"已生成 {len(hypotheses)} 个候选假设")
        return {"hypotheses": hypotheses}
    except Exception as exc:
        logger.exception("假设生成失败: {}", exc)
        return {
            "hypotheses": [],
            "investigation_errors": [f"hypothesizer: {exc}"],
        }


async def _gather_observations(service_name: str, user_input: str) -> tuple[str, str]:
    try:
        alerts = await _fetch_alerts(service_name)
    except Exception as exc:
        logger.warning(f"获取告警观测失败: {exc}")
        alerts = f"（告警查询失败: {exc}）"
    return alerts, await _fetch_experience(user_input)
