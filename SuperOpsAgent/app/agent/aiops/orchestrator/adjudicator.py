"""Adjudicator 节点：基于证据子集淘汰假设并派生定向取证指令。"""

import json
from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.agent.aiops.diagnosis_models import (
    AdjudicationDecision,
    EvidenceCard,
    Hypothesis,
)
from app.config import config
from app.core.llm_factory import LLMFactory

from .state import OrchestratorState

adjudicator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是运维诊断 Adjudicator，只依据提供的证据评审候选假设，不执行工具。

                职责：
                - eliminations：证据足以排除的假设。ruled_out_by 必须填写
                  证据中真实存在的 claim_id，reason 说明排除逻辑。
                - new_directives：关键结论缺少证据支撑时，派生定向取证指令
                  （objective 要具体到"查什么、期望看到什么"），max_iterations 保守取 3。
                - contradictions：记录互相冲突的证据。
                - converged：仅当某个假设获得多域证据一致支持、且竞争假设均被
                  淘汰或明显弱势时才置 true，并填写 converged_hypothesis_id。
                - 不得虚构 claim_id 或假设 id；引用前先核对输入清单。
                - 证据不足且无法派生有效指令时，不要强行收敛。
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


def _claims_index(evidence: list[EvidenceCard]) -> dict[str, EvidenceCard]:
    index: dict[str, EvidenceCard] = {}
    for card in evidence:
        for claim in card.claims:
            index[claim.claim_id] = card
    return index


def _validate_decision(
    decision: AdjudicationDecision,
    hypotheses: list[Hypothesis],
    claim_ids: set[str],
) -> str | None:
    hypothesis_ids = {h.id for h in hypotheses}
    for item in decision.eliminations:
        if item.hypothesis_id not in hypothesis_ids:
            return f"eliminations 引用了未知假设: {item.hypothesis_id}"
        unknown = [c for c in item.ruled_out_by if c not in claim_ids]
        if unknown:
            return f"eliminations 引用了不存在的 claim: {unknown}"
    for directive in decision.new_directives:
        unknown = [h for h in directive.hypothesis_ids if h not in hypothesis_ids]
        if unknown:
            return f"new_directives 引用了未知假设: {unknown}"
        if directive.target_domain not in ("metrics", "logs", "knowledge"):
            return f"new_directives 目标域非法: {directive.target_domain}"
    if decision.converged:
        if decision.converged_hypothesis_id not in hypothesis_ids:
            return f"converged_hypothesis_id 不在假设清单中: {decision.converged_hypothesis_id}"
    return None


async def adjudicator(state: OrchestratorState) -> dict[str, Any]:
    logger.info("=== Adjudicator：评审证据与假设 ===")
    hypotheses = state.get("hypotheses", [])
    evidence = state.get("evidence", [])
    try:
        decision = await _adjudicate(state, hypotheses, evidence)
    except Exception as exc:
        logger.exception("评审失败: {}", exc)
        decision = AdjudicationDecision(
            contradictions=[f"评审异常: {exc}"],
        )
    return {"pending_decision": decision}


async def _adjudicate(
    state: OrchestratorState,
    hypotheses: list[Hypothesis],
    evidence: list[EvidenceCard],
) -> AdjudicationDecision:
    llm = LLMFactory.create_qwen_chat_model(
        model=config.aiops_adjudicator_model or config.rag_model, temperature=0
    )
    chain = adjudicator_prompt | llm.with_structured_output(AdjudicationDecision)
    claim_ids = set(_claims_index(evidence))
    messages = [
        (
            "user",
            "候选假设：\n"
            + json.dumps(
                [h.model_dump(mode="json") for h in hypotheses], ensure_ascii=False, indent=2
            )
            + "\n\n证据卡（claims 为可信证据子集）：\n"
            + json.dumps(
                [card.model_dump(mode="json") for card in evidence],
                ensure_ascii=False,
                indent=2,
                default=str,
            )[:30000]
            + "\n\n原始任务："
            + state.get("input", ""),
        )
    ]
    # Planner 同款校验回喂重试：引用越界是可修正错误，不直接终止评审。
    for attempt in range(2):
        raw = await chain.ainvoke({"messages": messages})
        decision = (
            raw if isinstance(raw, AdjudicationDecision) else AdjudicationDecision.model_validate(raw)
        )
        error = _validate_decision(decision, hypotheses, claim_ids)
        if error is None:
            return decision
        logger.warning("评审决策校验失败，反馈错误后重试: {}", error)
        messages.append(
            (
                "user",
                f"上一次决策未通过校验：{error}\n"
                "claim_id 与 hypothesis_id 必须来自上方清单，请完整重新输出。",
            )
        )
    # 两次都失败时丢弃非法引用，保留可用的部分（宁可少淘汰，不可错淘汰）。
    logger.warning("评审决策两次校验失败，降级为仅保留合法引用")
    safe_eliminations = [
        item
        for item in decision.eliminations
        if item.hypothesis_id in {h.id for h in hypotheses}
        and all(c in claim_ids for c in item.ruled_out_by)
    ]
    return AdjudicationDecision(
        eliminations=safe_eliminations,
        contradictions=[f"评审校验失败，部分决策被丢弃: {error}"],
    )
