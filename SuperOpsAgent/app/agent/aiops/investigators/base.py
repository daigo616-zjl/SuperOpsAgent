"""Investigator 基座：ReAct 取证子图 + 机械证据记录。

LLM 只负责决定调用哪些工具、如何解读；证据出处（ClaimProvenance）
由代码从真实工具调用记录确定性构建，模型无法虚构 provenance。
"""

import time
from textwrap import dedent
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.core.llm_factory import LLMFactory
from app.tools.knowledge_tool import capture_retrieval_for_session

from ..diagnosis_models import (
    ClaimProvenance,
    Directive,
    EvidenceCard,
    EvidenceClaim,
    EvidencePolarity,
)
from ..tool_registry import get_domain_registry
from ..tool_runtime import args_digest, utc_now

DRAFT_PROMPT = dedent("""
    你是取证分析员。上面是你刚刚执行的工具调用记录（工具名、参数、原始输出）。

    请基于记录中的真实输出产出证据草稿：
    - 只允许引用真实发生过的工具调用：call_index 必须指向记录列表中的某一行。
    - excerpt 必须是该工具输出原文的忠实摘录，不得改写数值或编造内容。
    - statement 是一条独立、可核查的证据判断（一句话）。
    - polarity：supports 表示支持假设，refutes 表示反驳假设，neutral 表示无关事实。
    - hypothesis_ids 只能填写任务指令中列出的假设 ID。
    - confidence 反映该判断的确定性（0-1）。
    - summary 概述本轮取证结论。
""").strip()


class DraftClaim(BaseModel):
    """模型产出的证据草稿，call_index 指向真实工具调用记录。"""

    call_index: int = Field(ge=0)
    statement: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    polarity: EvidencePolarity = "neutral"
    hypothesis_ids: list[str] = Field(default_factory=list)
    output_path: str | None = Field(
        default=None, description="证据在工具输出中的点分字段路径，整个输出则为 null"
    )
    excerpt: str = Field(min_length=1)


class EvidenceDraft(BaseModel):
    """结构化输出：取证草稿，仅声明引用关系，不构建 provenance。"""

    summary: str = Field(min_length=1)
    claims: list[DraftClaim] = Field(min_length=1)


class ToolCallRecord(BaseModel):
    """一次真实工具调用的机械记录。"""

    call_index: int
    tool_name: str
    arguments: dict[str, Any]
    content: str


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def extract_tool_call_records(messages: list[Any]) -> list[ToolCallRecord]:
    """按 AIMessage.tool_calls 与 ToolMessage 的 tool_call_id 配对，还原真实调用序列。"""
    pending: dict[str, tuple[str, dict[str, Any]]] = {}
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                pending[call["id"]] = (call["name"], dict(call.get("args") or {}))
        elif isinstance(message, ToolMessage):
            entry = pending.pop(message.tool_call_id, None)
            if entry is None:
                continue
            tool_name, arguments = entry
            records.append(
                ToolCallRecord(
                    call_index=len(records),
                    tool_name=tool_name,
                    arguments=arguments,
                    content=_message_text(message.content),
                )
            )
    return records


def _user_prompt(directive: Directive, context: Any, hypotheses: list[dict[str, Any]]) -> str:
    context_json = (
        context.model_dump_json(indent=2) if hasattr(context, "model_dump_json") else context
    )
    now_ms = int(utc_now().timestamp() * 1000)
    lines = [
        f"取证目标：{directive.objective}",
        f"关注假设：{directive.hypothesis_ids or '（无指定，收集与诊断相关的事实）'}",
        f"当前时间戳（毫秒）：{now_ms}",
        f"诊断上下文：\n{context_json}",
    ]
    if hypotheses:
        lines.append("候选假设：")
        for item in hypotheses:
            lines.append(
                f"- {item.get('id')}: {item.get('statement')} "
                f"（预期支持: {item.get('expected_support') or '无'}；"
                f"预期反驳: {item.get('expected_refuting') or '无'}）"
            )
    return "\n".join(lines)


def _validate_draft(
    draft: EvidenceDraft,
    record_count: int,
    directive: Directive,
) -> str | None:
    """返回校验错误描述，None 表示通过。只校验机械完整性：
    call_index 指向真实调用、excerpt 非空。假设 ID 归属由清洗兜底。"""
    if not draft.claims:
        return "claims 不能为空"
    for index, claim in enumerate(draft.claims):
        if claim.call_index >= record_count:
            return f"claims[{index}].call_index={claim.call_index} 超出实际工具调用数 {record_count}"
        if not claim.excerpt.strip():
            return f"claims[{index}].excerpt 不能为空"
    return None


def _build_evidence_card(
    draft: EvidenceDraft,
    records: list[ToolCallRecord],
    directive: Directive,
    domain: str,
    round_number: int,
) -> EvidenceCard:
    claims: list[EvidenceClaim] = []
    for offset, draft_claim in enumerate(draft.claims):
        record = records[draft_claim.call_index]
        claims.append(
            EvidenceClaim(
                claim_id=f"ev-{directive.id}-{offset + 1}",
                statement=draft_claim.statement,
                confidence=draft_claim.confidence,
                polarity=draft_claim.polarity,
                hypothesis_ids=draft_claim.hypothesis_ids,
                provenance=ClaimProvenance(
                    tool_name=record.tool_name,
                    args_digest=args_digest(record.tool_name, record.arguments, record.content),
                    output_path=draft_claim.output_path,
                    excerpt=draft_claim.excerpt.strip()[:2000],
                ),
            )
        )
    return EvidenceCard(
        card_id=f"card-{directive.id}-r{round_number}",
        domain=domain,
        directive_id=directive.id,
        round=round_number,
        claims=claims,
        summary=draft.summary.strip(),
    )


async def run_investigation(
    directive: Directive,
    context: Any,
    *,
    domain: str,
    system_prompt: str,
    hypotheses: list[dict[str, Any]] | None = None,
    round_number: int = 0,
) -> EvidenceCard:
    """执行单个取证域的 ReAct 子图，产出带机械 provenance 的 EvidenceCard。"""
    logger.info(f"=== Investigator[{domain}]：执行指令 {directive.id} ===")
    started_at = time.monotonic()
    registry = await get_domain_registry(domain)
    model_name = config.aiops_investigator_model or config.rag_model
    llm = LLMFactory.create_qwen_chat_model(
        model=model_name,
        temperature=0,
        streaming=True,
        timeout=config.aiops_investigator_timeout,
        max_retries=config.aiops_investigator_llm_retries,
        stall_seconds=config.aiops_investigator_stall_seconds,
    )
    agent = create_react_agent(
        llm,
        tools=list(registry.handlers.values()),
        prompt=system_prompt,
    )
    messages = [("user", _user_prompt(directive, context, hypotheses or []))]
    # 知识检索 trace 以 directive_id 为键：Send 并行取证共享 thread_id，
    # 直接用会话 ID 会导致并行分支互相覆盖检索记录。
    with capture_retrieval_for_session(f"aiops-investigation:{directive.id}"):
        agent_result = await agent.ainvoke(
            {"messages": messages},
            config={"recursion_limit": directive.max_iterations * 2 + 6},
        )
    records = extract_tool_call_records(agent_result["messages"])
    if not records:
        raise RuntimeError(f"取证指令 {directive.id} 未产生任何工具调用")

    draft_chain = llm.with_structured_output(EvidenceDraft)
    draft_messages = [
        (
            "user",
            "工具调用记录：\n"
            + "\n".join(
                f"[{record.call_index}] tool={record.tool_name} args={record.arguments}\n"
                f"output（截断）: {record.content[:3000]}"
                for record in records
            )
            + "\n\n"
            + DRAFT_PROMPT,
        )
    ]
    # 与 Planner 相同的一次校验重试：结构化输出偶发引用越界时把
    # 明确错误反馈给模型重新声明，避免可修正错误终止整轮诊断。
    draft: EvidenceDraft | None = None
    last_error: str | None = None
    for _attempt in range(2):
        raw = await draft_chain.ainvoke(draft_messages)
        candidate = raw if isinstance(raw, EvidenceDraft) else EvidenceDraft.model_validate(raw)
        last_error = _validate_draft(candidate, len(records), directive)
        if last_error is None:
            draft = candidate
            break
        logger.warning("证据草稿校验失败，反馈错误后重试: {}", last_error)
        draft_messages.append(
            (
                "user",
                f"上一次草稿未通过校验：{last_error}\n"
                f"call_index 只能取 0 到 {len(records) - 1}，请完整重新输出。",
            )
        )
    if draft is None:
        raise ValueError(f"证据草稿校验失败: {last_error}")

    # 假设 ID 归属清洗：模型可能编造候选列表之外的 ID（重试也纠正不了），
    # 剔除未知 ID 保留证据卡，而不是把整卡作废——归属缺失可由评审兜底，
    # 取证产出的丢失不可恢复。
    allowed_hypothesis_ids = set(directive.hypothesis_ids) | {
        item.get("id") for item in hypotheses or [] if item.get("id")
    }
    for offset, claim in enumerate(draft.claims):
        unknown = [i for i in claim.hypothesis_ids if i not in allowed_hypothesis_ids]
        if unknown:
            logger.warning(
                "草稿 claims[{}] 引用了未知假设 {}，已剔除: {}",
                offset,
                unknown,
                sorted(allowed_hypothesis_ids),
            )
            draft.claims[offset].hypothesis_ids = [
                i for i in claim.hypothesis_ids if i in allowed_hypothesis_ids
            ]

    card = _build_evidence_card(draft, records, directive, domain, round_number)
    logger.info(
        f"Investigator[{domain}] 完成：{len(card.claims)} 条证据，"
        f"调用 {len(records)} 次工具，耗时 {time.monotonic() - started_at:.0f}s"
    )
    return card
