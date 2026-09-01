"""Reporter 节点：基于 Supervisor 给的 claim 白名单生成最终报告。"""

import json
import re
from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from loguru import logger

from app.agent.aiops.diagnosis_models import EvidenceCard, Hypothesis
from app.config import config
from app.core.llm_factory import LLMFactory

from .state import OrchestratorState

CLAIM_REF_PATTERN = re.compile(r"\[ev-[a-zA-Z0-9_-]+\]")


def _chunk_text(chunk: Any) -> str:
    """Extract user-visible text from a LangChain message chunk."""
    content_blocks = getattr(chunk, "content_blocks", None)
    if isinstance(content_blocks, list):
        return "".join(
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )

    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


reporter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是运维诊断 Reporter。依据白名单证据与假设裁决生成最终 Markdown 报告。

                报告结构：
                # 诊断报告
                ## 结论（根因或"无法确认"，并说明置信度）
                ## 关键证据（每条标注来源 claim_id，格式如 [ev-xxx-1]）
                ## 已排除的假设（说明排除依据）
                ## 处置建议

                规则：
                - 只允许引用白名单中的 claim_id，格式 [ev-...]；不得编造其他引用。
                - 未在证据中出现的数值一律不得写入报告。
                - 证据不足时如实写"无法确认"，并列出还缺什么证据。
                - 不要输出 JSON，只输出 Markdown。
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


def strip_unresolved_claims(report: str, whitelist: set[str]) -> tuple[str, list[str]]:
    """剥离白名单之外的 [ev-...] 引用并记录违规（防幻觉引用）。"""
    violations: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        claim_id = match.group(0)[1:-1]
        if claim_id in whitelist:
            return match.group(0)
        violations.append(claim_id)
        return ""

    return CLAIM_REF_PATTERN.sub(_replace, report), violations


async def reporter(state: OrchestratorState) -> dict[str, Any]:
    logger.info("=== Reporter：生成最终诊断报告 ===")
    evidence = state.get("evidence", [])
    whitelist = {claim.claim_id for card in evidence for claim in card.claims}
    converged_hypothesis_id = state.get("converged_hypothesis_id")
    try:
        report = await _generate_report(state, evidence, whitelist, converged_hypothesis_id)
    except Exception as exc:
        logger.exception("报告生成失败: {}", exc)
        report = _fallback_report(state, evidence, converged_hypothesis_id)

    report, violations = strip_unresolved_claims(report, whitelist)
    if violations:
        logger.warning(f"剥离了 {len(violations)} 个未定义 claim 引用: {violations}")
    if not report.strip():
        report = _fallback_report(state, evidence, converged_hypothesis_id)
    return {"response": report, "report_violations": violations}


def _user_message(
    state: OrchestratorState,
    evidence: list[EvidenceCard],
    whitelist: set[str],
    converged_hypothesis_id: str | None,
) -> str:
    return (
        "原始任务："
        + state.get("input", "")
        + "\n\nclaim 白名单（只可引用这些）：\n"
        + json.dumps(sorted(whitelist), ensure_ascii=False)
        + "\n\n证据卡：\n"
        + json.dumps(
            [card.model_dump(mode="json") for card in evidence],
            ensure_ascii=False,
            indent=2,
            default=str,
        )[:30000]
        + "\n\n假设裁决：\n"
        + json.dumps(
            {
                "hypotheses": [
                    h.model_dump(mode="json") for h in state.get("hypotheses", [])
                ],
                "converged_hypothesis_id": converged_hypothesis_id,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n请生成最终诊断报告。"
    )


async def _generate_report(
    state: OrchestratorState,
    evidence: list[EvidenceCard],
    whitelist: set[str],
    converged_hypothesis_id: str | None,
) -> str:
    llm = LLMFactory.create_qwen_chat_model(
        model=config.aiops_reporter_model or config.rag_model,
        temperature=0,
        streaming=True,
    )
    messages = [
        (
            "user",
            _user_message(state, evidence, whitelist, converged_hypothesis_id),
        )
    ]

    writer = None
    try:
        writer = get_stream_writer()
    except RuntimeError:
        pass

    parts: list[str] = []
    if writer is not None:
        # 流式输出与 legacy replanner 相同：Markdown chunk 走 custom
        # stream 逐段转发给前端（report_chunk），完整文本进 state。
        async for chunk in (reporter_prompt | llm).astream({"messages": messages}):
            text = _chunk_text(chunk)
            if not text:
                continue
            parts.append(text)
            writer({"type": "report_chunk", "stage": "final_report", "data": text})
    else:
        response = await (reporter_prompt | llm).ainvoke({"messages": messages})
        parts.append(_chunk_text(response))

    report = "".join(parts)
    if not report.strip():
        raise ValueError("流式报告为空")
    return report


def _fallback_report(
    state: OrchestratorState,
    evidence: list[EvidenceCard],
    converged_hypothesis_id: str | None,
) -> str:
    lines = ["# 诊断报告", "", f"目标服务：{state['context'].service_name}", ""]
    if converged_hypothesis_id:
        for hypothesis in state.get("hypotheses", []):
            if hypothesis.id == converged_hypothesis_id:
                lines.append(f"## 结论\n\n{hypothesis.statement}（假设 {hypothesis.id}）")
                break
    else:
        lines.append("## 结论\n\n无法确认根因：证据不足或诊断预算耗尽。")
    if evidence:
        lines.append("\n## 关键证据\n")
        for card in evidence:
            for claim in card.claims:
                lines.append(f"- [{claim.claim_id}] {claim.statement}")
    ruled_out = [
        h for h in state.get("hypotheses", []) if h.status == "ruled_out"
    ]
    if ruled_out:
        lines.append("\n## 已排除的假设\n")
        for hypothesis in ruled_out:
            lines.append(f"- {hypothesis.id}: {hypothesis.statement}")
    return "\n".join(lines)
