"""Supervisor 节点：确定性状态机路由 + 消息中转 + 预算闸口（不调 LLM）。"""

from typing import Any

from loguru import logger

from app.agent.aiops.diagnosis_models import (
    BudgetLedger,
    Directive,
    Hypothesis,
    SupervisorDecision,
)

from .state import OrchestratorState

DOMAIN_LABELS = {"metrics": "指标", "logs": "日志", "knowledge": "知识"}


def supervisor(state: OrchestratorState) -> dict[str, Any]:
    """推进一轮：应用评审信箱 → 预算检查 → 确定性路由决策。"""
    budget: BudgetLedger = state["budget"].model_copy(update={"round": state["budget"].round + 1})
    logger.info(
        f"=== Supervisor 第 {budget.round} 轮："
        f"假设 {len(state.get('hypotheses', []))}，证据 {len(state.get('evidence', []))} ==="
    )

    update: dict[str, Any] = {"budget": budget, "pending_decision": None}
    if pending := state.get("pending_decision"):
        update.update(_apply_adjudication(state, pending))

    hypotheses = update.get("hypotheses", state.get("hypotheses", []))
    converged_hypothesis_id = update.get(
        "converged_hypothesis_id", state.get("converged_hypothesis_id")
    )
    decision = _decide(state, update, hypotheses, converged_hypothesis_id)
    update["decision"] = decision
    if decision.action == "adjudicate":
        update["adjudicated_evidence_count"] = len(state.get("evidence", []))
    if decision.action == "dispatch":
        update["budget"] = budget.model_copy(
            update={
                "invocations": budget.invocations
                + sum(d.max_iterations for d in decision.directives)
                + 1  # 决策本身视为一次调用配额
            }
        )
    logger.info(f"Supervisor 决策={decision.action}，原因={decision.reason}")
    return update


def _apply_adjudication(
    state: OrchestratorState, decision: Any
) -> dict[str, Any]:
    """星型中转：AdjudicationDecision 只经 Supervisor 写回全局状态。"""
    update: dict[str, Any] = {
        "adjudications": [decision],
        "converged_hypothesis_id": decision.converged_hypothesis_id,
    }
    if decision.eliminations:
        eliminated = {item.hypothesis_id: item for item in decision.eliminations}
        hypotheses = []
        for hypothesis in state.get("hypotheses", []):
            item = eliminated.get(hypothesis.id)
            if item is not None:
                hypotheses.append(
                    hypothesis.model_copy(
                        update={
                            "status": "ruled_out",
                            "ruled_out_by": list(item.ruled_out_by),
                        }
                    )
                )
            else:
                hypotheses.append(hypothesis)
        update["hypotheses"] = hypotheses

    if decision.new_directives:
        existing = {d.id for d in state.get("directives", [])}
        directives = []
        for offset, directive in enumerate(decision.new_directives):
            if directive.id in existing:
                directive = directive.model_copy(update={"id": f"{directive.id}-b{offset}"})
            directives.append(directive)
        update["directives"] = directives
    return update


def _decide(
    state: OrchestratorState,
    update: dict[str, Any],
    hypotheses: list[Hypothesis],
    converged_hypothesis_id: str | None,
) -> SupervisorDecision:
    budget: BudgetLedger = update["budget"]
    evidence = state.get("evidence", [])
    directives = update.get("directives", state.get("directives", []))
    dispatched = set(state.get("dispatched", []))

    active = [h for h in hypotheses if h.status == "active"]
    if budget.remaining_rounds() <= 0 or budget.wall_exhausted():
        return SupervisorDecision(
            action="converge",
            converged_hypothesis_id=converged_hypothesis_id,
            reason="预算耗尽，按当前证据收敛输出",
        )
    if budget.remaining_invocations() <= 0:
        return SupervisorDecision(
            action="converge",
            converged_hypothesis_id=converged_hypothesis_id,
            reason="LLM 调用配额耗尽，按当前证据收敛输出",
        )
    if not hypotheses:
        if any(e.startswith("hypothesizer") for e in state.get("investigation_errors", [])):
            return SupervisorDecision(
                action="converge",
                reason="假设生成失败，无法继续诊断",
            )
        return SupervisorDecision(action="hypothesize", reason="尚无候选假设")
    if not active and converged_hypothesis_id is None:
        return SupervisorDecision(
            action="converge",
            reason="所有假设均被淘汰，输出排除性结论",
        )

    pending = [d for d in directives if d.id not in dispatched]
    if (
        pending or not state.get("directives")
    ) and budget.remaining_wall_seconds() < budget.min_dispatch_wall_seconds:
        return SupervisorDecision(
            action="converge",
            converged_hypothesis_id=converged_hypothesis_id,
            reason=(
                f"剩余墙钟 {budget.remaining_wall_seconds():.0f}s 不足以完成取证，"
                "按当前证据收敛输出"
            ),
        )
    if not pending and not state.get("directives"):
        pending = _default_directives(budget.round, active, state.get("input", ""))
        if pending:
            update["directives"] = list(state.get("directives", [])) + pending
            return SupervisorDecision(
                action="dispatch",
                directives=pending,
                reason=f"首轮派发 {len(pending)} 个域取证任务",
            )
    if pending:
        return SupervisorDecision(
            action="dispatch",
            directives=pending,
            reason=f"派发 {len(pending)} 个取证任务",
        )
    if len(evidence) > state.get("adjudicated_evidence_count", 0):
        return SupervisorDecision(
            action="adjudicate",
            reason=f"{len(evidence) - state.get('adjudicated_evidence_count', 0)} 张新证据卡待评审",
        )
    return SupervisorDecision(
        action="converge",
        converged_hypothesis_id=converged_hypothesis_id,
        reason="无新增证据与指令，收敛输出",
    )


def _default_directives(
    round_number: int, hypotheses: list[Hypothesis], user_input: str
) -> list[Directive]:
    """首轮由 Supervisor 生成三域默认取证任务（确定性，不调 LLM）。"""
    hypothesis_ids = [h.id for h in hypotheses]
    topic = user_input.strip()[:200] or "目标服务当前告警"
    return [
        Directive(
            id=f"r{round_number}-{domain}",
            target_domain=domain,
            objective=f"围绕症状「{topic}」在{DOMAIN_LABELS[domain]}域收集鉴别性证据",
            hypothesis_ids=hypothesis_ids,
        )
        for domain in ("metrics", "logs", "knowledge")
    ]
