"""星型编排图：supervisor 是唯一中枢，其余节点出边一律指回 supervisor。"""

import asyncio
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from loguru import logger

from app.agent.aiops.diagnosis_models import (
    BudgetLedger,
    Directive,
    EvidenceCard,
)
from app.agent.aiops.investigators import run_domain_investigation
from app.services.evidence_repository import evidence_repository

from .adjudicator import adjudicator
from .hypothesizer import hypothesizer
from .reporter import reporter
from .state import OrchestratorState, InvestigateTask
from .supervisor import supervisor

NODE_SUPERVISOR = "supervisor"
NODE_HYPOTHESIZER = "hypothesizer"
NODE_INVESTIGATE = "investigate"
NODE_ADJUDICATOR = "adjudicator"
NODE_REPORTER = "reporter"


async def investigate(task: InvestigateTask) -> dict[str, Any]:
    """取证节点：命令式调用域子图，入参 Directive、出参仅 EvidenceCard。"""
    directive = Directive.model_validate(task["directive"])
    budget = BudgetLedger.model_validate(task["budget"])
    logger.info(f"=== Investigate：{directive.id}（{directive.target_domain}）===")
    try:
        remaining = budget.max_wall_seconds - budget.elapsed_seconds()
        card: EvidenceCard = await asyncio.wait_for(
            run_domain_investigation(
                directive.target_domain,
                directive,
                task["context"],
                hypotheses=task["hypotheses"],
                round_number=task["round"],
            ),
            timeout=max(5.0, remaining),
        )
    except Exception as exc:
        logger.exception(f"取证任务 {directive.id} 失败: {exc}")
        return {
            "dispatched": [directive.id],
            "investigation_errors": [f"{directive.id}: {exc}"],
        }

    try:
        evidence_repository.append_evidence_card(
            task["session_id"], card, directive=directive
        )
    except Exception as exc:
        # 证据持久化失败不阻断诊断，Evidence Store 是旁路而非依赖。
        logger.warning(f"证据卡 {card.card_id} 持久化失败: {exc}")

    return {"dispatched": [directive.id], "evidence": [card]}


def route_from_supervisor(state: OrchestratorState) -> Any:
    """supervisor 的唯一条件边：确定性决策 → 下一跳。"""
    decision = state["decision"]
    if decision.action == "hypothesize":
        return NODE_HYPOTHESIZER
    if decision.action == "dispatch":
        dispatched = set(state.get("dispatched", []))
        pending = [d for d in decision.directives if d.id not in dispatched]
        if not pending:
            return NODE_SUPERVISOR
        return [
            Send(
                NODE_INVESTIGATE,
                InvestigateTask(
                    directive=directive.model_dump(mode="json"),
                    hypotheses=[
                        h.model_dump(mode="json")
                        for h in state.get("hypotheses", [])
                    ],
                    round=state["budget"].round,
                    session_id=state.get("session_id", "default"),
                    context=state["context"],
                    budget=state["budget"].model_dump(mode="json"),
                ),
            )
            for directive in pending
        ]
    if decision.action == "adjudicate":
        return NODE_ADJUDICATOR
    return NODE_REPORTER


def build_orchestrator_graph(
    checkpointer: BaseCheckpointSaver | None = None,
):
    """编译星型拓扑图。铁律：任意两个非 supervisor 节点之间没有边。"""
    workflow = StateGraph(OrchestratorState)
    workflow.add_node(NODE_SUPERVISOR, supervisor)
    workflow.add_node(NODE_HYPOTHESIZER, hypothesizer)
    workflow.add_node(NODE_INVESTIGATE, investigate)
    workflow.add_node(NODE_ADJUDICATOR, adjudicator)
    workflow.add_node(NODE_REPORTER, reporter)

    workflow.set_entry_point(NODE_SUPERVISOR)
    workflow.add_edge(NODE_HYPOTHESIZER, NODE_SUPERVISOR)
    workflow.add_edge(NODE_INVESTIGATE, NODE_SUPERVISOR)
    workflow.add_edge(NODE_ADJUDICATOR, NODE_SUPERVISOR)
    workflow.add_conditional_edges(
        NODE_SUPERVISOR,
        route_from_supervisor,
        {
            NODE_HYPOTHESIZER: NODE_HYPOTHESIZER,
            NODE_ADJUDICATOR: NODE_ADJUDICATOR,
            NODE_REPORTER: NODE_REPORTER,
            NODE_SUPERVISOR: NODE_SUPERVISOR,
        },
    )
    workflow.add_edge(NODE_REPORTER, END)

    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("星型编排图构建完成（supervisor 中心化路由）")
    return graph
