"""星型编排顶层状态。

子图（取证 Agent）不共享任何顶层键：指令经 Send 任务下发，
EvidenceCard 经 reducer 合并回顶层，一切消息由 Supervisor 中转。
"""

import operator
from typing import Annotated, TypedDict

from app.agent.aiops.diagnosis_models import (
    AdjudicationDecision,
    BudgetLedger,
    DiagnosisContext,
    Directive,
    EvidenceCard,
    Hypothesis,
    SupervisorDecision,
)


class InvestigateTask(TypedDict):
    """Send 下发的单条取证任务（Supervisor 组装，域 Agent 只看到它）。"""

    directive: dict
    hypotheses: list[dict]
    round: int
    session_id: str
    context: DiagnosisContext
    budget: dict


class OrchestratorState(TypedDict, total=False):
    input: str
    session_id: str
    context: DiagnosisContext
    hypotheses: list[Hypothesis]
    # append-only：所有历史指令与已派发标记，供确定性路由判断
    directives: Annotated[list[Directive], operator.add]
    dispatched: Annotated[list[str], operator.add]
    evidence: Annotated[list[EvidenceCard], operator.add]
    investigation_errors: Annotated[list[str], operator.add]
    adjudications: Annotated[list[AdjudicationDecision], operator.add]
    # Adjudicator 写、Supervisor 读后清空（星型中转的单向信箱）
    pending_decision: AdjudicationDecision | None
    decision: SupervisorDecision
    adjudicated_evidence_count: int
    converged_hypothesis_id: str | None
    report_violations: list[str]
    response: str
    budget: BudgetLedger
