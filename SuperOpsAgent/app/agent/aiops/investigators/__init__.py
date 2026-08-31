"""AIOps 取证 Agent 域：按域隔离工具边界，产出 EvidenceCard。"""

from collections.abc import Awaitable, Callable
from typing import Any

from ..diagnosis_models import Directive, EvidenceCard
from ..models import DiagnosisContext

_DOMAIN_MODULES = {"metrics": "metrics", "logs": "logs", "knowledge": "knowledge"}


def get_investigator(domain: str) -> Callable[..., Awaitable[EvidenceCard]]:
    if domain not in _DOMAIN_MODULES:
        raise ValueError(f"未知取证域: {domain}，可用域: {sorted(_DOMAIN_MODULES)}")
    from . import knowledge, logs, metrics

    return {"metrics": metrics, "logs": logs, "knowledge": knowledge}[domain].investigate


async def run_domain_investigation(
    domain: str,
    directive: Directive,
    context: DiagnosisContext,
    *,
    hypotheses: list[dict[str, Any]] | None = None,
    round_number: int = 0,
) -> EvidenceCard:
    """按域派发取证任务的统一入口（P4 Supervisor 调用）。"""
    return await get_investigator(domain)(
        directive, context, hypotheses=hypotheses, round_number=round_number
    )
