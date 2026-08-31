"""知识取证域：检索运维手册与历史经验，输出可执行的判别标准。"""

from textwrap import dedent

from ..diagnosis_models import Directive
from ..models import DiagnosisContext
from .base import run_investigation

knowledge_system_prompt = dedent("""
    你是知识取证 Agent，只负责通过 retrieve_knowledge 检索内部运维知识库，
    不做最终结论。

    工作方法：
    - 用取证目标中的症状关键词（如 "GC 停顿"、"OOM"、"连接池耗尽"）构造
      检索查询，最多检索 2-3 次不同措辞。
    - 从命中的运维手册（runbook）中提取：该症状的典型根因列表、
      每个根因的判别性证据（应查什么指标/日志、阈值是多少）。
    - 每条证据都须指明它支持或反驳哪条假设；与假设无关的事实保持 neutral。
    - 引用手册内容时保持原文，并注明来源条目。
""").strip()


async def investigate(
    directive: Directive,
    context: DiagnosisContext,
    *,
    hypotheses: list[dict] | None = None,
    round_number: int = 0,
):
    return await run_investigation(
        directive,
        context,
        domain="knowledge",
        system_prompt=knowledge_system_prompt,
        hypotheses=hypotheses,
        round_number=round_number,
    )
