"""指标取证域：告警优先，曲线判别根因与症状。"""

from textwrap import dedent

from ..diagnosis_models import Directive
from ..models import DiagnosisContext
from .base import run_investigation

metrics_system_prompt = dedent("""
    你是指标取证 Agent，只负责通过监控工具收集证据，不做最终结论。

    工作方法：
    - 先调用 query_active_alerts 获取当前告警，区分 critical 与 warning。
    - 再用 query_cpu_metrics / query_memory_metrics 查询最近 30 分钟曲线，
      时间窗口以诊断上下文与当前时间戳推算。
    - 区分根因信号与连带症状：例如 GC 压力会导致 CPU 升高，但 CPU 高本身
      不是根因；内存持续爬升或锯齿状回落（OOM 重启）是强根因信号。
    - 每条证据都须指明它支持或反驳哪条假设；与假设无关的事实保持 neutral。
    - 工具输出中的数值必须原样引用，禁止推算或四舍五入。
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
        domain="metrics",
        system_prompt=metrics_system_prompt,
        hypotheses=hypotheses,
        round_number=round_number,
    )
