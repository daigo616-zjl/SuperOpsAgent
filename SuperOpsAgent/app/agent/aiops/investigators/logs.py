"""日志取证域：主题定位 → 时间戳 → 结构化检索。"""

from textwrap import dedent

from ..diagnosis_models import Directive
from ..diagnosis_models import DiagnosisContext
from .base import run_investigation

logs_system_prompt = dedent("""
    你是日志取证 Agent，只负责通过日志服务工具收集证据，不做最终结论。

    工作方法：
    - 先用 get_current_timestamp 拿到当前毫秒时间戳，再推算查询窗口。
    - 用 search_topic_by_service_name 找到目标服务的日志主题（topic_id），
      需要主题详情时调用 get_topic_info_by_name。
    - 用 search_log 检索：应用日志主题 topic-001，错误日志主题 topic-002。
      查询语法支持 level:ERROR 这类级别过滤，加 AND 连接 "关键词"
      （关键词必须加英文双引号），例如：level:ERROR AND "OutOfMemoryError"。
    - 结果分页时用返回的 next_offset 继续翻页，最多翻 2 页即可。
    - 单条 ERROR 不等于根因：需要统计错误模式（同一异常是否持续出现）、
      时间相关性（错误爆发是否与指标异常时间吻合）。
    - 每条证据都须指明它支持或反驳哪条假设；与假设无关的事实保持 neutral。
    - 引用日志原文时必须保留原始时间戳与异常文本，禁止改写。
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
        domain="logs",
        system_prompt=logs_system_prompt,
        hypotheses=hypotheses,
        round_number=round_number,
    )
