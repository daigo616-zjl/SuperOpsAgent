"""Planner 节点：生成可校验、可直接执行的结构化诊断计划。"""

from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory
from app.tools import retrieve_knowledge

from .models import DiagnosticPlan
from .state import PlanExecuteState
from .tool_registry import get_tool_registry

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是运维诊断 Planner，只负责生成结构化计划，不执行工具。

                诊断上下文：
                {diagnosis_context}

                可用工具注册表（名称、用途、参数 JSON Schema）：
                {tool_registry}

                {experience_context}

                规划规则：
                - 每个步骤只能调用一次工具；需要多个工具时拆成多个步骤。
                - tool_name 必须来自工具注册表，arguments 必须符合对应 input_schema。
                - 服务名统一引用上下文，参数值写成
                  {{"source":"context","path":"service_name"}}，不要写死服务名。
                - 后续步骤可引用前置步骤输出，格式为
                  {{"source":"step","step_id":"步骤ID","path":"输出字段路径"}}。
                  如果前置输出本身就是所需标量可省略 path；数值可用 offset 做加法偏移，
                  例如当前毫秒时间戳减 15 分钟设置 offset=-900000。
                - 只能引用工具真实输出中存在的字段，不得猜测或虚构字段。
                  纯字符串输出必须省略 path，且不能对它使用数值 offset。
                - depends_on 使用步骤 id，且只能依赖之前声明的步骤。
                - id 使用英文字母开头，只包含英文字母、数字、下划线或连字符。
                - success_criteria 必须是 Executor 可机械判断的条件，不能写自然语言判断。
                - success_criteria.path 是工具输出中的点分字段路径；整个输出可用 null。
                - 不要加入“综合分析”或“生成报告”等无工具步骤，最终分析由 Replanner 完成。
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> dict[str, Any]:
    logger.info("=== Planner：制定结构化执行计划 ===")
    input_text = state.get("input", "")
    context = state["context"]

    try:
        experience_docs = ""
        try:
            result = await retrieve_knowledge.ainvoke({"query": input_text})
            if isinstance(result, str) and result.strip():
                experience_docs = result
        except Exception as exc:
            logger.warning(f"查询内部经验失败: {exc}")

        registry = await get_tool_registry()
        llm = LLMFactory.create_qwen_chat_model(model=config.rag_model, temperature=0)
        chain = planner_prompt | llm.with_structured_output(DiagnosticPlan)
        messages = [("user", input_text)]
        chain_input = {
            "messages": messages,
            "diagnosis_context": context.model_dump_json(indent=2),
            "tool_registry": registry.prompt_description(),
            "experience_context": (
                f"相关运维经验：\n{experience_docs}" if experience_docs else "无相关经验文档。"
            ),
        }

        # 结构化输出偶尔会漏写 depends_on 或返回错误字段类型。把明确的
        # Pydantic/注册表校验结果反馈给模型重试一次，避免一次可修正的计划
        # 错误直接结束整个诊断流程。
        for attempt in range(2):
            try:
                plan = await chain.ainvoke(chain_input)
                if isinstance(plan, dict):
                    plan = DiagnosticPlan.model_validate(plan)
                if not isinstance(plan, DiagnosticPlan):
                    raise ValueError("Planner 未返回有效的 DiagnosticPlan")
                registry.validate_plan(plan)
                logger.info(f"结构化计划已生成，共 {len(plan.steps)} 个步骤")
                return {"plan": plan}
            except Exception as planning_exc:
                if attempt == 1:
                    raise
                logger.warning("结构化计划校验失败，反馈错误后重试: {}", planning_exc)
                messages.append(
                    (
                        "user",
                        "上一次计划未通过结构校验，请修正后完整重新输出。"
                        f"校验错误：{planning_exc}\n"
                        "尤其检查 steps 必须为数组，且所有 step 输出引用都必须在 "
                        "depends_on 中声明对应步骤 ID。",
                    )
                )
    except Exception as exc:
        # Loguru 不能使用 logging 的 exc_info=True；异常文本里通常含有
        # JSON 花括号，误传该参数会触发二次 format 并掩盖原始错误。
        logger.exception("生成结构化计划失败: {}", exc)
        return {"response": f"诊断计划生成失败：{exc}"}
