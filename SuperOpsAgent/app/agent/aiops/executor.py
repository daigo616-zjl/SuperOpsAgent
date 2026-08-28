"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现
"""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.core.llm_factory import LLMFactory
from app.tools import get_current_time, retrieve_knowledge

from .state import PlanExecuteState


async def executor(state: PlanExecuteState) -> dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        # 获取本地工具
        local_tools = [get_current_time, retrieve_knowledge]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 监控取数步骤必须使用确定的 MCP 工具，不能交给 LLM 自由选择，
        # 否则模型可能只查询主题元数据而不读取实际指标/日志。
        required_result = await _execute_required_monitor_tool(task, mcp_tools)
        if required_result is not None:
            return {
                "plan": plan[1:],
                "past_steps": [(task, required_result)],
            }

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 创建 LLM（绑定工具）
        llm = LLMFactory.create_qwen_chat_model(model=config.rag_model, temperature=0)
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(
                content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""
            ),
            HumanMessage(content=f"请执行以下任务: {task}"),
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = (
                final_response.content
                if hasattr(final_response, "content")
                else str(final_response)
            )
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }


async def _execute_required_monitor_tool(task: str, mcp_tools: list) -> str | None:
    """根据计划步骤强制调用 MCP 监控工具，并返回原始结果。"""
    tools = {getattr(tool, "name", ""): tool for tool in mcp_tools}

    async def invoke(name: str, arguments: dict[str, Any]) -> Any:
        tool = tools.get(name)
        if tool is None:
            raise RuntimeError(f"MCP 工具不可用: {name}")
        logger.info(f"强制调用 MCP 工具: {name}, 参数={arguments}")
        return await tool.ainvoke(arguments)

    try:
        if "获取当前时间" in task or "时间基准" in task:
            result = await invoke("get_current_timestamp", {})
        elif "system-metrics" in task or "CPU" in task or "内存" in task:
            result = {
                "cpu": await invoke("query_cpu_metrics", {"service_name": "data-sync-service"}),
                "memory": await invoke("query_memory_metrics", {"service_name": "data-sync-service"}),
            }
        elif "application-logs" in task or "应用日志" in task or "日志证据" in task or "详细日志" in task:
            end_time_result = await invoke("get_current_timestamp", {})
            end_time = _extract_timestamp(end_time_result)
            if end_time is None:
                raise RuntimeError(f"无法解析 MCP 时间戳: {end_time_result!r}")
            topic_result = await invoke("get_topic_info_by_name", {"topic_name": "数据同步服务日志"})
            topic = _unwrap_tool_result(topic_result)
            topic_id = topic.get("topic_id") if isinstance(topic, dict) else "topic-001"
            start_time = end_time - 15 * 60 * 1000
            result = await invoke(
                "search_log",
                {"topic_id": topic_id, "start_time": start_time, "end_time": end_time, "limit": 100},
            )
        else:
            return None
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.error(f"强制调用监控 MCP 工具失败: {exc}")
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _unwrap_tool_result(value: Any) -> Any:
    """解包 LangChain MCP 结果中常见的 text/content/list 包装。"""
    if isinstance(value, list):
        if len(value) == 1:
            return _unwrap_tool_result(value[0])
        return [_unwrap_tool_result(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return _unwrap_tool_result(value["text"])
        if "content" in value and len(value) == 1:
            return _unwrap_tool_result(value["content"])
        return value
    if isinstance(value, str):
        try:
            return _unwrap_tool_result(json.loads(value))
        except json.JSONDecodeError:
            return value
    return value


def _extract_timestamp(value: Any) -> int | None:
    """从 MCP 工具返回值中提取毫秒时间戳。"""
    value = _unwrap_tool_result(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        for key in ("timestamp", "data", "result", "value"):
            if key in value:
                timestamp = _extract_timestamp(value[key])
                if timestamp is not None:
                    return timestamp
    if isinstance(value, str):
        match = re.search(r"\d{12,}", value)
        if match:
            return int(match.group(0))
    return None
