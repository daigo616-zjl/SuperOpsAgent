"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

from collections.abc import AsyncGenerator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from app.agent.aiops import PlanExecuteState, executor, planner, replanner
from app.agent.aiops.models import DiagnosisContext, DiagnosticPlan, StepExecutionResult
from app.config import config

# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务"""
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)  # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)  # planner -> executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)  # executor -> replanner

        # replanner 的条件边
        def should_continue(state: PlanExecuteState) -> str:
            """判断是否继续执行"""
            # 如果已经生成了最终响应，结束
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            # 如果还有计划步骤，继续执行
            plan = state.get("plan")
            results = state.get("execution_results", [])
            if plan:
                completed_ids = {result.step_id for result in results}
                remaining = sum(step.id not in completed_ids for step in plan.steps)
                logger.info(f"继续执行，剩余 {remaining} 个步骤")
                return NODE_EXECUTOR

            # 计划为空但没有响应，返回 replanner 生成响应
            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER, should_continue, {NODE_EXECUTOR: NODE_EXECUTOR, END: END}
        )

        # 编译工作流
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
        service_name: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID
            service_name: 目标服务；未提供时使用配置的默认值

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 初始化状态
            diagnosis_context = DiagnosisContext(
                service_name=service_name or config.aiops_default_service_name
            )
            initial_state: PlanExecuteState = {
                "input": user_input,
                "context": diagnosis_context,
                "plan": None,
                "execution_results": [],
                "response": "",
                "replan_count": 0,
            }

            # 流式执行工作流
            config_dict = {"configurable": {"thread_id": session_id}}

            async for event in self.graph.astream(
                input=initial_state, config=config_dict, stream_mode="updates"
            ):
                # 解析事件
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # 根据节点类型生成不同的事件
                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)

                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)

                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            # 安全地获取响应（处理 values 可能为 None 的情况）
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response,
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {"type": "error", "stage": "error", "message": f"任务执行出错: {str(e)}"}

    async def diagnose(
        self,
        session_id: str = "default",
        service_name: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID
            service_name: 目标服务；未提供时使用配置的默认值

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        # 使用固定的 AIOps 任务描述
        from textwrap import dedent

        aiops_task = dedent(
            """诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过"""
        )

        async for event in self.execute(
            aiops_task,
            session_id=session_id,
            service_name=service_name,
        ):
            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                # 将 response 包装为 diagnosis 格式
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {"status": "completed", "report": event.get("response", "")},
                }
            else:
                yield event

    def _format_planner_event(self, state: dict | None) -> dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {"type": "status", "stage": "planner", "message": "规划节点执行中"}

        plan = state.get("plan")
        plan_data = plan.model_dump(mode="json") if isinstance(plan, DiagnosticPlan) else None
        step_count = len(plan.steps) if isinstance(plan, DiagnosticPlan) else 0

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {step_count} 个步骤",
            "plan": plan_data,
        }

    def _format_executor_event(self, state: dict | None) -> dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {"type": "status", "stage": "executor", "message": "执行节点运行中"}

        execution_results = state.get("execution_results", [])
        plan = state.get("plan")

        if execution_results:
            last_result = execution_results[-1]
            if isinstance(last_result, StepExecutionResult):
                result_data = last_result.model_dump(mode="json")
                step_title = last_result.step_title
            else:
                result_data = last_result
                step_title = last_result.get("step_title", "未知步骤")
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成：{step_title}",
                "current_step": step_title,
                "result": result_data,
                "remaining_steps": (
                    max(0, len(plan.steps) - len(execution_results))
                    if isinstance(plan, DiagnosticPlan)
                    else None
                ),
            }
        else:
            return {"type": "status", "stage": "executor", "message": "开始执行步骤"}

    def _format_replanner_event(self, state: dict | None) -> dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {"type": "status", "stage": "replanner", "message": "评估节点运行中"}

        response = state.get("response", "")
        plan = state.get("plan")

        if response:
            # 已生成最终响应
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response,
            }
        else:
            # 重新规划
            return {
                "type": "status",
                "stage": "replanner",
                "message": "评估完成，继续执行或已更新剩余步骤",
                "plan": plan.model_dump(mode="json") if isinstance(plan, DiagnosticPlan) else None,
            }


# 全局单例
aiops_service = AIOpsService()
