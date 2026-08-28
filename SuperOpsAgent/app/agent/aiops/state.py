"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import TypedDict

from .models import DiagnosisContext, DiagnosticPlan, StepExecutionResult


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 状态"""

    # 用户输入（任务描述）
    input: str

    # 本次诊断统一上下文
    context: DiagnosisContext

    # 完整结构化执行计划；执行进度由 execution_results 推导
    plan: DiagnosticPlan | None

    # 已执行的步骤历史；节点返回完整列表，便于新请求显式清空 checkpoint 旧状态
    execution_results: list[StepExecutionResult]

    # 最终响应/报告
    response: str

    # 防止无限重新规划
    replan_count: int
