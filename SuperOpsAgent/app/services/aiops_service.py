"""
AIOps 诊断服务：星型多 Agent 编排（假设驱动鉴别诊断）

supervisor 中心化确定性路由 + 假设驱动鉴别诊断。
"""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.agent.aiops.diagnosis_models import (
    BudgetLedger,
    DiagnosisContext,
    SupervisorDecision,
)
from app.agent.aiops.orchestrator.graph import (
    NODE_ADJUDICATOR,
    NODE_HYPOTHESIZER,
    NODE_INVESTIGATE,
    NODE_REPORTER,
    NODE_SUPERVISOR,
    build_orchestrator_graph,
)
from app.agent.aiops.orchestrator.state import OrchestratorState
from app.config import config
from app.services.evidence_repository import evidence_repository


def _scenario_id() -> str:
    try:
        from mcp_servers.scenario_loader import get_active_scenario_name

        return get_active_scenario_name()
    except ImportError:
        return os.environ.get("MOCK_SCENARIO", "no-fault")


class _AsyncPostgresSaverAdapter(BaseCheckpointSaver):
    """把同步 PostgresSaver 桥接成异步接口。

    Windows 上 uvicorn 运行在 ProactorEventLoop，psycopg async 模式不支持；
    LangGraph 的 AsyncPregelLoop 又只调用 async 接口（不会自动回退到同步方法），
    因此用 to_thread 把同步实现包成 async。
    """

    def __init__(self, sync_saver: PostgresSaver) -> None:
        super().__init__()
        self._sync = sync_saver

    async def aget_tuple(self, config: dict[str, Any]) -> Any:
        return await asyncio.to_thread(self._sync.get_tuple, config)

    async def alist(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Any:
        for item in self._sync.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Any,
        metadata: dict[str, Any],
        new_versions: Any,
    ) -> Any:
        return await asyncio.to_thread(
            self._sync.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: dict[str, Any],
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(
            self._sync.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self._sync.delete_thread, thread_id)


class AIOpsService:
    """AIOps 诊断服务"""

    def __init__(self, checkpointer: BaseCheckpointSaver | None = None):
        """初始化服务

        checkpointer 默认惰性创建持久化的 PostgresSaver（编排状态
        checkpoint 进 PostgreSQL，进程重启后可按 thread_id 追溯/恢复）；
        测试或无 DB 环境可注入自定义实现（如 MemorySaver）。
        """
        self._injected_checkpointer = checkpointer
        self._multiagent_graph: Any | None = None
        self._graph_lock = asyncio.Lock()
        self._pool: ConnectionPool | None = None
        logger.info("AIOps Service 初始化完成（星型多 Agent 编排）")

    async def _ensure_graph(self) -> Any:
        if self._multiagent_graph is None:
            async with self._graph_lock:
                if self._multiagent_graph is None:
                    if self._injected_checkpointer is not None:
                        checkpointer = self._injected_checkpointer
                    else:
                        checkpointer = self._build_postgres_checkpointer()
                    self._multiagent_graph = build_orchestrator_graph(
                        checkpointer=checkpointer
                    )
        return self._multiagent_graph

    def _build_postgres_checkpointer(self) -> _AsyncPostgresSaverAdapter:
        dsn = config.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=config.database_pool_size,
            open=True,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        try:
            sync_saver = PostgresSaver(pool)
            # 幂等：首次启动创建 checkpoints/checkpoint_writes/checkpoint_blobs 表
            sync_saver.setup()
            saver = _AsyncPostgresSaverAdapter(sync_saver)
        except Exception:
            pool.close()
            raise
        self._pool = pool
        logger.info("AIOps 编排状态 checkpoint 已接入 PostgreSQL（持久化）")
        return saver

    async def aclose(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
        service_name: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行诊断流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID
            service_name: 目标服务；未提供时使用配置的默认值

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始多 Agent 诊断: {user_input}")

        diagnosis_context = DiagnosisContext(
            service_name=service_name or config.aiops_default_service_name
        )
        budget = BudgetLedger(
            max_rounds=config.aiops_max_rounds,
            max_invocations=config.aiops_max_invocations,
            max_wall_seconds=config.aiops_max_wall_seconds,
            min_dispatch_wall_seconds=config.aiops_min_dispatch_wall_seconds,
            investigation_wall_seconds=config.aiops_investigation_wall_seconds,
        )
        initial_state: OrchestratorState = {
            "input": user_input,
            "session_id": session_id,
            "context": diagnosis_context,
            "hypotheses": [],
            "directives": [],
            "dispatched": [],
            "evidence": [],
            "investigation_errors": [],
            "adjudications": [],
            "pending_decision": None,
            "decision": SupervisorDecision(action="hypothesize", reason="初始状态"),
            "adjudicated_evidence_count": 0,
            "converged_hypothesis_id": None,
            "report_violations": [],
            "response": "",
            "budget": budget,
        }

        self._start_session(session_id, diagnosis_context, budget)

        try:
            # 每次诊断一个独立 thread：编排状态全程 checkpoint 进 PostgreSQL，
            # thread_id 加请求级后缀避免并发串话
            thread_id = f"{session_id}:{uuid.uuid4().hex[:8]}"
            config_dict = {"configurable": {"thread_id": thread_id}}

            graph = await self._ensure_graph()
            report_streamed = False
            final_response = ""
            async for stream_mode, event in graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode=["updates", "custom"],
            ):
                if stream_mode == "custom":
                    if isinstance(event, dict) and event.get("type") == "report_chunk":
                        report_streamed = True
                        yield event
                    continue

                for node_name, node_output in event.items():
                    for sse_event in self._format_multiagent_event(
                        node_name, node_output
                    ):
                        if sse_event.get("type") == "report":
                            final_response = sse_event.get("report", "")
                            # chunk 已把完整报告发给客户端时跳过重复 report 事件，
                            # 最终文本由 complete 事件携带
                            if report_streamed:
                                continue
                        yield sse_event

            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response,
            }
            logger.info(f"[会话 {session_id}] 多 Agent 诊断完成")
        except Exception as e:
            logger.exception(f"[会话 {session_id}] 多 Agent 诊断失败: {e}")
            yield {
                "type": "error",
                "stage": "error",
                "message": f"智能运维诊断出错: {str(e)}",
            }
        finally:
            self._finish_session(session_id)

    def _start_session(
        self,
        session_id: str,
        context: DiagnosisContext,
        budget: BudgetLedger,
    ) -> None:
        try:
            evidence_repository.start_session(
                session_id,
                context.service_name,
                _scenario_id(),
                budget.model_dump(mode="json"),
            )
        except Exception as exc:
            logger.warning(f"诊断会话 {session_id} 登记失败（Evidence Store 旁路）: {exc}")

    def _finish_session(self, session_id: str) -> None:
        try:
            evidence_repository.finish_session(
                session_id, status="completed"
            )
        except Exception as exc:
            logger.warning(f"诊断会话 {session_id} 收尾失败（Evidence Store 旁路）: {exc}")

    def _format_multiagent_event(
        self, node_name: str, output: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """节点更新 → SSE 事件（契约映射见方案 P4 表格）"""
        if not isinstance(output, dict):
            return []

        if node_name == NODE_SUPERVISOR:
            decision: SupervisorDecision = output["decision"]
            budget = output.get("budget")
            round_number = budget.round if isinstance(budget, BudgetLedger) else "?"
            return [
                {
                    "type": "status",
                    "stage": "supervisor",
                    "message": f"第 {round_number} 轮：{decision.reason}",
                }
            ]

        if node_name == NODE_HYPOTHESIZER:
            hypotheses = output.get("hypotheses", [])
            if not hypotheses:
                errors = output.get("investigation_errors", [])
                message = "候选假设生成失败"
                if errors:
                    message += f": {errors[-1]}"
                return [{"type": "status", "stage": "hypothesizer", "message": message}]
            return [
                {
                    "type": "plan",
                    "stage": "hypotheses_created",
                    "message": f"已生成 {len(hypotheses)} 个候选假设，开始鉴别取证",
                    "plan": {
                        "hypotheses": [
                            h.model_dump(mode="json")
                            if hasattr(h, "model_dump")
                            else h
                            for h in hypotheses
                        ]
                    },
                }
            ]

        if node_name == NODE_INVESTIGATE:
            events: list[dict[str, Any]] = []
            for card in output.get("evidence", []):
                card_data = card.model_dump(mode="json") if hasattr(card, "model_dump") else card
                domain = card_data.get("domain", "")
                directive_id = card_data.get("directive_id", "")
                claim_count = len(card_data.get("claims", []))
                events.append(
                    {
                        "type": "step_complete",
                        "stage": "investigated",
                        "message": f"取证完成：{directive_id}（{domain}，{claim_count} 条证据）",
                        "current_step": directive_id,
                        "result": card_data,
                    }
                )
            for error in output.get("investigation_errors", []):
                events.append(
                    {
                        "type": "status",
                        "stage": "investigate_failed",
                        "message": f"取证任务失败：{error}",
                    }
                )
            return events

        if node_name == NODE_ADJUDICATOR:
            decision = output.get("pending_decision")
            if decision is None:
                return [{"type": "status", "stage": "adjudicator", "message": "评审完成"}]
            eliminations = len(decision.eliminations)
            new_directives = len(decision.new_directives)
            message = f"评审完成：淘汰 {eliminations} 个假设，新增 {new_directives} 个取证任务"
            if decision.converged:
                message = f"评审收敛：假设 {decision.converged_hypothesis_id} 获得支持"
            return [{"type": "status", "stage": "adjudicator", "message": message}]

        if node_name == NODE_REPORTER:
            response = output.get("response", "")
            return [
                {
                    "type": "report",
                    "stage": "final_report",
                    "message": "最终报告已生成",
                    "report": response,
                }
            ]

        return []

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


# 全局单例
aiops_service = AIOpsService()
