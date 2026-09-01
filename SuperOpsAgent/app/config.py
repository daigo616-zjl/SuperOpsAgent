"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperOpsAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 18000

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus Lite 配置
    milvus_lite_path: str = "./data/milvus.db"
    milvus_lite_db_name: str = "default"
    milvus_timeout: int = 10000  # 毫秒
    # Milvus Lite is an embedded gRPC server. Avoid frequent idle pings being
    # rejected by the server with GOAWAY too_many_pings.
    milvus_grpc_keepalive_time_ms: int = 60000
    milvus_grpc_keepalive_timeout_ms: int = 20000
    milvus_grpc_keepalive_permit_without_calls: bool = False

    # RAG 配置
    rag_top_k: int = 5
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考
    rag_temperature: float = 0.1
    rag_max_tokens: int = 1200
    rag_enable_thinking: bool = False
    rag_context_summary_model: str = "qwen3.5-flash"
    rag_context_summary_trigger_messages: int = 12
    rag_context_summary_keep_messages: int = 6
    rag_recall_size: int = 20
    rag_rerank_enabled: bool = True
    rag_rerank_model: str = "BAAI/bge-reranker-base"
    rag_rerank_timeout: int = 10
    rag_rerank_warmup_enabled: bool = True
    rag_rerank_warmup_timeout: int = 120
    rag_rrf_k: int = 60
    rag_query_rewrite_enabled: bool = True
    rag_query_rewrite_model: str = ""
    rag_query_rewrite_history_rounds: int = 3
    rag_query_rewrite_timeout: int = 5
    rag_query_rewrite_max_length: int = 200

    # LLM 稳定性配置
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_max_concurrency: int = 8
    llm_min_interval: float = 0.0
    llm_circuit_failure_threshold: int = 3
    llm_circuit_recovery_timeout: float = 30.0
    llm_retry_backoff: float = 0.25
    llm_fallback_model: str = "qwen-turbo"

    # 离线评测配置
    eval_model: str = "qwen-max"
    eval_output_dir: str = "eval/reports"
    eval_metric_timeout: int = 90
    eval_faithfulness_timeout: int = 360
    eval_faithfulness_statement_batch_size: int = 10
    eval_answer_correctness_timeout: int = 300
    eval_metric_max_concurrency: int = 2
    eval_client_max_retries: int = 3
    eval_recall_k: int = 20
    eval_hit_k: int = 5

    # Elasticsearch 配置
    es_scheme: str = "http"
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "biz"
    es_timeout: int = 10
    es_analyzer: str = "standard"
    es_search_analyzer: str = "standard"
    es_required: bool = False

    # PostgreSQL 权威知识库
    database_url: str = "postgresql+psycopg://superops:superops@localhost:5432/superops"
    database_pool_size: int = 5
    database_max_overflow: int = 5
    database_pool_timeout: int = 10
    index_worker_poll_seconds: float = 1.0
    index_worker_lease_seconds: int = 300
    index_worker_max_attempts: int = 8
    index_repair_interval_seconds: int = 300
    index_repair_batch_size: int = 100

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:18003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:18004/mcp"

    # AIOps 诊断默认上下文
    aiops_default_service_name: str = "data-sync-service"

    # AIOps 取证域工具名单（能力边界 = 决策权边界）
    aiops_domain_tools: dict[str, list[str]] = {
        "metrics": ["query_cpu_metrics", "query_memory_metrics", "query_active_alerts"],
        "logs": [
            "search_topic_by_service_name",
            "get_topic_info_by_name",
            "get_current_timestamp",
            "search_log",
        ],
        "knowledge": ["retrieve_knowledge"],
    }
    aiops_investigator_model: str | None = None  # 默认回退到 rag_model

    # AIOps 多 Agent 编排预算与角色模型
    aiops_max_rounds: int = Field(default=6, ge=1)
    aiops_max_invocations: int = Field(default=60, ge=1)
    # 墙钟需覆盖：首轮 60s 调用超时×重试的挂死 spell（实测 60+60+60=180s）
    # 加一轮正常取证与评审；过低会让 Provider 延迟尖峰期直接零证据收敛
    aiops_max_wall_seconds: float = Field(default=300.0, gt=0)
    # 剩余墙钟低于该值时不再派发新取证任务（一次取证 + 草稿生成约需 60-120s）
    aiops_min_dispatch_wall_seconds: float = Field(default=90.0, gt=0)
    # 单个取证任务的墙钟上限：LangGraph Send 分支需全部返回才交还
    # supervisor，某个域卡住时该上限保证其余域的证据仍能进入评审
    aiops_investigation_wall_seconds: float = Field(default=150.0, gt=0)
    # 取证 LLM 单次调用总超时（流式模式的整体兜底上限）
    aiops_investigator_timeout: float = Field(default=60.0, gt=0)
    # 取证 LLM 流式调用块间空档上限：超过即判定挂死并中止本次调用。
    # 非流式调用曾实测 >85s 静默挂死；流式下正常模型块间间隔远小于该值，
    # 挂死能在 stall 窗口内被识别，而不必等满 60s 调用超时
    aiops_investigator_stall_seconds: float = Field(default=20.0, gt=0)
    # 取证 LLM 调用内重试次数：流式挂死检测后单次失败成本降到 ~stall 窗口
    # （20s + 退避），任务预算内可以承受，默认恢复为 1
    aiops_investigator_llm_retries: int = Field(default=1, ge=0)
    aiops_hypothesizer_model: str | None = None  # 默认回退到 rag_model
    aiops_adjudicator_model: str | None = None  # 默认回退到 rag_model
    aiops_reporter_model: str | None = None  # 默认回退到 rag_model

    @property
    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }


# 全局配置实例
config = Settings()
