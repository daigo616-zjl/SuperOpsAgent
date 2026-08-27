"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Any

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

    # RAG 配置
    rag_top_k: int = 5
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考
    rag_temperature: float = 0.1
    rag_max_tokens: int = 1200
    rag_enable_thinking: bool = False
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

    # 离线评测配置
    eval_model: str = "qwen-max"
    eval_output_dir: str = "eval/reports"
    eval_metric_timeout: int = 90
    eval_faithfulness_timeout: int = 300
    eval_faithfulness_statement_batch_size: int = 10
    eval_answer_correctness_timeout: int = 240
    eval_metric_max_concurrency: int = 2
    eval_client_max_retries: int = 3

    # Elasticsearch 配置
    es_scheme: str = "http"
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "biz"
    es_timeout: int = 10
    es_analyzer: str = "standard"
    es_search_analyzer: str = "standard"
    es_required: bool = False

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:18003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:18004/mcp"

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
