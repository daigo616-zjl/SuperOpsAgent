from __future__ import annotations

"""知识检索工具 - 从向量数据库中检索相关信息"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from loguru import logger

_CAPTURED_RETRIEVAL_DOCS: dict[str, list[Document]] = {}

from app.config import config
from app.services.hybrid_search_service import hybrid_search_service
from app.services.query_rewrite_service import query_rewrite_service
from app.services.rerank_service import rerank_service


def set_captured_retrieval_docs(session_id: str, docs: list[Document]) -> None:
    _CAPTURED_RETRIEVAL_DOCS[session_id] = docs


def pop_captured_retrieval_docs(session_id: str) -> list[Document]:
    return _CAPTURED_RETRIEVAL_DOCS.pop(session_id, [])


def clear_captured_retrieval_docs(session_id: str) -> None:
    _CAPTURED_RETRIEVAL_DOCS.pop(session_id, None)


@tool(response_format="content_and_artifact")
def retrieve_knowledge(
    query: str,
    runtime_config: RunnableConfig | None = None,
) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题
    
    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    
    Args:
        query: 用户的问题或查询
        
    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: original_query='{query}'")

        session_id = None
        if runtime_config:
            session_id = runtime_config.get("configurable", {}).get("thread_id")
        rewritten_query = query_rewrite_service.rewrite_sync(query, session_id)
        logger.info(
            f"知识检索开始: original_query='{query}', retrieval_query='{rewritten_query}'"
        )

        candidates = hybrid_search_service.search_sync(
            rewritten_query,
            top_k=config.rag_recall_size,
        )

        if not candidates:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        if config.rag_rerank_enabled:
            docs = rerank_service.rerank(
                rewritten_query,
                candidates,
                top_k=config.rag_top_k,
            )
        else:
            docs = candidates[: config.rag_top_k]

        if session_id:
            set_captured_retrieval_docs(session_id, docs)

        # 格式化文档为上下文
        context = format_docs(docs)
        
        logger.info(
            f"检索到 {len(docs)} 个相关文档: original_query='{query}', retrieval_query='{rewritten_query}'"
        )
        return context, docs
        
    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
