import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory


class QueryRewriteService:
    def __init__(self) -> None:
        self.model_name = config.rag_query_rewrite_model or config.rag_model
        self.model = LLMFactory.create_qwen_chat_model(
            model=self.model_name,
            temperature=0,
            streaming=False,
        )

    def _get_recent_history(self, session_id: str) -> list[dict[str, str]]:
        try:
            from app.services.rag_agent_service import rag_agent_service

            checkpoint_tuple = rag_agent_service.checkpointer.get(
                {"configurable": {"thread_id": session_id}}
            )
            if not checkpoint_tuple:
                return []

            if hasattr(checkpoint_tuple, "checkpoint"):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore[attr-defined]
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            history: list[dict[str, str]] = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    continue

                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue
                content = message.content if hasattr(message, "content") else str(message)
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    ).strip()
                else:
                    content = str(content).strip()

                if not content:
                    continue

                history.append({"role": role, "content": content})

            rounds = config.rag_query_rewrite_history_rounds
            if rounds <= 0:
                return []
            return history[-(rounds * 2) :]
        except Exception as e:
            logger.warning(f"查询重写读取会话历史失败: session_id={session_id}, error={e}")
            return []

    def _build_rewrite_prompt(self, query: str, history: list[dict[str, str]]) -> str:
        history_text = "\n".join(f"[{item['role']}] {item['content']}" for item in history)
        if not history_text:
            history_text = "(无历史对话)"

        return (
            "你在做检索查询重写。\n"
            "请根据最近对话补全代词、省略和关键实体，让当前问题改写成更适合知识库检索的一行查询。\n"
            "保留错误码、英文术语、服务名、组件名、数字、时间范围。\n"
            "不要回答问题。\n"
            "不要编造上下文中不存在的信息。\n"
            "只输出一行纯文本 query。\n\n"
            f"最近对话:\n{history_text}\n\n"
            f"当前问题: {query}\n"
            "输出:"
        )

    async def rewrite(self, query: str, session_id: str | None) -> str:
        original_query = query
        if not query.strip():
            return original_query
        if not config.rag_query_rewrite_enabled:
            return original_query
        if not session_id:
            return original_query

        history = self._get_recent_history(session_id)
        prompt = self._build_rewrite_prompt(query, history)

        try:
            response = await asyncio.wait_for(
                self.model.ainvoke(prompt),
                timeout=config.rag_query_rewrite_timeout,
            )
            content = response.content if hasattr(response, "content") else response
            if isinstance(content, list):
                rewritten_query = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                rewritten_query = str(content).strip()

            fallback = (
                not rewritten_query or len(rewritten_query) > config.rag_query_rewrite_max_length
            )
            final_query = original_query if fallback else rewritten_query
            if fallback:
                logger.warning(
                    f"查询重写回退: original_query={original_query!r}, final_query={final_query!r}, "
                    f"rewrite_fallback=True, history_count={len(history)}"
                )
            else:
                logger.info(
                    f"查询重写成功: original_query={original_query!r}, final_query={final_query!r}, "
                    f"rewrite_fallback=False, history_count={len(history)}"
                )
            return final_query
        except Exception as e:
            logger.warning(
                f"查询重写失败并回退: original_query={original_query!r}, final_query={original_query!r}, "
                f"rewrite_fallback=True, history_count={len(history)}, error={e}"
            )
            return original_query

    def rewrite_sync(self, query: str, session_id: str | None) -> str:
        return asyncio.run(self.rewrite(query, session_id))


query_rewrite_service = QueryRewriteService()
