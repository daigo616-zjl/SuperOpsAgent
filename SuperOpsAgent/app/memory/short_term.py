"""短期记忆：Redis 滑动窗口 + 滚动摘要

数据结构（key 前缀 rag:s:{sid}:）：
- msgs   LIST   轮次消息 [{"role","content","ts"}]，在 keep 与 window 之间振荡
- sum    STRING 滚动摘要（被压缩消息的增量合并）
- seq    INCR   轮次号（兼作每轮 checkpoint thread_id 后缀）
- lock   压缩互斥锁

Redis 不可用时熔断降级：冷却期内直接返回空数据（调用方回退 checkpoint 路径），
冷却期满后半开探测，成功即恢复。
"""

import json
import time
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from app.config import config

KEY_PREFIX = "rag:s:"
BREAKER_COOLDOWN_SECONDS = 30.0
# 压缩连续失败时的安全上限，防止列表无限增长（超限静默丢最旧消息）
MAX_LIST_LENGTH = 200


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SummaryGenerator:
    """滚动摘要生成器（增量合并：旧摘要 + 被压缩消息 -> 新摘要）"""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from app.core.llm_factory import LLMFactory

            self._model = LLMFactory.create_qwen_chat_model(
                model=config.memory_extract_model,
                temperature=0,
                streaming=False,
                max_tokens=config.memory_extract_max_tokens,
                enable_thinking=False,
            )
        return self._model

    @staticmethod
    def _prompt(old_summary: str, messages: list[dict[str, Any]]) -> str:
        old_summary_text = old_summary.strip() or "（无）"
        dialogue = "\n".join(
            f"[{item.get('role', 'user')}] {item.get('content', '')}" for item in messages
        )
        return (
            "你在维护一段对话的滚动摘要。请把旧摘要与新压缩的对话内容合并为一份新摘要，"
            "用于替代完整历史供后续对话参考。\n"
            "要求：\n"
            "1. 保留对后续对话有用的信息：用户目标、已确认的事实与决策、服务名/错误码/阈值等关键实体、"
            "用户偏好与约束。\n"
            "2. 丢弃寒暄、重复内容和已被新对话推翻的旧结论。\n"
            "3. 按主题分点，简洁准确，不编造对话中不存在的信息，总长度不超过 500 字。\n"
            "只输出摘要正文。\n\n"
            f"旧摘要:\n{old_summary_text}\n\n"
            f"新压缩的对话:\n{dialogue}\n\n"
            "新摘要:"
        )

    async def summarize(self, old_summary: str, messages: list[dict[str, Any]]) -> str:
        model = self._get_model()
        response = await model.ainvoke(self._prompt(old_summary, messages))
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)
        return text.strip()


class ShortTermMemory:
    """Redis 短期记忆，带实例级熔断（冷却 + 半开探测）"""

    def __init__(
        self,
        client: Any | None = None,
        summarizer: Any | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._client = client
        self._summarizer = summarizer or SummaryGenerator()
        self._enabled = config.memory_enabled if enabled is None else enabled
        self._open_until = 0.0  # 熔断截止时间戳；<= now 表示闭合（可用）

    @staticmethod
    def keys_for(session_id: str) -> dict[str, str]:
        p = f"{KEY_PREFIX}{session_id}"
        return {"msgs": f"{p}:msgs", "sum": f"{p}:sum", "seq": f"{p}:seq", "lock": f"{p}:lock"}

    @property
    def tripped(self) -> bool:
        return time.monotonic() < self._open_until

    def _trip(self, exc: Exception) -> None:
        self._open_until = time.monotonic() + BREAKER_COOLDOWN_SECONDS
        logger.warning(
            f"短期记忆 Redis 故障，熔断 {BREAKER_COOLDOWN_SECONDS:.0f}s 降级到 checkpoint 路径: {exc}"
        )

    def _reset(self) -> None:
        self._open_until = 0.0

    async def _ensure_client(self):
        if self._client is not None:
            return self._client
        from app.memory.redis_client import redis_client_manager

        if not redis_client_manager.connected:
            await redis_client_manager.connect()
        self._client = redis_client_manager.get_client()
        self._reset()
        return self._client

    async def available(self) -> bool:
        """是否可用；冷却期满时尝试半开重连探测。"""
        if not self._enabled:
            return False
        if self.tripped:
            return False
        try:
            await self._ensure_client()
            return True
        except Exception as e:
            self._trip(e)
            return False

    # ---- 基础操作 ----

    async def next_seq(self, session_id: str) -> int | None:
        keys = self.keys_for(session_id)
        try:
            client = await self._ensure_client()
            pipe = client.pipeline()
            pipe.incr(keys["seq"])
            pipe.expire(keys["seq"], config.memory_redis_ttl_seconds)
            result = await pipe.execute()
            return int(result[0])
        except Exception as e:
            self._trip(e)
            return None

    async def build_context(self, session_id: str) -> tuple[str, list[dict[str, Any]]]:
        """返回 (滚动摘要, 窗口消息)；不可用时返回 ("", [])"""
        keys = self.keys_for(session_id)
        try:
            client = await self._ensure_client()
            summary = await client.get(keys["sum"]) or ""
            raw = await client.lrange(keys["msgs"], 0, -1)
            messages = []
            for item in raw:
                try:
                    messages.append(json.loads(item))
                except (TypeError, ValueError):
                    continue
            return summary, messages
        except Exception as e:
            self._trip(e)
            return "", []

    async def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> bool:
        """追加一轮对话并按需触发压缩。失败时熔断并返回 False（不抛出）。"""
        keys = self.keys_for(session_id)
        try:
            client = await self._ensure_client()
            pipe = client.pipeline()
            pipe.rpush(
                keys["msgs"],
                json.dumps({"role": "user", "content": user_text, "ts": _now_iso()}, ensure_ascii=False),
            )
            pipe.rpush(
                keys["msgs"],
                json.dumps(
                    {"role": "assistant", "content": assistant_text, "ts": _now_iso()},
                    ensure_ascii=False,
                ),
            )
            pipe.expire(keys["msgs"], config.memory_redis_ttl_seconds)
            pipe.expire(keys["sum"], config.memory_redis_ttl_seconds)
            pipe.expire(keys["seq"], config.memory_redis_ttl_seconds)
            await pipe.execute()
            await self._compress_if_needed(client, keys)
            return True
        except Exception as e:
            self._trip(e)
            return False

    async def _compress_if_needed(self, client, keys: dict[str, str]) -> None:
        llen = await client.llen(keys["msgs"])
        if llen < config.memory_window_messages:
            return
        if llen >= MAX_LIST_LENGTH:
            # 压缩持续失败的安全阀：静默丢弃最旧消息，只保留窗口尾部
            await client.ltrim(keys["msgs"], -config.memory_compress_keep, -1)
            logger.warning(f"短期记忆消息数达到安全上限 {MAX_LIST_LENGTH}，已静默截断")
            return

        compress_count = llen - config.memory_compress_keep
        if compress_count <= 0:
            return

        # 抢不到压缩锁则本轮跳过，下轮重试
        acquired = await client.set(keys["lock"], "1", nx=True, px=10000)
        if not acquired:
            return

        try:
            oldest_raw = await client.lrange(keys["msgs"], 0, compress_count - 1)
            oldest = []
            for item in oldest_raw:
                try:
                    oldest.append(json.loads(item))
                except (TypeError, ValueError):
                    continue
            if not oldest:
                return

            # 加锁后重读摘要，避免与并发压缩竞态
            old_summary = await client.get(keys["sum"]) or ""
            merged = await self._summarizer.summarize(old_summary, oldest)

            pipe = client.pipeline()
            pipe.set(keys["sum"], merged, ex=config.memory_redis_ttl_seconds)
            pipe.ltrim(keys["msgs"], compress_count, -1)
            await pipe.execute()
            logger.debug(f"短期记忆压缩完成: session 压缩 {compress_count} 条消息")
        except Exception as e:
            logger.warning(f"短期记忆压缩失败（保留原文，下轮重试）: {e}")
        finally:
            try:
                await client.delete(keys["lock"])
            except Exception:
                pass

    async def history(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """窗口内消息历史，可选最近 limit 条"""
        _, messages = await self.build_context(session_id)
        if limit is not None and limit > 0:
            messages = messages[-limit:]
        return messages

    async def get_summary(self, session_id: str) -> str:
        summary, _ = await self.build_context(session_id)
        return summary

    async def clear(self, session_id: str) -> bool:
        keys = self.keys_for(session_id)
        try:
            client = await self._ensure_client()
            await client.delete(keys["msgs"], keys["sum"], keys["seq"])
            return True
        except Exception as e:
            self._trip(e)
            return False


short_term_memory = ShortTermMemory()


def window_messages_to_langchain(messages: list[dict[str, Any]]):
    """将 Redis 窗口消息转换为 LangChain 消息列表"""
    from langchain_core.messages import AIMessage, HumanMessage

    converted = []
    for item in messages:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted
