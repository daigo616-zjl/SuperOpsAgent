"""长期记忆召回：三路融合（强事实 > ES 长文本 > 向量语义）"""

import asyncio
import re

from loguru import logger

from app.config import config
from app.memory.es_memory_store import es_memory_store
from app.memory.long_term_repository import long_term_repository
from app.memory.milvus_memory_store import milvus_memory_store
from app.memory.models import MemoryContext, MemoryHit

_SPLIT_RE = re.compile(r"[\s，。！？、；：,.!?;:()（）\[\]【】\"'`]+")
# 召回查询分词时保留的最小词长（英文 2 字符 / 中文 2 字）
MIN_TERM_LENGTH = 2


def extract_terms(query: str) -> list[str]:
    terms = []
    for token in _SPLIT_RE.split(query):
        token = token.strip()
        if len(token) >= MIN_TERM_LENGTH and token not in terms:
            terms.append(token)
    return terms[:12]


class MemoryRecallService:
    """三路召回按优先级融合；任一路故障独立降级，不影响其余两路"""

    async def recall(self, query: str, user_id: str) -> MemoryContext:
        if not query.strip():
            return MemoryContext()

        terms = extract_terms(query)
        facts_task = asyncio.to_thread(self._recall_facts, user_id, terms)
        es_task = asyncio.to_thread(
            es_memory_store.search, user_id, query, config.memory_es_top_k,
        )
        vec_task = asyncio.to_thread(
            milvus_memory_store.search, user_id, query, config.memory_vec_top_k,
        )
        facts_result, es_result, vec_result = await asyncio.gather(
            facts_task, es_task, vec_task, return_exceptions=True,
        )

        context = MemoryContext()
        if isinstance(facts_result, Exception):
            logger.warning(f"强事实召回降级: {facts_result}")
        else:
            context.facts = facts_result

        if isinstance(es_result, Exception):
            logger.warning(f"ES 长文本记忆召回降级: {es_result}")
        else:
            context.es_hits = es_result

        if isinstance(vec_result, Exception):
            logger.warning(f"向量语义记忆召回降级: {vec_result}")
        else:
            context.vec_hits = vec_result

        # 向量结果让步于 ES：按 content_hash 去重，ES 命中优先
        seen = {hit.content_hash for hit in context.es_hits if hit.content_hash}
        context.vec_hits = [
            hit for hit in context.vec_hits if not hit.content_hash or hit.content_hash not in seen
        ]

        if not context.is_empty():
            logger.info(
                f"长期记忆召回: 强事实={len(context.facts)}, ES={len(context.es_hits)}, "
                f"向量={len(context.vec_hits)}"
            )
        return context

    def _recall_facts(self, user_id: str, terms: list[str]) -> list[MemoryHit]:
        limit = config.memory_facts_max
        hits = long_term_repository.match_facts(user_id, terms, limit=limit)
        if len(hits) < limit:
            # 关键词未命中的全局事实按时间兜底补充
            matched_hashes = {hit.content_hash for hit in hits}
            for hit in long_term_repository.recent_facts(user_id, limit=limit):
                if hit.content_hash not in matched_hashes:
                    hits.append(hit)
                if len(hits) >= limit:
                    break
        return hits[:limit]

    @staticmethod
    def format_prompt_block(context: MemoryContext) -> str:
        """三块分区注入 system prompt，优先级显式声明"""
        blocks: list[str] = []
        if context.facts:
            facts_text = "\n".join(f"- {hit.content}" for hit in context.facts)
            blocks.append(f"[长期记忆-强事实]（必须遵守，不可被下述内容推翻）\n{facts_text}")
        if context.es_hits:
            es_text = "\n".join(f"- {hit.content}" for hit in context.es_hits)
            blocks.append(f"[长期记忆-相关经验]（与当前问题相关的历史对话与方案）\n{es_text}")
        if context.vec_hits:
            vec_text = "\n".join(f"- {hit.content}" for hit in context.vec_hits)
            blocks.append(f"[语义联想-仅供参考]（低置信度，与强事实冲突时以强事实为准）\n{vec_text}")
        return "\n\n".join(blocks)


memory_recall_service = MemoryRecallService()
