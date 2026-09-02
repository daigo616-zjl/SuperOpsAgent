"""长期记忆：PG 强事实仓储

强事实零容错：召回优先级最高，结果不可被 ES/向量召回推翻。
幂等锚点 content_hash = sha256(user_id + normalized content)，
同 subject 新事实将旧 active 记录置 superseded。
"""

import uuid
from typing import Any

from sqlalchemy import text

from app.core.postgres import postgres_manager
from app.memory.models import MemoryHit, content_hash


class LongTermRepository:
    @staticmethod
    def _hit(row: Any) -> MemoryHit:
        mapping = dict(row._mapping)
        return MemoryHit(
            content=mapping.get("content", ""),
            subject=mapping.get("subject", "") or "",
            score=1.0,
            content_hash=mapping.get("content_hash", "") or "",
        )

    def upsert_fact(
        self,
        user_id: str,
        content: str,
        subject: str = "",
        keywords: list[str] | None = None,
    ) -> str:
        """写入强事实；同 subject 旧 active 记录置 superseded。返回 content_hash。"""
        c_hash = content_hash(user_id, content)
        with postgres_manager.engine.begin() as connection:
            connection.execute(text("""
                update rag_memory_facts
                set status = 'superseded', updated_at = now()
                where user_id = :user_id and status = 'active'
                  and subject = :subject and subject <> ''
                  and content_hash <> :content_hash
            """), {"user_id": user_id, "subject": subject, "content_hash": c_hash})
            connection.execute(text("""
                insert into rag_memory_facts
                    (memory_id, user_id, content, subject, keywords, content_hash)
                values (:memory_id, :user_id, :content, :subject,
                        :keywords, :content_hash)
                on conflict (user_id, content_hash) where status = 'active' do nothing
            """), {
                "memory_id": str(uuid.uuid4()),
                "user_id": user_id,
                "content": content,
                "subject": subject,
                "keywords": list(keywords or []),
                "content_hash": c_hash,
            })
        return c_hash

    def match_facts(
        self, user_id: str, terms: list[str], limit: int = 20,
    ) -> list[MemoryHit]:
        """按 subject/content 关键词与 keywords 数组重叠召回强事实"""
        if not terms:
            return []
        patterns = [f"%{term}%" for term in terms if term.strip()]
        if not patterns:
            return []
        with postgres_manager.engine.connect() as connection:
            rows = connection.execute(text("""
                select content, subject, content_hash
                from rag_memory_facts
                where user_id = :user_id and status = 'active'
                  and (
                      content ilike any(cast(:patterns as text[]))
                      or subject ilike any(cast(:patterns as text[]))
                      or keywords && cast(:terms as text[])
                  )
                order by updated_at desc
                limit :limit
            """), {
                "user_id": user_id,
                "patterns": patterns,
                "terms": [t for t in terms if t.strip()],
                "limit": limit,
            }).all()
            return [self._hit(row) for row in rows]

    def recent_facts(self, user_id: str, limit: int = 20) -> list[MemoryHit]:
        """按时间兜底召回（全局事实，不依赖关键词命中）"""
        with postgres_manager.engine.connect() as connection:
            rows = connection.execute(text("""
                select content, subject, content_hash
                from rag_memory_facts
                where user_id = :user_id and status = 'active'
                order by updated_at desc
                limit :limit
            """), {"user_id": user_id, "limit": limit}).all()
            return [self._hit(row) for row in rows]


long_term_repository = LongTermRepository()
