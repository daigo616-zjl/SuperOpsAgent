"""分层记忆数据模型"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ExtractedItem:
    """LLM 抽取出的单条长期记忆"""

    type: str  # fact | text | semantic
    content: str
    subject: str = ""
    confidence: float = 1.0
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryHit:
    """召回命中的单条记忆"""

    content: str
    subject: str = ""
    score: float = 0.0
    content_hash: str = ""


@dataclass(slots=True)
class MemoryContext:
    """三路召回融合结果（按优先级排列：facts > es_hits > vec_hits）"""

    facts: list[MemoryHit] = field(default_factory=list)
    es_hits: list[MemoryHit] = field(default_factory=list)
    vec_hits: list[MemoryHit] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.facts or self.es_hits or self.vec_hits)


def content_hash(user_id: str, content: str) -> str:
    """三路存储统一的幂等锚点"""
    import hashlib

    normalized = " ".join(content.split()).strip().lower()
    return hashlib.sha256(f"{user_id}\n{normalized}".encode()).hexdigest()
