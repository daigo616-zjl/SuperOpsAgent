"""知识文档内容 Hash。"""

import hashlib


def compute_content_hash(content: str) -> str:
    """仅统一换行符后计算 UTF-8 SHA-256。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
