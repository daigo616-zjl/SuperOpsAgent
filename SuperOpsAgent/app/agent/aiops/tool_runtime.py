"""工具调用机械内核：证据出处摘要与时间戳，绝不调 LLM。"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def args_digest(tool_name: str, arguments: dict[str, Any], content: str = "") -> str:
    """工具调用的确定性摘要，作为证据 provenance 的出处指纹。"""
    payload = json.dumps(
        {"tool": tool_name, "args": arguments, "content": content[:500]},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def utc_now() -> datetime:
    return datetime.now(UTC)
