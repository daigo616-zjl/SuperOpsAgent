from __future__ import annotations

from pathlib import PurePath
from typing import Any


def normalize_source(source: str) -> str:
    """Normalize a source label so dataset paths and indexed file names can match."""
    normalized = source.strip().replace("\\", "/")
    return PurePath(normalized).name.casefold()


def relevant_sources_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Read one or more relevant source labels from an evaluation sample."""
    raw_sources = metadata.get("relevant_sources")
    if raw_sources is None:
        raw_sources = metadata.get("source")

    if isinstance(raw_sources, str):
        values = [raw_sources]
    elif isinstance(raw_sources, list):
        values = [value for value in raw_sources if isinstance(value, str)]
    else:
        values = []

    return list(dict.fromkeys(source for value in values if (source := normalize_source(value))))


def recall_at_k(
    relevant_sources: list[str],
    retrieved_sources: list[str],
    k: int,
) -> float | None:
    """Return document-level Recall@K, or None when no relevance labels exist."""
    relevant = {normalize_source(source) for source in relevant_sources if source.strip()}
    if not relevant:
        return None

    retrieved = {
        normalize_source(source)
        for source in retrieved_sources[: max(0, k)]
        if source.strip()
    }
    return len(relevant & retrieved) / len(relevant)


def hit_at_k(
    relevant_sources: list[str],
    retrieved_sources: list[str],
    k: int,
) -> float | None:
    """Return 1 when any relevant document occurs in the first K results."""
    recall = recall_at_k(relevant_sources, retrieved_sources, k)
    if recall is None:
        return None
    return float(recall > 0)
