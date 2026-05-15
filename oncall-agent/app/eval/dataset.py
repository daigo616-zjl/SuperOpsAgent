from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalSample:
    id: str
    question: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_eval_dataset(path: str) -> list[EvalSample]:
    samples: list[EvalSample] = []

    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc

        missing_fields = [field for field in ("id", "question", "ground_truth") if not payload.get(field)]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Missing required fields on line {line_number}: {missing}")

        metadata = payload.get("metadata")
        if metadata is None:
            metadata_dict: dict[str, Any] = {}
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        else:
            raise ValueError(f"Invalid metadata on line {line_number}: expected object")

        samples.append(
            EvalSample(
                id=str(payload["id"]),
                question=str(payload["question"]),
                ground_truth=str(payload["ground_truth"]),
                metadata=metadata_dict,
            )
        )

    return samples
