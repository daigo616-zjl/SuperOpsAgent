from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import config
from app.eval.ragas_runner import EvalReport


def build_report_path(dataset_path: str) -> Path:
    dataset_stem = Path(dataset_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.eval_output_dir) / f"{dataset_stem}-{timestamp}.json"


def write_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_report(input_path: str | Path) -> EvalReport:
    """Load a prior report for metric-only rescoring."""
    path = Path(input_path)
    return EvalReport.from_dict(json.loads(path.read_text(encoding="utf-8")))
