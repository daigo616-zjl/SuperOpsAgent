from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.eval.dataset import load_eval_dataset
from app.eval.ragas_runner import (
    normalize_metric_names,
    run_ragas_evaluation,
    score_existing_report,
)
from app.eval.report_writer import build_report_path, load_report, write_report
from app.eval.runtime import evaluation_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline ragas evaluation")
    parser.add_argument("--dataset", help="Path to the JSONL dataset")
    parser.add_argument(
        "--input-report",
        help="Reuse answers and contexts from an existing report without rerunning RAG",
    )
    parser.add_argument("--output", help="Optional output path for the JSON report")
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N samples",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated metrics, or all (default)",
    )
    return parser


def _parse_metrics(value: str) -> tuple[str, ...] | None:
    if value.strip().lower() == "all":
        return None
    return normalize_metric_names(value.split(","))


async def _run(
    dataset_path: str | None,
    output_path: str | None,
    limit: int | None,
    metric_names: tuple[str, ...] | None,
    input_report_path: str | None,
) -> Path:
    if not dataset_path and not input_report_path:
        raise ValueError("Provide either --dataset or --input-report")
    if dataset_path and input_report_path:
        raise ValueError("Use only one of --dataset and --input-report")

    if input_report_path:
        report = load_report(input_report_path)
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive")
            report.details = report.details[:limit]
            report.summary.total = len(report.details)
        selected_metrics = metric_names or ("faithfulness", "answer_correctness")
        report = await score_existing_report(report, selected_metrics)
        report_path = Path(output_path) if output_path else build_report_path(input_report_path)
    else:
        samples = load_eval_dataset(dataset_path or "")
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive")
            samples = samples[:limit]
        async with evaluation_runtime():
            report = await run_ragas_evaluation(samples, metric_names=metric_names)
        report_path = Path(output_path) if output_path else build_report_path(dataset_path or "")

    write_report(report.to_dict(), report_path)

    print(f"dataset path: {dataset_path}")
    print(f"total: {report.summary.total}")
    print(f"success: {report.summary.success}")
    print(f"failed: {report.summary.failed}")
    print(f"metrics summary: {report.summary.metrics}")
    print(f"report path: {report_path}")

    return report_path


def main() -> int:
    args = build_parser().parse_args()
    metric_names = _parse_metrics(args.metrics)
    asyncio.run(
        _run(
            args.dataset,
            args.output,
            args.limit,
            metric_names,
            args.input_report,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
