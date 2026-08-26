from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.eval.dataset import load_eval_dataset
from app.eval.ragas_runner import run_ragas_evaluation
from app.eval.report_writer import build_report_path, write_report
from app.eval.runtime import evaluation_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline ragas evaluation")
    parser.add_argument("--dataset", required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output", help="Optional output path for the JSON report")
    return parser


async def _run(dataset_path: str, output_path: str | None) -> Path:
    samples = load_eval_dataset(dataset_path)
    async with evaluation_runtime():
        report = await run_ragas_evaluation(samples)
        report_path = Path(output_path) if output_path else build_report_path(dataset_path)
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
    asyncio.run(_run(args.dataset, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
