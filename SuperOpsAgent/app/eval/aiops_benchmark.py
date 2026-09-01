"""AIOps 场景基准。

按剧本（MOCK_SCENARIO）起 mock MCP 服务，每剧本跑 N 次诊断，
用 EVAL_MODEL 评判根因命中率与幻觉率，落盘 eval/reports/aiops/。
（P5 阶段为 legacy/multiagent 双引擎 A/B 对比，legacy 于 P6 删除。）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from app.config import config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "mcp_servers" / "scenarios"
CLAIM_REF_PATTERN = re.compile(r"\[ev-[a-zA-Z0-9_-]+\]")
ROUND_PATTERN = re.compile(r"第\s*(\d+)\s*轮")

MCP_CLS_PORT = 18003
MCP_MONITOR_PORT = 18004
REQUIRED_MCP_TOOLS = {"query_active_alerts", "search_log", "query_cpu_metrics"}


def list_scenario_ids() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenarios(names: list[str]) -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    for name in names:
        path = SCENARIO_DIR / f"{name}.yaml"
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(scenario, dict) or not scenario.get("ground_truth", {}).get(
            "root_cause"
        ):
            raise ValueError(f"剧本缺少 ground_truth.root_cause: {path}")
        scenarios[name] = scenario
    return scenarios


def summarize_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    """从一次诊断的 SSE 事件流提取可观测指标（纯函数，可单测）。"""
    report = ""
    error = None
    claim_ids: set[str] = set()
    tool_outputs: list[Any] = []
    rounds = 0
    step_count = 0
    report_chunk_count = 0
    event_types: list[str] = []

    for event in events:
        event_type = event.get("type", "")
        event_types.append(event_type)
        if event_type == "complete":
            report = event.get("response", "") or ""
        elif event_type == "error":
            error = event.get("message", "")
        elif event_type == "report_chunk":
            report_chunk_count += 1
        elif event_type == "status":
            match = ROUND_PATTERN.search(event.get("message", ""))
            if match:
                rounds = max(rounds, int(match.group(1)))
        elif event_type == "step_complete":
            step_count += 1
            result = event.get("result")
            if isinstance(result, dict):
                tool_outputs.append(result)
                for claim in result.get("claims", []) or []:
                    if isinstance(claim, dict) and claim.get("claim_id"):
                        claim_ids.add(claim["claim_id"])

    refs = CLAIM_REF_PATTERN.findall(report)
    return {
        "report": report,
        "report_chars": len(report),
        "error": error,
        "claim_ids": sorted(claim_ids),
        "tool_outputs": tool_outputs,
        "rounds": rounds,
        "step_count": step_count,
        "report_chunk_count": report_chunk_count,
        "event_types": event_types,
        "ev_refs_total": len(refs),
        "ev_refs_unresolved": len([ref for ref in refs if ref[1:-1] not in claim_ids]),
    }


def claim_reference_metrics(summary: dict[str, Any]) -> float | None:
    """未解析证据引用占比；报告没有引用时返回 None（不参与聚合）。"""
    total = summary["ev_refs_total"]
    if total == 0:
        return None
    return summary["ev_refs_unresolved"] / total


def parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型回复中鲁棒地提取第一个 JSON 对象（容忍代码围栏与前后缀文本）。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


ROOT_CAUSE_PROMPT = """你是运维诊断评测员。判断诊断报告的结论是否与真实根因一致。

场景描述：{description}
真实根因：{root_cause}

评分规则：
- 报告明确指出的根因与真实根因实质一致（允许措辞不同、粒度略有差异）→ hit=true。
- 只罗列了可能原因但未确认根因、或结论方向相反 → hit=false。
- 若真实根因描述为"无故障"，报告结论为系统正常/未发现故障 → hit=true。

只输出 JSON：{{"hit": true, "reason": "一句话理由"}}"""

HALLUCINATION_PROMPT = """你是事实核查员。诊断报告如下，本次诊断收集到的原始工具输出（JSON）附后。

找出报告中缺乏工具输出支撑的事实性断言：
- 具体数值、状态、错误信息必须能在工具输出中找到对应来源，否则视为幻觉。
- 诊断推断、处置建议、常识性解释不算幻觉。
- 转述工具输出的内容算有支撑。

只输出 JSON：
{{"total_claims": 断言总数, "unsupported_claims": 无支撑断言数, "examples": ["无支撑断言原文，最多 5 条"]}}"""


def _truncate(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...（截断）"


class JudgeClient:
    """用 EVAL_MODEL 做判分；走独立 OpenAI 兼容客户端，不进 LLM 调用计数。"""

    def __init__(self, model: str | None = None):
        from openai import AsyncOpenAI

        self.model = model or config.eval_model
        self.client = AsyncOpenAI(
            api_key=config.dashscope_api_key, base_url=config.dashscope_api_base
        )

    async def _chat_json(self, prompt: str) -> dict[str, Any] | None:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return parse_json_object(content)

    async def judge_root_cause(
        self, report: str, scenario: dict[str, Any]
    ) -> dict[str, Any] | None:
        ground_truth = scenario["ground_truth"]
        prompt = ROOT_CAUSE_PROMPT.format(
            description=scenario.get("description", ""),
            root_cause=ground_truth["root_cause"],
        )
        return await self._chat_json(f"{prompt}\n\n诊断报告：\n{_truncate(report)}")

    async def judge_hallucination(
        self, report: str, tool_outputs: list[Any]
    ) -> dict[str, Any] | None:
        outputs = json.dumps(tool_outputs, ensure_ascii=False, default=str)
        prompt = f"{HALLUCINATION_PROMPT}\n\n诊断报告：\n{_truncate(report)}"
        prompt += f"\n\n工具输出：\n{_truncate(outputs)}"
        return await self._chat_json(prompt)


_LL_CALL_COUNT = {"count": 0}


def install_llm_call_counter() -> None:
    """给所有经 LLMFactory 创建的 ChatQwen 精确计数。

    ChatQwen._agenerate 与 _astream 是两条独立路径（前者直接走 super()，
    不经 _astream），所以两个都要挂，不会重复计数。"""
    import app.core.llm_factory as llm_factory_module
    from langchain_qwq import ChatQwen as _RealChatQwen

    class _CountingChatQwen(_RealChatQwen):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            _LL_CALL_COUNT["count"] += 1
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            _LL_CALL_COUNT["count"] += 1
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

    llm_factory_module.ChatQwen = _CountingChatQwen


class McpServerManager:
    """为每个剧本启动/停止两个 mock MCP 服务子进程。"""

    def __init__(self, output_dir: Path, ready_timeout: float = 90.0):
        self.output_dir = output_dir
        self.ready_timeout = ready_timeout
        self.processes: list[asyncio.subprocess.Process] = []

    async def _assert_ports_free(self, grace_seconds: float = 60.0) -> None:
        """端口被占时先等 grace_seconds（可能是上一个服务正在关闭或瞬时占用），
        超时后报错并指出占用者 PID。"""
        deadline = time.monotonic() + grace_seconds
        while True:
            occupier = None
            for port in (MCP_CLS_PORT, MCP_MONITOR_PORT):
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    await writer.wait_closed()
                    occupier = port
                    break
                except OSError:
                    continue
            if occupier is None:
                return
            if time.monotonic() >= deadline:
                pid = self._find_port_pid(occupier)
                raise RuntimeError(
                    f"端口 {occupier} 被进程 {pid or '未知'} 占用。"
                    "请先停掉 start-windows.bat 启动的 MCP 服务，或结束占用进程后重跑。"
                )
            print(f"    端口 {occupier} 暂被占用，{grace_seconds:.0f}s 内重试...")
            await asyncio.sleep(5)

    @staticmethod
    def _find_port_pid(port: int) -> int | None:
        try:
            output = subprocess.run(
                ["powershell", "-Command",
                 f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                 f"-ErrorAction SilentlyContinue).OwningProcess"],
                capture_output=True, text=True, timeout=15,
                encoding="utf-8", errors="replace",
            ).stdout.strip()
            return int(output.splitlines()[0]) if output else None
        except (ValueError, subprocess.SubprocessError):
            return None

    async def start(self, scenario_id: str) -> None:
        await self._assert_ports_free()
        env = dict(**__import__("os").environ)
        env["MOCK_SCENARIO"] = scenario_id
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        servers = {
            "cls": REPO_ROOT / "mcp_servers" / "cls_server.py",
            "monitor": REPO_ROOT / "mcp_servers" / "monitor_server.py",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, script in servers.items():
            log_path = self.output_dir / f"mcp-{scenario_id}-{name}.log"
            log_file = log_path.open("w", encoding="utf-8")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-X",
                "utf8",
                str(script),
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self.processes.append(process)

        await self._wait_ready()

    async def _wait_ready(self) -> None:
        from app.agent.aiops.tool_registry import get_tool_registry

        deadline = time.monotonic() + self.ready_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                registry = await get_tool_registry()
                missing = REQUIRED_MCP_TOOLS - set(registry.descriptors)
                if not missing:
                    return
                last_error = RuntimeError(f"MCP 工具缺失: {sorted(missing)}")
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(2)
        raise TimeoutError(f"MCP 服务未在 {self.ready_timeout}s 内就绪: {last_error}")

    async def stop(self) -> None:
        for process in self.processes:
            if process.returncode is None:
                process.terminate()
        for process in self.processes:
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        self.processes = []
        # 确认端口真正释放后再返回，避免下一个剧本的端口检查误判
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", MCP_CLS_PORT
                )
                writer.close()
                await writer.wait_closed()
            except OSError:
                return
            await asyncio.sleep(1)


async def run_diagnosis(scenario: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    from app.services.aiops_service import aiops_service

    user_input = f"诊断 {scenario['service_name']} 告警并给出根因分析"
    return [event async for event in aiops_service.execute(user_input, session_id=session_id)]


async def execute_run(
    scenario_id: str,
    scenario: dict[str, Any],
    run_index: int,
    judge: JudgeClient | None,
) -> dict[str, Any]:
    from app.services import aiops_service as aiops_service_module

    session_id = f"ab-{scenario_id}-{run_index}-{datetime.now(UTC).strftime('%H%M%S')}"
    _LL_CALL_COUNT["count"] = 0
    started = time.perf_counter()
    events = await run_diagnosis(scenario, session_id)
    wall_seconds = time.perf_counter() - started

    summary = summarize_run(events)
    result: dict[str, Any] = {
        "scenario": scenario_id,
        "run": run_index,
        "session_id": session_id,
        "wall_seconds": round(wall_seconds, 1),
        "llm_calls": _LL_CALL_COUNT["count"],
        "rounds": summary["rounds"],
        "step_count": summary["step_count"],
        "report_chars": summary["report_chars"],
        "ev_refs_total": summary["ev_refs_total"],
        "ev_refs_unresolved": summary["ev_refs_unresolved"],
        "unresolved_ref_ratio": claim_reference_metrics(summary),
        "error": summary["error"],
        "report": summary["report"],
        "judge_root_cause": None,
        "judge_hallucination": None,
    }

    if judge is not None and not summary["error"]:
        result["judge_root_cause"] = await judge.judge_root_cause(
            summary["report"], scenario
        )
        result["judge_hallucination"] = await judge.judge_hallucination(
            summary["report"], summary["tool_outputs"]
        )

    # 证据落盘诊断上下文，报告文本单独存文件避免主报告过大
    runs_dir = Path(config.eval_output_dir) / "aiops"
    runs_dir.mkdir(parents=True, exist_ok=True)
    detail_path = runs_dir / f"{session_id}.json"
    detail_path.write_text(
        json.dumps(
            {"summary": {k: v for k, v in result.items() if k != "report"},
             "events": events},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    result["detail_path"] = str(detail_path)

    try:
        aiops_service_module.evidence_repository.finish_session(
            session_id, status="completed"
        )
    except Exception:
        pass
    return result


def judge_hit(run: dict[str, Any]) -> bool | None:
    verdict = run.get("judge_root_cause")
    if not isinstance(verdict, dict):
        return None
    return bool(verdict.get("hit"))


def judge_hallucination_rate(run: dict[str, Any]) -> float | None:
    verdict = run.get("judge_hallucination")
    if not isinstance(verdict, dict):
        return None
    total = verdict.get("total_claims")
    unsupported = verdict.get("unsupported_claims")
    if not isinstance(total, (int, float)) or not isinstance(unsupported, (int, float)):
        return None
    if total <= 0:
        return 0.0
    return unsupported / total


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["scenario"]].append(run)

    aggregated: dict[str, dict[str, Any]] = {}
    for scenario, group in sorted(groups.items()):
        aggregated[scenario] = {
            "scenario": scenario,
            "runs": len(group),
            "hit_rate": _mean(
                [1.0 for r in group if judge_hit(r) is True]
                + [0.0 for r in group if judge_hit(r) is False]
            ),
            "hallucination_rate": _mean(
                [
                    rate
                    for r in group
                    if (rate := judge_hallucination_rate(r)) is not None
                ]
            ),
            "avg_wall_seconds": _mean([r["wall_seconds"] for r in group]),
            "avg_llm_calls": _mean([float(r["llm_calls"]) for r in group]),
            "avg_rounds": _mean([float(r["rounds"]) for r in group]),
        }
    return aggregated


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    scenario_ids = args.scenarios or list_scenario_ids()
    scenarios = load_scenarios(scenario_ids)
    output_dir = Path(config.eval_output_dir) / "aiops"
    output_dir.mkdir(parents=True, exist_ok=True)

    judge = None if args.no_judge else JudgeClient()
    install_llm_call_counter()
    manager = McpServerManager(output_dir)

    runs: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]
        print(f"=== 剧本 {scenario_id}: 启动 mock MCP 服务 ===")
        await manager.start(scenario_id)
        try:
            for run_index in range(args.runs):
                print(f"--- {scenario_id} / 第 {run_index + 1} 次 ---")
                run = await execute_run(scenario_id, scenario, run_index, judge)
                hit = judge_hit(run)
                print(
                    f"    wall={run['wall_seconds']}s llm={run['llm_calls']} "
                    f"rounds={run['rounds']} steps={run['step_count']} "
                    f"hit={hit} error={run['error']}"
                )
                runs.append(run)
        finally:
            # 每个剧本跑完必须停服务换剧本，否则下一个剧本会用到上一个的 mock 数据
            await manager.stop()

    aggregated = aggregate(runs)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "judge_model": None if judge is None else judge.model,
        "scenarios": scenario_ids,
        "runs_per_scenario": args.runs,
        "aggregate": aggregated,
        "runs": [
            {k: v for k, v in run.items() if k not in ("report", "detail_path")}
            for run in runs
        ],
    }
    report_path = output_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-aiops-benchmark.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("\n===== 基准汇总 =====")
    for key, agg in report["aggregate"].items():
        print(
            f"{key}: hit={agg['hit_rate']} hallucination={agg['hallucination_rate']} "
            f"wall={agg['avg_wall_seconds']}s llm={agg['avg_llm_calls']} "
            f"rounds={agg['avg_rounds']}"
        )
    print(f"报告: {report.get('report_path')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIOps 场景基准")
    parser.add_argument(
        "--scenarios", default="", help="逗号分隔剧本名，默认全部 5 个"
    )
    parser.add_argument("--runs", type=int, default=3, help="每剧本运行次数")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 判分")
    args = parser.parse_args(argv)
    args.scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    report = asyncio.run(run_benchmark(args))
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
