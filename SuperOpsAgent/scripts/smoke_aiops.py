"""AIOps SSE 接口冒烟测试（P6）。

起 mock MCP（MOCK_SCENARIO 选剧本）+ FastAPI，通过 HTTP SSE 调一次 /api/aiops
诊断，断言事件流完整：status / step_complete 出现、以 complete 结束、
报告非空、无 error 事件。跑完自动停掉全部子进程。

用法（需 DASHSCOPE_API_KEY，先停掉已在运行的服务）：
    python scripts/smoke_aiops.py                     # 默认 db-slow-query 剧本
    python scripts/smoke_aiops.py --scenario oom-kill
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

# Windows 控制台默认 GBK，打印 emoji/中文会崩；子进程已单独设 PYTHONUTF8
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "mcp_servers" / "scenarios"
MCP_CLS_PORT = 18003
MCP_MONITOR_PORT = 18004
LOG_DIR = REPO_ROOT / "eval" / "reports" / "aiops"

MCP_START_TIMEOUT = 90.0
# lifespan 预热含 Rerank 模型加载；huggingface 不可达时重试可达 2 分钟以上
API_START_TIMEOUT = 300.0
# 诊断墙钟预算 300s + 流式收尾余量；SSE 两个事件之间最长静默约 150s（任务上限）
DIAGNOSIS_TIMEOUT = 420.0
SSE_READ_TIMEOUT = 240.0


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_port(port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_in_use(port):
            return
        time.sleep(1)
    raise TimeoutError(f"端口 {port} 未在 {timeout:.0f}s 内就绪")


def wait_health(api_port: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    last_error: Exception | str | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{api_port}/api/health", timeout=5)
            # 任何 HTTP 响应都说明服务已就绪；503 是 ES 等外部依赖不可用的
            # 降级响应，AIOps 诊断不依赖这些组件
            return resp.status_code
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"/api/health 未在 {timeout:.0f}s 内可连: {last_error}")


def terminate(process: subprocess.Popen, log_file) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log_file.close()


def stream_diagnosis(api_port: int, service_name: str) -> list[dict]:
    url = f"http://127.0.0.1:{api_port}/api/aiops"
    timeout = httpx.Timeout(
        connect=10.0, read=SSE_READ_TIMEOUT, write=30.0, pool=30.0
    )
    events: list[dict] = []
    started = time.monotonic()
    with httpx.stream(
        "POST",
        url,
        json={"session_id": "smoke", "service_name": service_name},
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        data_lines: list[str] = []
        for line in resp.iter_lines():
            if time.monotonic() - started > DIAGNOSIS_TIMEOUT:
                raise TimeoutError(f"诊断超过 {DIAGNOSIS_TIMEOUT:.0f}s 未结束")
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
            elif not line.strip() and data_lines:
                payload = json.loads("\n".join(data_lines))
                events.append(payload)
                data_lines = []
                if payload.get("type") in ("complete", "error"):
                    break
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="AIOps SSE 接口冒烟测试")
    parser.add_argument("--scenario", default="db-slow-query")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "18000"))
    )
    args = parser.parse_args()

    scenario_path = SCENARIO_DIR / f"{args.scenario}.yaml"
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    service_name = scenario["service_name"]

    busy = [p for p in (args.port, MCP_CLS_PORT, MCP_MONITOR_PORT) if port_in_use(p)]
    if busy:
        print(f"❌ 端口被占用: {busy}，请先停掉已在运行的服务（make stop 或 start-windows.bat）")
        return 1

    env = dict(**os.environ)
    # HF_HUB_OFFLINE：模型已本地缓存；不设的话 rerank 预热会因 huggingface
    # 不可达反复重试，lifespan 拖到 5 分钟以上才完成
    env.update(
        MOCK_SCENARIO=args.scenario,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[subprocess.Popen, object]] = []
    for name, script in (
        ("cls", "mcp_servers/cls_server.py"),
        ("monitor", "mcp_servers/monitor_server.py"),
    ):
        log_file = (LOG_DIR / f"smoke-mcp-{name}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", str(REPO_ROOT / script)],
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log_file))

    api_proc = None
    api_log = None
    try:
        wait_port(MCP_CLS_PORT, MCP_START_TIMEOUT)
        wait_port(MCP_MONITOR_PORT, MCP_START_TIMEOUT)
        print("✅ mock MCP 服务就绪")

        api_log = (LOG_DIR / "smoke-api.log").open("w", encoding="utf-8")
        api_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", str(args.port),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=api_log,
            stderr=subprocess.STDOUT,
        )
        health_status = wait_health(args.port, API_START_TIMEOUT)
        note = "" if health_status == 200 else f"（health={health_status}，外部依赖降级，不影响诊断）"
        print(f"✅ FastAPI 就绪 (http://127.0.0.1:{args.port}){note}")

        print(f"▶️  开始诊断: scenario={args.scenario} service={service_name}（约 1-5 分钟）")
        started = time.monotonic()
        events = stream_diagnosis(args.port, service_name)
        wall = time.monotonic() - started

        types = [e.get("type") for e in events]
        complete = next((e for e in events if e.get("type") == "complete"), None)
        errors = [e for e in events if e.get("type") == "error"]
        # API 路径的 complete 事件把报告包在 diagnosis.report 里
        report = ""
        if complete:
            report = (
                (complete.get("diagnosis") or {}).get("report")
                or complete.get("response")
                or ""
            )
        step_count = types.count("step_complete")

        print(f"\n耗时 {wall:.0f}s，事件 {len(events)} 条（step {step_count}），报告 {len(report)} 字符")
        checks = [
            ("status 事件出现", "status" in types),
            ("step_complete 事件出现", step_count > 0),
            ("以 complete 结束", complete is not None),
            ("报告非空", len(report) > 0),
            ("无 error 事件", not errors),
        ]
        ok = True
        for label, passed in checks:
            print(f"  {'✅' if passed else '❌'} {label}")
            ok = ok and passed
        for error in errors[:3]:
            print(f"  error 事件: {error.get('message')}")
        print(f"\nAPI 日志: {LOG_DIR / 'smoke-api.log'}")
        print("✅ 冒烟 PASS" if ok else "❌ 冒烟 FAIL")
        return 0 if ok else 1
    finally:
        if api_proc is not None:
            terminate(api_proc, api_log)
        for proc, log_file in procs:
            terminate(proc, log_file)


if __name__ == "__main__":
    sys.exit(main())
