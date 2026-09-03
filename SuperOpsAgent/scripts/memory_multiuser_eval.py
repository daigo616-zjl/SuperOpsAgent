"""多用户分层记忆评测：数据集驱动（eval/fixtures/memory_multiuser_dataset.json）。

前置：服务进程已停止（Milvus Lite 文件锁单进程）。
覆盖能力：
  1. 跨会话精确召回（新 session + 同 user）
  2. 语义改写召回（口语化指代）
  3. 用户隔离（不同 user 记忆互不可见）
  4. 冷启动（无记忆用户不得幻觉出他人事实）
  5. 事实更正 supersede（旧事实状态落库 + 不再出现在召回答案中）
用法：PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/memory_multiuser_eval.py
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy import text

FIXTURE = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "memory_multiuser_dataset.json"

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  -> {detail}" if detail else ""))


async def wait_jobs_done(session_id: str, expected: int, timeout_seconds: int = 240) -> int:
    from app.core.postgres import postgres_manager

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with postgres_manager.engine.connect() as conn:
            row = conn.execute(text(
                "select count(*) from rag_memory_jobs where session_id = :s and status = 'done'"
            ), {"s": session_id}).scalar_one()
        if row >= expected:
            return row
        await asyncio.sleep(3)
    raise TimeoutError(f"outbox 任务超时未完成: {session_id} 期望 {expected}")


def evaluate_answer(answer: str, probe: dict) -> list[str]:
    failures = []
    for keyword in probe.get("expect_all", []):
        if keyword not in answer:
            failures.append(f"缺少关键词「{keyword}」")
    expect_any = probe.get("expect_any", [])
    if expect_any and not any(keyword in answer for keyword in expect_any):
        failures.append(f"缺少任一关键词 {expect_any}")
    for keyword in probe.get("excludes_all", []):
        if keyword in answer:
            failures.append(f"出现不应出现的关键词「{keyword}」")
    return failures


async def main() -> None:
    from app.core.es_client import es_client_manager
    from app.core.milvus_client import milvus_manager
    from app.core.postgres import postgres_manager
    from app.memory.memory_writer import memory_write_worker
    from app.memory.redis_client import redis_client_manager
    from app.services.rag_agent_service import rag_agent_service

    dataset = json.loads(FIXTURE.read_text(encoding="utf-8"))
    run_tag = uuid.uuid4().hex[:6]

    postgres_manager.connect()
    await redis_client_manager.connect()
    await es_client_manager.connect()
    milvus_manager.connect()
    memory_write_worker.start()

    # ---- 注册数据集用户（长期记忆主体） ----
    user_ids: dict[str, str] = {}
    with postgres_manager.engine.begin() as conn:
        for user in dataset["users"]:
            user_ids[user["key"]] = str(conn.execute(text(
                "insert into users (display_name) values (:name) returning id"
            ), {"name": user["display_name"]}).scalar_one())
    print(f"运行标识: mu-{run_tag}")
    print(f"用户映射: { {k: v[:8] for k, v in user_ids.items()} }\n")

    pending_jobs: dict[str, int] = {}  # session -> 期望 done 的 turn 数

    async def flush_memory_writes() -> None:
        for session_id, expected in pending_jobs.items():
            done = await wait_jobs_done(session_id, expected)
            print(f"[outbox] {session_id}: {done}/{expected} done")
        pending_jobs.clear()
        from app.config import config

        es_client_manager.get_sync_client().indices.refresh(index=config.es_memory_index)

    # ---- 按数据集顺序执行 flows ----
    for flow in dataset["flows"]:
        name = flow["name"]
        user_id = user_ids[flow["user"]]
        session_id = f"mu-{run_tag}-{flow['session']}"
        print(f"\n--- {name} (user={flow['user']}, session={session_id}) ---")

        if "turns" in flow:
            for i, question in enumerate(flow["turns"], 1):
                result = await rag_agent_service.query_with_context(
                    question, session_id, user_id=user_id,
                )
                print(f"  写入轮 {i} 回答: {result.answer[:60]}…")
                pending_jobs[session_id] = pending_jobs.get(session_id, 0) + 1

        if "probes" in flow:
            await flush_memory_writes()
            for probe in flow["probes"]:
                question = probe["question"]
                result = await rag_agent_service.query_with_context(
                    question, session_id, user_id=user_id,
                )
                answer = result.answer
                print(f"  问: {question}")
                print(f"  答: {answer[:150]}…")
                failures = evaluate_answer(answer, probe)
                record(probe["question"][:30], not failures, "; ".join(failures))

    await flush_memory_writes()

    # ---- 数据库状态校验 ----
    print("\n--- db_checks ---")
    with postgres_manager.engine.connect() as conn:
        for check in dataset["db_checks"]:
            user_id = user_ids[check["user"]]
            if check["type"] == "superseded":
                count = conn.execute(text(
                    "select count(*) from rag_memory_facts "
                    "where user_id = :u and status = 'superseded' and content like :pat"
                ), {"u": user_id, "pat": f"%{check['content_like']}%"}).scalar_one()
                record(check["name"], count >= 1, f"superseded 命中 {count} 条")
            elif check["type"] == "no_rows":
                count = conn.execute(text(
                    "select count(*) from rag_memory_facts where user_id = :u"
                ), {"u": user_id}).scalar_one()
                record(check["name"], count == 0, f"facts 行数 {count}")

    # ---- 汇总 ----
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== 多用户评测: {passed}/{len(results)} 通过 ===")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        sys.exit(1)
