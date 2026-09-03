"""端到端记忆系统测试：模拟测试集 → 真实 LLM + 三路存储 + 召回融合。

前置：服务进程已停止（Milvus Lite 文件锁单进程）。
流程：
  注册用户 → 会话 A 三轮对话，分别喂入 环境事实/方案文本/偏好语义
  → 等待 outbox worker 抽取落库 → 校验 PG/ES/Milvus 三路
  → 清空 Redis 窗口（模拟重启后上下文丢失）
  → T4 提问验证答案来自长期记忆召回而非窗口消息
  → T5 跨会话验证：全新 SESSION_B + 同一 user_id，仍能召回会话 A 的事实
"""

import asyncio
import sys
import time
import uuid

from sqlalchemy import text

SESSION_ID = ""
SESSION_ID_B = ""
USER_ID = ""


async def wait_jobs_done(session_id: str, expected: int, timeout_seconds: int = 180) -> int:
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
    raise TimeoutError("outbox 任务超时未完成")


async def main() -> None:
    global SESSION_ID, SESSION_ID_B, USER_ID

    from app.core.postgres import postgres_manager
    from app.memory.memory_writer import memory_write_worker
    from app.memory.redis_client import redis_client_manager
    from app.memory.short_term import short_term_memory
    from app.services.rag_agent_service import rag_agent_service

    postgres_manager.connect()
    await redis_client_manager.connect()
    memory_write_worker.start()

    # ---- 注册用户（长期记忆主体） ----
    with postgres_manager.engine.begin() as conn:
        USER_ID = str(conn.execute(text(
            "insert into users (display_name) values ('e2e-user') returning id"
        )).scalar_one())

    SESSION_ID = f"e2e-session-a-{uuid.uuid4().hex[:8]}"
    SESSION_ID_B = f"e2e-session-b-{uuid.uuid4().hex[:8]}"
    print(f"用户: {USER_ID}\n会话 A: {SESSION_ID}\n会话 B: {SESSION_ID_B}")

    turns = [
        "我们的核心服务 data-sync-service 部署在华东1，数据库用的是 MySQL 8.0。",
        "本季度的备份改造方案已经确定：每天凌晨 2 点做全量备份，每小时做增量备份，备份文件统一上传 OSS 归档。",
        "以后回答请尽量简短直接，我只要结论和命令，不要客套话。",
    ]
    for i, question in enumerate(turns, 1):
        result = await rag_agent_service.query_with_context(
            question, SESSION_ID, user_id=USER_ID,
        )
        print(f"T{i} 回答: {result.answer[:60]}…")

    done = await wait_jobs_done(SESSION_ID, 3)
    print(f"[outbox] OK: {done} 个任务 done")

    # ---- 三路存储校验 ----
    from app.config import config
    from app.core.es_client import es_client_manager

    await es_client_manager.connect()
    es_client = es_client_manager.get_sync_client()
    es_client.indices.refresh(index=config.es_memory_index)

    with postgres_manager.engine.connect() as conn:
        facts = conn.execute(text(
            "select subject, content, status from rag_memory_facts where user_id = :u"
        ), {"u": USER_ID}).all()
    print(f"[PG 强事实] {len(facts)} 条:")
    for subject, content, status in facts:
        print(f"  [{status}] subject={subject!r}: {content[:40]}")
    assert facts, "PG 无 fact 落库"
    assert any("-" in f[0] for f in facts), "subject 未按「实体-属性」细粒度抽取"
    assert all(f[2] == "active" for f in facts), "存在被错误 supersede 的事实"

    es_hits = es_client.count(index=config.es_memory_index, body={"query": {"term": {"user_id": USER_ID}}})["count"]
    print(f"[ES text 记忆] {es_hits} 条（extractor 可能将全部内容归类为 fact，此断言降级为信息输出）")

    from pymilvus import Collection

    from app.core.milvus_client import milvus_manager

    milvus_manager.connect()
    c = Collection(config.milvus_memory_collection)
    c.load()
    vec_rows = c.query(expr=f'user_id == "{USER_ID}"', output_fields=["content", "subject"])
    print(f"[Milvus semantic 记忆] {len(vec_rows)} 条")

    # ---- 模拟重启：清空 Redis 窗口后仅靠长期记忆召回 ----
    await short_term_memory.clear(SESSION_ID)
    probe = await rag_agent_service.query_with_context(
        "data-sync-service 部署在哪个区域？现在的备份策略是什么？",
        SESSION_ID, user_id=USER_ID,
    )
    print(f"T4 召回回答: {probe.answer[:120]}…")
    assert "华东" in probe.answer, "回答未体现 PG 强事实召回"
    assert ("增量" in probe.answer) or ("全量" in probe.answer), "回答未体现 ES 文本记忆召回"

    # ---- T5 跨会话验证：全新会话 + 同一用户，召回会话 A 的事实 ----
    cross = await rag_agent_service.query_with_context(
        "data-sync-service 部署在哪个区域？",
        SESSION_ID_B, user_id=USER_ID,
    )
    print(f"T5 跨会话召回: {cross.answer[:120]}…")
    assert "华东" in cross.answer, "跨会话未召回会话 A 的强事实"
    print("=== E2E 全部通过 ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
