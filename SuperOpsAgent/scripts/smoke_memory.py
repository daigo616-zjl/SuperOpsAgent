"""本地中间件冒烟：Redis 短期记忆 + 熔断降级 + PG 强事实 + Milvus 记忆 collection。"""

import asyncio
import sys

import redis.asyncio as aioredis
from loguru import logger
from sqlalchemy import text


async def smoke_redis_short_term() -> None:
    from app.memory.redis_client import redis_client_manager
    from app.memory.short_term import short_term_memory

    await redis_client_manager.connect()
    assert await redis_client_manager.health_check(), "Redis 健康检查失败"

    sid = "smoke-session-1"
    await short_term_memory.clear(sid)

    ok = await short_term_memory.append_turn(sid, "生产 MySQL 主库在哪？", "华东1 可用区A")
    assert ok, "append_turn 失败"
    seq1 = await short_term_memory.next_seq(sid)
    await short_term_memory.append_turn(sid, "从库延迟多少？", "正常情况下小于 100ms")
    seq2 = await short_term_memory.next_seq(sid)

    summary, window = await short_term_memory.build_context(sid)
    assert seq2 > seq1 >= 1, f"seq 异常: {seq1}, {seq2}"
    assert len(window) == 4, f"窗口应为 4 条消息，实际 {len(window)}"
    assert window[-1]["role"] == "assistant"
    history = await short_term_memory.history(sid, limit=2)
    assert len(history) == 2

    client = await short_term_memory._ensure_client()
    ttl = await client.ttl(f"rag:s:{sid}:msgs")
    assert ttl > 0, f"TTL 未续期: {ttl}"
    print(f"[Redis 短期记忆] OK: seq={seq2}, window={len(window)}, ttl={ttl}s, summary={summary!r}")

    await short_term_memory.clear(sid)
    summary2, window2 = await short_term_memory.build_context(sid)
    assert not window2 and not summary2
    print("[Redis clear] OK")


async def smoke_redis_breaker() -> None:
    from app.memory.redis_client import redis_client_manager
    from app.memory.short_term import ShortTermMemory, short_term_memory

    dead = aioredis.from_url(
        "redis://localhost:6390/0",
        socket_timeout=1, socket_connect_timeout=1, decode_responses=True,
    )
    original = redis_client_manager._client
    redis_client_manager._client = dead

    stm = ShortTermMemory()
    ok = await stm.append_turn("smoke-breaker", "q", "a")
    assert not ok and stm.tripped, "故障未触发熔断"
    print("[熔断] OK: 死端口写入失败后 tripped=True, append_turn 返回 False")

    await dead.aclose()
    redis_client_manager._client = original

    stm2 = ShortTermMemory()
    assert await stm2.available(), "恢复后新实例应可用"
    ok2 = await stm2.append_turn("smoke-breaker-2", "q", "a")
    assert ok2, "恢复后写入失败"
    await short_term_memory.clear("smoke-breaker-2")
    print("[熔断恢复] OK")


async def smoke_pg_facts() -> None:
    from app.core.postgres import postgres_manager
    from app.memory.long_term_repository import long_term_repository

    postgres_manager.connect()
    user = "smoke-user-pg"
    with postgres_manager.engine.begin() as conn:
        conn.exec_driver_sql("delete from rag_memory_facts where user_id = 'smoke-user-pg'")

    long_term_repository.upsert_fact(
        user, "生产环境 MySQL 主库在华东1 可用区A",
        subject="数据库", keywords=["mysql", "华东1"],
    )
    long_term_repository.upsert_fact(
        user, "生产环境 MySQL 主库已迁移至华东2 可用区B",
        subject="数据库", keywords=["mysql", "华东2"],
    )

    hits = long_term_repository.match_facts(user, ["mysql", "主库"], limit=20)
    assert any("华东2" in h.content for h in hits), "新事实未召回"
    assert all("华东1" not in h.content for h in hits), "旧事实未置 superseded"
    with postgres_manager.engine.connect() as conn:
        row = conn.execute(text(
            "select status, count(*) from rag_memory_facts where user_id = 'smoke-user-pg' group by status"
        )).all()
    print(f"[PG 强事实] OK: active 召回 {len(hits)} 条, 状态分布 {dict(row)}")


async def smoke_milvus_memory() -> None:
    from app.memory.milvus_memory_store import milvus_memory_store

    user = "smoke-user-vec"
    content = "夜巡脚本每天凌晨 2 点执行 mysqldump 全量备份"
    milvus_memory_store.upsert(user, content, subject="备份")
    hits = milvus_memory_store.search(user, "数据库备份是怎么做的")
    assert hits, "Milvus 语义召回为空"
    print(f"[Milvus 记忆] OK: 召回 {len(hits)} 条, top score={hits[0].score:.3f}, content={hits[0].content[:20]}…")


async def smoke_es_memory() -> None:
    from app.config import config
    from app.core.es_client import es_client_manager
    from app.memory.es_memory_store import es_memory_store

    await es_client_manager.connect()
    es_memory_store.ensure_index()

    user = "smoke-user-es"
    content = "上次处理 OOM 事故的结论是堆外内存泄漏，用 pmap 排查到 native 分配"
    es_memory_store.index(user, content, subject="事故复盘")
    es_memory_store.index(user, content, subject="事故复盘")  # 幂等：重复写入

    client = es_client_manager.get_sync_client()
    client.indices.refresh(index=config.es_memory_index)

    hits = es_memory_store.search(user, "OOM 事故当时结论是什么")
    assert hits, "ES BM25 召回为空"
    count = client.count(
        index=config.es_memory_index,
        body={"query": {"term": {"user_id": user}}},
    )["count"]
    assert count == 1, f"幂等失败: {count} 条 doc"
    print(f"[ES 记忆] OK: 召回 {len(hits)} 条, top score={hits[0].score:.3f}, doc 数={count}")

    client.delete_by_query(
        index=config.es_memory_index,
        body={"query": {"term": {"user_id": user}}},
        refresh=True,
    )


async def main() -> None:
    await smoke_redis_short_term()
    await smoke_redis_breaker()
    await smoke_pg_facts()
    await smoke_milvus_memory()
    await smoke_es_memory()
    print("=== 全部冒烟通过 ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        logger.exception(f"冒烟失败: {exc}")
        sys.exit(1)
