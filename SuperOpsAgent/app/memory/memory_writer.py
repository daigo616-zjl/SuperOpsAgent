"""长期记忆持久化：PG Outbox 入队 + 后台 Worker 消费

请求内同步入队（毫秒级 INSERT，不阻塞响应），Worker 线程异步执行
LLM 抽取与三路写入；进程重启不丢任务，多实例安全（skip locked）。
"""

import os
import socket
import uuid
from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from typing import Any

from loguru import logger
from sqlalchemy import text

from app.config import config
from app.core.postgres import postgres_manager
from app.memory.es_memory_store import es_memory_store
from app.memory.extractor import memory_extractor
from app.memory.long_term_repository import long_term_repository
from app.memory.milvus_memory_store import milvus_memory_store


class MemoryWriter:
    def __init__(self) -> None:
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._stop = Event()
        self._thread: Thread | None = None

    # ---- 入队（请求路径内调用，同步毫秒级） ----

    def enqueue_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: str,
        summary: str = "",
    ) -> bool:
        if not assistant_message.strip():
            return False
        try:
            with postgres_manager.engine.begin() as connection:
                connection.execute(text("""
                    insert into rag_memory_jobs
                        (job_id, session_id, user_id, user_message,
                         assistant_message, summary)
                    values (:job_id, :session_id, :user_id, :user_message,
                            :assistant_message, :summary)
                """), {
                    "job_id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_message": user_message[:8000],
                    "assistant_message": assistant_message[:8000],
                    "summary": (summary or "")[:4000],
                })
            return True
        except Exception as exc:
            logger.warning(f"长期记忆任务入队失败（不影响主链路）: {exc}")
            return False

    # ---- Worker ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="memory-write-worker", daemon=True)
        self._thread.start()
        logger.info(f"长期记忆写入 Worker 已启动: {self.worker_id}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._claim_job()
                if job is not None:
                    self._process(job)
                    continue
            except Exception as exc:
                logger.exception(f"长期记忆 Worker 循环异常: {exc}")
            self._stop.wait(config.memory_worker_poll_seconds)

    def _claim_job(self) -> dict[str, Any] | None:
        lease_cutoff = datetime.now(UTC) - timedelta(seconds=config.memory_worker_lease_seconds)
        with postgres_manager.engine.begin() as connection:
            connection.execute(text("""
                update rag_memory_jobs
                set status = case when attempts >= :max_attempts then 'dead' else 'pending' end,
                    lease_until = null, updated_at = now(),
                    last_error = coalesce(last_error, 'worker lease expired')
                where status = 'processing' and lease_until < :lease_cutoff
            """), {"lease_cutoff": lease_cutoff, "max_attempts": config.memory_worker_max_attempts})
            row = connection.execute(text("""
                with candidate as (
                    select id from rag_memory_jobs
                    where status = 'pending' and attempts < :max_attempts
                    order by id
                    for update skip locked
                    limit 1
                )
                update rag_memory_jobs j
                set status = 'processing',
                    attempts = attempts + 1,
                    lease_until = now() + (:lease_seconds * interval '1 second'),
                    updated_at = now()
                from candidate
                where j.id = candidate.id
                returning j.*
            """), {
                "max_attempts": config.memory_worker_max_attempts,
                "lease_seconds": config.memory_worker_lease_seconds,
            }).first()
        return dict(row._mapping) if row is not None else None

    def _process(self, job: dict[str, Any]) -> None:
        job_id = int(job["id"])
        try:
            import asyncio

            items = asyncio.run(
                memory_extractor.extract(
                    str(job["user_message"]),
                    str(job["assistant_message"]),
                    str(job["summary"] or ""),
                )
            )
            self._route_items(str(job["user_id"]), items)
            with postgres_manager.engine.begin() as connection:
                connection.execute(text("""
                    update rag_memory_jobs
                    set status = 'done', updated_at = now()
                    where id = :job_id
                """), {"job_id": job_id})
            if items:
                logger.info(
                    f"长期记忆写入完成: job={job_id}, fact/text/semantic 共 {len(items)} 条"
                )
        except Exception as exc:
            logger.warning(f"长期记忆任务失败: job={job_id}, error={exc}")
            self._fail_job(job_id, str(exc))

    def _route_items(self, user_id: str, items: list) -> None:
        for item in items:
            if item.type == "fact":
                long_term_repository.upsert_fact(
                    user_id, item.content, subject=item.subject,
                )
            elif item.type == "text":
                es_memory_store.index(user_id, item.content, subject=item.subject)
            elif item.type == "semantic":
                milvus_memory_store.upsert(user_id, item.content, subject=item.subject)

    def _fail_job(self, job_id: int, error: str) -> None:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(text("""
                select attempts from rag_memory_jobs
                where id = :job_id and status = 'processing' for update
            """), {"job_id": job_id}).first()
            if row is None:
                return
            dead = int(row.attempts) >= config.memory_worker_max_attempts
            delay = min(3600, 2 ** max(0, int(row.attempts) - 1))
            connection.execute(text("""
                update rag_memory_jobs
                set status = :status,
                    lease_until = now() + (:delay * interval '1 second'),
                    last_error = :error, updated_at = now()
                where id = :job_id and status = 'processing'
            """), {
                "job_id": job_id,
                "status": "dead" if dead else "pending",
                "delay": delay,
                "error": error[:4000],
            })


memory_write_worker = MemoryWriter()
