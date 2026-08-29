"""基于 PostgreSQL Outbox 的多实例安全索引 Worker。"""

import os
import socket
import time
import uuid
from threading import Event, Thread
from typing import Any

from loguru import logger

from app.config import config
from app.services.document_splitter_service import document_splitter_service
from app.services.es_store_manager import es_store_manager
from app.services.knowledge_repository import knowledge_repository
from app.services.vector_store_manager import vector_store_manager


class PostgresIndexWorker:
    def __init__(self) -> None:
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_audit = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="postgres-index-worker", daemon=True)
        self._thread.start()
        logger.info(f"PostgreSQL 索引 Worker 已启动: {self.worker_id}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = knowledge_repository.claim_job(self.worker_id)
                if job is not None:
                    self._process(job)
                    continue
                self._audit_if_due()
            except Exception as exc:
                logger.exception(f"索引 Worker 循环异常: {exc}")
            self._stop.wait(config.index_worker_poll_seconds)

    def _process(self, claimed: dict[str, Any]) -> None:
        job = knowledge_repository.get_job_document(int(claimed["id"]))
        if job.get("worker_id") != self.worker_id or job.get("status") != "processing":
            return
        lease_stop = Event()
        lease_thread = Thread(
            target=self._renew_lease,
            args=(int(job["id"]), lease_stop),
            name=f"index-lease-{job['id']}",
            daemon=True,
        )
        lease_thread.start()
        try:
            if job["event_type"] == "delete":
                self._process_delete(job)
            else:
                self._process_upsert(job)
        except Exception as exc:
            logger.exception(f"索引任务失败: {job['public_id']}, {exc}")
            knowledge_repository.fail_job(int(job["id"]), str(exc), self.worker_id)
        finally:
            lease_stop.set()
            lease_thread.join(timeout=2)

    def _renew_lease(self, job_id: int, stop: Event) -> None:
        interval = max(1.0, config.index_worker_lease_seconds / 3)
        while not stop.wait(interval):
            try:
                if not knowledge_repository.renew_lease(job_id, self.worker_id):
                    return
            except Exception as exc:
                logger.warning(f"索引任务续租失败: job={job_id}, {exc}")

    def _process_upsert(self, job: dict[str, Any]) -> None:
        if job["deleted_at"] is not None or int(job["current_version"]) != int(job["document_version"]):
            knowledge_repository.complete_superseded(int(job["id"]), self.worker_id)
            return

        document_id = str(job["document_public_id"])
        content_hash = str(job["current_content_hash"])
        index_version = str(uuid.uuid4())
        source = f"postgresql://knowledge/{document_id}"
        documents = document_splitter_service.split_document(str(job["content"]), source)
        chunk_ids = [str(uuid.uuid4()) for _ in documents]
        for document in documents:
            document.metadata.update({
                "document_id": document_id,
                "source_scope": "postgresql:knowledge",
                "source_path": str(job["source_path"]),
                "content_hash": content_hash,
                "_index_version": index_version,
                "_index_task_id": str(job["public_id"]),
                "_file_name": str(job["title"]),
            })

        vector_written = False
        es_written = False
        try:
            if documents:
                vector_written = True
                vector_store_manager.add_documents(documents, chunk_ids)
                es_store_manager.add_documents(documents, chunk_ids)
                es_written = True
            if not knowledge_repository.is_current_version(
                int(job["document_id"]), int(job["document_version"]), deleted=False
            ):
                vector_store_manager.delete_by_ids(chunk_ids)
                es_store_manager.delete_by_ids(chunk_ids)
                knowledge_repository.complete_superseded(int(job["id"]), self.worker_id)
                return
            vector_store_manager.delete_old_document_versions(document_id, index_version)
            es_store_manager.delete_old_document_versions(document_id, index_version)
            published = knowledge_repository.commit_registry_and_job(
                int(job["id"]), int(job["document_id"]), int(job["document_version"]),
                content_hash, index_version, chunk_ids,
                self.worker_id,
            )
            if not published:
                logger.info(f"索引候选版本已被更新文档取代: {job['public_id']}")
        except Exception:
            if vector_written and not es_written:
                try:
                    vector_store_manager.delete_by_ids(chunk_ids)
                except Exception as cleanup_error:
                    logger.warning(f"Milvus 候选版本清理失败: {cleanup_error}")
                try:
                    es_store_manager.delete_by_ids(chunk_ids)
                except Exception as cleanup_error:
                    logger.warning(f"ES 候选版本清理失败: {cleanup_error}")
            raise

    def _process_delete(self, job: dict[str, Any]) -> None:
        if not knowledge_repository.is_current_version(
            int(job["document_id"]), int(job["document_version"]), deleted=True
        ):
            knowledge_repository.complete_superseded(int(job["id"]), self.worker_id)
            return
        document_id = str(job["document_public_id"])
        vector_store_manager.delete_by_document_id(document_id)
        es_store_manager.delete_by_document_id(document_id)
        knowledge_repository.complete_delete(
            int(job["id"]), int(job["document_id"]), self.worker_id,
        )

    def _audit_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_audit < config.index_repair_interval_seconds:
            return
        self._last_audit = now
        for state in knowledge_repository.registry_audit_candidates(config.index_repair_batch_size):
            try:
                if state.get("index_version") is None:
                    knowledge_repository.enqueue_repair(str(state["public_id"]))
                    continue
                args = (
                    str(state["public_id"]), str(state["content_hash"]),
                    str(state["index_version"]),
                )
                expected = int(state["chunk_count"])
                if (
                    vector_store_manager.count_committed_version(*args) != expected
                    or es_store_manager.count_committed_version(*args) != expected
                ):
                    knowledge_repository.enqueue_repair(str(state["public_id"]))
            except Exception as exc:
                logger.warning(f"索引一致性巡检失败: {state.get('public_id')}, {exc}")
            finally:
                knowledge_repository.mark_registry_checked(int(state["id"]))


postgres_index_worker = PostgresIndexWorker()
