"""PostgreSQL 权威文档、索引注册表和事务 Outbox 仓储。"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.config import config
from app.core.postgres import postgres_manager
from app.services.document_hash import compute_content_hash


class DocumentConflictError(RuntimeError):
    pass


class DocumentNotFoundError(RuntimeError):
    pass


class KnowledgeRepository:
    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        return dict(row._mapping)

    def _enqueue(
        self, connection: Any, document_id: int, version: int,
        content_hash: str, event_type: str, *, requeue_dead: bool = False,
    ) -> None:
        connection.execute(text("""
            insert into knowledge_index_jobs
                (public_id, document_id, event_type, document_version, content_hash)
            values (:public_id, :document_id, :event_type, :version, :content_hash)
            on conflict (document_id, document_version, event_type) do update
            set status = case
                    when knowledge_index_jobs.status = 'processing' then 'processing'
                    when knowledge_index_jobs.status = 'dead' and not :requeue_dead then 'dead'
                    else 'pending'
                end,
                attempts = case
                    when knowledge_index_jobs.status in ('succeeded', 'superseded')
                         or (knowledge_index_jobs.status = 'dead' and :requeue_dead) then 0
                    else knowledge_index_jobs.attempts
                end,
                available_at = now(), last_error = null, updated_at = now()
        """), {
            "public_id": uuid.uuid4(), "document_id": document_id,
            "event_type": event_type, "version": version, "content_hash": content_hash,
            "requeue_dead": requeue_dead,
        })

    def create_document(self, title: str, source_path: str, content: str) -> dict[str, Any]:
        content_hash = compute_content_hash(content)
        public_id = uuid.uuid4()
        try:
            with postgres_manager.engine.begin() as connection:
                row = connection.execute(text("""
                    insert into knowledge_documents
                        (public_id, title, source_path, content, content_hash)
                    values (:public_id, :title, :source_path, :content, :content_hash)
                    returning *
                """), {
                    "public_id": public_id, "title": title.strip(),
                    "source_path": source_path.strip(), "content": content,
                    "content_hash": content_hash,
                }).one()
                document = self._row(row)
                self._enqueue(connection, document["id"], document["version"], content_hash, "upsert")
                return document
        except IntegrityError as exc:
            raise DocumentConflictError(f"文档路径已存在: {source_path}") from exc

    def upsert_uploaded_document(
        self, title: str, source_path: str, content: str,
    ) -> tuple[dict[str, Any], str]:
        """兼容上传入口；原文只写 PostgreSQL，不再写本地文件。"""
        normalized_title = title.strip()
        normalized_path = source_path.strip()
        content_hash = compute_content_hash(content)
        try:
            with postgres_manager.engine.begin() as connection:
                row = connection.execute(text("""
                    select * from knowledge_documents
                    where lower(source_path) = lower(:source_path) and deleted_at is null
                    for update
                """), {"source_path": normalized_path}).first()
                if row is None:
                    inserted = connection.execute(text("""
                        insert into knowledge_documents
                            (public_id, title, source_path, content, content_hash)
                        values (:public_id, :title, :source_path, :content, :content_hash)
                        returning *
                    """), {
                        "public_id": uuid.uuid4(), "title": normalized_title,
                        "source_path": normalized_path, "content": content,
                        "content_hash": content_hash,
                    }).one()
                    document = self._row(inserted)
                    self._enqueue(
                        connection, document["id"], document["version"], content_hash, "upsert",
                    )
                    return document, "queued"

                current = self._row(row)
                if current["content_hash"] == content_hash:
                    return current, "unchanged"
                updated = connection.execute(text("""
                    update knowledge_documents
                    set title = :title, content = :content, content_hash = :content_hash,
                        version = version + 1, updated_at = now()
                    where id = :id
                    returning *
                """), {
                    "id": current["id"], "title": normalized_title,
                    "content": content, "content_hash": content_hash,
                }).one()
                document = self._row(updated)
                self._enqueue(
                    connection, document["id"], document["version"], content_hash, "upsert",
                )
                return document, "queued"
        except IntegrityError as exc:
            raise DocumentConflictError(f"文档路径并发写入冲突: {source_path}") from exc

    def get_document(self, public_id: str, include_deleted: bool = False) -> dict[str, Any]:
        query = "select * from knowledge_documents where public_id = :public_id"
        if not include_deleted:
            query += " and deleted_at is null"
        with postgres_manager.engine.connect() as connection:
            row = connection.execute(text(query), {"public_id": public_id}).first()
        if row is None:
            raise DocumentNotFoundError(f"文档不存在: {public_id}")
        return self._row(row)

    def list_documents(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        with postgres_manager.engine.connect() as connection:
            total = connection.execute(text(
                "select count(*) from knowledge_documents where deleted_at is null"
            )).scalar_one()
            rows = connection.execute(text("""
                select d.*, r.status as index_status, r.indexed_at,
                       coalesce(r.chunk_count, 0) as chunk_count
                from knowledge_documents d
                left join knowledge_index_registry r on r.document_id = d.id
                where d.deleted_at is null
                order by d.updated_at desc, d.id desc
                limit :limit offset :offset
            """), {"limit": limit, "offset": offset}).all()
        return {"items": [self._row(row) for row in rows], "total": int(total)}

    def update_document(
        self, public_id: str, expected_version: int, title: str,
        source_path: str, content: str,
    ) -> dict[str, Any]:
        content_hash = compute_content_hash(content)
        try:
            with postgres_manager.engine.begin() as connection:
                current_row = connection.execute(text("""
                    select * from knowledge_documents
                    where public_id = :public_id and deleted_at is null
                    for update
                """), {"public_id": public_id}).first()
                if current_row is None:
                    raise DocumentNotFoundError(f"文档不存在: {public_id}")
                current = self._row(current_row)
                if int(current["version"]) != expected_version:
                    raise DocumentConflictError(
                        f"文档版本冲突，当前版本为 {current['version']}"
                    )
                if (
                    current["title"] == title.strip()
                    and current["source_path"] == source_path.strip()
                    and current["content_hash"] == content_hash
                ):
                    return current
                row = connection.execute(text("""
                    update knowledge_documents
                    set title = :title, source_path = :source_path, content = :content,
                        content_hash = :content_hash, version = version + 1, updated_at = now()
                    where id = :id
                    returning *
                """), {
                    "id": current["id"], "title": title.strip(),
                    "source_path": source_path.strip(), "content": content,
                    "content_hash": content_hash,
                }).one()
                document = self._row(row)
                self._enqueue(connection, document["id"], document["version"], content_hash, "upsert")
                return document
        except IntegrityError as exc:
            raise DocumentConflictError(f"文档路径已存在: {source_path}") from exc

    def delete_document(self, public_id: str, expected_version: int) -> dict[str, Any]:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(text("""
                update knowledge_documents
                set deleted_at = now(), version = version + 1, updated_at = now()
                where public_id = :public_id and deleted_at is null and version = :expected_version
                returning *
            """), {"public_id": public_id, "expected_version": expected_version}).first()
            if row is None:
                existing = connection.execute(text(
                    "select version from knowledge_documents where public_id = :public_id and deleted_at is null"
                ), {"public_id": public_id}).first()
                if existing is None:
                    raise DocumentNotFoundError(f"文档不存在: {public_id}")
                raise DocumentConflictError(f"文档版本冲突，当前版本为 {existing.version}")
            document = self._row(row)
            self._enqueue(
                connection, document["id"], document["version"],
                document["content_hash"], "delete",
            )
            return document

    def enqueue_repair(self, public_id: str, force: bool = False) -> dict[str, Any]:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(text("""
                select * from knowledge_documents
                where public_id = :public_id and deleted_at is null for update
            """), {"public_id": public_id}).first()
            if row is None:
                raise DocumentNotFoundError(f"文档不存在: {public_id}")
            document = self._row(row)
            self._enqueue(
                connection, document["id"], document["version"],
                document["content_hash"], "repair", requeue_dead=force,
            )
            return document

    def claim_job(self, worker_id: str) -> dict[str, Any] | None:
        lease_cutoff = datetime.now(UTC) - timedelta(seconds=config.index_worker_lease_seconds)
        with postgres_manager.engine.begin() as connection:
            connection.execute(text("""
                update knowledge_index_jobs
                set status = case when attempts >= :max_attempts then 'dead' else 'retry' end,
                    available_at = now(), worker_id = null, locked_at = null,
                    last_error = coalesce(last_error, 'worker lease expired'), updated_at = now()
                where status = 'processing' and locked_at < :lease_cutoff
            """), {"lease_cutoff": lease_cutoff, "max_attempts": config.index_worker_max_attempts})
            row = connection.execute(text("""
                with candidate as (
                    select j.id
                    from knowledge_index_jobs j
                    where j.status in ('pending', 'retry')
                      and j.available_at <= now()
                      and j.attempts < :max_attempts
                      and not exists (
                          select 1 from knowledge_index_jobs active
                          where active.document_id = j.document_id
                            and active.status = 'processing'
                      )
                    order by j.available_at, j.id
                    for update skip locked
                    limit 1
                )
                update knowledge_index_jobs j
                set status = 'processing', attempts = attempts + 1,
                    worker_id = :worker_id, locked_at = now(), updated_at = now()
                from candidate
                where j.id = candidate.id
                returning j.*
            """), {
                "worker_id": worker_id, "max_attempts": config.index_worker_max_attempts,
            }).first()
        return self._row(row) if row is not None else None

    def get_job_document(self, job_id: int) -> dict[str, Any]:
        with postgres_manager.engine.connect() as connection:
            row = connection.execute(text("""
                select j.*, d.public_id as document_public_id, d.title, d.source_path,
                       d.content, d.content_hash as current_content_hash,
                       d.version as current_version, d.deleted_at
                from knowledge_index_jobs j
                join knowledge_documents d on d.id = j.document_id
                where j.id = :job_id
            """), {"job_id": job_id}).one()
        return self._row(row)

    def is_current_version(self, document_id: int, version: int, deleted: bool) -> bool:
        with postgres_manager.engine.connect() as connection:
            row = connection.execute(text("""
                select version, deleted_at from knowledge_documents where id = :document_id
            """), {"document_id": document_id}).first()
        return bool(
            row is not None and int(row.version) == version
            and (row.deleted_at is not None) == deleted
        )

    def renew_lease(self, job_id: int, worker_id: str) -> bool:
        with postgres_manager.engine.begin() as connection:
            result = connection.execute(text("""
                update knowledge_index_jobs set locked_at = now(), updated_at = now()
                where id = :job_id and status = 'processing' and worker_id = :worker_id
            """), {"job_id": job_id, "worker_id": worker_id})
            return result.rowcount == 1

    @staticmethod
    def _owns_job(connection: Any, job_id: int, worker_id: str) -> bool:
        row = connection.execute(text("""
            select 1 from knowledge_index_jobs
            where id = :job_id and status = 'processing' and worker_id = :worker_id
            for update
        """), {"job_id": job_id, "worker_id": worker_id}).first()
        return row is not None

    def commit_registry_and_job(
        self, job_id: int, document_id: int, document_version: int,
        content_hash: str, index_version: str, chunk_ids: list[str], worker_id: str,
    ) -> bool:
        with postgres_manager.engine.begin() as connection:
            if not self._owns_job(connection, job_id, worker_id):
                return False
            current = connection.execute(text("""
                select version, content_hash, deleted_at
                from knowledge_documents where id = :document_id for update
            """), {"document_id": document_id}).one()
            if (
                current.deleted_at is not None
                or int(current.version) != document_version
                or current.content_hash != content_hash
            ):
                self._finish_job(connection, job_id, "superseded", worker_id)
                return False
            connection.execute(text("""
                insert into knowledge_index_registry
                    (document_id, content_hash, document_version, index_version,
                     chunk_ids, chunk_count, status, last_error, indexed_at, updated_at)
                values (:document_id, :content_hash, :document_version, :index_version,
                        cast(:chunk_ids as jsonb), :chunk_count, 'ready', null, now(), now())
                on conflict (document_id) do update
                set content_hash = excluded.content_hash,
                    document_version = excluded.document_version,
                    index_version = excluded.index_version,
                    chunk_ids = excluded.chunk_ids,
                    chunk_count = excluded.chunk_count,
                    status = 'ready', last_error = null,
                    indexed_at = now(), updated_at = now()
            """), {
                "document_id": document_id, "content_hash": content_hash,
                "document_version": document_version, "index_version": index_version,
                "chunk_ids": json.dumps(chunk_ids), "chunk_count": len(chunk_ids),
            })
            self._finish_job(connection, job_id, "succeeded", worker_id)
            return True

    def complete_delete(self, job_id: int, document_id: int, worker_id: str) -> bool:
        with postgres_manager.engine.begin() as connection:
            if not self._owns_job(connection, job_id, worker_id):
                return False
            connection.execute(text(
                "delete from knowledge_index_registry where document_id = :document_id"
            ), {"document_id": document_id})
            self._finish_job(connection, job_id, "succeeded", worker_id)
            return True

    def complete_superseded(self, job_id: int, worker_id: str) -> None:
        with postgres_manager.engine.begin() as connection:
            self._finish_job(connection, job_id, "superseded", worker_id)

    @staticmethod
    def _finish_job(connection: Any, job_id: int, status: str, worker_id: str) -> None:
        connection.execute(text("""
            update knowledge_index_jobs
            set status = :status, completed_at = now(), updated_at = now(),
                worker_id = null, locked_at = null
            where id = :job_id and status = 'processing' and worker_id = :worker_id
        """), {"job_id": job_id, "status": status, "worker_id": worker_id})

    def fail_job(self, job_id: int, error: str, worker_id: str) -> None:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(text(
                """select attempts from knowledge_index_jobs
                   where id = :job_id and status = 'processing' and worker_id = :worker_id
                   for update"""
            ), {"job_id": job_id, "worker_id": worker_id}).first()
            if row is None:
                return
            dead = int(row.attempts) >= config.index_worker_max_attempts
            delay = min(3600, 2 ** max(0, int(row.attempts) - 1))
            connection.execute(text("""
                update knowledge_index_jobs
                set status = :status, available_at = now() + (:delay * interval '1 second'),
                    last_error = :error, worker_id = null, locked_at = null, updated_at = now()
                where id = :job_id and status = 'processing' and worker_id = :worker_id
            """), {
                "job_id": job_id, "status": "dead" if dead else "retry",
                "delay": delay, "error": error[:4000],
                "worker_id": worker_id,
            })

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with postgres_manager.engine.connect() as connection:
            rows = connection.execute(text("""
                select j.public_id as task_id, j.event_type as operation, j.status,
                       j.attempts as retry_count, j.last_error as error,
                       d.public_id as document_id, d.title, d.source_path,
                       j.created_at, j.updated_at
                from knowledge_index_jobs j
                join knowledge_documents d on d.id = j.document_id
                order by j.id desc limit :limit
            """), {"limit": limit}).all()
        return [self._row(row) for row in rows]

    def retry_job(self, public_id: str) -> dict[str, Any]:
        with postgres_manager.engine.begin() as connection:
            row = connection.execute(text("""
                update knowledge_index_jobs
                set status = 'retry', attempts = 0, available_at = now(),
                    last_error = null, worker_id = null, locked_at = null, updated_at = now()
                where public_id = :public_id and status in ('retry', 'dead')
                returning *
            """), {"public_id": public_id}).first()
            if row is None:
                raise DocumentConflictError("任务不存在或当前不可重试")
            return self._row(row)

    def registry_audit_candidates(self, limit: int) -> list[dict[str, Any]]:
        with postgres_manager.engine.connect() as connection:
            rows = connection.execute(text("""
                select d.id, d.public_id, d.content_hash, d.version,
                       r.index_version, r.chunk_count
                from knowledge_documents d
                left join knowledge_index_registry r on r.document_id = d.id
                where d.deleted_at is null
                order by coalesce(r.checked_at, 'epoch'::timestamptz), d.id
                limit :limit
            """), {"limit": limit}).all()
        return [self._row(row) for row in rows]

    def mark_registry_checked(self, document_id: int) -> None:
        with postgres_manager.engine.begin() as connection:
            connection.execute(text("""
                update knowledge_index_registry set checked_at = now()
                where document_id = :document_id
            """), {"document_id": document_id})


knowledge_repository = KnowledgeRepository()
