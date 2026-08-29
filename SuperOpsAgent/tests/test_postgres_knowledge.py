import json
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.documents import Document
from starlette.datastructures import UploadFile

from app.api.file import upload_file
from app.services.document_hash import compute_content_hash
from app.services.postgres_index_worker import PostgresIndexWorker


def test_content_hash_normalizes_only_line_endings() -> None:
    assert compute_content_hash("a\r\nb\r") == compute_content_hash("a\nb\n")
    assert compute_content_hash("a b") != compute_content_hash("a  b")


@pytest.mark.asyncio
async def test_upload_writes_authoritative_content_without_local_file(monkeypatch) -> None:
    captured = {}

    def upsert(title, source_path, content):
        captured.update(title=title, source_path=source_path, content=content)
        return {"public_id": "doc-1"}, "unchanged"

    monkeypatch.setattr(
        "app.api.file.knowledge_repository.upsert_uploaded_document", upsert,
    )
    response = await upload_file(UploadFile(BytesIO(b"# hello\n"), filename="hello.md"))
    payload = json.loads(response.body)

    assert captured == {
        "title": "hello.md", "source_path": "hello.md", "content": "# hello\n",
    }
    assert payload["data"]["index_status"] == "unchanged"


def test_schema_and_claim_query_support_transactional_multi_instance_worker() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = (root / "migrations" / "001_postgres_knowledge.sql").read_text(encoding="utf-8")
    repository = (root / "app" / "services" / "knowledge_repository.py").read_text(
        encoding="utf-8"
    )
    main = (root / "app" / "main.py").read_text(encoding="utf-8")

    assert "knowledge_documents" in schema
    assert "knowledge_index_registry" in schema
    assert "knowledge_index_jobs" in schema
    assert "for update skip locked" in repository.lower()
    assert "vector_index_service" not in main
    assert "knowledge_docs_dir" not in main


def test_worker_dual_writes_and_publishes_registry(monkeypatch) -> None:
    events = []

    class Repository:
        def is_current_version(self, document_id, version, deleted):
            return True

        def commit_registry_and_job(self, *args):
            events.append("publish")
            return True

    class Store:
        def __init__(self, name):
            self.name = name

        def add_documents(self, documents, ids):
            events.append(f"{self.name}:add")
            assert documents[0].metadata["source_scope"] == "postgresql:knowledge"

        def delete_old_document_versions(self, document_id, keep_version):
            events.append(f"{self.name}:cleanup")

    monkeypatch.setattr("app.services.postgres_index_worker.knowledge_repository", Repository())
    monkeypatch.setattr("app.services.postgres_index_worker.vector_store_manager", Store("milvus"))
    monkeypatch.setattr("app.services.postgres_index_worker.es_store_manager", Store("es"))
    monkeypatch.setattr(
        "app.services.postgres_index_worker.document_splitter_service.split_document",
        lambda content, source: [Document(page_content=content, metadata={"_source": source})],
    )
    job = {
        "id": 9,
        "public_id": "task-1",
        "document_id": 4,
        "document_public_id": "doc-1",
        "document_version": 2,
        "current_version": 2,
        "current_content_hash": "a" * 64,
        "deleted_at": None,
        "source_path": "runbook.md",
        "title": "Runbook",
        "content": "hello",
    }

    PostgresIndexWorker()._process_upsert(job)

    assert events == [
        "milvus:add", "es:add", "milvus:cleanup", "es:cleanup", "publish",
    ]


def test_consistency_audit_enqueues_repair_and_rotates_cursor(monkeypatch) -> None:
    repaired = []
    checked = []

    class Repository:
        def registry_audit_candidates(self, limit):
            return [{
                "id": 7, "public_id": "doc-7", "content_hash": "b" * 64,
                "index_version": "index-1", "chunk_count": 2,
            }]

        def enqueue_repair(self, public_id):
            repaired.append(public_id)

        def mark_registry_checked(self, document_id):
            checked.append(document_id)

    class Store:
        def count_committed_version(self, *args):
            return 1

    monkeypatch.setattr("app.services.postgres_index_worker.knowledge_repository", Repository())
    monkeypatch.setattr("app.services.postgres_index_worker.vector_store_manager", Store())
    monkeypatch.setattr("app.services.postgres_index_worker.es_store_manager", Store())
    worker = PostgresIndexWorker()
    worker._audit_if_due()

    assert repaired == ["doc-7"]
    assert checked == [7]
