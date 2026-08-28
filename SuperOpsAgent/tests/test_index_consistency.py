from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.index_task_store import IndexTaskStore
from app.services.vector_index_service import VectorIndexService


def test_index_task_store_persists_status_and_compensation(tmp_path: Path) -> None:
    store = IndexTaskStore(tmp_path / "tasks.json")
    task = store.create(file_path="/tmp/a.md", staged_file_path=None, version="v1")
    store.update(task["task_id"], status="partial_success", error="es unavailable")
    store.add_compensation(task["task_id"], {"store": "milvus", "operation": "delete_by_ids"})

    reloaded = IndexTaskStore(tmp_path / "tasks.json").get("v1")
    assert reloaded is not None
    assert reloaded["status"] == "partial_success"
    assert reloaded["compensations"][0]["operation"] == "delete_by_ids"


def test_index_writes_new_version_before_deleting_old_data(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("new content", encoding="utf-8")
    service = VectorIndexService()
    service.task_store = IndexTaskStore(tmp_path / "tasks.json")
    captured: dict[str, object] = {}

    class Store:
        def add_documents(self, documents, ids):
            captured["documents"] = documents
            captured["ids"] = ids

        def delete_old_versions(self, file_path, keep_version):
            captured.setdefault("deletes", []).append((file_path, keep_version))

        def delete_by_ids(self, ids):
            raise AssertionError("成功路径不应回滚")

    monkeypatch.setattr("app.services.vector_index_service.vector_store_manager", Store())
    monkeypatch.setattr("app.services.vector_index_service.es_store_manager", Store())
    monkeypatch.setattr(
        "app.services.vector_index_service.document_splitter_service.split_document",
        lambda content, path: [Document(page_content=content, metadata={"_source": path})],
    )

    task = service.index_single_file(str(source))

    assert task["status"] == "success"
    assert captured["documents"][0].metadata["_index_version"] == task["index_version"]
    assert len(captured["deletes"]) == 2


def test_es_failure_keeps_old_file_and_records_cleanup_failure(tmp_path: Path, monkeypatch) -> None:
    final_path = tmp_path / "knowledge.md"
    final_path.write_text("old content", encoding="utf-8")
    staged_path = tmp_path / ".knowledge.md.uploading"
    staged_path.write_text("new content", encoding="utf-8")
    service = VectorIndexService()
    service.task_store = IndexTaskStore(tmp_path / "tasks.json")

    class VectorStore:
        def add_documents(self, documents, ids):
            pass

        def delete_old_versions(self, file_path, keep_version):
            raise AssertionError("双写未成功时不应删除旧版本")

        def delete_by_ids(self, ids):
            raise RuntimeError("milvus cleanup unavailable")

    class EsStore:
        def add_documents(self, documents, ids):
            raise RuntimeError("es unavailable")

        def delete_old_versions(self, file_path, keep_version):
            raise AssertionError("双写未成功时不应删除旧版本")

        def delete_by_ids(self, ids):
            pass

    monkeypatch.setattr("app.services.vector_index_service.vector_store_manager", VectorStore())
    monkeypatch.setattr("app.services.vector_index_service.es_store_manager", EsStore())
    monkeypatch.setattr(
        "app.services.vector_index_service.document_splitter_service.split_document",
        lambda content, path: [Document(page_content=content, metadata={"_source": path})],
    )

    with pytest.raises(RuntimeError):
        service.index_single_file(str(final_path), staged_file_path=str(staged_path))

    assert final_path.read_text(encoding="utf-8") == "old content"
    assert staged_path.exists()
    failed = service.task_store.list({"failed"})[0]
    assert failed["status"] == "failed"
    assert failed["compensations"][0]["store"] == "milvus"
