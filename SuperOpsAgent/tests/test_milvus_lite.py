from pathlib import Path

from app.config import Settings, config
from app.core.es_client import EsClientManager
from app.core.milvus_client import MilvusClientManager


def test_milvus_lite_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.milvus_lite_path == "./data/milvus.db"
    assert settings.milvus_lite_db_name == "default"
    assert settings.es_analyzer == "standard"
    assert settings.es_search_analyzer == "standard"


def test_milvus_lite_creates_collection(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "milvus-test.db"
    monkeypatch.setattr(config, "milvus_lite_path", str(database_path))
    monkeypatch.setattr(config, "milvus_lite_db_name", "default")

    manager = MilvusClientManager()
    try:
        client = manager.connect()

        assert "biz" in client.list_collections()
        assert manager.health_check() is True
        assert manager.get_collection().indexes[0].params["index_type"] == "FLAT"
    finally:
        manager.close()


def test_elasticsearch_index_uses_configured_analyzers(monkeypatch) -> None:
    monkeypatch.setattr(config, "es_analyzer", "standard")
    monkeypatch.setattr(config, "es_search_analyzer", "standard")

    content_mapping = EsClientManager()._index_body()["mappings"]["properties"]["content"]

    assert content_mapping["analyzer"] == "standard"
    assert content_mapping["search_analyzer"] == "standard"
