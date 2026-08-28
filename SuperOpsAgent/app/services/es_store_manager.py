"""Elasticsearch 存储管理器模块"""

from datetime import datetime, UTC

from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.core.es_client import es_client_manager


class EsStoreManager:
    """Elasticsearch 存储管理器"""

    def _build_source(self, document: Document) -> dict:
        metadata = document.metadata or {}
        return {
            "content": document.page_content,
            "source": metadata.get("_source"),
            "file_name": metadata.get("_file_name"),
            "extension": metadata.get("_extension"),
            "h1": metadata.get("h1"),
            "h2": metadata.get("h2"),
            "h3": metadata.get("h3"),
            "metadata": metadata,
            "index_version": metadata.get("_index_version"),
            "index_task_id": metadata.get("_index_task_id"),
            "indexed_at": datetime.now(UTC).isoformat(),
        }

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        if len(documents) != len(ids):
            raise ValueError("documents 与 ids 数量不一致")

        client = es_client_manager.get_sync_client()
        actions = [
            {
                "_index": config.es_index,
                "_id": chunk_id,
                "_source": self._build_source(document),
            }
            for chunk_id, document in zip(ids, documents)
        ]

        success, errors = bulk(client, actions, raise_on_error=False)
        if errors:
            raise RuntimeError(f"ES bulk 部分失败: {errors[:3]}")

        logger.info(f"批量写入 Elasticsearch 完成: success={success}")

    def delete_by_source(self, file_path: str) -> int:
        client = es_client_manager.get_sync_client()
        response = client.delete_by_query(
            index=config.es_index,
            body={"query": {"term": {"source": file_path}}},
            conflicts="proceed",
            refresh=True,
        )
        deleted = int(response.get("deleted", 0))
        logger.info(f"删除 Elasticsearch 文件旧数据: {file_path}, 删除数量: {deleted}")
        return deleted

    def delete_old_versions(self, file_path: str, keep_version: str) -> int:
        client = es_client_manager.get_sync_client()
        response = client.delete_by_query(
            index=config.es_index,
            body={
                "query": {
                    "bool": {
                        "filter": [{"term": {"source": file_path}}],
                        "must_not": [{"term": {"index_version": keep_version}}],
                    }
                }
            },
            conflicts="proceed",
            refresh=True,
        )
        deleted = int(response.get("deleted", 0))
        logger.info(f"删除 Elasticsearch 旧版本: {file_path}, 保留版本={keep_version}, 数量={deleted}")
        return deleted

    def delete_by_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0

        client = es_client_manager.get_sync_client()
        actions = [
            {
                "_op_type": "delete",
                "_index": config.es_index,
                "_id": chunk_id,
            }
            for chunk_id in ids
        ]
        _, errors = bulk(client, actions, raise_on_error=False)
        if errors:
            raise RuntimeError(f"按 ID 删除 Elasticsearch 文档时出现部分失败: {errors[:3]}")
        deleted = len(ids)
        logger.info(f"按 ID 删除 Elasticsearch 文档完成: 删除数量={deleted}")
        return deleted


es_store_manager = EsStoreManager()
