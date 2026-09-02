"""长期记忆：Milvus 非结构化语义存储（仅做语义补充，最低优先级）

独立于业务 collection biz（biz 由索引 worker 管理版本生命周期）。
使用 IP 度量：text-embedding-v4 向量已归一化，内积即余弦相似度，
可直接与 memory_vec_min_score 阈值比较。id = content_hash 天然幂等。
"""

from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from app.config import config
from app.core.milvus_client import milvus_manager
from app.memory.models import MemoryHit, content_hash
from app.services.vector_embedding_service import vector_embedding_service


class MilvusMemoryStore:
    ID_MAX_LENGTH = 64
    CONTENT_MAX_LENGTH = 8000
    USER_ID_MAX_LENGTH = 128
    SUBJECT_MAX_LENGTH = 512

    _collection: Collection | None = None

    @property
    def collection_name(self) -> str:
        return config.milvus_memory_collection

    def _ensure_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection

        # ORM Collection/utility 依赖 default 别名连接；显式建连，不依赖调用方先初始化
        milvus_manager.connect()
        if not bool(utility.has_collection(self.collection_name)):
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR,
                    max_length=self.ID_MAX_LENGTH, is_primary=True,
                ),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR,
                    dim=1024,
                ),
                FieldSchema(
                    name="content", dtype=DataType.VARCHAR,
                    max_length=self.CONTENT_MAX_LENGTH,
                ),
                FieldSchema(
                    name="user_id", dtype=DataType.VARCHAR,
                    max_length=self.USER_ID_MAX_LENGTH,
                ),
                FieldSchema(
                    name="subject", dtype=DataType.VARCHAR,
                    max_length=self.SUBJECT_MAX_LENGTH,
                ),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="RAG long-term semantic memory",
                enable_dynamic_field=False,
            )
            collection = Collection(name=self.collection_name, schema=schema)
            collection.create_index(
                field_name="vector",
                index_params={"metric_type": "IP", "index_type": "FLAT", "params": {}},
            )
            logger.info(f"创建长期记忆 Milvus collection: {self.collection_name}")
        else:
            collection = Collection(self.collection_name)

        collection.load()
        self._collection = collection
        return collection

    def _escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def upsert(self, user_id: str, content: str, subject: str = "") -> str:
        c_hash = content_hash(user_id, content)
        collection = self._ensure_collection()
        vector = vector_embedding_service.embed_query(content)
        collection.upsert([[c_hash], [vector], [content[: self.CONTENT_MAX_LENGTH]], [user_id], [subject]])
        return c_hash

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[MemoryHit]:
        if not query.strip():
            return []
        collection = self._ensure_collection()
        vector = vector_embedding_service.embed_query(query)
        results = collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {}},
            limit=top_k,
            expr=f'user_id == "{self._escape(user_id)}"',
            output_fields=["content", "subject"],
        )
        hits: list[MemoryHit] = []
        for result in results:
            for item in result:
                if float(item.score) < config.memory_vec_min_score:
                    continue
                entity = item.entity
                hits.append(
                    MemoryHit(
                        content=entity.get("content", ""),
                        subject=entity.get("subject", "") or "",
                        score=float(item.score),
                        content_hash=item.id,
                    )
                )
        return hits


milvus_memory_store = MilvusMemoryStore()
