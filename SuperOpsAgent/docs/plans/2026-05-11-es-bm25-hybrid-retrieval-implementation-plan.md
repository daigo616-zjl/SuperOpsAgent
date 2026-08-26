# Elasticsearch + BM25 Hybrid Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 Milvus 向量检索链路上增加 Elasticsearch BM25 关键词检索，并通过 RRF 融合让 `retrieve_knowledge` 默认走 hybrid retrieval。

**Architecture:** 保持现有上传、分块、Agent 工具签名不变，在索引侧新增 ES 双写，在检索侧新增 BM25 服务和 hybrid 融合服务。生命周期、健康检查和部署编排都挂到现有 FastAPI + Docker Compose 结构上，失败时按设计文档约定做回滚或降级。

**Tech Stack:** FastAPI, Pydantic Settings, LangChain, Milvus, Elasticsearch 8.x, elasticsearch-py, Docker Compose, Loguru

---

### Task 1: Add Elasticsearch dependency and configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`

**Step 1: Add Elasticsearch dependency**

在 `pyproject.toml` 的 `dependencies` 中加入：

```toml
"elasticsearch>=8.13,<9",
```

**Step 2: Add Elasticsearch settings**

在 `app/config.py` 的 `Settings` 中加入：

```python
es_host: str = "localhost"
es_port: int = 9200
es_index: str = "biz"
es_timeout: int = 10

rag_recall_size: int = 20
rag_rrf_k: int = 60
```

保持现有 `rag_top_k` 不变。

**Step 3: Verify dependency and config parse**

Run:
```bash
python -c "from app.config import config; print(config.es_host, config.es_port, config.rag_recall_size, config.rag_rrf_k)"
```

Expected: 打印默认 ES 配置和 RAG 参数。

**Step 4: Commit**

```bash
git add pyproject.toml app/config.py
git commit -m "feat: add elasticsearch retrieval config"
```

### Task 2: Add Elasticsearch client manager

**Files:**
- Create: `app/core/es_client.py`
- Modify: `app/core/__init__.py` (only if exports are used in this repo)

**Step 1: Create ES client manager**

新建 `app/core/es_client.py`，实现 `EsClientManager`，职责包括：
- 管理同步 `Elasticsearch` 和异步 `AsyncElasticsearch` client
- `connect()` 时检查 index 是否存在，不存在就按设计文档创建 `biz` mapping
- 提供 `get_sync_client()`、`get_async_client()`、`health_check()`、`close()`
- 所有配置从 `config` 读取

建议骨架：

```python
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.exceptions import NotFoundError
from loguru import logger

from app.config import config


class EsClientManager:
    def __init__(self) -> None:
        self._sync_client: Elasticsearch | None = None
        self._async_client: AsyncElasticsearch | None = None

    async def connect(self) -> None:
        if self._sync_client is not None and self._async_client is not None:
            return

        hosts = [f"http://{config.es_host}:{config.es_port}"]
        self._sync_client = Elasticsearch(hosts=hosts, request_timeout=config.es_timeout)
        self._async_client = AsyncElasticsearch(hosts=hosts, request_timeout=config.es_timeout)

        if not self._sync_client.indices.exists(index=config.es_index):
            self._sync_client.indices.create(index=config.es_index, body=self._index_body())

    def _index_body(self) -> dict:
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {"default": {"type": "BM25"}},
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                    "source": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "extension": {"type": "keyword"},
                    "h1": {"type": "keyword"},
                    "h2": {"type": "keyword"},
                    "h3": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                    "indexed_at": {"type": "date"},
                }
            },
        }
```

**Step 2: Add singleton**

文件尾部增加：

```python
es_client_manager = EsClientManager()
```

**Step 3: Verify import**

Run:
```bash
python -c "from app.core.es_client import es_client_manager; print(type(es_client_manager).__name__)"
```

Expected: 输出 `EsClientManager`。

**Step 4: Commit**

```bash
git add app/core/es_client.py app/core/__init__.py
git commit -m "feat: add elasticsearch client manager"
```

### Task 3: Add Elasticsearch store manager for chunk write/delete

**Files:**
- Create: `app/services/es_store_manager.py`

**Step 1: Create document payload builder**

新建 `app/services/es_store_manager.py`，实现把 LangChain `Document` 转为 ES `_source` 的逻辑，字段与设计文档一致：`content`、`source`、`file_name`、`extension`、`h1`、`h2`、`h3`、`metadata`、`indexed_at`。

**Step 2: Add bulk index method**

实现：

```python
def add_documents(self, documents: list[Document], ids: list[str]) -> None:
```

要求：
- 使用同步 client
- 使用 `elasticsearch.helpers.bulk`
- `raise_on_error=False`
- 如果 `errors` 非空，抛出 `RuntimeError`

**Step 3: Add delete-by-source method**

实现：

```python
def delete_by_source(self, file_path: str) -> int:
```

要求：
- 用 `delete_by_query`
- `term` 查询 `source`
- 返回删除数量

**Step 4: Add delete-by-ids method**

实现：

```python
def delete_by_ids(self, ids: list[str]) -> int:
```

要求：
- 对每个 id 走 bulk delete action
- 返回尝试删除的数量即可

**Step 5: Add singleton**

文件尾部增加：

```python
es_store_manager = EsStoreManager()
```

**Step 6: Verify import**

Run:
```bash
python -c "from app.services.es_store_manager import es_store_manager; print(type(es_store_manager).__name__)"
```

Expected: 输出 `EsStoreManager`。

**Step 7: Commit**

```bash
git add app/services/es_store_manager.py
git commit -m "feat: add elasticsearch store manager"
```

### Task 4: Extend Milvus store manager for deterministic IDs and rollback

**Files:**
- Modify: `app/services/vector_store_manager.py`

**Step 1: Change add_documents signature**

把：

```python
def add_documents(self, documents: List[Document]) -> List[str]:
```

改成：

```python
def add_documents(self, documents: List[Document], ids: List[str]) -> List[str]:
```

删除内部 UUID 生成逻辑，直接使用调用方传入的 `ids`。

**Step 2: Add delete_by_ids**

新增：

```python
def delete_by_ids(self, ids: List[str]) -> int:
    collection = milvus_manager.get_collection()
    quoted_ids = ", ".join([f'"{item}"' for item in ids])
    expr = f"id in [{quoted_ids}]"
    result = collection.delete(expr)
    return result.delete_count if hasattr(result, "delete_count") else 0
```

**Step 3: Keep existing APIs intact**

保留 `delete_by_source()` 和 `get_vector_store()` 等外部接口，避免额外重构。

**Step 4: Verify import and signature**

Run:
```bash
python -c "from app.services.vector_store_manager import vector_store_manager; print(hasattr(vector_store_manager, 'delete_by_ids'))"
```

Expected: 输出 `True`。

**Step 5: Commit**

```bash
git add app/services/vector_store_manager.py
git commit -m "feat: support explicit chunk ids in milvus store"
```

### Task 5: Update indexing flow to dual-write with rollback

**Files:**
- Modify: `app/services/vector_index_service.py`

**Step 1: Import UUID and ES manager**

在 `app/services/vector_index_service.py` 中引入：
- `uuid`
- `es_store_manager`

**Step 2: Update delete phase**

在 `index_single_file()` 中，删除旧数据时改为先删 Milvus，再删 ES：

```python
vector_store_manager.delete_by_source(normalized_path)
es_store_manager.delete_by_source(normalized_path)
```

保持任一删除失败即抛错。

**Step 3: Generate chunk IDs in service layer**

在 `documents = document_splitter_service.split_document(...)` 之后生成：

```python
chunk_ids = [str(uuid.uuid4()) for _ in documents]
```

**Step 4: Dual write with rollback**

把写入逻辑改成：

```python
vector_store_manager.add_documents(documents, chunk_ids)
try:
    es_store_manager.add_documents(documents, chunk_ids)
except Exception:
    vector_store_manager.delete_by_ids(chunk_ids)
    raise
```

**Step 5: Preserve empty-document behavior**

如果 `documents` 为空，继续保留当前 warning 分支，不做额外逻辑。

**Step 6: Verify module import**

Run:
```bash
python -c "from app.services.vector_index_service import vector_index_service; print(type(vector_index_service).__name__)"
```

Expected: 输出 `VectorIndexService`。

**Step 7: Commit**

```bash
git add app/services/vector_index_service.py
git commit -m "feat: dual write chunks to milvus and elasticsearch"
```

### Task 6: Add BM25 search service

**Files:**
- Create: `app/services/bm25_search_service.py`

**Step 1: Create BM25 result model or reuse simple dict/object**

新建服务，返回结果结构要至少包含：
- `id`
- `content`
- `score`
- `metadata`

可以复用 `VectorSearchService.SearchResult` 的风格，新建一个轻量结果类即可。

**Step 2: Implement async BM25 search**

实现：

```python
async def search(self, query: str, top_k: int) -> list[BM25SearchResult]:
```

查询体：

```python
{
    "size": top_k,
    "query": {"match": {"content": query}},
    "_source": ["content", "source", "file_name", "extension", "h1", "h2", "h3", "metadata"],
}
```

**Step 3: Normalize metadata shape**

返回的 metadata 要尽量和当前 Milvus 文档保持兼容：
- `_source`
- `_file_name`
- `_extension`
- `h1`
- `h2`
- `h3`
- 其余原始 metadata 合并保留

**Step 4: Add singleton**

```python
bm25_search_service = BM25SearchService()
```

**Step 5: Verify import**

Run:
```bash
python -c "from app.services.bm25_search_service import bm25_search_service; print(type(bm25_search_service).__name__)"
```

Expected: 输出 `BM25SearchService`。

**Step 6: Commit**

```bash
git add app/services/bm25_search_service.py
git commit -m "feat: add bm25 search service"
```

### Task 7: Add hybrid search service with RRF and downgrade

**Files:**
- Create: `app/services/hybrid_search_service.py`
- Modify: `app/services/vector_search_service.py` (only if a helper conversion method is needed)

**Step 1: Create async hybrid search**

新建 `HybridSearchService`，实现：

```python
async def search(self, query: str, top_k: int) -> list[Document]:
```

要求：
- 空 query 直接返回 `[]`
- 向量检索用 `asyncio.to_thread(vector_search_service.search_similar_documents, query, config.rag_recall_size)`
- BM25 检索用 `bm25_search_service.search(query, config.rag_recall_size)`
- 用 `asyncio.gather(..., return_exceptions=True)` 收集两路结果

**Step 2: Add downgrade handling**

规则：
- 一路异常：记录 `WARNING`，使用另一路结果
- 两路都异常：返回 `[]`
- 两路都空：返回 `[]`

**Step 3: Add result-to-document conversion**

在 hybrid service 内实现统一转换函数，把向量结果和 BM25 结果都转成 `Document`，并确保 `doc.metadata["id"] = chunk_id`，便于 RRF 聚合。

**Step 4: Implement RRF**

实现：

```python
def _rrf_fuse(self, vector_hits, bm25_hits, top_k: int) -> list[Document]:
```

规则：
- 平滑系数用 `config.rag_rrf_k`
- 按 chunk id 聚合
- 分数降序取前 `top_k`

**Step 5: Add sync wrapper**

实现：

```python
def search_sync(self, query: str, top_k: int) -> list[Document]:
    return asyncio.run(self.search(query, top_k))
```

**Step 6: Add singleton**

```python
hybrid_search_service = HybridSearchService()
```

**Step 7: Verify import**

Run:
```bash
python -c "from app.services.hybrid_search_service import hybrid_search_service; print(type(hybrid_search_service).__name__)"
```

Expected: 输出 `HybridSearchService`。

**Step 8: Commit**

```bash
git add app/services/hybrid_search_service.py app/services/vector_search_service.py
git commit -m "feat: add hybrid retrieval with rrf"
```

### Task 8: Switch knowledge tool from vector-only to hybrid retrieval

**Files:**
- Modify: `app/tools/knowledge_tool.py`

**Step 1: Replace vector-store retrieval path**

删除当前：

```python
vector_store = vector_store_manager.get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": config.rag_top_k})
docs = retriever.invoke(query)
```

改成：

```python
docs = hybrid_search_service.search_sync(query, top_k=config.rag_top_k)
```

**Step 2: Keep tool signature unchanged**

保留：
- `@tool(response_format="content_and_artifact")`
- `def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]`
- `format_docs()` 返回结构

**Step 3: Preserve empty-result behavior**

继续在 `not docs` 时返回：

```python
return "没有找到相关信息。", []
```

**Step 4: Verify import**

Run:
```bash
python -c "from app.tools.knowledge_tool import retrieve_knowledge; print(retrieve_knowledge.name)"
```

Expected: 正常输出工具名。

**Step 5: Commit**

```bash
git add app/tools/knowledge_tool.py
git commit -m "feat: route knowledge tool through hybrid retrieval"
```

### Task 9: Wire Elasticsearch lifecycle and health check

**Files:**
- Modify: `app/main.py`
- Modify: `app/api/health.py`

**Step 1: Connect ES on startup**

在 `app/main.py` 的 lifespan 启动阶段，在 Milvus 连接之后加入：

```python
logger.info("🔌 正在连接 Elasticsearch...")
await es_client_manager.connect()
logger.info("✅ Elasticsearch 连接成功")
```

**Step 2: Close ES on shutdown**

在 shutdown 阶段加入：

```python
logger.info("🔌 正在关闭 Elasticsearch 连接...")
await es_client_manager.close()
```

如果 `close()` 设计为同步，则对应去掉 `await`，但整个实现里建议统一做成 async close。

**Step 3: Extend health API**

在 `app/api/health.py` 中增加 ES 检查结果，并把 overall status 判定为：
- Milvus 和 ES 都健康：200
- 任一不健康：503

同时保留当前响应包裹结构：

```json
{
  "code": 200,
  "message": "服务运行正常",
  "data": { ... }
}
```

只是在 `data` 里新增 `elasticsearch` 字段。

**Step 4: Verify startup imports**

Run:
```bash
python -c "from app.main import app; print(app.title)"
```

Expected: 正常输出应用名，无导入错误。

**Step 5: Commit**

```bash
git add app/main.py app/api/health.py
git commit -m "feat: add elasticsearch lifecycle and health check"
```

### Task 10: Add Elasticsearch service to Docker Compose

**Files:**
- Modify: `vector-database.yml`

**Step 1: Add Elasticsearch service**

在 `vector-database.yml` 中加入 `elasticsearch` 服务，参数按设计文档：
- image: `infinilabs/elasticsearch-ik:8.13.4`
- `discovery.type=single-node`
- `xpack.security.enabled=false`
- `ES_JAVA_OPTS=-Xms512m -Xmx512m`
- 暴露 `9200:9200`
- 添加 data volume
- 添加 healthcheck

**Step 2: Keep existing Milvus services unchanged**

不要顺手整理 compose 结构，不改现有 etcd/minio/standalone/attu 行为。

**Step 3: Verify compose syntax**

Run:
```bash
docker compose -f vector-database.yml config
```

Expected: 成功输出规范化 compose 配置。

**Step 4: Commit**

```bash
git add vector-database.yml
git commit -m "feat: add elasticsearch service to compose"
```

### Task 11: Update README for Elasticsearch-based hybrid retrieval

**Files:**
- Modify: `README.md`

**Step 1: Update feature and stack description**

在 README 中把 RAG 描述从纯向量检索改为 hybrid retrieval，至少体现：
- Milvus + Elasticsearch
- BM25 + 向量召回

**Step 2: Update startup docs**

把 Docker Compose 启动说明改成包含 ES，Windows 启动等待时间描述改成更保守一些，避免仍写死“约 5-10 秒”。

**Step 3: Update exposed services**

在文档中新增：
- Elasticsearch: `http://localhost:9200`

**Step 4: Verify markdown presence**

Run:
```bash
python -c "from pathlib import Path; print('Elasticsearch' in Path('README.md').read_text(encoding='utf-8'))"
```

Expected: 输出 `True`。

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update readme for hybrid retrieval"
```

### Task 12: End-to-end bring-up verification without tests

**Files:**
- Modify: none

**Step 1: Install updated dependencies**

Run:
```bash
pip install -e .
```

Expected: 安装 `elasticsearch` 依赖成功。

**Step 2: Start infrastructure**

Run:
```bash
docker compose -f vector-database.yml up -d
```

Expected: Milvus 和 Elasticsearch 容器启动。

**Step 3: Verify Elasticsearch is reachable**

Run:
```bash
curl http://localhost:9200
```

Expected: 返回 ES cluster info JSON。

**Step 4: Start app**

Run:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

Expected: 启动日志同时显示 Milvus 和 Elasticsearch 连接成功。

**Step 5: Verify health endpoint**

另开终端运行：
```bash
curl http://localhost:9900/health
```

Expected: 响应 `data` 中同时包含 `milvus` 和 `elasticsearch` 健康状态。

**Step 6: Re-index documents**

如需重建数据，运行现有初始化或重新上传文档，让 Milvus 与 ES 从头对齐。

**Step 7: Smoke check retrieval path**

调用现有上传接口和对话接口，确认服务日志中能看到：
- ES 写入完成
- BM25 检索结果数
- 向量检索结果数
- hybrid / RRF 融合结果数

**Step 8: Final commit if verification required changes**

```bash
git status --short
```

Expected: 无额外意外修改。
