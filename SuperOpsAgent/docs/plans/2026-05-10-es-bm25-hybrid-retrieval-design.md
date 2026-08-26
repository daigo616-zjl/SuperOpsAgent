# Elasticsearch + BM25 双路召回 RAG 设计

> 日期：2026-05-10
> 状态：已对齐，待实现
> 目标：在现有 Milvus 向量检索基础上增加 Elasticsearch BM25 关键词检索，通过 RRF 融合实现 hybrid retrieval。

## 1. 背景

当前 RAG 链路只有一路：DashScope embedding → Milvus 相似度检索（L2 距离 + IVF_FLAT）。
纯向量检索对**专有名词、缩写、数字、错误码**等字面量匹配不敏感，运维场景里这类词高频出现（如 OOM、5xx、CPU、IOPS）。
引入 BM25 与向量检索互补，是当前业界 hybrid RAG 的标准做法。

## 2. 决策汇总

| 决策点 | 选择 |
|---|---|
| ES 索引粒度 | chunk 级别同步双写（与 Milvus 对齐） |
| 融合算法 | RRF (Reciprocal Rank Fusion) |
| 中文分词 | IK 分词器（`ik_max_word` 索引 / `ik_smart` 查询） |
| ES 部署 | 加到现有 `vector-database.yml` |
| 双写一致性 | 任一侧失败则回滚 |
| 检索入口 | `retrieve_knowledge` 默认走 hybrid，不暴露开关 |
| ES 版本/鉴权 | ES 8.x，本地免鉴权 |
| 镜像 | `infinilabs/elasticsearch-ik:8.13.4`（IK 预装） |
| 历史数据 | 不做迁移，`make init` 重建 |

## 3. 架构总览

```
                       ┌──────────────────────┐
   /api/upload  ──►    │ vector_index_service │
                       │  (索引编排器)         │
                       └─────┬────────┬───────┘
                             │        │   (同一批 chunks，事务式双写)
                             │        │
                  ┌──────────▼┐     ┌─▼──────────────┐
                  │ Milvus    │     │ Elasticsearch  │
                  │ (vector)  │     │ (BM25 + 原文)   │
                  └──────────┬┘     └─┬──────────────┘
                             │        │
                       ┌─────▼────────▼─────┐
                       │ HybridSearchService │
                       │  ・并行召回         │
                       │  ・RRF 融合         │
                       │  ・top-k 截断       │
                       └─────────┬───────────┘
                                 │
                       ┌─────────▼───────────┐
                       │ retrieve_knowledge  │  ← LangGraph Agent 工具
                       └─────────────────────┘
```

### 新增组件

| 文件 | 职责 |
|---|---|
| `app/core/es_client.py` | ES 客户端工厂（同步 + 异步），index 初始化、健康检查 |
| `app/services/es_store_manager.py` | chunk 级别的 ES 写入 / 删除（与 `vector_store_manager` 对称） |
| `app/services/bm25_search_service.py` | BM25 异步检索 |
| `app/services/hybrid_search_service.py` | 双路召回 + RRF 融合 + 降级 |

### 改动现有组件

| 文件 | 改动 |
|---|---|
| `vector-database.yml` | 增加 `elasticsearch` 服务（8.x + IK） |
| `app/config.py` | 增加 ES 配置项 + RRF 参数 |
| `app/services/vector_index_service.py` | 改为「先写 Milvus、再写 ES、任一失败回滚」 |
| `app/services/vector_store_manager.py` | 新增 `delete_by_ids()` 用于回滚；UUID 生成上移到调用方 |
| `app/tools/knowledge_tool.py` | `retrieve_knowledge` 内部从纯向量切到 hybrid |
| `app/api/health.py` | 增加 ES 健康检查 |
| `app/main.py` | lifespan 中连接/释放 ES |
| `pyproject.toml` | 增加 `elasticsearch>=8.13,<9` |

> 上层 Agent 工具 `retrieve_knowledge` 的签名和返回结构**保持不变**，LangGraph、AIOps、UI 链路无感。

## 4. 数据模型 & ES Mapping

### Chunk 在两侧的字段对照

| 字段 | Milvus (现有) | Elasticsearch (新增) | 说明 |
|---|---|---|---|
| 主键 | `id` (VARCHAR, ≤100) | `_id` (string) | **同一个 UUID**，跨库可对齐 |
| 向量 | `vector` (FLOAT_VECTOR, 1024) | — | 仅 Milvus |
| 正文 | `content` (VARCHAR, ≤8000) | `content` (text, IK 分词) | ES 用 BM25 |
| 元数据 | `metadata` (JSON) | 平铺 + `metadata` (object) | 关键字段单独抽出 |

### `biz` index mapping

```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "similarity": { "default": { "type": "BM25" } }
  },
  "mappings": {
    "properties": {
      "content":    { "type": "text",
                      "analyzer": "ik_max_word",
                      "search_analyzer": "ik_smart" },
      "source":     { "type": "keyword" },
      "file_name":  { "type": "keyword" },
      "extension":  { "type": "keyword" },
      "h1":         { "type": "keyword" },
      "h2":         { "type": "keyword" },
      "h3":         { "type": "keyword" },
      "metadata":   { "type": "object", "enabled": false },
      "indexed_at": { "type": "date" }
    }
  }
}
```

设计要点：
- `number_of_replicas: 0`：单机本地部署，没必要副本
- `ik_max_word` 索引 / `ik_smart` 查询：IK 推荐组合（细粒度索引提升召回，粗粒度查询提升精度）
- `source` / `file_name` keyword：用于按文件精确删除（`term` 查询）
- `metadata` `enabled: false`：不索引但保留原文，避免动态 schema 漂移
- `h1/h2/h3` 平铺：v1 不参与 BM25 加权，仅用于结果展示；v2 可加 multi_match 加权

### 写入文档形态

`vector_index_service` 切完 chunks 后构造两份载荷：

**Milvus (现有，不变)**：
```python
Document(
  page_content=chunk_text,
  metadata={"_source": ..., "_file_name": ..., "_extension": ..., "h1": ..., ...}
)
```

**Elasticsearch (新增)**：
```python
{
  "_index": "biz",
  "_id": chunk_uuid,
  "_source": {
    "content":   chunk_text,
    "source":    metadata.get("_source"),
    "file_name": metadata.get("_file_name"),
    "extension": metadata.get("_extension"),
    "h1":        metadata.get("h1"),
    "h2":        metadata.get("h2"),
    "h3":        metadata.get("h3"),
    "metadata":  metadata,
    "indexed_at": now_iso8601()
  }
}
```

### 按文件删除对齐

Milvus: `metadata["_source"] == "<path>"`
ES: `{ "query": { "term": { "source": "<path>" } } }` (delete_by_query)

两侧用同一个 normalized path（`Path.as_posix()`，复用现有约定）。

## 5. 写入流程（含双写回滚）

```
索引一个文件的完整流程：

  ┌─ Phase 1: 删除旧数据 ────────────────┐
  │  Milvus.delete_by_source(path)      │
  │  ES.delete_by_source(path)          │
  │  (任一失败都报错；旧数据残留可接受    │
  │   因为下次重试是幂等的)              │
  └──────────────────────────────────────┘
                 │
  ┌─ Phase 2: 双写新数据 ────────────────┐
  │  ① Milvus.add_documents(docs, ids)  │
  │     成功 → 进入 ②                    │
  │     失败 → 抛错（无需回滚 ES）        │
  │                                      │
  │  ② ES.bulk_index(docs, ids)         │
  │     成功 → 完成                      │
  │     失败 → 回滚：                    │
  │             Milvus.delete_by_ids(ids)│
  │             再抛原错                 │
  └──────────────────────────────────────┘
```

**顺序：Milvus 先、ES 后。** 理由：
- Milvus 写入慢（embedding API 调用）、失败概率高
- ES 写入快、稳定，放第二步可减少回滚发生
- ES 失败时回滚 Milvus，是更小的代价

### 失败语义

| 失败点 | 用户看到 | 系统状态 |
|---|---|---|
| Phase 1 Milvus 删除失败 | HTTP 500 | 两侧都未变更 |
| Phase 1 ES 删除失败 | HTTP 500 | Milvus 旧数据已删、ES 残留 → 重试时再删一次（幂等） |
| Phase 2 ① Milvus 写失败 | HTTP 500 | 旧数据已清空，该文件暂时无数据（重试可恢复） |
| Phase 2 ② ES 写失败 | HTTP 500 | 触发 `delete_by_ids` 回滚 → 两侧都为空（重试可恢复） |

### 关键新增方法

`vector_store_manager.delete_by_ids(ids: List[str]) -> int`：
```python
def delete_by_ids(self, ids: List[str]) -> int:
    expr = f'id in {ids}'
    result = milvus_manager.get_collection().delete(expr)
    return result.delete_count
```

### UUID 上移

现状 UUID 在 `vector_store_manager.add_documents()` 内部生成，外部拿不到。
改为在 `vector_index_service.index_single_file()` 内生成 UUID 列表，分别传给两侧 store_manager。

### ES 批量写入

```python
from elasticsearch.helpers import bulk
actions = [
    {"_index": "biz", "_id": cid, "_source": payload}
    for cid, payload in zip(chunk_ids, payloads)
]
success, errors = bulk(es_sync_client, actions, raise_on_error=False)
if errors:
    raise RuntimeError(f"ES bulk 部分失败: {errors[:3]}")
```

### 同步/异步边界

- **写入路径用同步 `Elasticsearch` 客户端**：与现有 `vector_index_service` 同步语义对齐，改动小
- **检索路径用异步 `AsyncElasticsearch` 客户端**：BM25 与向量两路 `asyncio.gather` 并发
- 两个 client 实例由 `EsClientManager` 统一管理生命周期

## 6. 检索流程（双路召回 + RRF）

### 调用链

```
retrieve_knowledge(query)            ← LangGraph 工具，签名不变
        │
        ▼
hybrid_search_service.search(query, top_k=3)
        │
        ├──► asyncio.gather(
        │       vector_search_service.search_async(query, k=20),  ← to_thread 包装
        │       bm25_search_service.search(query, k=20),          ← 新增
        │    )
        │
        ▼
RRF 融合（k=60，按 chunk_id 聚合）
        │
        ▼
按 RRF 分数降序，返回 top_k 个 Document
```

### 两路召回

每路召回 `recall_size = 20` 条，给 RRF 留候选空间。

**向量路**：复用 `vector_search_service.search_similar_documents(query, top_k=20)`，用 `asyncio.to_thread()` 包装。

**BM25 路**（新增 `bm25_search_service.search()`）：
```python
body = {
    "size": 20,
    "query": { "match": { "content": query } },
    "_source": ["content", "source", "file_name", "h1", "h2", "h3", "metadata"]
}
resp = await es_async_client.search(index="biz", body=body)
```

v1 仅对 `content` 字段 `match`，不对 h1/h2/h3 加权。

### RRF 算法

```python
def rrf_fuse(vector_hits, bm25_hits, k=60, top_k=3):
    scores = {}                           # chunk_id -> rrf_score
    payload = {}                          # chunk_id -> Document
    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0) + 1 / (k + rank)
        payload[hit.id] = to_document(hit)
    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0) + 1 / (k + rank)
        payload.setdefault(hit.id, to_document(hit))
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [payload[cid] for cid, _ in ranked]
```

`k=60` 是业界默认（Cormack 2009），不参数化。
**靠 chunk_id（UUID）做去重和聚合** —— 这是第 4 段对齐两侧主键的核心目的。

### `retrieve_knowledge` 改动

```python
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    docs = hybrid_search_service.search_sync(query, top_k=config.rag_top_k)
    if not docs:
        return "没有找到相关信息。", []
    return format_docs(docs), docs
```

**只换中间这一行**，签名 / 返回结构不变。

`hybrid_search_service.search_sync` 内部用 `asyncio.run()` 跑两路并发（LangGraph `@tool` 默认同步调用，工具调用本身在 `to_thread` 里跑）。

### 边界情况 / 降级

| 情况 | 行为 |
|---|---|
| ES 不可用 | BM25 路抛错 → 降级为纯向量（捕获、返回 `[]`） |
| Milvus 不可用 | 向量路抛错 → 降级为纯 BM25 |
| 两路都失败 | 返回 `"没有找到相关信息。", []` |
| 两路都返回空 | 同上 |
| Query 为空 | 提前判，不发请求 |

降级时打 `WARNING` 日志便于发现问题。

### 配置项

新增到 `app/config.py`：
```python
es_host: str = "localhost"
es_port: int = 9200
es_index: str = "biz"
es_timeout: int = 10

rag_recall_size: int = 20   # 每路召回数
rag_rrf_k: int = 60         # RRF 平滑系数
# rag_top_k 已存在，复用
```

## 7. 部署 / 启动

### 7.1 `vector-database.yml` 增量

```yaml
  elasticsearch:
    container_name: rag-elasticsearch
    image: infinilabs/elasticsearch-ik:8.13.4
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - cluster.name=rag-es
    ulimits:
      memlock: { soft: -1, hard: -1 }
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/elasticsearch:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -fs http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    ports:
      - "9200:9200"
```

镜像 `infinilabs/elasticsearch-ik:8.13.4` 是 IK 作者维护、预装好 IK 的镜像，免去手动安装插件。
`xpack.security.enabled=false` 必需，否则 ES 8.x 默认强制 HTTPS + 鉴权。

### 7.2 索引初始化

`EsClientManager.connect()` 检查 `biz` index：
- 不存在 → 用上文 mapping 创建
- 已存在 → 跳过（不做 mapping 校验，改 mapping 需手动删 index）

在 `app/main.py` lifespan 启动钩子里 `await es_client_manager.connect()`，与 Milvus 并列。

### 7.3 健康检查

```json
GET /api/health
{
  "status": "ok",
  "milvus": "ok",
  "elasticsearch": "ok"
}
```

### 7.4 文档 / 脚本同步

| 文件 | 改动 |
|---|---|
| `README.md` | 增加 ES 端口（9200）、依赖、启动等待时间提示 |
| `start-windows.bat` | 不用改（compose 自动起所有 service） |
| `Makefile` | 不用改 |
| `pyproject.toml` | 增加 `elasticsearch>=8.13,<9` |

### 7.5 数据迁移

**不写迁移脚本，让 `make init` 重新跑一遍上传。** 理由：
- 文档量小（5 个 md 文件）
- 双写从头跑保证两侧 ID 一致
- 写迁移工具不划算

## 8. 测试策略

### 单元测试（`tests/services/`）

- `test_bm25_search_service.py` — mock `AsyncElasticsearch`，验证 query body
- `test_hybrid_search_service.py` — mock 两路结果，验证 RRF 排序、降级
- `test_vector_index_service.py` — mock 两侧 store_manager，验证回滚路径

### 集成测试（可选）

`tests/integration/test_index_and_search.py`：
- 起一个文件 → 走完整双写 → 走 hybrid_search
- 断言 chunk_id 在两侧一致
- pytest marker `@pytest.mark.integration`，CI 跳过、本地 `pytest -m integration` 触发

### 手工验收清单

1. `docker compose up` 后访问 `http://localhost:9200` 确认 ES 返回 cluster info
2. `make init` 后，去 ES `GET /biz/_count` 看 chunks 数与 Milvus 对得上
3. 调 `/api/chat` 问运维问题，日志能看到「BM25 召回 N / 向量召回 M / RRF top-3」
4. 杀掉 ES 容器，再问一次，确认降级生效（WARNING 但不报错）

## 9. v1 不做（YAGNI）

- IK 自定义运维词典
- reranker 重排
- hybrid 检索的 `mode` 参数（不暴露给 LLM）
- Milvus → ES 反向迁移工具
- ES 副本 / 集群
- h1/h2/h3 BM25 加权
- search_after / 滚动分页
