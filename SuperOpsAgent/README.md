# SuperOpsAgent

## PostgreSQL 权威 RAG 文档库

知识文档原文、索引注册表和索引任务均保存在 PostgreSQL。Milvus 与
Elasticsearch 是可重建的派生索引；服务启动不会扫描 `aiops-docs`、`uploads`
或任何本地 Markdown 目录，也不再读取 `data/knowledge_index_state.json` 和
`uploads/.index_tasks.json`。

先创建 PostgreSQL 数据库并配置 `.env`：

```dotenv
DATABASE_URL=postgresql+psycopg://superops:superops@localhost:5432/superops
```

启动时会幂等执行 `migrations/001_postgres_knowledge.sql`。文档可从网页的
“文档管理”入口新增、查看、修改、删除和手动重建，也可调用：

- `POST /api/knowledge/documents`
- `GET /api/knowledge/documents`
- `GET|PUT|DELETE /api/knowledge/documents/{document_id}`
- `POST /api/knowledge/documents/{document_id}/reindex`

文档写入与 Outbox 入队在同一 PostgreSQL 事务内完成。后台 Worker 使用
`FOR UPDATE SKIP LOCKED` 领取任务，写入 Milvus 和 Elasticsearch 后才提交索引
注册表。失败任务指数退避重试；租约超时可被其他实例接管。周期巡检会比较注册表
与两个索引的版本和分片数，发现缺失后自动产生 repair 任务。

旧目录仅在升级时显式导入一次，绝不会随服务启动自动执行：

```bash
python scripts/import_markdown_to_postgres.py --directory aiops-docs
```
