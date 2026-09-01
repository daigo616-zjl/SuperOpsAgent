# SuperOpsAgent

## 本地启动

项目使用 Milvus Lite，不需要单独启动 Milvus Server；Elasticsearch 和 PostgreSQL
需要提前在本机运行。首次配置可复制 `.env.example` 为 `.env`，填写
`DASHSCOPE_API_KEY` 和数据库连接信息。

Windows 下可直接运行：

```bat
start-windows.bat
```

启动脚本会依次启动 CLS MCP、Monitor MCP 和 FastAPI。服务健康检查地址为：

```text
http://localhost:18000/api/health
```

`.env` 仅用于本地配置，不应提交到 Git；提交配置模板请使用 `.env.example`。

## AIOps 多 Agent 诊断（星型编排）

`/api/aiops` 的诊断默认由**星型拓扑多 Agent 编排**驱动（`app/agent/aiops/`）：

- **Supervisor** 是唯一中枢，确定性状态机路由（不调 LLM）：无假设 → 假设生成；
  无指令 → 确定性生成 metrics/logs/knowledge 三域取证任务并扇出；有新证据 →
  评审（淘汰/加钻/收敛）；预算耗尽 → 按当前证据收敛输出。
- **Investigators（取证域）** 只与 Supervisor 通信，各自跑 ReAct 子图调用
  MCP 工具；证据出处（ClaimProvenance）由代码从真实工具调用记录确定性构建，
  模型无法虚构 provenance。
- **Reporter** 流式生成最终报告，并按证据卡 claim 白名单剥离未支撑的
  `[ev-*]` 引用（反幻觉）。

引擎通过 `AIOPS_ENGINE` 切换（`multiagent` 默认 / `legacy` 保留旧
plan-execute-replan 流程用于 A/B 对比）。

### 预算与超时配置

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `AIOPS_MAX_ROUNDS` | 6 | 编排最大轮数 |
| `AIOPS_MAX_INVOCATIONS` | 60 | LLM 迭代配额（每轮派发预扣各任务 max_iterations） |
| `AIOPS_MAX_WALL_SECONDS` | 240 | 诊断墙钟预算 |
| `AIOPS_MIN_DISPATCH_WALL_SECONDS` | 90 | 剩余墙钟低于该值不再派发新取证任务 |
| `AIOPS_INVESTIGATION_WALL_SECONDS` | 120 | 单个取证任务墙钟上限（Send 并行分支需全部返回，单域卡住时该上限兜底） |
| `AIOPS_INVESTIGATOR_TIMEOUT` | 60 | 取证 LLM 单次调用超时（需小于单任务上限，挂死后重试才来得及在任务预算内完成） |
| `AIOPS_INVESTIGATOR_MODEL` 等 | 回退 `RAG_MODEL` | 各角色模型可单独覆盖 |

### Mock 故障剧本

`mcp_servers/scenarios/*.yaml` 提供剧本化故障注入（告警清单、指标曲线塑形、
日志错误模式、`ground_truth.root_cause`）。通过 `MOCK_SCENARIO` 环境变量选择，
可用剧本：`db-slow-query`、`distractor-cpu`、`gc-pressure`、`no-fault`、
`oom-kill`。同一剧本数据按时间窗确定性播种。

### A/B 双引擎基准

```bash
# 先停掉 start-windows.bat 启动的服务（基准需要独占 18003/18004 端口）
python scripts/run_aiops_scenarios.py                # 5 剧本 × 2 引擎 × 3 次
python scripts/run_aiops_scenarios.py --runs 1       # 快速全量（10 次）
python scripts/run_aiops_scenarios.py --scenarios gc-pressure --engines multiagent
python scripts/run_aiops_scenarios.py --no-judge     # 只跑诊断不评分
```

每个剧本独占启动一组 mock MCP 子进程，跑完即停。判分使用 `EVAL_MODEL`
（默认 qwen-max）做根因命中与幻觉审计；门禁：**multiagent 命中率 ≥ legacy
且 幻觉率 < legacy**，exit code 反映门禁结果。明细写入
`eval/reports/aiops/`（已 gitignore）。

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

## Milvus Lite 连接配置

Milvus Lite 以本地 `data/milvus.db` 文件运行，项目会通过 PyMilvus 连接。为避免
嵌入式 gRPC 服务因空闲连接频繁发送 ping 而返回
`GOAWAY too_many_pings`，默认使用以下 keepalive 配置：

```dotenv
MILVUS_GRPC_KEEPALIVE_TIME_MS=60000
MILVUS_GRPC_KEEPALIVE_TIMEOUT_MS=20000
MILVUS_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS=false
```

这条 `GOAWAY` 日志中的 `127.0.0.1:11xxx` 通常是 Milvus Lite 的动态 gRPC 端口，
不是 DashScope 长连接异常。修改上述配置后需要重启 API 或重新运行评测。

## RAG 评测

运行完整 Ragas 评测：

```bash
python -m app.eval.cli \
  --dataset eval/fixtures/sample_ragas_dataset.jsonl \
  --output eval/reports/sample_ragas_dataset.json
```

也可以只评测指定指标，例如将 Faithfulness 和 Answer Correctness 拆开运行：

```bash
python -m app.eval.cli --dataset eval/fixtures/sample_ragas_dataset.jsonl \
  --limit 5 --metrics faithfulness \
  --output eval/reports/first5-faithfulness.json

python -m app.eval.cli --dataset eval/fixtures/sample_ragas_dataset.jsonl \
  --limit 5 --metrics answer_correctness \
  --output eval/reports/first5-answer-correctness.json
```

如果已有评测报告，可复用其中已经生成的 `answer` 和 `retrieved_contexts`，
只重新评分指定指标，不再执行 RAG 生成和检索：

```bash
python -m app.eval.cli \
  --input-report eval/reports/sample_ragas_dataset-50-controlled.json \
  --metrics faithfulness,answer_correctness \
  --output eval/reports/sample_ragas_dataset-50-rescored.json
```

Faithfulness 默认超时为 360 秒，Answer Correctness 默认超时为 300 秒；
两项评分相互独立，某一项失败不会阻塞另一项。

评测会执行答案生成、Milvus + Elasticsearch 双路召回、RRF、Rerank、Top-K，
并统计 Faithfulness、Answer Relevancy、Answer Correctness、Context Relevance、
Recall@20 和 Hit@5。评测前请确认 Rerank 模型可用、MCP 服务已启动，且
Milvus Lite 没有被其他进程以冲突方式占用。
