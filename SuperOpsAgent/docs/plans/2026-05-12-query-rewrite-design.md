# RAG 查询重写设计

> 日期：2026-05-12
> 状态：已对齐，待实现
> 目标：在现有 hybrid retrieval 前增加查询重写能力，结合最近会话上下文补全指代并增强检索查询，提高 RAG 召回效果。

## 1. 背景

当前 RAG 检索入口是 `app/tools/knowledge_tool.py` 中的 `retrieve_knowledge(query)`，内部直接调用 `hybrid_search_service.search_sync(query, top_k)`。

现有链路已经具备向量检索 + BM25 双路召回，但输入 query 仍然是用户原始问题。对于运维场景中的追问、代词、省略和口语化表达，原始 query 往往不利于检索，例如：

- `这个报错怎么处理`
- `上面那个服务为什么会挂`
- `这个告警和 5xx 有关系吗`

这些问法在多轮对话里语义完整，但脱离上下文后检索信号不足。需要在检索前增加一层查询重写，将用户问题改写为更适合 BM25 和向量检索的单条查询。

## 2. 决策汇总

| 决策点 | 选择 |
|---|---|
| 重写位置 | `retrieve_knowledge` 内部、`hybrid_search_service` 之前 |
| 重写组件 | 新增 `QueryRewriteService` |
| 是否结合上下文 | 是，读取最近 3 轮会话 |
| 输出形态 | 单条 rewritten query |
| 检索输入 | 只使用 rewritten query |
| 失败策略 | 回退原始 query |
| session 透传 | `contextvars` request-scoped 上下文 |
| 检索层改动 | 不改 `HybridSearchService` 职责 |
| 测试范围 | 本次不新增覆盖测试 |

## 3. 架构总览

```
用户问题
   │
   ▼
RagAgentService.query/query_stream
   │
   │  设置 request-scoped session_id
   ▼
LangGraph Agent
   │
   ▼
retrieve_knowledge(query)
   │
   ├──► QueryRewriteService.rewrite(query, session_id)
   │         ├── 读取最近 3 轮会话历史
   │         ├── 调用模型生成单条检索 query
   │         └── 失败时回退原 query
   │
   ▼
hybrid_search_service.search_sync(rewritten_query, top_k)
   │
   ▼
返回检索文档给 Agent
```

### 新增组件

| 文件 | 职责 |
|---|---|
| `app/services/query_rewrite_service.py` | 查询重写、历史提取、失败回退 |
| `app/core/request_context.py` | 管理当前请求的 `session_id` 上下文 |

### 改动现有组件

| 文件 | 改动 |
|---|---|
| `app/services/rag_agent_service.py` | 在调用 agent 前后设置/清理 `session_id` 上下文 |
| `app/tools/knowledge_tool.py` | 检索前接入 query rewrite |
| `app/config.py` | 增加 query rewrite 配置项 |

## 4. 组件设计

### 4.1 QueryRewriteService

建议接口：

```python
async def rewrite(self, query: str, session_id: str) -> str
```

同时提供同步包装：

```python
def rewrite_sync(self, query: str, session_id: str) -> str
```

职责约束：
- 输入当前 query 和会话 ID
- 读取最近 3 轮会话历史
- 调用轻量模型进行单次重写
- 仅返回一条纯文本查询
- 不回答问题、不生成多条 query、不输出 JSON
- 任何异常都回退原 query

### 4.2 Request Context

新增一个轻量 request-scoped 上下文模块，基于 `contextvars.ContextVar` 保存当前 `session_id`。

建议提供：

```python
set_current_session_id(session_id: str)
get_current_session_id() -> str | None
reset_current_session_id(token)
```

使用方式：
- `RagAgentService.query/query_stream` 在调用 agent 前设置 `session_id`
- `retrieve_knowledge` 内部读取当前 `session_id`
- 调用结束后在 `finally` 中 reset，避免污染后续请求

选择 `contextvars` 而不是改 tool 签名或耦合 LangChain runtime config，原因是侵入更小、异步场景更稳定、实现最直接。

## 5. 数据流与重写规则

### 5.1 历史窗口

v1 只读取最近 3 轮消息，最多包含：
- 3 条 user 消息
- 3 条 assistant 消息

目的：
- 覆盖“这个报错 / 那个服务 / 上面那个告警”这类常见追问
- 避免把更早的话题错误带入当前检索
- 控制模型输入长度与成本

### 5.2 重写目标

模型提示词要明确限制为：
- 补全代词和省略信息
- 保留上文出现的服务名、错误码、组件名、英文术语、时间范围
- 保留原始字面量，如 `OOM`、`5xx`、`CrashLoopBackOff`、实例 ID、告警名
- 输出为适合检索的一句话
- 不要回答问题
- 不要编造上下文中不存在的信息
- 不要扩展成多个查询

### 5.3 输出示例

输入：
- 当前问题：`这个报错怎么处理`
- 上文：`k8s 里 payment-service 出现 CrashLoopBackOff`

输出：

```text
k8s payment-service CrashLoopBackOff 报错处理方法
```

## 6. 接入方式

### 6.1 `retrieve_knowledge` 改动

当前：

```python
docs = hybrid_search_service.search_sync(query, top_k=config.rag_top_k)
```

调整后：

```python
session_id = get_current_session_id()
rewritten_query = query_rewrite_service.rewrite_sync(query, session_id) if session_id else query
docs = hybrid_search_service.search_sync(rewritten_query, top_k=config.rag_top_k)
```

保持以下不变：
- `retrieve_knowledge(query: str)` 签名不变
- 返回结构不变
- `format_docs()` 不变
- `HybridSearchService` 不感知 query rewrite

### 6.2 `RagAgentService` 改动

在 `query()` 和 `query_stream()` 中：
- 调 agent 前设置当前 `session_id`
- 在 `finally` 中清理上下文

这样工具层不需要显式接收 session_id 参数，也不需要修改 Agent tool schema。

## 7. 错误处理与降级

### 7.1 回退规则

以下情况统一回退原始 query：
- query 为空
- session_id 缺失
- 会话历史为空且模型重写失败
- 模型调用超时
- 模型调用异常
- 返回空串
- 返回明显异常的长文本

### 7.2 系统行为

| 场景 | 行为 |
|---|---|
| 重写成功 | 用 rewritten query 检索 |
| 重写失败 | 记录 warning，回退原 query |
| 重写和检索都失败 | 维持现有检索层错误处理 |

结论：query rewrite 是增强层，不改变 RAG 基础可用性。

## 8. 配置项

新增到 `app/config.py`：

```python
rag_query_rewrite_enabled: bool = True
rag_query_rewrite_model: str = ""
rag_query_rewrite_history_rounds: int = 3
rag_query_rewrite_timeout: int = 5
rag_query_rewrite_max_length: int = 200
```

建议：
- `rag_query_rewrite_model` 为空时默认回退到 `rag_model`
- `rag_query_rewrite_max_length` 用于过滤异常输出，避免把长段回答误当检索 query

## 9. 日志与可观测性

至少记录：
- `original_query`
- `rewritten_query`
- `rewrite_used_history`
- `rewrite_fallback`

日志要求：
- 不打印整段长历史
- 重写失败打 `WARNING`
- 正常重写打 `INFO`

## 10. 本次不做

- 多 query 扩展
- reranker 重排
- query 分类路由
- 前端显示 rewritten query
- 测试补充与覆盖率建设
- 针对 h1/h2/h3 的特殊查询加权

## 11. 最终结论

v1 采用最小侵入方案：
- 增加 `QueryRewriteService`
- 在 `retrieve_knowledge` 中接入查询重写
- 使用 `contextvars` 透传 `session_id`
- 结合最近 3 轮会话进行单次重写
- 仅使用 rewritten query 检索
- 失败时无条件回退原 query

该方案与现有 RAG 架构兼容性最好，改动集中，便于后续逐步演进到多 query、query routing 或 reranking。