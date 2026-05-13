# Query Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 `retrieve_knowledge` 检索入口前增加基于会话上下文的单次查询重写能力，并在重写失败时自动回退到原始 query。

**Architecture:** 保持 `retrieve_knowledge(query)` 和 `HybridSearchService` 的外部行为不变，在工具层前增加 `QueryRewriteService` 作为增强层。`RagAgentService` 通过 `contextvars` 透传当前 `session_id`，查询重写服务读取最近 3 轮会话历史，生成单条适合检索的 rewritten query，再交给现有 hybrid retrieval 链路执行。

**Tech Stack:** Python, FastAPI, LangChain, LangGraph, ChatQwen, Pydantic Settings, contextvars, Loguru

---

### Task 1: Add query rewrite configuration

**Files:**
- Modify: `app/config.py`

**Step 1: Add query rewrite settings**

在 `Settings` 的 RAG 配置段落下加入：

```python
    rag_query_rewrite_enabled: bool = True
    rag_query_rewrite_model: str = ""
    rag_query_rewrite_history_rounds: int = 3
    rag_query_rewrite_timeout: int = 5
    rag_query_rewrite_max_length: int = 200
```

要求：
- 保持和现有 `rag_*` 命名风格一致
- 不新增复杂 validator，保持最小实现

**Step 2: Verify config import**

Run:
```bash
python -c "from app.config import config; print(config.rag_query_rewrite_enabled, config.rag_query_rewrite_history_rounds, config.rag_query_rewrite_max_length)"
```

Expected: 输出 `True 3 200`。

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add query rewrite config"
```

### Task 2: Add request-scoped session context helpers

**Files:**
- Create: `app/core/request_context.py`

**Step 1: Create contextvar storage**

新建 `app/core/request_context.py`，写入：

```python
from contextvars import ContextVar, Token

_current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)


def set_current_session_id(session_id: str) -> Token:
    return _current_session_id.set(session_id)


def get_current_session_id() -> str | None:
    return _current_session_id.get()


def reset_current_session_id(token: Token) -> None:
    _current_session_id.reset(token)
```

要求：
- 只做 `session_id` 透传，不顺手抽象成通用 request context
- 使用标准库 `contextvars`，不引入第三方依赖

**Step 2: Verify import**

Run:
```bash
python -c "from app.core.request_context import get_current_session_id; print(get_current_session_id())"
```

Expected: 输出 `None`。

**Step 3: Commit**

```bash
git add app/core/request_context.py
git commit -m "feat: add request session context helpers"
```

### Task 3: Wire session context into RagAgentService

**Files:**
- Modify: `app/services/rag_agent_service.py`

**Step 1: Import request context helpers**

在文件顶部 imports 中加入：

```python
from app.core.request_context import (
    reset_current_session_id,
    set_current_session_id,
)
```

**Step 2: Set and reset session context in `query()`**

在 `query()` 方法中，`await self._initialize_agent()` 之后、调用 `self.agent.ainvoke(...)` 之前加入：

```python
session_token = set_current_session_id(session_id)
```

并把现有主体包进 `try ... finally`，在 `finally` 中加入：

```python
reset_current_session_id(session_token)
```

要求：
- 只包裹 agent 执行和结果提取阶段
- 保留现有异常日志逻辑
- 不改变返回值结构

**Step 3: Set and reset session context in `query_stream()`**

在 `query_stream()` 中同样加入：

```python
session_token = set_current_session_id(session_id)
```

并在最外层 `try` 中用 `finally` 清理：

```python
reset_current_session_id(session_token)
```

要求：
- 确保流式异常、正常完成两种路径都会 reset
- 不修改现有 token yield 结构

**Step 4: Verify import and module load**

Run:
```bash
python -c "from app.services.rag_agent_service import RagAgentService; print(RagAgentService.__name__)"
```

Expected: 输出 `RagAgentService`。

**Step 5: Commit**

```bash
git add app/services/rag_agent_service.py
git commit -m "feat: pass session id through request context"
```

### Task 4: Add query rewrite service

**Files:**
- Create: `app/services/query_rewrite_service.py`

**Step 1: Create service skeleton and singleton**

新建文件，定义：

```python
class QueryRewriteService:
    def __init__(self) -> None:
        self.model_name = config.rag_query_rewrite_model or config.rag_model
        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0,
            streaming=False,
        )

query_rewrite_service = QueryRewriteService()
```

要求：
- 复用现有 `ChatQwen` 集成
- `temperature=0`
- 不加入额外缓存、重试、复杂策略

**Step 2: Add history extraction helper**

实现一个私有方法，直接从 `rag_agent_service.checkpointer` 读取会话消息，提取最近 N 轮：

```python
def _get_recent_history(self, session_id: str) -> list[dict[str, str]]:
```

要求：
- 使用 `config.rag_query_rewrite_history_rounds` 控制轮数
- 过滤系统消息
- 只保留 `role` 和 `content`
- assistant / user 各自都保留
- 若读不到历史，返回空列表

建议复用 `rag_agent_service.py:316-365` 的读取思路，但不要直接依赖整个 `RagAgentService` 实例方法。

**Step 3: Add prompt builder**

实现：

```python
def _build_rewrite_prompt(self, query: str, history: list[dict[str, str]]) -> str:
```

要求 prompt 明确约束：
- 你在做检索查询重写
- 根据最近对话补全代词、省略和关键实体
- 保留错误码、英文术语、服务名、组件名、数字、时间范围
- 不要回答问题
- 不要编造上下文中不存在的信息
- 只输出一行纯文本 query

历史建议格式：

```text
[user] ...
[assistant] ...
```

最后附上：

```text
当前问题: <query>
输出:
```

**Step 4: Add async rewrite method**

实现：

```python
async def rewrite(self, query: str, session_id: str | None) -> str:
```

逻辑要求：
- `query.strip()` 为空时直接返回原 query
- `rag_query_rewrite_enabled` 为 `False` 时直接返回原 query
- `session_id` 为空时直接返回原 query
- 读取 history
- 构造 prompt
- 用 `asyncio.wait_for(self.model.ainvoke(prompt), timeout=config.rag_query_rewrite_timeout)` 调模型
- 提取文本结果并 `strip()`
- 返回空串或长度超过 `rag_query_rewrite_max_length` 时回退原 query
- 记录 `INFO` / `WARNING` 日志，日志字段至少包含原 query、最终 query、是否回退

**Step 5: Add sync wrapper**

实现：

```python
def rewrite_sync(self, query: str, session_id: str | None) -> str:
    return asyncio.run(self.rewrite(query, session_id))
```

要求：
- 不额外处理事件循环嵌套；按当前仓库风格保持最小实现

**Step 6: Verify import**

Run:
```bash
python -c "from app.services.query_rewrite_service import query_rewrite_service; print(type(query_rewrite_service).__name__)"
```

Expected: 输出 `QueryRewriteService`。

**Step 7: Commit**

```bash
git add app/services/query_rewrite_service.py
git commit -m "feat: add query rewrite service"
```

### Task 5: Route knowledge retrieval through query rewrite

**Files:**
- Modify: `app/tools/knowledge_tool.py`

**Step 1: Import request context and rewrite service**

在 imports 中加入：

```python
from app.core.request_context import get_current_session_id
from app.services.query_rewrite_service import query_rewrite_service
```

**Step 2: Rewrite query before retrieval**

在 `retrieve_knowledge()` 中，把：

```python
docs = hybrid_search_service.search_sync(query, top_k=config.rag_top_k)
```

改成：

```python
session_id = get_current_session_id()
rewritten_query = query_rewrite_service.rewrite_sync(query, session_id)
docs = hybrid_search_service.search_sync(rewritten_query, top_k=config.rag_top_k)
```

要求：
- `session_id` 为 `None` 时仍可安全运行，因为 rewrite service 内部会回退
- 保持工具签名不变
- 保持 `format_docs()` 和返回结构不变

**Step 3: Add retrieval log**

在检索前后保留/补充日志，至少能看到：
- 原始 query
- 最终检索 query

不要打印完整会话历史。

**Step 4: Verify import**

Run:
```bash
python -c "from app.tools.knowledge_tool import retrieve_knowledge; print(retrieve_knowledge.name)"
```

Expected: 正常输出工具名。

**Step 5: Commit**

```bash
git add app/tools/knowledge_tool.py
git commit -m "feat: rewrite knowledge queries before retrieval"
```

### Task 6: Manual verification without adding tests

**Files:**
- Modify: none

**Step 1: Verify all modules import cleanly**

Run:
```bash
python -c "from app.services.rag_agent_service import rag_agent_service; from app.tools.knowledge_tool import retrieve_knowledge; from app.services.query_rewrite_service import query_rewrite_service; print('ok')"
```

Expected: 输出 `ok`。

**Step 2: Run a local rewrite smoke check**

Run:
```bash
python - <<'PY'
from app.services.query_rewrite_service import query_rewrite_service
print(query_rewrite_service.rewrite_sync('这个报错怎么处理', 'demo-session'))
PY
```

Expected: 命令能返回字符串；如果没有历史或模型不可用，也应安全回退而不是崩溃。

**Step 3: Run app import check**

Run:
```bash
python -c "from app.main import app; print(app.title)"
```

Expected: 正常输出应用标题，无导入错误。

**Step 4: Smoke check retrieval path**

启动应用后，用现有 `/api/chat` 或等价入口发送一组多轮问题，例如：
1. `payment-service 出现 CrashLoopBackOff`
2. `这个报错怎么处理`

Expected:
- 服务无报错
- 日志能看到 query rewrite 被调用
- 第二轮的最终检索 query 包含 `payment-service` 和 `CrashLoopBackOff` 这类上文实体，或在失败时明确回退原 query

**Step 5: Commit any verification-driven fixes**

```bash
git status --short
```

Expected: 只有预期文件变更，没有额外意外修改。
