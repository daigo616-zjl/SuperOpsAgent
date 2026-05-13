# Cross-Encoder Rerank Design

> 目标：在现有 query rewrite + hybrid retrieval 链路后增加一层开源 cross-encoder rerank，对 hybrid 召回的 20 个候选进行重排序，最终取前 5 个结果给 LLM。

## 1. 背景与目标

当前知识检索链路已经具备两层增强：

1. `query rewrite`：结合最近会话上下文，将用户问题改写为更适合检索的单条 query。
2. `hybrid retrieval`：通过向量检索 + Elasticsearch BM25 双路召回，并用 RRF 融合得到最终候选。

现状下，`retrieve_knowledge()` 会直接把 hybrid 返回的结果交给 `format_docs()` 提供给 LLM。该链路能够显著改善召回覆盖，但仍存在一个问题：

- 候选被召回出来，不代表顺序已经最优。
- 尤其当 `rag_recall_size` 增大到 20 时，前几条文档未必是与当前 query 最相关的内容。

因此需要在“召回完成”之后增加一层最终精排，让 LLM 接收到的上下文更干净、更相关。

本次设计目标明确限定为：

- 保留现有 `query rewrite -> hybrid retrieval` 主体架构；
- 对 hybrid 的候选结果执行一次 cross-encoder rerank；
- 候选数固定为 20，精排后返回前 5；
- rerank 异常时回退到 hybrid 原始前 5，保证主链路可用。

## 2. 方案决策

### 2.1 已确认决策

| 决策项 | 结论 |
| --- | --- |
| rerank 类型 | 使用普通开源 cross-encoder rerank 模型 |
| 接入位置 | 放在 `retrieve_knowledge` 内部，位于 hybrid retrieval 之后、`format_docs` 之前 |
| 候选规模 | `hybrid recall 20 -> rerank -> top 5` |
| 失败策略 | rerank 失败时回退到 hybrid 前 5 |
| 部署方式 | 进程内直接加载模型，不单独拆服务 |
| 最终返回数量 | 继续由 `rag_top_k` 控制，默认值调整为 5 |

### 2.2 选择原因

该方案的核心思路是：召回与精排解耦。

- `HybridSearchService` 继续专注“召回 + 融合”；
- `RerankService` 只负责“对候选做相关性精排”；
- `knowledge_tool` 作为编排层串联 rewrite / retrieval / rerank / format。

这样做的好处是：

- 保持现有 tool 对外行为不变；
- 不让 hybrid service 职责膨胀；
- 后续如果要替换模型或把 rerank 独立部署，只需替换 rerank 层实现。

## 3. 架构落点

### 3.1 新链路

第一版目标链路如下：

```text
用户问题
  -> query rewrite
  -> hybrid recall (20)
  -> cross-encoder rerank
  -> top 5
  -> format_docs
  -> LLM
```

其中：

- query rewrite 继续输出适合检索的 `rewritten_query`
- hybrid retrieval 使用 `rewritten_query` 返回 20 个 `Document`
- rerank 也使用同一个 `rewritten_query` 对 20 个文档再次打分
- 最终只保留前 5 个 `Document` 给 `format_docs()`

### 3.2 为什么 rerank 不放进 `HybridSearchService`

如果将 rerank 直接并入 `HybridSearchService`，该服务会同时承担：

1. 双路召回
2. RRF 融合
3. 模型精排

这会使服务职责开始发散。当前阶段更合理的分层应是：

- 检索层：负责召回
- 精排层：负责重排序
- 工具层：负责整条链路编排

因此 rerank 应作为 retrieval enhancement，而不是 retrieval core logic。

## 4. 服务设计

### 4.1 新增服务

新增文件：

- `app/services/rerank_service.py`

建议的服务接口：

```python
class RerankService:
    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        ...
```

### 4.2 职责边界

`RerankService` 的职责仅包括：

1. 接收 `query` 和候选 `docs`
2. 构造 `(query, doc.page_content)` pairs
3. 调用 cross-encoder 模型为每个文档打分
4. 按分数降序排序
5. 返回前 `top_k` 个原始 `Document`
6. 在异常场景下降级到原始输入的前 `top_k`

明确不负责：

- 召回逻辑
- 文档格式化
- query rewrite
- BM25 / 向量分融合
- 多模型调度

### 4.3 模型加载策略

第一版采用进程内模型加载，建议使用懒加载：

- 应用启动时不强制加载 rerank 模型；
- 第一次进入 rerank 时再初始化模型；
- 后续复用同一个模型实例。

这样可以避免：

- 启动时额外拉长应用初始化时间；
- 本地开发或非检索场景下无意义占用内存；
- rerank 初始化失败直接影响整个服务启动。

## 5. 代码改动范围

### 5.1 `app/tools/knowledge_tool.py`

当前逻辑：

- 读取 `session_id`
- 调用 `query_rewrite_service.rewrite_sync()`
- 调用 `hybrid_search_service.search_sync(rewritten_query, top_k=config.rag_top_k)`
- 直接 `format_docs()`

建议调整为：

1. 先调用 `hybrid_search_service.search_sync(rewritten_query, top_k=config.rag_recall_size)`
2. 再调用 `rerank_service.rerank(rewritten_query, docs, top_k=config.rag_top_k)`
3. 使用 rerank 后的结果继续 `format_docs()`

这里强调使用 `rewritten_query` 作为 rerank 输入，而不是原始 query。原因是：

- 召回阶段已经基于 rewritten query 获取候选；
- 如果 rerank 改用原始 query，排序标准会与召回标准错位；
- 当前系统已经明确把 rewritten query 视为“更适合 retrieval 的语义表达”，rerank 应沿用同一语义基准。

### 5.2 `app/services/hybrid_search_service.py`

原则上不需要改变其职责，只保持：

- 双路召回
- RRF 融合
- 返回 `Document` 列表

唯一需要配合的是：

- `knowledge_tool` 不再用 `config.rag_top_k` 作为 hybrid 的截断值；
- hybrid 返回规模改由 `config.rag_recall_size` 控制。

### 5.3 `app/config.py`

建议配置调整如下：

- 保留 `rag_recall_size = 20`：召回候选数
- 保留 `rag_top_k` 作为最终返回给 LLM 的文档数量，默认值从 `3` 调整为 `5`
- 新增：
  - `rag_rerank_enabled: bool = True`
  - `rag_rerank_model: str = ""`
  - `rag_rerank_timeout: int = 10`

第一版不额外新增 `rag_rerank_candidate_k`，直接复用 `rag_recall_size`，避免配置语义重复。

### 5.4 `pyproject.toml`

需要补充进程内运行 open-source reranker 所需依赖。具体包选择可以在 implementation plan 阶段再定，但设计上已确定此处会有变更。

## 6. 异常处理与降级

### 6.1 降级原则

系统已明确选择：

- rerank 失败时，不中断主链路；
- 直接回退到 hybrid 返回结果的前 `rag_top_k`。

### 6.2 触发降级的异常范围

以下情况都应视为 rerank 失败：

- 模型初始化失败
- 模型推理异常
- 推理超时
- 返回分数数量与文档数量不一致
- query 为空或 docs 为空时无法执行有效 rerank

这些错误不应向 `retrieve_knowledge()` 上抛成致命错误，而应在 `RerankService` 内部处理并返回 fallback 结果。

### 6.3 不做的复杂策略

第一版明确不做：

- 自动重试
- 多级 fallback 模型
- rerank 结果缓存
- 分数阈值裁剪
- 基于 rerank score 的混合加权回写

目标是优先验证 rerank 是否能稳定提升上下文质量，而不是一次性做成完整检索平台。

## 7. 日志与可观测性

建议在 rerank 层补充最小可观测日志。

### 7.1 正常日志

至少记录：

- original query
- rewritten query
- hybrid 返回数量
- rerank 输入数量
- rerank 输出数量
- rerank 耗时

### 7.2 降级日志

当发生降级时，记录：

- 降级原因
- fallback 行为（返回 hybrid top_k）

例如日志语义应能表达：

- `Rerank 完成: candidates=20, returned=5, duration_ms=...`
- `Rerank 降级: reason=model_timeout, fallback=hybrid_top_k`

这能帮助排查两个关键问题：

1. rerank 是否真实执行
2. 当回答质量异常时，是排序效果问题，还是 rerank 压根没有成功跑通

## 8. 第一版明确不做

为了避免范围膨胀，本次设计明确排除以下内容：

- 独立 rerank HTTP / gRPC 服务
- GPU / CPU 自动调度
- 多模型切换与 A/B 实验
- 标题字段额外加权
- 原始 query 与 rewritten query 双路打分
- 动态扩召回池（如 50 -> rerank -> 5）
- rerank score 阈值过滤
- 结果缓存或离线评估框架

这些都可以作为后续演进方向，但不属于当前最小可落地范围。

## 9. 最终结论

本次 rerank 设计结论如下：

- 在现有 `query rewrite -> hybrid retrieval` 后插入一层进程内 open-source cross-encoder rerank；
- hybrid retrieval 负责召回 20 个候选；
- `RerankService` 使用 `rewritten_query` 对 20 个 `Document` 重新打分排序；
- 最终返回前 5 个结果给 `format_docs()` 和 LLM；
- 若 rerank 失败，则直接回退到 hybrid 前 5；
- 保持 tool 签名不变，最小化对现有系统的侵入。

该设计兼顾了最小改动、可用性和后续可演进性，适合作为现有 RAG 检索链路的下一步增强。