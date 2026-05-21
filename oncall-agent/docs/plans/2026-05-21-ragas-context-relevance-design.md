# Ragas Context Relevance Design

> **Goal:** 在现有离线 ragas 评测框架中补充“检索上下文是否与问题相关”的评测维度，通过把评测时真实检索到的内容显式传给 ragas，新增对检索质量的离线观测能力。

**Architecture:** 保持当前离线评测入口与报告结构不变，在单条样本执行过程中额外采集真实检索结果，写入评测明细，并在构造 ragas dataset 时显式增加 `retrieved_contexts` 字段，用于执行 `ContextRelevance` 指标。回答质量指标继续保留，与检索相关性指标同批次输出。

**Tech Stack:** Python, ragas, datasets, JSONL, pathlib

---

## 1. 目标与范围

本次目标是补充“检索结果本身是否和问题相关”的离线评测能力，不改变线上问答主链路，也不重构现有评测工具结构。

本次范围：
- 保留现有离线评测入口
- 保留现有回答质量指标
- 在评测时采集真实检索上下文
- 新增 `ContextRelevance` 指标
- 在报告中输出每条样本的检索内容与相关评分

本次不做：
- 在线服务化评测
- 独立拆分新的检索评测系统
- 在第一版加入 context precision / context recall
- 固化检索结果到数据集文件

## 2. 方案选择

采用低改动方案：在现有评测流程内补齐检索上下文采集与 ragas 字段映射，而不是拆出独立检索评测子流程。

选择该方案的原因：
- 对现有代码侵入最小
- 能直接复用已有样本读取、回答生成、报告输出流程
- 一次执行即可同时得到回答质量与检索质量结果
- 为后续扩展更多 context 指标保留空间

## 3. 数据结构调整

现有 `EvalDetail` 仅记录 question、ground_truth、answer、scores、error，需要补充检索上下文字段。

建议新增：
- `retrieved_contexts: list[str]`

用途：
- 保留每条样本实际检索到的文本列表
- 为 ragas 的 context 类指标提供显式输入
- 让最终报告可以直接定位低分样本的检索问题

如果后续需要更强分析能力，再考虑增加文档 id、来源、分数等结构化信息；本次只保留文本内容列表即可。

## 4. 评测流程调整

当前流程是：
1. 读取数据集
2. 逐条生成最终回答
3. 组装 ragas dataset
4. 执行评分

调整后流程是：
1. 读取数据集
2. 对每条样本执行检索并收集 `retrieved_contexts`
3. 基于同一批检索结果生成最终回答
4. 组装包含 `question`、`answer`、`ground_truth`、`retrieved_contexts` 的 ragas dataset
5. 执行回答质量指标与 `ContextRelevance` 指标
6. 输出总体摘要与逐条明细

这样可以保证：
- context 指标评估的是评测当次真实检索结果
- `faithfulness` 与 context 指标共享同一份上下文输入
- 评测结果更容易对齐排查

## 5. Ragas 输入字段

当前 dataset payload 仅包含：
- `question`
- `answer`
- `ground_truth`

需要新增：
- `retrieved_contexts`

每条样本的 `retrieved_contexts` 应为字符串列表，内容是检索阶段返回并实际参与回答生成的文本片段。

如果现有检索返回对象包含 metadata，本次在送入 ragas 前只提取文本字段，避免第一版引入额外映射复杂度。

## 6. 指标设计

现有指标保留：
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`

本次新增：
- `ContextRelevance`

指标职责划分：
- `ContextRelevance`：检索上下文和问题是否相关
- `faithfulness`：回答内容是否被检索上下文支撑
- `answer_relevancy`：回答是否切题
- `answer_correctness`：回答是否接近参考答案

这样可以把“检索是否找对”和“回答是否答对”分开观察。

## 7. 检索上下文采集策略

实现关键不在 ragas，而在于如何从现有知识检索链路中拿到评测时真实使用的检索结果。

推荐优先级：
1. 复用现有检索函数，新增一个评测可调用的显式返回接口
2. 如果回答生成链路内部已经持有检索结果，则让评测流程可读出该结果
3. 避免通过日志解析或旁路推断检索内容

目标是让评测代码拿到“真实检索内容”，而不是手工构造或复刻一份近似结果。

## 8. 报告输出调整

逐条结果建议新增输出：
- `retrieved_contexts`
- `context_relevance` 得分

总体摘要中新增：
- `context_relevance` 均分

这样报告既可用于总体对比，也可直接定位某条样本是“没检到对的内容”还是“检到了但回答没用好”。

## 9. 风险与约束

主要约束：
- 需要确认当前检索链路是否存在稳定的可复用入口
- 需要确认 ragas 当前版本下 `ContextRelevance` 的字段命名与调用方式
- 若回答生成流程内部包含 query rewrite 或 rerank，评测采集的上下文应尽量与真实生成链路保持一致

主要风险：
- 如果评测侧单独调用检索逻辑，但与真实回答链路不一致，结果会失真
- 如果上下文文本过长，可能影响评测成本与稳定性

第一版以“尽量复用真实链路、最少额外抽象”为原则。

## 10. 后续扩展

后续可按需扩展：
- `ContextPrecision`
- `ContextRecall`
- rerank 前后检索质量对比
- query rewrite 前后检索质量对比
- 将检索结果固化到数据集，形成稳定 benchmark
