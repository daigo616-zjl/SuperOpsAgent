# Ragas Evaluation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在项目内新增一个独立的离线 ragas 评测模块，读取用户提供的测试集并自动输出最终回答质量评分。

**Architecture:** 评测能力只作为项目内工具存在，不接入现有 API、前端或在线服务链路。评测脚本读取 JSONL 测试集，逐条调用现有最终回答生成逻辑，整理成 ragas 所需的评测数据后执行批量评分，并把汇总结果与逐条结果落盘到报告目录。

**Tech Stack:** Python, ragas, JSONL, pathlib, argparse, loguru

---

## 1. 目标与边界

本次只做“最终回答效果”评测，不做检索层单独评分，也不把评测能力暴露成服务接口。

明确边界：
- 评测代码保留在仓库内
- 输入是用户构建的数据集文件
- 输出是自动评分结果
- 不改线上主链路
- 不做在线评测、不做前端集成

## 2. 数据集格式

第一版采用 JSONL，每行一条样本，便于后续手工维护和增量追加。

最低字段：
- `id`
- `question`
- `ground_truth`

建议预留字段：
- `metadata`
- `reference_contexts`（后续如果想分析检索质量再补）

目录建议：
- `eval/datasets/`：测试集
- `eval/reports/`：评测结果
- `eval/fixtures/`：示例数据

## 3. 评测流程

1. 读取 JSONL 测试集
2. 校验样本字段
3. 逐条生成最终回答
4. 组装 ragas 输入数据
5. 执行批量评分
6. 输出总体摘要和逐条明细
7. 失败样本记录错误并继续执行

## 4. 指标选择

第一版只评估最终回答质量，建议优先使用：
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`（依赖可用时启用）

后续如有需要，再扩展更多指标或分组统计。

## 5. 结果产物

每次执行都输出一个报告文件，包含：
- 数据集信息
- 总样本数
- 各项指标均分
- 每条样本的 question、answer、scores、error

报告建议保存为 JSON，方便后续脚本或人工读取。

## 6. 明确不做

第一版不做：
- 测试集自动生成
- 在线评测接口
- 可视化大盘
- 检索层单独评测
- 复杂失败重试

## 7. 后续扩展

后续可以按需加入：
- 自动生成测试集
- 更多 ragas 指标
- 多数据集批跑
- 按分类字段聚合统计
- CSV 或 HTML 报告
