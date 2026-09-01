"""AIOps 星型多 Agent 编排。

中心化 Supervisor 确定性路由：hypothesize → Send 并行取证
（metrics/logs/knowledge）→ adjudicate → converge → reporter。
Agent 间零直连，一切消息经 Supervisor 中转。
"""
