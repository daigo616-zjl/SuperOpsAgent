"""剧本化 Mock 数据加载器。

通过环境变量 ``MOCK_SCENARIO`` 选择故障剧本（默认 ``no-fault``）。
所有剧本文件位于 ``mcp_servers/scenarios/*.yaml``，每个剧本包含：
- 告警清单（供 ``query_active_alerts`` 返回）
- 指标曲线塑形参数（供 monitor_server 生成 CPU/内存曲线）
- 注入日志的错误/警告模式（供 cls_server 混入噪声日志）
- ground_truth.root_cause（供 P5 基准评判根因命中率）

确定性保证：任意 ``(scenario_id, topic, minute_bucket)`` 三元组经
``minute_seed`` 映射为固定随机种子，保证同一时间窗内多次查询、
分页偏移之间生成的日志完全一致。
"""

from __future__ import annotations

import hashlib
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
DEFAULT_SCENARIO = "no-fault"
SCENARIO_ENV_VAR = "MOCK_SCENARIO"

# 默认噪声密度（行/分钟），剧本可用 noise.lines_per_minute 覆盖
DEFAULT_NOISE_RANGE = (20, 60)


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


@lru_cache(maxsize=None)
def load_scenario(name: str) -> dict[str, Any]:
    path = SCENARIO_DIR / f"{name}.yaml"
    if not path.is_file():
        available = ", ".join(list_scenarios())
        raise FileNotFoundError(
            f"未找到剧本 '{name}'（MOCK_SCENARIO={SCENARIO_ENV_VAR}）。可用剧本: {available}"
        )
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict) or "id" not in scenario:
        raise ValueError(f"剧本文件格式非法: {path}")
    if "ground_truth" not in scenario or not scenario["ground_truth"].get("root_cause"):
        raise ValueError(f"剧本缺少 ground_truth.root_cause: {path}")
    return scenario


def get_active_scenario_name() -> str:
    return os.environ.get(SCENARIO_ENV_VAR, DEFAULT_SCENARIO).strip() or DEFAULT_SCENARIO


def get_active_scenario() -> dict[str, Any]:
    return load_scenario(get_active_scenario_name())


def minute_seed(scenario_id: str, topic: str, minute_bucket: str) -> int:
    digest = hashlib.sha256(f"{scenario_id}|{topic}|{minute_bucket}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(scenario_id: str, topic: str, minute_bucket: str) -> random.Random:
    return random.Random(minute_seed(scenario_id, topic, minute_bucket))


def noise_range(scenario: dict[str, Any]) -> tuple[int, int]:
    override = (scenario.get("noise") or {}).get("lines_per_minute")
    if isinstance(override, (list, tuple)) and len(override) == 2:
        return int(override[0]), int(override[1])
    return DEFAULT_NOISE_RANGE
