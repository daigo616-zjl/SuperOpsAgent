"""P1 Mock 剧本化测试：剧本指纹可区分、日志生成确定、分页稳定、查询语法正确。

全部直调 mcp_servers 模块级 helper（fastmcp 会把工具包装成 FunctionTool），
不启动 MCP 服务、不消耗 token。
"""

from datetime import datetime

import pytest

from mcp_servers import cls_server as cls
from mcp_servers import monitor_server as monitor
from mcp_servers import scenario_loader

BASE = datetime(2026, 8, 31, 10, 0, 0)
BASE_MS = int(BASE.timestamp() * 1000)
END_MS = BASE_MS + 5 * 60 * 1000  # 5 分钟窗口

FAULT_SCENARIOS = ["gc-pressure", "db-slow-query", "oom-kill", "distractor-cpu"]

SCENARIO_SIGNATURES = {
    "gc-pressure": "GC pause",
    "db-slow-query": "timeout waiting for connection from pool",
    "oom-kill": "OutOfMemoryError",
    "distractor-cpu": "thread pool 'sync-worker' exhausted",
}


def _use_scenario(monkeypatch, name: str) -> None:
    monkeypatch.setenv("MOCK_SCENARIO", name)


def _logs(monkeypatch, name: str):
    _use_scenario(monkeypatch, name)
    scenario = scenario_loader.get_active_scenario()
    return cls._generate_logs_for_window(scenario, BASE_MS, END_MS)


# ------------------------------------------------------------
# 剧本加载与指纹
# ------------------------------------------------------------

def test_all_scenarios_load_with_ground_truth():
    names = scenario_loader.list_scenarios()
    assert set(FAULT_SCENARIOS + ["no-fault"]) <= set(names)
    for name in names:
        scenario = scenario_loader.load_scenario(name)
        assert scenario["ground_truth"]["root_cause"], name


def test_unknown_scenario_raises_with_available_list():
    with pytest.raises(FileNotFoundError) as exc_info:
        scenario_loader.load_scenario("no-such-scenario")
    assert "gc-pressure" in str(exc_info.value)


def test_minute_seed_deterministic_and_varies():
    seed_1 = scenario_loader.minute_seed("gc-pressure", "logs", "2026-08-31 10:00")
    seed_2 = scenario_loader.minute_seed("gc-pressure", "logs", "2026-08-31 10:00")
    seed_3 = scenario_loader.minute_seed("gc-pressure", "logs", "2026-08-31 10:01")
    seed_4 = scenario_loader.minute_seed("oom-kill", "logs", "2026-08-31 10:00")
    assert seed_1 == seed_2
    assert seed_1 != seed_3
    assert seed_1 != seed_4


def test_active_scenario_follows_env(monkeypatch):
    _use_scenario(monkeypatch, "oom-kill")
    assert scenario_loader.get_active_scenario_name() == "oom-kill"
    assert scenario_loader.get_active_scenario()["id"] == "oom-kill"
    _use_scenario(monkeypatch, "no-fault")
    assert scenario_loader.get_active_scenario()["id"] == "no-fault"


@pytest.mark.parametrize("name,signature", SCENARIO_SIGNATURES.items())
def test_fault_scenarios_inject_discriminative_log_signature(monkeypatch, name, signature):
    logs = _logs(monkeypatch, name)
    messages = [entry[2] for entry in logs]
    assert any(signature in message for message in messages), (name, signature)
    # 噪声体量：5 分钟窗口应有数百行混合日志
    assert len(logs) >= 5 * 15


def test_no_fault_scenario_has_no_injected_errors(monkeypatch):
    logs = _logs(monkeypatch, "no-fault")
    for _, level, message in logs:
        if level == "ERROR":
            # 仅允许低频"已自愈"噪声错误
            assert "已自愈" in message
    scenario = scenario_loader.load_scenario("no-fault")
    assert scenario["alerts"] == []
    assert scenario["log_patterns"] == []


# ------------------------------------------------------------
# 确定性与分页稳定性
# ------------------------------------------------------------

def test_log_generation_is_deterministic(monkeypatch):
    first = _logs(monkeypatch, "gc-pressure")
    second = _logs(monkeypatch, "gc-pressure")
    assert first == second


def test_pagination_stable_and_no_overlap(monkeypatch):
    _use_scenario(monkeypatch, "gc-pressure")
    scenario = scenario_loader.get_active_scenario()
    all_logs = [
        entry for entry in cls._generate_logs_for_window(scenario, BASE_MS, END_MS)
        if BASE_MS <= entry[0] <= END_MS
    ]
    assert len(all_logs) > 30  # 保证有分页空间

    page_size = 20
    page_1 = cls.search_log.fn(
        topic_id="topic-001", start_time=BASE_MS, end_time=END_MS, limit=page_size, offset=0
    )
    page_2 = cls.search_log.fn(
        topic_id="topic-001", start_time=BASE_MS, end_time=END_MS,
        limit=page_size, offset=page_1["next_offset"],
    )
    assert page_1["has_more"] is True
    assert page_1["total"] == page_size
    assert page_1["total_matched"] == len(all_logs)
    assert page_2["total"] == page_size
    # 分页内容与确定性全量切片一致，且页间无重叠
    expected_page_1 = all_logs[:page_size]
    expected_page_2 = all_logs[page_size:2 * page_size]
    assert [(log["level"], log["message"]) for log in page_1["logs"]] == [
        (level, message) for _, level, message in expected_page_1
    ]
    assert [(log["level"], log["message"]) for log in page_2["logs"]] == [
        (level, message) for _, level, message in expected_page_2
    ]


# ------------------------------------------------------------
# 查询语法 / 主题 / 上限
# ------------------------------------------------------------

def test_query_syntax_level_and_keyword(monkeypatch):
    _use_scenario(monkeypatch, "gc-pressure")
    result = cls.search_log.fn(
        topic_id="topic-001", start_time=BASE_MS, end_time=END_MS,
        query='level:ERROR AND "GC overhead"', limit=100,
    )
    assert result["total_matched"] > 0
    for log in result["logs"]:
        assert log["level"] == "ERROR"
        assert "gc overhead" in log["message"].lower()


def test_query_keyword_case_insensitive(monkeypatch):
    _use_scenario(monkeypatch, "oom-kill")
    lower = cls.search_log.fn(
        topic_id="topic-001", start_time=BASE_MS, end_time=END_MS,
        query='"outofmemoryerror"', limit=100,
    )
    upper = cls.search_log.fn(
        topic_id="topic-001", start_time=BASE_MS, end_time=END_MS,
        query='"OutOfMemoryError"', limit=100,
    )
    assert lower["total_matched"] == upper["total_matched"] > 0


def test_error_topic_only_returns_error_level(monkeypatch):
    _use_scenario(monkeypatch, "gc-pressure")
    result = cls.search_log.fn(
        topic_id="topic-002", start_time=BASE_MS, end_time=END_MS, limit=1000,
    )
    assert result["total"] > 0
    assert all(log["level"] == "ERROR" for log in result["logs"])


def test_limit_capped_at_1000(monkeypatch):
    _use_scenario(monkeypatch, "gc-pressure")
    result = cls.search_log.fn(
        topic_id="topic-001", start_time=BASE_MS, end_time=END_MS, limit=2000,
    )
    assert result["limit"] == 1000


def test_unknown_topic_returns_error(monkeypatch):
    _use_scenario(monkeypatch, "gc-pressure")
    result = cls.search_log.fn(
        topic_id="topic-999", start_time=BASE_MS, end_time=END_MS,
    )
    assert result["error"].startswith("主题不存在")
    assert result["logs"] == []


# ------------------------------------------------------------
# 监控：告警与指标塑形
# ------------------------------------------------------------

def test_alerts_per_scenario(monkeypatch):
    _use_scenario(monkeypatch, "gc-pressure")
    response = monitor._active_alerts_response("data-sync-service", None)
    names = [alert["alert_name"] for alert in response["alerts"]]
    assert "JvmGCPauseHigh" in names
    critical = monitor._active_alerts_response("data-sync-service", "critical")
    assert {alert["severity"] for alert in critical["alerts"]} == {"critical"}


def test_alerts_empty_for_no_fault(monkeypatch):
    _use_scenario(monkeypatch, "no-fault")
    response = monitor._active_alerts_response("data-sync-service", None)
    assert response["total"] == 0 and response["alerts"] == []


def _metric(monkeypatch, scenario_name, metric_key, metric_name, threshold, jitter, pressure_key,
            window_minutes=60):
    _use_scenario(monkeypatch, scenario_name)
    end = datetime(2026, 8, 31, 10 + window_minutes // 60, window_minutes % 60, 0)
    return monitor._generate_metric_response(
        service_name="data-sync-service",
        metric_key=metric_key,
        metric_name=metric_name,
        default_start=10.0,
        default_peak=96.0,
        alert_threshold=threshold,
        jitter=jitter,
        pressure_key=pressure_key,
        alert_message_high="test",
        start_time="2026-08-31 10:00:00",
        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        interval="1m",
    )


def test_metric_curves_distinguish_scenarios(monkeypatch):
    gc_memory = _metric(monkeypatch, "gc-pressure", "memory", "memory_usage_percent",
                        70.0, 1.0, "memory_pressure")
    normal_memory = _metric(monkeypatch, "db-slow-query", "memory", "memory_usage_percent",
                            70.0, 1.0, "memory_pressure")
    assert gc_memory["statistics"]["max"] > 80
    assert normal_memory["statistics"]["max"] < 60
    assert gc_memory["statistics"]["memory_pressure"] is True
    assert normal_memory["statistics"]["memory_pressure"] is False


def test_distractor_cpu_spikes_like_legacy(monkeypatch):
    cpu = _metric(monkeypatch, "distractor-cpu", "cpu", "cpu_usage_percent",
                  80.0, 2.0, "spike_detected")
    assert cpu["statistics"]["max"] > 80
    assert cpu["statistics"]["spike_detected"] is True


def test_metric_curve_is_deterministic(monkeypatch):
    first = _metric(monkeypatch, "oom-kill", "memory", "memory_usage_percent",
                    70.0, 1.0, "memory_pressure")
    second = _metric(monkeypatch, "oom-kill", "memory", "memory_usage_percent",
                     70.0, 1.0, "memory_pressure")
    assert first["data_points"] == second["data_points"]


def test_sawtooth_memory_shows_restart_cycles(monkeypatch):
    response = _metric(monkeypatch, "oom-kill", "memory", "memory_usage_percent",
                       70.0, 1.0, "memory_pressure")
    values = [point["value"] for point in response["data_points"]]
    # 锯齿特征：出现过 >90% 的峰值，且峰值后回落到 50% 以下
    peak_index = values.index(max(values))
    assert max(values) > 90
    assert min(values) < 50


def test_memory_points_carry_gb_fields(monkeypatch):
    response = _metric(monkeypatch, "gc-pressure", "memory", "memory_usage_percent",
                       70.0, 1.0, "memory_pressure")
    point = response["data_points"][0]
    assert point["total_gb"] == 8.0
    assert point["used_gb"] == round(point["value"] / 100.0 * 8.0, 2)
