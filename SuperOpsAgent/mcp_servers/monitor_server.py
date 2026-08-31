"""智能运维监控 MCP Server

本地实现的监控服务 MCP Server，提供：
- 监控数据查询（CPU、内存、磁盘、网络等）
- 进程信息查询
- 历史工单查询
- 服务信息查询

用于支持运维 Agent 的故障排查场景。
"""

import logging
import functools
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

try:
    from mcp_servers.scenario_loader import get_active_scenario, rng_for
except ImportError:  # 以脚本方式直接运行时
    from scenario_loader import get_active_scenario, rng_for

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Monitor_MCP_Server")

mcp = FastMCP("Monitor")


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数（排除self等）
        if kwargs:
            # 使用 json.dumps 格式化参数，处理可能的序列化错误
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                summary = {k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                          for k, v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


# ============================================================
# 辅助函数
# ============================================================

def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间。

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # 返回默认时间（当前时间 + 偏移）
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """生成时间序列字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量
        format_str: 时间格式字符串

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime(format_str)


# ============================================================
# 剧本化指标曲线生成
# ============================================================

def _parse_interval(interval: str) -> int:
    interval_minutes = 1
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60
    return interval_minutes


def _curve_value(mode: str, start: float, peak: float, t_index: int,
                 total_points: int, rng, jitter: float) -> float:
    """按剧本模式计算第 t_index 个数据点的基础值。"""
    if mode == "spike_up":
        # 旧版 CPU 曲线形态：低位几分钟后快速冲高（干扰项场景）
        if t_index < 3:
            value = start + t_index * 0.5
        else:
            growth = (peak - start) / 8.0
            value = min(start + (t_index - 2) * growth, peak)
    elif mode == "step_climb":
        # 阶梯式抬升（GC 压力下的内存特征）
        step = max(1, total_points // 5)
        progress = min(1.0, (t_index // step + 1) / 5.0)
        value = start + (peak - start) * progress
    elif mode == "sawtooth_oom":
        # 锯齿状：爬升至峰值后骤降（OOMKilled 重启特征）
        cycle = 12
        r = t_index % cycle
        value = start + (peak - start) * (r / (cycle - 1))
    elif mode == "plateau_moderate":
        # 中度升高后维持平台
        progress = min(1.0, t_index / 8.0)
        value = start + (peak - start) * progress
    else:  # normal
        value = start
    value += rng.uniform(-jitter, jitter)
    return round(max(0.0, min(100.0, value)), 1)


def _generate_metric_response(
    service_name: str,
    metric_key: str,
    metric_name: str,
    default_start: float,
    default_peak: float,
    alert_threshold: float,
    jitter: float,
    pressure_key: str,
    alert_message_high: str,
    start_time: Optional[str],
    end_time: Optional[str],
    interval: str,
) -> Dict[str, Any]:
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)
    interval_minutes = _parse_interval(interval)
    scenario = get_active_scenario()
    cfg = (scenario.get("metrics") or {}).get(metric_key) or {}
    mode = cfg.get("mode", "normal")
    start = float(cfg.get("start", default_start))
    peak = float(cfg.get("peak", default_peak))
    total_points = max(1, int((end_dt - start_dt).total_seconds() // 60) + 1)

    data_points = []
    current_time = start_dt
    t_index = 0
    while current_time <= end_dt:
        rng = rng_for(scenario["id"], f"metric:{metric_key}",
                      current_time.strftime("%Y%m%d%H%M"))
        value = _curve_value(mode, start, peak, t_index, total_points, rng, jitter)
        point: Dict[str, Any] = {
            "timestamp": current_time.strftime("%H:%M"),
            "value": value,
        }
        if metric_name == "memory_usage_percent":
            point["used_gb"] = round(value / 100.0 * 8.0, 2)
            point["total_gb"] = 8.0
        data_points.append(point)
        current_time += timedelta(minutes=interval_minutes)
        t_index += 1

    if not data_points:
        return {
            "service_name": service_name,
            "metric_name": metric_name,
            "interval": interval,
            "data_points": [],
            "statistics": {},
        }

    values = [d["value"] for d in data_points]
    max_value = max(values)
    pressure = max_value > alert_threshold
    statistics = {
        "avg": round(sum(values) / len(values), 2),
        "max": max_value,
        "min": min(values),
        pressure_key: pressure,
        "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else max_value, 2),
    }
    return {
        "service_name": service_name,
        "metric_name": metric_name,
        "interval": interval,
        "data_points": data_points,
        "statistics": statistics,
        "alert_info": {
            "triggered": pressure,
            "threshold": alert_threshold,
            "message": alert_message_high if pressure else f"{metric_name} 正常",
        },
    }


# ============================================================
# 监控数据查询工具
# ============================================================

@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询服务的 CPU 使用率监控数据。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"
        
        start_time: 开始时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 10:00:00"
            默认值: 如果不传，默认为当前时间的1小时前
            注意: 必须使用字符串格式，而非时间戳
        
        end_time: 结束时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 11:00:00"
            默认值: 如果不传，默认为当前时间
            注意: 必须使用字符串格式，而非时间戳
        
        interval: 数据聚合间隔（可选）
            可选值: "1m" (1分钟), "5m" (5分钟), "1h" (1小时)
            默认值: "1m"
            说明: 控制数据点的时间间隔

    Returns:
        Dict: CPU 监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (cpu_usage_percent)
            - interval: 数据聚合间隔
            - data_points: 数据点列表，每个点包含:
                * timestamp: 时间点（格式: HH:MM）
                * value: CPU 使用率百分比
            - statistics: 统计信息
                * average: 平均值
                * max: 最大值
                * min: 最小值
            - alert: 告警信息（如有）
                * triggered: 是否触发告警
                * threshold: 告警阈值
                * message: 告警消息
    
    使用示例:
        # 示例1: 使用默认时间（最近1小时）
        query_cpu_metrics(service_name="data-sync-service")
        
        # 示例2: 指定时间范围
        query_cpu_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00",
            end_time="2026-02-14 11:00:00",
            interval="5m"
        )
        
        # 示例3: 只指定开始时间（结束时间自动为当前时间）
        query_cpu_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00"
        )
    """
    return _generate_metric_response(
        service_name=service_name,
        metric_key="cpu",
        metric_name="cpu_usage_percent",
        default_start=10.0,
        default_peak=96.0,
        alert_threshold=80.0,
        jitter=2.0,
        pressure_key="spike_detected",
        alert_message_high="CPU 使用率持续超过 80% 阈值",
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询服务的内存使用监控数据。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"
        
        start_time: 开始时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 10:00:00"
            默认值: 如果不传，默认为当前时间的1小时前
            注意: 必须使用字符串格式，而非时间戳
        
        end_time: 结束时间（可选，字符串类型）
            格式: "YYYY-MM-DD HH:MM:SS"
            示例: "2026-02-14 11:00:00"
            默认值: 如果不传，默认为当前时间
            注意: 必须使用字符串格式，而非时间戳
        
        interval: 数据聚合间隔（可选）
            可选值: "1m" (1分钟), "5m" (5分钟), "1h" (1小时)
            默认值: "1m"

    Returns:
        Dict: 内存监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (memory_usage_percent)
            - interval: 数据聚合间隔
            - data_points: 数据点列表，每个点包含:
                * timestamp: 时间点（格式: HH:MM）
                * value: 内存使用率百分比
                * used_gb: 已使用内存（GB）
                * total_gb: 总内存（GB）
            - statistics: 统计信息
                * average: 平均值
                * max: 最大值
                * min: 最小值
            - alert: 告警信息（如有）
                * triggered: 是否触发告警
                * threshold: 告警阈值
                * message: 告警消息
    
    使用示例:
        # 示例1: 使用默认时间（最近1小时）
        query_memory_metrics(service_name="data-sync-service")
        
        # 示例2: 指定时间范围
        query_memory_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00",
            end_time="2026-02-14 11:00:00",
            interval="5m"
        )
    """
    return _generate_metric_response(
        service_name=service_name,
        metric_key="memory",
        metric_name="memory_usage_percent",
        default_start=30.0,
        default_peak=85.0,
        alert_threshold=70.0,
        jitter=1.0,
        pressure_key="memory_pressure",
        alert_message_high="内存使用率超过 70% 阈值，存在内存压力",
        start_time=start_time,
        end_time=end_time,
        interval=interval,
    )


def _active_alerts_response(service_name: str, level: Optional[str]) -> Dict[str, Any]:
    scenario = get_active_scenario()
    scenario_service = str(scenario.get("service_name", "")).lower()
    requested = service_name.lower()
    matched = scenario_service in requested or requested in scenario_service

    alerts = []
    if matched:
        triggered_at = (datetime.now() - timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M:%S")
        for alert in scenario.get("alerts") or []:
            if level and str(alert.get("severity", "")).lower() != level.lower():
                continue
            alerts.append({
                "alert_name": alert.get("alert_name"),
                "severity": alert.get("severity"),
                "description": alert.get("description"),
                "triggered_at": triggered_at,
            })

    return {
        "service_name": service_name,
        "total": len(alerts),
        "alerts": alerts,
        "message": f"查询到 {len(alerts)} 条活跃告警" if alerts else "当前无活跃告警",
    }


@mcp.tool()
@log_tool_call
def query_active_alerts(
    service_name: str,
    level: Optional[str] = None
) -> Dict[str, Any]:
    """查询服务当前活跃的告警清单。

    排障的第一步：先看有哪些告警触发，再决定取证方向。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"

        level: 告警级别过滤（可选）
            可选值: "critical"（严重）, "warning"（警告）, "info"（提示）
            默认值: 不传则返回全部级别的告警

    Returns:
        Dict: 活跃告警清单
            - service_name: 服务名称
            - total: 告警数量
            - alerts: 告警列表，每条包含:
                * alert_name: 告警名称（如 JvmGCPauseHigh、PodOOMKilled）
                * severity: 告警级别
                * description: 告警描述（含触发条件与关键数值）
                * triggered_at: 触发时间
            - message: 查询状态消息

    使用示例:
        # 示例1: 查询全部活跃告警
        query_active_alerts(service_name="data-sync-service")

        # 示例2: 只看严重告警
        query_active_alerts(service_name="data-sync-service", level="critical")
    """
    return _active_alerts_response(service_name, level)




if __name__ == "__main__":
    # 使用 streamable-http 模式，运行在 8004 端口
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(os.getenv("MCP_MONITOR_PORT", "18004")),
        path="/mcp",
    )
