"""腾讯云 CLS (Cloud Log Service) MCP Server

本地实现的 CLS 日志服务 MCP Server，提供日志查询、检索和分析功能。
"""

import logging
import functools
import json
import os
import re
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

try:
    from mcp_servers.scenario_loader import get_active_scenario, noise_range, rng_for
except ImportError:  # 以脚本方式直接运行时
    from scenario_loader import get_active_scenario, noise_range, rng_for

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLS_MCP_Server")

mcp = FastMCP("CLS")


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
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int) -> str:
    """生成基于基准时间的时间字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（以毫秒为单位）。
    
    此工具用于获取标准的毫秒时间戳，可用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数
    
    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000
    
    使用示例:
        # 获取当前时间
        current = get_current_timestamp()
        
        # 计算15分钟前的时间
        fifteen_min_ago = current - (15 * 60 * 1000)
        
        # 计算1小时前的时间
        one_hour_ago = current - (60 * 60 * 1000)
        
        # 用于搜索最近15分钟的日志
        search_log(
            topic_id="topic-001",
            start_time=fifteen_min_ago,
            end_time=current
        )
    """
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str, Any]:
    """根据地区名称搜索对应的地区参数。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
            - region_code: 地区代码
            - region_name: 地区名称
            - available: 是否可用
    """
    # 模拟地区映射表（实际应该从配置或数据库读取）
    region_mapping = {
        "北京": {"region_code": "ap-beijing", "region_name": "北京", "available": True},
        "上海": {"region_code": "ap-shanghai", "region_name": "上海", "available": True},
        "广州": {"region_code": "ap-guangzhou", "region_name": "广州", "available": True},
    }

    result = region_mapping.get(region_name)
    if result:
        return result
    else:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": f"未找到地区: {region_name}"
        }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(topic_name: str, region_code: Optional[str] = None) -> Dict[str, Any]:
    """根据主题名称搜索相关的主题信息。

    Args:
        topic_name: 主题名称
        region_code: 地区代码（可选）

    Returns:
        Dict: 包含主题信息的字典
            - topic_id: 主题ID
            - topic_name: 主题名称
            - region_code: 所属地区
            - create_time: 创建时间
            - log_count: 日志数量
    """
    mock_topics = [
        {
            "topic_id": "topic-001",
            "topic_name": "数据同步服务日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "服务应用日志"
        }
    ]

    # 根据名称和地区筛选
    for topic in mock_topics:
        if topic["topic_name"] == topic_name:
            if region_code is None or topic["region_code"] == region_code:
                return topic

    return {
        "topic_id": None,
        "topic_name": topic_name,
        "region_code": region_code,
        "error": f"未找到主题: {topic_name}"
    }


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
    service_name: str,
    region_code: Optional[str] = None,
    fuzzy: bool = True
) -> Dict[str, Any]:
    """根据服务名称搜索相关的日志主题信息，支持模糊搜索。
    
    此工具用于根据服务名称查找对应的日志主题（topic），便于后续进行日志查询。
    
    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service", "sync", "data-sync"
            说明: 当 fuzzy=True 时，支持部分匹配
        
        region_code: 地区代码（可选）
            示例: "ap-beijing", "ap-shanghai"
            说明: 如果指定，只返回该地区的主题
        
        fuzzy: 是否启用模糊搜索（可选，默认 True）
            True: 部分匹配，例如 "sync" 可以匹配 "data-sync-service"
            False: 精确匹配，必须完全一致
    
    Returns:
        Dict: 搜索结果
            - total: 匹配到的主题数量
            - topics: 主题列表，每个主题包含:
                * topic_id: 主题ID（用于后续日志查询）
                * topic_name: 主题名称
                * service_name: 服务名称
                * region_code: 所属地区
                * create_time: 创建时间
                * log_count: 日志数量
                * description: 主题描述
            - query: 查询条件
    
    使用示例:
        # 示例1: 模糊搜索（推荐）
        search_topic_by_service_name(service_name="data-sync")
        # 可以匹配: "data-sync-service", "data-sync-worker" 等
        
        # 示例2: 精确搜索
        search_topic_by_service_name(
            service_name="data-sync-service",
            fuzzy=False
        )
        
        # 示例3: 指定地区搜索
        search_topic_by_service_name(
            service_name="sync",
            region_code="ap-beijing"
        )
        
        # 示例4: 查找后进行日志搜索的完整流程
        # 步骤1: 根据服务名查找 topic
        result = search_topic_by_service_name(service_name="data-sync-service")
        
        # 步骤2: 获取 topic_id
        topic_id = result["topics"][0]["topic_id"]  # "topic-001"
        
        # 步骤3: 使用 topic_id 查询日志
        current_ts = get_current_timestamp()
        start_ts = current_ts - (15 * 60 * 1000)
        search_log(
            topic_id=topic_id,
            start_time=start_ts,
            end_time=current_ts
        )
    """
    # Mock 主题数据（实际应该从配置或数据库读取）
    mock_topics = [
        {
            "topic_id": "topic-001",
            "topic_name": "数据同步服务日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "数据同步服务的应用日志，包含同步任务执行情况"
        },
        {
            "topic_id": "topic-002",
            "topic_name": "数据同步服务错误日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "数据同步服务的错误日志"
        },
        {
            "topic_id": "topic-003",
            "topic_name": "API网关服务日志",
            "service_name": "api-gateway-service",
            "region_code": "ap-shanghai",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "API网关服务日志"
        }
    ]
    
    matched_topics = []
    
    # 搜索逻辑
    for topic in mock_topics:
        # 地区筛选
        if region_code and topic["region_code"] != region_code:
            continue
        
        # 服务名称匹配
        topic_service_name = topic.get("service_name", "")
        
        if fuzzy:
            # 模糊匹配：服务名包含查询字符串，或查询字符串包含服务名
            if (service_name.lower() in topic_service_name.lower() or 
                topic_service_name.lower() in service_name.lower()):
                matched_topics.append(topic)
        else:
            # 精确匹配
            if topic_service_name == service_name:
                matched_topics.append(topic)
    
    return {
        "total": len(matched_topics),
        "topics": matched_topics,
        "query": {
            "service_name": service_name,
            "region_code": region_code,
            "fuzzy": fuzzy
        },
        "message": f"找到 {len(matched_topics)} 个匹配的日志主题" if matched_topics else f"未找到服务 '{service_name}' 的日志主题"
    }


# ============================================================
# 剧本化日志生成（确定性：按分钟桶播种，分页稳定）
# ============================================================

NOISE_INFO_TEMPLATES = [
    "正在同步元数据……",
    "心跳检测正常",
    "同步任务 #{n} 执行完成，耗时 {ms} ms",
    "刷新本地缓存完成",
    "处理分片 {n}/16 完成",
]
NOISE_WARN_TEMPLATES = [
    "下游调用超时，已自动重试成功",
    "线程池活跃度 {n}%，接近告警阈值",
]
NOISE_ERROR_TEMPLATES = [
    "下游短暂超时（{ms} ms），重试后成功，已自愈",
]

TOPIC_APP_LOG = "topic-001"     # 全量日志（含噪声与剧本注入行）
TOPIC_ERROR_LOG = "topic-002"   # 仅 ERROR 级别


def _generate_logs_for_minute(scenario: Dict[str, Any], minute_start_ms: int) -> list:
    """生成某个分钟桶内的日志行 [(epoch_ms, level, message), ...]。"""
    minute_start = datetime.fromtimestamp(minute_start_ms / 1000)
    minute_bucket = minute_start.strftime("%Y-%m-%d %H:%M")
    rng = rng_for(scenario["id"], "logs", minute_bucket)
    lines = []

    def fmt(template: str) -> str:
        return template.format(n=rng.randint(1, 999), ms=rng.randint(20, 4000))

    lo, hi = noise_range(scenario)
    for _ in range(rng.randint(lo, hi)):
        lines.append((minute_start_ms + rng.randint(0, 59) * 1000,
                      "INFO", fmt(rng.choice(NOISE_INFO_TEMPLATES))))
    if rng.random() < 0.15:
        lines.append((minute_start_ms + rng.randint(0, 59) * 1000,
                      "WARN", fmt(rng.choice(NOISE_WARN_TEMPLATES))))
    # 低频瞬时 ERROR（自愈型），防止“见 ERROR 即根因”
    if rng.random() < 0.05:
        lines.append((minute_start_ms + rng.randint(0, 59) * 1000,
                      "ERROR", fmt(rng.choice(NOISE_ERROR_TEMPLATES))))
    for pattern in scenario.get("log_patterns") or []:
        for _ in range(int(pattern.get("per_minute", 1))):
            lines.append((minute_start_ms + rng.randint(0, 59) * 1000,
                          str(pattern.get("level", "ERROR")).upper(),
                          str(pattern.get("message", ""))))

    lines.sort(key=lambda item: item[0])
    return lines


def _generate_logs_for_window(scenario: Dict[str, Any], start_ms: int, end_ms: int) -> list:
    logs = []
    first_minute = (start_ms // 60000) * 60000
    minute = first_minute
    while minute <= end_ms:
        logs.extend(_generate_logs_for_minute(scenario, minute))
        minute += 60000
    return logs


def _parse_log_query(query: Optional[str]) -> tuple:
    """解析 CLS 风格查询语法: level:ERROR AND "关键词"。

    Returns:
        (level, keywords): level 为大写级别或 None；keywords 为小写关键词列表。
    """
    level = None
    keywords = []
    if not query:
        return level, keywords
    tokens = re.split(r"\s+AND\s+", query.strip(), flags=re.IGNORECASE)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        level_match = re.fullmatch(r"level\s*:\s*(\w+)", token, re.IGNORECASE)
        if level_match:
            level = level_match.group(1).upper()
            continue
        if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
            token = token[1:-1]
        keywords.append(token.lower())
    return level, keywords


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int,
    end_time: int,
    query: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """基于提供的查询参数搜索日志。

    Args:
        topic_id: 主题ID（必填）
            示例: "topic-001"
        
        start_time: 开始时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            获取方式: 
            1. 使用 get_current_timestamp() 工具获取当前时间戳
            2. 计算历史时间: current_timestamp - (分钟数 * 60 * 1000)
            示例: 
            - 当前时间: 1708012345000
            - 15分钟前: 1708012345000 - (15 * 60 * 1000) = 1708011445000
            - 1小时前: 1708012345000 - (60 * 60 * 1000) = 1708008745000
        
        end_time: 结束时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            通常使用 get_current_timestamp() 工具获取当前时间作为结束时间
            示例: 1708012345000
        
        query: 查询语句（可选，CLS 查询语法）
            支持 AND 组合的过滤条件:
            - level:ERROR        按日志级别过滤（ERROR/WARN/INFO）
            - "GC"               按关键词过滤（支持引号或裸词）
            - 组合示例: level:ERROR AND "GC overhead"

        limit: 单页返回条数限制（默认100，最大1000，可选）

        offset: 分页偏移量（默认0，配合 next_offset/has_more 翻页）

    Returns:
        Dict: 搜索结果
            - topic_id: 主题ID
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - query: 查询语句
            - limit: 结果限制
            - total: 实际返回的日志条数（当前页）
            - total_matched: 满足查询条件的日志总条数（跨所有页）
            - offset: 当前页起始偏移量
            - next_offset: 下一页偏移量（无更多数据时为 null）
            - has_more: 是否还有更多数据
            - logs: 日志列表，每条日志包含:
                * timestamp: 日志时间（格式: YYYY-MM-DD HH:MM:SS）
                * level: 日志级别
                * message: 日志内容
            - took_ms: 查询耗时（毫秒）
            - message: 查询状态消息
    
    使用示例:
        # 步骤1: 获取当前时间戳
        current_ts = get_current_timestamp()  # 返回: 1708012345000
        
        # 步骤2: 计算开始时间（15分钟前）
        start_ts = current_ts - (15 * 60 * 1000)  # 1708011445000
        
        # 步骤3: 搜索日志
        search_log(
            topic_id="topic-001",
            start_time=start_ts,     # int类型: 1708011445000
            end_time=current_ts,     # int类型: 1708012345000
            limit=100
        )
    """
    # 仅支持应用日志与错误日志两个主题
    if topic_id not in (TOPIC_APP_LOG, TOPIC_ERROR_LOG):
        return {
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "offset": offset,
            "total": 0,
            "total_matched": 0,
            "logs": [],
            "took_ms": 0,
            "error": f"主题不存在: {topic_id}",
            "message": f"错误: 未找到主题 {topic_id}，请检查 topic_id 是否正确"
        }

    scenario = get_active_scenario()
    level_filter, keywords = _parse_log_query(query)
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))

    all_logs = _generate_logs_for_window(scenario, start_time, end_time)
    matched = []
    for epoch_ms, level, message in all_logs:
        if not (start_time <= epoch_ms <= end_time):
            continue
        if topic_id == TOPIC_ERROR_LOG and level != "ERROR":
            continue
        if level_filter and level != level_filter:
            continue
        if keywords and not all(k in message.lower() for k in keywords):
            continue
        matched.append((epoch_ms, level, message))

    total_matched = len(matched)
    page = matched[offset:offset + limit]
    has_more = offset + limit < total_matched

    logs = [
        {
            "timestamp": datetime.fromtimestamp(epoch_ms / 1000).strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
        }
        for epoch_ms, level, message in page
    ]

    return {
        "topic_id": topic_id,
        "start_time": start_time,
        "end_time": end_time,
        "query": query,
        "limit": limit,
        "offset": offset,
        "total": len(logs),
        "total_matched": total_matched,
        "next_offset": offset + len(logs) if has_more else None,
        "has_more": has_more,
        "logs": logs,
        "took_ms": 50,
        "message": f"成功查询 {len(logs)} 条日志，共匹配 {total_matched} 条"
    }



if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(os.getenv("MCP_CLS_PORT", "18003")),
        path="/mcp",
    )
