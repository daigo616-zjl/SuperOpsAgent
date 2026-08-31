"""统一的 AIOps 工具注册表。"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.agent.mcp_client import get_mcp_client_with_retry
from app.config import config
from app.tools import get_current_time, retrieve_knowledge

from .models import DiagnosticPlan


class ToolDescriptor(BaseModel):
    name: str
    description: str = ""
    source: Literal["local", "mcp"]
    input_schema: dict[str, Any] = Field(default_factory=dict)


class UnknownToolError(LookupError):
    pass


class InvalidToolArgumentsError(ValueError):
    pass


@dataclass
class ToolRegistry:
    descriptors: dict[str, ToolDescriptor]
    handlers: dict[str, Any]

    def get_descriptor(self, name: str) -> ToolDescriptor:
        try:
            return self.descriptors[name]
        except KeyError as exc:
            raise UnknownToolError(f"工具不存在: {name}") from exc

    def validate_arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.get_descriptor(name)
        handler = self.handlers[name]
        args_schema = getattr(handler, "args_schema", None)
        if args_schema is None:
            return arguments
        # langchain-mcp-adapters intentionally exposes MCP inputSchema as a plain
        # JSON Schema dict. StructuredTool validates that form during ainvoke;
        # model_validate is only available on Pydantic model classes.
        if isinstance(args_schema, dict):
            return arguments
        if not hasattr(args_schema, "model_validate"):
            raise InvalidToolArgumentsError(
                f"工具 {name} 的参数 Schema 类型不受支持: {type(args_schema).__name__}"
            )
        try:
            validated = args_schema.model_validate(arguments)
        except ValidationError as exc:
            raise InvalidToolArgumentsError(str(exc)) from exc
        return validated.model_dump(exclude_none=True)

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        self.get_descriptor(name)
        return await self.handlers[name].ainvoke(arguments)

    def prompt_description(self) -> str:
        return json.dumps(
            [descriptor.model_dump() for descriptor in self.descriptors.values()],
            ensure_ascii=False,
            indent=2,
        )

    def validate_plan(self, plan: DiagnosticPlan) -> None:
        """对无需解析运行时引用的计划部分做静态校验。"""
        for step in plan.steps:
            descriptor = self.get_descriptor(step.tool_call.tool_name)
            schema = descriptor.input_schema
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
            supplied = set(step.tool_call.arguments)
            missing = required - supplied
            if missing:
                raise InvalidToolArgumentsError(f"步骤 {step.id} 缺少工具参数: {sorted(missing)}")
            if schema.get("additionalProperties") is False:
                unknown = supplied - set(properties)
                if unknown:
                    raise InvalidToolArgumentsError(
                        f"步骤 {step.id} 包含未知工具参数: {sorted(unknown)}"
                    )


def _input_schema(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()
    args = getattr(tool, "args", None)
    return args if isinstance(args, dict) else {}


async def get_tool_registry() -> ToolRegistry:
    """从本地工具和 MCP 工具构建同一份注册表快照。"""
    local_tools = [get_current_time, retrieve_knowledge]
    mcp_client = await get_mcp_client_with_retry()
    mcp_tools = await mcp_client.get_tools()

    descriptors: dict[str, ToolDescriptor] = {}
    handlers: dict[str, Any] = {}
    for source, tools in (("local", local_tools), ("mcp", mcp_tools)):
        for tool in tools:
            name = getattr(tool, "name", "")
            if not name:
                continue
            if name in descriptors:
                raise ValueError(f"工具名称重复，无法构建注册表: {name}")
            descriptors[name] = ToolDescriptor(
                name=name,
                description=getattr(tool, "description", "") or "",
                source=source,
                input_schema=_input_schema(tool),
            )
            handlers[name] = tool
    return ToolRegistry(descriptors=descriptors, handlers=handlers)


def get_domain_tool_names(domain: str) -> list[str]:
    names = config.aiops_domain_tools.get(domain)
    if names is None:
        raise ValueError(f"未知取证域: {domain}，可用域: {sorted(config.aiops_domain_tools)}")
    return list(names)


async def get_domain_registry(domain: str) -> ToolRegistry:
    """按取证域切分工具注册表：能力边界 = 决策权边界。

    取证 Agent 只能看到本域工具名单内的工具；名单中配置了但注册表里
    不存在的工具视为部署错误，立即失败。
    """
    allowed = set(get_domain_tool_names(domain))
    full = await get_tool_registry()
    missing = allowed - set(full.descriptors)
    if missing:
        raise UnknownToolError(f"取证域 {domain} 缺少已配置工具: {sorted(missing)}")
    return ToolRegistry(
        descriptors={
            name: descriptor
            for name, descriptor in full.descriptors.items()
            if name in allowed
        },
        handlers={
            name: handler for name, handler in full.handlers.items() if name in allowed
        },
    )
