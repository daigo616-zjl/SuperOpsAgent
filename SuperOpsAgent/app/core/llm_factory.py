"""LLM 工厂类

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用阿里云 DashScope
这种方式便于后续切换到其他支持 OpenAI API 的模型提供商

支持的模型提供商（只需修改 base_url 和 api_key）：
- 阿里云 DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
- OpenAI: https://api.openai.com/v1
- Azure OpenAI: https://{resource}.openai.azure.com
- 其他兼容 OpenAI API 的服务
"""

from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from pydantic import SecretStr

from app.config import config
from app.core.llm_resilience import ResilientChatModel, build_resilient_model


class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ResilientChatModel:
        model = model or config.dashscope_model
        base_url = base_url or config.dashscope_api_base
        resolved_api_key = SecretStr(api_key or config.dashscope_api_key)

        # 参考：https://help.aliyun.com/zh/model-studio/getting-started/models
        extra_body = {}
        extra_body["stream"] = streaming

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=resolved_api_key,
            extra_body=extra_body if extra_body else None,
        )

        fallback = None
        if config.llm_fallback_model and config.llm_fallback_model != model:
            fallback = ChatOpenAI(
                model=config.llm_fallback_model,
                temperature=temperature,
                streaming=streaming,
                base_url=base_url,
                api_key=resolved_api_key,
                extra_body=extra_body if extra_body else None,
            )
        return build_resilient_model(
            llm,
            model_name=model,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
            max_concurrency=config.llm_max_concurrency,
            min_interval=config.llm_min_interval,
            failure_threshold=config.llm_circuit_failure_threshold,
            recovery_timeout=config.llm_circuit_recovery_timeout,
            retry_backoff=config.llm_retry_backoff,
            fallback=fallback,
        )

    @staticmethod
    def create_qwen_chat_model(
        model: str,
        temperature: float = 0.7,
        streaming: bool = False,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        stall_seconds: float | None = None,
    ) -> ResilientChatModel:
        """创建地域、凭据配置一致的 ChatQwen 客户端。"""
        resolved_timeout = timeout if timeout is not None else config.llm_timeout
        resolved_retries = (
            max_retries if max_retries is not None else config.llm_max_retries
        )
        llm = ChatQwen(
            model=model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            base_url=base_url or config.dashscope_api_base,
            api_key=SecretStr(api_key or config.dashscope_api_key),
        )
        fallback = None
        if config.llm_fallback_model and config.llm_fallback_model != model:
            fallback = ChatQwen(
                model=config.llm_fallback_model,
                temperature=temperature,
                streaming=streaming,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                base_url=base_url or config.dashscope_api_base,
                api_key=SecretStr(api_key or config.dashscope_api_key),
            )
        return build_resilient_model(
            llm,
            model_name=model,
            timeout=resolved_timeout,
            max_retries=resolved_retries,
            max_concurrency=config.llm_max_concurrency,
            min_interval=config.llm_min_interval,
            failure_threshold=config.llm_circuit_failure_threshold,
            recovery_timeout=config.llm_circuit_recovery_timeout,
            retry_backoff=config.llm_retry_backoff,
            fallback=fallback,
            stall_timeout=stall_seconds,
        )


# 全局 LLM 工厂实例
llm_factory = LLMFactory()
