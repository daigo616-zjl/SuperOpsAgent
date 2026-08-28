from typing import Any

import app.core.llm_factory as llm_factory_module
from app.config import Settings, config
from app.core.llm_factory import LLMFactory


def test_dashscope_api_base_defaults_to_beijing() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.dashscope_api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_qwen_factory_passes_configured_api_base(monkeypatch) -> None:
    captured: list[dict[str, Any]] = []

    class FakeChatQwen:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(llm_factory_module, "ChatQwen", FakeChatQwen)
    monkeypatch.setattr(config, "dashscope_api_base", "https://example.test/v1")
    monkeypatch.setattr(config, "dashscope_api_key", "test-key")

    LLMFactory.create_qwen_chat_model(
        model="test-model",
        temperature=0.1,
        streaming=True,
        max_tokens=1200,
        enable_thinking=False,
    )

    primary = captured[0]
    assert primary["base_url"] == "https://example.test/v1"
    assert primary["api_key"].get_secret_value() == "test-key"
    assert primary["model"] == "test-model"
    assert primary["temperature"] == 0.1
    assert primary["streaming"] is True
    assert primary["max_tokens"] == 1200
    assert primary["enable_thinking"] is False

    assert len(captured) == 2
    fallback = captured[1]
    assert fallback["model"] == config.llm_fallback_model
    assert fallback["base_url"] == "https://example.test/v1"


def test_rag_generation_defaults_are_conservative() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.rag_top_k == 5
    assert settings.rag_temperature == 0.1
    assert settings.rag_max_tokens == 1200
    assert settings.rag_enable_thinking is False


def test_context_summary_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.rag_context_summary_model == "qwen3.5-flash"
    assert settings.rag_context_summary_trigger_messages == 12
    assert settings.rag_context_summary_keep_messages == 6
