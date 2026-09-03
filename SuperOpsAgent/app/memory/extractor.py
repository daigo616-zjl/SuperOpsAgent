"""长期记忆抽取器：LLM 三分类抽取（fact / text / semantic）"""

import json
import re
from typing import Any

from loguru import logger

from app.config import config
from app.memory.models import ExtractedItem

VALID_TYPES = {"fact", "text", "semantic"}
# 敏感信息黑名单：命中的内容一律不写入长期记忆
SENSITIVE_PATTERNS = (
    "密码", "口令", "凭证", "token", "secret", "api key", "api_key", "password",
)


class MemoryExtractor:
    def _new_model(self):
        """每次抽取新建模型实例：worker 在短命 event loop 中运行，
        共享实例会把 SDK 层的 loop 绑定状态带进已关闭的 loop"""
        from app.core.llm_factory import LLMFactory

        return LLMFactory.create_qwen_chat_model(
            model=config.memory_extract_model,
            temperature=0,
            streaming=False,
            max_tokens=config.memory_extract_max_tokens,
            enable_thinking=False,
        )

    @staticmethod
    def _prompt(
        user_msg: str,
        assistant_msg: str,
        summary: str,
        existing_facts: list[tuple[str, str]] | None = None,
    ) -> str:
        existing_section = ""
        if existing_facts:
            lines = "\n".join(f"- {subject}: {content}" for subject, content in existing_facts)
            existing_section = (
                "该用户已有的强事实（subject: content）:\n"
                f"{lines}\n\n"
                "若新事实描述的是已有某条事实的同一实体同一属性（即使表述不同），"
                "必须原样复用那条事实的 subject——新事实会覆盖旧事实；"
                "只有全新属性才新建「实体-属性」subject。\n\n"
            )
        return (
            "你在从一轮对话中抽取值得长期记住的信息。\n"
            "只抽取持久信息：环境事实（服务名/IP/阈值/配置/归属关系）、用户偏好与约束、"
            "已确认的决策与结论。\n"
            "禁止抽取：瞬时问答内容、寒暄、当前时间状态、任何凭证或敏感信息。\n"
            "每条记忆必须是原子单句，content 可独立理解，"
            "一条只描述一个属性（如部署区域和数据库版本要拆成两条）。\n"
            "subject 必须是「实体-属性」形式的细粒度标签（如 用户-姓名、用户-职业、"
            "服务-部署位置、系统-备份策略），禁止使用「用户」这类宽泛主语——"
            "同 subject 的新事实会覆盖旧事实，粒度不够细会导致无关记忆互相顶掉。\n"
            f"{existing_section}"
            "分类规则：\n"
            "- fact：可独立验证的确定性事实（如'用户的服务 data-sync-service 部署在华东1'），"
            "必须能在对话原文中找到依据\n"
            "- text：较长的历史总结、需求描述、已确认方案等半结构化内容\n"
            "- semantic：无法结构化的模糊语义内容（沟通风格、情绪、隐性偏好）\n"
            "允许输出空数组。只输出 JSON，不要输出其他文字。\n"
            '格式：{"items":[{"type":"fact|text|semantic","content":"...","subject":"实体-属性","confidence":0.0-1.0}]}\n\n'
            f"历史摘要（供参考，不要重复抽取其中已有内容）:\n{(summary or '（无）')[:800]}\n\n"
            f"用户消息:\n{user_msg[:2000]}\n\n"
            f"助手回复:\n{assistant_msg[:2000]}\n\n"
            "JSON:"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def extract(
        self,
        user_msg: str,
        assistant_msg: str,
        summary: str = "",
        existing_facts: list[tuple[str, str]] | None = None,
    ) -> list[ExtractedItem]:
        if not user_msg.strip() or not assistant_msg.strip():
            return []

        model = self._new_model()
        response = await model.ainvoke(
            self._prompt(user_msg, assistant_msg, summary, existing_facts)
        )
        raw = response.content if hasattr(response, "content") else response
        if isinstance(raw, list):
            raw = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in raw
            )

        parsed = self._parse_json(str(raw))
        if parsed is None:
            logger.warning("长期记忆抽取输出无法解析为 JSON，本轮跳过")
            return []

        items: list[ExtractedItem] = []
        for entry in parsed.get("items", []):
            if not isinstance(entry, dict):
                continue
            item_type = str(entry.get("type", "")).strip().lower()
            content = str(entry.get("content", "")).strip()
            if item_type not in VALID_TYPES or not content:
                continue
            if any(pattern in content.lower() for pattern in SENSITIVE_PATTERNS):
                logger.debug("长期记忆抽取命中敏感信息黑名单，丢弃")
                continue
            try:
                confidence = float(entry.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            if confidence < config.memory_extract_confidence_min:
                continue
            items.append(
                ExtractedItem(
                    type=item_type,
                    content=content,
                    subject=str(entry.get("subject", "")).strip(),
                    confidence=confidence,
                )
            )
        return items


memory_extractor = MemoryExtractor()
