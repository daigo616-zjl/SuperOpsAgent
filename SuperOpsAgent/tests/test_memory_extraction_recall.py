"""长期记忆抽取与召回融合测试（mock LLM 与三路存储，无真实中间件）"""

from types import SimpleNamespace

import pytest

from app.memory import recall_service as recall_module
from app.memory.extractor import MemoryExtractor
from app.memory.models import MemoryContext, MemoryHit
from app.memory.recall_service import MemoryRecallService, extract_terms


def _model_returning(text: str) -> SimpleNamespace:
    async def ainvoke(_prompt):
        return SimpleNamespace(content=text)

    return SimpleNamespace(ainvoke=ainvoke)


# ------------------------------------------------------------
# 抽取器
# ------------------------------------------------------------

async def test_extract_parses_valid_items(monkeypatch):
    extractor = MemoryExtractor()
    monkeypatch.setattr(
        extractor, "_get_model",
        lambda: _model_returning(
            '{"items":[{"type":"fact","content":"用户的服务是 data-sync-service",'
            '"subject":"服务","confidence":0.95},'
            '{"type":"semantic","content":"用户偏好简洁回答","subject":"用户","confidence":0.8}]}'
        ),
    )
    items = await extractor.extract("我的服务是什么配置", "你的服务是 data-sync-service")
    assert [item.type for item in items] == ["fact", "semantic"]
    assert items[0].content == "用户的服务是 data-sync-service"


async def test_extract_strips_code_fence(monkeypatch):
    extractor = MemoryExtractor()
    monkeypatch.setattr(
        extractor, "_get_model",
        lambda: _model_returning('```json\n{"items":[{"type":"text","content":"方案A已确认","confidence":0.9}]}\n```'),
    )
    items = await extractor.extract("用方案A", "好的，已确认方案A")
    assert len(items) == 1 and items[0].type == "text"


async def test_extract_filters_invalid_low_confidence_and_sensitive(monkeypatch):
    extractor = MemoryExtractor()
    monkeypatch.setattr(
        extractor, "_get_model",
        lambda: _model_returning(
            '{"items":['
            '{"type":"unknown","content":"x","confidence":0.9},'
            '{"type":"fact","content":"","confidence":0.9},'
            '{"type":"fact","content":"低置信内容","confidence":0.3},'
            '{"type":"fact","content":"我的密码是123456","confidence":0.99},'
            '{"type":"fact","content":"生产库阈值是80%","confidence":0.9}]}'
        ),
    )
    items = await extractor.extract("q", "a")
    assert [item.content for item in items] == ["生产库阈值是80%"]


async def test_extract_garbage_output_returns_empty(monkeypatch):
    extractor = MemoryExtractor()
    monkeypatch.setattr(
        extractor, "_get_model", lambda: _model_returning("抱歉我无法输出 JSON"),
    )
    assert await extractor.extract("q", "a") == []


async def test_extract_empty_inputs_short_circuit(monkeypatch):
    extractor = MemoryExtractor()
    monkeypatch.setattr(extractor, "_get_model", lambda: pytest.fail("不应调用模型"))
    assert await extractor.extract("", "a") == []
    assert await extractor.extract("q", "") == []


# ------------------------------------------------------------
# 召回融合
# ------------------------------------------------------------

async def test_recall_fuses_with_priority_and_dedupe(monkeypatch):
    service = MemoryRecallService()

    def fake_facts(user_id, terms, limit=20):
        return [MemoryHit(content="强事实1", content_hash="f1")]

    def fake_es(user_id, query, top_k):
        return [
            MemoryHit(content="ES命中A", content_hash="e1", score=12.0),
            MemoryHit(content="ES命中B", content_hash="", score=8.0),
        ]

    def fake_vec(user_id, query, top_k):
        return [
            MemoryHit(content="与ES重复", content_hash="e1", score=0.9),  # 应被去重
            MemoryHit(content="纯语义", content_hash="v1", score=0.85),
            MemoryHit(content="低分语义", content_hash="v2", score=0.2),  # 存储层已过滤阈值，理论上不出现
        ]

    monkeypatch.setattr(recall_module.long_term_repository, "match_facts", fake_facts)
    monkeypatch.setattr(recall_module.long_term_repository, "recent_facts", lambda *a, **k: [])
    monkeypatch.setattr(recall_module.es_memory_store, "search", fake_es)
    monkeypatch.setattr(recall_module.milvus_memory_store, "search", fake_vec)

    ctx = await service.recall("data-sync-service 内存告警怎么排查", "user-1")

    assert [h.content for h in ctx.facts] == ["强事实1"]
    assert [h.content for h in ctx.es_hits] == ["ES命中A", "ES命中B"]
    # e1 与 ES 去重后被剔除
    assert [h.content for h in ctx.vec_hits] == ["纯语义", "低分语义"]


async def test_recall_degrades_per_store(monkeypatch):
    service = MemoryRecallService()

    def fail(*args, **kwargs):
        raise ConnectionError("store down")

    monkeypatch.setattr(recall_module.long_term_repository, "match_facts", fail)
    monkeypatch.setattr(recall_module.long_term_repository, "recent_facts", lambda *a, **k: [])
    monkeypatch.setattr(recall_module.es_memory_store, "search", fail)
    monkeypatch.setattr(recall_module.milvus_memory_store, "search", fail)

    ctx = await service.recall("查询", "user-1")
    assert ctx.is_empty() is True


async def test_recall_empty_query_short_circuit():
    service = MemoryRecallService()
    assert service.recall.__name__ == "recall"
    ctx = await service.recall("  ", "user-1")
    assert ctx.is_empty() is True


def test_extract_terms_filters_short_and_duplicated():
    terms = extract_terms("data-sync-service 的 CPU, CPU 告警！a")
    assert terms.count("CPU") == 1
    assert "a" not in terms
    assert "data-sync-service" in terms


def test_format_prompt_block_declares_priority():
    ctx = MemoryContext(
        facts=[MemoryHit(content="事实F")],
        es_hits=[MemoryHit(content="经验E")],
        vec_hits=[MemoryHit(content="联想V")],
    )
    block = MemoryRecallService.format_prompt_block(ctx)
    assert block.index("[长期记忆-强事实]") < block.index("[长期记忆-相关经验]")
    assert block.index("[长期记忆-相关经验]") < block.index("[语义联想-仅供参考]")
    assert "不可被下述内容推翻" in block

    empty_block = MemoryRecallService.format_prompt_block(MemoryContext())
    assert empty_block == ""
