from app.services.rag_agent_service import RagAgentService


def test_rag_system_prompt_enforces_grounded_concise_answers() -> None:
    prompt = RagAgentService._build_system_prompt(object())

    assert "必须先调用\n   retrieve_knowledge" in prompt
    assert "只能把 retrieve_knowledge 返回的内容作为事实依据" in prompt
    assert "知识库中没有足够信息回答该问题" in prompt
    assert "只回答用户所问" in prompt
    assert "不使用常识或模型记忆补全缺失事实" in prompt
