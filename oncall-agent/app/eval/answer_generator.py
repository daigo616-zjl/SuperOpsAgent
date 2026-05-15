from __future__ import annotations

from uuid import uuid4

from app.services.rag_agent_service import rag_agent_service


class EvalAnswerGenerationError(RuntimeError):
    pass


async def generate_answer(question: str, session_id: str | None = None) -> str:
    eval_session_id = session_id or f"eval-{uuid4().hex}"

    try:
        return await rag_agent_service.query(question=question, session_id=eval_session_id)
    except Exception as exc:
        raise EvalAnswerGenerationError(str(exc)) from exc
