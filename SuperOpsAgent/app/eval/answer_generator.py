from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.services.rag_agent_service import rag_agent_service


class EvalAnswerGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class EvalAnswerResult:
    answer: str
    retrieved_contexts: list[str]
    retrieval_attempted: bool
    retrieval_candidate_sources: list[str]
    reranked_sources: list[str]


async def generate_answer_with_context(
    question: str,
    session_id: str | None = None,
) -> EvalAnswerResult:
    eval_session_id = session_id or f"eval-{uuid4().hex}"

    try:
        result = await rag_agent_service.query_with_context(
            question=question,
            session_id=eval_session_id,
        )
    except Exception as exc:
        raise EvalAnswerGenerationError(str(exc)) from exc

    return EvalAnswerResult(
        answer=result.answer,
        retrieved_contexts=result.retrieved_contexts,
        retrieval_attempted=result.retrieval_attempted,
        retrieval_candidate_sources=result.retrieval_candidate_sources,
        reranked_sources=result.reranked_sources,
    )


async def generate_answer(question: str, session_id: str | None = None) -> str:
    result = await generate_answer_with_context(question, session_id=session_id)
    return result.answer
