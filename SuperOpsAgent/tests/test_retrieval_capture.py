from langchain_core.documents import Document

from app.tools.knowledge_tool import (
    _ACTIVE_RETRIEVAL_SESSION,
    RetrievalTrace,
    capture_retrieval_for_session,
    clear_captured_retrieval_trace,
    pop_captured_retrieval_trace,
    set_captured_retrieval_trace,
)


def test_capture_retrieval_session_is_scoped() -> None:
    assert _ACTIVE_RETRIEVAL_SESSION.get() is None

    with capture_retrieval_for_session("eval-session"):
        assert _ACTIVE_RETRIEVAL_SESSION.get() == "eval-session"

    assert _ACTIVE_RETRIEVAL_SESSION.get() is None


def test_retrieval_trace_preserves_candidate_and_reranked_orders() -> None:
    session_id = "eval-trace"
    candidates = [Document(page_content="candidate", metadata={"_file_name": "a.md"})]
    ranked_docs = [Document(page_content="ranked", metadata={"_file_name": "b.md"})]
    final_docs = ranked_docs[:]

    set_captured_retrieval_trace(
        session_id,
        RetrievalTrace(
            candidates=candidates,
            ranked_docs=ranked_docs,
            final_docs=final_docs,
        ),
    )

    trace = pop_captured_retrieval_trace(session_id)
    assert trace is not None
    assert trace.candidates == candidates
    assert trace.ranked_docs == ranked_docs
    assert trace.final_docs == final_docs
    assert pop_captured_retrieval_trace(session_id) is None
    clear_captured_retrieval_trace(session_id)
