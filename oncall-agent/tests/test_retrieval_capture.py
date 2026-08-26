from app.tools.knowledge_tool import (
    _ACTIVE_RETRIEVAL_SESSION,
    capture_retrieval_for_session,
)


def test_capture_retrieval_session_is_scoped() -> None:
    assert _ACTIVE_RETRIEVAL_SESSION.get() is None

    with capture_retrieval_for_session("eval-session"):
        assert _ACTIVE_RETRIEVAL_SESSION.get() == "eval-session"

    assert _ACTIVE_RETRIEVAL_SESSION.get() is None
