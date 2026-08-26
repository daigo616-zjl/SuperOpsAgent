# Cross-Encoder Rerank Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an in-process open-source cross-encoder rerank layer after hybrid retrieval so the system recalls 20 candidates, reranks them, and returns the top 5 documents to the LLM with fallback to hybrid top 5 on rerank failure.

**Architecture:** Keep `retrieve_knowledge()` as the orchestration point. Let `HybridSearchService` continue to handle recall and RRF fusion only, then pass its `Document` candidates into a new `RerankService` that lazily loads a local rerank model, scores `(query, doc.page_content)` pairs, sorts the candidates, and falls back to the original order on timeout or inference errors.

**Tech Stack:** Python 3.11, FastAPI, LangChain `Document`, Pydantic Settings, pytest, pytest-mock, open-source cross-encoder rerank dependency

---

### Task 1: Add rerank configuration

**Files:**
- Modify: `app/config.py:37-53`
- Modify: `docs/plans/2026-05-13-rerank-design.md` (only if implementation reveals config naming mismatch)

**Step 1: Write the failing test**

Create a new config test file:

- Create: `tests/test_config.py`

Add a test that asserts the new config attributes exist with the expected defaults:

```python
from app.config import Settings


def test_rerank_settings_defaults():
    settings = Settings()

    assert settings.rag_recall_size == 20
    assert settings.rag_top_k == 5
    assert settings.rag_rerank_enabled is True
    assert settings.rag_rerank_model == ""
    assert settings.rag_rerank_timeout == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_rerank_settings_defaults -v`
Expected: FAIL because `rag_rerank_*` fields do not exist and `rag_top_k` is still `3`.

**Step 3: Write minimal implementation**

Modify `app/config.py`:

- Change:

```python
rag_top_k: int = 3
```

To:

```python
rag_top_k: int = 5
```

- Keep:

```python
rag_recall_size: int = 20
```

- Add:

```python
rag_rerank_enabled: bool = True
rag_rerank_model: str = ""
rag_rerank_timeout: int = 10
```

Place the rerank fields directly under the existing RAG settings.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_rerank_settings_defaults -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add rerank configuration"
```

### Task 2: Add the rerank service skeleton with fallback behavior

**Files:**
- Create: `app/services/rerank_service.py`
- Create: `tests/services/test_rerank_service.py`

**Step 1: Write the failing test**

Create `tests/services/test_rerank_service.py` with three tests:

```python
from langchain_core.documents import Document

from app.services.rerank_service import RerankService


def test_rerank_returns_top_k_documents_by_score(mocker):
    service = RerankService()
    docs = [
        Document(page_content="doc-a", metadata={"id": "a"}),
        Document(page_content="doc-b", metadata={"id": "b"}),
        Document(page_content="doc-c", metadata={"id": "c"}),
    ]

    mocker.patch.object(service, "_score_pairs", return_value=[0.1, 0.9, 0.5])

    ranked = service.rerank("query", docs, top_k=2)

    assert [doc.metadata["id"] for doc in ranked] == ["b", "c"]


def test_rerank_falls_back_to_original_order_on_scoring_error(mocker):
    service = RerankService()
    docs = [
        Document(page_content="doc-a", metadata={"id": "a"}),
        Document(page_content="doc-b", metadata={"id": "b"}),
    ]

    mocker.patch.object(service, "_score_pairs", side_effect=RuntimeError("boom"))

    ranked = service.rerank("query", docs, top_k=1)

    assert [doc.metadata["id"] for doc in ranked] == ["a"]


def test_rerank_returns_empty_for_empty_inputs():
    service = RerankService()

    assert service.rerank("query", [], top_k=5) == []
    assert service.rerank("   ", [Document(page_content="doc", metadata={})], top_k=5) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_rerank_service.py -v`
Expected: FAIL because `app/services/rerank_service.py` does not exist.

**Step 3: Write minimal implementation**

Create `app/services/rerank_service.py` with a minimal service:

```python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config


class RerankService:
    def __init__(self) -> None:
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            raise NotImplementedError
        return self._model

    def _score_pairs(self, query: str, docs: list[Document]) -> list[float]:
        raise NotImplementedError

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not query.strip() or not docs or top_k <= 0:
            return []

        try:
            scores = self._score_pairs(query, docs)
            if len(scores) != len(docs):
                raise ValueError("score length mismatch")
            ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as exc:
            logger.warning(f"Rerank 降级: {exc}")
            return docs[:top_k]


rerank_service = RerankService()
```

Keep the implementation deliberately minimal so the tests pass before model integration.

**Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_rerank_service.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/rerank_service.py tests/services/test_rerank_service.py
git commit -m "feat: add rerank service skeleton"
```

### Task 3: Add lazy model loading and timeout-bounded scoring

**Files:**
- Modify: `app/services/rerank_service.py`
- Modify: `tests/services/test_rerank_service.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

Extend `tests/services/test_rerank_service.py` with two tests:

```python
def test_rerank_loads_model_once(mocker):
    service = RerankService()
    fake_model = object()
    load_model = mocker.patch.object(service, "_load_model", return_value=fake_model)

    assert service._get_model() is fake_model
    assert service._get_model() is fake_model
    assert load_model.call_count == 1


def test_rerank_falls_back_on_timeout(mocker):
    service = RerankService()
    docs = [
        Document(page_content="doc-a", metadata={"id": "a"}),
        Document(page_content="doc-b", metadata={"id": "b"}),
    ]

    mocker.patch.object(service, "_score_pairs", side_effect=TimeoutError("timeout"))

    ranked = service.rerank("query", docs, top_k=2)

    assert [doc.metadata["id"] for doc in ranked] == ["a", "b"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_rerank_service.py -v`
Expected: FAIL because `_load_model()` and lazy caching behavior are not implemented.

**Step 3: Write minimal implementation**

Update `app/services/rerank_service.py`:

- Add a `_load_model()` method that imports the chosen rerank dependency and loads `config.rag_rerank_model`.
- Change `_get_model()` to:

```python
def _get_model(self) -> Any:
    if self._model is None:
        self._model = self._load_model()
    return self._model
```

- Replace the `NotImplementedError` placeholder in `_score_pairs()` with a timeout-bounded implementation. Structure it like this:

```python
def _predict_scores(self, query: str, docs: list[Document]) -> list[float]:
    model = self._get_model()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    return [float(score) for score in scores]


def _score_pairs(self, query: str, docs: list[Document]) -> list[float]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self._predict_scores, query, docs)
        try:
            return future.result(timeout=config.rag_rerank_timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError("rerank timeout") from exc
```

- Add a guard in `_load_model()` so an empty `config.rag_rerank_model` raises a clear `ValueError` and triggers fallback.

Add the chosen rerank dependency to `pyproject.toml` under `[project].dependencies`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_rerank_service.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/rerank_service.py tests/services/test_rerank_service.py pyproject.toml
git commit -m "feat: add lazy rerank model loading"
```

### Task 4: Route knowledge retrieval through rerank

**Files:**
- Modify: `app/tools/knowledge_tool.py:10-57`
- Modify: `tests/tools/test_knowledge_tool.py`

**Step 1: Write the failing test**

Create `tests/tools/test_knowledge_tool.py` with two tests:

```python
from langchain_core.documents import Document

from app.tools.knowledge_tool import retrieve_knowledge


def test_retrieve_knowledge_reranks_hybrid_results(mocker):
    hybrid_docs = [
        Document(page_content="doc-a", metadata={"id": "a", "_file_name": "a.md"}),
        Document(page_content="doc-b", metadata={"id": "b", "_file_name": "b.md"}),
    ]
    reranked_docs = [hybrid_docs[1]]

    rewrite = mocker.patch("app.tools.knowledge_tool.query_rewrite_service.rewrite_sync", return_value="rewritten")
    hybrid = mocker.patch("app.tools.knowledge_tool.hybrid_search_service.search_sync", return_value=hybrid_docs)
    rerank = mocker.patch("app.tools.knowledge_tool.rerank_service.rerank", return_value=reranked_docs)

    content, artifact = retrieve_knowledge.func("original question", runtime_config=None)

    rewrite.assert_called_once_with("original question", None)
    hybrid.assert_called_once_with("rewritten", top_k=20)
    rerank.assert_called_once_with("rewritten", hybrid_docs, top_k=5)
    assert artifact == reranked_docs
    assert "【参考资料 1】" in content
    assert "b.md" in content


def test_retrieve_knowledge_returns_empty_message_when_no_hybrid_docs(mocker):
    mocker.patch("app.tools.knowledge_tool.query_rewrite_service.rewrite_sync", return_value="rewritten")
    mocker.patch("app.tools.knowledge_tool.hybrid_search_service.search_sync", return_value=[])
    rerank = mocker.patch("app.tools.knowledge_tool.rerank_service.rerank")

    content, artifact = retrieve_knowledge.func("question", runtime_config=None)

    rerank.assert_not_called()
    assert content == "没有找到相关信息。"
    assert artifact == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_knowledge_tool.py -v`
Expected: FAIL because `knowledge_tool.py` does not import or call `rerank_service`.

**Step 3: Write minimal implementation**

Modify `app/tools/knowledge_tool.py`:

- Add:

```python
from app.services.rerank_service import rerank_service
```

- Replace:

```python
docs = hybrid_search_service.search_sync(rewritten_query, top_k=config.rag_top_k)
```

With:

```python
candidates = hybrid_search_service.search_sync(
    rewritten_query,
    top_k=config.rag_recall_size,
)

if not candidates:
    logger.warning("未检索到相关文档")
    return "没有找到相关信息。", []

if config.rag_rerank_enabled:
    docs = rerank_service.rerank(
        rewritten_query,
        candidates,
        top_k=config.rag_top_k,
    )
else:
    docs = candidates[: config.rag_top_k]
```

Keep the rest of the function behavior unchanged.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_knowledge_tool.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/tools/knowledge_tool.py tests/tools/test_knowledge_tool.py
git commit -m "feat: route knowledge retrieval through rerank"
```

### Task 5: Add rerank logging and mismatch fallback coverage

**Files:**
- Modify: `app/services/rerank_service.py`
- Modify: `tests/services/test_rerank_service.py`

**Step 1: Write the failing test**

Add a test that verifies score-length mismatch falls back cleanly:

```python
def test_rerank_falls_back_when_score_length_mismatches(mocker):
    service = RerankService()
    docs = [
        Document(page_content="doc-a", metadata={"id": "a"}),
        Document(page_content="doc-b", metadata={"id": "b"}),
    ]

    mocker.patch.object(service, "_score_pairs", return_value=[0.5])

    ranked = service.rerank("query", docs, top_k=2)

    assert [doc.metadata["id"] for doc in ranked] == ["a", "b"]
```

Optionally add a log assertion if the repository already uses `caplog`; if not, keep the test focused on fallback behavior.

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_rerank_service.py::test_rerank_falls_back_when_score_length_mismatches -v`
Expected: FAIL if mismatch handling is missing or incomplete.

**Step 3: Write minimal implementation**

Ensure `app/services/rerank_service.py` logs both success and fallback:

```python
logger.info(
    f"Rerank 完成: candidates={len(docs)}, returned={len(result)}"
)
```

and keep this guard:

```python
if len(scores) != len(docs):
    raise ValueError("score length mismatch")
```

Do not add retries, score thresholds, or extra weighting.

**Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_rerank_service.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/services/rerank_service.py tests/services/test_rerank_service.py
git commit -m "feat: add rerank fallback logging"
```

### Task 6: Verify the final retrieval behavior at the unit level

**Files:**
- Modify: `tests/services/test_rerank_service.py`
- Modify: `tests/tools/test_knowledge_tool.py`

**Step 1: Write the failing test**

Add one integration-style unit test that exercises the happy path through `retrieve_knowledge` with 20 candidates and 5 returned docs:

```python
def test_retrieve_knowledge_returns_top_k_reranked_docs(mocker):
    docs = [
        Document(page_content=f"doc-{i}", metadata={"id": str(i), "_file_name": f"{i}.md"})
        for i in range(20)
    ]
    reranked = docs[10:15]

    mocker.patch("app.tools.knowledge_tool.query_rewrite_service.rewrite_sync", return_value="rewritten")
    mocker.patch("app.tools.knowledge_tool.hybrid_search_service.search_sync", return_value=docs)
    mocker.patch("app.tools.knowledge_tool.rerank_service.rerank", return_value=reranked)

    content, artifact = retrieve_knowledge.func("question", runtime_config=None)

    assert len(artifact) == 5
    assert artifact == reranked
    assert "10.md" in content
    assert "14.md" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_knowledge_tool.py::test_retrieve_knowledge_returns_top_k_reranked_docs -v`
Expected: FAIL until the rerank path is wired as designed.

**Step 3: Write minimal implementation**

Only fix the exact behavior needed for the test if previous tasks did not already satisfy it. Do not add broader abstractions.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_knowledge_tool.py -v && pytest tests/services/test_rerank_service.py -v`
Expected: PASS for all rerank-related tests.

**Step 5: Commit**

```bash
git add tests/tools/test_knowledge_tool.py tests/services/test_rerank_service.py
git commit -m "test: cover rerank retrieval flow"
```

### Task 7: Update the design-adjacent docs to reflect the new default behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-05-13-rerank-design.md` (only if implementation changed a user-visible decision)

**Step 1: Write the failing test**

No automated test is required here. Instead, identify the exact README lines that currently describe hybrid retrieval without rerank.

**Step 2: Verify the doc gap exists**

Run: `python -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8'); print('rerank' in text.lower())"`
Expected: `False` or no mention of rerank behavior.

**Step 3: Write minimal implementation**

Update `README.md` so the RAG description reflects:

- query rewrite before retrieval
- hybrid recall (vector + BM25)
- rerank before handing context to the LLM

Keep the edit concise; do not add a long architecture section unless the README already has one.

**Step 4: Verify the doc update**

Run: `python -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8'); print('rerank' in text.lower())"`
Expected: `True`.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: mention rerank retrieval stage"
```

## Final verification checklist

After completing all tasks, run only the targeted checks for the changed area:

1. `pytest tests/test_config.py -v`
2. `pytest tests/services/test_rerank_service.py -v`
3. `pytest tests/tools/test_knowledge_tool.py -v`

If the rerank dependency introduces import-time issues, also run:

4. `python -c "from app.services.rerank_service import rerank_service; print(type(rerank_service).__name__)"`

Expected final state:

- `retrieve_knowledge()` recalls 20 docs, reranks them, returns 5
- rerank is guarded by config
- rerank failures fall back to hybrid top 5
- rerank model is lazily loaded
- rerank behavior is covered by focused unit tests
- README matches the new retrieval pipeline
