# Ragas Context Relevance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the offline ragas evaluation flow so each sample includes the real retrieved contexts used during evaluation and reports a `ContextRelevance` score alongside the existing answer-quality metrics.

**Architecture:** Keep the current CLI entrypoint and report shape, but teach the evaluation flow to capture retrieval output explicitly before answer generation. Add a small evaluation-specific return type for answer generation, pass `retrieved_contexts` into the ragas dataset, and append `ContextRelevance` to the metric list without changing the online request path.

**Tech Stack:** Python 3.11, ragas 0.4.x, datasets, LangChain `Document`, pytest

---

### Task 1: Add a focused retrieval helper for evaluation

**Files:**
- Modify: `app/tools/knowledge_tool.py:16-107`
- Create: `tests/tools/test_knowledge_tool.py`

**Step 1: Write the failing test**

Create `tests/tools/test_knowledge_tool.py` with two tests that exercise a new plain-Python helper instead of the LangChain tool wrapper:

```python
from langchain_core.documents import Document

from app.tools.knowledge_tool import retrieve_knowledge_documents


def test_retrieve_knowledge_documents_returns_reranked_docs(mocker):
    docs = [
        Document(page_content="doc-a", metadata={"_file_name": "a.md"}),
        Document(page_content="doc-b", metadata={"_file_name": "b.md"}),
    ]
    mocker.patch("app.tools.knowledge_tool.query_rewrite_service.rewrite_sync", return_value="rewritten")
    mocker.patch("app.tools.knowledge_tool.hybrid_search_service.search_sync", return_value=docs)
    mocker.patch("app.tools.knowledge_tool.rerank_service.rerank", return_value=[docs[1]])

    result = retrieve_knowledge_documents("question", session_id="eval-1")

    assert [doc.page_content for doc in result] == ["doc-b"]


def test_retrieve_knowledge_documents_returns_empty_list_when_no_candidates(mocker):
    mocker.patch("app.tools.knowledge_tool.query_rewrite_service.rewrite_sync", return_value="rewritten")
    mocker.patch("app.tools.knowledge_tool.hybrid_search_service.search_sync", return_value=[])

    assert retrieve_knowledge_documents("question", session_id="eval-1") == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_knowledge_tool.py -v`
Expected: FAIL because `retrieve_knowledge_documents` does not exist.

**Step 3: Write minimal implementation**

Refactor `app/tools/knowledge_tool.py` so the retrieval orchestration lives in a shared helper:

```python
def retrieve_knowledge_documents(query: str, session_id: str | None = None) -> list[Document]:
    rewritten_query = query_rewrite_service.rewrite_sync(query, session_id)
    candidates = hybrid_search_service.search_sync(
        rewritten_query,
        top_k=config.rag_recall_size,
    )
    if not candidates:
        return []
    if config.rag_rerank_enabled:
        return rerank_service.rerank(rewritten_query, candidates, top_k=config.rag_top_k)
    return candidates[: config.rag_top_k]
```

Then change `retrieve_knowledge()` to call `retrieve_knowledge_documents()` and keep using `format_docs()` for the returned documents. Do not change the tool signature.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_knowledge_tool.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/tools/knowledge_tool.py tests/tools/test_knowledge_tool.py
git commit -m "refactor: expose retrieval helper for evaluation"
```

### Task 2: Return retrieved contexts from the evaluation answer generator

**Files:**
- Modify: `app/eval/answer_generator.py:1-18`
- Create: `tests/eval/test_answer_generator.py`

**Step 1: Write the failing test**

Create `tests/eval/test_answer_generator.py` with a test that asserts the evaluation layer can return both answer text and retrieved contexts:

```python
from langchain_core.documents import Document

from app.eval.answer_generator import generate_answer_with_context


async def test_generate_answer_with_context_returns_answer_and_contexts(mocker):
    docs = [
        Document(page_content="ctx-1", metadata={}),
        Document(page_content="ctx-2", metadata={}),
    ]
    mocker.patch("app.eval.answer_generator.retrieve_knowledge_documents", return_value=docs)
    mocker.patch("app.eval.answer_generator.rag_agent_service.query", return_value="final answer")

    result = await generate_answer_with_context("question", session_id="eval-1")

    assert result.answer == "final answer"
    assert result.retrieved_contexts == ["ctx-1", "ctx-2"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_answer_generator.py -v`
Expected: FAIL because `generate_answer_with_context` does not exist.

**Step 3: Write minimal implementation**

In `app/eval/answer_generator.py`:

- Add a dataclass:

```python
@dataclass(slots=True)
class EvalAnswerResult:
    answer: str
    retrieved_contexts: list[str]
```

- Keep `generate_answer()` as a compatibility wrapper.
- Add:

```python
async def generate_answer_with_context(question: str, session_id: str | None = None) -> EvalAnswerResult:
    eval_session_id = session_id or f"eval-{uuid4().hex}"
    docs = retrieve_knowledge_documents(question, session_id=eval_session_id)
    try:
        answer = await rag_agent_service.query(question=question, session_id=eval_session_id)
    except Exception as exc:
        raise EvalAnswerGenerationError(str(exc)) from exc
    return EvalAnswerResult(
        answer=answer,
        retrieved_contexts=[doc.page_content for doc in docs],
    )
```

- Update `generate_answer()` to call `generate_answer_with_context()` and return only `.answer`.

Import `retrieve_knowledge_documents` from `app.tools.knowledge_tool`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_answer_generator.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/eval/answer_generator.py tests/eval/test_answer_generator.py
git commit -m "feat: return retrieved contexts in eval answer generation"
```

### Task 3: Store retrieved contexts in evaluation details

**Files:**
- Modify: `app/eval/ragas_runner.py:19-131`
- Create: `tests/eval/test_ragas_runner.py`

**Step 1: Write the failing test**

Create `tests/eval/test_ragas_runner.py` with a test that verifies successful details retain retrieved contexts:

```python
from app.eval.dataset import EvalSample
from app.eval.ragas_runner import run_ragas_evaluation


async def test_run_ragas_evaluation_stores_retrieved_contexts(mocker):
    sample = EvalSample(id="1", question="q", ground_truth="gt")
    mocker.patch(
        "app.eval.ragas_runner.generate_answer_with_context",
        return_value=mocker.Mock(answer="ans", retrieved_contexts=["ctx-1", "ctx-2"]),
    )
    mocker.patch("app.eval.ragas_runner.evaluate", return_value=mocker.Mock(scores=[]))

    report = await run_ragas_evaluation([sample])

    assert report.details[0].retrieved_contexts == ["ctx-1", "ctx-2"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_ragas_runner.py::test_run_ragas_evaluation_stores_retrieved_contexts -v`
Expected: FAIL because `EvalDetail` has no `retrieved_contexts` field and the runner still calls `generate_answer()`.

**Step 3: Write minimal implementation**

In `app/eval/ragas_runner.py`:

- Extend `EvalDetail` with:

```python
retrieved_contexts: list[str] = field(default_factory=list)
```

- Replace the `generate_answer()` call with `generate_answer_with_context()`.
- On success, set both `detail.answer` and `detail.retrieved_contexts`.
- Keep current error handling and summary counting unchanged.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_ragas_runner.py::test_run_ragas_evaluation_stores_retrieved_contexts -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/eval/ragas_runner.py tests/eval/test_ragas_runner.py
git commit -m "feat: store retrieved contexts in eval details"
```

### Task 4: Pass retrieved contexts into the ragas dataset

**Files:**
- Modify: `app/eval/ragas_runner.py:69-77`
- Modify: `tests/eval/test_ragas_runner.py`

**Step 1: Write the failing test**

Extend `tests/eval/test_ragas_runner.py` with a dataset payload test:

```python
from app.eval.ragas_runner import EvalDetail, _build_dataset_payload


def test_build_dataset_payload_includes_retrieved_contexts():
    detail = EvalDetail(
        id="1",
        question="q",
        ground_truth="gt",
        answer="ans",
        retrieved_contexts=["ctx-1", "ctx-2"],
    )

    dataset = _build_dataset_payload([detail])

    assert dataset.column_names == ["question", "answer", "ground_truth", "retrieved_contexts"]
    assert dataset[0]["retrieved_contexts"] == ["ctx-1", "ctx-2"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_ragas_runner.py::test_build_dataset_payload_includes_retrieved_contexts -v`
Expected: FAIL because `_build_dataset_payload()` does not include the new column.

**Step 3: Write minimal implementation**

Modify `_build_dataset_payload()` in `app/eval/ragas_runner.py` to include:

```python
"retrieved_contexts": [detail.retrieved_contexts for detail in successful_details],
```

Keep the success filtering logic unchanged.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_ragas_runner.py::test_build_dataset_payload_includes_retrieved_contexts -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/eval/ragas_runner.py tests/eval/test_ragas_runner.py
git commit -m "feat: add retrieved contexts to ragas dataset"
```

### Task 5: Add the ContextRelevance metric to the evaluation metric list

**Files:**
- Modify: `app/eval/ragas_runner.py:8-12,60-66`
- Modify: `tests/eval/test_ragas_runner.py`

**Step 1: Write the failing test**

Extend `tests/eval/test_ragas_runner.py` with a metric-list test:

```python
from ragas.metrics.collections import ContextRelevance

from app.eval.ragas_runner import _build_metrics


def test_build_metrics_includes_context_relevance(mocker):
    mocker.patch("app.eval.ragas_runner._build_evaluator_llm", return_value=object())

    metrics = _build_metrics()

    assert any(isinstance(metric, ContextRelevance) for metric in metrics)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_ragas_runner.py::test_build_metrics_includes_context_relevance -v`
Expected: FAIL because `_build_metrics()` only returns three metrics.

**Step 3: Write minimal implementation**

Update `app/eval/ragas_runner.py` imports and metric builder:

```python
from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, ContextRelevance
```

Then append:

```python
ContextRelevance(llm=evaluator_llm),
```

Keep the current evaluator LLM reuse pattern.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_ragas_runner.py::test_build_metrics_includes_context_relevance -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add app/eval/ragas_runner.py tests/eval/test_ragas_runner.py
git commit -m "feat: add context relevance metric"
```

### Task 6: Verify summary and detail output include the new metric data

**Files:**
- Modify: `app/eval/ragas_runner.py:80-131`
- Modify: `tests/eval/test_ragas_runner.py`

**Step 1: Write the failing test**

Extend `tests/eval/test_ragas_runner.py` with an end-to-end unit test for score extraction:

```python
from app.eval.dataset import EvalSample


async def test_run_ragas_evaluation_records_context_relevance_score(mocker):
    sample = EvalSample(id="1", question="q", ground_truth="gt")
    mocker.patch(
        "app.eval.ragas_runner.generate_answer_with_context",
        return_value=mocker.Mock(answer="ans", retrieved_contexts=["ctx"]),
    )
    mocker.patch(
        "app.eval.ragas_runner.evaluate",
        return_value=mocker.Mock(scores=[{"context_relevance": 0.8, "faithfulness": 1.0}]),
    )

    report = await run_ragas_evaluation([sample])

    assert report.summary.metrics["context_relevance"] == 0.8
    assert report.details[0].scores["context_relevance"] == 0.8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_ragas_runner.py::test_run_ragas_evaluation_records_context_relevance_score -v`
Expected: FAIL until the new metric is present and propagated through the detail/summary structures.

**Step 3: Write minimal implementation**

Only adjust code if the previous tasks did not already satisfy the assertions. Do not add a special-case branch for `context_relevance`; the generic score extraction should handle it naturally.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_ragas_runner.py -v`
Expected: PASS for all ragas runner tests.

**Step 5: Commit**

```bash
git add app/eval/ragas_runner.py tests/eval/test_ragas_runner.py
git commit -m "test: cover context relevance reporting"
```

### Task 7: Add a dataset loader test that preserves optional metadata only

**Files:**
- Create: `tests/eval/test_dataset.py`
- Modify: `app/eval/dataset.py` only if the test reveals a current bug

**Step 1: Write the failing test**

Create `tests/eval/test_dataset.py` with a regression test that confirms the existing JSONL schema remains unchanged:

```python
import json

from app.eval.dataset import load_eval_dataset


def test_load_eval_dataset_keeps_current_required_fields(tmp_path):
    dataset = tmp_path / "sample.jsonl"
    dataset.write_text(
        json.dumps({"id": "1", "question": "q", "ground_truth": "gt", "metadata": {"tag": "a"}}),
        encoding="utf-8",
    )

    samples = load_eval_dataset(str(dataset))

    assert len(samples) == 1
    assert samples[0].metadata == {"tag": "a"}
```

**Step 2: Run test to verify it fails or passes**

Run: `pytest tests/eval/test_dataset.py -v`
Expected: PASS if no accidental schema drift was introduced; FAIL only if earlier edits broke the loader.

**Step 3: Write minimal implementation**

Only modify `app/eval/dataset.py` if the test exposes a regression. Do not add `retrieved_contexts` as a required input field for the dataset file.

**Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_dataset.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/eval/test_dataset.py app/eval/dataset.py
git commit -m "test: protect eval dataset schema"
```

### Task 8: Update the design-adjacent docs to mention retrieval-quality evaluation

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-05-14-ragas-evaluation-design.md`

**Step 1: Write the failing test**

No automated test is required. Instead, identify the exact places that still say the evaluation only covers final-answer quality.

**Step 2: Verify the doc gap exists**

Run: `python -c "from pathlib import Path; text = Path('docs/plans/2026-05-14-ragas-evaluation-design.md').read_text(encoding='utf-8'); print('检索' in text and '单独评分' in text)"`
Expected: `True`, showing the old document still excludes retrieval evaluation.

**Step 3: Write minimal implementation**

Update the docs so they now say:
- the offline evaluation still centers on final answers
- it also records real retrieved contexts during evaluation
- it adds a retrieval-quality metric for context relevance

Keep the edits short and factual. Do not rewrite the whole design history.

**Step 4: Verify the doc update**

Run: `python -c "from pathlib import Path; text = Path('docs/plans/2026-05-14-ragas-evaluation-design.md').read_text(encoding='utf-8'); print('ContextRelevance' in text or '上下文相关性' in text)"`
Expected: `True`.

**Step 5: Commit**

```bash
git add README.md docs/plans/2026-05-14-ragas-evaluation-design.md
git commit -m "docs: mention context relevance evaluation"
```

## Final verification checklist

After completing all tasks, run only the targeted checks for the changed area:

1. `pytest tests/tools/test_knowledge_tool.py -v`
2. `pytest tests/eval/test_answer_generator.py -v`
3. `pytest tests/eval/test_ragas_runner.py -v`
4. `pytest tests/eval/test_dataset.py -v`
5. `python -c "from app.eval.ragas_runner import _build_metrics; print([type(metric).__name__ for metric in _build_metrics()])"`

Expected final state:

- the evaluation path can retrieve the same document texts it scores
- `EvalDetail` includes `retrieved_contexts`
- the ragas dataset includes `retrieved_contexts`
- `_build_metrics()` includes `ContextRelevance`
- report summaries and per-sample scores include `context_relevance`
- the input dataset schema remains `id/question/ground_truth` with optional `metadata`
- the online query path remains unchanged for normal users
