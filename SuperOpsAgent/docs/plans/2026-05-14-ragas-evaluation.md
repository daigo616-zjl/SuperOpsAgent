# Ragas Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在仓库内增加一个独立的离线 ragas 评测工具，读取 JSONL 测试集，批量生成最终回答并输出评分报告。

**Architecture:** 保持现有 FastAPI 与聊天接口不变，新增 `eval/` 目录存放数据集与报告，新增评测模块负责读取样本、调用现有回答生成逻辑、构造 ragas 数据集并写出结果。执行入口使用独立脚本，不通过 HTTP 接口触发。

**Tech Stack:** Python 3.11, ragas, argparse, json, pathlib, loguru

---

### Task 1: Add ragas dependency and evaluation configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`

**Step 1: Add ragas dependency**

在 `pyproject.toml` 的 `dependencies` 中加入 `ragas`，版本范围按实现时确认的当前兼容版本填写。

建议先放在主依赖中，因为评测工具和主仓库一起安装会更直接。

**Step 2: Add minimal evaluation settings**

在 `app/config.py` 中新增最小评测配置，例如：

```python
eval_model: str = "qwen-max"
eval_output_dir: str = "eval/reports"
```

如果实现时发现不需要配置项，则只保留 `eval_output_dir`，不要过度设计。

**Step 3: Verify config import**

Run:
```bash
python -c "from app.config import config; print(config.eval_output_dir)"
```

Expected: 输出 `eval/reports` 或实现时定义的默认值。

**Step 4: Commit**

```bash
git add pyproject.toml app/config.py
git commit -m "feat: add ragas evaluation dependency and config"
```

### Task 2: Add evaluation dataset and report directories

**Files:**
- Create: `eval/datasets/.gitkeep`
- Create: `eval/reports/.gitkeep`
- Create: `eval/fixtures/sample_ragas_dataset.jsonl`

**Step 1: Create dataset and report directories**

创建 `eval/datasets/` 和 `eval/reports/`，用 `.gitkeep` 保持目录进入版本控制。

**Step 2: Add sample dataset**

创建 `eval/fixtures/sample_ragas_dataset.jsonl`，至少包含两条样例数据，格式如下：

```json
{"id":"case-001","question":"CPU 持续升高时应该先看什么指标？","ground_truth":"应先查看 CPU 使用率趋势、负载、异常进程以及相关监控指标。"}
{"id":"case-002","question":"磁盘告警出现时先检查什么？","ground_truth":"应先检查磁盘使用率、inode、增长趋势以及异常目录或日志文件。"}
```

**Step 3: Verify sample file exists**

Run:
```bash
python -c "from pathlib import Path; print(Path('eval/fixtures/sample_ragas_dataset.jsonl').exists())"
```

Expected: 输出 `True`。

**Step 4: Commit**

```bash
git add eval/datasets/.gitkeep eval/reports/.gitkeep eval/fixtures/sample_ragas_dataset.jsonl
git commit -m "chore: add ragas evaluation directories and sample dataset"
```

### Task 3: Add dataset schema and loader

**Files:**
- Create: `app/eval/dataset.py`

**Step 1: Add sample model**

在 `app/eval/dataset.py` 中定义一个轻量数据结构，例如：

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class EvalSample:
    id: str
    question: str
    ground_truth: str
    metadata: dict[str, object] = field(default_factory=dict)
```

**Step 2: Add JSONL loader**

实现：

```python
def load_eval_dataset(path: str) -> list[EvalSample]:
```

要求：
- 逐行读取 JSONL
- 忽略空行
- 校验 `id/question/ground_truth`
- 校验失败时抛出带行号的 `ValueError`

**Step 3: Add serialization helper if useful**

如实现结果输出时需要，可增加：

```python
def sample_to_dict(sample: EvalSample) -> dict:
```

仅在后续模块复用时添加，避免无意义抽象。

**Step 4: Verify loader with sample file**

Run:
```bash
python -c "from app.eval.dataset import load_eval_dataset; print(len(load_eval_dataset('eval/fixtures/sample_ragas_dataset.jsonl')))"
```

Expected: 输出 `2`。

**Step 5: Commit**

```bash
git add app/eval/dataset.py
git commit -m "feat: add ragas evaluation dataset loader"
```

### Task 4: Add final-answer generation adapter for evaluation

**Files:**
- Create: `app/eval/answer_generator.py`
- Modify: `app/services/rag_agent_service.py` (only if a cleaner non-stream evaluation entry is needed)

**Step 1: Reuse existing final-answer path**

在 `app/eval/answer_generator.py` 中封装一个评测专用入口，例如：

```python
async def generate_answer(question: str, session_id: str) -> str:
    return await rag_agent_service.query(question=question, session_id=session_id)
```

**Step 2: Keep integration internal**

不要走 HTTP 请求，不要新增 API。

如果 `rag_agent_service.query()` 当前过于耦合，可在 `app/services/rag_agent_service.py` 内补一个更明确的非流式内部方法，但不要改变现有 API 行为。

**Step 3: Normalize failures**

评测适配层要把异常包装成可记录的错误信息，供 runner 决定“单条失败后继续”。

**Step 4: Verify import path**

Run:
```bash
python -c "from app.eval.answer_generator import generate_answer; print(callable(generate_answer))"
```

Expected: 输出 `True`。

**Step 5: Commit**

```bash
git add app/eval/answer_generator.py app/services/rag_agent_service.py
git commit -m "feat: add evaluation answer generator"
```

### Task 5: Add ragas runner and metric selection

**Files:**
- Create: `app/eval/ragas_runner.py`

**Step 1: Add runner input and output structures**

定义评测结果结构，至少覆盖：
- summary
- details
- errors

例如：

```python
from dataclasses import dataclass, field

@dataclass(slots=True)
class EvalDetail:
    id: str
    question: str
    ground_truth: str
    answer: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None
```

**Step 2: Build ragas dataset payload**

实现将成功生成 answer 的样本转换成 ragas `evaluate()` 可接受的数据集格式。

至少包含：
- `question`
- `answer`
- `ground_truth`

如果实现时当前 ragas 版本要求使用 Hugging Face Dataset 或特定对象格式，就按文档要求组装。

**Step 3: Select initial metrics**

第一版优先接入：
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`（仅在当前版本兼容时启用）

若某指标初始化依赖额外 LLM wrapper，则在 runner 中集中处理，不把细节泄漏到脚本层。

**Step 4: Support partial-failure execution**

当某条样本生成 answer 失败时：
- 记录到 `details`
- 不参与 ragas 评分集
- 整体继续执行

如果全部样本都失败，则返回一个明确错误，不写空评分。

**Step 5: Verify runner import**

Run:
```bash
python -c "from app.eval.ragas_runner import EvalDetail; print(EvalDetail.__name__)"
```

Expected: 输出 `EvalDetail`。

**Step 6: Commit**

```bash
git add app/eval/ragas_runner.py
git commit -m "feat: add ragas evaluation runner"
```

### Task 6: Add report writer

**Files:**
- Create: `app/eval/report_writer.py`

**Step 1: Add timestamped output path builder**

实现一个帮助函数，例如：

```python
def build_report_path(dataset_path: str) -> Path:
```

要求：
- 输出到 `eval/reports/`
- 文件名包含时间戳
- 文件名可包含数据集 stem

**Step 2: Add JSON report writer**

实现：

```python
def write_report(report: dict, output_path: Path) -> Path:
```

要求：
- 自动创建父目录
- UTF-8 编码
- `ensure_ascii=False`
- 缩进输出，便于人工查看

**Step 3: Verify writer import**

Run:
```bash
python -c "from app.eval.report_writer import build_report_path; print(build_report_path('eval/fixtures/sample_ragas_dataset.jsonl').suffix)"
```

Expected: 输出 `.json`。

**Step 4: Commit**

```bash
git add app/eval/report_writer.py
git commit -m "feat: add ragas evaluation report writer"
```

### Task 7: Add CLI entry script for offline evaluation

**Files:**
- Create: `scripts/run_ragas_eval.py`

**Step 1: Add CLI argument parsing**

支持至少这些参数：
- `--dataset`：必填，JSONL 路径
- `--output`：可选，自定义报告路径

不要一开始加太多参数。

**Step 2: Orchestrate the evaluation flow**

脚本负责：
1. 调用 `load_eval_dataset()`
2. 调用评测 runner 生成结果
3. 调用 `write_report()` 落盘
4. 在终端打印摘要

**Step 3: Print concise summary**

终端至少打印：
- dataset path
- total
- success
- failed
- metrics summary
- report path

**Step 4: Verify help output**

Run:
```bash
python scripts/run_ragas_eval.py --help
```

Expected: 正常打印命令帮助。

**Step 5: Commit**

```bash
git add scripts/run_ragas_eval.py
git commit -m "feat: add ragas evaluation cli"
```

### Task 8: Add minimal tests for dataset and report modules

**Files:**
- Create: `tests/eval/test_dataset.py`
- Create: `tests/eval/test_report_writer.py`

**Step 1: Write dataset loader tests**

至少覆盖：
- 正常加载两条样本
- 缺字段时报错
- 空行被忽略

示例：

```python
def test_load_eval_dataset_reads_jsonl(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        '{"id":"1","question":"q","ground_truth":"a"}\n',
        encoding="utf-8",
    )

    samples = load_eval_dataset(str(path))

    assert len(samples) == 1
    assert samples[0].id == "1"
```

**Step 2: Write report writer tests**

至少覆盖：
- 自动创建目录
- 正常写出 JSON
- 文件内容可读取

**Step 3: Run targeted tests**

Run:
```bash
pytest tests/eval/test_dataset.py tests/eval/test_report_writer.py -q
```

Expected: PASS。

**Step 4: Commit**

```bash
git add tests/eval/test_dataset.py tests/eval/test_report_writer.py
git commit -m "test: cover ragas evaluation dataset and report writer"
```

### Task 9: Add README usage section for offline evaluation

**Files:**
- Modify: `README.md`

**Step 1: Add evaluation overview**

在 README 中新增一个简短章节，说明：
- 这是离线评测工具
- 输入是 JSONL 数据集
- 输出是 JSON 报告

**Step 2: Add command example**

加入示例：

```bash
python scripts/run_ragas_eval.py --dataset eval/fixtures/sample_ragas_dataset.jsonl
```

**Step 3: Add dataset format example**

展示一行 JSONL 示例即可，不写成长篇文档。

**Step 4: Verify text presence**

Run:
```bash
python -c "from pathlib import Path; print('run_ragas_eval.py' in Path('README.md').read_text(encoding='utf-8'))"
```

Expected: 输出 `True`。

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document offline ragas evaluation"
```

### Task 10: Smoke-check the CLI flow

**Files:**
- Modify: none

**Step 1: Install updated dependencies**

Run:
```bash
pip install -e .
```

Expected: `ragas` 安装成功。

**Step 2: Run the CLI on the sample dataset**

Run:
```bash
python scripts/run_ragas_eval.py --dataset eval/fixtures/sample_ragas_dataset.jsonl
```

Expected: 命令完成，终端打印摘要，并在 `eval/reports/` 下生成报告文件。

**Step 3: Inspect the generated report file**

Run:
```bash
python -c "from pathlib import Path; files = sorted(Path('eval/reports').glob('*.json')); print(files[-1].name if files else 'missing')"
```

Expected: 输出最新报告文件名。

**Step 4: Final commit if verification required changes**

```bash
git status --short
```

Expected: 无额外意外修改。
