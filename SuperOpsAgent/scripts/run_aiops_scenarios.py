"""AIOps 双引擎 A/B 场景基准入口（实现见 app/eval/aiops_benchmark.py）。

用法（需 DASHSCOPE_API_KEY，先停掉已运行的 MCP 服务）：
    python scripts/run_aiops_scenarios.py                       # 5 剧本 × 2 引擎 × 3 次
    python scripts/run_aiops_scenarios.py --scenarios gc-pressure --runs 1
    python scripts/run_aiops_scenarios.py --no-judge            # 只跑诊断不判分
"""

import sys

from app.eval.aiops_benchmark import main

if __name__ == "__main__":
    sys.exit(main())
