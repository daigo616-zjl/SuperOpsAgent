"""AIOps 基准纯函数单测（Fake 事件流，不依赖 MCP / API key）。"""

from app.eval.aiops_benchmark import (
    aggregate,
    claim_reference_metrics,
    judge_hallucination_rate,
    judge_hit,
    load_scenarios,
    parse_json_object,
    summarize_run,
)


def multiagent_events() -> list[dict]:
    return [
        {"type": "status", "stage": "supervisor", "message": "第 1 轮：无假设，生成"},
        {"type": "plan", "stage": "hypotheses_created", "plan": {"hypotheses": [1, 2]}},
        {"type": "status", "stage": "supervisor", "message": "第 2 轮：派发取证"},
        {
            "type": "step_complete",
            "stage": "investigated",
            "result": {
                "domain": "logs",
                "directive_id": "r2-logs",
                "claims": [
                    {"claim_id": "ev-r2-logs-1", "statement": "Full GC 频繁"},
                    {"claim_id": "ev-r2-logs-2", "statement": "STW 超阈值"},
                ],
            },
        },
        {"type": "report_chunk", "stage": "final_report", "data": "## 结论"},
        {
            "type": "complete",
            "stage": "complete",
            "response": "## 结论\nGC 压力过高 [ev-r2-logs-1] 与 [ev-bogus-9]",
        },
    ]


def plain_events() -> list[dict]:
    return [
        {
            "type": "plan",
            "stage": "plan_created",
            "plan": {"goal": "诊断", "steps": [{"id": "query_cpu"}]},
        },
        {
            "type": "step_complete",
            "stage": "step_executed",
            "result": {"step_id": "query_cpu", "status": "succeeded"},
        },
        {"type": "report", "stage": "final_report", "report": "# 报告\nCPU 偏高"},
        {"type": "complete", "stage": "complete", "response": "# 报告\nCPU 偏高"},
    ]


class TestSummarizeRun:
    def test_multiagent_extracts_report_claims_and_rounds(self) -> None:
        summary = summarize_run(multiagent_events())

        assert summary["report"].startswith("## 结论")
        assert summary["claim_ids"] == ["ev-r2-logs-1", "ev-r2-logs-2"]
        assert summary["rounds"] == 2
        assert summary["step_count"] == 1
        assert summary["report_chunk_count"] == 1
        assert summary["ev_refs_total"] == 2
        assert summary["ev_refs_unresolved"] == 1
        assert summary["error"] is None

    def test_plain_events_have_no_claims_or_rounds(self) -> None:
        summary = summarize_run(plain_events())

        assert summary["claim_ids"] == []
        assert summary["rounds"] == 0
        assert summary["ev_refs_total"] == 0
        assert summary["tool_outputs"] == [
            {"step_id": "query_cpu", "status": "succeeded"}
        ]

    def test_error_event_recorded(self) -> None:
        summary = summarize_run([{"type": "error", "message": "智能运维诊断出错: x"}])

        assert summary["error"] == "智能运维诊断出错: x"
        assert summary["report"] == ""


class TestClaimReferenceMetrics:
    def test_ratio(self) -> None:
        summary = summarize_run(multiagent_events())
        assert claim_reference_metrics(summary) == 0.5

    def test_none_when_no_refs(self) -> None:
        summary = summarize_run(plain_events())
        assert claim_reference_metrics(summary) is None


class TestParseJsonObject:
    def test_plain(self) -> None:
        assert parse_json_object('{"hit": true}') == {"hit": True}

    def test_fenced_with_prose(self) -> None:
        text = '好的，结果如下：\n```json\n{"hit": false, "reason": "不匹配"}\n```\n完毕'
        assert parse_json_object(text) == {"hit": False, "reason": "不匹配"}

    def test_invalid_returns_none(self) -> None:
        assert parse_json_object("完全没有 JSON") is None
        assert parse_json_object("{broken") is None


class TestAggregate:
    def make_run(self, scenario, run, hit, hall_rate) -> dict:
        return {
            "scenario": scenario,
            "run": run,
            "wall_seconds": 10.0,
            "llm_calls": 20,
            "rounds": 3,
            "judge_root_cause": (
                None if hit is None else {"hit": hit, "reason": "x"}
            ),
            "judge_hallucination": (
                None
                if hall_rate is None
                else {"total_claims": 10, "unsupported_claims": int(hall_rate * 10)}
            ),
        }

    def test_aggregate_groups_by_scenario(self) -> None:
        runs = [
            self.make_run("gc-pressure", 0, True, 0.2),
            self.make_run("gc-pressure", 1, False, 0.4),
            self.make_run("oom-kill", 0, True, 0.1),
        ]

        agg = aggregate(runs)
        assert agg["gc-pressure"]["hit_rate"] == 0.5
        assert agg["gc-pressure"]["hallucination_rate"] == 0.3
        assert agg["oom-kill"]["hit_rate"] == 1.0
        assert agg["oom-kill"]["hallucination_rate"] == 0.1

    def test_judge_hallucination_rate_zero_claims(self) -> None:
        run = self.make_run("s", 0, True, None)
        run["judge_hallucination"] = {"total_claims": 0, "unsupported_claims": 0}
        assert judge_hallucination_rate(run) == 0.0
        assert judge_hit(run) is True


class TestLoadScenarios:
    def test_loads_ground_truth(self) -> None:
        scenarios = load_scenarios(["gc-pressure"])
        assert scenarios["gc-pressure"]["service_name"] == "data-sync-service"
        assert "root_cause" in scenarios["gc-pressure"]["ground_truth"]
