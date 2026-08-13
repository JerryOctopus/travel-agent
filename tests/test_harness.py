from __future__ import annotations

from travel_agent.harness import (
    AgentHarness,
    HarnessCase,
    HarnessCaseResult,
    HarnessEnvironment,
    HarnessSuiteResult,
    HarnessTurnResult,
    aggregate_case_results,
    aggregate_real_multi_agent_results,
    build_harness_report,
    load_cases_json,
)
from travel_agent.harness.chinatravel import build_chinatravel_summary
from travel_agent.harness.live_tools import LiveToolsHarnessRunner
from travel_agent.harness.reporting import evaluate_release_gates


def test_harness_runs_single_turn_case(offline_settings):
    harness = AgentHarness(
        settings=offline_settings,
        environment=HarnessEnvironment(mode="offline"),
    )
    result = harness.run_case(
        HarnessCase(
            case_id="hangzhou_3d",
            turns=["杭州三天随便"],
            expected_city="杭州",
            expected_days=3,
            expect_clarification=False,
            expect_itinerary=True,
            expected_tools=["plan_and_critique"],
        )
    )

    assert result.errors == []
    assert result.passed is True
    assert result.final_profile["destination"] == "杭州"
    assert result.final_profile["days"] == 3
    assert "itinerary" in result.final_artifacts
    assert result.metrics["tool_coverage_ok"] is True


def test_harness_runs_multi_turn_case_with_history(offline_settings):
    harness = AgentHarness(
        settings=offline_settings,
        environment=HarnessEnvironment(mode="offline"),
    )
    result = harness.run_case(
        HarnessCase(
            case_id="beijing_multi_turn",
            turns=["我想去北京", "玩两天，喜欢历史"],
            expected_city="北京",
            expected_days=2,
            expect_clarification=False,
            expect_itinerary=True,
        )
    )

    assert len(result.turns) == 2
    assert result.turns[0].clarification is True
    assert result.turns[1].clarification is False
    assert result.final_profile["destination"] == "北京"
    assert result.final_profile["days"] == 2
    assert "itinerary" in result.final_artifacts


def test_harness_loads_existing_eval_cases():
    cases = load_cases_json("eval/cases.json")

    assert cases
    assert cases[0].case_id == "beijing_history_food_2d"
    assert cases[0].turns == ["帮我规划北京两天，喜欢历史和美食，不要太累"]


def test_harness_reporting_keeps_nl_eval_summary_shape(offline_settings):
    harness = AgentHarness(
        settings=offline_settings,
        environment=HarnessEnvironment(mode="offline"),
    )
    cases = [
        HarnessCase(
            case_id="beijing_case",
            turns=["北京两天历史游，随便"],
            expected_city="北京",
            expected_days=2,
            expect_itinerary=True,
        ),
        HarnessCase(
            case_id="missing_case",
            turns=["我想轻松一点，喜欢自然和美食"],
            expect_clarification=True,
        ),
    ]

    summary = aggregate_case_results(harness.run_cases(cases), cases)
    report = build_harness_report(summary)

    assert summary["sample_size"] == 2
    assert summary["city_accuracy"] == 1.0
    assert summary["days_accuracy"] == 1.0
    assert summary["clarification_accuracy"] == 1.0
    assert "task_completion_rate" in summary
    assert "Harness NL Eval Report" in report


def test_harness_reporting_aggregates_real_multi_agent_shape():
    case = HarnessCase(case_id="real_case", turns=["杭州三天随便"])
    result = HarnessCaseResult(
        case_id="real_case",
        turns=[
            HarnessTurnResult(
                user_message="杭州三天随便",
                reply_text="done",
                tool_trace=["search_poi", "recommend_candidates", "plan_and_critique"],
                used_real_agent=True,
                clarification=False,
                profile={"destination": "杭州", "days": 3},
                artifacts={},
            )
        ],
        final_profile={"destination": "杭州", "days": 3},
        final_artifacts={
            "itinerary": {"critic": {"passed": True}},
            "agent_trace": {
                "items": [
                    {
                        "agent": "attraction",
                        "kind": "subagent",
                        "status": "completed",
                        "detail": {"tool_trace": ["search_poi"]},
                    },
                    {
                        "agent": "hotel",
                        "kind": "subagent",
                        "status": "completed",
                        "detail": {"tool_trace": ["search_hotel"]},
                    },
                    {
                        "agent": "restaurant",
                        "kind": "subagent",
                        "status": "completed",
                        "detail": {"tool_trace": ["search_restaurant"]},
                    },
                    {
                        "agent": "transport",
                        "kind": "subagent",
                        "status": "completed",
                        "detail": {"tool_trace": ["plan_route"]},
                    },
                    {
                        "agent": "planner",
                        "kind": "subagent",
                        "status": "completed",
                        "detail": {
                            "tool_trace": ["recommend_candidates", "plan_and_critique"]
                        },
                    },
                ]
            },
        },
        metrics={
            "critic_passed": True,
            "environment_pass": True,
            "constraint_pass": True,
            "preference_pass": True,
            "final_pass": True,
        },
    )

    summary = aggregate_real_multi_agent_results(
        [result],
        [case],
        allowed_by_agent={
            "attraction": {"search_poi", "check_weather"},
            "hotel": {"search_hotel"},
            "restaurant": {"search_restaurant", "estimate_budget"},
            "transport": {"plan_route", "estimate_budget"},
            "planner": {"build_constraints", "recommend_candidates", "plan_and_critique"},
        },
    )

    assert summary["sample_size"] == 1
    assert summary["used_real_agent_rate"] == 1.0
    assert summary["full_multi_agent_pass_rate"] == 1.0


def test_harness_suite_result_and_chinatravel_summary_shape():
    result = HarnessSuiteResult(
        suite="chinatravel-mini-dev",
        mode="deterministic",
        environment="offline",
        case_count=2,
        attempted_count=2,
        metrics={
            "official_eval_available": True,
            "delivery_rate": 1.0,
            "schema_pass_rate": 100.0,
            "fpr": 0.5,
        },
        rows=[{"query_id": "a", "delivered": True}],
        artifacts={
            "suite": "mini-dev",
            "split": "human",
            "mode": "deterministic",
            "llm_provider": "rule",
            "llm_model": "test",
            "architecture": "multi_agent_full_v3",
            "predictions_dir": "/tmp/predictions",
            "diagnostics_dir": None,
            "prediction_file_count": 2,
            "multi_agent_metrics": None,
            "cumulative_multi_agent_metrics": None,
        },
    )

    summary = build_chinatravel_summary(result)

    assert summary["suite"] == "mini-dev"
    assert summary["metrics"]["fpr"] == 0.5
    assert summary["prediction_file_count"] == 2


def test_live_tools_harness_records_missing_replay_snapshot(tmp_path):
    case_root = tmp_path / "cases"
    case_root.mkdir()
    (case_root / "shadow_dev.json").write_text(
        '[{"id":"missing_snapshot_case","city":"杭州"}]',
        encoding="utf-8",
    )

    result = LiveToolsHarnessRunner(
        snapshot_root=tmp_path / "snapshots",
        case_root=case_root,
    ).run(provider="chinatravel", mode="replay", suite="shadow-dev")

    assert result.case_count == 1
    assert result.metrics["api_success_rate"] == 0.0
    assert result.rows[0]["operation"] == "snapshot"
    assert "missing snapshot" in result.rows[0]["error"]


def test_release_gates_report_blockers_for_failed_chinatravel():
    summary = evaluate_release_gates(
        {
            "agent-nl": {
                "task_completion_rate": 1.0,
                "clarification_accuracy": 1.0,
                "critic_pass_rate": 1.0,
                "fpr": 1.0,
            },
            "chinatravel-mini": {
                "metrics": {
                    "delivery_rate": 1.0,
                    "schema_pass_rate": 100.0,
                    "fpr": 0.0,
                }
            },
        }
    )

    assert summary["status"] == "blocked"
    assert summary["gates"]["agent-nl"]["passed"] is True
    assert summary["gates"]["chinatravel-mini"]["passed"] is False
