from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from travel_agent.agent.lc_tools import build_tools
from travel_agent.agent.evaluation_trace import EvaluationTraceCallback
from travel_agent.agent.session import build_session
from travel_agent.evaluation.chinatravel_adapter import (
    build_chinatravel_provider,
    preflight_chinatravel,
)
from travel_agent.harness import (
    HarnessCase,
    HarnessCaseResult,
    HarnessSuiteResult,
    HarnessTurnResult,
)
from travel_agent.harness.cases import load_cases_json
from travel_agent.harness.faults import FaultInjectingProvider
from travel_agent.harness.live_tools import _redact_snapshot
from travel_agent.harness.product import (
    DEFAULT_PRODUCT_CASES,
    DEFAULT_PRODUCT_DEV_CASES,
    aggregate_product_rows,
    attribute_failures,
    decide_fine_tuning,
    validate_absolute_return_deadline_evidence,
    wilson_interval,
    _configuration_fingerprint,
    _execution_case,
    _product_case_output,
    _product_row,
    _tool_snapshot_fingerprint,
    write_product_run,
)
from travel_agent.harness.validators import validate_case_result


def test_production_v1_dataset_has_192_cases_with_frozen_split_discipline() -> None:
    expected = {
        "dev.jsonl": (34, "6460b21b8a7c95a5b989fb17472e2dc67c52af16d2a81c2f1290efcf221d91f3"),
        "core_frozen.jsonl": (94, "86e7636e722936612788b578664e34bfaa40fb8fac4b03890ab06b0f30705178"),
        "challenge_frozen.jsonl": (34, "f7fccae88d0d6d91929163cb1828a31ed51345be69e7b9e0ba67bf6ea88e4c78"),
        "shadow_frozen.jsonl": (30, "d9b4be53bb2259f690ce080415b5315b2510c54cb1d1f6a9fb50f480bd66d63f"),
    }
    for name, (count, digest) in expected.items():
        path = DEFAULT_PRODUCT_CASES.parent / name
        assert sum(bool(line.strip()) for line in path.read_bytes().splitlines()) == count
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_formal_run_fingerprints_config_without_hashing_secrets(offline_settings) -> None:
    configured = replace(
        offline_settings,
        llm=replace(
            offline_settings.llm,
            provider="deepseek",
            model="deepseek-chat",
            api_key="first-secret",
        ),
        amap=replace(offline_settings.amap, web_key="first-amap-secret"),
    )
    rotated_secrets = replace(
        configured,
        llm=replace(configured.llm, api_key="second-secret"),
        amap=replace(configured.amap, web_key="second-amap-secret"),
    )
    changed_temperature = replace(
        configured,
        llm=replace(configured.llm, temperature=0.7),
    )

    assert _configuration_fingerprint(configured, "configured") == (
        _configuration_fingerprint(rotated_secrets, "configured")
    )
    assert _configuration_fingerprint(configured, "configured") != (
        _configuration_fingerprint(changed_temperature, "configured")
    )
    assert _tool_snapshot_fingerprint(configured, "configured") == (
        _tool_snapshot_fingerprint(rotated_secrets, "configured")
    )
    assert _tool_snapshot_fingerprint(configured, "configured") != (
        _tool_snapshot_fingerprint(configured, "local")
    )


def test_absolute_return_deadline_accepts_explicit_relative_last_day_semantics() -> None:
    case = HarnessCase(
        case_id="ungrounded-deadline",
        turns=["10月去杭州五天。", "最后一天17点前返程。"],
        split="dev",
        snapshot_date="2026-08-11T12:00:00+08:00",
        gold_constraints_tree={
            "date_start": "2026-10-01",
            "duration_days": 5,
            "return_deadline": "2026-10-05T17:00:00+08:00",
        },
    )

    errors, frozen_warnings = validate_absolute_return_deadline_evidence([case])

    assert errors == []
    assert frozen_warnings == []


def test_absolute_return_deadline_rejects_relative_time_that_disagrees_with_gold() -> None:
    case = HarnessCase(
        case_id="mismatched-relative-deadline",
        turns=["10月去杭州五天。", "最后一天18点前返程。"],
        split="dev",
        gold_constraints_tree={
            "duration_days": 5,
            "return_deadline": "2026-10-05T17:00:00+08:00",
        },
    )

    errors, _warnings = validate_absolute_return_deadline_evidence([case])

    assert errors == [
        "mismatched-relative-deadline: absolute return_deadline date 2026-10-05 "
        "must be derivable from explicit user date_start/date_end and duration_days"
    ]


def test_absolute_return_deadline_accepts_user_grounded_start_and_duration() -> None:
    case = HarnessCase(
        case_id="grounded-deadline",
        turns=["2026年10月1日起杭州五天。", "最后一天17点前返程。"],
        split="dev",
        gold_constraints_tree={
            "date_start": "2026-10-01",
            "duration_days": 5,
            "return_deadline": "2026-10-05T17:00:00+08:00",
        },
    )

    assert validate_absolute_return_deadline_evidence([case]) == ([], [])


def test_frozen_ungrounded_absolute_return_deadline_is_report_only() -> None:
    case = HarnessCase(
        case_id="frozen-ungrounded-deadline",
        turns=["10月1日起杭州五天。", "17点前返程。"],
        split="core_frozen",
        gold_constraints_tree={
            "date_start": "2026-10-01",
            "duration_days": 5,
            "return_deadline": "2026-10-05T17:00:00+08:00",
        },
    )

    errors, frozen_warnings = validate_absolute_return_deadline_evidence([case])

    assert errors == []
    assert frozen_warnings == [
        "frozen-ungrounded-deadline: absolute return_deadline date 2026-10-05 "
        "must be derivable from explicit user date_start/date_end and duration_days"
    ]


def test_production_dataset_loader_parses_external_schema() -> None:
    cases = load_cases_json(DEFAULT_PRODUCT_DEV_CASES)
    case = cases[0]

    assert case.turns and all(turn.strip() for turn in case.turns)
    assert case.gold_constraints_tree
    assert case.fixture_id
    assert case.architecture_policy.get("shared_tool") == "plan_and_critique"
    assert case.metadata.get("title")


def test_shadow_split_is_rejected_for_multi_variant_comparison(monkeypatch) -> None:
    import scripts.eval_ablation as ablation

    monkeypatch.setattr(
        "sys.argv",
        ["eval_ablation.py", "--variants", "v0,v3", "--product-split", "shadow_frozen"],
    )
    with pytest.raises(SystemExit, match="当前发布周期"):
        ablation.main()


def test_single_variant_is_allowed_on_shadow_split(monkeypatch, offline_settings) -> None:
    import scripts.eval_ablation as ablation

    monkeypatch.setattr(
        "sys.argv",
        ["eval_ablation.py", "--variants", "v3", "--product-split", "shadow_frozen"],
    )
    monkeypatch.setattr(ablation, "load_settings", lambda: offline_settings)
    with pytest.raises(SystemExit, match="当前发布周期"):
        ablation.main()


def test_product_executions_isolate_users_between_cases_and_repeats() -> None:
    case = HarnessCase(case_id="isolation", turns=["杭州两天"])
    first = _execution_case(case, 1)
    repeated = _execution_case(case, 2)
    another_run = _execution_case(case, 1, run_namespace="run-b")
    explicit = _execution_case(
        HarnessCase(case_id="two-users", turns=["a", "b"], user_ids=["owner", "other"]),
        1,
    )

    assert first.metadata["user_id"] != repeated.metadata["user_id"]
    assert first.metadata["user_id"] != another_run.metadata["user_id"]
    assert "run-b" in another_run.metadata["user_id"]
    assert explicit.user_ids[0] != "owner"
    assert explicit.user_ids[0] != explicit.user_ids[1]


def test_tool_argument_and_memory_assertions_are_scored() -> None:
    case = HarnessCase(
        case_id="trace",
        turns=["杭州两天历史游"],
        required_tools=["search_poi"],
        forbidden_tools=["search_hotel"],
        tool_argument_assertions=[
            {"tool": "search_poi", "argument": "city", "op": "eq", "value": "杭州"}
        ],
        expected_memory=[
            {"scope": "stable_profile", "field": "interests", "op": "contains", "value": "history"}
        ],
    )
    result = HarnessCaseResult(
        case_id="trace",
        turns=[
            HarnessTurnResult(
                user_message="杭州两天历史游",
                reply_text="done",
                tool_trace=["search_poi"],
                used_real_agent=True,
                clarification=False,
                profile={},
                artifacts={},
                tool_calls=[{"name": "search_poi", "arguments": {"city": "杭州"}, "status": "ok"}],
                memory_snapshot={"stable_profile": {"interests": ["history"]}},
            )
        ],
        final_profile={},
        final_artifacts={},
    )

    metrics = validate_case_result(case, result)

    assert metrics["tool_coverage_ok"] is True
    assert metrics["forbidden_tools_ok"] is True
    assert metrics["tool_arguments_ok"] is True
    assert metrics["memory_ok"] is True
    assert metrics["llm_judge"] == {
        "status": "not_run",
        "rubric": None,
        "reason": "missing_expected_artifact",
    }


def test_langchain_tool_trace_redacts_sensitive_arguments(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.evaluation_trace_enabled = True
    tool = next(item for item in build_tools(ctx, offline_settings) if item.name == "update_travel_profile")

    tool.invoke({"destination": "杭州", "days": 2})

    record = ctx.evaluation_trace[-1]
    assert record["name"] == "update_travel_profile"
    assert record["arguments"] == {"destination": "杭州", "days": 2}
    assert record["status"] == "ok"


def test_langchain_tool_trace_records_structured_error_code(
    offline_settings, monkeypatch
) -> None:
    from travel_agent.agent import toolkit

    ctx = build_session(persist=False)
    ctx.evaluation_trace_enabled = True
    monkeypatch.setattr(
        toolkit,
        "search_poi",
        lambda *_args, **_kwargs: {
            "isError": True,
            "error_code": "NOT_FOUND",
            "summary": "没有匹配结果",
        },
    )
    tool = next(item for item in build_tools(ctx, offline_settings) if item.name == "search_poi")

    tool.invoke({"city": "测试城", "max_results": 10})

    record = ctx.evaluation_trace[-1]
    assert record["status"] == "error"
    assert record["error_code"] == "NOT_FOUND"


def test_restaurant_tool_normalizes_numeric_per_capita_budget(
    offline_settings, monkeypatch
) -> None:
    from travel_agent.agent import toolkit

    captured = {}

    def fake_search(_ctx, city, cuisine, area, budget_level, max_results):
        captured.update({
            "city": city,
            "area": area,
            "budget_level": budget_level,
            "max_results": max_results,
        })
        return {"isError": False, "restaurants": []}

    monkeypatch.setattr(toolkit, "search_restaurant", fake_search)
    ctx = build_session(persist=False)
    tool = next(item for item in build_tools(ctx, offline_settings) if item.name == "search_restaurant")

    tool.invoke({
        "city": "测试城",
        "area": "江畔商圈",
        "budget_level": "人均100元左右",
        "max_results": 10,
    })

    assert captured["budget_level"] == "mid"


def test_plan_tool_ends_react_loop_after_structured_itinerary(offline_settings) -> None:
    ctx = build_session(persist=False)
    tools = {tool.name: tool for tool in build_tools(ctx, offline_settings)}

    assert tools["plan_and_critique"].return_direct is True
    assert tools["search_poi"].return_direct is False


def test_fault_provider_is_deterministic() -> None:
    ctx = build_session(persist=False)
    empty = FaultInjectingProvider(
        ctx.provider,
        {"operation": "search_pois", "mode": "empty"},
    )
    timeout = FaultInjectingProvider(
        ctx.provider,
        {"operation": "get_weather", "mode": "timeout"},
    )

    assert empty.search_pois("杭州") == []
    with pytest.raises(TimeoutError, match="injected"):
        timeout.get_weather("杭州")


def test_failure_attribution_separates_model_and_tool_failures() -> None:
    case = HarnessCase(
        case_id="failure",
        turns=["杭州两天"],
        expect_itinerary=True,
        required_tools=["search_poi"],
        failure_injection={"operation": "search_pois", "mode": "empty"},
    )
    result = HarnessCaseResult(
        case_id="failure",
        turns=[
            HarnessTurnResult(
                user_message="杭州两天",
                reply_text="查询为空，建议稍后重试",
                tool_trace=[],
                used_real_agent=True,
                clarification=False,
                profile={},
                artifacts={},
            )
        ],
        final_profile={},
        final_artifacts={},
        metrics={"tool_coverage_ok": False, "fault_recovery_ok": True},
    )

    assert attribute_failures(case, result) == ["model_tool_selection", "model_intent"]


def test_fine_tuning_gate_requires_remediation_and_systematic_frozen_failures() -> None:
    rows = [
        {
            "case_id": f"case-{index}",
            "split": "core_frozen" if index < 94 else "challenge_frozen",
            "failure_attributions": ["model_tool_arguments"] if index < 18 else [],
        }
        for index in range(128)
    ]

    before = decide_fine_tuning(rows, remediation_complete=False)
    after = decide_fine_tuning(rows, remediation_complete=True)

    assert before["decision"] == "prompt_or_workflow_first"
    assert after["decision"] == "recommend_sft"
    assert after["triggered_categories"] == {"model_tool_arguments": 18}


def test_fine_tuning_gate_requires_full_frozen_run() -> None:
    rows = [
        {"case_id": f"case-{index}", "split": "core_frozen", "failure_attributions": []}
        for index in range(90)
    ]

    decision = decide_fine_tuning(rows, remediation_complete=True)

    assert decision["decision"] == "insufficient_frozen_evidence"


def test_wilson_interval_reports_counts_and_bounds() -> None:
    interval = wilson_interval(50, 100)

    assert interval is not None
    assert interval["successes"] == 50
    assert interval["total"] == 100
    assert interval["low"] == pytest.approx(0.4038, abs=0.001)
    assert interval["high"] == pytest.approx(0.5962, abs=0.001)


def test_artifact_first_rates_have_applicable_confidence_intervals() -> None:
    base = {
        "repeat": 1,
        "split": "dev",
        "passed": True,
        "tool_trace": [],
        "strict_task_success": True,
        "artifact_type_match": True,
        "failure_reason": None,
        "gating_passed": True,
        "grounding_ok": True,
        "authorization_ok": True,
        "architecture_policy_ok": True,
        "expected_outcome_match": True,
        "task_completed": True,
        "hard_constraints_ok": True,
        "soft_preferences_ok": True,
        "critic_passed": True,
        "clarification_ok": True,
        "tool_schema_valid": True,
        "fault_recovery_ok": True,
        "memory_ok": True,
        "turn_state_ok": True,
        "turn_state_accuracy": 1.0,
        "tool_required_count": 0,
        "tool_required_hit_count": 0,
        "tool_selected_count": 0,
        "tool_unexpected_count": 0,
        "tool_argument_assertion_count": 0,
        "tool_argument_pass_count": 0,
        "duration_ms": 1.0,
        "total_tokens": 1,
        "estimated_cost_usd": 0.0,
        "model_call_count": 1,
        "tool_call_count": 0,
        "dispatch_count": 0,
        "failure_attributions": [],
        "planner_required": 0,
        "planner_admitted": 0,
        "planner_budget_exhausted": 0,
        "dispatch_total": 0,
        "duplicate_dispatches": 0,
        "artifact_reuse_total": 0,
        "artifact_reused": 0,
        "stale_artifact_reused": 0,
        "execution_total": 1,
        "system_error": 0,
    }
    rows = [
        {
            **base,
            "case_id": "full",
            "expected_artifact_type": "full_itinerary",
            "planner_required": 1,
            "planner_admitted": 1,
            "dispatch_total": 3,
            "artifact_reuse_total": 2,
            "artifact_reused": 1,
        },
        {
            **base,
            "case_id": "route",
            "expected_artifact_type": "route_plan",
            "dispatch_total": 1,
        },
    ]

    metrics = aggregate_product_rows(rows, remediation_complete=False)

    intervals = metrics["confidence_intervals_95"]
    assert intervals["overall_strict_success_rate"]["total"] == 2
    assert intervals["artifact_type_match_rate"]["total"] == 2
    assert intervals["full_itinerary_success_rate"]["total"] == 1
    assert intervals["non_itinerary_success_rate"]["total"] == 1
    assert intervals["expected_itinerary_missing_rate"]["successes"] == 0
    assert metrics["planner_admission_rate"] == 1.0
    assert metrics["planner_budget_exhaustion_rate"] == 0.0
    assert metrics["duplicate_dispatch_rate"] == 0.0
    assert metrics["artifact_reuse_rate"] == 0.5
    assert metrics["system_error_rate"] == 0.0


def test_chinatravel_preflight_and_provider_fail_fast(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="preflight failed"):
        preflight_chinatravel(tmp_path)
    with pytest.raises(RuntimeError, match="official database missing"):
        build_chinatravel_provider(tmp_path, allow_synthetic=False)


def test_live_snapshot_redaction_is_recursive() -> None:
    payload = {"key": "secret", "nested": [{"token": "secret", "city": "杭州"}]}

    assert _redact_snapshot(payload) == {
        "key": "[REDACTED]",
        "nested": [{"token": "[REDACTED]", "city": "杭州"}],
    }


def test_model_callback_records_duration_and_usage_without_prompts() -> None:
    trace = []
    callback = EvaluationTraceCallback(trace, model="qwen3.7-plus", phase="react")
    run_id = uuid4()
    callback.on_chat_model_start({}, [[]], run_id=run_id)
    response = SimpleNamespace(
        llm_output={
            "model_name": "qwen3.7-plus",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        },
        generations=[],
    )

    callback.on_llm_end(response, run_id=run_id)

    assert trace == [
        {
            "kind": "model",
            "phase": "react",
                "model": "qwen3.7-plus",
                "input_tokens": 12,
                "estimated_input_tokens": 1,
                "input_shape": {
                    "message_chars": 0,
                    "tool_schema_chars": 2,
                    "message_count": 0,
                },
                "output_tokens": 4,
            "total_tokens": 16,
            "visible_output_tokens": 4,
            "reasoning_tokens": None,
            "provider_completion_tokens": 4,
            "finish_reason": None,
            "truncated": False,
            "duration_ms": trace[0]["duration_ms"],
            "status": "ok",
            "error": None,
        }
    ]
    assert trace[0]["duration_ms"] is not None


def test_product_row_counts_completion_when_fallback_outputs_itinerary() -> None:
    case = HarnessCase(case_id="fallback", turns=["杭州两天"], expect_itinerary=True)
    result = HarnessCaseResult(
        case_id="fallback",
        turns=[
            HarnessTurnResult(
                user_message="杭州两天",
                reply_text="（ReAct agent 调用失败：connection failed，已降级到离线兜底）\n\nfallback plan",
                tool_trace=["plan_and_critique"],
                used_real_agent=False,
                clarification=False,
                profile={},
                artifacts={"itinerary": {"summary": "fallback"}},
                model_calls=[
                    {
                        "kind": "model",
                        "status": "error",
                        "error": "HTTPError: connection failed",
                    }
                ],
            )
        ],
        final_profile={},
        final_artifacts={"itinerary": {"summary": "fallback"}},
        metrics={},
    )

    row = _product_row(case, result, 1)

    assert row["passed"] is False
    assert row["task_completed"] is True
    assert row["task_completed_via_fallback"] is True
    assert row["react_attempted"] is True
    assert row["used_real_agent"] is False
    assert row["delivery_mode"] == "fallback"
    assert row["model_error_call_count"] == 1


def test_product_run_persists_inspectable_itinerary_per_case(tmp_path) -> None:
    case = HarnessCase(
        case_id="persisted-plan",
        turns=["杭州两天，喜欢历史"],
        split="dev",
        category="explicit_interests",
    )
    itinerary = {
        "summary": "杭州历史两日游",
        "critic": {"passed": True},
        "itinerary": {"city": "杭州", "days": []},
        "api_key": "must-not-leak",
    }
    result = HarnessCaseResult(
        case_id=case.case_id,
        turns=[
            HarnessTurnResult(
                user_message=case.turns[0],
                reply_text="第一天参观博物馆。",
                tool_trace=["plan_and_critique"],
                used_real_agent=True,
                clarification=False,
                profile={"destination": "杭州", "days": 2},
                artifacts={"itinerary": itinerary},
            )
        ],
        final_profile={"destination": "杭州", "days": 2},
        final_artifacts={"itinerary": itinerary},
        passed=True,
        metrics={
            "critic_passed": True,
            "llm_judge": {
                "status": "not_run",
                "rubric": "full_itinerary",
                "reason": "judge_not_attached",
            },
        },
    )
    output = _product_case_output(case, result, 1)
    suite = HarnessSuiteResult(
        suite="agent-product",
        mode="product",
        environment="real_agent",
        case_count=1,
        attempted_count=1,
        metrics={"sample_size": 1},
        rows=[{"case_id": case.case_id, "repeat": 1, "passed": True}],
        artifacts={"model": "test-model", "_case_outputs": [output]},
    )

    summary = write_product_run(suite, tmp_path, run_id="test-run")

    case_path = tmp_path / "runs" / "test-run" / "cases" / "persisted-plan__repeat-1.json"
    saved_case = json.loads(case_path.read_text(encoding="utf-8"))
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_case["turns"][0]["reply_text"] == "第一天参观博物馆。"
    assert saved_case["final_itinerary"]["critic"]["passed"] is True
    assert saved_case["final_itinerary"]["api_key"] == "[REDACTED]"
    assert saved_case["evaluation"]["independent_judge"] is None
    assert saved_case["evaluation"]["rule_metrics"]["llm_judge"] == {
        "status": "not_run",
        "rubric": "full_itinerary",
        "reason": "judge_not_attached",
    }
    assert summary["artifacts"]["case_output_count"] == 1
    assert saved_summary["rows"][0]["case_output_file"] == str(case_path)
    assert "_case_outputs" not in saved_summary["artifacts"]
