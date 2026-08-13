from __future__ import annotations

import json
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
    attribute_failures,
    decide_fine_tuning,
    validate_product_dataset,
    wilson_interval,
    _execution_case,
    _product_case_output,
    _product_row,
    write_product_run,
)
from travel_agent.harness.validators import validate_case_result


def test_production_v1_dataset_has_192_cases_with_frozen_split_discipline() -> None:
    cases = load_cases_json(DEFAULT_PRODUCT_CASES)

    validation = validate_product_dataset(cases)

    assert validation.valid is True, validation.errors
    assert validation.split_counts == {
        "dev": 34,
        "core_frozen": 94,
        "challenge_frozen": 34,
        "shadow_frozen": 30,
    }
    assert len(validation.subset_counts) >= 15
    assert all(case.gold_outcome for case in cases)


def test_production_dataset_loader_parses_external_schema() -> None:
    cases = load_cases_json(DEFAULT_PRODUCT_CASES)
    case = next(item for item in cases if item.split == "dev")

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
    with pytest.raises(SystemExit, match="影子集"):
        ablation.main()


def test_single_variant_is_allowed_on_shadow_split(monkeypatch, offline_settings) -> None:
    import scripts.eval_ablation as ablation

    monkeypatch.setattr(
        "sys.argv",
        ["eval_ablation.py", "--variants", "v3", "--product-split", "shadow_frozen"],
    )
    monkeypatch.setattr(ablation, "load_settings", lambda: offline_settings)
    # 单 variant 不触发影子禁令；离线 rule provider 会拦在 LLM 检查。
    with pytest.raises(RuntimeError, match="真实 LLM"):
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


def test_langchain_tool_trace_redacts_sensitive_arguments(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.evaluation_trace_enabled = True
    tool = next(item for item in build_tools(ctx, offline_settings) if item.name == "update_travel_profile")

    tool.invoke({"destination": "杭州", "days": 2})

    record = ctx.evaluation_trace[-1]
    assert record["name"] == "update_travel_profile"
    assert record["arguments"] == {"destination": "杭州", "days": 2}
    assert record["status"] == "ok"


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
        metrics={"critic_passed": True},
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
    assert summary["artifacts"]["case_output_count"] == 1
    assert saved_summary["rows"][0]["case_output_file"] == str(case_path)
    assert "_case_outputs" not in saved_summary["artifacts"]
