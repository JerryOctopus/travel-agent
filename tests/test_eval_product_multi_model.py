from __future__ import annotations

import json
from dataclasses import replace

import pytest

from travel_agent.harness.result import HarnessCaseResult, HarnessTurnResult

from scripts.eval_product_multi_model import (
    DEFAULT_RELAY_MODELS,
    ALL_AGENT_ROLES,
    acquire_run_lock,
    build_state_template,
    build_parser,
    multi_agent_fields,
    is_quota_error,
    load_pending_case_outputs,
    parse_models,
    pick_api_key,
    provider_tool_quota_error,
    provider_reported_models,
    run_tool_preflight,
    relay_mode_enabled,
    recorded_model_identity,
    save_pending_case_output,
    save_state,
)


def test_tool_preflight_reports_failure_without_any_model_probe(monkeypatch) -> None:
    calls: list[str] = []
    settings = type("Settings", (), {"amap": object()})()

    def fail_amap(_settings):
        calls.append("amap")
        return {
            "ok": False,
            "checks": {"place_search": {"ok": False, "infocode": "10044"}},
        }

    monkeypatch.setattr(
        "scripts.eval_product_multi_model.preflight_amap", fail_amap
    )

    result = run_tool_preflight(settings, "configured")

    assert calls == ["amap"]
    assert result["ok"] is False


def test_run_lock_rejects_concurrent_writer(tmp_path) -> None:
    first = acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            acquire_run_lock(tmp_path)
    finally:
        first.close()

    second = acquire_run_lock(tmp_path)
    second.close()


def test_default_relay_models_keep_requested_order() -> None:
    assert parse_models(DEFAULT_RELAY_MODELS, "qwen") == [
        ("qwen", "qwen-max"),
        ("qwen", "qwen3.6-flash"),
        ("qwen", "qwen3.7-flash-2026-07-15"),
        ("qwen", "qwen-long"),
        ("qwen", "qwen3.5-flash-2026-02-23"),
        ("qwen", "qwen-plus"),
        ("glm", "glm-5"),
    ]

    assert parse_models(["deepseek:deepseek-v4-pro"], "qwen") == [
        ("deepseek", "deepseek-v4-pro")
    ]


def test_provider_credentials_are_not_reused_across_providers(monkeypatch) -> None:
    monkeypatch.delenv("TRAVEL_AGENT_DEEPSEEK_API_KEY", raising=False)
    assert pick_api_key("qwen", None, "qwen", "qwen-secret") == "qwen-secret"
    assert pick_api_key("deepseek", None, "qwen", "qwen-secret") is None

    monkeypatch.setenv("TRAVEL_AGENT_DEEPSEEK_API_KEY", "deepseek-secret")
    assert (
        pick_api_key("deepseek", None, "qwen", "qwen-secret")
        == "deepseek-secret"
    )


def test_context_limit_is_not_misclassified_as_account_quota() -> None:
    assert is_quota_error("AllocationQuota.FreeTierOnly") is True
    assert is_quota_error("insufficient balance") is True
    assert is_quota_error("maximum context length exceeded") is False


def test_state_uses_provider_specific_base_urls() -> None:
    state = build_state_template(
        "run-1",
        [("qwen", "qwen-test"), ("deepseek", "deepseek-test")],
        None,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    assert state["models"][0]["base_url"].startswith("https://dashscope")
    assert state["models"][1]["base_url"] == "https://api.deepseek.com/v1"


def test_single_model_run_records_actual_identity() -> None:
    models = [{"provider": "siliconflow", "model": "deepseek-ai/DeepSeek-V4-Flash"}]

    assert recorded_model_identity(models) == (
        "siliconflow",
        "deepseek-ai/DeepSeek-V4-Flash",
    )
    assert recorded_model_identity([*models, {"provider": "qwen", "model": "qwen-max"}]) == (
        "quota-relay",
        "quota-relay",
    )


def test_single_model_execution_is_not_reported_as_model_relay() -> None:
    assert relay_mode_enabled([
        {"provider": "deepseek", "model": "deepseek-v4-flash"}
    ]) is False
    assert relay_mode_enabled([
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "qwen", "model": "qwen-max"},
    ]) is True


def test_provider_reported_model_identity_is_recorded_separately() -> None:
    result = _result(["attraction", "transport", "planner"])
    result.turns[0].model_calls.extend([
        {"model": "provider-resolved-a"},
        {"model": "provider-resolved-a"},
        {"model": "provider-resolved-b"},
    ])

    assert provider_reported_models(result) == [
        "provider-resolved-a", "provider-resolved-b",
    ]


def test_provider_tool_quota_error_detects_amap_abort() -> None:
    result = _result(["attraction", "transport", "planner"])
    result = replace(
        result,
        turns=[
            replace(
                result.turns[0],
                error=(
                    "ProviderRateLimitError: AMap rate limit: "
                    "USER_DAILY_QUERY_OVER_LIMIT (10044)"
                ),
            )
        ],
    )

    assert "10044" in provider_tool_quota_error(result)


def test_provider_tool_quota_error_detects_wrapped_incomplete_reply() -> None:
    result = _result(["hotel", "transport"])
    result = replace(
        result,
        turns=[
            replace(
                result.turns[0],
                reply_text=(
                    "当前方案尚不可交付。未解决：ProviderRateLimitError: "
                    "AMap rate limit: USER_DAILY_QUERY_OVER_LIMIT (10044)。"
                ),
            )
        ],
    )

    assert "10044" in provider_tool_quota_error(result)


def test_provider_tool_quota_error_ignores_model_quota() -> None:
    result = _result(["attraction", "transport", "planner"])
    result.turns[0].model_calls.append(
        {"status": "error", "error": "insufficient balance"}
    )

    assert provider_tool_quota_error(result) is None


def test_local_model_token_cap_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["eval_product_multi_model.py"])

    args = build_parser().parse_args()

    assert args.max_model_tokens == 0


def test_resume_reloads_atomically_persisted_case_outputs(tmp_path) -> None:
    pending_dir = tmp_path / "pending_cases"
    first = {
        "case": {"case_id": "dev_002"},
        "execution": {"repeat": 1},
        "turns": [{"reply_text": "second"}],
    }
    second = {
        "case": {"case_id": "dev_001"},
        "execution": {"repeat": 1},
        "turns": [{"reply_text": "first"}],
    }

    save_pending_case_output(pending_dir, first)
    save_pending_case_output(pending_dir, second)

    restored = load_pending_case_outputs(pending_dir)

    assert [item["case"]["case_id"] for item in restored] == ["dev_001", "dev_002"]
    assert not list(pending_dir.glob("*.tmp"))


def test_relay_state_write_is_atomic(tmp_path) -> None:
    state_path = tmp_path / "relay_state.json"

    save_state(state_path, {"completed_case_ids": ["dev_001"]})

    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "completed_case_ids": ["dev_001"]
    }
    assert not state_path.with_suffix(".json.tmp").exists()


def _result(successful_agents: list[str], *, real: bool = True) -> HarnessCaseResult:
    agent_items = [
        {
            "request_id": "req-1",
            "task_id": f"task_{agent}",
            "agent": agent,
            "kind": "subagent",
            "status": "completed" if agent in successful_agents else "failed",
            "detail": {"tool_trace": []},
        }
        for agent in ALL_AGENT_ROLES
    ]
    turn = HarnessTurnResult(
        user_message="杭州三天",
        reply_text="done",
        tool_trace=[],
        used_real_agent=real,
        clarification=False,
        profile={},
        artifacts={},
    )
    return HarnessCaseResult(
        case_id="case-1",
        turns=[turn],
        final_profile={},
        final_artifacts={
            "itinerary": {"days": []},
            "agent_trace": {"items": agent_items},
        },
        metrics={
            "expected_artifact_type": "full_itinerary",
            "artifact_type_match": True,
        },
    )


def test_multi_agent_strict_success_requires_artifact_scoped_real_agents() -> None:
    # A plain full itinerary requires attraction, transport and planner; the
    # optional hotel/restaurant roles are not fabricated requirements.
    required = ["attraction", "transport", "planner"]
    complete = multi_agent_fields(_result(required))
    assert complete["multi_agent_strict_success"] is True
    assert complete["multi_agent_missing_agents"] == []

    partial = multi_agent_fields(_result(["attraction", "transport"]))
    assert partial["multi_agent_strict_success"] is False
    assert partial["multi_agent_missing_agents"] == ["planner"]

    fallback = multi_agent_fields(_result(required, real=False))
    assert fallback["multi_agent_strict_success"] is False
