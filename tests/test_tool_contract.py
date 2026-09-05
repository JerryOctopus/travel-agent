from __future__ import annotations

import json
import threading
import time

import pytest

from travel_agent.agent import toolkit
from travel_agent.agent.lc_tools import build_tools
from travel_agent.agent.session import (
    RequestControl,
    RequestCancelledError,
    build_session,
    reset_task_meta,
    set_current_task_meta,
)
from travel_agent.agent.tool_contract import (
    ToolExecutionPolicy,
    contracted_tool,
    normalize_tool_arguments,
    tool_execution_policy,
)


def test_invalid_city_is_rejected_before_provider(monkeypatch) -> None:
    ctx = build_session(session_id="contract-city", persist=False)
    called = False

    def should_not_run(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(ctx.provider, "search_pois", should_not_run)
    result = toolkit.search_hotel(ctx, city="")
    assert result == {
        "isError": True,
        "summary": "city 不能为空。",
        "error_code": "INVALID_INPUT",
        "retryable": False,
        "details": {"field": "city"},
    }
    assert called is False


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"days": 0}, "days"),
        ({"companions": -1}, "companions"),
        ({"budget_level": "luxury"}, "budget_level"),
    ],
)
def test_budget_semantics_are_validated(kwargs, field) -> None:
    ctx = build_session(session_id=f"contract-budget-{field}", persist=False)
    ctx.profile.destination = "杭州"
    result = toolkit.estimate_budget(ctx, **kwargs)
    assert result["isError"] is True
    assert result["error_code"] == "INVALID_INPUT"
    assert result["retryable"] is False
    assert result["details"]["field"] == field


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("经济", "low"),
        ("中档", "mid"),
        ("中高端", "high"),
    ],
)
def test_model_facing_budget_aliases_are_normalized(value, expected) -> None:
    assert normalize_tool_arguments(
        "estimate_budget", {"city": "杭州", "budget_level": value}
    )["budget_level"] == expected


def test_model_facing_poi_category_moves_named_taxonomy_into_search_terms() -> None:
    assert normalize_tool_arguments(
        "search_poi",
        {"city": "北京", "interests": ["历史"], "category": "故宫"},
    ) == {
        "city": "北京",
        "interests": ["历史", "故宫"],
        "category": None,
    }
    assert normalize_tool_arguments(
        "search_poi", {"city": "苏州", "category": "景点"}
    )["category"] == "scenic"


def test_route_rejects_equal_or_unknown_session_pois() -> None:
    ctx = build_session(session_id="contract-route", persist=False)
    equal = toolkit.plan_route(ctx, "poi-1", "poi-1")
    assert equal["error_code"] == "INVALID_INPUT"
    assert equal["retryable"] is False

    missing = toolkit.plan_route(ctx, "poi-1", "poi-2")
    assert missing["error_code"] == "NOT_FOUND"
    assert missing["retryable"] is True
    assert missing["details"]["missing_poi_ids"] == ["poi-1", "poi-2"]


def test_provider_exception_is_standardized(monkeypatch) -> None:
    ctx = build_session(session_id="contract-provider", persist=False)

    def fail(**_kwargs):
        raise ConnectionError("provider offline")

    monkeypatch.setattr(ctx.provider, "search_pois", fail)
    result = toolkit.search_hotel(ctx, city="杭州")
    assert result["isError"] is True
    assert result["error_code"] == "UPSTREAM_UNAVAILABLE"
    assert result["retryable"] is True


def test_rate_limit_exception_has_stable_retryable_code() -> None:
    ctx = build_session(session_id="contract-rate-limit", persist=False)

    @contracted_tool("search_hotel")
    def limited(_ctx, city=None):
        raise RuntimeError("429 rate limit")

    result = limited(ctx, city="杭州")
    assert result["error_code"] == "RATE_LIMITED"
    assert result["retryable"] is True


def test_success_envelope_remains_backward_compatible() -> None:
    ctx = build_session(session_id="contract-success", persist=False)
    ctx.profile.destination = "杭州"
    result = toolkit.estimate_budget(ctx, days=2)
    assert result["isError"] is False
    assert result["summary"]
    assert "error_code" not in result
    assert ctx.store.get_record(result["artifact_id"])["kind"] == "budget"


def test_planner_requires_explicit_existing_artifacts() -> None:
    ctx = build_session(session_id="contract-planner", persist=False)
    token = set_current_task_meta(
        {"agent": "planner", "request_id": "req", "task_id": "planner-1"}
    )
    try:
        absent = toolkit.recommend_candidates(ctx, artifact_ids=None)
        assert absent["error_code"] == "PERMISSION_DENIED"
        assert absent["retryable"] is False
    finally:
        reset_task_meta(token)

    token = set_current_task_meta(
        {
            "agent": "planner",
            "request_id": "req",
            "task_id": "planner-1",
            "artifact_ids": ["artifact-missing"],
        }
    )
    try:
        missing = toolkit.recommend_candidates(ctx)
        assert missing["error_code"] == "PERMISSION_DENIED"
        assert missing["details"]["missing_artifact_ids"] == ["artifact-missing"]
    finally:
        reset_task_meta(token)


def test_planner_accepts_artifacts_bound_in_task_context() -> None:
    ctx = build_session(session_id="contract-planner-bound", persist=False)
    artifact_id = ctx.store.put("candidates", {"pois": []})
    token = set_current_task_meta(
        {
            "agent": "planner",
            "request_id": "req",
            "task_id": "planner-1",
            "artifact_ids": [artifact_id],
        }
    )
    try:
        result = toolkit.recommend_candidates(ctx)
        assert result["error_code"] == "NOT_FOUND"
        assert result["error_code"] != "PERMISSION_DENIED"
    finally:
        reset_task_meta(token)


def test_planner_rejects_existing_but_unbound_artifact() -> None:
    ctx = build_session(session_id="contract-planner-unbound", persist=False)
    bound_id = ctx.store.put("candidates", {"pois": []})
    unbound_id = ctx.store.put("candidates", {"pois": []})
    token = set_current_task_meta(
        {
            "agent": "planner",
            "request_id": "req",
            "task_id": "planner-1",
            "artifact_ids": [bound_id],
        }
    )
    try:
        result = toolkit.recommend_candidates(ctx, artifact_ids=[unbound_id])
        assert result["error_code"] == "PERMISSION_DENIED"
        assert result["details"]["unbound_artifact_ids"] == [unbound_id]
    finally:
        reset_task_meta(token)


def test_success_without_owned_artifact_is_contract_violation() -> None:
    ctx = build_session(session_id="contract-artifact", persist=False)

    @contracted_tool("search_poi")
    def broken(_ctx, city=None):
        return {"isError": False, "summary": "pretend success"}

    result = broken(ctx, city="杭州")
    assert result["error_code"] == "CONTRACT_VIOLATION"
    assert result["retryable"] is False


def test_foreign_artifact_cannot_satisfy_tool_success() -> None:
    ctx = build_session(session_id="contract-foreign-artifact", persist=False)

    @contracted_tool("search_poi")
    def broken(local_ctx, city=None):
        artifact_id = local_ctx.store.put(
            "candidates",
            {"pois": []},
            request_id="foreign-request",
            task_id="foreign-task",
            agent="attraction",
        )
        return {"isError": False, "summary": "pretend success", "artifact_id": artifact_id}

    token = set_current_task_meta(
        {"agent": "attraction", "request_id": "request", "task_id": "task"}
    )
    try:
        result = broken(ctx, city="杭州")
        assert result["error_code"] == "CONTRACT_VIOLATION"
        assert result["details"]["field"] == "request_id"
    finally:
        reset_task_meta(token)


def test_local_and_mcp_paths_share_error_contract(offline_settings) -> None:
    from travel_agent.mcp_server import _call

    local_ctx = build_session(session_id="contract-local", persist=False)
    local_tool = next(tool for tool in build_tools(local_ctx, offline_settings) if tool.name == "search_poi")
    local = json.loads(local_tool.invoke({"city": ""}))
    remote = _call("contract-remote", None, toolkit.search_poi, "")
    for key in ("isError", "summary", "error_code", "retryable", "details"):
        assert remote[key] == local[key]


def test_request_cancellation_is_not_normalized() -> None:
    ctx = build_session(session_id="contract-cancel", persist=False)
    ctx.request_control = RequestControl("request-contract-cancel")
    ctx.request_control.cancel()
    with pytest.raises(RequestCancelledError):
        toolkit.check_weather(ctx, "杭州")


def test_execution_policy_is_fail_closed_and_public() -> None:
    assert tool_execution_policy("search_poi") == ToolExecutionPolicy(
        parallel_safe=True,
        mutates_session=True,
        artifact_kinds=("candidates",),
    )
    assert tool_execution_policy("unknown") == ToolExecutionPolicy()


def test_parallel_safe_tools_overlap_within_session() -> None:
    ctx = build_session(session_id="contract-parallel", persist=False)
    active = 0
    peak = 0
    guard = threading.Lock()

    @contracted_tool("search_poi")
    def slow(local_ctx, city=None):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.08)
        artifact_id = local_ctx.store.put("candidates", {"pois": []})
        with guard:
            active -= 1
        return {"isError": False, "summary": "ok", "artifact_id": artifact_id}

    threads = [threading.Thread(target=slow, args=(ctx, city)) for city in ("杭州", "上海")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
    assert peak == 2


def test_serial_tools_do_not_overlap_in_same_session() -> None:
    ctx = build_session(session_id="contract-serial", persist=False)
    active = 0
    peak = 0
    guard = threading.Lock()

    @contracted_tool("build_constraints")
    def slow(local_ctx):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        artifact_id = local_ctx.store.put("constraints", {})
        with guard:
            active -= 1
        return {"isError": False, "summary": "ok", "artifact_id": artifact_id}

    threads = [threading.Thread(target=slow, args=(ctx,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
    assert peak == 1


def test_serial_tools_from_different_sessions_do_not_block_each_other() -> None:
    contexts = [
        build_session(session_id="contract-serial-a", persist=False),
        build_session(session_id="contract-serial-b", persist=False),
    ]
    barrier = threading.Barrier(2)

    @contracted_tool("build_constraints")
    def synchronized(local_ctx):
        barrier.wait(timeout=1)
        artifact_id = local_ctx.store.put("constraints", {})
        return {"isError": False, "summary": "ok", "artifact_id": artifact_id}

    threads = [threading.Thread(target=synchronized, args=(ctx,)) for ctx in contexts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()
