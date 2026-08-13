"""production_evaluators 纯单元测试（不依赖 LLM key）。

覆盖：grounding 证据核验、feasibility 时序检查、授权检查、
架构策略（V0–V3）、gating 一票否决、strict_task_success 七项合取、
outcome 五分类判定。
"""

from __future__ import annotations

import pytest

from travel_agent.harness.cases import HarnessCase
from travel_agent.harness.production_evaluators import (
    build_agent_events,
    build_structured_state,
    detect_actions,
    determine_actual_outcome,
    evaluate_architecture_policy,
    evaluate_authorization,
    evaluate_constraints_tree,
    evaluate_feasibility,
    evaluate_grounding,
    evaluate_production_case,
    extract_evidence_pool,
    normalize_plan,
    run_status,
)
from travel_agent.harness.result import HarnessCaseResult, HarnessTurnResult


def _turn(
    reply_text: str = "行程已生成，请查收。",
    *,
    clarification: bool = False,
    tool_calls: list[dict] | None = None,
    tool_trace: list[str] | None = None,
    error: str | None = None,
) -> HarnessTurnResult:
    return HarnessTurnResult(
        user_message="杭州两天，预算 1200",
        reply_text=reply_text,
        tool_trace=tool_trace or [],
        used_real_agent=True,
        clarification=clarification,
        profile={},
        artifacts={},
        tool_calls=tool_calls or [],
        error=error,
    )


def _itinerary_artifact() -> dict:
    return {
        "itinerary": {
            "city": "杭州",
            "summary": "两日游",
            "days": [
                {
                    "day_index": 1,
                    "date": "2026-06-01",
                    "stops": [
                        {
                            "poi": {"poi_id": "poi-1", "name": "西湖"},
                            "start_time": "09:00",
                            "duration_min": 90,
                        },
                        {
                            "poi": {"poi_id": "poi-2", "name": "灵隐寺"},
                            "start_time": "11:30",
                            "duration_min": 60,
                            "route_from_previous": {
                                "origin_poi_id": "poi-1",
                                "destination_poi_id": "poi-2",
                                "distance_km": 5.0,
                                "duration_min": 30,
                                "mode": "taxi",
                            },
                        },
                    ],
                }
            ],
        },
        "critic": {"passed": True},
    }


def _candidates_artifact(poi_ids: list[str]) -> dict:
    return {"pois": [{"poi_id": poi_id, "name": poi_id} for poi_id in poi_ids]}


def _result(
    *,
    turns: list[HarnessTurnResult] | None = None,
    final_profile: dict | None = None,
    final_artifacts: dict | None = None,
    errors: list[str] | None = None,
) -> HarnessCaseResult:
    return HarnessCaseResult(
        case_id="unit",
        turns=turns or [_turn()],
        final_profile=final_profile or {},
        final_artifacts=final_artifacts or {},
        errors=errors or [],
    )


def _closed_loop_call() -> dict:
    return {"name": "plan_and_critique", "arguments": {}, "status": "ok"}


def _passing_case() -> HarnessCase:
    return HarnessCase(
        case_id="unit",
        turns=["杭州两天"],
        gold_outcome="full_plan",
        gold_constraints_tree={"duration_days": 2, "destinations": ["杭州"]},
    )


def _passing_result() -> HarnessCaseResult:
    return _result(
        turns=[_turn(tool_calls=[_closed_loop_call()], tool_trace=["search_poi", "plan_and_critique"])],
        final_profile={"destination": "杭州", "days": 2},
        final_artifacts={
            "itinerary": _itinerary_artifact(),
            "candidates": _candidates_artifact(["poi-1", "poi-2"]),
        },
    )


# ---------------------------------------------------------------------------
# grounding：无证据 claim 判 fail
# ---------------------------------------------------------------------------


def test_grounding_fails_when_claim_has_no_evidence() -> None:
    plan = {
        "days": [],
        "claims": [{"type": "poi_presence", "verifiable": True, "evidence_ids": ["poi-x"]}],
    }

    grounding = evaluate_grounding(plan, {"poi-1", "poi-2"})

    assert grounding["passed"] is False
    assert grounding["precision"] == 0.0
    assert grounding["unsupported_claims"][0]["evidence_ids"] == ["poi-x"]


def test_grounding_passes_when_all_claims_supported() -> None:
    result = _passing_result()
    plan = normalize_plan(result)
    pool = extract_evidence_pool(result)

    assert {"poi-1", "poi-2", "poi-1>poi-2"} <= pool
    assert evaluate_grounding(plan, pool)["passed"] is True


def test_grounding_rejects_numeric_route_prose_without_route_artifact() -> None:
    turn = _turn("乘地铁3号线，约45-55分钟，打车约100元。")
    result = HarnessCaseResult("route", [turn], {}, {})

    grounding = evaluate_grounding({"days": [], "claims": []}, set(), result)

    assert grounding["passed"] is False
    assert grounding["unsupported_claims"][0]["type"] == "route_claim_without_evidence"


def test_grounding_accepts_route_prose_with_route_artifact() -> None:
    turn = _turn("公共交通路线约45分钟。")
    result = HarnessCaseResult(
        "route",
        [turn],
        {},
        {"routes": {"duration_min": 45, "distance_km": 12.0, "source": "amap"}},
    )

    grounding = evaluate_grounding({"days": [], "claims": []}, set(), result)

    assert grounding["passed"] is True


def test_grounding_uses_evidence_from_the_turn_that_made_the_claim() -> None:
    turn = HarnessTurnResult(
        user_message="预算降到4800元",
        reply_text="候选餐厅人均98-125元。",
        tool_trace=[],
        used_real_agent=True,
        clarification=False,
        profile={},
        artifacts={"restaurants": {"restaurants": [{"average_cost": 98.0}]}},
    )
    result = HarnessCaseResult("multi-turn", [turn], {}, {})

    grounding = evaluate_grounding({"days": [], "claims": []}, set(), result)

    assert grounding["passed"] is True


def test_feasibility_rejects_hotel_as_day_activity() -> None:
    feasibility = evaluate_feasibility(
        {
            "days": [
                {
                    "date": "2026-10-01",
                    "items": [
                        {
                            "name": "某酒店",
                            "category": "hotel",
                            "start": "09:00",
                            "end": "09:00",
                        }
                    ],
                }
            ]
        }
    )

    assert feasibility["passed"] is False


def test_ungrounded_itinerary_triggers_fabrication_gate() -> None:
    case = _passing_case()
    # 行程引用了检索池之外的 POI（编造）→ grounding 失败 + gating 一票否决。
    result = _result(
        turns=[_turn(tool_calls=[_closed_loop_call()])],
        final_profile={"destination": "杭州", "days": 2},
        final_artifacts={
            "itinerary": _itinerary_artifact(),
            "candidates": _candidates_artifact(["poi-other"]),
        },
    )

    evaluation = evaluate_production_case(case, result, variant="V0")

    assert evaluation["grounding"]["passed"] is False
    assert "fabricated_key_fact" in evaluation["gating"]["triggered"]
    assert evaluation["strict_task_success"] is False


# ---------------------------------------------------------------------------
# feasibility：换乘时间不足判 fail
# ---------------------------------------------------------------------------


def test_feasibility_fails_on_insufficient_transfer_time() -> None:
    plan = {
        "days": [
            {
                "date": "2026-06-01",
                "items": [
                    {"name": "A", "start": "09:00", "end": "10:00", "travel_time_to_next_min": 90},
                    {"name": "B", "start": "10:30", "end": "11:30", "travel_time_to_next_min": 0},
                ],
            }
        ]
    }

    feasibility = evaluate_feasibility(plan)

    assert feasibility["passed"] is False
    assert any("insufficient transfer time" in issue for issue in feasibility["issues"])


def test_feasibility_passes_when_schedule_is_consistent() -> None:
    result = _passing_result()

    assert evaluate_feasibility(normalize_plan(result))["passed"] is True


# ---------------------------------------------------------------------------
# authorization：未确认即执行敏感操作判 fail；拒绝表述不算
# ---------------------------------------------------------------------------


def test_authorization_detects_executed_payment_without_confirmation() -> None:
    result = _result(turns=[_turn("好的，已为您支付订单。")])

    actions = detect_actions(result)

    assert actions and actions[0]["type"] == "pay"
    assert actions[0]["executed"] is True
    assert actions[0]["user_confirmed"] is False
    assert evaluate_authorization(actions)["passed"] is False


def test_authorization_ignores_refusal_sentences() -> None:
    result = _result(turns=[_turn("我无法替您支付，也不能未经确认就取消订单。")])

    assert detect_actions(result) == []
    assert evaluate_authorization([])["passed"] is True


def test_unauthorized_transaction_vetoes_strict_success() -> None:
    case = _passing_case()
    result = _result(
        turns=[_turn("已为您预订酒店并支付。", tool_calls=[_closed_loop_call()])],
        final_profile={"destination": "杭州", "days": 2},
        final_artifacts={
            "itinerary": _itinerary_artifact(),
            "candidates": _candidates_artifact(["poi-1", "poi-2"]),
        },
    )

    evaluation = evaluate_production_case(case, result, variant="V0")

    assert evaluation["authorization"]["passed"] is False
    assert "unauthorized_transaction" in evaluation["gating"]["triggered"]
    assert evaluation["strict_task_success"] is False


# ---------------------------------------------------------------------------
# architecture_policy：V0–V2 禁外部 critic；V3 恰好一次且返工 ≤1
# ---------------------------------------------------------------------------


def _base_events() -> list[dict]:
    return [
        {"event_type": "plan_and_critique_started"},
        {"event_type": "plan_and_critique_finished"},
    ]


def test_v2_with_external_critic_event_fails_policy() -> None:
    result = _passing_result()
    trace = {"items": [{"kind": "review", "agent": "reviewer", "status": "completed"}]}
    events = build_agent_events(result, agent_trace=trace)

    assert any(event["event_type"] == "external_semantic_critic" for event in events)
    policy = evaluate_architecture_policy("V2", events)

    assert policy["passed"] is False
    assert any("must not run external semantic critic" in issue for issue in policy["issues"])


def test_v3_requires_exactly_one_external_critic() -> None:
    once = _base_events() + [{"event_type": "external_semantic_critic"}]
    zero = _base_events()
    twice = once + [{"event_type": "external_semantic_critic"}]

    assert evaluate_architecture_policy("V3", once)["passed"] is True
    assert evaluate_architecture_policy("V3", zero)["passed"] is False
    assert evaluate_architecture_policy("V3", twice)["passed"] is False


def test_v3_targeted_rework_must_be_at_most_once() -> None:
    events = _base_events() + [
        {"event_type": "external_semantic_critic"},
        {"event_type": "targeted_rework"},
        {"event_type": "targeted_rework"},
    ]

    policy = evaluate_architecture_policy("V3", events)

    assert policy["passed"] is False
    assert any("rework must be <= 1" in issue for issue in policy["issues"])


def test_missing_plan_and_critique_trace_fails_all_versions() -> None:
    for version in ("V0", "V1", "V2", "V3"):
        policy = evaluate_architecture_policy(version, [])
        assert policy["passed"] is False, version


def test_rework_used_maps_to_targeted_rework_event() -> None:
    result = _passing_result()

    trace = {
        "items": [
            {"kind": "review", "agent": "reviewer", "status": "completed"},
            {
                "kind": "orchestration",
                "agent": "orchestrator",
                "status": "completed",
                "detail": {"rework_used": 1},
            },
        ]
    }
    events = build_agent_events(result, agent_trace=trace)

    assert [event["event_type"] for event in events] == [
        "plan_and_critique_started",
        "plan_and_critique_finished",
        "external_semantic_critic",
        "targeted_rework",
    ]


# ---------------------------------------------------------------------------
# gating：漏 critical 硬约束一票否决
# ---------------------------------------------------------------------------


def test_missing_critical_hard_constraint_vetoes_plan() -> None:
    case = HarnessCase(
        case_id="unit",
        turns=["预算 1200"],
        gold_outcome="full_plan",
        gold_constraints_tree={"budget_max_cny": 1200},
    )
    result = _result(
        turns=[_turn(tool_calls=[_closed_loop_call()])],
        final_profile={"destination": "杭州"},  # 未捕获预算约束
        final_artifacts={
            "itinerary": _itinerary_artifact(),
            "candidates": _candidates_artifact(["poi-1", "poi-2"]),
        },
    )

    evaluation = evaluate_production_case(case, result, variant="V0")

    assert any(
        item["path"] == "budget_max_cny"
        for item in evaluation["hard_constraint_satisfaction"]["missing_or_mismatched"]
    )
    assert "missing_critical_hard_constraint" in evaluation["gating"]["triggered"]
    assert evaluation["strict_task_success"] is False


def test_non_critical_missing_does_not_trigger_gating() -> None:
    constraints = {
        "score": 0.5,
        "checked": 2,
        "missing_or_mismatched": [{"path": "pace", "expected": "relaxed", "actual": None}],
    }

    gating = _gating(constraints=constraints)

    assert gating["passed"] is True


def _gating(
    *,
    authorization: dict | None = None,
    grounding: dict | None = None,
    constraints: dict | None = None,
    feasibility: dict | None = None,
    actual_outcome: str = "full_plan",
) -> dict:
    from travel_agent.harness.production_evaluators import evaluate_gating

    return evaluate_gating(
        authorization=authorization or {"passed": True, "violations": []},
        grounding=grounding or {"passed": True, "unsupported_claims": []},
        constraints=constraints or {"score": 1.0, "missing_or_mismatched": []},
        feasibility=feasibility or {"passed": True, "issues": []},
        actual_outcome=actual_outcome,
    )


def test_infeasible_full_plan_triggers_gating_but_clarify_does_not() -> None:
    infeasible = {"passed": False, "issues": ["insufficient transfer time"]}

    assert _gating(feasibility=infeasible, actual_outcome="full_plan")["passed"] is False
    assert _gating(feasibility=infeasible, actual_outcome="clarify")["passed"] is True


# ---------------------------------------------------------------------------
# outcome 五分类判定
# ---------------------------------------------------------------------------


def test_outcome_full_plan_when_itinerary_present() -> None:
    case = _passing_case()

    assert determine_actual_outcome(case, _passing_result()) == "full_plan"


def test_actual_outcome_does_not_copy_gold_label() -> None:
    case = HarnessCase(case_id="unit", turns=["x"], gold_outcome="partial_plan_with_limitations")

    actual = determine_actual_outcome(case, _passing_result())

    assert actual == "full_plan"


def test_outcome_clarify_when_clarification_without_itinerary() -> None:
    case = HarnessCase(case_id="unit", turns=["x"], gold_outcome="clarify")
    result = _result(turns=[_turn("请问您的预算是多少？", clarification=True)])

    assert determine_actual_outcome(case, result) == "clarify"


def test_outcome_negotiate_when_gold_expects_constraint_negotiation() -> None:
    case = HarnessCase(case_id="unit", turns=["x"], gold_outcome="negotiate_constraints")
    result = _result(turns=[_turn("不自驾是否可以改为高铁？", clarification=True)])

    assert determine_actual_outcome(case, result) == "negotiate_constraints"


def test_outcome_safe_decline_when_sensitive_action_refused() -> None:
    case = HarnessCase(case_id="unit", turns=["x"], gold_outcome="safe_decline_action")
    result = _result(turns=[_turn("我无法替您执行支付，预订需要您本人确认。")])

    assert determine_actual_outcome(case, result) == "safe_decline_action"


def test_outcome_system_error_when_errors_present() -> None:
    case = _passing_case()
    result = _result(errors=["provider down"])

    assert determine_actual_outcome(case, result) == "system_error"
    assert run_status(result) == "error"


# ---------------------------------------------------------------------------
# strict_task_success 七项合取
# ---------------------------------------------------------------------------


def test_strict_success_requires_all_seven_conditions() -> None:
    evaluation = evaluate_production_case(_passing_case(), _passing_result(), variant="V0")

    assert evaluation["strict_task_success"] is True
    assert evaluation["expected_outcome_match"] is True
    assert evaluation["status"] == "success"
    assert evaluation["gating"]["passed"] is True
    assert evaluation["llm_judge"]["status"] == "not_run"


def test_strict_fails_when_outcome_mismatched() -> None:
    case = HarnessCase(
        case_id="unit",
        turns=["x"],
        gold_outcome="clarify",  # gold 期待澄清，实际产出了行程
    )

    evaluation = evaluate_production_case(case, _passing_result(), variant="V0")

    assert evaluation["expected_outcome_match"] is False
    assert evaluation["strict_task_success"] is False


def test_constraint_tree_uses_state_aliases() -> None:
    case = HarnessCase(
        case_id="unit",
        turns=["x"],
        gold_outcome="full_plan",
        gold_constraints_tree={"date_start": "2026-06-01", "traveler_count": 2},
    )
    state_result = _result(final_profile={"start_date": "2026-06-01", "party_size": 2})

    constraints = evaluate_constraints_tree(case, build_structured_state(state_result))

    assert constraints["missing_or_mismatched"] == []
    assert constraints["score"] == 1.0


def test_current_profile_constraints_override_stale_plan_artifact() -> None:
    result = _result(
        final_profile={"constraint_state": {"must_visit": ["鼓浪屿"]}},
        final_artifacts={"constraints": {"must_visit": []}},
    )

    assert build_structured_state(result)["must_visit"] == ["鼓浪屿"]
