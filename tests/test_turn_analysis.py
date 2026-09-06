"""轮次分析测试：task_type 识别、动态必填槽位、复杂信号与 LLM 兜底校验。"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from travel_agent.agent.runtime import _missing_required_slots, run_production_turn
from travel_agent.agent.intent import MessageKind
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import (
    DeliveryIntent,
    REQUIRED_SLOTS,
    TaskType,
    TurnAnalysis,
    _merge_llm_payload,
    analyze_travel_turn,
    build_rule_constraint_state,
    build_rule_patches,
    classify_task_type_rule_based,
    has_complex_signals,
    _analyze_with_llm,
    TURN_ANALYSIS_MAX_OUTPUT_TOKENS,
    TURN_ANALYSIS_TIMEOUT_SECONDS,
)
from travel_agent.agent.turn_lifecycle import (
    _is_weather_advice_turn,
    _merge_constraint_state,
    prepare_turn,
)
from travel_agent.profile_patch import PatchOp
from travel_agent.settings import LLMSettings, Settings


def _ctx_with_itinerary():
    ctx = build_session(persist=False)
    ctx.store.put(
        "itinerary",
        {"itinerary": {"city": "南京", "days": [{"day_index": 1}, {"day_index": 2}], "summary": "南京两日"}, "critic": {"passed": True}},
    )
    return ctx


def test_activity_candidate_selection_is_lightweight_poi_advice() -> None:
    assert (
        classify_task_type_rule_based("从春熙路附近选4个不同类型的下午活动")
        == TaskType.POI_ADVICE
    )


def test_candidate_ranking_from_shared_origin_requires_transport_evidence(
    offline_settings,
) -> None:
    message = (
        "候选是甲博物馆、乙公园和丙展馆。"
        "请根据从中央车站出发、18:00结束的条件排序。"
    )
    analysis = analyze_travel_turn(
        message,
        build_session(persist=False),
        offline_settings,
    )

    assert analysis.task_type == TaskType.CANDIDATE_COMPARISON
    assert analysis.constraint_state["target_anchor"] == "中央车站"
    assert "accessibility" in analysis.constraint_state["comparison_dimensions"]


def test_exact_activity_end_time_is_distinct_from_before_deadline() -> None:
    exact = build_rule_constraint_state("广州一日游，晚上20:30结束", {})
    before = build_rule_constraint_state("广州一日游，20:30前结束", {})

    assert exact["activity_end_deadline"] == "20:30"
    assert exact["activity_end_target"] == "20:30"
    assert "target_anchor" not in exact
    assert before["activity_end_deadline"] == "20:30"
    assert "activity_end_target" not in before


def test_destination_comparison_by_route_time_is_candidate_comparison() -> None:
    assert (
        classify_task_type_rule_based(
            "从车站出发，比较去甲馆、乙园和丙滩三个地点的公共交通时间，选一个"
        )
        == TaskType.CANDIDATE_COMPARISON
    )


def test_transport_mode_comparison_for_one_leg_is_route_plan() -> None:
    assert (
        classify_task_type_rule_based("从甲站去乙地，比较公共交通和打车")
        == TaskType.ROUTE_PLAN
    )


def test_origin_is_anchor_for_multi_destination_route_comparison() -> None:
    message = (
        "2027年4月3日上午从中央车站出发，比较去甲展馆、乙公园和丙海滩"
        "三个地点的公共交通时间，选一个半日活动。"
    )
    state = build_rule_constraint_state(
        message,
        build_rule_patches(message, TaskType.CANDIDATE_COMPARISON),
    )

    assert state["origin"] == "中央车站"
    assert state["target_anchor"] == "中央车站"
    assert state["comparison_candidates"] == ["甲展馆", "乙公园", "丙海滩"]
    assert "destination_name" not in state
    assert "destination_area" not in state


def test_explicit_local_adjustment_is_recognized_without_session_context() -> None:
    message = "我已经有两日行程，请只根据天气判断第二天是否替换，不要重写整份行程。"
    assert classify_task_type_rule_based(message) == TaskType.LOCAL_ADJUSTMENT


def test_unknown_request_does_not_default_to_full_trip_plan() -> None:
    assert classify_task_type_rule_based("帮我看看") == TaskType.UNKNOWN
    assert classify_task_type_rule_based("帮我预订机票") == TaskType.SAFE_DECLINE


def test_task_types_use_canonical_routing_goal_values() -> None:
    assert TaskType.FULL_TRIP_PLAN.value == "full_itinerary"
    assert TaskType.ROUTE_QUERY.value == "route_plan"
    assert TaskType.POI_ADVICE.value == "candidate_comparison"
    assert TaskType.ITINERARY_REVISION.value == "itinerary_patch"
    assert TaskType.LOCAL_ADJUSTMENT.value == "local_adjustment_advice"


def test_turn_lifecycle_extracts_rule_profile_once(monkeypatch, offline_settings) -> None:
    from travel_agent import workflow_rules
    from travel_agent.agent import intent, preferences, turn_analysis
    from travel_agent.agent.turn_lifecycle import prepare_turn

    original = workflow_rules.extract_profile_rule_based
    calls = 0

    def counted(message: str):
        nonlocal calls
        calls += 1
        return original(message)

    monkeypatch.setattr(turn_analysis, "extract_profile_rule_based", counted)
    monkeypatch.setattr(intent, "extract_profile_rule_based", counted)
    monkeypatch.setattr(preferences, "extract_profile_rule_based", counted)

    prepared = prepare_turn(
        "杭州三天，喜欢历史",
        build_session(persist=False),
        offline_settings,
        [],
    )

    assert prepared.analysis.task_type == TaskType.FULL_TRIP_PLAN
    assert calls == 1


def test_unknown_task_returns_clarification_before_engine(offline_settings) -> None:
    from travel_agent.agent.turn_lifecycle import prepare_turn

    prepared = prepare_turn(
        "推荐旅行保险",
        build_session(persist=False),
        offline_settings,
        [],
    )

    assert prepared.analysis.task_type == TaskType.UNKNOWN
    assert prepared.early_reply is not None
    assert prepared.early_reply.status == "clarification_required"
    assert prepared.early_reply.failure_reason == "unknown_task_type"


def test_constraint_followup_with_active_trip_is_not_unknown(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "厦门"
    ctx.profile.days = 4

    analysis = analyze_travel_turn("住宿放在思明区。", ctx, offline_settings)

    assert analysis.task_type == TaskType.FULL_TRIP_PLAN


def test_meal_arrangement_is_lightweight_poi_advice() -> None:
    message = "明天有雨的话，帮我安排公司附近晚餐，步行10分钟以内"
    assert classify_task_type_rule_based(message) == TaskType.POI_ADVICE


def test_activity_count_is_not_extracted_as_must_visit(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "厦门玩三天，每天最多安排3个主要活动，酒店每晚600元以内。",
        ctx,
        offline_settings,
    )

    assert "must_visit" not in analysis.patches
    assert analysis.constraint_state.get("must_visit") in (None, [])
    assert analysis.constraint_state["max_major_activities_per_day"] == 3


def test_venues_pending_feasibility_check_are_candidates_not_must_visits(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "周一想去甲博物馆、乙公园，请先核验这些地点当天是否适合安排，再生成路线。",
        ctx,
        offline_settings,
    )

    assert "must_visit" not in analysis.patches
    assert analysis.constraint_state.get("must_visit") in (None, [])
    assert analysis.constraint_state["candidate_attractions"] == ["甲博物馆", "乙公园"]


def test_candidate_list_with_active_pruning_request_is_not_hardened_to_must_visit(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "从测试城南站出发，想去甲博物馆、乙公园和丙古街，帮我判断能否都去，不能就主动删减。",
        ctx,
        offline_settings,
    )

    assert "must_visit" not in analysis.patches
    assert analysis.constraint_state.get("must_visit") in (None, [])
    assert analysis.constraint_state["candidate_attractions"] == ["甲博物馆", "乙公园", "丙古街"]
    assert analysis.constraint_state["candidate_only"] is True


def test_hotel_comparison_candidates_strip_request_and_action_prefixes() -> None:
    variants = {
        "请比较鼓楼和河西住在哪边更合适": ["鼓楼", "河西"],
        "请对比住在老城还是住在新区哪个更方便": ["老城", "新区"],
    }

    for message, expected in variants.items():
        state = build_rule_constraint_state(message, {})
        assert state["comparison_candidates"] == expected
        assert "lodging_area" not in state


def test_hotel_comparison_with_candidates_does_not_require_plan_artifact(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "南京"

    prepared = prepare_turn(
        "请比较鼓楼和河西住在哪边更合适",
        ctx,
        offline_settings,
        [],
    )

    assert prepared.early_reply is None
    assert ctx.store.latest("itinerary") is None
    assert ctx.profile.constraint_state["comparison_candidates"] == ["鼓楼", "河西"]


def test_hotel_comparison_with_one_candidate_still_requests_candidates(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "南京"

    prepared = prepare_turn(
        "请比较鼓楼住哪里更方便",
        ctx,
        offline_settings,
        [],
    )

    assert prepared.early_reply is not None
    assert prepared.early_reply.failure_reason == "missing_comparison_candidates"
    assert "comparison_candidates" not in ctx.profile.constraint_state
    assert "lodging_area" not in ctx.profile.constraint_state


def test_candidate_discovery_location_satisfies_destination_gate(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)

    prepared = prepare_turn(
        "在临海湖畔附近找3家适合聚餐的餐厅，人均120元以内。",
        ctx,
        offline_settings,
        [],
    )

    assert prepared.early_reply is None
    assert ctx.profile.destination is None
    assert ctx.profile.constraint_state["location"] == "临海湖畔附近"
    assert _missing_required_slots(ctx.profile, TaskType.CANDIDATE_COMPARISON) == []


def test_weather_condition_does_not_bypass_domain_artifact_delivery(
    offline_settings,
) -> None:
    restaurant_ctx = build_session(persist=False)
    restaurant = prepare_turn(
        "明天下雨的话，在滨河大道附近找晚餐，人均100元以内。",
        restaurant_ctx,
        offline_settings,
        [],
    )
    assert restaurant.early_reply is None
    assert not _is_weather_advice_turn(restaurant)

    local_ctx = build_session(persist=False)
    local = prepare_turn(
        "我已有两日行程，只判断第二天山景是否因下雨替换，并给室内备选。",
        local_ctx,
        offline_settings,
        [],
    )
    assert local.early_reply is None
    assert not _is_weather_advice_turn(local)
    # Turn-local selectors reach workers and validation, while the durable
    # itinerary constraint state remains unchanged by an advice request.
    effective = local.turn_inputs["profile"]["constraint_state"]
    assert effective["weather_condition"] == "rain_if_true"
    assert effective["need_indoor_backup"] is True
    assert effective["referenced_day_index"] == 2
    assert "destinations" not in effective
    assert "weather_condition" not in local_ctx.profile.constraint_state
    assert "need_indoor_backup" not in local_ctx.profile.constraint_state


def test_lightweight_reply_audits_ephemeral_constraints_without_persisting_them(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn(
        "我有自己的行程，只判断第二天户外活动是否因下雨替换，并给室内备选。",
        ctx=ctx,
        settings=offline_settings,
    )

    turn_state = reply.profile["constraint_state"]
    assert turn_state["weather_condition"] == "rain_if_true"
    assert turn_state["need_indoor_backup"] is True
    assert turn_state["referenced_day_index"] == 2
    assert "weather_condition" not in ctx.profile.constraint_state
    assert "need_indoor_backup" not in ctx.profile.constraint_state


def test_restaurant_request_phrase_is_not_promoted_to_must_visit(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "明天下雨的话，安排公司附近晚餐，地点在滨河大道，人均100元。",
        ctx,
        offline_settings,
    )

    assert analysis.task_type == TaskType.CANDIDATE_COMPARISON
    assert analysis.constraint_state.get("must_visit") in (None, [])


def test_destination_cities_are_not_hardened_into_must_visits(
    offline_settings,
) -> None:
    for message, city in (("我想去上海", "上海"), ("我想去成都", "成都")):
        analysis = analyze_travel_turn(
            message,
            build_session(persist=False),
            offline_settings,
        )
        assert analysis.patches["destination"].value == city
        assert "must_visit" not in analysis.patches
        assert "must_visit" not in analysis.constraint_state
        assert analysis.constraint_state["destination_city"] == city


def test_destination_city_does_not_create_must_visit_constraint_event(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)

    prepared = prepare_turn("我想去广州", ctx, offline_settings, [])

    assert prepared.early_reply is not None
    assert prepared.early_reply.failure_reason == "missing_slot"
    assert ctx.profile.destination == "广州"
    assert ctx.profile.must_visit == []
    assert "must_visit" not in ctx.profile.constraint_state
    assert all(
        event["field"] != "must_visit"
        for event in ctx.profile.constraint_state["_constraint_events"]
    )


def test_named_attractions_remain_must_visits(offline_settings) -> None:
    for message, venue in (("我想去故宫", "故宫"), ("必须去陕西历史博物馆", "陕西历史博物馆")):
        analysis = analyze_travel_turn(
            message,
            build_session(persist=False),
            offline_settings,
        )
        assert analysis.patches["must_visit"].value == [venue]
        assert analysis.constraint_state["must_visit"] == [venue]


def test_explicit_must_visit_remains_hard_when_verification_is_requested(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "故宫必须去，请先核验开放时间。",
        ctx,
        offline_settings,
    )

    assert analysis.patches["must_visit"].value == ["故宫"]


def test_lodging_and_reverse_must_visit_are_both_preserved(
    offline_settings,
) -> None:
    analysis = analyze_travel_turn(
        "带孩子去北京两天，住王府井附近，故宫必须去。",
        build_session(persist=False),
        offline_settings,
    )

    assert analysis.constraint_state["lodging_area"] == "王府井附近"
    assert analysis.constraint_state["must_visit"] == ["故宫"]


def test_must_preserve_extracts_object_after_action(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "最后方案里必须保留西湖，预算不要增加。",
        ctx,
        offline_settings,
    )

    assert analysis.patches["must_visit"].value == ["西湖"]
    assert analysis.constraint_state["must_visit"] == ["西湖"]


def test_daily_walking_not_exceeding_km_is_preserved(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "住西湖东侧，每天步行不超过6公里。",
        ctx,
        offline_settings,
    )

    assert analysis.constraint_state["max_walking_km_per_day"] == 6
    assert "transport_mode" not in analysis.patches


def test_long_horizon_constraint_followups_use_canonical_state(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "厦门"
    ctx.profile.days = 4

    budget = analyze_travel_turn("最初预算6000元。", ctx, offline_settings)
    lodging = analyze_travel_turn("住宿放在思明区。", ctx, offline_settings)
    dietary = analyze_travel_turn("其中一人不吃海鲜。", ctx, offline_settings)

    assert budget.kind == MessageKind.TRAVEL
    assert budget.constraint_state["budget_max_cny"] == 6000
    assert lodging.kind == MessageKind.TRAVEL
    assert lodging.constraint_state["lodging_area"] == "思明区"
    assert dietary.kind == MessageKind.TRAVEL
    assert dietary.constraint_state["dietary"] == ["不吃海鲜"]
    interest_patch = dietary.patches.get("interests")
    assert interest_patch is None or "food" not in list(interest_patch.value or [])
    assert "food_preference" not in dietary.patches


def test_rule_budget_field_wins_over_llm_total_budget_alias() -> None:
    rule = TurnAnalysis(
        kind=MessageKind.TRAVEL,
        task_type=TaskType.FULL_TRIP_PLAN,
        constraint_state={"budget_max_cny": 6000.0},
    )

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "constraint_state": {"budget_total_cny": 6000},
        },
        rule,
    )

    assert merged.constraint_state["budget_max_cny"] == 6000
    assert "budget_total_cny" not in merged.constraint_state


def test_replacement_followup_adds_new_must_visit(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    analysis = analyze_travel_turn("改去上海自然博物馆。", ctx, offline_settings)

    assert analysis.task_type == TaskType.FULL_ITINERARY
    assert analysis.constraint_state["must_visit"] == ["上海自然博物馆"]


def test_llm_merge_keeps_rule_replacement_must_visit_without_profile_patch() -> None:
    rule = TurnAnalysis(
        kind=MessageKind.TRAVEL,
        task_type=TaskType.ITINERARY_REVISION,
        constraint_state={"removed": ["虎丘"], "must_visit": ["苏州博物馆"]},
    )

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "itinerary_revision",
            "constraint_state": {"removed": ["虎丘"]},
        },
        rule,
    )

    assert merged.constraint_state["must_visit"] == ["苏州博物馆"]


def test_llm_constraint_state_rejects_non_numeric_numeric_fields() -> None:
    rule = TurnAnalysis(kind=MessageKind.TRAVEL, task_type=TaskType.FULL_TRIP_PLAN)

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "constraint_state": {
                "hotel_budget_per_night_cny": "降档",
                "budget_max_cny": 3500,
            },
        },
        rule,
    )

    assert "hotel_budget_per_night_cny" not in merged.constraint_state
    assert merged.constraint_state["budget_max_cny"] == 3500


def test_meal_appointment_uses_canonical_fixed_event(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    analysis = analyze_travel_turn("第二天晚饭已经约在陆家嘴。", ctx, offline_settings)

    assert analysis.constraint_state["fixed_events"] == [{
        "day": 2,
        "start": "18:00",
        "end": "20:00",
        "location": "陆家嘴",
    }]
    assert analysis.constraint_state["user_owned_unspecified_fixed_event_locations"] == [
        "陆家嘴"
    ]


def test_lodging_confirmation_preserves_more_specific_existing_area(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.destination = "上海"
    ctx.profile.days = 4
    ctx.profile.constraint_state["lodging_area"] = "人民广场附近"

    analysis = analyze_travel_turn(
        "住宿仍住人民广场，不要跟着晚饭换酒店。",
        ctx,
        offline_settings,
    )

    assert analysis.constraint_state["lodging_area"] == "人民广场附近"


def test_generic_lodging_preserve_does_not_create_literal_area(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.constraint_state["lodging_area"] = "西湖东侧"

    analysis = analyze_travel_turn("住宿区域不要改。", ctx, offline_settings)

    assert analysis.constraint_state["lodging_area"] == "西湖东侧"
    assert analysis.kind == MessageKind.TRAVEL
    assert analysis.task_type == TaskType.FULL_ITINERARY


def test_dietary_adjective_and_broad_fixed_event_are_canonical(offline_settings) -> None:
    ctx = _ctx_with_itinerary()

    dietary = analyze_travel_turn("回到行程，饮食要清淡。", ctx, offline_settings)
    event = analyze_travel_turn("第三天下午安排自由活动。", ctx, offline_settings)

    assert dietary.constraint_state["dietary"] == ["清淡"]
    assert event.constraint_state["fixed_events"] == [{
        "day": 3,
        "start": "14:00",
        "end": "18:00",
        "location": "自由活动",
    }]
    assert "自由活动" not in event.constraint_state.get("must_visit", [])


def test_last_day_deadline_is_full_rebuild_not_one_day_trip(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.days = 5

    analysis = analyze_travel_turn("最后一天17点前到杭州东站。", ctx, offline_settings)

    assert analysis.task_type == TaskType.FULL_ITINERARY
    assert "duration_days" not in analysis.constraint_state
    assert "days" not in analysis.patches
    assert analysis.constraint_state["return_deadline"] == "17:00"
    assert analysis.constraint_state["return_location"] == "杭州东站"


def test_final_version_followup_routes_to_full_rebuild(offline_settings) -> None:
    ctx = _ctx_with_itinerary()

    analysis = analyze_travel_turn(
        "按全部条件出最终版，别恢复已删除的地点。",
        ctx,
        offline_settings,
    )

    assert analysis.kind == MessageKind.TRAVEL
    assert analysis.task_type == TaskType.FULL_ITINERARY


def test_global_constraint_changes_never_route_to_itinerary_patch(
    offline_settings,
) -> None:
    variants = (
        "整体预算改成5000元，请调整现有方案。",
        "住宿区域改到老城区。",
        "全程饮食要清淡。",
        "必须去中央博物馆。",
        "最后一天17:00前返回车站。",
    )
    for message in variants:
        analysis = analyze_travel_turn(message, _ctx_with_itinerary(), offline_settings)
        assert analysis.task_type == TaskType.FULL_ITINERARY, message


def test_specific_fixed_event_requires_atomic_full_rebuild(offline_settings) -> None:
    analysis = analyze_travel_turn(
        "第二天10:00到12:00固定参加预约活动。",
        _ctx_with_itinerary(),
        offline_settings,
    )
    assert analysis.task_type == TaskType.FULL_ITINERARY
    assert analysis.delivery_intent == DeliveryIntent.REBUILD_NOW


def test_specific_day_activity_change_still_routes_to_patch(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    analysis = analyze_travel_turn(
        "请判断第二天户外活动是否因下雨替换，并给室内备选。",
        ctx,
        offline_settings,
    )
    assert analysis.task_type == TaskType.ITINERARY_PATCH


def test_removed_shorthand_is_normalized_to_known_full_entity() -> None:
    from travel_agent.agent.turn_lifecycle import _merge_constraint_state

    state = {
        "must_visit": ["西湖", "良渚博物院"],
        "candidate_attractions": ["良渚博物院"],
    }

    _merge_constraint_state(
        state,
        {"removed": ["良渚"], "must_visit": ["浙江省博物馆"]},
    )

    assert state["removed"] == ["良渚博物院"]
    assert state["must_visit"] == ["西湖", "浙江省博物馆"]
    assert "candidate_attractions" not in state


def test_cancelled_venue_cannot_become_destination_via_llm() -> None:
    rule = TurnAnalysis(
        kind=MessageKind.TRAVEL,
        task_type=TaskType.ITINERARY_REVISION,
        constraint_state={"removed": ["迪士尼"]},
    )

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "itinerary_revision",
            "constraint_state": {"destinations": ["迪士"]},
        },
        rule,
    )

    assert "destinations" not in merged.constraint_state


def test_standalone_cancel_and_budget_increase_are_deterministic(offline_settings) -> None:
    ctx = _ctx_with_itinerary()

    cancelled = analyze_travel_turn("不去迪士尼了，排队太久。", ctx, offline_settings)
    budget = analyze_travel_turn(
        "预算最多再加500，改成6500元。",
        ctx,
        offline_settings,
    )

    assert cancelled.constraint_state["removed"] == ["迪士尼"]
    assert "destinations" not in cancelled.constraint_state
    assert budget.constraint_state["budget_max_cny"] == 6500


def test_cancelled_venue_never_overwrites_destination_without_current_itinerary(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "上海"
    ctx.profile.constraint_state = {"destinations": ["上海"], "destination_city": "上海"}

    analysis = analyze_travel_turn("不去迪士尼了，排队太久。", ctx, offline_settings)

    assert analysis.patches.get("destination") is None
    assert "destinations" not in analysis.constraint_state
    assert analysis.constraint_state["removed"] == ["迪士尼"]


def test_child_and_relaxed_pace_followup_are_deterministic(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    analysis = analyze_travel_turn("有个8岁孩子，节奏别太赶。", ctx, offline_settings)

    assert analysis.constraint_state["child_age"] == 8
    assert analysis.constraint_state["pace"] == "relaxed"


def test_want_to_see_before_comma_remains_a_hard_visit(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "想看西湖，但不要安排太多步行。",
        ctx,
        offline_settings,
    )

    assert analysis.patches["must_visit"].value == ["西湖"]


def test_constraint_supplement_with_existing_plan_is_revision() -> None:
    assert (
        classify_task_type_rule_based("补充一下，每天步行控制在6公里以内", _ctx_with_itinerary())
        == TaskType.ITINERARY_REVISION
    )


def test_llm_cannot_downgrade_full_plan_and_bypass_gates() -> None:
    rule = TurnAnalysis(kind=MessageKind.TRAVEL, task_type=TaskType.FULL_TRIP_PLAN)

    merged = _merge_llm_payload(
        {"kind": "travel", "task_type": "poi_advice", "slots": {}},
        rule,
    )

    assert merged.task_type == TaskType.FULL_TRIP_PLAN


def test_llm_cannot_harden_candidates_into_must_visit() -> None:
    rule = TurnAnalysis(
        kind=MessageKind.TRAVEL,
        task_type=TaskType.FULL_TRIP_PLAN,
        constraint_state={"candidate_attractions": ["甲博物馆", "乙公园"]},
    )

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "slots": {"must_visit": {"op": "set", "value": ["甲博物馆", "乙公园"]}},
            "constraint_state": {"must_visit": ["甲博物馆", "乙公园"]},
        },
        rule,
    )

    assert "must_visit" not in merged.patches
    assert "must_visit" not in merged.constraint_state
    assert merged.constraint_state["candidate_attractions"] == ["甲博物馆", "乙公园"]


def test_llm_cannot_infer_walking_mode_from_walking_limit() -> None:
    rule = TurnAnalysis(
        kind=MessageKind.TRAVEL,
        task_type=TaskType.FULL_TRIP_PLAN,
        constraint_state={"max_walking_km_per_day": 6},
    )

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "slots": {"transport_mode": {"op": "set", "value": "walk"}},
            "constraint_state": {
                "max_walking_km_per_day": 6,
                "transport_mode": "walk",
            },
        },
        rule,
    )

    assert "transport_mode" not in merged.patches
    assert "transport_mode" not in merged.constraint_state


def test_llm_cannot_infer_lodging_downgrade_from_total_budget() -> None:
    rule = TurnAnalysis(
        kind=MessageKind.TRAVEL,
        task_type=TaskType.FULL_TRIP_PLAN,
        constraint_state={"budget_max_cny": 5000},
    )

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "constraint_state": {
                "budget_max_cny": 5000,
                "lodging_flexibility": "can_downgrade",
            },
        },
        rule,
    )

    assert "lodging_flexibility" not in merged.constraint_state


def test_explicit_lodging_downgrade_survives_llm_merge(offline_settings) -> None:
    analysis = analyze_travel_turn(
        "两个人去海滨城市三天，住宿可以降档，总预算5000元。",
        build_session(persist=False),
        offline_settings,
    )

    assert analysis.constraint_state["lodging_flexibility"] == "can_downgrade"


def test_llm_candidate_list_drops_count_and_budget_fragments() -> None:
    rule = TurnAnalysis(kind=MessageKind.TRAVEL, task_type=TaskType.FULL_TRIP_PLAN)

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "slots": {},
            "constraint_state": {
                "candidate_attractions": ["3个主要活动", "酒店预算每晚600元以内"]
            },
        },
        rule,
    )

    assert "candidate_attractions" not in merged.constraint_state


def test_llm_candidate_list_drops_itinerary_instruction_fragments() -> None:
    rule = TurnAnalysis(kind=MessageKind.TRAVEL, task_type=TaskType.FULL_TRIP_PLAN)

    merged = _merge_llm_payload(
        {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "slots": {},
            "constraint_state": {
                "candidate_attractions": ["三天行程", "不要让其他项目", "预约冲突"]
            },
        },
        rule,
    )

    assert "candidate_attractions" not in merged.constraint_state


# --------------------------------------------------------------------------- #
# task_type 规则识别（用户给出的四个典型用例）
# --------------------------------------------------------------------------- #
def test_route_query_is_not_asked_for_days() -> None:
    ctx = build_session(persist=False)
    assert classify_task_type_rule_based("虹桥机场到外滩怎么走", ctx) == TaskType.ROUTE_QUERY
    assert _missing_required_slots(ctx.profile, TaskType.ROUTE_QUERY) == []


def test_route_query_supports_from_to_with_go_wording() -> None:
    ctx = build_session(persist=False)
    message = "从上海虹桥站去迪士尼度假区，比较公共交通和打车"
    assert classify_task_type_rule_based(message, ctx) == TaskType.ROUTE_QUERY


def test_full_plan_cue_beats_embedded_route_context() -> None:
    message = "从上海虹桥去杭州一日游，不自驾，也不要安排太多步行"
    assert classify_task_type_rule_based(message) == TaskType.FULL_TRIP_PLAN


def test_one_day_trip_sets_days_without_clarification() -> None:
    patches = build_rule_patches("帮我规划2026年10月1日上海一日游", TaskType.FULL_TRIP_PLAN)
    assert patches["days"].value == 1


def test_date_range_derives_duration_days() -> None:
    patches = build_rule_patches("2026年9月14日至16日去西安", TaskType.FULL_TRIP_PLAN)
    assert patches["days"].value == 3


def test_hotel_advice_only_requires_destination() -> None:
    ctx = build_session(persist=False)
    assert classify_task_type_rule_based("新街口和夫子庙住哪里方便", ctx) == TaskType.POI_ADVICE
    assert REQUIRED_SLOTS[TaskType.POI_ADVICE] == ("destination",)
    assert _missing_required_slots(ctx.profile, TaskType.POI_ADVICE) == ["destination"]
    ctx.profile.destination = "南京"
    assert _missing_required_slots(ctx.profile, TaskType.POI_ADVICE) == []


def test_revision_is_not_asked_and_days_not_misextracted() -> None:
    ctx = _ctx_with_itinerary()
    message = "把第二天的中山陵删掉"
    assert classify_task_type_rule_based(message, ctx) == TaskType.ITINERARY_REVISION
    patches = build_rule_patches(message, TaskType.ITINERARY_REVISION)
    assert "days" not in patches
    assert "destination" not in patches
    assert _missing_required_slots(ctx.profile, TaskType.ITINERARY_REVISION) == []


def test_day_advice_only_requires_destination() -> None:
    ctx = build_session(persist=False)
    assert classify_task_type_rule_based("明天下雨去哪里", ctx) == TaskType.DAY_ADVICE
    assert REQUIRED_SLOTS[TaskType.DAY_ADVICE] == ("destination",)


def test_full_trip_plan_keeps_destination_and_days() -> None:
    ctx = build_session(persist=False)
    assert classify_task_type_rule_based("帮我规划杭州三天", ctx) == TaskType.FULL_TRIP_PLAN
    assert _missing_required_slots(ctx.profile, TaskType.FULL_TRIP_PLAN) == ["destination", "days"]


def test_full_trip_convenience_and_cost_words_do_not_create_comparison_contract(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "安排海州一日游，两个人，交通方便且总费用控制在1200元内。",
        ctx,
        offline_settings,
    )
    assert analysis.task_type == TaskType.FULL_ITINERARY
    assert "comparison_dimensions" not in analysis.constraint_state


def test_route_query_run_turn_does_not_clarify_days(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("虹桥机场到外滩怎么走", ctx=ctx, settings=offline_settings)
    assert reply.clarification is False
    assert reply.failure_reason != "missing_slot"


def test_route_only_with_date_and_arranged_trip_never_asks_days(offline_settings) -> None:
    ctx = build_session(persist=False)
    message = (
        "我已经订好酒店和景点，只需要规划2027年3月8日从北城机场到"
        "南区酒店带的公共交通路线，并给一个打车备选。不要再推荐景点和餐厅。"
    )
    assert classify_task_type_rule_based(message, ctx) == TaskType.ROUTE_PLAN
    prepared = prepare_turn(message, ctx, offline_settings, [])
    assert prepared.analysis.task_type == TaskType.ROUTE_PLAN
    assert prepared.early_reply is None
    assert ctx.profile.constraint_state["public_transport_required"] is True
    assert ctx.profile.constraint_state["taxi_backup"] is True


def test_state_only_followup_does_not_launch_search_or_planner(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    prepared = prepare_turn(
        "补充一下条件：每天步行不超过6公里，先记住，之后再出最终版。",
        ctx,
        offline_settings,
        [("user", "请安排五天行程")],
    )
    assert prepared.early_reply is not None
    assert prepared.early_reply.planner_status == "deferred_state_update"
    assert ctx.profile.constraint_state["max_walking_km_per_day"] == 6


def test_ordinal_day_is_reference_not_duration_variant() -> None:
    patches = build_rule_patches("第三天下午保留自由活动", TaskType.LOCAL_ADJUSTMENT)
    state = build_rule_constraint_state("第三天下午保留自由活动", patches)
    assert "days" not in patches
    assert "duration_days" not in state
    assert state["referenced_day_index"] == 3
    assert state["fixed_events"][0]["day"] == 3


def test_hotel_advice_run_turn_only_asks_destination(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("新街口和夫子庙住哪里方便", ctx=ctx, settings=offline_settings)
    assert reply.clarification is True
    assert reply.failure_reason == "missing_slot"
    assert "destination" in reply.text
    assert "days" not in reply.text


# --------------------------------------------------------------------------- #
# 复杂信号检测
# --------------------------------------------------------------------------- #
def test_complex_signals_detected() -> None:
    assert has_complex_signals("苏州不去了，换成杭州")
    assert has_complex_signals("周五晚上到、周日下午走")
    assert has_complex_signals("上海、杭州、苏州一共六天")
    assert has_complex_signals("下雨就不去崂山")
    assert has_complex_signals("把那里放在第二天下午")
    assert not has_complex_signals("帮我规划北京三天")


def test_simple_message_skips_llm(monkeypatch, offline_settings) -> None:
    settings = Settings(
        llm=LLMSettings(provider="openai", api_key="key"),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )
    ctx = build_session(persist=False)

    def should_not_call(*args, **kwargs):
        raise AssertionError("规则高置信且无复杂信号时不应调用 LLM")

    monkeypatch.setattr("travel_agent.agent.turn_analysis._analyze_with_llm", should_not_call)
    analysis = analyze_travel_turn("帮我规划杭州三天", ctx, settings)
    assert analysis.source == "rule"
    assert analysis.patches["destination"].value == "杭州"
    assert analysis.patches["days"].value == 3


def test_llm_turn_analysis_uses_bounded_json_request(monkeypatch, offline_settings) -> None:
    bound: dict = {}
    built: dict = {}

    class Model:
        def bind(self, **kwargs):
            bound.update(kwargs)
            return self

        def invoke(self, messages, config=None):
            return SimpleNamespace(
                content=(
                    '{"kind":"travel","task_type":"full_trip_plan",'
                    '"slots":{},"constraint_state":{}}'
                )
            )

    def build(_settings, **kwargs):
        built.update(kwargs)
        return Model()

    monkeypatch.setattr("travel_agent.agent.runtime._build_chat_model", build)
    settings = Settings(
        llm=LLMSettings(provider="openai", api_key="key", timeout_seconds=60),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )

    payload = _analyze_with_llm(
        "杭州两日游，如果下雨就改室内",
        build_session(persist=False),
        settings,
        [],
        None,
    )

    assert payload["task_type"] == "full_trip_plan"
    assert built == {"timeout_seconds": TURN_ANALYSIS_TIMEOUT_SECONDS, "max_retries": 0}
    assert bound == {
        "max_tokens": TURN_ANALYSIS_MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }


def test_complex_message_triggers_llm_and_validates(monkeypatch, offline_settings) -> None:
    settings = Settings(
        llm=LLMSettings(provider="fake", api_key="key"),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )
    ctx = build_session(persist=False)

    def fake_analyze(user_message, ctx, settings, history, evaluation_trace):
        return {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "slots": {
                "destination": {"op": "set", "value": "杭州"},
                "days": {"op": "set", "value": 6},
                "budget_level": {"op": "set", "value": "luxury"},  # 非法枚举，应被丢弃
                "interests": {"op": "set", "value": ["历史", "food"]},
            },
            "constraint_state": {
                "return_deadline": "18:30",
                "self_driving_allowed": False,
            },
        }

    monkeypatch.setattr("travel_agent.agent.turn_analysis._analyze_with_llm", fake_analyze)
    analysis = analyze_travel_turn("上海、杭州、苏州一共六天", ctx, settings)
    assert analysis.source == "llm"
    assert analysis.task_type == TaskType.FULL_TRIP_PLAN
    assert analysis.patches["destination"].value == "杭州"
    assert analysis.patches["days"].value == 6
    assert "budget_level" not in analysis.patches
    assert "history" in analysis.patches["interests"].value
    assert analysis.constraint_state["return_deadline"] == "18:30"
    assert analysis.constraint_state["self_driving_allowed"] is False


def test_rule_constraint_state_parses_compound_recommendation_ban(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "只规划公共交通路线，不要再推荐景点和餐厅。",
        ctx,
        offline_settings,
    )

    assert analysis.constraint_state["exclude"] == ["景点推荐", "餐厅推荐"]


def test_wheelchair_implies_accessibility_priority(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn(
        "一行三人，其中一人使用轮椅，必须优先无障碍路线。",
        ctx,
        offline_settings,
    )

    assert analysis.constraint_state["wheelchair_user"] is True
    assert analysis.constraint_state["accessibility_priority"] is True


def test_llm_failure_falls_back_to_rules(monkeypatch, offline_settings) -> None:
    settings = Settings(
        llm=LLMSettings(provider="fake", api_key="key"),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )
    ctx = build_session(persist=False)

    def broken_analyze(*args, **kwargs):
        raise ValueError("bad json")

    monkeypatch.setattr("travel_agent.agent.turn_analysis._analyze_with_llm", broken_analyze)
    # 复杂信号触发 LLM 调用，失败后必须回退规则结果
    analysis = analyze_travel_turn("帮我规划杭州三天，如果下雨就改成室内景点", ctx, settings)
    assert analysis.source == "rule"
    assert analysis.patches["destination"].value == "杭州"
    assert analysis.patches["days"].value == 3


def test_llm_clear_semantics(monkeypatch, offline_settings) -> None:
    settings = Settings(
        llm=LLMSettings(provider="fake", api_key="key"),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )
    ctx = build_session(persist=False)

    def fake_analyze(user_message, ctx, settings, history, evaluation_trace):
        return {
            "kind": "travel",
            "task_type": "full_trip_plan",
            "slots": {
                "destination": {"op": "set", "value": "杭州"},
                "start_date": {"op": "clear"},
            },
        }

    monkeypatch.setattr("travel_agent.agent.turn_analysis._analyze_with_llm", fake_analyze)
    analysis = analyze_travel_turn("苏州不去了，换成杭州，日期先不定", ctx, settings)
    assert analysis.patches["destination"].op == PatchOp.SET
    assert analysis.patches["start_date"].op == PatchOp.CLEAR


def test_delivery_intent_separates_declarative_state_from_immediate_rebuild(
    offline_settings,
) -> None:
    deferred_ctx = _ctx_with_itinerary()
    deferred = analyze_travel_turn(
        "整体预算改成5000元。", deferred_ctx, offline_settings
    )
    assert deferred.task_type == TaskType.FULL_ITINERARY
    assert deferred.delivery_intent == DeliveryIntent.STATE_UPDATE_ONLY

    immediate_ctx = _ctx_with_itinerary()
    immediate = analyze_travel_turn(
        "整体预算改成5000元，请立即调整现有方案。",
        immediate_ctx,
        offline_settings,
    )
    assert immediate.task_type == TaskType.FULL_ITINERARY
    assert immediate.delivery_intent == DeliveryIntent.REBUILD_NOW


def test_delivery_intent_uses_existing_structured_budget_for_immediate_rebuild(
    offline_settings,
) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.destination = "南京"
    ctx.profile.days = 2
    ctx.profile.budget_limit = 8000
    ctx.profile.constraint_state["budget_max_cny"] = 8000

    analysis = analyze_travel_turn("总预算改为6200元。", ctx, offline_settings)

    assert analysis.task_type == TaskType.FULL_ITINERARY
    assert analysis.constraint_state["budget_max_cny"] == 6200.0
    assert analysis.delivery_intent == DeliveryIntent.REBUILD_NOW


def test_pending_state_beats_structured_replacement_until_final_delivery(
    offline_settings,
) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.destination = "南京"
    ctx.profile.days = 2
    ctx.profile.budget_limit = 8000
    ctx.profile.constraint_state.update({
        "budget_max_cny": 8000,
        "_plan_status": "rebuild_pending",
    })

    update = analyze_travel_turn("总预算改为6200元。", ctx, offline_settings)
    final = analyze_travel_turn("按全部条件生成最终版。", ctx, offline_settings)

    assert update.delivery_intent == DeliveryIntent.STATE_UPDATE_ONLY
    assert final.delivery_intent == DeliveryIntent.REBUILD_NOW


def test_pending_state_accepts_common_final_delivery_phrasings(
    offline_settings,
) -> None:
    for message in (
        "生成最终计划，保留当前全部硬约束。",
        "最后方案里必须保留西湖，预算不要增加。",
        "再确认一次：必去地点和忌口都保留。",
    ):
        ctx = _ctx_with_itinerary()
        ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"

        analysis = analyze_travel_turn(message, ctx, offline_settings)

        assert analysis.delivery_intent == DeliveryIntent.REBUILD_NOW
        assert analysis.task_type == TaskType.FULL_ITINERARY


def test_pending_state_without_current_plan_defers_updates_but_accepts_final_request(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "上海"
    ctx.profile.days = 4
    ctx.profile.constraint_state.update({
        "destination_city": "上海",
        "duration_days": 4,
        "_plan_status": "rebuild_pending",
    })

    update = analyze_travel_turn("预算改成6500元。", ctx, offline_settings)
    final = analyze_travel_turn(
        "生成最终计划，保留当前全部硬约束。", ctx, offline_settings
    )

    assert update.delivery_intent == DeliveryIntent.STATE_UPDATE_ONLY
    assert final.delivery_intent == DeliveryIntent.REBUILD_NOW


def test_delivery_intent_limits_patch_to_concrete_schedule_scope(
    offline_settings,
) -> None:
    ctx = _ctx_with_itinerary()
    local = analyze_travel_turn(
        "把第二天下午的博物馆替换成室内展览。", ctx, offline_settings
    )
    assert local.task_type == TaskType.ITINERARY_PATCH
    assert local.delivery_intent == DeliveryIntent.LOCAL_PATCH


def test_weather_interjection_is_lightweight_even_with_plan_context(
    offline_settings,
) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"
    analysis = analyze_travel_turn("明天天气怎么样？", ctx, offline_settings)
    assert analysis.task_type == TaskType.CANDIDATE_COMPARISON
    assert analysis.delivery_intent == DeliveryIntent.LIGHTWEIGHT_ADVICE
    assert "start_date" not in analysis.patches


def test_clothing_weather_interjection_never_mutates_durable_trip_state(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "苏州"
    ctx.profile.days = 3
    _merge_constraint_state(
        ctx.profile.constraint_state,
        {"destination_city": "苏州", "duration_days": 3},
        source_turn=1,
    )
    from travel_agent.artifact_policy import update_constraint_version

    update_constraint_version(ctx.profile)
    before = copy.deepcopy(ctx.profile.constraint_state)

    prepared = prepare_turn(
        "顺便问一句，杭州十月通常需要带外套吗？",
        ctx,
        offline_settings,
        [],
    )

    assert prepared.analysis.delivery_intent == DeliveryIntent.LIGHTWEIGHT_ADVICE
    assert prepared.analysis.constraint_state["weather_condition"] == "query"
    assert prepared.turn_inputs["profile"]["constraint_state"]["destination_city"] == "杭州"
    assert ctx.profile.destination == "苏州"
    assert ctx.profile.constraint_state == before


def test_explicit_total_budget_survives_canonical_state_merge() -> None:
    state: dict[str, object] = {}

    _merge_constraint_state(
        state,
        {"budget_max_cny": 1800.0, "budget_total_cny": 1800.0},
        source_turn=1,
    )

    assert state["budget_max_cny"] == 1800.0
    assert state["budget_total_cny"] == 1800.0


def test_trip_date_range_is_not_removed_by_dated_fixed_event(
    offline_settings,
) -> None:
    analysis = analyze_travel_turn(
        "2026年10月3日至5日去成都。10月4日15:00到17:00已经预约三星堆讲解，请安排完整行程。",
        build_session(persist=False),
        offline_settings,
    )

    assert analysis.constraint_state["date_start"] == "2026-10-03"
    assert analysis.constraint_state["date_end"] == "2026-10-05"
    assert analysis.constraint_state["fixed_events"]


def test_prepaid_lodging_phrase_is_not_a_lodging_area(offline_settings) -> None:
    analysis = analyze_travel_turn(
        "两个人去长沙两天，总预算1800元，住宿已经花了600元。",
        build_session(persist=False),
        offline_settings,
    )

    assert "lodging_area" not in analysis.constraint_state
    assert analysis.constraint_state["budget_total_cny"] == 1800.0
    assert analysis.constraint_state["prepaid_lodging_cny"] == 600.0
