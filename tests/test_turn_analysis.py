"""轮次分析测试：task_type 识别、动态必填槽位、复杂信号与 LLM 兜底校验。"""

from __future__ import annotations

from types import SimpleNamespace

from travel_agent.agent.runtime import _missing_required_slots, run_production_turn
from travel_agent.agent.intent import MessageKind
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import (
    REQUIRED_SLOTS,
    TaskType,
    TurnAnalysis,
    _merge_llm_payload,
    analyze_travel_turn,
    build_rule_patches,
    classify_task_type_rule_based,
    has_complex_signals,
    _analyze_with_llm,
    TURN_ANALYSIS_MAX_OUTPUT_TOKENS,
    TURN_ANALYSIS_TIMEOUT_SECONDS,
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


def test_unknown_request_does_not_default_to_full_trip_plan() -> None:
    assert classify_task_type_rule_based("帮我看看") == TaskType.UNKNOWN
    assert classify_task_type_rule_based("帮我预订机票") == TaskType.UNKNOWN


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

    assert analysis.task_type == TaskType.ITINERARY_REVISION
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
    assert analysis.task_type == TaskType.ITINERARY_REVISION


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


def test_last_day_deadline_is_revision_not_one_day_trip(offline_settings) -> None:
    ctx = _ctx_with_itinerary()
    ctx.profile.days = 5

    analysis = analyze_travel_turn("最后一天17点前到杭州东站。", ctx, offline_settings)

    assert analysis.task_type == TaskType.ITINERARY_REVISION
    assert "duration_days" not in analysis.constraint_state
    assert "days" not in analysis.patches
    assert analysis.constraint_state["return_deadline"] == "17:00"
    assert analysis.constraint_state["return_location"] == "杭州东站"


def test_final_version_followup_stays_in_travel_chain(offline_settings) -> None:
    ctx = _ctx_with_itinerary()

    analysis = analyze_travel_turn(
        "按全部条件出最终版，别恢复已删除的地点。",
        ctx,
        offline_settings,
    )

    assert analysis.kind == MessageKind.TRAVEL
    assert analysis.task_type == TaskType.ITINERARY_REVISION


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


def test_route_query_run_turn_does_not_clarify_days(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("虹桥机场到外滩怎么走", ctx=ctx, settings=offline_settings)
    assert reply.clarification is False
    assert reply.failure_reason != "missing_slot"


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
