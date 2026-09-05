from __future__ import annotations

from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import (
    TaskType,
    build_rule_constraint_state,
    build_rule_patches,
    classify_task_type_rule_based,
)
from travel_agent.agent.turn_lifecycle import prepare_turn
from travel_agent.evaluation.artifact_contract import artifact_content_valid
from travel_agent.orchestration.multi_agent.dispatch_rules import agents_required_for_turn
from travel_agent.orchestration.multi_agent.engine import _materialize_lightweight_artifact
from travel_agent.orchestration.multi_agent.schemas import STATUS_COMPLETED, SubagentResult


def _result(
    ctx,
    kind: str,
    payload: dict,
    *,
    task_id: str = "task",
    agent: str = "attraction",
) -> SubagentResult:
    artifact_id = ctx.store.put(kind, payload, request_id="req", task_id=task_id, agent=agent)
    return SubagentResult(
        request_id="req",
        task_id=task_id,
        agent=agent,
        status=STATUS_COMPLETED,
        evidence=[{"artifact_id": artifact_id, "kind": kind}],
    )


def _state(message: str) -> dict:
    task_type = classify_task_type_rule_based(message)
    return build_rule_constraint_state(
        message,
        build_rule_patches(message, task_type),
    )


def test_route_evidence_materializes_a_route_plan_artifact() -> None:
    ctx = build_session(persist=False)
    route = _result(
        ctx,
        "routes",
        {
            "origin_poi_id": "origin-1",
            "origin_name": "中央车站",
            "destination_poi_id": "destination-1",
            "destination_name": "湖畔酒店区",
            "mode": "public_transport",
            "duration_min": 35,
            "distance_km": 12.0,
            "source": "fixture",
        },
        agent="transport",
    )

    _materialize_lightweight_artifact(
        ctx,
        TaskType.ROUTE_PLAN,
        [route],
        [],
        request_id="req",
    )

    artifact = ctx.store.latest("route_plan")
    assert artifact["origin"] == "中央车站"
    assert artifact["destination"] == "湖畔酒店区"
    assert artifact_content_valid("route_plan", artifact)


def test_two_regions_produce_candidate_artifact_with_exact_requested_candidates() -> None:
    ctx = build_session(persist=False)
    message = "去南京时只比较新街口和夫子庙两个住宿区域，哪个更适合父母？"
    ctx.profile.destination = "南京"
    ctx.profile.constraint_state = _state(message)
    result = _result(
        ctx,
        "candidates",
        {
            "pois": [
                {"name": "新街口无障碍服务点", "tags": ["适老", "无障碍"], "rating": 4.5},
                {"name": "夫子庙无障碍服务点", "tags": ["适老", "无障碍"], "rating": 4.2},
            ]
        },
    )

    _materialize_lightweight_artifact(
        ctx, TaskType.CANDIDATE_COMPARISON, [result], [], request_id="req"
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert [item["name"] for item in artifact["candidates"]] == ["新街口", "夫子庙"]
    assert artifact_content_valid("candidate_comparison", artifact)
    assert set(agents_required_for_turn(
        TaskType.CANDIDATE_COMPARISON,
        task_brief=message,
        inputs={"profile": {"constraint_state": ctx.profile.constraint_state}},
    )) == {"hotel", "transport"}


def test_lodging_downgrade_routes_full_plan_through_hotel_worker() -> None:
    assert "hotel" in agents_required_for_turn(
        TaskType.FULL_ITINERARY,
        task_brief="预算降低，住宿可以降档，请重新生成完整行程",
        inputs={
            "profile": {
                "constraint_state": {"lodging_flexibility": "can_downgrade"}
            }
        },
    )


def test_historical_lodging_downgrade_does_not_retrigger_hotel_worker() -> None:
    assert "hotel" not in agents_required_for_turn(
        TaskType.FULL_ITINERARY,
        task_brief="增加一个室内备选，再生成最终版",
        inputs={
            "profile": {
                "constraint_state": {"lodging_flexibility": "can_downgrade"}
            }
        },
    )


def test_region_to_fixed_anchor_comparison_filters_unrelated_route_evidence() -> None:
    ctx = build_session(persist=False)
    message = "比较老城和新区到中央车站的通勤时间"
    ctx.profile.constraint_state = _state(message)
    good = _result(
        ctx,
        "routes",
        {"origin_name": "老城", "destination_name": "中央车站", "duration_min": 20},
        task_id="good",
        agent="transport",
    )
    second = _result(
        ctx,
        "routes",
        {"origin_name": "新区", "destination_name": "中央车站", "duration_min": 25},
        task_id="second",
        agent="transport",
    )
    bad = _result(
        ctx,
        "routes",
        {"origin_name": "无关公园", "destination_name": "无关商场", "duration_min": 2},
        task_id="bad",
        agent="transport",
    )

    _materialize_lightweight_artifact(
        ctx, TaskType.CANDIDATE_COMPARISON, [good, second, bad], [], request_id="req"
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert [item["name"] for item in artifact["candidates"]] == ["老城", "新区"]
    assert artifact["recommendation"]["candidate"] == "老城"
    assert len(artifact["evidence"]) == 2
    assert artifact_content_valid("candidate_comparison", artifact)
    assert "transport" in agents_required_for_turn(
        TaskType.CANDIDATE_COMPARISON,
        task_brief=message,
        inputs={"profile": {"constraint_state": ctx.profile.constraint_state}},
    )


def test_candidate_comparison_rejects_artifact_id_only_placeholder() -> None:
    ctx = build_session(persist=False)
    message = "比较甲区和乙区哪个氛围更安静"
    ctx.profile.constraint_state = _state(message)
    placeholder = SubagentResult(
        request_id="req",
        task_id="attraction-placeholder",
        agent="attraction",
        status=STATUS_COMPLETED,
        evidence=[{"artifact_id": "candidates_missing", "kind": "candidates"}],
    )

    _materialize_lightweight_artifact(
        ctx,
        TaskType.CANDIDATE_COMPARISON,
        [placeholder],
        [],
        request_id="req",
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert artifact["evidence"] == []
    assert artifact["recommendation"] is None
    assert not artifact_content_valid("candidate_comparison", artifact)


def test_discovery_results_are_promoted_to_grounded_candidate_artifact() -> None:
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {
        "location": "湖畔商务区附近",
        "specific_restaurant_recommendation": True,
        "top_n": 3,
    }
    restaurants = _result(
        ctx,
        "restaurants",
        {
            "restaurants": [
                {"name": "松风餐厅", "category": "food", "rating": 4.7, "average_cost": 90},
                {"name": "青禾小馆", "category": "food", "rating": 4.5, "average_cost": 110},
                {"name": "云水茶膳", "category": "food", "rating": 4.3, "average_cost": 80},
                {"name": "远山饭店", "category": "food", "rating": 4.1, "average_cost": 70},
            ]
        },
        agent="restaurant",
    )

    _materialize_lightweight_artifact(
        ctx,
        TaskType.CANDIDATE_COMPARISON,
        [restaurants],
        [],
        request_id="req",
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert [item["name"] for item in artifact["candidates"]] == [
        "松风餐厅", "青禾小馆", "云水茶膳",
    ]
    assert artifact["recommendation"]["candidate"] == "松风餐厅"
    assert artifact_content_valid("candidate_comparison", artifact)


def test_anchored_dinner_discovery_routes_only_relevant_workers() -> None:
    message = "明天下雨时，在湖畔大道附近找晚餐，步行10分钟内，人均100元"
    state = _state(message)

    assert set(agents_required_for_turn(
        TaskType.CANDIDATE_COMPARISON,
        task_brief=message,
        inputs={"profile": {"constraint_state": state}},
    )) == {"restaurant", "transport"}


def test_discovery_deduplicates_entrances_and_prefers_diverse_categories() -> None:
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {"top_n": 4, "diversity_required": True}
    candidates = _result(
        ctx,
        "candidates",
        {
            "pois": [
                {"name": "古城景区东门", "canonical_name": "古城景区东门", "parent_poi_id": "old-town", "category": "scenic", "rating": 4.8},
                {"name": "古城景区西门", "canonical_name": "古城景区西门", "parent_poi_id": "old-town", "category": "scenic", "rating": 4.7},
                {"name": "城市博物馆", "category": "museum", "rating": 4.6},
                {"name": "滨江公园", "category": "park", "rating": 4.5},
                {"name": "山水湿地", "category": "nature", "rating": 4.4},
            ]
        },
    )

    _materialize_lightweight_artifact(
        ctx,
        TaskType.CANDIDATE_COMPARISON,
        [candidates],
        [],
        request_id="req",
    )

    artifact = ctx.store.latest("candidate_comparison")
    names = [item["name"] for item in artifact["candidates"]]
    assert len(names) == 4
    assert sum("古城景区" in name for name in names) == 1
    assert {"城市博物馆", "滨江公园", "山水湿地"}.issubset(names)
    assert artifact_content_valid("candidate_comparison", artifact)


def test_structured_itinerary_selects_patch_and_materializes_before_after() -> None:
    ctx = build_session(persist=False)
    plan_id = ctx.store.put(
        "itinerary",
        {"itinerary": {"days": [{"day_index": 1}, {"day_index": 2, "stops": ["户外活动"]}]}},
        agent="planner",
    )
    message = "请判断第二天户外活动是否因下雨替换，并给室内备选"
    assert classify_task_type_rule_based(message, ctx) == TaskType.ITINERARY_PATCH
    ctx.profile.constraint_state = _state(message)
    weather = _result(ctx, "weather", {"condition": "大雨"}, task_id="weather")
    candidates = _result(ctx, "candidates", {"pois": [{"name": "室内展馆"}]}, task_id="poi")

    _materialize_lightweight_artifact(
        ctx,
        TaskType.ITINERARY_PATCH,
        [weather, candidates],
        [],
        request_id="req",
        existing_plan_artifact_id=plan_id,
    )

    artifact = ctx.store.latest("itinerary_patch")
    assert artifact["before"]["day_index"] == 2
    assert artifact["after"]["local_adjustment"]["replacement"] == "室内展馆"
    assert artifact["affected_periods"] == [2]
    assert artifact_content_valid("itinerary_patch", artifact)


def test_no_itinerary_but_sufficient_local_context_produces_advice_not_patch() -> None:
    ctx = build_session(persist=False)
    message = "请判断第二天户外活动是否因下雨替换，并给室内备选"
    assert classify_task_type_rule_based(message, ctx) == TaskType.LOCAL_ADJUSTMENT_ADVICE
    ctx.profile.constraint_state = _state(message)
    weather = _result(ctx, "weather", {"condition": "小雨"}, task_id="weather")
    candidates = _result(ctx, "candidates", {"pois": [{"name": "室内展馆"}]}, task_id="poi")

    _materialize_lightweight_artifact(
        ctx, TaskType.LOCAL_ADJUSTMENT_ADVICE, [weather, candidates], [], request_id="req"
    )

    artifact = ctx.store.latest("local_adjustment_advice")
    assert artifact["apply_status"] == "advice_only_no_itinerary_modified"
    assert ctx.store.latest("itinerary_patch") is None
    assert artifact_content_valid("local_adjustment_advice", artifact)


def test_insufficient_local_adjustment_asks_only_for_relevant_fragment(offline_settings) -> None:
    ctx = build_session(persist=False)
    prepared = prepare_turn("我已有行程，不要重写，只调整一下。", ctx, offline_settings, [])
    assert prepared.early_reply is not None
    assert "日期和对应活动" in prepared.early_reply.text
    assert "你想去哪个" not in prepared.early_reply.text
    assert "总天数" in prepared.early_reply.text and "不需要提供" in prepared.early_reply.text


def test_city_and_nearby_anchor_wording_variants_are_separated() -> None:
    for message in (
        "今晚在成都的太古里附近找晚餐",
        "想住成都城里的太古里周边",
        "请找成都市太古里一带的安静餐厅",
    ):
        state = _state(message)
        assert state["destination_city"] == "成都"
        assert state["location_anchor"] == "太古里"
        assert state["location"] == "成都太古里附近"


def test_counterexamples_do_not_collapse_all_requests_to_one_artifact() -> None:
    assert classify_task_type_rule_based("从机场到火车站怎么走，比较地铁和打车") == TaskType.ROUTE_PLAN
    assert classify_task_type_rule_based("请规划完整的三日行程") == TaskType.FULL_ITINERARY
    assert classify_task_type_rule_based("比较甲区和乙区哪个更方便") == TaskType.CANDIDATE_COMPARISON
    assert not artifact_content_valid(
        "candidate_comparison", {"subject": "区域", "candidates": [{"name": "甲区"}]}
    )
