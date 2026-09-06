from __future__ import annotations

from travel_agent.agent.response_summary import (
    build_plan_reply_text,
    looks_like_hallucinated_itinerary,
)
from travel_agent.agent.runtime import _finalize_reply_text, _reply_from_outcome
from travel_agent.agent.session import ArtifactStore, build_session
from travel_agent.orchestration.multi_agent.engine import TurnOutcome
from travel_agent.orchestration.multi_agent.schemas import STATUS_INCOMPLETE
from travel_agent.schemas import TravelProfile


def test_looks_like_hallucinated_itinerary() -> None:
    bad = "⚠️ **Critic 检测到关键问题**\n我将**人工介入修正**\n#### Day 1｜外滩"
    assert looks_like_hallucinated_itinerary(bad)
    assert not looks_like_hallucinated_itinerary("上海3天行程已生成，请看右侧卡片。")


def test_build_plan_reply_text_is_short() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="上海", days=3, interests=["food", "nature"])
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {
                "summary": "上海3天轻松行程",
                "days": [
                    {
                        "day_index": 1,
                        "theme": "外滩",
                        "stops": [{"name": "外滩"}, {"name": "豫园"}],
                    }
                ],
            },
            "critic": {"passed": True},
            "original_issue_count": 1,
            "final_issue_count": 0,
            "revision_notes": ["调整了雨天安排"],
        },
    )
    text = build_plan_reply_text(ctx)
    assert "右侧面板" in text
    assert "第1天" in text
    assert "人工介入" not in text
    assert len(text) < 500


def test_plan_reply_hides_internal_reviewer_instruction() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="成都", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "成都一日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "revision_notes": [
                "已携带 Reviewer 定向要求重新规划：补充上午活动",
                "补充必去地点 `武侯祠`。",
            ],
        },
    )

    text = build_plan_reply_text(ctx)

    assert "Reviewer" not in text
    assert "补充必去地点" in text


def test_plan_reply_does_not_claim_delivery_when_budget_is_exceeded() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=3)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "杭州三日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "budget_plan": {
                "user_limit_cny": 3000,
                "total_expected_cny": 3240,
                "total_low_cny": 2800,
                "total_high_cny": 3700,
                "within_user_limit": False,
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "尚不可交付" in text
    assert "已排好" not in text


def test_no_current_artifact_never_claims_sidebar_or_map() -> None:
    ctx = build_session(persist=False)
    rejected = ctx.store.put(
        "itinerary",
        {
            "artifact_status": "validation_failure",
            "itinerary": {"summary": "失败候选", "days": []},
            "critic": {"passed": False, "issues": []},
        },
    )

    text = build_plan_reply_text(ctx, rejected)

    assert "当前没有" in text
    assert "右侧面板" not in text
    assert "地图" not in text
    assert "已排好" not in text


def test_rejected_candidate_discards_contradictory_reviewer_or_planner_text() -> None:
    ctx = build_session(persist=False)
    rejected = ctx.store.put(
        "itinerary",
        {
            "artifact_status": "validation_failure",
            "itinerary": {"summary": "失败候选", "days": []},
            "critic": {"passed": False, "issues": []},
        },
        agent="planner",
    )
    outcome = TurnOutcome(
        status=STATUS_INCOMPLETE,
        reply="行程已排好，地图路线请查看右侧面板。",
        plan_artifact_id=rejected,
        cards=[{"type": "summary"}],
        map_payload={"routes": []},
    )

    reply = _reply_from_outcome(ctx, outcome)

    assert "尚不可交付" in reply.text
    assert "已排好" not in reply.text
    assert "右侧面板" not in reply.text
    assert reply.cards == [] and reply.map_payload is None


def test_estimated_route_is_never_described_as_map_verified() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {
                "summary": "测试城一日游",
                "days": [{
                    "day_index": 1,
                    "stops": [{
                        "poi": {
                            "poi_id": "p1",
                            "name": "测试馆",
                            "source": "provider",
                            "verification_status": "verified",
                        },
                        "route_from_previous": {
                            "origin_poi_id": "origin",
                            "destination_poi_id": "p1",
                            "duration_min": 20,
                            "distance_km": 3.0,
                            "source": "deterministic_speed_fallback",
                            "evidence_status": "deterministic_estimate",
                        },
                    }],
                }],
            },
            "critic": {"passed": True, "issues": []},
        },
    )

    text = build_plan_reply_text(ctx)

    assert "路线为估算结果" in text
    assert "已核验地图路线" not in text


def test_plan_reply_uses_actual_non_overlapping_meal_strategy() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {
                "summary": "测试城一日游",
                "days": [{
                    "day_index": 1,
                    "stops": [
                        {"name": "甲馆", "start_time": "09:00", "duration_min": 120},
                        {"name": "乙园", "start_time": "12:00", "duration_min": 150},
                        {"name": "丙街", "start_time": "15:00", "duration_min": 105},
                    ],
                }],
            },
            "critic": {"passed": True, "issues": []},
            "meal_strategy": {
                "scheduled_meals": [{
                    "day_index": 1,
                    "name": "晚餐时段（当日活动区域就近自行安排）",
                    "start_time": "17:30",
                    "end_time": "18:30",
                    "is_reservation_only": True,
                }],
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "17:30–18:30" in text
    assert "12:00–13:00" not in text


def test_return_plan_uses_the_actual_trip_city_in_reply() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="南京", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "南京一日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "return_plan": {
                "required": True,
                "from_city": "南京",
                "to_location": "南京南站",
                "activity_cutoff": "18:00",
                "arrival_deadline": "19:00",
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "结束南京活动" in text
    assert "结束杭州活动" not in text


def test_finalize_prefers_artifact_over_llm() -> None:
    ctx = build_session(persist=False)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "杭州3天", "days": []},
            "critic": {"passed": True},
            "original_issue_count": 0,
            "final_issue_count": 0,
        },
    )
    llm = "#### Day 1｜" + "x" * 800
    out = _finalize_reply_text(ctx, ["plan_and_critique"], llm)
    assert "Day 1｜" not in out
    assert "右侧面板" in out or "杭州" in out


def test_plan_reply_surfaces_structured_return_plan_without_inventing_train() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "杭州一日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "return_plan": {
                "required": True,
                "activity_cutoff": "18:30",
                "arrival_deadline": "21:30",
                "to_location": "上海",
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "18:30" in text and "21:30" in text and "上海" in text
    assert "实时车次" in text


def test_plan_reply_surfaces_selected_lodging_and_budget_limit() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="北京", days=2)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "北京两日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "lodging_plan": {
                "status": "recommended_not_booked",
                "hotel": {"name": "王府井大饭店"},
                "rooms": 1,
                "nights": 1,
                "lodging_subtotal_cny": 585,
            },
            "budget_plan": {
                "total_low_cny": 1675.5,
                "total_expected_cny": 1965,
                "total_high_cny": 2254.5,
                "user_limit_cny": 4500,
                "within_user_limit": True,
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "住宿推荐（未预订）：王府井大饭店" in text
    assert "不超过上限 ¥4500" in text
