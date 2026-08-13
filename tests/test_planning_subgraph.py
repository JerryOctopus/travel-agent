from __future__ import annotations

from pathlib import Path

from travel_agent.planning import (
    _daily_opening_window,
    apply_structured_schedule_constraints,
    build_simple_itinerary,
)
from travel_agent.schemas import ItineraryDay, ItineraryStop, POI, ScoredPOI
from travel_agent.planning_subgraph import plan_and_critique
from travel_agent.providers import build_tool_provider
from travel_agent.recommendation import score_pois
from travel_agent.schemas import TravelProfile

POI_PATH = Path(__file__).resolve().parents[1] / "data" / "seed" / "pois.json"


def _ranked(profile: TravelProfile):
    provider = build_tool_provider(POI_PATH)
    candidates = provider.search_pois(city=profile.destination or "", max_results=50)
    return score_pois(candidates, profile), provider


def test_plan_and_critique_outputs_itinerary():
    profile = TravelProfile(destination="杭州", days=2, interests=["nature", "food"])
    ranked, provider = _ranked(profile)
    result = plan_and_critique(ranked, profile, route_estimator=provider)

    assert result.itinerary.city == "杭州"
    assert len(result.itinerary.days) == 2
    assert result.final_issue_count <= result.original_issue_count


def test_plan_and_critique_closes_the_loop():
    # 排序或 reviser 均可覆盖博物馆兴趣；不要求为了测试而先生成坏初稿。
    profile = TravelProfile(destination="杭州", days=1, interests=["museum"])
    ranked, provider = _ranked(profile)
    result = plan_and_critique(ranked, profile, route_estimator=provider)

    assert result.critic_result.passed is True
    assert "interest_not_covered" not in {
        issue.code for issue in result.critic_result.issues
    }


def test_plan_and_critique_handles_multi_interest_single_day():
    profile = TravelProfile(
        destination="杭州",
        days=1,
        interests=["history", "food", "nature", "museum"],
    )
    ranked, provider = _ranked(profile)
    result = plan_and_critique(ranked, profile, route_estimator=provider)

    assert result.critic_result.passed is True
    assert "interest_not_covered" not in {
        issue.code for issue in result.critic_result.issues
    }


def test_candidate_selection_reserves_all_evidenced_must_visits_before_meal():
    profile = TravelProfile(
        destination="杭州",
        days=1,
        must_visit=["西湖", "中国茶叶博物馆"],
    )
    ranked, provider = _ranked(profile)

    itinerary = build_simple_itinerary(
        ranked, profile, provider, preserve_must_visit_capacity=True
    )
    names = [stop.poi.name for stop in itinerary.days[0].stops]

    assert any("西湖" in name for name in names)
    assert any("中国茶叶博物馆" in name for name in names)


def test_candidate_selection_prefers_meal_near_hard_must_visit():
    must = POI(
        poi_id="west-lake",
        name="西湖",
        city="杭州",
        category="scenic",
        lat=30.25,
        lng=120.16,
        rating=4.8,
        popularity=1.0,
        tags=["nature"],
        estimated_duration_min=90,
        price_level="free",
    )
    near_food = POI(
        poi_id="near-food",
        name="西湖边餐厅",
        city="杭州",
        category="food",
        lat=30.251,
        lng=120.161,
        rating=4.2,
        popularity=0.6,
        tags=["food"],
        estimated_duration_min=60,
        price_level="mid",
    )
    far_food = POI(
        poi_id="far-food",
        name="高分远郊餐厅",
        city="杭州",
        category="food",
        lat=30.8,
        lng=120.8,
        rating=5.0,
        popularity=1.0,
        tags=["food"],
        estimated_duration_min=60,
        price_level="mid",
    )
    museum = POI(
        poi_id="museum",
        name="西湖博物馆",
        city="杭州",
        category="museum",
        lat=30.26,
        lng=120.17,
        rating=4.6,
        popularity=0.8,
        tags=["museum"],
        estimated_duration_min=60,
        price_level="free",
    )
    profile = TravelProfile(destination="杭州", days=1, must_visit=["西湖"])
    ranked = [
        ScoredPOI(must, 1.0, []),
        ScoredPOI(far_food, 0.99, []),
        ScoredPOI(near_food, 0.5, []),
        ScoredPOI(museum, 0.4, []),
    ]

    itinerary = build_simple_itinerary(
        ranked, profile, preserve_must_visit_capacity=True
    )

    names = [stop.poi.name for stop in itinerary.days[0].stops]
    assert "西湖边餐厅" in names
    assert "高分远郊餐厅" not in names


def test_daily_opening_window_parses_explicit_hours_only():
    assert _daily_opening_window("周一至周日 09:00-16:30") == (540, 990)
    assert _daily_opening_window("00:00-24:00") is None
    assert _daily_opening_window("all_day") is None


def test_structured_schedule_sorts_by_known_opening_time_before_retiming() -> None:
    evening = POI(
        "night", "大唐不夜城", "西安", "nightlife", 34.214, 108.966,
        4.5, 0.8, ["culture"], 120, "free", opening_hours="18:00-23:00",
    )
    meal = POI(
        "meal", "清真午餐", "西安", "food", 34.22, 108.96,
        4.5, 0.8, ["food", "halal"], 60, "mid", opening_hours="09:00-22:00",
    )
    days = [ItineraryDay(1, "test", [
        ItineraryStop(evening, "09:30", 120, ""),
        ItineraryStop(meal, "11:30", 60, ""),
    ])]
    profile = TravelProfile(destination="西安", days=1)

    scheduled = apply_structured_schedule_constraints(days, [], profile, None)

    assert [stop.poi.name for stop in scheduled[0].stops] == ["清真午餐", "大唐不夜城"]
    assert scheduled[0].stops[0].start_time == "11:30"
    assert scheduled[0].stops[1].start_time >= "18:00"


def test_cross_city_return_deadline_reserves_terminal_buffer() -> None:
    poi = POI(
        poi_id="west-lake",
        name="西湖",
        city="杭州",
        category="scenic",
        lat=30.25,
        lng=120.16,
        rating=4.8,
        popularity=1.0,
        tags=["nature"],
        estimated_duration_min=90,
        price_level="free",
    )
    late = ItineraryStop(poi=poi, start_time="19:45", duration_min=90, note="test")
    profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"return_deadline": "21:30", "return_location": "上海"},
    )

    days = apply_structured_schedule_constraints(
        [ItineraryDay(day_index=1, theme="test", stops=[late])],
        [ScoredPOI(poi=poi, score=1.0, reasons=[])],
        profile,
        None,
    )

    assert days[0].stops == []


def test_non_actionable_return_warning_does_not_reschedule_valid_stops() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        must_visit=["西湖"],
        constraint_state={"return_deadline": "21:30", "return_location": "上海"},
    )
    ranked, provider = _ranked(profile)

    result = plan_and_critique(ranked, profile, route_estimator=provider)

    assert result.iterations == 0
    assert len(result.itinerary.days[0].stops) >= 2
    assert "return_leg_requires_verification" in {
        issue.code for issue in result.critic_result.issues
    }


def test_lunch_can_shift_to_1230_instead_of_being_pushed_to_dinner() -> None:
    scenic = POI(
        poi_id="scenic",
        name="上午景点",
        city="苏州",
        category="scenic",
        lat=31.31,
        lng=120.60,
        rating=4.8,
        popularity=1.0,
        tags=["classic"],
        estimated_duration_min=150,
        price_level="mid",
    )
    lunch = POI(
        poi_id="lunch",
        name="附近午餐",
        city="苏州",
        category="food",
        lat=31.32,
        lng=120.61,
        rating=4.7,
        popularity=0.9,
        tags=["food"],
        estimated_duration_min=60,
        price_level="mid",
    )
    afternoon = POI(
        poi_id="garden",
        name="下午园林",
        city="苏州",
        category="scenic",
        lat=31.33,
        lng=120.62,
        rating=4.6,
        popularity=0.8,
        tags=["garden"],
        estimated_duration_min=120,
        price_level="mid",
        opening_hours="07:30-17:30",
    )
    profile = TravelProfile(
        destination="苏州",
        days=1,
        constraint_state={"activity_end_deadline": "17:30"},
    )

    itinerary = build_simple_itinerary(
        [
            ScoredPOI(scenic, 1.0, []),
            ScoredPOI(lunch, 0.9, []),
            ScoredPOI(afternoon, 0.8, []),
        ],
        profile,
    )

    assert [(stop.poi.category, stop.start_time) for stop in itinerary.days[0].stops] == [
        ("scenic", "09:30"),
        ("food", "12:30"),
        ("scenic", "14:30"),
    ]


def test_distant_hard_must_visits_are_spread_across_days() -> None:
    def poi(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="西安",
            category="museum",
            lat=34.3,
            lng=lng,
            rating=4.8,
            popularity=0.9,
            tags=["history"],
            estimated_duration_min=90,
            price_level="mid",
        )

    wall = poi("wall", "西安城墙", 108.94)
    warriors = poi("warriors", "秦始皇兵马俑博物馆", 109.28)
    optional = [poi(f"p{index}", f"普通景点{index}", 108.90 + index * 0.03) for index in range(4)]
    profile = TravelProfile(
        destination="西安",
        days=3,
        must_visit=["西安城墙", "兵马俑"],
    )
    ranked = [ScoredPOI(item, 1.0 - index / 10, []) for index, item in enumerate([wall, warriors, *optional])]

    itinerary = build_simple_itinerary(
        ranked,
        profile,
        preserve_must_visit_capacity=True,
    )

    anchor_days = {
        day.day_index
        for day in itinerary.days
        if any(stop.poi.poi_id in {"wall", "warriors"} for stop in day.stops)
    }
    assert len(anchor_days) == 2
