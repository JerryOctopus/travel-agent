"""36 non-frozen synthetic release-readiness cases (9 families x 4 variants)."""

from __future__ import annotations

import pytest

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType, classify_task_type_rule_based
from travel_agent.constraint_events import merge_constraint_update
from travel_agent.critic import critique_itinerary
from travel_agent.plan_invariants import validate_plan_artifact
from travel_agent.poi_evidence import poi_avoid_match
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    TravelProfile,
)


READINESS_CASES = [
    *(('intercity_evidence', index) for index in range(4)),
    *(('accessibility_fail_closed', index) for index in range(4)),
    *(('daily_dietary_coverage', index) for index in range(4)),
    *(('return_route_closure', index) for index in range(4)),
    *(('ferry_queue_transfer', index) for index in range(4)),
    *(('fixed_appointment_meal_conflict', index) for index in range(4)),
    *(('artifact_routing', index) for index in range(4)),
    *(('budget_feasibility', index) for index in range(4)),
    *(('state_overwrite', index) for index in range(4)),
]


@pytest.mark.parametrize(
    ("family", "variant"),
    READINESS_CASES,
    ids=[f"rr_{index + 1:02d}_{family}_v{variant + 1}" for index, (family, variant) in enumerate(READINESS_CASES)],
)
def test_release_readiness_case(family: str, variant: int) -> None:
    evaluators = {
        "intercity_evidence": _intercity_evidence,
        "accessibility_fail_closed": _accessibility_fail_closed,
        "daily_dietary_coverage": _daily_dietary_coverage,
        "return_route_closure": _return_route_closure,
        "ferry_queue_transfer": _ferry_queue_transfer,
        "fixed_appointment_meal_conflict": _fixed_appointment_meal_conflict,
        "artifact_routing": _artifact_routing,
        "budget_feasibility": _budget_feasibility,
        "state_overwrite": _state_overwrite,
    }
    evaluators[family](variant)


def test_readiness_contract_is_exactly_nine_families_and_36_cases() -> None:
    assert len(READINESS_CASES) == 36
    assert {family for family, _ in READINESS_CASES} == {
        "intercity_evidence",
        "accessibility_fail_closed",
        "daily_dietary_coverage",
        "return_route_closure",
        "ferry_queue_transfer",
        "fixed_appointment_meal_conflict",
        "artifact_routing",
        "budget_feasibility",
        "state_overwrite",
    }
    assert all(sum(item[0] == family for item in READINESS_CASES) == 4 for family in {item[0] for item in READINESS_CASES})


def _poi(poi_id: str, name: str, category: str = "scenic", *, popularity: float = 0.8, tags: list[str] | None = None) -> POI:
    return POI(
        poi_id, name, "合成城", category, 30.0, 120.0, 4.5,
        popularity, list(tags or []), 90, "mid", source="synthetic_fixture",
        entity_type="restaurant" if category == "food" else "attraction",
        verification_status="verified",
    )


def _intercity_evidence(variant: int) -> None:
    deadline = ("20:30", "21:00", "21:30", "22:00")[variant]
    profile = TravelProfile(
        destination="甲城",
        days=1,
        constraint_state={"return_location": "乙城中央站", "return_deadline": deadline},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [{"poi": {"poi_id": "last"}}]}]}
    terminal = {
        "origin_poi_id": "last", "destination_poi_id": "local-station",
        "origin_name": "末站", "destination_name": "甲城东站", "duration_min": 35,
        "source": "synthetic_provider", "evidence_status": "provider_verified",
    }
    intercity = {
        "origin_poi_id": "local-station", "destination_poi_id": "remote-station",
        "origin_name": "甲城东站", "destination_name": "乙城中央站", "duration_min": 75 + variant * 5,
        "source": "synthetic_provider", "evidence_status": "provider_verified",
    }
    result = toolkit._build_return_plan(
        profile,
        itinerary,
        {"transport": [{"payload": terminal}, {"payload": intercity}]},
    )
    assert result is not None
    assert result["terminal_transfer"] == terminal
    assert result["intercity_segment"]["status"] == "verified_route"


def _accessibility_fail_closed(variant: int) -> None:
    stop = ItineraryStop(_poi(f"p{variant}", f"普通景点{variant}"), "09:30", 90, "")
    state = {"wheelchair_user": True} if variant % 2 == 0 else {"accessibility_priority": True}
    profile = TravelProfile(destination="合成城", days=1, constraint_state=state)
    result = critique_itinerary(Itinerary("合成城", [ItineraryDay(1, "", [stop])], ""), profile)
    assert "accessibility_evidence_missing" in {issue.code for issue in result.issues}


def _daily_dietary_coverage(variant: int) -> None:
    rules = ("仅清真餐厅", "严格素食", "不吃海鲜", "无麸质")[variant]
    meal = _poi(f"meal{variant}", f"合规餐厅{variant}", "food", tags=[rules, "清真", "halal"])
    scenic = _poi(f"s{variant}", f"普通景点{variant}")
    itinerary = Itinerary(
        "合成城",
        [
            ItineraryDay(1, "", [ItineraryStop(meal, "12:00", 60, "")]),
            ItineraryDay(2, "", [ItineraryStop(scenic, "09:30", 90, "")]),
        ],
        "",
    )
    profile = TravelProfile(
        destination="合成城",
        days=2,
        interests=["food"],
        constraint_state={"dietary": [rules]},
    )
    result = critique_itinerary(itinerary, profile)
    assert "daily_meal_missing" in {issue.code for issue in result.issues}


def _return_route_closure(variant: int) -> None:
    profile = TravelProfile(
        destination="合成城",
        days=1,
        constraint_state={
            "origin": f"出发枢纽{variant}",
            "return_location": f"返程枢纽{variant}",
            "return_deadline": f"{17 + variant}:00",
        },
    )
    payload = {
        "itinerary": {"days": [{"day_index": 1, "stops": [{"poi": {
            "poi_id": "last", "name": "末站", "source": "synthetic_provider",
            "verification_status": "verified",
        }}]}]},
        "required_route_anchors": {"legs": []},
    }
    result = validate_plan_artifact(payload, profile)
    codes = {issue["code"] for issue in result["issues"]}
    assert "trip_origin_route_missing" in codes
    assert "return_route_missing" in codes


def _ferry_queue_transfer(variant: int) -> None:
    start = ("11:30", "11:45", "12:00", "12:15")[variant]
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": start,
        "duration_min": 90,
        "poi": {"category": "transport", "name": f"轮渡码头活动{variant}"},
    }]}]}
    strategy = toolkit._build_meal_strategy(TravelProfile(destination="合成城", days=1), itinerary, {})
    for meal in strategy["scheduled_meals"]:
        meal_start = toolkit._clock_to_minute(meal["start_time"])
        occupied_start = toolkit._clock_to_minute(start)
        assert meal_start is not None and occupied_start is not None
        assert toolkit._meal_window_fits(meal_start, [(occupied_start, occupied_start + 90)])
    queue_terms = ("不去排队很久的地方", "避开网红点", "避免长时间排队", "不要热门排队景点")
    risky = _poi(f"hot{variant}", f"高热度轮渡点{variant}", popularity=0.95 + variant * 0.01)
    queue_profile = TravelProfile(destination="合成城", constraint_state={"avoid": [queue_terms[variant]]})
    assert poi_avoid_match(risky, queue_profile) is not None


def _fixed_appointment_meal_conflict(variant: int) -> None:
    start = ("11:30", "11:45", "12:00", "12:15")[variant]
    profile = TravelProfile(
        destination="合成城",
        days=1,
        constraint_state={"fixed_events": [{
            "day": 1,
            "start": start,
            "end": toolkit._offset_clock(start, 120),
            "location": f"固定预约场馆{variant}",
        }]},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": start,
        "duration_min": 120,
        "poi": {"category": "museum", "name": f"固定预约场馆{variant}"},
    }]}]}
    strategy = toolkit._build_meal_strategy(profile, itinerary, {})
    occupied_start = toolkit._clock_to_minute(start)
    assert occupied_start is not None
    assert all(
        toolkit._meal_window_fits(
            toolkit._clock_to_minute(meal["start_time"]),
            [(occupied_start, occupied_start + 120)],
        )
        for meal in strategy["scheduled_meals"]
    )


def test_meal_reservation_keeps_time_for_route_to_next_stop() -> None:
    day = {
        "day_index": 1,
        "stops": [
            {"start_time": "09:00", "duration_min": 150},
            {
                "start_time": "13:00",
                "duration_min": 120,
                "route_from_previous": {"duration_min": 59},
            },
        ],
    }

    assert toolkit._available_meal_reservation(day) == ("17:30", "18:30")


def _artifact_routing(variant: int) -> None:
    prompts = (
        ("只规划机场到酒店的公交路线，不推荐景点", TaskType.ROUTE_PLAN),
        ("比较这三个候选活动，给出推荐理由", TaskType.CANDIDATE_COMPARISON),
        ("原行程不变，只给附近室内备选建议", TaskType.LOCAL_ADJUSTMENT_ADVICE),
        ("安排合成城三天完整行程", TaskType.FULL_ITINERARY),
    )
    prompt, expected = prompts[variant]
    assert classify_task_type_rule_based(prompt) == expected


def _budget_feasibility(variant: int) -> None:
    limit = (900, 1200, 1800, 2400)[variant]
    profile = TravelProfile(
        destination="合成城",
        days=2,
        party_size=2,
        budget_limit=limit,
        constraint_state={"budget_max_cny": limit, "lodging_flexibility": "can_downgrade"},
    )
    estimate = {
        "days": 2, "companions": 2, "hotel": 1000, "meals": 500,
        "tickets": 300, "inner_city_transport": 200, "total_low": 1800, "total_high": 2400,
    }
    result = toolkit._build_budget_plan(profile, {"budgets": [{"payload": estimate}]}, None)
    assert result["user_limit_cny"] == limit
    assert result["within_user_limit"] is (result["expected_total"] <= limit)


def _state_overwrite(variant: int) -> None:
    place = f"候选地点{variant}"
    state: dict = {}
    merge_constraint_update(state, {"budget_max_cny": 3000 + variant * 100, "must_visit": [place]}, source_turn=1)
    merge_constraint_update(state, {"budget_max_cny": 2200 + variant * 100, "removed": [place]}, source_turn=2)
    assert state["budget_max_cny"] == 2200 + variant * 100
    assert place in state["removed"]
    assert place not in state.get("must_visit", [])
    assert any(event["operation"] == "remove" for event in state["_constraint_events"])
