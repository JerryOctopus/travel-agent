from __future__ import annotations

from travel_agent.planning import build_simple_itinerary
from travel_agent.critic import critique_itinerary
from dataclasses import replace

from travel_agent.schemas import POI, RouteInfo, ScoredPOI, TravelProfile


def _poi(
    poi_id: str,
    lng: float,
    lat: float,
    category: str = "scenic",
    tags: list[str] | None = None,
) -> ScoredPOI:
    return ScoredPOI(
        poi=POI(
            poi_id=poi_id,
            name=poi_id,
            city="杭州",
            category=category,
            lat=lat,
            lng=lng,
            rating=4.5,
            popularity=0.8,
            tags=tags or ["nature"],
            estimated_duration_min=60 if category == "food" else 90,
            price_level="mid",
        ),
        score=1.0,
        reasons=["test"],
    )


def test_geo_cluster_groups_nearby_pois_on_same_day():
    profile = TravelProfile(destination="杭州", days=2, pace="standard")
    ranked = [
        _poi("west-1", 120.10, 30.20),
        _poi("west-2", 120.11, 30.21),
        _poi("east-1", 120.30, 30.25),
        _poi("east-2", 120.31, 30.26),
    ]

    itinerary = build_simple_itinerary(ranked, profile)

    day1_ids = {stop.poi.poi_id for stop in itinerary.days[0].stops}
    day2_ids = {stop.poi.poi_id for stop in itinerary.days[1].stops}

    assert {"west-1", "west-2"}.issubset(day1_ids) or {"west-1", "west-2"}.issubset(day2_ids)
    assert {"east-1", "east-2"}.issubset(day1_ids) or {"east-1", "east-2"}.issubset(day2_ids)
    assert day1_ids.isdisjoint(day2_ids)


def test_food_pois_are_scheduled_at_meal_times() -> None:
    profile = TravelProfile(destination="杭州", days=1, pace="standard", interests=["food"])
    ranked = [
        _poi("breakfast-wrong", 120.10, 30.20, category="food", tags=["food"]),
        _poi("lunch-or-dinner", 120.11, 30.21, category="food", tags=["food"]),
        _poi("west-lake", 120.12, 30.22, category="scenic", tags=["nature"]),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    food_times = [
        stop.start_time
        for stop in itinerary.days[0].stops
        if stop.poi.category == "food"
    ]

    assert "09:30" not in food_times
    assert food_times == ["11:30", "18:00"]


def test_meal_precedes_evening_only_activity() -> None:
    profile = TravelProfile(destination="西安", days=1, pace="standard")
    meal = _poi("清真晚餐", 108.96, 34.22, category="food", tags=["food", "halal"])
    evening = _poi("大唐不夜城", 108.966, 34.214, category="nightlife", tags=["culture"])
    evening = replace(evening, poi=replace(evening.poi, opening_hours="18:00-23:00"))

    itinerary = build_simple_itinerary([evening, meal], profile)

    assert [stop.start_time for stop in itinerary.days[0].stops] == ["17:30", "18:45"]
    assert [stop.poi.name for stop in itinerary.days[0].stops] == ["清真晚餐", "大唐不夜城"]


def test_complete_plan_reserves_one_evidenced_meal_per_day() -> None:
    profile = TravelProfile(destination="杭州", days=2, pace="standard")
    ranked = [
        *[_poi(f"sight-{index}", 120.10 + index / 100, 30.20) for index in range(8)],
        _poi("meal-1", 120.25, 30.22, category="food", tags=["food"]),
        _poi("meal-2", 120.26, 30.23, category="food", tags=["food"]),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    meals = [
        stop
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.category == "food"
    ]

    assert len(meals) == 2
    assert all(
        sum(stop.poi.category == "food" for stop in day.stops) == 1
        for day in itinerary.days
    )


def test_reserved_meals_prefer_distinct_brands() -> None:
    profile = TravelProfile(destination="北京", days=2, pace="standard")
    ranked = [
        *[_poi(f"sight-{index}", 116.30 + index / 100, 39.90) for index in range(4)],
        _poi("四季民福烤鸭店(东安门店)", 116.40, 39.91, category="food"),
        _poi("四季民福烤鸭店(灯市口店)", 116.41, 39.92, category="food"),
        _poi("护国寺小吃(王府井店)", 116.42, 39.93, category="food"),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    food_names = [
        stop.poi.name
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.category == "food"
    ]

    assert food_names == ["四季民福烤鸭店(东安门店)", "护国寺小吃(王府井店)"]


def test_known_forbidden_city_route_uses_gate_to_north_order() -> None:
    profile = TravelProfile(destination="北京", days=1, pace="intensive")
    ranked = [
        _poi("景山公园", 116.397, 39.925),
        _poi("故宫博物院", 116.397, 39.916, category="museum"),
        _poi("天安门广场", 116.397, 39.905),
        _poi("午餐", 116.398, 39.910, category="food", tags=["food"]),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    activity_names = [
        stop.poi.name for stop in itinerary.days[0].stops if stop.poi.category != "food"
    ]

    assert activity_names == ["天安门广场", "故宫博物院", "景山公园"]


def test_dietary_constraint_reserves_verified_meal_capacity() -> None:
    profile = TravelProfile(
        destination="西安",
        days=2,
        pace="standard",
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    ranked = [
        *[_poi(f"sight-{index}", 120.10 + index / 100, 30.20) for index in range(8)],
        _poi("halal-1", 120.25, 30.22, category="food", tags=["food", "halal"]),
        _poi("halal-2", 120.26, 30.23, category="food", tags=["food", "halal"]),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    food_ids = {
        stop.poi.poi_id
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.category == "food"
    }

    assert food_ids == {"halal-1", "halal-2"}


def test_dietary_constraint_excludes_unverified_food_candidates() -> None:
    profile = TravelProfile(
        destination="西安",
        days=1,
        pace="standard",
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    ranked = [
        _poi("ordinary-food", 120.10, 30.20, category="food", tags=["food"]),
        _poi("halal-food", 120.11, 30.21, category="food", tags=["food", "halal"]),
        _poi("sight", 120.12, 30.22),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    food_ids = {
        stop.poi.poi_id
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.category == "food"
    }

    assert food_ids == {"halal-food"}


def test_activity_start_times_do_not_duplicate_when_day_is_full() -> None:
    profile = TravelProfile(destination="杭州", days=1, pace="intensive")
    ranked = [
        _poi("stop-1", 120.10, 30.20),
        _poi("stop-2", 120.11, 30.21),
        _poi("stop-3", 120.12, 30.22),
        _poi("stop-4", 120.13, 30.23),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    start_times = [stop.start_time for stop in itinerary.days[0].stops]

    assert len(start_times) == 4
    assert len(set(start_times)) == 4
    assert start_times == sorted(start_times)


def test_transfer_overflow_drops_stop_instead_of_clamping_to_2359() -> None:
    class VerySlowRoutes:
        def estimate_route(self, origin, destination, mode="public_transport"):
            return RouteInfo(
                origin_poi_id=origin.poi_id,
                destination_poi_id=destination.poi_id,
                distance_km=60,
                duration_min=900,
                mode=mode,
                source="test",
            )

    profile = TravelProfile(destination="杭州", days=1, pace="standard")
    itinerary = build_simple_itinerary(
        [_poi("first", 120.1, 30.2), _poi("second", 121.1, 31.2)],
        profile,
        route_estimator=VerySlowRoutes(),
    )

    assert all(stop.start_time != "23:59" for stop in itinerary.days[0].stops)
    assert len(itinerary.days[0].stops) == 1


def test_fixed_event_is_moved_to_literal_day_and_time() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=3,
        pace="standard",
        constraint_state={
            "fixed_events": [
                {"day": 2, "start": "14:30", "end": "16:30", "location": "历史博物馆"}
            ]
        },
    )
    ranked = [
        _poi("stop-1", 120.10, 30.20),
        _poi("stop-2", 120.11, 30.21),
        _poi("历史博物馆", 120.12, 30.22, category="museum"),
        _poi("stop-4", 120.13, 30.23),
        _poi("stop-5", 120.14, 30.24),
        _poi("stop-6", 120.15, 30.25),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    matches = [
        (day.day_index, stop.start_time, stop.duration_min)
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.name == "历史博物馆"
    ]

    assert matches == [(2, "14:30", 120)]
    assert all(
        issue.code != "fixed_event_missing_or_conflicting"
        for issue in critique_itinerary(itinerary, profile).issues
    )


def test_activity_deadline_drops_late_flexible_stops() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        pace="intensive",
        constraint_state={"activity_end_deadline": "17:30"},
    )
    ranked = [_poi(f"stop-{index}", 120.10 + index / 100, 30.20) for index in range(4)]

    itinerary = build_simple_itinerary(ranked, profile)

    assert all(
        int(stop.start_time[:2]) * 60 + int(stop.start_time[3:]) + stop.duration_min <= 17 * 60 + 30
        for stop in itinerary.days[0].stops
    )
    assert critique_itinerary(itinerary, profile).passed is True


def test_deadline_prioritizes_must_visit_stops() -> None:
    profile = TravelProfile(
        destination="上海",
        days=1,
        pace="standard",
        must_visit=["外滩", "豫园"],
        constraint_state={"activity_end_deadline": "18:00"},
    )
    ranked = [
        _poi("上海豫园", 121.49, 31.23),
        _poi("南京路步行街", 121.47, 31.23, category="shopping"),
        _poi("外滩", 121.50, 31.24),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    names = [stop.poi.name for stop in itinerary.days[0].stops]

    assert "上海豫园" in names
    assert "外滩" in names
    assert critique_itinerary(itinerary, profile).passed is True


def test_restaurant_area_name_does_not_satisfy_must_visit() -> None:
    profile = TravelProfile(destination="上海", days=1, must_visit=["豫园"])
    ranked = [
        _poi("外滩家宴·上海菜(外滩豫园店)", 121.49, 31.23, category="food"),
        _poi("外滩", 121.50, 31.24),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    result = critique_itinerary(itinerary, profile)

    assert any(issue.code == "must_visit_missing" for issue in result.issues)


def test_named_candidates_reserve_verified_open_venues_not_closed_museum() -> None:
    profile = TravelProfile(
        destination="天津",
        days=1,
        constraint_state={
            "candidate_attractions": ["天津博物馆", "五大道", "意式风情区"],
            "weekday": "周一",
        },
    )
    closed_museum = _poi("天津博物馆", 117.20, 39.08, category="museum")
    closed_museum = replace(
        closed_museum,
        poi=replace(closed_museum.poi, opening_hours="周一 全天不开放"),
    )
    ranked = [
        closed_museum,
        _poi("瓷房子", 117.19, 39.12),
        _poi("天津五大道文化旅游区", 117.18, 39.11),
        _poi("天津·海河意式风情区", 117.21, 39.13, category="shopping"),
    ]

    itinerary = build_simple_itinerary(ranked, profile, preserve_must_visit_capacity=True)
    names = [stop.poi.name for stop in itinerary.days[0].stops]

    assert "天津五大道文化旅游区" in names
    assert "天津·海河意式风情区" in names
    assert "天津博物馆" not in names
