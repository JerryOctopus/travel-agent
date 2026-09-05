from __future__ import annotations

from travel_agent.planning import apply_structured_schedule_constraints, build_simple_itinerary
from travel_agent.critic import critique_itinerary
from dataclasses import replace

from travel_agent.schemas import Itinerary, ItineraryDay, ItineraryStop, POI, RouteInfo, ScoredPOI, TravelProfile


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


def test_geo_cluster_balances_two_activities_per_day_when_supply_allows() -> None:
    profile = TravelProfile(destination="测试城", days=3, pace="standard")
    ranked = [
        _poi("west", 120.00, 30.20),
        _poi("center-1", 120.10, 30.20),
        _poi("center-2", 120.11, 30.20),
        _poi("east-1", 120.20, 30.20),
        _poi("east-2", 120.21, 30.20),
        _poi("east-3", 120.22, 30.20),
    ]

    itinerary = build_simple_itinerary(ranked, profile)

    assert [len(day.stops) for day in itinerary.days] == [2, 2, 2]


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


def test_structured_rescheduling_moves_delayed_meal_to_next_meal_window() -> None:
    sight = _poi("remote-sight", 120.1, 30.2)
    meal = _poi("verified-meal", 120.2, 30.3, category="food", tags=["food"])
    day = ItineraryDay(1, "", [
        ItineraryStop(sight.poi, "09:30", 180, ""),
        ItineraryStop(meal.poi, "11:30", 60, ""),
    ])

    constrained = apply_structured_schedule_constraints(
        [day], [sight, meal], TravelProfile(destination="杭州", days=1, interests=["food"]), None
    )

    food_stop = next(stop for stop in constrained[0].stops if stop.poi.category == "food")
    assert food_stop.start_time == "12:45"


def test_major_activity_start_does_not_consume_only_lunch_window() -> None:
    profile = TravelProfile(destination="杭州", days=1, pace="standard")
    morning = _poi("morning", 120.10, 30.20)
    morning = replace(
        morning, poi=replace(morning.poi, estimated_duration_min=120)
    )
    afternoon = _poi("afternoon", 120.11, 30.21)

    class Routes:
        def estimate_route(self, origin, destination, mode):
            return RouteInfo(
                origin_poi_id=origin.poi_id,
                destination_poi_id=destination.poi_id,
                distance_km=2.0,
                duration_min=15,
                mode=mode,
                source="provider",
            )

    constrained = apply_structured_schedule_constraints(
        [ItineraryDay(1, "", [
            ItineraryStop(morning.poi, "09:00", 120, ""),
            ItineraryStop(afternoon.poi, "11:30", 90, ""),
        ])],
        [morning, afternoon],
        profile,
        Routes(),
    )

    assert constrained[0].stops[1].start_time == "13:00"


def test_meal_precedes_evening_only_activity() -> None:
    profile = TravelProfile(destination="西安", days=1, pace="standard", interests=["food"])
    meal = _poi("清真晚餐", 108.96, 34.22, category="food", tags=["food", "halal"])
    evening = _poi("大唐不夜城", 108.966, 34.214, category="nightlife", tags=["culture"])
    evening = replace(evening, poi=replace(evening.poi, opening_hours="18:00-23:00"))

    itinerary = build_simple_itinerary([evening, meal], profile)

    assert [stop.start_time for stop in itinerary.days[0].stops] == ["17:30", "18:45"]
    assert [stop.poi.name for stop in itinerary.days[0].stops] == ["清真晚餐", "大唐不夜城"]


def test_named_night_activity_is_not_scheduled_in_daylight_when_open_all_day() -> None:
    profile = TravelProfile(destination="杭州", days=1, pace="relaxed")
    light_show = _poi("钱江新城灯光秀", 120.21, 30.25, tags=["sightseeing"])
    light_show = replace(
        light_show,
        poi=replace(light_show.poi, opening_hours="周一至周日 00:00-24:00"),
    )

    itinerary = build_simple_itinerary([light_show], profile)

    assert itinerary.days[0].stops[0].start_time >= "18:00"


def test_explicit_food_plan_reserves_one_evidenced_meal_per_day() -> None:
    profile = TravelProfile(destination="杭州", days=2, pace="standard", interests=["food"])
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
    profile = TravelProfile(destination="北京", days=2, pace="standard", interests=["food"])
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


def test_seafood_avoidance_excludes_food_without_positive_dietary_evidence() -> None:
    profile = TravelProfile(
        destination="厦门",
        days=1,
        pace="standard",
        constraint_state={"dietary": ["不吃海鲜"]},
    )
    ranked = [
        _poi("generic-food", 118.10, 24.45, category="food", tags=["food", "local"]),
        _poi("vegetarian-food", 118.11, 24.46, category="food", tags=["food", "vegetarian"]),
        _poi("sight", 118.12, 24.47),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    food_ids = {
        stop.poi.poi_id
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.category == "food"
    }

    assert food_ids == {"vegetarian-food"}


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


def test_low_ranked_afternoon_fixed_venue_and_nearby_anchor_reserve_capacity() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=2,
        pace="standard",
        constraint_state={
            "fixed_events": [{
                "day": 2,
                "start": "15:00",
                "end": "17:00",
                "location": "邻城专题博物馆",
            }],
        },
    )
    fixed = _poi("邻城专题博物馆", 120.80, 30.80, category="museum")
    nearby = _poi("遗址公园", 120.81, 30.80)
    ranked = [
        *[_poi(f"high-score-{index}", 120.10 + index / 100, 30.20) for index in range(8)],
        replace(nearby, score=0.2),
        replace(fixed, score=0.1),
    ]

    itinerary = build_simple_itinerary(
        ranked, profile, preserve_must_visit_capacity=True
    )
    day_two_names = {stop.poi.name for stop in itinerary.days[1].stops}

    assert "邻城专题博物馆" in day_two_names
    assert "遗址公园" in day_two_names


def test_location_only_fixed_block_reserves_transfer_buffer() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        pace="standard",
        constraint_state={
            "fixed_events": [{
                "day": 1,
                "start": "18:00",
                "end": "20:00",
                "location": "江畔商圈",
            }],
            "user_owned_unspecified_fixed_event_locations": ["江畔商圈"],
        },
    )
    late = _poi("late-stop", 120.10, 30.20)
    day = ItineraryDay(1, "", [
        ItineraryStop(late.poi, "16:30", 90, ""),
    ])

    constrained = apply_structured_schedule_constraints(
        [day], [late], profile, None
    )

    assert constrained[0].stops == []


def test_user_owned_location_only_meal_block_does_not_require_fabricated_poi() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=2,
        pace="relaxed",
        constraint_state={
            "fixed_events": [{
                "day": 2,
                "start": "18:00",
                "end": "20:00",
                "location": "江畔商圈",
            }],
            "user_owned_unspecified_fixed_event_locations": ["江畔商圈"],
        },
    )
    ranked = [
        _poi("day-one-a", 120.10, 30.20),
        _poi("day-one-b", 120.11, 30.21),
        _poi("day-two-a", 120.12, 30.22),
        _poi("江畔商圈观景台", 120.13, 30.23),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    issues = critique_itinerary(itinerary, profile).issues

    assert all(issue.code != "fixed_event_missing_or_conflicting" for issue in issues)
    assert all(stop.poi.name != "江畔商圈" for day in itinerary.days for stop in day.stops)


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


def test_exact_activity_end_target_aligns_last_flexible_activity() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        pace="relaxed",
        constraint_state={
            "activity_end_deadline": "20:30",
            "activity_end_target": "20:30",
        },
    )
    ranked = [
        _poi("上午博物馆", 120.10, 30.20, category="museum"),
        _poi("城市观景台", 120.11, 30.21),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    final = itinerary.days[0].stops[-1]

    assert int(final.start_time[:2]) * 60 + int(final.start_time[3:]) + final.duration_min == 20 * 60 + 30
    assert critique_itinerary(itinerary, profile).passed is True


def test_conditional_outdoor_avoidance_window_is_reserved_fail_safe() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        must_visit=["城市古祠", "城市观景塔"],
        constraint_state={
            "conditional_avoid_window": {
                "condition": "high_temperature",
                "start": "12:00",
                "end": "15:00",
                "avoid": "long_outdoor_activity",
            },
            "activity_end_deadline": "20:30",
            "activity_end_target": "20:30",
        },
    )
    hall = replace(
        _poi("城市古祠", 120.10, 30.20).poi,
        estimated_duration_min=150,
        indoor=False,
    )
    tower = replace(
        _poi("城市观景塔", 120.11, 30.21).poi,
        estimated_duration_min=150,
        indoor=False,
    )
    unsafe = ItineraryDay(1, "", [
        ItineraryStop(hall, "09:00", 150, ""),
        ItineraryStop(tower, "12:00", 150, ""),
    ])

    assert any(
        issue.code == "conditional_avoid_window_conflict"
        for issue in critique_itinerary(
            Itinerary(city="测试城", summary="", days=[unsafe]),
            profile,
        ).issues
    )

    constrained = apply_structured_schedule_constraints(
        [unsafe],
        [ScoredPOI(hall, 1.0, []), ScoredPOI(tower, 0.9, [])],
        profile,
        None,
    )
    stops = constrained[0].stops

    assert stops[0].start_time == "09:00"
    assert stops[1].start_time == "18:00"
    assert critique_itinerary(
        Itinerary(city="测试城", summary="", days=constrained),
        profile,
    ).passed is True


def test_provider_parent_relation_dedupes_scenic_subvenue() -> None:
    profile = TravelProfile(destination="测试城", days=1, pace="relaxed")
    parent = replace(
        _poi("古祠堂", 120.10, 30.20),
        poi=replace(
            _poi("古祠堂", 120.10, 30.20).poi,
            source_poi_id="provider-parent",
        ),
    )
    child = replace(
        _poi("古祠广场", 120.1001, 30.2001),
        poi=replace(
            _poi("古祠广场", 120.1001, 30.2001).poi,
            parent_poi_id="provider-parent",
        ),
        score=0.9,
    )

    itinerary = build_simple_itinerary([parent, child], profile)

    assert [stop.poi.name for stop in itinerary.days[0].stops] == ["古祠堂"]


def test_provider_parent_relation_dedupes_differently_named_museum_entities() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        pace="relaxed",
        must_visit=["古王朝兵俑馆"],
    )
    parent = replace(
        _poi("古王朝帝陵博物院", 120.10, 30.20),
        poi=replace(
            _poi("古王朝帝陵博物院", 120.10, 30.20).poi,
            source_poi_id="provider-parent",
        ),
    )
    child = replace(
        _poi("古王朝兵俑馆", 120.1001, 30.2001),
        poi=replace(
            _poi("古王朝兵俑馆", 120.1001, 30.2001).poi,
            parent_poi_id="provider-parent",
            source_poi_id="provider-child",
        ),
        score=0.9,
    )

    itinerary = build_simple_itinerary([child, parent], profile)

    assert [stop.poi.name for stop in itinerary.days[0].stops] == ["古王朝兵俑馆"]


def test_provider_parent_relation_dedupes_sibling_subvenues() -> None:
    profile = TravelProfile(destination="测试城", days=1, pace="relaxed")
    hall = replace(
        _poi("古塔", 120.10, 30.20),
        poi=replace(
            _poi("古塔", 120.10, 30.20).poi,
            parent_poi_id="provider-complex",
            source_poi_id="provider-hall",
        ),
    )
    pavilion = replace(
        _poi("古塔景区-夕照亭", 120.1001, 30.2001),
        poi=replace(
            _poi("古塔景区-夕照亭", 120.1001, 30.2001).poi,
            parent_poi_id="provider-complex",
            source_poi_id="provider-pavilion",
        ),
        score=0.9,
    )

    itinerary = build_simple_itinerary([hall, pavilion], profile)

    assert [stop.poi.name for stop in itinerary.days[0].stops] == ["古塔"]


def test_deadline_prioritizes_must_visit_stops() -> None:
    profile = TravelProfile(
        destination="上海",
        days=1,
        pace="standard",
        must_visit=["外滩", "豫园"],
        constraint_state={"activity_end_deadline": "18:00"},
    )
    exact = _poi("上海豫园", 121.49, 31.23)
    exact = replace(exact, poi=replace(exact.poi, aliases=["豫园"]))
    ranked = [
        exact,
        _poi("南京路步行街", 121.47, 31.23, category="shopping"),
        _poi("外滩", 121.50, 31.24),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    names = [stop.poi.name for stop in itinerary.days[0].stops]

    assert "上海豫园" in names
    assert "外滩" in names
    assert critique_itinerary(itinerary, profile).passed is True


def test_deadline_keeps_three_hard_venues_with_compact_required_slots() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        pace="standard",
        must_visit=["必访甲馆", "必访乙馆", "必访丙馆"],
        constraint_state={"activity_end_deadline": "19:00"},
    )
    ranked = [
        replace(
            _poi(name, 120.10 + index / 100, 30.20),
            poi=replace(
                _poi(name, 120.10 + index / 100, 30.20).poi,
                estimated_duration_min=120,
            ),
        )
        for index, name in enumerate(profile.must_visit)
    ]

    class Routes:
        def estimate_route(self, origin, destination, mode):
            return RouteInfo(
                origin_poi_id=origin.poi_id,
                destination_poi_id=destination.poi_id,
                distance_km=3.0,
                duration_min=30,
                mode=mode,
                source="provider",
            )

    initial = build_simple_itinerary(
        ranked, profile, route_estimator=Routes(), preserve_must_visit_capacity=True
    )
    constrained = apply_structured_schedule_constraints(
        list(initial.days), ranked, profile, Routes()
    )
    stops = constrained[0].stops

    assert {stop.poi.name for stop in stops} == set(profile.must_visit)
    assert max(
        int(stop.start_time[:2]) * 60
        + int(stop.start_time[3:])
        + stop.duration_min
        for stop in stops
    ) <= 19 * 60


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
    unrelated_closed = _poi("天津自然博物馆", 117.22, 39.09, category="museum")
    unrelated_closed = replace(
        unrelated_closed,
        poi=replace(unrelated_closed.poi, opening_hours="周一 全天不开放"),
    )
    ranked = [
        closed_museum,
        unrelated_closed,
        _poi("瓷房子", 117.19, 39.12),
        _poi("天津五大道文化旅游区", 117.18, 39.11),
        _poi("天津·海河意式风情区", 117.21, 39.13, category="shopping"),
    ]

    itinerary = build_simple_itinerary(ranked, profile, preserve_must_visit_capacity=True)
    names = [stop.poi.name for stop in itinerary.days[0].stops]

    assert "天津五大道文化旅游区" in names
    assert "天津·海河意式风情区" in names
    assert "天津博物馆" not in names
    assert "天津自然博物馆" not in names


def test_named_candidate_prefers_requested_venue_over_higher_ranked_subvenue() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        pace="relaxed",
        constraint_state={"candidate_attractions": ["五大道", "河畔公园"]},
    )
    ranked = [
        _poi("五大道历史博物馆", 117.19, 39.12),
        _poi("测试城五大道文化旅游区", 117.18, 39.11),
        _poi("河畔公园", 117.21, 39.13),
    ]

    itinerary = build_simple_itinerary(ranked, profile)
    names = [stop.poi.name for stop in itinerary.days[0].stops]

    assert "测试城五大道文化旅游区" in names
    assert "五大道历史博物馆" not in names
