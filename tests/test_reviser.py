from travel_agent.critic import critique_itinerary
from travel_agent.data_loader import load_seed_pois
from travel_agent.planning import build_simple_itinerary
from travel_agent.recommendation import score_pois
from travel_agent.reviser import (
    _fill_sparse_activity_days,
    _fill_missing_meal_days,
    _repair_long_routes,
    _reconcile_revision_notes,
    _schedule_stops,
    revise_itinerary,
)
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    RouteInfo,
    ScoredPOI,
    TravelProfile,
)
from travel_agent.tools import search_poi
from travel_agent.workflow import DEFAULT_POI_PATH


def test_reviser_adds_missing_must_visit_when_candidate_exists() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["food"],
        must_visit=["故宫博物院"],
        pace="relaxed",
    )
    pois = load_seed_pois(DEFAULT_POI_PATH)
    candidates = search_poi(pois, city="北京", query_tags=profile.interests)
    ranked = score_pois(candidates, profile)
    itinerary = build_simple_itinerary(ranked, profile)
    critic_result = critique_itinerary(itinerary, profile)

    revised, revised_result, notes = revise_itinerary(
        itinerary=itinerary,
        ranked_pois=score_pois(search_poi(pois, city="北京"), profile),
        profile=profile,
        critic_result=critic_result,
    )

    names = [
        stop.poi.name
        for day in revised.days
        for stop in day.stops
    ]
    assert any("故宫" in name for name in names)
    assert notes
    assert "must_visit_missing" not in {issue.code for issue in revised_result.issues}


def test_reviser_does_not_replace_one_must_visit_with_another() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=[],
        must_visit=["故宫博物院", "天坛公园"],
        pace="relaxed",
    )
    pois = load_seed_pois(DEFAULT_POI_PATH)
    initial = build_simple_itinerary(
        score_pois(search_poi(pois, city="北京", query_tags=["food"]), profile),
        profile,
    )
    critic_result = critique_itinerary(initial, profile)

    revised, revised_result, _notes = revise_itinerary(
        itinerary=initial,
        ranked_pois=score_pois(search_poi(pois, city="北京"), profile),
        profile=profile,
        critic_result=critic_result,
    )

    names = [stop.poi.name for day in revised.days for stop in day.stops]
    assert any("故宫" in name for name in names)
    assert any("天坛" in name for name in names)
    assert revised_result.passed is True


def test_reviser_replaces_optional_endpoint_of_overlong_route() -> None:
    def poi(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="青岛",
            category="scenic",
            lat=36.06,
            lng=lng,
            rating=4.7,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=90,
            price_level="mid",
        )

    remote = poi("remote", "崂山风景区", 120.65)
    center = poi("center", "五四广场", 120.38)
    nearby = poi("nearby", "音乐广场", 120.381)
    itinerary = Itinerary(
        city="青岛",
        summary="test",
        days=[
            ItineraryDay(
                day_index=1,
                theme="test",
                stops=[
                    ItineraryStop(remote, "09:30", 90, ""),
                    ItineraryStop(
                        center,
                        "14:30",
                        90,
                        "",
                        RouteInfo("remote", "center", 50, 219, "public_transport"),
                    ),
                ],
            )
        ],
    )
    profile = TravelProfile(
        destination="青岛",
        days=1,
        constraint_state={"optional_remove": ["崂山"]},
    )
    critic = critique_itinerary(itinerary, profile)

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(nearby, 1.0, [])],
        profile,
        critic,
    )

    assert [stop.poi.name for stop in revised.days[0].stops] == ["音乐广场", "五四广场"]
    assert any("超长通勤" in note for note in notes)


def test_explicit_relaxed_pace_drops_unverified_replacement_and_keeps_free_time() -> None:
    def poi(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category="scenic",
            lat=30.0,
            lng=lng,
            rating=4.6,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=90,
            price_level="mid",
        )

    required = poi("required", "核心景点", 120.0)
    remote = poi("remote", "可选景点", 120.1)
    straight_line_nearby = poi("nearby", "直线距离近的候选", 120.01)
    itinerary = Itinerary(
        city="测试城",
        summary="test",
        days=[
            ItineraryDay(
                day_index=1,
                theme="test",
                stops=[
                    ItineraryStop(required, "09:30", 90, ""),
                    ItineraryStop(
                        remote,
                        "14:30",
                        90,
                        "",
                        RouteInfo("required", "remote", 8, 65, "public_transport"),
                    ),
                ],
            )
        ],
    )
    profile = TravelProfile(
        destination="测试城",
        days=1,
        pace="relaxed",
        must_visit=["核心景点"],
        constraint_state={"pace": "relaxed", "must_visit": ["核心景点"]},
    )

    revised, result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(straight_line_nearby, 1.0, [])],
        profile,
        critique_itinerary(itinerary, profile),
    )

    assert [stop.poi.poi_id for stop in revised.days[0].stops] == ["required"]
    assert "route_too_long" not in {issue.code for issue in result.issues}
    assert any("显式轻松节奏" in note for note in notes)


def test_long_route_repair_prefers_reusing_nearby_meal_over_dropping_activity() -> None:
    activity = POI("activity", "城内博物馆", "测试城", "museum", 30.0, 120.0, 4.5, 0.8, ["history"], 90, "mid")
    remote_meal = POI("remote-meal", "远郊合规餐厅", "测试城", "food", 30.0, 120.8, 4.5, 0.8, ["food", "halal"], 60, "mid")
    nearby_meal = POI("near-meal", "城内合规餐厅", "测试城", "food", 30.0, 120.01, 4.5, 0.8, ["food", "halal"], 60, "mid")
    itinerary = Itinerary(
        city="测试城",
        summary="test",
        days=[
            ItineraryDay(1, "test", [ItineraryStop(nearby_meal, "12:00", 60, "")]),
            ItineraryDay(2, "test", [
                ItineraryStop(activity, "09:30", 90, ""),
                ItineraryStop(remote_meal, "17:30", 60, "", RouteInfo("activity", "remote-meal", 80, 180, "public_transport")),
            ]),
        ],
    )
    profile = TravelProfile(
        destination="测试城",
        days=2,
        constraint_state={"dietary": ["halal"]},
    )

    repaired, notes = _repair_long_routes(
        itinerary,
        [ScoredPOI(activity, 0.9, []), ScoredPOI(remote_meal, 0.8, []), ScoredPOI(nearby_meal, 0.7, [])],
        profile,
    )

    assert [stop.poi.poi_id for stop in repaired.days[1].stops] == ["activity", "near-meal"]
    assert any("替换" in note for note in notes)


def test_feasibility_shortlist_drops_optional_stop_when_no_lunch_window() -> None:
    def attraction(poi_id: str, name: str) -> POI:
        return POI(poi_id, name, "测试城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 120, "mid")

    first, second, third = (
        attraction("a", "甲博物馆"),
        attraction("b", "乙公园"),
        attraction("c", "丙古街"),
    )
    itinerary = Itinerary("测试城", [ItineraryDay(1, "test", [
        ItineraryStop(first, "09:00", 120, ""),
        ItineraryStop(second, "12:00", 120, "", RouteInfo("a", "b", 5, 30, "public_transport")),
        ItineraryStop(third, "15:00", 150, "", RouteInfo("b", "c", 5, 30, "public_transport")),
    ])], "test")
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "candidate_only": True,
            "candidate_attractions": ["甲博物馆", "乙公园", "丙古街"],
            "return_deadline": "19:00",
            "return_location": "测试城南站",
        },
    )
    initial = critique_itinerary(itinerary, profile)

    revised, result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(first, 0.9, []), ScoredPOI(second, 0.8, []), ScoredPOI(third, 0.7, [])],
        profile,
        initial,
    )

    assert "meal_break_missing" in {issue.code for issue in initial.issues}
    assert len(revised.days[0].stops) == 2
    assert "meal_break_missing" not in {issue.code for issue in result.issues}
    assert any("用餐窗口" in note for note in notes)


def test_reviser_repairs_isolated_activity_before_central_meal() -> None:
    def poi(poi_id: str, name: str, category: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="西安",
            category=category,
            lat=34.30,
            lng=lng,
            rating=4.5,
            popularity=0.8,
            tags=["food", "halal"] if category == "food" else ["history"],
            estimated_duration_min=60,
            price_level="mid",
        )

    must_visit = poi("must", "兵马俑", "museum", 109.28)
    meal = poi("meal", "清真临潼餐厅", "food", 109.15)
    remote_optional = poi("remote", "市区可选景点", "scenic", 108.95)
    nearby_activity = poi("nearby", "华清宫", "scenic", 109.21)
    alternate_meal = poi("alternate-meal", "清真市区餐厅", "food", 108.96)
    itinerary = Itinerary(
        city="西安",
        summary="test",
        days=[ItineraryDay(1, "test", [
            ItineraryStop(must_visit, "09:30", 90, ""),
            ItineraryStop(
                meal,
                "12:00",
                60,
                "",
                RouteInfo("must", "meal", 12, 40, "public_transport"),
            ),
            ItineraryStop(
                remote_optional,
                "14:30",
                90,
                "",
                RouteInfo("meal", "remote", 24, 79, "public_transport"),
            ),
        ])],
    )
    profile = TravelProfile(
        destination="西安",
        days=1,
        must_visit=["兵马俑"],
        constraint_state={"dietary": ["仅清真餐厅"]},
    )

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(nearby_activity, 0.8, []), ScoredPOI(alternate_meal, 0.8, [])],
        profile,
        critique_itinerary(itinerary, profile),
    )

    names = [stop.poi.name for stop in revised.days[0].stops]
    assert "清真临潼餐厅" in names
    assert "市区可选景点" not in names
    assert "华清宫" in names
    assert any("市区可选景点" in note and "华清宫" in note for note in notes)


def test_reviser_reduces_load_for_hard_daily_walking_cap() -> None:
    def poi(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="杭州",
            category="scenic",
            lat=30.25,
            lng=lng,
            rating=4.5,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=90,
            price_level="mid",
        )

    first = poi("first", "甲公园", 120.10)
    middle = poi("middle", "乙公园", 120.14)
    last = poi("last", "丙公园", 120.18)
    replacement = poi("replacement", "丁公园", 120.15)
    itinerary = Itinerary(
        city="杭州",
        summary="test",
        days=[
            ItineraryDay(
                day_index=1,
                theme="test",
                stops=[
                    ItineraryStop(first, "09:30", 90, ""),
                    ItineraryStop(
                        middle,
                        "14:30",
                        90,
                        "",
                        RouteInfo("first", "middle", 4.0, 50, "walk", walking_distance_km=4.0),
                    ),
                    ItineraryStop(
                        last,
                        "17:00",
                        90,
                        "",
                        RouteInfo("middle", "last", 4.0, 50, "walk", walking_distance_km=4.0),
                    ),
                ],
            )
        ],
    )
    profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"max_walking_km_per_day": 6},
    )
    critic = critique_itinerary(itinerary, profile)

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(replacement, 1.0, [])],
        profile,
        critic,
    )

    assert len(revised.days[0].stops) == 2
    assert all(stop.poi.poi_id != "replacement" for stop in revised.days[0].stops)
    assert any("每日步行上限" in note for note in notes)


def test_reviser_selects_longest_route_when_walking_cap_is_not_active() -> None:
    def poi(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="西安",
            category="scenic",
            lat=34.25,
            lng=lng,
            rating=4.5,
            popularity=0.8,
            tags=["history"],
            estimated_duration_min=90,
            price_level="mid",
        )

    first = poi("first", "甲景点", 108.90)
    second = poi("second", "乙景点", 108.95)
    itinerary = Itinerary(
        city="西安",
        summary="test",
        days=[ItineraryDay(
            day_index=1,
            theme="test",
            stops=[
                ItineraryStop(first, "09:30", 90, ""),
                ItineraryStop(
                    second,
                    "15:00",
                    90,
                    "",
                    RouteInfo("first", "second", 10.0, 180, "public_transport"),
                ),
            ],
        )],
    )
    profile = TravelProfile(destination="西安", days=1)
    critic = critique_itinerary(itinerary, profile)

    revised, _result, notes = revise_itinerary(itinerary, [], profile, critic)

    assert len(revised.days[0].stops) == 1
    assert any("超长通勤" in note for note in notes)


def test_reviser_replaces_remote_food_with_dietary_compliant_food() -> None:
    def food(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="西安",
            category="food",
            lat=34.25,
            lng=lng,
            rating=4.5,
            popularity=0.8,
            tags=["food", "halal"],
            estimated_duration_min=60,
            price_level="mid",
        )

    scenic = POI(
        poi_id="wall",
        name="西安城墙",
        city="西安",
        category="culture",
        lat=34.25,
        lng=108.95,
        rating=4.8,
        popularity=1.0,
        tags=["history"],
        estimated_duration_min=120,
        price_level="mid",
    )
    remote = food("remote-halal", "清真远郊餐厅", 109.50)
    nearby = food("near-halal", "清真城墙餐厅", 108.951)
    itinerary = Itinerary(
        city="西安",
        summary="test",
        days=[ItineraryDay(
            day_index=1,
            theme="test",
            stops=[
                ItineraryStop(scenic, "09:30", 120, ""),
                ItineraryStop(
                    remote,
                    "15:00",
                    60,
                    "",
                    RouteInfo("wall", "remote-halal", 45.0, 180, "public_transport"),
                ),
            ],
        )],
    )
    profile = TravelProfile(
        destination="西安",
        days=1,
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    critic = critique_itinerary(itinerary, profile)

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(nearby, 1.0, [])],
        profile,
        critic,
    )

    assert [stop.poi.name for stop in revised.days[0].stops] == ["西安城墙", "清真城墙餐厅"]
    assert any("替换" in note and "清真城墙餐厅" in note for note in notes)


def test_reviser_removes_optional_stop_before_distant_fixed_event() -> None:
    def poi(poi_id: str, name: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="成都",
            category="museum",
            lat=30.65,
            lng=lng,
            rating=4.8,
            popularity=0.9,
            tags=["history"],
            estimated_duration_min=90,
            price_level="mid",
        )

    optional = poi("optional", "武侯祠", 104.05)
    fixed = poi("fixed", "三星堆博物馆", 104.22)
    replacement = poi("replacement", "锦里古街", 104.06)
    itinerary = Itinerary(
        city="成都",
        summary="test",
        days=[ItineraryDay(
            day_index=2,
            theme="test",
            stops=[
                ItineraryStop(optional, "09:30", 90, ""),
                ItineraryStop(
                    fixed,
                    "15:00",
                    120,
                    "用户已确认的固定时段活动",
                    RouteInfo("optional", "fixed", 58.0, 132, "public_transport"),
                ),
            ],
        )],
    )
    profile = TravelProfile(
        destination="成都",
        days=3,
        must_visit=["三星堆博物馆"],
        constraint_state={
            "fixed_events": [{
                "day": 2,
                "start": "15:00",
                "end": "17:00",
                "location": "三星堆博物馆",
            }]
        },
    )
    critic = critique_itinerary(itinerary, profile)

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(replacement, 1.0, [])],
        profile,
        critic,
    )

    assert [stop.poi.name for stop in revised.days[0].stops] == ["三星堆博物馆"]
    assert any("固定时段活动" in note for note in notes)


def test_reviser_adds_nearby_activity_to_sparse_multiday_day() -> None:
    def poi(poi_id: str, name: str, category: str, lng: float) -> POI:
        return POI(
            poi_id=poi_id,
            name=name,
            city="西安",
            category=category,
            lat=34.26,
            lng=lng,
            rating=4.5,
            popularity=0.8,
            tags=["food"] if category == "food" else ["history"],
            estimated_duration_min=60,
            price_level="mid",
        )

    days = [
        ItineraryDay(
            day_index=index,
            theme="test",
            stops=[
                ItineraryStop(poi(f"a{index}", f"景点{index}", "scenic", 108.9 + index / 100), "09:30", 90, ""),
                ItineraryStop(poi(f"f{index}", f"餐厅{index}", "food", 108.91 + index / 100), "12:00", 60, ""),
            ],
        )
        for index in range(1, 4)
    ]
    itinerary = Itinerary(city="西安", summary="test", days=days)
    profile = TravelProfile(destination="西安", days=3)
    ranked = [
        ScoredPOI(poi(f"extra{index}", f"下午景点{index}", "museum", 108.92 + index / 100), 0.8, [])
        for index in range(1, 4)
    ]

    revised, result, notes = revise_itinerary(
        itinerary,
        ranked,
        profile,
        critique_itinerary(itinerary, profile),
    )

    assert all(sum(stop.poi.category != "food" for stop in day.stops) == 2 for day in revised.days)
    assert "daily_activity_sparse" not in {issue.code for issue in result.issues}
    assert len(notes) == 3


def test_relaxed_day_meal_does_not_consume_second_activity_capacity() -> None:
    activity = POI(
        "activity", "上午景点", "测试城", "scenic", 30.0, 120.0,
        4.5, 0.8, ["history"], 90, "mid",
    )
    meal = POI(
        "meal", "午餐", "测试城", "food", 30.0, 120.01,
        4.5, 0.8, ["food"], 60, "mid",
    )
    afternoon = POI(
        "afternoon", "下午博物馆", "测试城", "museum", 30.0, 120.02,
        4.5, 0.8, ["history"], 90, "mid",
    )
    itinerary = Itinerary(
        "测试城",
        [
            ItineraryDay(1, "test", [
                ItineraryStop(activity, "09:30", 90, ""),
                ItineraryStop(meal, "11:30", 60, ""),
            ]),
            ItineraryDay(2, "test", []),
        ],
        "test",
    )
    profile = TravelProfile(destination="测试城", days=2, pace="relaxed")

    revised, notes = _fill_sparse_activity_days(
        itinerary, [ScoredPOI(afternoon, 0.8, [])], profile
    )

    assert [stop.poi.poi_id for stop in revised.days[0].stops] == [
        "activity", "meal", "afternoon",
    ]
    assert any("第1天下午" in note for note in notes)


def test_reviser_restores_missing_meal_after_other_repairs() -> None:
    scenic = POI("scenic", "兵马俑", "西安", "museum", 34.38, 109.28, 4.9, 1.0, ["history"], 90, "mid")
    meal = POI("meal", "清真附近餐厅", "西安", "food", 34.37, 109.27, 4.5, 0.8, ["food", "halal", "清真"], 60, "mid")
    itinerary = Itinerary(
        city="西安",
        summary="test",
        days=[ItineraryDay(1, "test", [ItineraryStop(scenic, "09:30", 90, "")])],
    )
    profile = TravelProfile(
        destination="西安",
        days=1,
        must_visit=["兵马俑"],
        constraint_state={"dietary": ["仅清真餐厅"]},
    )

    revised, result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(meal, 0.8, [])],
        profile,
        critique_itinerary(itinerary, profile),
    )

    assert any(stop.poi.category == "food" for stop in revised.days[0].stops)
    assert "daily_meal_missing" not in {issue.code for issue in result.issues}
    assert any("补全第1天用餐" in note for note in notes)


def test_meal_repair_uses_nearest_day_anchor_not_farthest_day_anchor() -> None:
    near = POI("near", "近端景点", "合成城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 90, "mid")
    remote = POI("remote", "远端景点", "合成城", "scenic", 30.0, 120.5, 4.5, 0.8, [], 90, "mid")
    meal = POI("meal", "近端餐厅", "合成城", "food", 30.0, 120.01, 4.5, 0.8, ["food"], 60, "mid")
    itinerary = Itinerary(
        city="合成城",
        summary="test",
        days=[ItineraryDay(1, "test", [
            ItineraryStop(near, "09:00", 90, ""),
            ItineraryStop(remote, "15:00", 90, ""),
        ])],
    )
    profile = TravelProfile(destination="合成城", days=1, interests=["food"])

    revised, notes = _fill_missing_meal_days(
        itinerary,
        [ScoredPOI(meal, 0.8, [])],
        profile,
    )

    assert any(stop.poi.poi_id == "meal" for stop in revised.days[0].stops)
    assert any("补全第1天用餐" in note for note in notes)


def test_meal_repair_can_reuse_grounded_restaurant_across_trip_days() -> None:
    day_one = POI(
        "day-one", "第一天景点", "合成城", "scenic",
        30.0, 120.0, 4.5, 0.8, [], 90, "mid",
    )
    day_two = POI(
        "day-two", "第二天景点", "合成城", "museum",
        30.01, 120.01, 4.5, 0.8, [], 90, "mid",
    )
    meal = POI(
        "meal", "有证据餐厅", "合成城", "food",
        30.005, 120.005, 4.5, 0.8, ["food"], 60, "mid",
    )
    itinerary = Itinerary(
        city="合成城",
        summary="test",
        days=[
            ItineraryDay(1, "test", [
                ItineraryStop(day_one, "09:30", 90, ""),
                ItineraryStop(meal, "11:30", 60, ""),
            ]),
            ItineraryDay(2, "test", [
                ItineraryStop(day_two, "09:30", 90, ""),
            ]),
        ],
    )
    profile = TravelProfile(destination="合成城", days=2, interests=["food"])

    revised, notes = _fill_missing_meal_days(
        itinerary,
        [ScoredPOI(meal, 0.8, [])],
        profile,
    )

    assert [
        stop.poi.poi_id
        for stop in revised.days[1].stops
        if stop.poi.category == "food"
    ] == ["meal"]
    assert any("补全第2天用餐" in note for note in notes)


def test_explicit_food_plan_prefers_distant_grounded_meal_to_missing_day() -> None:
    activity = POI(
        "activity", "远郊景点", "合成城", "scenic",
        30.0, 120.0, 4.5, 0.8, [], 90, "mid",
    )
    meal = POI(
        "meal", "有证据清真餐厅", "合成城", "food",
        30.0, 120.3, 4.5, 0.8, ["food", "清真"], 60, "mid",
    )
    itinerary = Itinerary(
        city="合成城",
        summary="test",
        days=[ItineraryDay(1, "test", [
            ItineraryStop(activity, "09:30", 90, ""),
        ])],
    )
    profile = TravelProfile(
        destination="合成城",
        days=1,
        interests=["food"],
        food_preference=["清真"],
        constraint_state={"dietary": ["仅清真餐厅"]},
    )

    revised, notes = _fill_missing_meal_days(
        itinerary,
        [ScoredPOI(meal, 0.8, [])],
        profile,
    )

    assert [stop.poi.poi_id for stop in revised.days[0].stops] == [
        "activity", "meal",
    ]
    assert any("补全第1天用餐" in note for note in notes)


def test_reviser_schedules_evening_only_restaurant_after_opening() -> None:
    activity = POI("activity", "上午景点", "合成城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 90, "mid")
    meal = POI(
        "meal", "晚间餐厅", "合成城", "food", 30.0, 120.01,
        4.5, 0.8, ["food"], 60, "mid", opening_hours="16:00-04:00",
    )

    scheduled = _schedule_stops([
        ItineraryStop(activity, "09:00", 90, ""),
        ItineraryStop(meal, "12:00", 60, ""),
    ])

    meal_stop = next(stop for stop in scheduled if stop.poi.category == "food")
    assert meal_stop.start_time == "16:00"


def test_reviser_keeps_a_late_lunch_in_the_lunch_window() -> None:
    activity = POI(
        "activity", "上午景点", "合成城", "scenic",
        30.0, 120.0, 4.5, 0.8, [], 150, "mid",
    )
    meal = POI(
        "meal", "午餐餐厅", "合成城", "food",
        30.0, 120.01, 4.5, 0.8, ["food"], 60, "mid",
    )

    scheduled = _schedule_stops([
        ItineraryStop(activity, "09:00", 150, ""),
        ItineraryStop(meal, "11:30", 60, ""),
    ])

    meal_stop = next(stop for stop in scheduled if stop.poi.category == "food")
    assert meal_stop.start_time == "12:30"


def test_reviser_never_claims_an_addition_absent_from_final_itinerary() -> None:
    existing = POI("existing", "已有景点", "测试城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 90, "mid")
    museum = POI("museum", "候选博物馆", "测试城", "museum", 31.0, 121.0, 4.5, 0.8, ["museum"], 90, "mid")
    itinerary = Itinerary(
        city="测试城",
        summary="test",
        days=[ItineraryDay(1, "test", [ItineraryStop(existing, "09:30", 90, "")])],
    )
    profile = TravelProfile(destination="测试城", days=1, interests=["museum"], pace="relaxed")

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(museum, 1.0, [])],
        profile,
        critique_itinerary(itinerary, profile),
    )
    final_names = {stop.poi.name for day in revised.days for stop in day.stops}

    assert all(
        "补充" not in note or any(f"`{name}`" in note for name in final_names)
        for note in notes
    )


def test_accumulated_revision_notes_drop_later_removed_candidate() -> None:
    kept = POI("kept", "最终景点", "测试城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 90, "mid")
    itinerary = Itinerary(
        city="测试城", summary="test",
        days=[ItineraryDay(1, "test", [ItineraryStop(kept, "09:30", 90, "")])],
    )

    notes = _reconcile_revision_notes(
        ["为覆盖 `museum` 偏好，补充 `已移除博物馆`。", "保留真实说明。"],
        itinerary,
        TravelProfile(destination="测试城", days=1),
    )

    assert notes == ["保留真实说明。"]


def test_revision_notes_drop_stale_add_and_remove_claims() -> None:
    kept = POI("kept", "最终景点", "测试城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 90, "mid")
    itinerary = Itinerary(
        city="测试城", summary="test",
        days=[ItineraryDay(1, "test", [ItineraryStop(kept, "09:30", 90, "")])],
    )

    notes = _reconcile_revision_notes(
        [
            "为补全第1天下午行程，增加 `已移除景点`。",
            "为修复超长通勤，删减可选地点 `最终景点`。",
            "保留真实说明。",
        ],
        itinerary,
        TravelProfile(destination="测试城", days=1),
    )

    assert notes == ["保留真实说明。"]


def test_reviser_does_not_reuse_a_global_activity_on_sparse_days() -> None:
    only = POI("only", "唯一景点", "测试城", "scenic", 30.0, 120.0, 4.5, 0.8, [], 90, "mid")
    itinerary = Itinerary(
        city="测试城",
        summary="test",
        days=[
            ItineraryDay(1, "test", [ItineraryStop(only, "09:30", 90, "")]),
            ItineraryDay(2, "test", []),
            ItineraryDay(3, "test", []),
        ],
    )
    profile = TravelProfile(destination="测试城", days=3)

    revised, _result, _notes = revise_itinerary(
        itinerary,
        [ScoredPOI(only, 1.0, [])],
        profile,
        critique_itinerary(itinerary, profile),
    )

    assert sum(
        stop.poi.poi_id == "only"
        for day in revised.days
        for stop in day.stops
    ) == 1


def test_reviser_does_not_add_a_distant_meal_after_route_repair() -> None:
    scenic = POI("scenic", "核心景点", "测试城", "scenic", 30.0, 120.0, 4.8, 0.9, [], 90, "mid")
    distant = POI("meal", "远端餐厅", "测试城", "food", 31.0, 121.0, 4.5, 0.8, ["food"], 60, "mid")
    itinerary = Itinerary(
        city="测试城", summary="test",
        days=[ItineraryDay(1, "test", [ItineraryStop(scenic, "09:30", 90, "")])],
    )
    profile = TravelProfile(destination="测试城", days=1, pace="standard")

    revised, _result, notes = revise_itinerary(
        itinerary,
        [ScoredPOI(distant, 1.0, [])],
        profile,
        critique_itinerary(itinerary, profile),
    )

    assert all(stop.poi.category != "food" for stop in revised.days[0].stops)
    assert all("远端餐厅" not in note for note in notes)
