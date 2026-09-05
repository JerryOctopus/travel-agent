from travel_agent.critic import critique_itinerary
from travel_agent.data_loader import load_seed_pois
from travel_agent.planning import build_simple_itinerary
from travel_agent.recommendation import score_pois
from travel_agent.reviser import _reconcile_revision_notes, revise_itinerary
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
