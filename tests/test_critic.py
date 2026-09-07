from travel_agent.critic import critique_itinerary
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    RouteInfo,
    TravelProfile,
)


def test_critic_passes_when_itinerary_covers_interests() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["history", "food"],
        pace="relaxed",
    )
    itinerary = Itinerary(
        city="北京",
        summary="北京1天轻松行程",
        days=[
            ItineraryDay(
                day_index=1,
                theme="历史与美食",
                stops=[
                    ItineraryStop(
                        poi=_poi("故宫博物院", "culture", ["history", "culture"]),
                        start_time="09:30",
                        duration_min=150,
                        note="历史文化核心景点",
                    ),
                    ItineraryStop(
                        poi=_poi("南锣鼓巷", "food", ["food", "local"]),
                        start_time="13:30",
                        duration_min=90,
                        note="本地美食体验",
                    ),
                ],
            )
        ],
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is True
    assert result.issues == []


def test_critic_reports_missing_must_visit_and_uncovered_interest() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["food"],
        must_visit=["故宫"],
        pace="relaxed",
    )
    itinerary = Itinerary(
        city="北京",
        summary="北京1天轻松行程",
        days=[
            ItineraryDay(
                day_index=1,
                theme="自然体验",
                stops=[
                    ItineraryStop(
                        poi=_poi("景山公园", "scenic", ["nature"]),
                        start_time="09:30",
                        duration_min=90,
                        note="轻松游览",
                    )
                ],
            )
        ],
    )

    result = critique_itinerary(itinerary, profile)
    codes = {issue.code for issue in result.issues}

    assert result.passed is False
    assert "interest_not_covered" in codes
    assert "must_visit_missing" in codes
    assert "day_too_sparse" in codes
    assert "daily_meal_missing" in codes


def test_explicit_dietary_requirement_makes_missing_daily_meal_an_error() -> None:
    profile = TravelProfile(
        destination="西安",
        days=2,
        interests=["food"],
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    itinerary = Itinerary(
        city="西安",
        summary="test",
        days=[
            ItineraryDay(1, "test", [ItineraryStop(_poi("城墙", "scenic", ["history"]), "09:30", 90, "")]),
            ItineraryDay(2, "test", [ItineraryStop(_poi("兵马俑", "museum", ["history"]), "09:30", 90, "")]),
        ],
    )

    result = critique_itinerary(itinerary, profile)

    missing_meals = [issue for issue in result.issues if issue.code == "daily_meal_missing"]
    assert len(missing_meals) == 2
    assert all(issue.severity == "error" for issue in missing_meals)
    assert result.passed is False


def test_critic_reports_duplicate_restaurant_brand() -> None:
    profile = TravelProfile(destination="北京", days=1, pace="standard")
    itinerary = Itinerary(
        city="北京",
        summary="test",
        days=[
            ItineraryDay(
                day_index=1,
                theme="test",
                stops=[
                    ItineraryStop(
                        poi=_poi("四季民福烤鸭店(东安门店)", "food", ["food"]),
                        start_time="11:30",
                        duration_min=60,
                        note="test",
                    ),
                    ItineraryStop(
                        poi=_poi("四季民福烤鸭店(灯市口店)", "food", ["food"]),
                        start_time="18:00",
                        duration_min=60,
                        note="test",
                    ),
                ],
            )
        ],
    )

    codes = {issue.code for issue in critique_itinerary(itinerary, profile).issues}

    assert "duplicate_food_brand" in codes


def test_warning_only_critic_result_passes() -> None:
    profile = TravelProfile(destination="杭州", days=1, pace="standard")
    itinerary = Itinerary(
        city="杭州",
        days=[
            ItineraryDay(
                day_index=1,
                theme="test",
                stops=[
                    ItineraryStop(
                        poi=_poi("西湖", "scenic", []),
                        start_time="09:30",
                        duration_min=90,
                        note="test",
                        route_from_previous=RouteInfo(
                            origin_poi_id="a",
                            destination_poi_id="b",
                            distance_km=10,
                            duration_min=61,
                            mode="public_transport",
                            source="test",
                        ),
                    )
                ],
            )
        ],
        summary="test",
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is True
    assert {issue.severity for issue in result.issues} == {"warning"}


def test_gross_route_violation_is_error() -> None:
    profile = TravelProfile(destination="杭州", days=1, pace="standard")
    itinerary = Itinerary(
        city="杭州",
        days=[
            ItineraryDay(
                day_index=1,
                theme="test",
                stops=[
                    ItineraryStop(
                        poi=_poi("西湖", "scenic", []),
                        start_time="09:30",
                        duration_min=90,
                        note="test",
                        route_from_previous=RouteInfo(
                            origin_poi_id="a",
                            destination_poi_id="b",
                            distance_km=60,
                            duration_min=180,
                            mode="walk",
                            source="test",
                        ),
                    )
                ],
            )
        ],
        summary="test",
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is False
    assert any(issue.severity == "error" for issue in result.issues)


def test_critic_flags_fast_cross_city_backtracking() -> None:
    profile = TravelProfile(destination="测试城", days=1, pace="standard")
    first = POI("remote", "远郊必去点", "测试城", "scenic", 30.0, 120.50, 4.8, 0.9, [], 90, "mid")
    middle = POI("city", "市区可选点", "测试城", "museum", 30.0, 120.10, 4.5, 0.8, [], 90, "mid")
    last = POI("remote-meal", "远郊餐厅", "测试城", "food", 30.0, 120.51, 4.5, 0.8, ["food"], 60, "mid")
    itinerary = Itinerary(
        city="测试城",
        days=[ItineraryDay(1, "test", [
            ItineraryStop(first, "09:00", 90, ""),
            ItineraryStop(
                middle,
                "13:00",
                90,
                "",
                RouteInfo("remote", "city", 38.0, 39, "taxi", source="test"),
            ),
            ItineraryStop(
                last,
                "17:30",
                60,
                "",
                RouteInfo("city", "remote-meal", 39.0, 42, "taxi", source="test"),
            ),
        ])],
        summary="test",
    )

    result = critique_itinerary(itinerary, profile)

    assert "route_backtracking" in {issue.code for issue in result.issues}


def test_critic_reports_route_too_long() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["history"],
        pace="relaxed",
    )
    itinerary = Itinerary(
        city="北京",
        summary="北京1天轻松行程",
        days=[
            ItineraryDay(
                day_index=1,
                theme="历史文化",
                stops=[
                    ItineraryStop(
                        poi=_poi("故宫博物院", "culture", ["history"]),
                        start_time="09:30",
                        duration_min=120,
                        note="历史文化核心景点",
                    ),
                    ItineraryStop(
                        poi=_poi("八达岭长城", "culture", ["history"]),
                        start_time="13:30",
                        duration_min=150,
                        note="历史文化核心景点",
                        route_from_previous=RouteInfo(
                            origin_poi_id="故宫博物院",
                            destination_poi_id="八达岭长城",
                            distance_km=65.0,
                            duration_min=95,
                            mode="public_transport",
                            source="test",
                        ),
                    ),
                ],
            )
        ],
    )

    result = critique_itinerary(itinerary, profile)
    codes = {issue.code for issue in result.issues}

    assert result.passed is True
    assert "route_too_long" in codes
    assert "daily_route_too_long" in codes
    assert all(issue.severity == "warning" for issue in result.issues if issue.code in codes)


def test_critic_rejects_food_without_required_halal_evidence() -> None:
    profile = TravelProfile(
        destination="西安",
        days=1,
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    itinerary = Itinerary(
        city="西安",
        summary="test",
        days=[ItineraryDay(
            day_index=1,
            theme="test",
            stops=[ItineraryStop(
                poi=_poi("普通餐厅", "food", ["food"]),
                start_time="11:30",
                duration_min=60,
                note="test",
            )],
        )],
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is False
    assert "dietary_constraint_violated" in {issue.code for issue in result.issues}


def test_structured_specific_interests_are_not_satisfied_by_broad_categories() -> None:
    profile = TravelProfile(
        destination="苏州",
        days=1,
        interests=["nature"],
        constraint_state={"interests": ["园林"]},
    )
    itinerary = Itinerary(
        city="苏州",
        summary="test",
        days=[ItineraryDay(
            day_index=1,
            theme="test",
            stops=[
                ItineraryStop(
                    poi=_poi("七里山塘景区", "scenic", ["classic", "sightseeing"]),
                    start_time="09:30",
                    duration_min=90,
                    note="test",
                ),
                ItineraryStop(
                    poi=_poi("本地餐厅", "food", ["food"]),
                    start_time="11:30",
                    duration_min=60,
                    note="test",
                ),
            ],
        )],
    )

    result = critique_itinerary(itinerary, profile)

    assert any(
        issue.code == "interest_not_covered" and "园林" in issue.message
        for issue in result.issues
    )


def test_history_attraction_interest_accepts_verified_history_category() -> None:
    profile = TravelProfile(
        destination="测试城", days=1,
        constraint_state={"interests": ["历史景点"]},
    )
    itinerary = Itinerary(
        city="测试城", summary="test",
        days=[ItineraryDay(1, "历史", [ItineraryStop(
            poi=_poi("城市历史博物馆", "museum", ["history"]),
            start_time="09:30", duration_min=90, note="test",
        )])],
    )

    result = critique_itinerary(itinerary, profile)

    assert not any(
        issue.code == "interest_not_covered" and "历史景点" in issue.message
        for issue in result.issues
    )


def test_accessibility_priority_fails_when_evidence_is_missing() -> None:
    profile = TravelProfile(
        destination="苏州",
        days=1,
        constraint_state={"wheelchair_user": True, "accessibility_priority": True},
    )
    itinerary = Itinerary(
        city="苏州",
        summary="test",
        days=[ItineraryDay(
            day_index=1,
            theme="test",
            stops=[ItineraryStop(
                poi=_poi("拙政园", "scenic", ["garden"]),
                start_time="09:30",
                duration_min=90,
                note="test",
            )],
        )],
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is False
    assert "accessibility_evidence_missing" in {issue.code for issue in result.issues}


def test_terrain_avoidance_accepts_verified_nonwalking_route_without_claiming_accessibility() -> None:
    first = _poi("城市广场", "scenic", [])
    second = _poi("城市展馆", "museum", [])
    itinerary = Itinerary("测试城", [ItineraryDay(1, "", [
        ItineraryStop(first, "09:30", 60, ""),
        ItineraryStop(
            second,
            "11:00",
            60,
            "",
            RouteInfo(
                first.poi_id, second.poi_id, 3.0, 15, "taxi",
                source="amap", evidence_status="provider_verified",
            ),
        ),
    ])], "")
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"avoid": ["连续爬坡", "长楼梯"]},
    )

    result = critique_itinerary(itinerary, profile)

    codes = {issue.code for issue in result.issues}
    assert "accessibility_evidence_missing" not in codes


def test_explicit_stairs_and_hills_avoidance_fails_closed_without_access_evidence() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"avoid": ["连续爬坡", "长楼梯"]},
    )
    itinerary = Itinerary(
        city="测试城",
        summary="test",
        days=[ItineraryDay(1, "test", [
            ItineraryStop(_poi("普通景点", "scenic", []), "09:30", 90, "")
        ])],
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is False
    assert "accessibility_evidence_missing" in {issue.code for issue in result.issues}


def test_commercial_amenity_cannot_satisfy_must_visit() -> None:
    from travel_agent.critic import poi_matches_must_visit

    assert not poi_matches_must_visit(
        _poi("兵马俑旅游纪念品综合超市", "shopping", ["shopping"]), "兵马俑"
    )
    assert not poi_matches_must_visit(
        _poi("兵马俑旅游广场", "scenic", ["sightseeing"]), "兵马俑"
    )
    assert not poi_matches_must_visit(
        _poi("兵马俑枢纽", "scenic", ["sightseeing"]), "兵马俑"
    )
    confirmed_alias = _poi("秦始皇兵马俑博物馆", "museum", ["history"])
    confirmed_alias = POI(**{**confirmed_alias.__dict__, "aliases": ["兵马俑"]})
    assert poi_matches_must_visit(confirmed_alias, "兵马俑")


def test_multiday_day_with_only_one_activity_and_meal_is_sparse() -> None:
    profile = TravelProfile(destination="北京", days=3)
    itinerary = Itinerary(
        city="北京",
        summary="test",
        days=[
            ItineraryDay(
                day_index=index,
                theme="test",
                stops=[
                    ItineraryStop(_poi(f"景点{index}", "scenic", []), "09:30", 90, ""),
                    ItineraryStop(_poi(f"餐厅{index}", "food", ["food"]), "12:00", 60, ""),
                ],
            )
            for index in range(1, 4)
        ],
    )

    result = critique_itinerary(itinerary, profile)

    assert sum(issue.code == "daily_activity_sparse" for issue in result.issues) == 3


def test_generic_plan_without_specific_restaurant_is_not_an_error() -> None:
    profile = TravelProfile(destination="天津", days=1)
    itinerary = Itinerary(
        city="天津",
        summary="天津一日",
        days=[ItineraryDay(
            day_index=1,
            theme="citywalk",
            stops=[
                ItineraryStop(_poi("五大道", "scenic", []), "09:30", 120, ""),
                ItineraryStop(_poi("意式风情区", "scenic", []), "14:00", 120, ""),
            ],
        )],
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is True
    assert "daily_meal_missing" not in {issue.code for issue in result.issues}


def test_ticket_center_cannot_satisfy_museum_identity() -> None:
    from travel_agent.critic import poi_matches_must_visit

    assert not poi_matches_must_visit(
        _poi("天津博物馆-票务中心", "scenic", ["sightseeing"]), "天津博物馆"
    )


def _poi(name: str, category: str, tags: list[str]) -> POI:
    return POI(
        poi_id=name,
        name=name,
        city="北京",
        category=category,
        lat=39.9,
        lng=116.4,
        rating=4.5,
        popularity=0.8,
        tags=tags,
        estimated_duration_min=90,
        price_level="mid",
    )
