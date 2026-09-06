from dataclasses import replace

import pytest

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.critic import critique_itinerary, poi_matches_must_visit
from travel_agent.poi_evidence import (
    canonical_entity_match_evidence,
    normalize_candidate_requirement,
    normalize_poi_entity,
    partition_verified_candidates,
)
from travel_agent.route_evidence import (
    SHORT_WALK_DISTANCE_KM,
    canonical_route_evidence_status,
    itinerary_route_violations,
    normalize_route_evidence,
    route_can_prove_hard_feasibility,
    route_supports_endpoints,
)
from travel_agent.plan_invariants import validate_plan_artifact
from travel_agent.planning import rebind_itinerary_routes
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    RouteInfo,
    TravelProfile,
)


def _poi(
    poi_id: str,
    name: str,
    *,
    category: str = "scenic",
    entity_type: str = "attraction",
    status: str = "verified",
    source: str = "provider",
) -> POI:
    return POI(
        poi_id=poi_id,
        name=name,
        city="测试城",
        category=category,
        lat=30.0,
        lng=120.0,
        rating=4.5,
        popularity=0.8,
        tags=[],
        estimated_duration_min=90,
        price_level="mid",
        source=source,
        canonical_name=name,
        entity_type=entity_type,
        source_poi_id=poi_id,
        verification_status=status,
    )


@pytest.mark.parametrize(
    ("name", "entity_type"),
    [
        ("自然美学美容院", "beauty_service"),
        ("海滨汽车美容中心", "automotive_service"),
        ("博物馆文创商店", "retail"),
    ],
)
def test_provider_non_attraction_entities_never_enter_activity_candidates(name, entity_type) -> None:
    verified, rejected = partition_verified_candidates([_poi("x", name, entity_type=entity_type)])
    assert verified == []
    assert rejected[0].verification_status == "wrong_entity"


def test_legitimate_attraction_is_not_rejected_by_name_vocabulary() -> None:
    poi = _poi("coast", "蓝湾自然保护地", entity_type="attraction")
    verified, rejected = partition_verified_candidates([poi])
    assert verified == [poi]
    assert rejected == []


@pytest.mark.parametrize(
    "name",
    ["星巴克臻选店", "滨海咖啡馆", "旧城主题餐厅", "瑞幸咖啡中心店"],
)
def test_food_business_name_conflict_rejects_provider_scenic_taxonomy(name) -> None:
    verified, rejected = partition_verified_candidates([_poi("food-business", name)])

    assert verified == []
    assert rejected[0].verification_status == "wrong_entity"
    assert "餐饮商业实体" in str(rejected[0].verification_reason)


def test_cafe_museum_name_is_not_rejected_as_food_business() -> None:
    poi = _poi("coffee-museum", "咖啡文化博物馆", category="museum", entity_type="museum")
    verified, rejected = partition_verified_candidates([poi])

    assert verified == [poi]
    assert rejected == []


@pytest.mark.parametrize("name", ["城市博物馆（建设中）", "新馆在建暂未开放"])
def test_construction_or_unopened_entity_is_not_plannable(name) -> None:
    verified, rejected = partition_verified_candidates([
        _poi("future", name, category="museum", entity_type="museum")
    ])

    assert verified == []
    assert rejected[0].verification_status == "closed"


def test_candidate_requirement_removes_trip_request_suffix_not_entity_name() -> None:
    assert normalize_candidate_requirement("省博物馆一日游") == "省博物馆"
    assert normalize_candidate_requirement("古城参观") == "古城"
    assert normalize_candidate_requirement("一日游博物馆") == "一日游博物馆"


def test_main_museum_branch_suffix_covers_parent_venue_name() -> None:
    museum = _poi(
        "museum-main",
        "甲乙博物馆(本馆)",
        category="museum",
        entity_type="museum",
    )

    evidence = canonical_entity_match_evidence(museum, "甲乙博物馆")

    assert evidence is not None
    assert evidence["match_method"] in {"canonical_name_variant", "qualified_containment"}
    assert poi_matches_must_visit(museum, "甲乙博物馆")


def test_formal_museum_prefix_can_cover_specific_colloquial_venue_name() -> None:
    museum = _poi(
        "formal-museum",
        "古王朝青铜遗址博物馆",
        category="museum",
        entity_type="museum",
    )

    evidence = canonical_entity_match_evidence(museum, "青铜遗址")

    assert evidence is not None
    assert poi_matches_must_visit(museum, "青铜遗址")


def test_city_named_main_museum_preserves_specific_city_identity() -> None:
    museum = replace(
        _poi(
            "museum-city-main",
            "甲城博物馆(本馆)",
            category="museum",
            entity_type="museum",
        ),
        city="甲城市",
    )

    assert canonical_entity_match_evidence(museum, "甲城博物馆") is not None
    assert poi_matches_must_visit(museum, "甲城博物馆")


def test_region_museum_name_does_not_match_unrelated_specialty_museum() -> None:
    geology = replace(
        _poi(
            "museum-geology",
            "甲省地质博物馆",
            category="museum",
            entity_type="museum",
        ),
        city="乙城市",
    )

    assert canonical_entity_match_evidence(geology, "甲省博物院") is None


def test_museum_identity_accepts_optional_administrative_suffix() -> None:
    museum = replace(
        _poi(
            "museum-province",
            "甲乙省博物馆",
            category="museum",
            entity_type="museum",
        ),
        city="测试城",
    )

    evidence = canonical_entity_match_evidence(museum, "甲乙博物院")

    assert evidence is not None
    assert evidence["match_method"] == "administrative_museum_variant"


def test_specialty_museum_does_not_cover_general_regional_museum() -> None:
    geology = replace(
        _poi(
            "museum-geology",
            "甲乙省地质博物馆",
            category="museum",
            entity_type="museum",
        ),
        city="测试城",
    )

    assert canonical_entity_match_evidence(geology, "甲乙省博物馆") is None


def test_museum_named_after_scenic_area_does_not_cover_the_area() -> None:
    museum = _poi(
        "lake-museum",
        "甲乙湖博物馆",
        category="museum",
        entity_type="museum",
    )

    assert canonical_entity_match_evidence(museum, "甲乙湖") is None


def test_direct_formal_museum_suffix_can_cover_unqualified_venue_name() -> None:
    museum = _poi(
        "palace-museum",
        "甲乙宫博物院",
        category="museum",
        entity_type="museum",
    )

    assert canonical_entity_match_evidence(museum, "甲乙宫") is not None


def test_natural_landmark_is_not_covered_by_same_prefixed_district_venue() -> None:
    hall = _poi(
        "heritage-hall",
        "甲乙湖区非物质文化遗产馆",
        category="museum",
        entity_type="museum",
    )
    square = _poi(
        "culture-square",
        "甲乙湖文化广场",
        category="scenic",
        entity_type="attraction",
    )
    scenic_area = _poi(
        "scenic-area",
        "甲乙湖风景名胜区",
        category="scenic",
        entity_type="attraction",
    )

    assert canonical_entity_match_evidence(hall, "甲乙湖") is None
    assert canonical_entity_match_evidence(square, "甲乙湖") is None
    assert canonical_entity_match_evidence(scenic_area, "甲乙湖") is not None


@pytest.mark.parametrize("annex_type", ["ticket_office", "parking", "visitor_center"])
def test_annex_entity_cannot_satisfy_venue_body(annex_type) -> None:
    annex = normalize_poi_entity(_poi("annex", "科学馆附属设施", entity_type=annex_type))
    assert not poi_matches_must_visit(annex, "科学馆")


def test_confirmed_alias_and_exact_name_cover_must_visit_but_substring_does_not() -> None:
    exact = _poi("exact", "城市历史馆", category="museum", entity_type="museum")
    alias = replace(exact, poi_id="alias", canonical_name="市立历史博物院", aliases=["城市历史馆"])
    substring = replace(exact, poi_id="shop", name="城市历史馆纪念品店", canonical_name="城市历史馆纪念品店")
    assert poi_matches_must_visit(exact, "城市历史馆")
    assert poi_matches_must_visit(alias, "城市历史馆")
    assert not poi_matches_must_visit(substring, "城市历史馆")


def test_scenic_subvenue_does_not_impersonate_requested_parent_landmark() -> None:
    child = replace(
        _poi("child", "甲城云湖风景名胜区-名人故居"),
        parent_poi_id="parent-provider-id",
    )

    assert canonical_entity_match_evidence(child, "云湖") is None
    assert not poi_matches_must_visit(child, "云湖")


def test_access_road_does_not_satisfy_named_scenic_requirement() -> None:
    road = _poi("road", "青铜遗址连接线")

    assert canonical_entity_match_evidence(road, "青铜遗址") is None
    assert not poi_matches_must_visit(road, "青铜遗址")


def test_explicit_child_can_cover_parent_but_not_mechanically_equal_unrelated_parent() -> None:
    child = replace(
        _poi("child", "湿地科普馆", category="museum", entity_type="museum"),
        parent_poi_id="wetland",
        parent_canonical_name="北岸湿地公园",
        coverage_relation="child_covers_parent",
    )
    assert poi_matches_must_visit(child, "北岸湿地公园")
    assert not poi_matches_must_visit(child, "南岸湿地公园")


@pytest.mark.parametrize(
    ("provider_name", "required_name", "category", "entity_type"),
    [
        ("测试城云山风景名胜区", "云山", "scenic", "attraction"),
        ("测试城市自然博物馆东馆", "测试城自然博物馆", "museum", "museum"),
    ],
)
def test_provider_canonical_name_variants_match_with_auditable_evidence(
    provider_name: str,
    required_name: str,
    category: str,
    entity_type: str,
) -> None:
    poi = _poi("canonical-id", provider_name, category=category, entity_type=entity_type)
    evidence = canonical_entity_match_evidence(poi, required_name)
    assert evidence is not None
    assert evidence["requested_name"] == required_name
    assert evidence["original_name"] == provider_name
    assert evidence["canonical_id"] == "canonical-id"
    assert evidence["match_method"] == "canonical_name_variant"
    assert poi_matches_must_visit(poi, required_name)


@pytest.mark.parametrize(
    ("provider_name", "required_name", "category", "entity_type"),
    [
        ("测试城西湖风景区", "湖", "scenic", "attraction"),
        ("测试城市历史博物馆", "博物馆", "museum", "museum"),
        ("测试城市自然博物馆", "别城市自然博物馆", "museum", "museum"),
    ],
)
def test_broad_or_wrong_city_names_do_not_match_by_containment(
    provider_name: str,
    required_name: str,
    category: str,
    entity_type: str,
) -> None:
    poi = _poi("canonical-id", provider_name, category=category, entity_type=entity_type)
    assert canonical_entity_match_evidence(poi, required_name) is None
    assert not poi_matches_must_visit(poi, required_name)


def test_closed_and_evidence_insufficient_candidates_are_not_plannable() -> None:
    closed = _poi("closed", "城市展览馆", status="closed")
    unknown = _poi("unknown", "山海体验中心", status="evidence_insufficient")
    verified, rejected = partition_verified_candidates([closed, unknown])
    assert verified == []
    assert {poi.verification_status for poi in rejected} == {"closed", "evidence_insufficient"}


def _route(origin: str, destination: str, *, distance: float = 2.0, source: str = "provider") -> RouteInfo:
    return normalize_route_evidence(RouteInfo(
        origin_poi_id=origin,
        destination_poi_id=destination,
        distance_km=distance,
        duration_min=20,
        mode="public_transport",
        source=source,
    ))


def test_route_evidence_is_bound_to_exact_ordered_endpoint_ids() -> None:
    route = _route("garden-a", "garden-b")
    assert route_supports_endpoints(route, "garden-a", "garden-b")
    assert not route_supports_endpoints(route, "garden-c", "garden-d")
    assert not route_supports_endpoints(route, "garden-b", "garden-a")


def test_final_adjacent_stop_route_mismatch_is_error() -> None:
    a, b = _poi("a", "甲馆"), _poi("b", "乙馆")
    itinerary = Itinerary("测试城", [ItineraryDay(1, "", [
        ItineraryStop(a, "09:30", 60, ""),
        ItineraryStop(b, "11:00", 60, "", _route("old-a", "old-b")),
    ])], "")
    violations = itinerary_route_violations(itinerary)
    assert violations[0][0] == "route_endpoint_mismatch"
    result = critique_itinerary(itinerary, TravelProfile(destination="测试城", days=1))
    assert any(issue.code == "route_endpoint_mismatch" and issue.severity == "error" for issue in result.issues)


def test_replacing_stop_invalidates_old_route_and_rebinds_final_ids() -> None:
    a, old_b, new_b = _poi("a", "甲馆"), _poi("old-b", "旧乙馆"), _poi("new-b", "新乙馆")
    itinerary = Itinerary("测试城", [ItineraryDay(1, "", [
        ItineraryStop(a, "09:30", 60, ""),
        ItineraryStop(new_b, "11:00", 60, "", _route("a", "old-b")),
    ])], "")

    class Estimator:
        def estimate_route(self, origin, destination, mode):
            return _route(origin.poi_id, destination.poi_id, source="amap")

    rebound = rebind_itinerary_routes(
        itinerary, TravelProfile(destination="测试城", days=1), Estimator()
    )
    route = rebound.days[0].stops[1].route_from_previous
    assert route_supports_endpoints(route, "a", "new-b")
    assert not route_supports_endpoints(route, "a", old_b.poi_id)


def test_walking_cap_rebinds_transit_with_unknown_walk_to_verified_taxi() -> None:
    a, b = _poi("a", "甲馆"), _poi("b", "乙馆")
    itinerary = Itinerary("测试城", [ItineraryDay(1, "", [
        ItineraryStop(a, "09:30", 60, ""),
        ItineraryStop(b, "11:00", 60, "", _route("a", "b", source="haversine_recovery_estimate")),
    ])], "")

    class Estimator:
        calls: list[str] = []

        def estimate_route(self, origin, destination, mode):
            self.calls.append(mode)
            if mode == "taxi":
                return normalize_route_evidence(RouteInfo(
                    origin_poi_id=origin.poi_id,
                    destination_poi_id=destination.poi_id,
                    distance_km=2.0,
                    duration_min=8,
                    mode="taxi",
                    source="amap",
                ))
            return _route(origin.poi_id, destination.poi_id, source="amap")

    estimator = Estimator()
    rebound = rebind_itinerary_routes(
        itinerary,
        TravelProfile(
            destination="测试城",
            days=1,
            constraint_state={"max_walking_km_per_day": 2},
        ),
        estimator,
    )

    route = rebound.days[0].stops[1].route_from_previous
    assert estimator.calls == ["public_transport", "taxi"]
    assert route is not None
    assert route.mode == "taxi"
    assert route.source == "amap"
    assert route.evidence_status == "provider_verified"


@pytest.mark.parametrize("context", ["fixed_appointment", "return_deadline", "accessibility", "intercity"])
def test_haversine_route_cannot_prove_hard_feasibility(context) -> None:
    estimate = _route("a", "b", source="haversine_recovery_estimate")
    assert estimate.evidence_status == "haversine_estimate"
    assert not route_can_prove_hard_feasibility(estimate, context)


def test_provider_route_is_legal_hard_feasibility_counterexample() -> None:
    assert route_can_prove_hard_feasibility(_route("a", "b", source="amap"), "fixed_appointment")


def test_provider_verified_requires_complete_actual_route_evidence() -> None:
    complete = {
        "origin_poi_id": "a",
        "destination_poi_id": "b",
        "duration_min": 20,
        "distance_km": 3.2,
        "source": "amap",
        "evidence_status": "provider_verified",
    }
    assert canonical_route_evidence_status(complete) == "provider_verified"
    assert canonical_route_evidence_status({**complete, "duration_min": 0}) == "unavailable"
    assert canonical_route_evidence_status({**complete, "evidence_status": "unavailable"}) == "unavailable"


def test_provider_verified_empty_anchor_fails_with_explicit_reason_code() -> None:
    profile = TravelProfile(destination="测试城", days=1)
    payload = {
        "itinerary": {"days": [{"day_index": 1, "stops": []}]},
        "critic": {"passed": True, "issues": []},
        "required_route_anchors": {
            "legs": [{
                "kind": "fixed_event_transfer",
                "required_name": "预约地点",
                "evidence_status": "provider_verified",
                "routes": [],
            }]
        },
    }

    validation = validate_plan_artifact(payload, profile)
    codes = {item["code"] for item in validation["issues"]}
    assert "provider_verified_routes_empty" in codes
    assert "provider_verified_route_evidence_missing" in codes


def test_fixed_event_plan_and_required_anchor_conflict_fails_closed() -> None:
    event = {"day": 1, "start": "14:00", "end": "15:00", "location": "预约地点"}
    profile = TravelProfile(
        destination="测试城", days=1, constraint_state={"fixed_events": [event]}
    )
    payload = {
        "itinerary": {"days": [{"day_index": 1, "stops": []}]},
        "critic": {"passed": True, "issues": []},
        "fixed_event_plan": {"events": [{
            **event,
            "route_evidence_status": "provider_verified",
            "route_evidence_reference": None,
            "recommended_departure": "13:00",
            "required_buffer_min": 15,
        }]},
        "required_route_anchors": {"legs": [{
            "kind": "fixed_event_transfer",
            "required_name": "预约地点",
            "evidence_status": "unavailable",
            "routes": [],
        }]},
    }

    validation = validate_plan_artifact(payload, profile)
    codes = {item["code"] for item in validation["issues"]}
    assert "fixed_event_route_status_mismatch" in codes
    assert "fixed_event_provider_evidence_missing" in codes


def test_short_unverified_mode_is_normalized_to_walk_at_configured_boundary() -> None:
    below = _route("a", "b", distance=SHORT_WALK_DISTANCE_KM, source="haversine_estimate")
    above = _route("a", "b", distance=SHORT_WALK_DISTANCE_KM + 0.01, source="haversine_estimate")
    explicit = _route("a", "b", distance=SHORT_WALK_DISTANCE_KM, source="amap")
    assert below.mode == "walk"
    assert above.mode == "public_transport"
    assert explicit.mode == "public_transport"


def test_concrete_hotel_gets_daily_first_and_last_route_anchors() -> None:
    profile = TravelProfile(destination="测试城", days=2)
    itinerary = {
        "days": [
            {"day_index": day, "stops": [
                {"start_time": "09:30", "poi": _poi(f"first-{day}", f"首站{day}").__dict__},
                {"start_time": "14:00", "poi": _poi(f"last-{day}", f"末站{day}").__dict__},
            ]}
            for day in (1, 2)
        ]
    }
    lodging = {"status": "recommended_not_booked", "hotel": {
        "hotel_id": "hotel-1", "name": "中心酒店", "city": "测试城",
        "lat": 30.1, "lng": 120.1, "source": "provider", "rating": 4.5,
    }}

    class Estimator:
        def estimate_route(self, origin, destination, mode):
            return _route(origin.poi_id, destination.poi_id, source="amap")

    anchors = toolkit._build_lodging_route_anchors(profile, itinerary, lodging, Estimator())
    assert anchors["status"] == "concrete_hotel_anchored"
    assert len(anchors["daily_routes"]) == 2
    for day in anchors["daily_routes"]:
        assert day["legs"][0]["origin_poi_id"] == "hotel-1"
        assert day["legs"][1]["destination_poi_id"] == "hotel-1"
        assert day["recommended_hotel_departure"] is not None
    assert anchors["main_activity_area_conflict"] is False


def test_area_only_lodging_does_not_fabricate_exact_routes() -> None:
    profile = TravelProfile(destination="测试城", days=2, hotel_area="中心区")
    anchors = toolkit._build_lodging_route_anchors(
        profile, {"days": []}, {"status": "evidence_unavailable"}, object()
    )
    assert anchors == {"status": "area_anchor_only", "daily_routes": []}


def test_origin_and_return_anchors_require_exact_endpoint_route_pairs() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "origin": "中央车站",
            "return_location": "中央车站",
            "return_deadline": "19:00",
        },
    )
    itinerary = {"days": [{"stops": [
        {"poi": {"poi_id": "a", "name": "甲馆"}},
        {"poi": {"poi_id": "b", "name": "乙馆"}},
    ]}]}
    domain = {"transport": [
        {"payload": {
            "origin_poi_id": "station", "destination_poi_id": "a",
            "origin_name": "中央车站", "destination_name": "甲馆", "source": "amap",
            "duration_min": 20, "distance_km": 3.0,
        }},
        {"payload": {
            "origin_poi_id": "b", "destination_poi_id": "station",
            "origin_name": "乙馆", "destination_name": "中央车站", "source": "amap",
            "duration_min": 20, "distance_km": 3.0,
        }},
    ]}
    anchors, issues = toolkit._build_required_route_anchors(profile, itinerary, domain)
    assert issues == []
    assert [(leg["origin_poi_id"], leg["destination_poi_id"]) for leg in anchors["legs"]] == [
        ("station", "a"), ("b", "station")
    ]


def test_return_deadline_fails_closed_when_only_unrelated_route_exists() -> None:
    profile = TravelProfile(
        destination="测试城", days=1,
        constraint_state={"return_location": "中央车站", "return_deadline": "19:00"},
    )
    itinerary = {"days": [{"stops": [{"poi": {"poi_id": "b", "name": "乙馆"}}]}]}
    domain = {"transport": [{"payload": {
        "origin_poi_id": "x", "destination_poi_id": "station",
        "origin_name": "别处", "destination_name": "中央车站", "source": "amap",
    }}]}
    anchors, issues = toolkit._build_required_route_anchors(profile, itinerary, domain)
    assert anchors["legs"][0]["evidence_status"] == "unavailable"
    assert [issue.code for issue in issues] == ["return_route_evidence_insufficient"]


def test_required_route_anchor_prefers_new_verified_repair_over_old_fallback() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"return_location": "中央车站", "return_deadline": "19:00"},
    )
    itinerary = {
        "days": [{
            "day_index": 1,
            "stops": [{"poi": {"poi_id": "final", "name": "最终景点"}}],
        }]
    }
    fallback = {
        "origin_poi_id": "final",
        "destination_poi_id": "station",
        "origin_name": "最终景点",
        "destination_name": "中央车站",
        "distance_km": 2.0,
        "duration_min": 20,
        "mode": "public_transport",
        "source": "haversine_recovery_estimate",
        "evidence_status": "haversine_estimate",
    }
    repaired = {
        **fallback,
        "duration_min": 24,
        "source": "amap",
        "evidence_status": "provider_verified",
        "recommended_latest_departure": "18:06",
        "required_buffer_min": 30,
    }
    domain = {"transport": [{"payload": fallback}, {"payload": repaired}]}

    anchors, issues = toolkit._build_required_route_anchors(
        profile, itinerary, domain
    )

    assert issues == []
    closure = anchors["last_stop_to_return_location"]
    assert closure["source"] == "amap"
    assert closure["duration_min"] == 24
    assert closure["evidence_status"] == "provider_verified"


def test_post_plan_return_closure_binds_actual_final_stop_and_deadline_buffer() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "return_location": "中央车站",
            "return_deadline": "2026-10-01T19:00+08:00",
        },
    )
    final_stop = _poi("planner-final", "最终景点")
    station = _poi(
        "station-1",
        "中央车站(地铁站)",
        category="transport",
        entity_type="transport",
        status="wrong_entity",
        source="amap",
    )

    class Provider:
        searches = 0

        def search_pois(self, **kwargs):
            assert kwargs["max_results"] == 5
            self.searches += 1
            return [station]

        def estimate_route(self, origin, destination, mode):
            assert origin.poi_id == "planner-final"
            assert destination.poi_id == "station-1"
            return _route(origin.poi_id, destination.poi_id, source="amap")

    provider = Provider()
    ctx.provider = provider
    domain = {"transport": []}
    route = toolkit._close_post_plan_return_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        domain,
        provider,
    )

    assert provider.searches == 1
    assert route["origin_poi_id"] == "planner-final"
    assert route["destination_poi_id"] == "station-1"
    assert route["evidence_status"] == "provider_verified"
    assert route["recommended_latest_departure"] == "2026-10-01T18:10+08:00"
    assert route["required_buffer_min"] == 30
    anchors, issues = toolkit._build_required_route_anchors(
        ctx.profile,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        domain,
    )
    closure = anchors["last_stop_to_return_location"]
    assert issues == []
    assert closure["origin_poi_id"] == "planner-final"
    assert closure["destination_poi_id"] == "station-1"
    assert closure["duration_min"] == 20
    assert closure["distance_km"] == 2.0


def test_broad_return_city_reuses_more_specific_origin_endpoint() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "origin": "甲城中央车站",
            "return_location": "甲城",
            "return_deadline": "21:30",
        },
    )
    final_stop = _poi("planner-final", "最终景点")
    station = replace(
        _poi(
            "station-1",
            "甲城中央车站",
            category="transport",
            entity_type="transport",
            status="wrong_entity",
            source="amap",
        ),
        city="甲城市",
    )

    class Provider:
        queries: list[list[str]] = []

        def search_pois(self, **kwargs):
            self.queries.append(kwargs["query_tags"])
            return [station]

        def estimate_route(self, origin, destination, mode):
            return _route(origin.poi_id, destination.poi_id, source="amap")

    provider = Provider()
    ctx.provider = provider
    domain = {"transport": []}

    route = toolkit._close_post_plan_return_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        domain,
        provider,
    )

    assert provider.queries == [["甲城中央车站"]]
    assert route is not None
    assert route["destination_poi_id"] == "station-1"
    assert route["return_location_name"] == "甲城"


def test_cross_city_return_retries_live_transit_after_bound_estimator_is_unverified() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="乙城",
        days=1,
        constraint_state={
            "origin": "甲城中央车站",
            "return_location": "甲城",
            "return_deadline": "21:30",
        },
    )
    final_stop = replace(_poi("planner-final", "最终景点"), city="乙城市")
    station = replace(
        _poi(
            "station-1",
            "甲城中央车站",
            category="transport",
            entity_type="transport",
            status="wrong_entity",
            source="amap",
        ),
        city="甲城市",
    )

    class LiveProvider:
        calls: list[str] = []

        def search_pois(self, **_kwargs):
            return [station]

        def estimate_route(self, origin, destination, mode):
            self.calls.append(mode)
            return replace(
                _route(origin.poi_id, destination.poi_id, source="amap"),
                mode=mode,
                duration_min=95,
            )

    class BoundEstimator:
        def estimate_route(self, origin, destination, mode):
            return replace(
                _route(
                    origin.poi_id,
                    destination.poi_id,
                    source="haversine_recovery_estimate",
                ),
                mode=mode,
                duration_min=500,
            )

    provider = LiveProvider()
    ctx.provider = provider
    route = toolkit._close_post_plan_return_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        {"transport": []},
        BoundEstimator(),
    )

    assert provider.calls == ["public_transport"]
    assert route is not None
    assert route["source"] == "amap"
    assert route["evidence_status"] == "provider_verified"
    assert route["duration_min"] == 95


def test_cross_city_origin_retries_live_transit_after_bound_estimator_is_unverified() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="乙城",
        days=1,
        constraint_state={"origin": "甲城中央车站"},
    )
    first_stop = replace(_poi("planner-first", "首个景点"), city="乙城市")
    station = replace(
        _poi(
            "station-1",
            "甲城中央车站",
            category="transport",
            entity_type="transport",
            status="wrong_entity",
            source="amap",
        ),
        city="甲城市",
    )

    class LiveProvider:
        calls: list[str] = []

        def search_pois(self, **_kwargs):
            return [station]

        def estimate_route(self, origin, destination, mode):
            self.calls.append(mode)
            return replace(
                _route(origin.poi_id, destination.poi_id, source="amap"),
                mode=mode,
                duration_min=88,
            )

    class BoundEstimator:
        def estimate_route(self, origin, destination, mode):
            return replace(
                _route(
                    origin.poi_id,
                    destination.poi_id,
                    source="haversine_recovery_estimate",
                ),
                mode=mode,
                duration_min=500,
            )

    provider = LiveProvider()
    ctx.provider = provider
    route = toolkit._close_post_plan_origin_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": first_stop.__dict__}]}]},
        {"transport": []},
        BoundEstimator(),
    )

    assert provider.calls == ["public_transport"]
    assert route is not None
    assert route["source"] == "amap"
    assert route["evidence_status"] == "provider_verified"
    assert route["duration_min"] == 88
def test_same_city_return_retries_verified_taxi_when_transit_is_unverified() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "return_location": "中央车站",
            "return_deadline": "19:00",
        },
    )
    final_stop = _poi("planner-final", "最终景点")
    station = _poi(
        "station-1",
        "中央车站",
        category="transport",
        entity_type="transport",
        status="wrong_entity",
        source="amap",
    )

    class Provider:
        calls: list[str] = []

        def search_pois(self, **_kwargs):
            return [station]

        def estimate_route(self, origin, destination, mode):
            self.calls.append(mode)
            route = _route(
                origin.poi_id,
                destination.poi_id,
                source="amap" if mode == "taxi" else "haversine_recovery_estimate",
            )
            return replace(route, mode=mode)

    provider = Provider()
    ctx.provider = provider
    route = toolkit._close_post_plan_return_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        {"transport": []},
        provider,
    )

    assert provider.calls == ["public_transport", "taxi"]
    assert route is not None
    assert route["mode"] == "taxi"
    assert route["evidence_status"] == "provider_verified"


def test_same_city_return_replaces_existing_unverified_route_with_taxi() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "return_location": "中央车站",
            "return_deadline": "19:00",
        },
    )
    final_stop = _poi("planner-final", "最终景点")
    station = _poi(
        "station-1",
        "中央车站",
        category="transport",
        entity_type="transport",
        status="wrong_entity",
        source="amap",
    )
    ctx.remember_pois([station])

    class Provider:
        calls: list[str] = []

        def estimate_route(self, origin, destination, mode):
            self.calls.append(mode)
            return replace(
                _route(origin.poi_id, destination.poi_id, source="amap"),
                mode=mode,
            )

    provider = Provider()
    existing = {
        **_route(
            "planner-final", "station-1", source="haversine_recovery_estimate"
        ).__dict__,
        "destination_name": "中央车站",
    }
    route = toolkit._close_post_plan_return_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        {"transport": [{"payload": existing}]},
        provider,
    )

    assert provider.calls == ["public_transport"]
    assert route is not None
    assert route["source"] == "amap"
    assert route["evidence_status"] == "provider_verified"


def test_post_plan_return_rebinds_verified_route_for_same_named_fresh_origin_id() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "return_location": "中央车站",
            "return_deadline": "19:00",
        },
    )
    final_stop = _poi("fresh-final-id", "最终景点")
    station = _poi(
        "station-1",
        "中央车站",
        category="transport",
        entity_type="transport",
        status="wrong_entity",
        source="amap",
    )
    ctx.remember_pois([station])
    existing = {
        **_route("old-final-id", "station-1", source="amap").__dict__,
        "origin_name": "最终景点",
        "destination_name": "中央车站",
    }

    class Provider:
        def estimate_route(self, *_args, **_kwargs):
            raise AssertionError("verified same-identity route must be reused")

    route = toolkit._close_post_plan_return_route(
        ctx,
        {"days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]},
        {"transport": [{"payload": existing}]},
        Provider(),
    )

    assert route is not None
    assert route["origin_poi_id"] == "fresh-final-id"
    assert route["destination_poi_id"] == "station-1"
    assert route["evidence_status"] == "provider_verified"


def test_post_plan_return_closure_trims_late_optional_tail_and_rebinds_route() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "return_location": "中央车站",
            "return_deadline": "17:00",
        },
    )
    prior = _poi("prior-stop", "午后公园")
    late = _poi("late-stop", "可选夜市")
    station = _poi(
        "station-1",
        "中央车站",
        category="transport",
        entity_type="transport",
        status="wrong_entity",
        source="amap",
    )
    ctx.remember_pois([station])

    class Provider:
        origins: list[str] = []

        def estimate_route(self, origin, destination, mode):
            self.origins.append(origin.poi_id)
            return _route(origin.poi_id, destination.poi_id, source="amap")

    provider = Provider()
    itinerary = {
        "days": [{
            "day_index": 1,
            "stops": [
                {"poi": prior.__dict__, "start_time": "14:00", "duration_min": 60},
                {"poi": late.__dict__, "start_time": "16:30", "duration_min": 60},
            ],
        }]
    }
    domain = {"transport": []}

    route = toolkit._close_post_plan_return_route(
        ctx, itinerary, domain, provider
    )

    assert [stop["poi"]["poi_id"] for stop in itinerary["days"][0]["stops"]] == [
        "prior-stop"
    ]
    assert provider.origins == ["late-stop", "prior-stop"]
    assert route is not None
    assert route["origin_poi_id"] == "prior-stop"
    assert route["destination_poi_id"] == "station-1"
    assert route["hard_feasibility_proven"] is True


def test_post_plan_return_endpoint_lookup_failure_stays_fail_closed() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={
            "return_location": "中央车站",
            "return_deadline": "19:00",
        },
    )
    final_stop = _poi("planner-final", "最终景点")

    class Provider:
        searches = 0

        def search_pois(self, **_kwargs):
            self.searches += 1
            return [_poi("unrelated", "无关地点")]

        def estimate_route(self, *_args, **_kwargs):
            raise AssertionError("unresolved endpoint must not be routed")

    provider = Provider()
    ctx.provider = provider
    itinerary = {
        "days": [{"day_index": 1, "stops": [{"poi": final_stop.__dict__}]}]
    }
    domain = {"transport": []}

    assert toolkit._close_post_plan_return_route(
        ctx, itinerary, domain, provider
    ) is None
    assert provider.searches == 1
    anchors, issues = toolkit._build_required_route_anchors(
        ctx.profile, itinerary, domain
    )
    assert anchors["last_stop_to_return_location"]["origin_poi_id"] == "planner-final"
    assert anchors["last_stop_to_return_location"]["destination_poi_id"] is None
    assert [issue.code for issue in issues] == ["return_route_evidence_insufficient"]


def test_post_plan_origin_closure_binds_explicit_origin_to_actual_first_stop() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"origin": "上海虹桥"},
    )
    first = replace(_poi("west-lake", "西湖"), city="杭州")
    station = replace(
        _poi(
            "hongqiao",
            "上海虹桥站",
            category="transport",
            entity_type="transport",
            status="wrong_entity",
            source="amap",
        ),
        city="上海",
    )

    class Provider:
        def search_pois(self, **kwargs):
            assert kwargs["query_tags"] == ["上海虹桥"]
            return [station]

        def estimate_route(self, origin, destination, mode):
            assert (origin.poi_id, destination.poi_id) == ("hongqiao", "west-lake")
            return _route(origin.poi_id, destination.poi_id, source="amap")

    provider = Provider()
    ctx.provider = provider
    domain = {"transport": []}
    itinerary = {
        "days": [{"day_index": 1, "stops": [{"poi": first.__dict__}]}]
    }

    route = toolkit._close_post_plan_origin_route(
        ctx, itinerary, domain, provider
    )
    anchors, issues = toolkit._build_required_route_anchors(
        ctx.profile, itinerary, domain
    )

    assert route is not None
    assert route["evidence_status"] == "provider_verified"
    assert anchors["legs"][0]["origin_poi_id"] == "hongqiao"
    assert anchors["legs"][0]["destination_poi_id"] == "west-lake"
    assert issues == []


def test_post_plan_location_only_fixed_event_gets_grounded_transfer_without_fake_stop() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=2,
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
    prior = _poi("prior-stop", "城市博物馆", category="museum", entity_type="museum")
    area = _poi(
        "area-anchor",
        "江畔商圈(地铁站)",
        category="transport",
        entity_type="transport",
        status="wrong_entity",
        source="amap",
    )

    class Provider:
        calls: list[str] = []

        def search_pois(self, **kwargs):
            assert kwargs["query_tags"] == ["江畔商圈"]
            return [area]

        def estimate_route(self, origin, destination, mode):
            assert origin.poi_id == "prior-stop"
            assert destination.poi_id == "area-anchor"
            self.calls.append(mode)
            route = _route(
                origin.poi_id,
                destination.poi_id,
                source="amap" if mode == "taxi" else "haversine_recovery_estimate",
            )
            return replace(route, mode=mode)

    provider = Provider()
    ctx.provider = provider
    itinerary = {"days": [
        {"day_index": 1, "stops": []},
        {"day_index": 2, "stops": [{
            "poi": prior.__dict__, "start_time": "14:00", "duration_min": 90,
        }]},
    ]}
    domain = {"transport": []}

    routes = toolkit._close_post_plan_fixed_event_routes(
        ctx, itinerary, domain, provider
    )
    anchors, issues = toolkit._build_required_route_anchors(
        ctx.profile, itinerary, domain
    )

    assert len(routes) == 1
    assert provider.calls == ["public_transport", "taxi"]
    assert routes[0]["mode"] == "taxi"
    assert routes[0]["fixed_event_name"] == "江畔商圈"
    leg = anchors["fixed_event_transfer"]
    assert issues == []
    assert leg["event_poi_id"] == "area-anchor"
    assert leg["evidence_status"] == "provider_verified"
    assert not any(
        stop.get("poi", {}).get("poi_id") == "area-anchor"
        for day in itinerary["days"] for stop in day["stops"]
    )


def test_post_plan_location_only_fixed_event_reuses_session_verified_area_candidate() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=2,
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
    prior = _poi("prior-stop", "城市博物馆", category="museum", entity_type="museum")
    area = _poi("area-anchor", "江畔商圈中心绿地", source="amap")

    class Provider:
        calls: list[str] = []

        def search_pois(self, **_kwargs):
            raise AssertionError("bound verified candidate should be reused")

        def estimate_route(self, origin, destination, mode):
            assert origin.poi_id == "prior-stop"
            assert destination.poi_id == "area-anchor"
            self.calls.append(mode)
            return _route(origin.poi_id, destination.poi_id, source="amap")

    provider = Provider()
    ctx.provider = provider
    ctx.remember_pois([area])
    itinerary = {"days": [
        {"day_index": 1, "stops": []},
        {"day_index": 2, "stops": [{
            "poi": prior.__dict__, "start_time": "14:00", "duration_min": 90,
        }]},
    ]}
    domain = {"transport": []}

    routes = toolkit._close_post_plan_fixed_event_routes(
        ctx, itinerary, domain, provider
    )

    assert len(routes) == 1
    assert provider.calls == [ctx.profile.transport_mode]
    assert routes[0]["destination_poi_id"] == "area-anchor"
    assert routes[0]["evidence_status"] == "provider_verified"


def test_unspecified_fixed_event_anchor_exposes_exact_repair_pair() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=2,
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
    prior = _poi("prior-stop", "城市博物馆", category="museum", entity_type="museum")
    itinerary = {"days": [
        {"day_index": 2, "stops": [{
            "poi": prior.__dict__, "start_time": "14:00", "duration_min": 90,
        }]},
    ]}
    domain = {"transport": [{"payload": {
        **_route("unrelated", "area-anchor", source="amap").__dict__,
        "origin_name": "无关地点",
        "destination_name": "江畔商圈中心绿地",
    }}]}

    anchors, issues = toolkit._build_required_route_anchors(profile, itinerary, domain)

    leg = anchors["fixed_event_transfer"]
    assert leg["origin_poi_id"] == "prior-stop"
    assert leg["event_poi_id"] == "area-anchor"
    assert leg["evidence_status"] == "unavailable"
    assert [issue.code for issue in issues] == ["fixed_event_route_evidence_insufficient"]


def test_fixed_event_closure_prefers_repaired_predecessor_route_endpoint() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=2,
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
    prior = _poi("prior-stop", "城市博物馆", category="museum", entity_type="museum")
    station = _poi(
        "station", "江畔商圈(地铁站)", category="transport", entity_type="transport",
        source="amap",
    )
    area = _poi("area-anchor", "江畔商圈中心绿地", source="amap")
    ctx.remember_pois([station, area])

    class Estimator:
        def estimate_route(self, origin, destination, mode):
            assert origin.poi_id == "prior-stop"
            assert destination.poi_id == "area-anchor"
            return _route(origin.poi_id, destination.poi_id, source="amap")

    itinerary = {"days": [
        {"day_index": 2, "stops": [{
            "poi": prior.__dict__, "start_time": "14:00", "duration_min": 90,
        }]},
    ]}
    domain = {"transport": [{"payload": {
        **_route("prior-stop", "area-anchor", source="amap").__dict__,
        "origin_name": "城市博物馆",
        "destination_name": "江畔商圈中心绿地",
    }}]}

    routes = toolkit._close_post_plan_fixed_event_routes(
        ctx, itinerary, domain, Estimator()
    )

    assert len(routes) == 1
    assert routes[0]["destination_poi_id"] == "area-anchor"
    assert routes[0]["fixed_event_name"] == "江畔商圈"


def test_post_plan_named_fixed_event_binds_route_to_scheduled_venue() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=2,
        constraint_state={
            "fixed_events": [{
                "day": 2,
                "start": "14:30",
                "end": "16:30",
                "location": "历史博物馆",
            }],
        },
    )
    prior = _poi("prior-stop", "古城墙", category="scenic", entity_type="attraction")
    venue = _poi("event-venue", "历史博物馆", category="museum", entity_type="museum")

    class Provider:
        def search_pois(self, **kwargs):
            raise AssertionError("scheduled fixed-event venue should be reused")

        def estimate_route(self, origin, destination, mode):
            assert origin.poi_id == "prior-stop"
            assert destination.poi_id == "event-venue"
            return _route(origin.poi_id, destination.poi_id, source="amap")

    provider = Provider()
    ctx.provider = provider
    itinerary = {"days": [
        {"day_index": 1, "stops": []},
        {"day_index": 2, "stops": [
            {"poi": prior.__dict__, "start_time": "10:00", "duration_min": 90},
            {"poi": venue.__dict__, "start_time": "14:30", "duration_min": 120},
        ]},
    ]}
    wrong_origin_route = {
        **_route("unrelated-stop", "event-venue", source="amap").__dict__,
        "fixed_event_name": "历史博物馆",
    }
    domain = {"transport": [{"payload": wrong_origin_route}]}

    routes = toolkit._close_post_plan_fixed_event_routes(
        ctx, itinerary, domain, provider
    )
    anchors, issues = toolkit._build_required_route_anchors(
        ctx.profile, itinerary, domain
    )

    assert len(routes) == 1
    assert routes[0]["origin_poi_id"] == "prior-stop"
    assert routes[0]["destination_poi_id"] == "event-venue"
    assert routes[0]["fixed_event_name"] == "历史博物馆"
    assert anchors["fixed_event_transfer"]["evidence_status"] == "provider_verified"
    assert issues == []


def test_rebind_retries_low_confidence_route_for_final_exact_endpoints() -> None:
    first, second = _poi("a", "甲馆"), _poi("b", "乙馆")
    itinerary = Itinerary("测试城", [ItineraryDay(1, "", [
        ItineraryStop(first, "09:30", 60, ""),
        ItineraryStop(
            second,
            "11:00",
            60,
            "",
            _route("a", "b", source="haversine_recovery_estimate"),
        ),
    ])], "")

    class Estimator:
        calls = 0

        def estimate_route(self, origin, destination, mode):
            self.calls += 1
            return _route(origin.poi_id, destination.poi_id, source="amap")

    estimator = Estimator()
    rebound = rebind_itinerary_routes(
        itinerary, TravelProfile(destination="测试城", days=1), estimator
    )

    assert estimator.calls == 1
    assert rebound.days[0].stops[1].route_from_previous.source == "amap"
