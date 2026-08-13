from __future__ import annotations

import threading
import time
from pathlib import Path

from travel_agent.agent import toolkit
from travel_agent.agent.serde import poi_brief
from travel_agent.agent.session import build_session, session_tool_lock
from travel_agent.schemas import POI, ScoredPOI, TravelProfile

POI_PATH = Path(__file__).resolve().parents[1] / "data" / "seed" / "pois.json"


def test_poi_brief_caps_provider_text_fields() -> None:
    poi = POI(
        poi_id="p1",
        name="测试景点",
        city="上海",
        category="scenic",
        lat=31.2,
        lng=121.4,
        rating=4.5,
        popularity=0.8,
        tags=["classic"],
        estimated_duration_min=90,
        price_level="mid",
        opening_hours="开放说明" * 200,
        address="很长地址" * 100,
    )

    brief = poi_brief(poi)

    assert len(brief["opening_hours"]) == 240
    assert len(brief["address"]) == 120


def test_commercial_amenities_are_filtered_from_ranked_candidates():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="西安", days=1)
    artifact_id = ctx.store.put(
        "candidates",
        {
            "pois": [
                {
                    "poi_id": "shop",
                    "name": "兵马俑旅游纪念品店",
                    "city": "西安",
                    "category": "shopping",
                    "lat": 34.38,
                    "lng": 109.28,
                    "rating": 4.9,
                    "popularity": 1.0,
                    "tags": ["shopping"],
                    "estimated_duration_min": 60,
                    "price_level": "mid",
                },
                {
                    "poi_id": "museum",
                    "name": "秦始皇兵马俑博物馆",
                    "city": "西安",
                    "category": "museum",
                    "lat": 34.39,
                    "lng": 109.29,
                    "rating": 4.8,
                    "popularity": 0.9,
                    "tags": ["history"],
                    "estimated_duration_min": 90,
                    "price_level": "mid",
                },
            ]
        },
    )

    result = toolkit.recommend_candidates(ctx, artifact_ids=[artifact_id])
    ranked = ctx.store.get(result["artifact_id"])["pois"]

    assert [item["poi"]["name"] for item in ranked] == ["秦始皇兵马俑博物馆"]


def test_ranked_candidates_collapse_terracotta_subvenues_to_main_museum():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="西安", days=3)
    base = {
        "city": "西安",
        "lat": 34.38,
        "lng": 109.28,
        "rating": 4.8,
        "popularity": 0.9,
        "tags": ["history"],
        "estimated_duration_min": 90,
        "price_level": "mid",
    }
    artifact_id = ctx.store.put(
        "candidates",
        {
            "pois": [
                {**base, "poi_id": "main", "name": "秦始皇兵马俑博物馆", "category": "museum"},
                {**base, "poi_id": "pit1", "name": "秦兵马俑壹号坑大厅", "category": "scenic"},
                {**base, "poi_id": "pit3", "name": "秦兵马俑三号坑遗址", "category": "museum"},
            ]
        },
    )

    result = toolkit.recommend_candidates(ctx, artifact_ids=[artifact_id])
    names = [item["poi"]["name"] for item in ctx.store.get(result["artifact_id"])["pois"]]

    assert names == ["秦始皇兵马俑博物馆"]


def _session():
    return build_session(persist=False, poi_path=POI_PATH)


def test_full_tool_pipeline():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2, interests=["nature", "food"])

    search = toolkit.search_poi(ctx)
    assert search["isError"] is False
    assert search["count"] > 0

    weather = toolkit.check_weather(ctx)
    assert weather["isError"] is False

    rec = toolkit.recommend_candidates(ctx)
    assert rec["isError"] is False
    assert rec["count"] > 0

    plan = toolkit.plan_and_critique(ctx)
    assert plan["isError"] is False
    assert plan["final_issue_count"] <= plan["original_issue_count"]

    cards = toolkit.render_itinerary(ctx)
    assert any(c["type"] == "day" for c in cards["cards"])

    map_payload = toolkit.render_map(ctx)
    assert len(map_payload["markers"]) > 0


def test_request_travel_info_when_missing():
    ctx = _session()
    toolkit.update_travel_profile(ctx, interests=["food"])
    info = toolkit.request_travel_info(ctx)
    assert "destination" in info["missing_fields"]
    assert info["question"]


def test_plan_requires_recommend_first():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2)
    result = toolkit.plan_and_critique(ctx)
    assert result["isError"] is True


def test_plan_merges_bound_restaurants_when_ranked_contains_only_attractions():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2)
    candidates = toolkit.search_poi(ctx)
    ranked = toolkit.recommend_candidates(ctx, artifact_ids=[candidates["artifact_id"]])
    restaurants = toolkit.search_restaurant(ctx)

    plan = toolkit.plan_and_critique(
        ctx,
        artifact_ids=[ranked["artifact_id"], restaurants["artifact_id"]],
    )

    assert plan["isError"] is False
    payload = ctx.store.get(plan["artifact_id"])
    assert all(
        any(stop["poi"]["category"] == "food" for stop in day["stops"])
        for day in payload["itinerary"]["days"]
    )


def test_planner_rejects_generic_candidate_food_without_restaurant_binding():
    generic_area = POI(
        "nanluo", "南锣鼓巷", "北京", "food", 39.93, 116.4,
        4.8, 0.9, ["food", "culture"], 90, "mid", source="seed",
    )
    restaurant = POI(
        "restaurant-1", "具体餐厅", "北京", "food", 39.92, 116.41,
        4.6, 0.8, ["food"], 60, "mid", source="amap",
    )
    ranked = {"pois": [toolkit._scored_to_dict(ScoredPOI(generic_area, 1.0, []))]}
    records = [{
        "kind": "restaurants",
        "artifact_id": "restaurants-1",
        "payload": {"restaurants": [toolkit.poi_to_dict(restaurant)]},
    }]

    merged = toolkit._merge_ranked_planner_inputs(
        ranked, records, TravelProfile(destination="北京", days=1),
    )

    assert [item.poi.name for item in merged] == ["具体餐厅"]


def test_halal_restaurant_search_falls_back_and_keeps_only_evidenced_matches(monkeypatch):
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="西安", days=1)
    ctx.profile.constraint_state = {"dietary": ["仅清真餐厅"]}
    halal = POI("halal", "清真测试餐厅", "西安", "food", 34.2, 108.9, 4.5, 0.8, ["halal"], 60, "mid")
    ordinary = POI("ordinary", "普通餐厅", "西安", "food", 34.2, 108.91, 4.6, 0.9, ["food"], 60, "mid")
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [ordinary] if kwargs.get("query_tags") else [ordinary, halal]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_restaurant(ctx, cuisine="清真")

    assert len(calls) == 2
    assert [item["name"] for item in result["restaurants"]] == ["清真测试餐厅"]


def test_halal_restaurant_search_broadens_when_exact_results_do_not_cover_trip(monkeypatch):
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="西安", days=3)
    ctx.profile.constraint_state = {"dietary": ["仅清真餐厅"]}
    exact = POI("exact", "清真精确餐厅", "西安", "food", 34.2, 108.9, 4.5, 0.8, ["halal"], 60, "mid")
    broad_1 = POI("broad-1", "清真候选一", "西安", "food", 34.21, 108.91, 4.4, 0.7, ["halal"], 60, "mid")
    broad_2 = POI("broad-2", "清真候选二", "西安", "food", 34.22, 108.92, 4.3, 0.6, ["halal"], 60, "mid")
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [exact] if kwargs.get("query_tags") else [exact, broad_1, broad_2]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_restaurant(ctx, cuisine="清真")

    assert len(calls) == 2
    assert [item["name"] for item in result["restaurants"]] == [
        "清真精确餐厅", "清真候选一", "清真候选二",
    ]


def test_halal_restaurant_search_uses_explicit_query_after_sparse_broad_search(monkeypatch):
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="西安", days=2)
    ctx.profile.constraint_state = {"dietary": ["仅清真餐厅"]}
    halal_1 = POI("h1", "清真餐厅一", "西安", "food", 34.2, 108.9, 4.5, 0.8, ["halal"], 60, "mid")
    halal_2 = POI("h2", "清真餐厅二", "西安", "food", 34.21, 108.91, 4.4, 0.7, ["halal"], 60, "mid")
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [halal_1, halal_2] if kwargs.get("query_tags") == ["清真餐厅"] else []

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_restaurant(ctx, cuisine="清真", area="兵马俑")

    assert len(calls) == 3
    assert [item["name"] for item in result["restaurants"]] == ["清真餐厅一", "清真餐厅二"]


def test_worker_hotel_area_hint_falls_back_to_grounded_city_results(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="西安", days=3)
    grounded = POI(
        "hotel-1", "西安大酒店", "西安", "hotel", 34.26, 108.95,
        4.6, 0.8, ["hotel"], 0, "mid", source="amap",
    )
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [] if kwargs.get("query_tags") else [grounded]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_hotel(ctx, area="钟楼")

    assert len(calls) == 2
    assert result["hotels"][0]["name"] == "西安大酒店"
    assert result["hotels"][0]["area"] is None


def test_build_return_plan_reserves_cross_city_buffer_without_inventing_inventory() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"return_deadline": "21:30", "return_location": "上海"},
    )
    itinerary = {
        "days": [{"stops": [{"poi": {"poi_id": "west-lake"}}]}],
    }
    route = {
        "origin_poi_id": "west-lake",
        "destination_poi_id": "hangzhou-station",
        "origin_name": "西湖",
        "destination_name": "杭州站",
        "duration_min": 30,
        "mode": "public_transport",
        "source": "amap",
    }
    return_route = {
        "origin_poi_id": "hangzhou-east",
        "destination_poi_id": "shanghai-hongqiao",
        "origin_name": "杭州东站",
        "destination_name": "上海虹桥站",
        "duration_min": 80,
        "mode": "public_transport",
        "source": "amap",
    }

    result = toolkit._build_return_plan(
        profile,
        itinerary,
        {"transport": [
            {"artifact_id": "route-1", "payload": route},
            {"artifact_id": "route-2", "payload": return_route},
        ]},
    )

    assert result is not None
    assert result["activity_cutoff"] == "18:30"
    assert result["arrival_deadline"] == "21:30"
    assert result["terminal_transfer"] == route
    assert result["intercity_segment"]["status"] == "route_estimate_only"
    assert result["intercity_segment"]["route_evidence"] == return_route
    assert "车次" not in result["intercity_segment"]


def test_lodging_and_budget_plans_select_grounded_hotel_and_compare_limit() -> None:
    profile = TravelProfile(destination="北京", days=2, party_size=2, budget_limit=4500)
    domain_inputs = {
        "hotels": [{
            "artifact_id": "hotel-1",
            "payload": {"hotels": [
                {"name": "虚拟酒店", "source": "mock", "rating": 5.0, "price_per_night": 200},
                {"name": "王府井大饭店", "source": "amap", "rating": 4.7, "price_per_night": 585},
            ]},
        }],
        "budgets": [{
            "artifact_id": "budget-1",
            "payload": {
                "days": 2, "companions": 2, "hotel": 550, "meals": 720,
                "tickets": 480, "inner_city_transport": 180,
                "total_low": 1640.5, "total_high": 2219.5,
            },
        }],
    }

    lodging = toolkit._build_lodging_plan(profile, domain_inputs)
    budget = toolkit._build_budget_plan(profile, domain_inputs, lodging)

    assert lodging["status"] == "recommended_not_booked"
    assert lodging["hotel"]["name"] == "王府井大饭店"
    assert lodging["lodging_subtotal_cny"] == 585
    assert budget["breakdown_cny"]["lodging"] == 585
    assert budget["total_expected_cny"] == 1965
    assert budget["total_high_cny"] == 2254.5
    assert budget["within_user_limit"] is True


def test_lodging_plan_fails_closed_when_grounded_hotel_misses_required_area() -> None:
    profile = TravelProfile(
        destination="厦门",
        days=4,
        hotel_area="思明区",
        constraint_state={"lodging_area": "思明区"},
    )
    domain_inputs = {"hotels": [{
        "artifact_id": "hotel-1",
        "payload": {"hotels": [{
            "name": "同安酒店", "area": None, "address": "祥平街道烧灰综合楼",
            "source": "amap", "price_per_night": 300, "rating": 4.5,
        }]},
    }]}

    lodging = toolkit._build_lodging_plan(profile, domain_inputs)

    assert lodging["status"] == "evidence_unavailable"
    assert lodging["area_requirement"] == "思明区"


def test_constraint_tree_budget_overrides_stale_compact_budget() -> None:
    profile = TravelProfile(
        destination="杭州", days=2, budget_limit=5000,
        constraint_state={"budget_max_cny": 4500},
    )
    domain_inputs = {"budgets": [{
        "artifact_id": "budget-1",
        "payload": {
            "days": 2, "companions": 2, "hotel": 1000, "meals": 1000,
            "tickets": 1000, "inner_city_transport": 1000,
            "total_low": 3600, "total_high": 4600,
        },
    }]}

    budget = toolkit._build_budget_plan(profile, domain_inputs, None)

    assert budget["user_limit_cny"] == 4500
    assert budget["user_limit_basis"] == "constraint_total_budget"


def test_budget_plan_multiplies_per_person_limit_and_surfaces_high_risk() -> None:
    profile = TravelProfile(
        destination="西安",
        days=3,
        constraint_state={"traveler_count": 3, "budget_per_person_cny": 2000},
    )
    domain_inputs = {"budgets": [{
        "artifact_id": "budget-1",
        "payload": {
            "days": 3, "companions": 3, "hotel": 2200, "meals": 1620,
            "tickets": 1080, "inner_city_transport": 405,
            "total_low": 4509.25, "total_high": 6100.75,
        },
    }]}

    budget = toolkit._build_budget_plan(profile, domain_inputs, None)

    assert budget["user_limit_cny"] == 6000
    assert budget["user_limit_basis"] == "per_person_times_people"
    assert budget["total_expected_cny"] == 5305
    assert budget["within_user_limit"] is True
    assert budget["risk_high_exceeds_limit"] is True


def test_budget_plan_reads_constraint_total_and_uses_grounded_low_scenario() -> None:
    profile = TravelProfile(
        destination="北京",
        days=3,
        party_size=2,
        constraint_state={"budget_max_cny": 3000, "max_walking_km_per_day": 6},
    )
    domain_inputs = {"budgets": [{
        "artifact_id": "budget-1",
        "payload": {
            "days": 3, "companions": 2, "hotel": 1170, "meals": 1080,
            "tickets": 720, "inner_city_transport": 270,
            "total_low": 2764.5, "total_high": 3726,
        },
    }]}

    budget = toolkit._build_budget_plan(profile, domain_inputs, None)
    mobility = toolkit._build_mobility_plan(
        profile,
        {"days": [{"day_index": 1, "stops": [{
            "name": "故宫",
            "route_from_previous": {
                "origin_name": "酒店", "destination_name": "故宫",
                "walking_distance_km": None,
            },
        }]}]},
    )

    assert budget["user_limit_cny"] == 3000
    assert budget["user_limit_basis"] == "constraint_total_budget"
    assert budget["status"] == "budget_optimized_low_scenario"
    assert budget["total_expected_cny"] == 2764.5
    assert budget["within_user_limit"] is True
    assert mobility["days"][0]["taxi_fallback_required"] is True


def test_collect_domain_inputs_bounds_previous_itinerary_ancestry() -> None:
    previous = {
        "itinerary": {"city": "苏州", "days": [{"day_index": 1, "stops": []}]},
        "critic": {"passed": True, "issues": []},
        "domain_inputs": {
            "previous_itineraries": [{
                "artifact_id": "older-plan",
                "payload": {"domain_inputs": {"previous_itineraries": ["recursive"]}},
            }],
        },
        "original_itinerary": {"large": "draft"},
        "source_artifact_ids": ["older-plan"],
        "budget_plan": {"total_expected_cny": 2000},
    }

    collected = toolkit._collect_domain_inputs([
        {"artifact_id": "plan-1", "kind": "itinerary", "payload": previous},
    ])
    embedded = collected["previous_itineraries"][0]["payload"]

    assert embedded["itinerary"]["city"] == "苏州"
    assert embedded["budget_plan"]["total_expected_cny"] == 2000
    assert "domain_inputs" not in embedded
    assert "original_itinerary" not in embedded
    assert "source_artifact_ids" not in embedded


def test_planner_dedupes_city_wall_aliases() -> None:
    wall = POI("wall-1", "西安城墙", "西安市", "scenic", 34.26, 108.94, 4.8, 0.9, [], 90, "mid")
    ming_wall = POI("wall-2", "西安明城墙", "西安市", "scenic", 34.27, 108.95, 4.7, 0.8, [], 90, "mid")

    result = toolkit._dedupe_plannable_entities(
        [wall, ming_wall], TravelProfile(destination="西安", days=2, must_visit=["西安城墙"]),
    )

    assert [poi.name for poi in result] == ["西安城墙"]


def test_estimate_budget_uses_constraint_tree_traveler_count() -> None:
    ctx = _session()
    ctx.profile = TravelProfile(
        destination="北京",
        days=2,
        companions="孩子",
        constraint_state={"traveler_count": 2},
    )

    result = toolkit.estimate_budget(ctx)

    assert result["budget"]["companions"] == 2
    assert result["budget"]["meals"] == 720


def test_explicit_hotel_area_filters_unrelated_results(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="北京", days=2, hotel_area="王府井附近")
    matching = POI(
        "matching", "王府井大饭店", "北京", "hotel", 39.91, 116.41,
        4.7, 0.9, ["hotel"], 60, "mid", address="王府井大街1号",
    )
    unrelated = POI(
        "unrelated", "亮马桥酒店", "北京", "hotel", 39.95, 116.46,
        4.8, 0.9, ["hotel"], 60, "mid", address="亮马桥路1号",
    )
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [unrelated, matching]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_hotel(ctx)

    assert calls[0]["query_tags"] == ["王府井附近"]
    assert [item["name"] for item in result["hotels"]] == ["王府井大饭店"]
    assert result["hotels"][0]["area"] == "王府井附近"


def test_explicit_hotel_area_does_not_fabricate_mock_when_no_match(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="北京", days=2, hotel_area="王府井附近")
    monkeypatch.setattr(ctx.provider, "search_pois", lambda **_kwargs: [])

    result = toolkit.search_hotel(ctx)

    assert result["count"] == 0
    assert result["hotels"] == []


def test_search_poi_requires_city():
    ctx = _session()
    result = toolkit.search_poi(ctx)
    assert result["isError"] is True


def test_session_tool_lock_is_per_session():
    assert session_tool_lock("sess_a") is session_tool_lock("sess_a")
    assert session_tool_lock("sess_a") is not session_tool_lock("sess_b")


def test_lc_tools_do_not_hold_session_lock_during_provider_io(monkeypatch):
    """Provider I/O may overlap; shared-state writes remain internally locked."""
    from travel_agent.agent.lc_tools import build_tools

    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2)
    tools = {tool.name: tool for tool in build_tools(ctx)}

    active = 0
    max_active = 0
    guard = threading.Lock()
    real_check_weather = toolkit.check_weather

    def slow_check_weather(ctx_arg, city=None):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return real_check_weather(ctx_arg, city)

    monkeypatch.setattr(toolkit, "check_weather", slow_check_weather)

    threads = [
        threading.Thread(target=lambda: tools["check_weather"].invoke({"city": "杭州"}))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 4
