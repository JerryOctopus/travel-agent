from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

from travel_agent.agent import toolkit
from travel_agent.agent.serde import poi_brief, poi_to_dict
from travel_agent.agent.session import build_session, session_tool_lock
from travel_agent.critic import critique_itinerary
from travel_agent.planning_subgraph import PlanAndCritiqueResult
from travel_agent.providers import ProviderRateLimitError
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    RouteInfo,
    ScoredPOI,
    TravelProfile,
)

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


def test_search_poi_does_not_hide_rate_limit_from_supplemental_city_query(monkeypatch):
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2)
    calls = 0

    def search_pois(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [ctx.provider.pois[0]]
        raise ProviderRateLimitError(
            "AMap rate limit: USER_DAILY_QUERY_OVER_LIMIT (10044)"
        )

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_poi(ctx, interests=["nature"], max_results=30)

    assert result["isError"] is True
    assert result["error_code"] == "RATE_LIMITED"


def test_search_poi_does_not_repeat_city_supply_after_full_provider_page(monkeypatch):
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2)
    calls = []
    page = [
        POI(
            poi_id=f"amap-{index}",
            name=f"自然景点{index}",
            city="杭州",
            category="scenic",
            lat=30.2 + index / 10000,
            lng=120.1 + index / 10000,
            rating=4.5,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=90,
            price_level="unknown",
            source="amap",
            entity_type="attraction",
            verification_status="verified",
        )
        for index in range(25)
    ]

    def search_pois(**kwargs):
        calls.append(kwargs)
        return page

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_poi(ctx, interests=["nature"], max_results=30)

    assert result["isError"] is False
    assert len(calls) == 1


def test_multi_day_sparse_search_adds_verified_category_supply(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="合成城", days=4)
    base = dict(
        city="合成城", lat=30.0, lng=120.0, rating=4.5,
        popularity=0.8, tags=["history"], estimated_duration_min=90,
        price_level="unknown", source="amap", entity_type="attraction",
        verification_status="verified",
    )
    initial = [POI("base-1", "综合景点一", category="scenic", **base)]
    scenic = [
        POI(f"scenic-{index}", f"自然景点{index}", category="scenic", **base)
        for index in range(3)
    ]
    museums = [
        POI(f"museum-{index}", f"历史博物馆{index}", category="museum", **base)
        for index in range(3)
    ]
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        if kwargs.get("category") == "scenic":
            return scenic
        if kwargs.get("category") == "museum":
            return museums
        return initial

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_poi(ctx, interests=["history"], max_results=30)

    assert result["isError"] is False
    assert result["count"] == 7
    assert {call.get("category") for call in calls} >= {"scenic", "museum"}


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


def test_plan_does_not_add_bound_restaurants_without_explicit_food_request():
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
        all(stop["poi"]["category"] != "food" for stop in day["stops"])
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


def test_planner_keeps_candidate_food_that_matches_explicit_specific_interest():
    cafe = POI(
        "cafe-1", "海湾咖啡馆", "厦门", "food", 24.45, 118.10,
        4.8, 0.9, ["coffee", "cafe"], 60, "mid", source="amap",
    )
    ranked = {"pois": [toolkit._scored_to_dict(ScoredPOI(cafe, 1.0, []))]}
    records = [{
        "kind": "candidates",
        "artifact_id": "candidates-1",
        "payload": {"pois": [toolkit.poi_to_dict(cafe)]},
    }]
    profile = TravelProfile(
        destination="厦门",
        days=1,
        interests=["food"],
        constraint_state={"interests": ["咖啡店"]},
    )

    merged = toolkit._merge_ranked_planner_inputs(ranked, records, profile)

    assert [item.poi.name for item in merged] == ["海湾咖啡馆"]


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


def test_restaurant_search_binds_local_area_into_provider_query(monkeypatch):
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=1)
    restaurant = POI(
        "meal", "青禾餐厅", "杭州", "food", 30.2, 120.1,
        4.5, 0.8, ["food"], 60, "mid",
    )
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [restaurant]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_restaurant(ctx, area="杭州滨江区星光大道")

    assert calls[0]["query_tags"] == ["杭州滨江区星光大道"]
    assert calls[-1]["query_tags"] == ["杭州滨江区星光大道", "餐厅"]
    assert calls[-1]["category"] == "food"
    assert result["restaurants"][0]["name"] == "青禾餐厅"


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

    assert len(calls) == 4
    assert [item["name"] for item in result["restaurants"]] == ["清真餐厅一", "清真餐厅二"]


def test_area_restaurant_search_keeps_provider_nearby_results_without_address_name_match(
    monkeypatch,
) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="合成城", days=2)
    ctx.profile.constraint_state = {"dietary": ["仅清真餐厅"]}
    anchor = POI(
        "anchor", "远端景区", "合成城", "scenic", 34.38, 109.27,
        4.8, 0.9, [], 120, "unknown", source="amap", entity_type="attraction",
    )
    nearby = POI(
        "nearby", "清真风味餐厅", "合成城", "food", 34.381, 109.271,
        4.6, 0.8, ["halal"], 60, "unknown", address="东环路",
        source="amap", entity_type="restaurant",
    )
    distant = POI(
        "distant", "清真市区餐厅", "合成城", "food", 34.20, 109.00,
        4.7, 0.9, ["halal"], 60, "unknown", address="中心街",
        source="amap", entity_type="restaurant",
    )

    def search_pois(**kwargs):
        if kwargs.get("query_tags") == ["远端景区"]:
            return [anchor]
        return [distant]

    nearby_calls = []

    def search_pois_nearby(**kwargs):
        nearby_calls.append(kwargs)
        return [nearby]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)
    monkeypatch.setattr(ctx.provider, "search_pois_nearby", search_pois_nearby, raising=False)

    result = toolkit.search_restaurant(ctx, cuisine="清真", area="远端景区")

    assert nearby_calls[0]["anchor"].poi_id == "anchor"
    assert nearby_calls[0]["category"] == "food"
    assert [item["name"] for item in result["restaurants"]][:1] == ["清真风味餐厅"]


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

    assert len(calls) == 3
    assert calls[1]["query_tags"] == ["钟楼 酒店"]
    assert calls[1]["category"] is None
    assert result["hotels"][0]["name"] == "西安大酒店"
    assert result["hotels"][0]["area"] is None


def test_explicit_user_hotel_area_accepts_provider_query_relevance(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(
        ctx, destination="杭州", days=3, hotel_area="湖区东侧"
    )
    grounded = POI(
        "hotel-area-1", "雅致酒店", "杭州", "hotel", 30.26, 120.17,
        4.6, 0.8, ["hotel"], 0, "mid", source="amap",
        entity_type="hotel",
    )
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        return [grounded] if kwargs.get("query_tags") == ["湖区东侧"] else []

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_hotel(ctx)

    assert len(calls) == 2
    assert calls[1]["query_tags"] == ["湖区"]
    assert result["count"] == 1
    assert result["hotels"][0]["area"] == "湖区东侧"


def test_directional_hotel_area_is_validated_against_landmark_coordinates(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(
        ctx, destination="测试城", days=3, hotel_area="中心湖东侧"
    )
    west = POI(
        "hotel-west", "便宜酒店", "测试城", "hotel", 30.0, 119.98,
        4.8, 0.9, ["hotel"], 0, "low", source="amap", entity_type="hotel",
    )
    east = POI(
        "hotel-east", "东岸酒店", "测试城", "hotel", 30.0, 120.03,
        4.5, 0.8, ["hotel"], 0, "low", source="amap", entity_type="hotel",
    )
    landmark = POI(
        "lake", "中心湖风景区", "测试城", "scenic", 30.0, 120.0,
        4.9, 1.0, ["classic"], 120, "unknown", source="amap",
        entity_type="attraction",
    )

    def search_pois(**kwargs):
        if kwargs.get("query_tags") == ["中心湖东侧"]:
            return [west, east]
        if kwargs.get("query_tags") == ["中心湖"]:
            return [landmark]
        return []

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_hotel(ctx, budget_level="low")

    assert [item["name"] for item in result["hotels"]] == ["东岸酒店"]
    assert result["hotels"][0]["area"] == "中心湖东侧"


def test_directional_hotel_area_broadens_then_rechecks_coordinates(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(
        ctx, destination="测试城", days=3, hotel_area="中心湖东侧"
    )
    west = POI(
        "hotel-west", "西岸酒店", "测试城", "hotel", 30.0, 119.98,
        4.8, 0.9, ["hotel"], 0, "low", source="amap", entity_type="hotel",
    )
    east = POI(
        "hotel-east", "东岸酒店", "测试城", "hotel", 30.0, 120.03,
        4.5, 0.8, ["hotel"], 0, "low", source="amap", entity_type="hotel",
    )
    landmark = POI(
        "lake", "中心湖风景区", "测试城", "scenic", 30.0, 120.0,
        4.9, 1.0, ["classic"], 120, "unknown", source="amap",
        entity_type="attraction",
    )

    def search_pois(**kwargs):
        if kwargs.get("query_tags") == ["中心湖东侧"]:
            return [west]
        if kwargs.get("query_tags") == ["中心湖"]:
            return [landmark]
        if kwargs.get("query_tags") is None and kwargs.get("category") == "hotel":
            return [west, east]
        return []

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_hotel(ctx, budget_level="low")

    assert [item["name"] for item in result["hotels"]] == ["东岸酒店"]
    assert result["hotels"][0]["area"] == "中心湖东侧"


def test_mobility_ranking_expands_locally_to_fill_multi_day_plan() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=3,
        constraint_state={"elderly": True, "max_walking_km_per_day": 6.0},
    )
    ranked = [
        ScoredPOI(
            POI(
                f"p{index}", f"本地景点{index}", "测试城", "scenic",
                30.0, 120.0 + offset, 4.5, 0.8, [], 90, "mid",
            ),
            1.0 - index / 100,
            [],
        )
        for index, offset in enumerate((0.01, 0.02, 0.03, 0.04, 0.12, 0.14), start=1)
    ]
    ranked.append(ScoredPOI(
        POI("remote", "远郊景点", "测试城", "scenic", 30.0, 120.5,
            4.9, 1.0, [], 90, "mid"),
        1.0,
        [],
    ))

    prioritized = toolkit._prioritize_ranked_near_lodging(
        ranked,
        {"hotel": {"lat": 30.0, "lng": 120.0}},
        profile,
    )

    assert len([item for item in prioritized if item.poi.category != "food"]) == 6
    assert "remote" not in {item.poi.poi_id for item in prioritized}


def test_fixed_event_plan_uses_hotel_departure_leg_when_event_is_first_stop() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=3,
        start_date="2026-10-03",
        hotel_area="中心区",
        constraint_state={
            "fixed_events": [{
                "date": "2026-10-04",
                "start": "15:00",
                "end": "17:00",
                "location": "遗址博物馆",
            }],
        },
    )
    required = {
        "legs": [{
            "kind": "fixed_event_transfer",
            "required_name": "遗址博物馆",
            "event_poi_id": "event",
            "origin_poi_id": None,
            "routes": [{
                "origin_poi_id": "subvenue",
                "destination_poi_id": "event",
                "duration_min": 5,
                "source": "amap",
                "evidence_status": "provider_verified",
            }],
        }],
    }
    lodging_routes = {
        "daily_routes": [{
            "day_index": 2,
            "legs": [{
                "position": "hotel_to_first_stop",
                "origin_poi_id": "hotel",
                "destination_poi_id": "event",
                "distance_km": 40.0,
                "duration_min": 147,
                "source": "haversine_recovery_estimate",
                "evidence_status": "haversine_estimate",
            }],
        }],
    }

    plan = toolkit._build_fixed_event_plan(
        profile, {}, required, lodging_routes
    )
    event = plan["events"][0]

    assert event["recommended_departure"] == "12:18"
    assert event["route_evidence_reference"]["origin_poi_id"] == "hotel"
    assert event["route_evidence_status"] == "haversine_estimate"


def test_required_fixed_event_anchor_rebinds_to_verified_hotel_leg() -> None:
    anchors = {
        "legs": [{
            "kind": "fixed_event_transfer",
            "required_name": "遗址博物馆",
            "origin_poi_id": None,
            "event_poi_id": "event",
            "evidence_status": "provider_verified",
            "routes": [{
                "origin_poi_id": "internal-hall",
                "destination_poi_id": "event",
                "distance_km": 0.5,
                "duration_min": 5,
                "source": "amap",
                "evidence_status": "provider_verified",
            }],
        }],
    }
    lodging = {
        "daily_routes": [{
            "day_index": 2,
            "legs": [{
                "position": "hotel_to_first_stop",
                "origin_poi_id": "hotel",
                "destination_poi_id": "event",
                "distance_km": 48.0,
                "duration_min": 98,
                "source": "amap",
                "evidence_status": "provider_verified",
            }],
        }],
    }

    rebound = toolkit._bind_lodging_fixed_event_routes(anchors, lodging)
    leg = rebound["fixed_event_transfer"]

    assert leg["origin_poi_id"] == "hotel"
    assert leg["routes"][0]["origin_poi_id"] == "hotel"
    assert leg["evidence_status"] == "provider_verified"


def test_fixed_event_closure_reuses_unlabelled_exact_endpoint_route() -> None:
    ctx = _session()
    ctx.profile.destination = "测试城"
    ctx.profile.days = 1
    ctx.profile.constraint_state = {
        "fixed_events": [{
            "day": 1,
            "start": "14:30",
            "end": "16:30",
            "location": "预约博物馆",
        }],
    }
    origin = POI(
        "origin", "上午景点", "测试城", "scenic", 30.0, 120.0,
        4.5, 0.8, ["classic"], 90, "mid", source="amap",
    )
    event = POI(
        "event", "预约博物馆", "测试城", "museum", 30.1, 120.1,
        4.8, 0.9, ["history"], 120, "mid", source="amap",
    )
    ctx.remember_pois([origin, event])
    itinerary = {"days": [{"day_index": 1, "stops": [
        {"start_time": "09:00", "duration_min": 90, "poi": poi_to_dict(origin)},
        {"start_time": "14:30", "duration_min": 120, "poi": poi_to_dict(event)},
    ]}]}
    domain_inputs = {"transport": [{
        "artifact_id": "exact-route",
        "payload": {
            "origin_poi_id": "origin",
            "destination_poi_id": "event",
            "origin_name": "上午景点",
            "destination_name": "预约博物馆",
            "mode": "public_transport",
            "duration_min": 20,
            "distance_km": 4.0,
            "source": "amap",
            "evidence_status": "provider_verified",
        },
    }]}

    class UnexpectedEstimator:
        def estimate_route(self, *_args, **_kwargs):
            raise AssertionError("exact endpoint evidence should be reused")

    closed = toolkit._close_post_plan_fixed_event_routes(
        ctx, itinerary, domain_inputs, UnexpectedEstimator()
    )

    assert len(closed) == 1
    assert closed[0]["fixed_event_name"] == "预约博物馆"
    assert closed[0]["required_buffer_min"] == 15
    assert closed[0]["recommended_latest_departure"] == "13:55"
    assert any(
        entry.get("post_plan_fixed_event_closure") is True
        and entry["payload"]["fixed_event_name"] == "预约博物馆"
        for entry in domain_inputs["transport"]
    )


def test_amap_hotel_without_price_evidence_is_not_rejected_as_wrong_budget_tier(
    monkeypatch,
) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=3)
    hotel = POI(
        "hotel-1", "湖畔酒店", "杭州", "hotel", 30.25, 120.15,
        4.5, 0.8, ["hotel"], 0, "unknown", source="amap",
        entity_type="hotel",
    )
    monkeypatch.setattr(ctx.provider, "search_pois", lambda **_kwargs: [hotel])

    result = toolkit.search_hotel(ctx, budget_level="low")

    assert result["hotels"][0]["name"] == "湖畔酒店"
    assert result["hotels"][0]["budget_level"] == "low"


def test_hotel_search_without_requested_area_starts_with_central_area_evidence(
    monkeypatch,
) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="合成城", days=3)
    central = POI(
        "central-hotel", "中心酒店", "合成城", "hotel", 30.25, 120.15,
        4.6, 0.8, ["hotel"], 0, "unknown", source="amap",
        entity_type="hotel",
    )
    remote = POI(
        "remote-hotel", "远郊酒店", "合成城", "hotel", 30.8, 120.8,
        4.8, 0.9, ["hotel"], 0, "unknown", source="amap",
        entity_type="hotel",
    )
    calls = []

    def search_pois(**kwargs):
        calls.append(kwargs)
        if kwargs.get("query_tags") == ["市中心", "核心商圈"]:
            return [central]
        return [remote]

    monkeypatch.setattr(ctx.provider, "search_pois", search_pois)

    result = toolkit.search_hotel(ctx)

    assert len(calls) == 1
    assert calls[0]["query_tags"] == ["市中心", "核心商圈"]
    assert [item["name"] for item in result["hotels"]] == ["中心酒店"]
    assert result["hotels"][0]["area"] == "核心商圈"


def test_hotel_search_applies_explicit_nightly_budget_to_estimated_candidates(
    monkeypatch,
) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="合成城", days=3)
    ctx.profile.constraint_state = {"hotel_budget_per_night_cny": 600}
    candidates = [
        POI(
            f"hotel-{index}", f"酒店{index}", "合成城", "hotel",
            30.25, 120.15, 4.6, 0.8, ["hotel"], 0, "unknown",
            source="amap", entity_type="hotel",
        )
        for index in range(1, 4)
    ]
    monkeypatch.setattr(ctx.provider, "search_pois", lambda **_kwargs: candidates)

    result = toolkit.search_hotel(ctx)

    assert [item["price_per_night"] for item in result["hotels"]] == [585]
    assert result["hotel_budget_per_night_cny"] == 600.0


def test_entity_dedupe_prefers_main_museum_over_directional_branch() -> None:
    profile = TravelProfile(destination="测试城", days=2, must_visit=["甲乙博物馆"])
    main = POI(
        "main", "甲乙博物馆(本馆)", "测试城", "museum", 30.0, 120.0,
        4.5, 0.8, ["history"], 90, "unknown", source="amap",
        canonical_name="甲乙博物馆(本馆)", entity_type="museum",
    )
    west = replace(main, poi_id="west", name="甲乙博物馆西馆", canonical_name="甲乙博物馆西馆")

    deduped = toolkit._dedupe_plannable_entities([west, main], profile)

    assert [poi.poi_id for poi in deduped] == ["main"]


def test_entity_dedupe_handles_city_prefixed_main_and_directional_branch() -> None:
    profile = TravelProfile(destination="甲城", days=2, must_visit=["甲城博物馆"])
    main = POI(
        "main", "甲城博物馆(本馆)", "甲城市", "museum", 30.0, 120.0,
        4.8, 0.9, ["history"], 120, "unknown", source="amap",
        canonical_name="甲城博物馆(本馆)", entity_type="museum",
    )
    west = replace(
        main,
        poi_id="west",
        name="甲城博物馆西馆",
        canonical_name="甲城博物馆西馆",
    )

    deduped = toolkit._dedupe_plannable_entities([west, main], profile)

    assert [poi.poi_id for poi in deduped] == ["main"]


def test_entity_dedupe_keeps_distinct_museum_inside_scenic_parent() -> None:
    scenic = POI(
        "lake", "测试湖风景名胜区", "测试城", "scenic", 30.0, 120.0,
        4.9, 1.0, [], 150, "unknown", source="amap",
        source_poi_id="lake", entity_type="attraction",
    )
    museum = POI(
        "museum", "测试省博物馆(湖畔馆区)", "测试城", "museum", 30.01, 120.01,
        4.8, 0.9, [], 120, "unknown", source="amap",
        source_poi_id="museum", parent_poi_id="lake", entity_type="museum",
    )
    profile = TravelProfile(
        destination="测试城", days=2,
        must_visit=["测试湖", "测试省博物馆"],
    )

    deduped = toolkit._dedupe_plannable_entities([scenic, museum], profile)

    assert {poi.poi_id for poi in deduped} == {"lake", "museum"}


def test_entity_dedupe_keeps_scenic_area_distinct_from_same_named_museum() -> None:
    profile = TravelProfile(destination="杭州", days=2, must_visit=["西湖"])
    scenic = POI(
        "lake", "杭州西湖风景名胜区", "杭州市", "scenic", 30.2, 120.1,
        4.8, 0.9, ["nature"], 120, "free", source="amap",
        canonical_name="杭州西湖风景名胜区", entity_type="attraction",
    )
    museum = POI(
        "lake-museum", "西湖博物馆", "杭州市", "museum", 30.2, 120.1,
        4.7, 0.8, ["history"], 90, "free", source="amap",
        canonical_name="西湖博物馆", entity_type="museum",
    )

    deduped = toolkit._dedupe_plannable_entities([museum, scenic], profile)

    assert {poi.poi_id for poi in deduped} == {"lake", "lake-museum"}


def test_entity_dedupe_prioritizes_required_museum_over_named_annex() -> None:
    profile = TravelProfile(
        destination="杭州", days=2, must_visit=["浙江省博物馆"]
    )
    museum = POI(
        "museum", "浙江省博物馆(孤山馆区)", "杭州市", "museum", 30.2, 120.1,
        4.7, 0.8, ["history"], 120, "free", source="amap",
        canonical_name="浙江省博物馆(孤山馆区)", entity_type="museum",
    )
    annex = replace(
        museum,
        poi_id="annex",
        name="浙江省博物馆-精品馆",
        canonical_name="浙江省博物馆-精品馆",
    )

    deduped = toolkit._dedupe_plannable_entities([annex, museum], profile)

    assert [poi.poi_id for poi in deduped] == ["museum"]


def test_entity_dedupe_collapses_explicit_provider_parent_and_child() -> None:
    profile = TravelProfile(destination="测试城", days=1, must_visit=["古王朝兵俑馆"])
    parent = POI(
        "parent", "古王朝帝陵博物院", "测试城", "museum", 30.0, 120.0,
        4.7, 0.9, ["history"], 120, "unknown", source="amap",
        source_poi_id="provider-parent",
    )
    child = POI(
        "child", "古王朝兵俑馆", "测试城", "museum", 30.0001, 120.0001,
        4.8, 0.9, ["history"], 90, "unknown", source="amap",
        parent_poi_id="provider-parent",
        source_poi_id="provider-child",
    )

    deduped = toolkit._dedupe_plannable_entities([parent, child], profile)

    assert [poi.poi_id for poi in deduped] == ["child"]


def test_entity_dedupe_collapses_provider_siblings_in_one_complex() -> None:
    profile = TravelProfile(destination="测试城", days=2, must_visit=["古塔"])
    requested = POI(
        "requested", "古塔", "测试城", "scenic", 30.0, 120.0,
        4.8, 0.9, ["history"], 90, "unknown", source="amap",
        parent_poi_id="provider-complex", source_poi_id="provider-requested",
    )
    pavilion = replace(
        requested,
        poi_id="pavilion",
        name="古塔景区-夕照亭",
        source_poi_id="provider-pavilion",
    )

    deduped = toolkit._dedupe_plannable_entities([pavilion, requested], profile)

    assert [poi.poi_id for poi in deduped] == ["requested"]


def test_hotel_search_applies_authorized_downgrade_under_total_budget(monkeypatch) -> None:
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="厦门", days=4, budget_level="mid")
    ctx.profile.constraint_state = {
        "budget_max_cny": 4800,
        "lodging_flexibility": "can_downgrade",
    }
    hotel = POI(
        "hotel-1", "海湾酒店", "厦门", "hotel", 24.4, 118.1,
        4.5, 0.8, ["hotel"], 0, "unknown", source="amap",
        entity_type="hotel",
    )
    monkeypatch.setattr(ctx.provider, "search_pois", lambda **_kwargs: [hotel])

    result = toolkit.search_hotel(ctx, budget_level="mid")

    assert result["hotels"][0]["budget_level"] == "low"
    assert result["hotels"][0]["price_per_night"] < 400


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
    assert result["intercity_segment"]["status"] == "verified_route"
    assert result["intercity_segment"]["route_evidence"] == return_route
    assert "车次" not in result["intercity_segment"]


def test_same_city_return_plan_uses_actual_final_stop_route() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"return_deadline": "19:00", "return_location": "测试城南站"},
    )
    itinerary = {"days": [{"stops": [
        {"poi": {"poi_id": "first"}},
        {"poi": {"poi_id": "final"}},
    ]}]}
    stale_route = {
        "origin_poi_id": "first", "destination_poi_id": "station",
        "origin_name": "第一站", "destination_name": "测试城南站",
        "duration_min": 25, "source": "amap", "evidence_status": "provider_verified",
    }
    final_route = {
        "origin_poi_id": "final", "destination_poi_id": "station",
        "origin_name": "最终站", "destination_name": "测试城南站",
        "duration_min": 35, "source": "amap", "evidence_status": "provider_verified",
    }

    result = toolkit._build_return_plan(
        profile,
        itinerary,
        {"transport": [{"payload": stale_route}, {"payload": final_route}]},
    )

    assert result is not None
    assert result["terminal_transfer"] == final_route


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


def test_budget_plan_includes_grounded_intercity_distance_allowance() -> None:
    profile = TravelProfile(
        destination="目的城",
        days=1,
        party_size=2,
        budget_limit=1200,
    )
    domain_inputs = {
        "budgets": [{
            "artifact_id": "budget-1",
            "payload": {
                "days": 1,
                "companions": 2,
                "hotel": 0,
                "meals": 180,
                "tickets": 100,
                "inner_city_transport": 50,
                "total_low": 280.5,
                "total_high": 379.5,
            },
        }],
        "transport": [
            {
                "artifact_id": "outbound",
                "post_plan_origin_closure": True,
                "payload": {
                    "origin_poi_id": "origin",
                    "destination_poi_id": "first",
                    "distance_km": 180,
                    "duration_min": 120,
                    "mode": "public_transport",
                    "source": "amap",
                    "evidence_status": "provider_verified",
                },
            },
            {
                "artifact_id": "return",
                "post_plan_endpoint_closure": True,
                "payload": {
                    "origin_poi_id": "last",
                    "destination_poi_id": "return",
                    "distance_km": 180,
                    "duration_min": 120,
                    "mode": "public_transport",
                    "source": "amap",
                    "evidence_status": "provider_verified",
                },
            },
        ],
    }

    budget = toolkit._build_budget_plan(profile, domain_inputs, None)

    assert budget["intercity_transport_estimate_cny"] == 360
    assert budget["transport"] == 410
    assert budget["total_expected_cny"] == 690
    assert budget["within_user_limit"] is True
    assert budget["intercity_estimate_method"] == "verified_distance_allowance_not_live_fare"


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


def test_prepaid_lodging_without_new_search_request_is_user_owned() -> None:
    profile = TravelProfile(
        destination="苏州",
        days=2,
        constraint_state={"prepaid_lodging_cny": 600},
    )

    lodging = toolkit._build_lodging_plan(profile, {})

    assert lodging == {
        "required": True,
        "explicit_requirement": False,
        "status": "user_owned_prepaid",
        "nights": 1,
        "prepaid_lodging_cny": 600.0,
        "evidence_status": "user_provided",
        "source_artifact_ids": [],
    }


def test_prepaid_lodging_keeps_explicit_area_search_fail_closed() -> None:
    profile = TravelProfile(
        destination="苏州",
        days=2,
        hotel_area="平江路附近",
        constraint_state={
            "prepaid_lodging_cny": 600,
            "lodging_area": "平江路附近",
        },
    )

    lodging = toolkit._build_lodging_plan(profile, {})

    assert lodging["status"] == "evidence_unavailable"
    assert lodging["explicit_requirement"] is True
    assert lodging["area_requirement"] == "平江路附近"


def test_mobility_sensitive_ranking_prioritizes_local_candidates_and_must_visit() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        must_visit=["核心湖"],
        constraint_state={"elderly": True, "max_walking_km_per_day": 6.0},
    )
    near = POI("near", "近处公园", "测试城", "scenic", 30.01, 120.01, 4.5, 0.8, [], 90, "mid")
    far = POI("far", "远郊寺庙", "测试城", "scenic", 30.30, 120.30, 4.9, 1.0, [], 90, "mid")
    required = POI("required", "核心湖风景区", "测试城", "scenic", 30.20, 120.20, 4.8, 0.9, [], 90, "mid")
    ranked = [
        ScoredPOI(far, 1.0, []),
        ScoredPOI(near, 0.9, []),
        ScoredPOI(required, 0.8, []),
    ]

    prioritized = toolkit._prioritize_ranked_near_lodging(
        ranked,
        {"hotel": {"lat": 30.0, "lng": 120.0}},
        profile,
    )

    assert [item.poi.poi_id for item in prioritized] == ["near", "required"]


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


def test_mobility_sensitive_budget_reserves_taxi_fallback_allowance() -> None:
    profile = TravelProfile(
        destination="测试城", days=3, party_size=2,
        constraint_state={"elderly": True, "max_walking_km_per_day": 4},
    )
    domain_inputs = {"budgets": [{
        "artifact_id": "budget-1",
        "payload": {
            "days": 3, "companions": 2, "hotel": 900, "meals": 600,
            "tickets": 300, "inner_city_transport": 200,
            "total_low": 1800, "total_high": 2400,
        },
    }]}

    budget = toolkit._build_budget_plan(profile, domain_inputs, None)

    assert budget["mobility_fallback_transport_cny"] == 240
    assert budget["transport"] == 440
    assert budget["total_expected_cny"] == 2240
    assert "行动不便" in budget["note"]


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
    assert budget["status"] == "estimate"
    assert budget["within_user_limit"] is False
    assert budget["mobility_fallback_transport_cny"] == 240
    assert budget["total_expected_cny"] == 3480
    assert mobility["days"][0]["taxi_fallback_required"] is True


def test_mobility_plan_counts_verified_taxi_as_zero_route_walking() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        constraint_state={"max_walking_km_per_day": 2},
    )

    mobility = toolkit._build_mobility_plan(
        profile,
        {"days": [{"day_index": 1, "stops": [{
            "name": "故宫",
            "route_from_previous": {
                "origin_name": "酒店",
                "destination_name": "故宫",
                "origin_poi_id": "hotel-1",
                "destination_poi_id": "palace-1",
                "distance_km": 4.2,
                "duration_min": 18,
                "mode": "taxi",
                "source": "amap",
                "evidence_status": "provider_verified",
                "walking_distance_km": None,
            },
        }]}]},
    )

    assert mobility is not None
    assert mobility["days"][0]["known_walking_km"] == 0
    assert mobility["days"][0]["unknown_walking_legs"] == []
    assert mobility["days"][0]["taxi_fallback_required"] is False


def test_mobility_plan_addresses_hill_and_stair_avoidance_without_numeric_cap() -> None:
    profile = TravelProfile(
        destination="重庆",
        days=1,
        constraint_state={
            "elderly": True,
            "avoid": ["连续爬坡", "长楼梯"],
        },
    )
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": "18:00",
        "duration_min": 90,
        "poi": {"poi_id": "view-1", "name": "滨江夜景观景台"},
    }]}]}

    mobility = toolkit._build_mobility_plan(profile, itinerary)

    assert mobility is not None
    assert mobility["avoidance_requirements"] == ["连续爬坡", "长楼梯"]
    assert mobility["venue_internal_access"][0]["poi_id"] == "view-1"
    assert mobility["venue_internal_access"][0]["fallback"] == "replace_candidate"
    assert "不承诺" in mobility["policy"]


def test_walking_cap_policy_does_not_invent_internal_accessibility_requirement() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"elderly": True, "max_walking_km_per_day": 6.0},
    )

    mobility = toolkit._build_mobility_plan(
        profile,
        {"days": [{"day_index": 1, "stops": []}]},
    )

    assert mobility is not None
    assert mobility["venue_internal_access"] == []
    assert "无台阶入口" not in mobility["policy"]


def test_meal_reservation_avoids_overlapping_scheduled_activity() -> None:
    profile = TravelProfile(destination="测试城", days=1)
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": "11:45",
        "duration_min": 90,
        "poi": {"category": "museum", "name": "城市博物馆"},
    }]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    meal = strategy["scheduled_meals"][0]
    assert meal["start_time"] == "13:30"
    assert meal["end_time"] == "14:30"


def test_meal_strategy_does_not_publish_a_false_window_when_day_has_no_gap() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"activity_end_deadline": "18:00"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {"start_time": "09:00", "duration_min": 120, "poi": {"category": "museum", "name": "上午活动"}},
        {"start_time": "12:00", "duration_min": 120, "poi": {"category": "scenic", "name": "中午活动"}},
        {"start_time": "15:00", "duration_min": 150, "poi": {"category": "scenic", "name": "下午活动"}},
    ]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    assert strategy["scheduled_meals"] == []
    assert strategy["unscheduled_meal_days"] == [{
        "day_index": 1,
        "reason": "no_non_overlapping_meal_window",
    }]


def test_meal_strategy_uses_early_lunch_gap_before_inbound_transfer() -> None:
    profile = TravelProfile(
        destination="合成城",
        days=1,
        constraint_state={"dietary": ["清淡"]},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "09:00", "duration_min": 120,
            "poi": {"category": "museum", "name": "上午活动"},
        },
        {
            "start_time": "13:00", "duration_min": 90,
            "route_from_previous": {"duration_min": 17},
            "poi": {"category": "scenic", "name": "下午活动"},
        },
    ]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    lunch = next(
        meal for meal in strategy["scheduled_meals"]
        if meal.get("period") == "lunch"
    )
    assert (lunch["start_time"], lunch["end_time"]) == ("11:15", "12:15")
    assert strategy["unscheduled_meal_days"] == []


def test_reviewer_sparse_day_directive_materializes_in_rebuilt_itinerary() -> None:
    first = POI(
        "first", "第一天景点", "合成城", "scenic",
        30.0, 120.0, 4.5, 0.8, [], 90, "mid",
    )
    second = POI(
        "second", "第二天景点", "合成城", "museum",
        30.01, 120.01, 4.5, 0.8, [], 90, "mid",
    )
    extra_one = POI(
        "extra-one", "第一天下午景点", "合成城", "museum",
        30.0, 120.01, 4.5, 0.8, [], 90, "mid",
    )
    extra_two = POI(
        "extra-two", "第二天下午景点", "合成城", "scenic",
        30.01, 120.02, 4.5, 0.8, [], 90, "mid",
    )
    itinerary = Itinerary(
        city="合成城",
        summary="test",
        days=[
            ItineraryDay(1, "test", [ItineraryStop(first, "09:30", 90, "")]),
            ItineraryDay(2, "test", [ItineraryStop(second, "09:30", 90, "")]),
        ],
    )
    profile = TravelProfile(destination="合成城", days=2, pace="relaxed")
    result = PlanAndCritiqueResult(
        itinerary=itinerary,
        original_itinerary=itinerary,
        critic_result=critique_itinerary(itinerary, profile),
    )
    ctx = _session()
    ctx.profile = profile

    revised = toolkit._apply_revision_directives(
        result,
        [
            ScoredPOI(extra_one, 0.8, []),
            ScoredPOI(extra_two, 0.8, []),
        ],
        ctx,
        {"reviewer_issue_types": ["daily_activity_sparse"]},
    )

    assert [
        sum(stop.poi.category != "food" for stop in day.stops)
        for day in revised.itinerary.days
    ] == [2, 2]
    assert "daily_activity_sparse" not in {
        issue.code for issue in revised.critic_result.issues
    }


def test_meal_reservation_leaves_transfer_buffer_before_and_after_activities() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"activity_end_deadline": "21:00"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "15:00",
            "duration_min": 90,
            "poi": {"category": "museum", "name": "下午活动"},
        },
        {
            "start_time": "17:30",
            "duration_min": 90,
            "poi": {"category": "shopping", "name": "傍晚活动"},
        },
    ]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    dinner = next(
        meal for meal in strategy["scheduled_meals"]
        if meal["name"].startswith("晚餐")
    )
    assert dinner["start_time"] == "19:15"
    assert dinner["end_time"] == "20:15"


def test_last_day_meal_reservation_stays_before_return_activity_cutoff() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"return_deadline": "17:00", "return_location": "杭州东站"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "09:00",
            "duration_min": 120,
            "poi": {"category": "museum", "name": "省博物馆"},
        },
        {
            "start_time": "12:00",
            "duration_min": 150,
            "poi": {"category": "scenic", "name": "湖滨公园"},
        },
    ]}]}
    return_plan = {"activity_cutoff": "16:00"}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {}, return_plan)

    meal = strategy["scheduled_meals"][0]
    assert meal["start_time"] == "14:45"
    assert meal["end_time"] == "15:45"


def test_post_plan_return_closure_uses_live_provider_when_bound_routes_are_only_geometry() -> None:
    final = POI(
        "museum", "湖畔博物馆", "测试城市", "museum", 30.1, 120.1,
        4.6, 0.8, ["history"], 90, "mid", source="amap",
        entity_type="museum", verification_status="verified",
    )
    station = POI(
        "station", "测试城东站", "测试城市", "transport", 30.2, 120.2,
        4.5, 0.8, ["railway"], 30, "mid", source="amap",
        entity_type="transport", verification_status="verified",
    )

    class Provider:
        def search_pois(self, **_kwargs):
            return [station]

        def estimate_route(self, origin, destination, mode="public_transport"):
            assert origin.poi_id == final.poi_id
            assert destination.poi_id == station.poi_id
            return RouteInfo(
                origin.poi_id, destination.poi_id, 12.0, 35, mode,
                source="amap", evidence_status="provider_verified",
            )

    class BoundGeometryOnly:
        def estimate_route(self, origin, destination, mode="public_transport"):
            return RouteInfo(
                origin.poi_id, destination.poi_id, 12.0, 90, mode,
                source="haversine_recovery_estimate", evidence_status="deterministic_estimate",
            )

    ctx = _session()
    ctx.provider = Provider()
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"return_location": "测试城东站", "return_deadline": "17:00"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": "13:00", "duration_min": 90, "poi": poi_to_dict(final),
    }]}]}
    domain_inputs = {}

    route = toolkit._close_post_plan_return_route(
        ctx, itinerary, domain_inputs, BoundGeometryOnly()
    )

    assert route is not None
    assert route["source"] == "amap"
    assert route["evidence_status"] == "provider_verified"
    assert route["hard_feasibility_proven"] is True


def test_late_activity_deadline_reserves_both_lunch_and_dinner() -> None:
    profile = TravelProfile(
        destination="广州",
        days=1,
        constraint_state={"activity_end_deadline": "20:30"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": "09:00",
        "duration_min": 120,
        "poi": {"category": "museum", "name": "城市博物馆"},
    }]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    assert [meal["name"].split("时段", 1)[0] for meal in strategy["scheduled_meals"]] == [
        "午餐", "晚餐",
    ]
    assert strategy["scheduled_meals"][1]["end_time"] <= "20:30"


def test_multi_day_plan_reserves_dinner_without_artificially_extending_stops() -> None:
    profile = TravelProfile(destination="合成城", days=2)
    itinerary = {"days": [
        {"day_index": day_index, "stops": [{
            "start_time": "09:00", "duration_min": 120,
            "poi": {"category": "museum", "name": f"博物馆{day_index}"},
        }]}
        for day_index in (1, 2)
    ]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    assert [
        (meal["day_index"], meal["name"].split("时段", 1)[0])
        for meal in strategy["scheduled_meals"]
    ] == [(1, "午餐"), (1, "晚餐"), (2, "午餐"), (2, "晚餐")]


def test_concrete_dinner_still_reserves_dietary_lunch_each_day() -> None:
    profile = TravelProfile(
        destination="合成城",
        days=1,
        interests=["food"],
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "09:00",
            "duration_min": 120,
            "poi": {"category": "museum", "name": "城市博物馆"},
        },
        {
            "start_time": "18:00",
            "duration_min": 60,
            "poi": {
                "category": "food",
                "name": "清真风味餐厅",
                "source": "amap",
                "verification_status": "verified",
            },
        },
    ]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    assert [meal["name"] for meal in strategy["scheduled_meals"]] == [
        "午餐时段（当日活动区域就近自行安排）", "清真风味餐厅",
    ]
    assert strategy["unscheduled_meal_days"] == []


def test_dietary_strategy_adds_grounded_lunch_when_only_dinner_is_scheduled() -> None:
    profile = TravelProfile(
        destination="合成城", days=1,
        constraint_state={"dietary": ["仅清真餐厅"]},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "09:00", "duration_min": 120,
            "poi": {"category": "museum", "name": "城市博物馆"},
        },
        {
            "start_time": "18:00", "duration_min": 60,
            "poi": {
                "poi_id": "dinner", "category": "food", "name": "清真晚餐",
                "source": "amap", "verification_status": "verified",
                "tags": ["halal"],
            },
        },
    ]}]}
    domain_inputs = {"restaurants": [{
        "artifact_id": "restaurants-1",
        "payload": {"status": "verified", "restaurants": [{
            "poi_id": "lunch", "name": "清真午餐候选", "city": "合成城",
            "category": "food", "lat": 30.0, "lng": 120.0,
            "rating": 4.5, "popularity": 0.8, "tags": ["halal"],
            "estimated_duration_min": 60, "price_level": "mid",
            "source": "amap", "verification_status": "verified",
        }]},
    }]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, domain_inputs)

    lunch = next(meal for meal in strategy["scheduled_meals"] if meal.get("period") == "lunch")
    assert lunch["name"] == "清真午餐候选"
    assert lunch["poi_id"] == "lunch"
    assert lunch["source"] == "amap"
    assert lunch["is_reservation_only"] is True


def test_early_dinner_is_reserved_before_long_evening_activity() -> None:
    profile = TravelProfile(
        destination="广州",
        days=1,
        constraint_state={"activity_end_deadline": "20:30"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "09:00",
            "duration_min": 120,
            "poi": {"category": "museum", "name": "城市博物馆"},
        },
        {
            "start_time": "18:00",
            "duration_min": 150,
            "poi": {"category": "scenic", "name": "城市夜景"},
        },
    ]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    dinners = [
        meal for meal in strategy["scheduled_meals"] if meal["name"].startswith("晚餐")
    ]
    assert dinners == [{
        "day_index": 1,
        "period": "dinner",
        "name": "晚餐时段（当日活动区域就近自行安排）",
        "start_time": "16:30",
        "end_time": "17:30",
        "source": "deterministic_schedule_reservation",
        "is_reservation_only": True,
    }]


def test_early_dinner_moves_before_inbound_route_to_late_activity() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"activity_end_deadline": "20:30"},
    )
    itinerary = {"days": [{"day_index": 1, "stops": [
        {
            "start_time": "09:00",
            "duration_min": 150,
            "poi": {"category": "museum", "name": "上午活动"},
        },
        {
            "start_time": "18:00",
            "duration_min": 150,
            "route_from_previous": {"duration_min": 68},
            "poi": {"category": "scenic", "name": "晚间活动"},
        },
    ]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    dinner = next(
        meal for meal in strategy["scheduled_meals"]
        if meal["name"].startswith("晚餐")
    )
    assert dinner["start_time"] == "15:30"
    assert dinner["end_time"] == "16:30"


def test_evening_meal_fallback_is_labeled_as_dinner() -> None:
    profile = TravelProfile(destination="测试城", days=1)
    itinerary = {"days": [{"day_index": 1, "stops": [{
        "start_time": "11:00",
        "duration_min": 240,
        "poi": {"category": "museum", "name": "大型博物馆"},
    }]}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    meal = strategy["scheduled_meals"][0]
    assert meal["start_time"] == "17:30"
    assert meal["name"] == "晚餐时段（当日活动区域就近自行安排）"


def test_dietary_meal_strategy_exposes_fail_closed_selection_policy() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"dietary": ["不吃海鲜", "不太辣"]},
    )
    itinerary = {"days": [{"day_index": 1, "stops": []}]}

    strategy = toolkit._build_meal_strategy(profile, itinerary, {})

    assert strategy["dietary_policy"]["mode"] == "confirm_or_replace"
    assert strategy["dietary_policy"]["requirements"] == ["不吃海鲜", "不太辣"]
    assert "无法确认则更换" in strategy["dietary_policy"]["instruction"]


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
