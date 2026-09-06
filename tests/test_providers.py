import json
from dataclasses import replace

import pytest

from travel_agent.providers import (
    AmapToolProvider,
    FallbackToolProvider,
    LocalToolProvider,
    ProviderRateLimitError,
)
from travel_agent.schemas import POI, RouteInfo, WeatherInfo


class _JSONResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _poi(name: str = "西湖") -> POI:
    return POI(
        poi_id="p1",
        name=name,
        city="杭州",
        category="scenic",
        lat=30.25,
        lng=120.15,
        rating=4.8,
        popularity=0.9,
        tags=["nature"],
        estimated_duration_min=120,
        price_level="free",
    )


def test_local_tool_provider_searches_seed_pois() -> None:
    provider = LocalToolProvider([_poi()])

    results = provider.search_pois("杭州", query_tags=["nature"])

    assert results[0].name == "西湖"
    assert results[0].source == "seed"


def test_fallback_tool_provider_uses_local_when_primary_fails() -> None:
    class BrokenProvider:
        def search_pois(self, *args, **kwargs):
            raise RuntimeError("api error")

        def get_weather(self, city: str) -> WeatherInfo:
            raise RuntimeError("api error")

    fallback = LocalToolProvider([_poi()])
    provider = FallbackToolProvider(primary=BrokenProvider(), fallback=fallback)

    assert provider.search_pois("杭州")[0].name == "西湖"
    assert provider.get_weather("杭州").source == "mock"
    assert provider.estimate_route(_poi("西湖"), _poi("灵隐寺")).source == "haversine_estimate"


def test_fallback_tool_provider_retries_transient_primary_route_once() -> None:
    calls = 0

    class FlakyPrimary:
        def estimate_route(self, origin, destination, mode):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("temporary route connection failure")
            return RouteInfo(
                origin_poi_id=origin.poi_id,
                destination_poi_id=destination.poi_id,
                distance_km=2.1,
                duration_min=18,
                mode=mode,
                source="amap",
                evidence_status="provider_verified",
            )

    provider = FallbackToolProvider(
        primary=FlakyPrimary(),
        fallback=LocalToolProvider([_poi()]),
    )

    route = provider.estimate_route(
        replace(_poi("甲馆"), poi_id="origin"),
        replace(_poi("乙馆"), poi_id="destination"),
    )

    assert calls == 2
    assert route.source == "amap"
    assert route.evidence_status == "provider_verified"


def test_fallback_tool_provider_never_retries_provider_quota_route() -> None:
    calls = 0

    class LimitedPrimary:
        def estimate_route(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise ProviderRateLimitError(
                "AMap rate limit: USER_DAILY_QUERY_OVER_LIMIT (10044)"
            )

    provider = FallbackToolProvider(
        primary=LimitedPrimary(),
        fallback=LocalToolProvider([_poi()]),
    )

    with pytest.raises(ProviderRateLimitError, match="10044"):
        provider.estimate_route(_poi("甲馆"), _poi("乙馆"))
    assert calls == 1


def test_fallback_tool_provider_does_not_hide_primary_rate_limit() -> None:
    class LimitedProvider:
        def search_pois(self, *args, **kwargs):
            raise ProviderRateLimitError("AMap rate limit: USER_DAILY_QUERY_OVER_LIMIT (10044)")

    provider = FallbackToolProvider(
        primary=LimitedProvider(),
        fallback=LocalToolProvider([_poi()]),
    )

    with pytest.raises(ProviderRateLimitError, match="10044"):
        provider.search_pois("杭州")


def test_fallback_tool_provider_delegates_nearby_search_and_preserves_rate_limits() -> None:
    anchor = _poi("锚点")
    nearby = replace(_poi("附近餐厅"), category="food")

    class Primary:
        def search_pois_nearby(self, *args, **kwargs):
            assert kwargs["anchor"] is anchor
            return [nearby]

    provider = FallbackToolProvider(
        primary=Primary(),
        fallback=LocalToolProvider([]),
    )

    assert provider.search_pois_nearby("杭州", anchor=anchor)[0].name == "附近餐厅"


def test_local_tool_provider_estimates_route() -> None:
    provider = LocalToolProvider([_poi()])
    route = provider.estimate_route(_poi("西湖"), _poi("灵隐寺"), mode="taxi")

    assert route.origin_poi_id == "p1"
    assert route.destination_poi_id == "p1"
    assert route.distance_km >= 0
    assert route.duration_min >= 5
    assert route.mode == "taxi"
    assert route.source == "haversine_estimate"


def test_amap_provider_maps_poi_payload(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert path == "/v5/place/text"
        assert params["region"] == "杭州"
        assert params["city_limit"] == "true"
        assert params["page_size"] == "20"
        assert params["page_num"] == "1"
        assert params["show_fields"] == "business"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "B001",
                    "name": "西湖风景名胜区",
                    "type": "风景名胜;风景名胜;国家级景点",
                    "location": "120.149,30.259",
                    "adname": "西湖区",
                    "address": "龙井路1号",
                    "business": {
                        "rating": "4.8",
                        "opentime_today": "09:00-17:00",
                        "opentime_week": "周一至周日 09:00-17:00",
                    },
                }
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("杭州", query_tags=["nature"])

    assert results[0].source == "amap"
    assert results[0].opening_hours == "周一至周日 09:00-17:00"
    assert results[0].name == "西湖风景名胜区"
    assert results[0].category == "scenic"
    assert results[0].rating == 4.8
    assert results[0].address == "西湖区龙井路1号"


def test_amap_place_cache_is_shared_across_provider_instances(monkeypatch) -> None:
    AmapToolProvider.clear_place_cache()
    calls = 0

    def fake_get_json(self, path: str, params: dict[str, str]) -> dict:
        nonlocal calls
        calls += 1
        return {
            "status": "1",
            "pois": [{
                "id": "B001",
                "name": "西湖风景名胜区",
                "type": "风景名胜;风景名胜;国家级景点",
                "location": "120.149,30.259",
            }],
        }

    monkeypatch.setattr(AmapToolProvider, "_get_json", fake_get_json)
    first = AmapToolProvider(api_key="same-key")
    second = AmapToolProvider(api_key="same-key")

    assert first.search_pois("杭州", category="景点")
    assert second.search_pois("杭州", category="景点")
    assert calls == 1


def test_amap_place_cache_never_caches_rate_limit(monkeypatch) -> None:
    AmapToolProvider.clear_place_cache()
    calls = 0

    def fake_get_json(self, path: str, params: dict[str, str]) -> dict:
        nonlocal calls
        calls += 1
        return {
            "status": "0",
            "info": "USER_DAILY_QUERY_OVER_LIMIT",
            "infocode": "10044",
        }

    monkeypatch.setattr(AmapToolProvider, "_get_json", fake_get_json)
    provider = AmapToolProvider(api_key="same-key")

    for _ in range(2):
        with pytest.raises(ProviderRateLimitError, match="10044"):
            provider.search_pois("杭州", category="景点")
    assert calls == 2


def test_amap_provider_raises_rate_limit_instead_of_returning_empty(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")
    monkeypatch.setattr(
        provider,
        "_get_json",
        lambda *_args, **_kwargs: {
            "status": "0",
            "info": "USER_DAILY_QUERY_OVER_LIMIT",
            "infocode": "10044",
        },
    )

    with pytest.raises(ProviderRateLimitError, match="10044"):
        provider.search_pois("杭州", category="景点")


def test_amap_get_json_retries_transient_qps_limit(monkeypatch) -> None:
    import travel_agent.providers as provider_module

    payloads = iter([
        {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10021"},
        {"status": "1", "pois": []},
    ])
    calls = 0

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _JSONResponse(next(payloads))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(provider_module, "_AMAP_MIN_REQUEST_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)

    payload = AmapToolProvider(api_key="fake-key")._get_json("/v5/place/text", {})

    assert payload["status"] == "1"
    assert calls == 2


def test_amap_get_json_does_not_retry_daily_quota(monkeypatch) -> None:
    import travel_agent.providers as provider_module

    calls = 0

    def fake_urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _JSONResponse({
            "status": "0",
            "info": "USER_DAILY_QUERY_OVER_LIMIT",
            "infocode": "10044",
        })

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(provider_module, "_AMAP_MIN_REQUEST_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)

    payload = AmapToolProvider(api_key="fake-key")._get_json("/v5/place/text", {})

    assert payload["infocode"] == "10044"
    assert calls == 1


def test_amap_food_search_uses_taxonomy_filter_and_rejects_non_food_payload(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert path == "/v5/place/text"
        assert params["types"] == "050000"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "office",
                    "name": "商务大厦",
                    "type": "商务住宅;楼宇;商务写字楼",
                    "location": "120.1,30.2",
                },
                {
                    "id": "meal",
                    "name": "青禾餐厅",
                    "type": "餐饮服务;中餐厅",
                    "location": "120.11,30.21",
                },
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("杭州", query_tags=["星光大道"], category="food")

    assert [item.name for item in results] == ["青禾餐厅"]


def test_amap_nearby_search_uses_anchor_radius_and_food_taxonomy(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")
    anchor = POI(
        "anchor", "合成景区", "合成城", "scenic", 34.384, 109.278,
        4.8, 0.9, [], 120, "unknown",
    )

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert path == "/v5/place/around"
        assert params["location"] == "109.278000,34.384000"
        assert params["radius"] == "8000"
        assert params["region"] == "合成城"
        assert params["city_limit"] == "true"
        assert params["sortrule"] == "distance"
        assert params["types"] == "050000"
        assert params["keywords"] == "清真餐厅"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "near-meal",
                    "name": "清真风味餐厅",
                    "type": "餐饮服务;中餐厅",
                    "location": "109.279,34.385",
                    "address": "景区东路",
                },
                {
                    "id": "near-shop",
                    "name": "纪念品店",
                    "type": "购物服务;特色商业街",
                    "location": "109.280,34.386",
                },
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois_nearby(
        "合成城",
        anchor,
        query_tags=["清真餐厅"],
        category="food",
        radius_m=8000,
    )

    assert [item.name for item in results] == ["清真风味餐厅"]


def test_amap_search_normalizes_chinese_category_before_taxonomy_filter(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert params["types"] == "110000"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "scenic",
                    "name": "海滨公园",
                    "type": "风景名胜;公园广场;公园",
                    "location": "118.1,24.4",
                },
                {
                    "id": "salon",
                    "name": "海滨造型",
                    "type": "生活服务;美容美发店",
                    "location": "118.11,24.41",
                },
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("厦门", category="景点")

    assert [item.name for item in results] == ["海滨公园"]
    assert results[0].category == "scenic"


def test_amap_search_does_not_exact_filter_unrecognized_named_category(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert "types" not in params
        assert "鼓浪屿" in params["keywords"]
        return {
            "status": "1",
            "pois": [
                {
                    "id": "gulangyu",
                    "name": "鼓浪屿",
                    "type": "风景名胜;风景名胜;国家级景点",
                    "location": "118.06,24.45",
                }
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("厦门", category="鼓浪屿")

    assert [item.name for item in results] == ["鼓浪屿"]


def test_amap_v5_search_uses_pipe_separated_keywords_and_keeps_type_filter(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert path == "/v5/place/text"
        assert params["keywords"] == "鼓浪屿|海边|景点"
        assert params["types"] == "110000"
        assert params["city_limit"] == "false"
        return {"status": "1", "pois": []}

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    provider.search_pois(
        "厦门", query_tags=["海边", "鼓浪屿"], category="景点"
    )


def test_amap_v5_search_treats_generic_chinese_interests_as_city_scoped(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert params["keywords"] == "海边|自然风光|景点"
        assert params["city_limit"] == "true"
        return {"status": "1", "pois": []}

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    provider.search_pois("厦门", query_tags=["海边", "nature"], category="景点")


def test_amap_v5_search_drops_mobility_constraints_from_venue_keywords(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert path == "/v5/place/text"
        assert params["keywords"] == "西湖"
        assert params["city_limit"] == "false"
        return {"status": "1", "pois": []}

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    provider.search_pois("杭州", query_tags=["西湖", "老年人", "少步行"])


def test_amap_hotel_area_query_remains_city_scoped(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert params["keywords"] == "人民广场附近"
        assert params["types"] == "100000"
        assert params["city_limit"] == "true"
        return {"status": "1", "pois": []}

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    provider.search_pois(
        "上海", query_tags=["人民广场附近"], category="hotel"
    )


def test_amap_named_venue_query_allows_cross_city_and_keeps_actual_city(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert params["region"] == "成都"
        assert params["city_limit"] == "false"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "sxd",
                    "name": "三星堆博物馆",
                    "cityname": "广汉",
                    "type": "科教文化服务;博物馆",
                    "location": "104.2,31.0",
                    "business": {"rating": "4.9"},
                }
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("成都", query_tags=["三星堆博物馆"])

    assert results[0].city == "广汉"
    assert results[0].name == "三星堆博物馆"


def test_amap_provider_maps_transit_stop_as_non_activity(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")
    monkeypatch.setattr(
        provider,
        "_get_json",
        lambda *_args, **_kwargs: {
            "status": "1",
            "pois": [
                {
                    "id": "bus-1",
                    "name": "苏州博物馆本馆(公交站)",
                    "location": "120.0,31.0",
                    "type": "交通设施服务;公交车站",
                    "biz_ext": {},
                }
            ],
        },
    )

    result = provider.search_pois("苏州", ["苏州博物馆"])

    assert result[0].category == "transport"
    assert result[0].estimated_duration_min == 0


def test_amap_provider_uses_varied_duration_by_poi_type(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        return {
            "status": "1",
            "pois": [
                {
                    "id": "B001",
                    "name": "城市观景台",
                    "type": "风景名胜;风景名胜;观景台",
                    "location": "120.149,30.259",
                    "biz_ext": {"rating": "4.8"},
                },
                {
                    "id": "B002",
                    "name": "国家森林公园",
                    "type": "风景名胜;风景名胜;国家森林公园",
                    "location": "120.150,30.260",
                    "biz_ext": {"rating": "4.7"},
                },
                {
                    "id": "B003",
                    "name": "本地小吃街",
                    "type": "购物服务;特色商业街;步行街",
                    "location": "120.151,30.261",
                    "biz_ext": {"rating": "4.6"},
                },
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("杭州", query_tags=["nature"])
    durations = [poi.estimated_duration_min for poi in results]

    assert durations == [90, 150, 90]
    assert len(set(durations)) > 1


def test_amap_provider_fetches_weather_with_geocode(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        if path == "/v3/geocode/geo":
            return {"status": "1", "geocodes": [{"adcode": "330100"}]}
        return {
            "status": "1",
            "lives": [{"weather": "小雨", "temperature": "24"}],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    weather = provider.get_weather("杭州")

    assert weather.city == "杭州"
    assert weather.condition == "小雨"
    assert weather.temperature_c == 24
    assert weather.source == "amap"


def test_amap_provider_estimates_public_transport_route(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert path == "/v3/direction/transit/integrated"
        assert params["origin"]
        assert params["destination"]
        return {
            "status": "1",
            "route": {
                "transits": [
                    {
                        "distance": "6800",
                        "duration": "2100",
                        "walking_distance": "850",
                    }
                ]
            },
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    route = provider.estimate_route(_poi("西湖"), _poi("灵隐寺"))

    assert route.distance_km == 6.8
    assert route.duration_min == 35
    assert route.walking_distance_km == 0.85
    assert route.source == "amap"


def test_amap_route_cache_is_shared_across_provider_instances(monkeypatch) -> None:
    AmapToolProvider.clear_route_cache()
    calls = 0

    def fake_get_json(self, path: str, params: dict[str, str]) -> dict:
        nonlocal calls
        calls += 1
        return {
            "status": "1",
            "route": {
                "transits": [{
                    "distance": "6800",
                    "duration": "2100",
                    "walking_distance": "850",
                }]
            },
        }

    monkeypatch.setattr(AmapToolProvider, "_get_json", fake_get_json)
    first = AmapToolProvider(api_key="same-key")
    second = AmapToolProvider(api_key="same-key")
    origin, destination = _poi("西湖"), _poi("灵隐寺")

    assert first.estimate_route(origin, destination).source == "amap"
    assert second.estimate_route(origin, destination).source == "amap"
    assert calls == 1


def test_amap_route_cache_never_caches_rate_limit(monkeypatch) -> None:
    AmapToolProvider.clear_route_cache()
    calls = 0

    def fake_get_json(self, path: str, params: dict[str, str]) -> dict:
        nonlocal calls
        calls += 1
        return {
            "status": "0",
            "info": "USER_DAILY_QUERY_OVER_LIMIT",
            "infocode": "10044",
        }

    monkeypatch.setattr(AmapToolProvider, "_get_json", fake_get_json)
    provider = AmapToolProvider(api_key="same-key")

    for _ in range(2):
        with pytest.raises(ProviderRateLimitError, match="10044"):
            provider.estimate_route(_poi("西湖"), _poi("灵隐寺"))
    assert calls == 2


def test_amap_provider_retries_one_empty_route_response(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")
    calls = 0

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "1", "route": {"transits": []}}
        return {
            "status": "1",
            "route": {
                "transits": [{
                    "distance": "6800",
                    "duration": "2100",
                    "walking_distance": "850",
                }]
            },
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    route = provider.estimate_route(_poi("西湖"), _poi("灵隐寺"))

    assert calls == 2
    assert route.source == "amap"
    assert route.evidence_status == "provider_verified"


def test_amap_provider_uses_verified_walk_for_short_empty_transit(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")
    origin = replace(_poi("甲馆"), poi_id="origin", lng=121.492497, lat=31.227714)
    destination = replace(
        _poi("乙馆"), poi_id="destination", lng=121.490405, lat=31.239137
    )
    calls: list[str] = []

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        calls.append(path)
        if path == "/v3/direction/transit/integrated":
            return {"status": "1", "route": {"transits": []}}
        assert path == "/v3/direction/walking"
        return {
            "status": "1",
            "route": {"paths": [{"distance": "1400", "duration": "1080"}]},
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    route = provider.estimate_route(origin, destination, mode="public_transport")

    assert calls == [
        "/v3/direction/transit/integrated",
        "/v3/direction/transit/integrated",
        "/v3/direction/walking",
    ]
    assert route.origin_poi_id == "origin"
    assert route.destination_poi_id == "destination"
    assert route.mode == "walk"
    assert route.source == "amap"
    assert route.evidence_status == "provider_verified"
