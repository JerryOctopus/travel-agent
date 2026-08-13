from travel_agent.providers import AmapToolProvider, FallbackToolProvider, LocalToolProvider
from travel_agent.schemas import POI, WeatherInfo


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
        assert path == "/v3/place/text"
        assert params["city"] == "杭州"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "B001",
                    "name": "西湖风景名胜区",
                    "type": "风景名胜;风景名胜;国家级景点",
                    "location": "120.149,30.259",
                    "biz_ext": {
                        "rating": "4.8",
                        "open_time": "09:00-17:00",
                        "opentime2": "周一至周日 09:00-17:00",
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


def test_amap_named_venue_query_allows_cross_city_and_keeps_actual_city(monkeypatch) -> None:
    provider = AmapToolProvider(api_key="fake-key")

    def fake_get_json(path: str, params: dict[str, str]) -> dict:
        assert params["city"] == "成都"
        assert params["citylimit"] == "false"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "sxd",
                    "name": "三星堆博物馆",
                    "cityname": "广汉",
                    "type": "科教文化服务;博物馆",
                    "location": "104.2,31.0",
                    "biz_ext": {"rating": "4.9"},
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
