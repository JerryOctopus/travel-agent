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
                    "biz_ext": {"rating": "4.8"},
                }
            ],
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    results = provider.search_pois("杭州", query_tags=["nature"])

    assert results[0].source == "amap"
    assert results[0].name == "西湖风景名胜区"
    assert results[0].category == "scenic"
    assert results[0].rating == 4.8


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
                    }
                ]
            },
        }

    monkeypatch.setattr(provider, "_get_json", fake_get_json)

    route = provider.estimate_route(_poi("西湖"), _poi("灵隐寺"))

    assert route.distance_km == 6.8
    assert route.duration_min == 35
    assert route.source == "amap"
