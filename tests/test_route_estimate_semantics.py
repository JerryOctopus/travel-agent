from travel_agent.schemas import POI
from travel_agent.tools import estimate_route_minutes


def _poi(poi_id: str, lat: float, lng: float) -> POI:
    return POI(poi_id, poi_id, "测试城", "scenic", lat, lng, 4.5, 0.8, [], 60, "mid")


def test_public_transport_fallback_includes_access_and_wait_overhead() -> None:
    origin = _poi("a", 31.2304, 121.4737)
    destination = _poi("b", 31.2488, 121.4930)

    distance, duration = estimate_route_minutes(origin, destination, "public_transport")

    assert 1.0 < distance < 3.0
    assert duration >= 15
