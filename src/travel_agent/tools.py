from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from travel_agent.schemas import POI, TransportMode


def search_poi(
    pois: list[POI],
    city: str,
    query_tags: list[str] | None = None,
    category: str | None = None,
    max_results: int = 20,
) -> list[POI]:
    """Retrieve local seed POIs by city, optional category, and tags."""
    tags = set(query_tags or [])
    candidates = [poi for poi in pois if poi.city.lower() == city.lower()]
    if category:
        candidates = [poi for poi in candidates if poi.category == category]
    if tags:
        candidates = [
            poi for poi in candidates if tags.intersection(set(poi.tags))
        ]
    return candidates[:max_results]


def estimate_route_minutes(
    origin: POI,
    destination: POI,
    mode: TransportMode = "public_transport",
) -> tuple[float, int]:
    distance_km = haversine_km(origin.lat, origin.lng, destination.lat, destination.lng)
    speed_kmh = {
        "walk": 4.5,
        "public_transport": 18.0,
        "taxi": 25.0,
        "drive": 28.0,
    }[mode]
    overhead_min = {
        "walk": 0,
        "public_transport": 12,
        "taxi": 8,
        "drive": 8,
    }[mode]
    duration_min = max(5, round(distance_km / speed_kmh * 60) + overhead_min)
    return round(distance_km, 2), duration_min


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lng / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(a))
