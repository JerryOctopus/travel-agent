"""POI / 行程等结构与 JSON 友好 dict 之间的互转。

工具的入参/出参必须是 JSON 可序列化的（LLM tool calling 约束），因此这里集中
处理 dataclass <-> dict 的转换，避免在各工具里重复且容易出错的手写序列化。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from travel_agent.schemas import (
    POI,
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    RouteInfo,
)


def poi_to_dict(poi: POI) -> dict[str, Any]:
    return asdict(poi)


def poi_from_dict(data: dict[str, Any]) -> POI:
    return POI(
        poi_id=str(data["poi_id"]),
        name=str(data["name"]),
        city=str(data["city"]),
        category=str(data["category"]),
        lat=float(data["lat"]),
        lng=float(data["lng"]),
        rating=float(data.get("rating", 4.0)),
        popularity=float(data.get("popularity", 0.5)),
        tags=list(data.get("tags", [])),
        estimated_duration_min=int(data.get("estimated_duration_min", 90)),
        price_level=str(data.get("price_level", "mid")),
        indoor=bool(data.get("indoor", False)),
        opening_hours=data.get("opening_hours"),
        source=str(data.get("source", "seed")),
    )


def poi_brief(poi: POI) -> dict[str, Any]:
    """给 LLM 看的紧凑摘要，省略经纬度等无关字段以节省 token。"""
    return {
        "poi_id": poi.poi_id,
        "name": poi.name,
        "category": poi.category,
        "rating": poi.rating,
        "tags": poi.tags,
        "indoor": poi.indoor,
        "duration_min": poi.estimated_duration_min,
        "source": poi.source,
    }


def route_to_dict(route: RouteInfo | None) -> dict[str, Any] | None:
    if route is None:
        return None
    return asdict(route)


def stop_to_dict(stop: ItineraryStop) -> dict[str, Any]:
    return {
        "poi": poi_to_dict(stop.poi),
        "start_time": stop.start_time,
        "duration_min": stop.duration_min,
        "note": stop.note,
        "route_from_previous": route_to_dict(stop.route_from_previous),
    }


def day_to_dict(day: ItineraryDay) -> dict[str, Any]:
    return {
        "day_index": day.day_index,
        "theme": day.theme,
        "stops": [stop_to_dict(stop) for stop in day.stops],
    }


def itinerary_to_dict(itinerary: Itinerary) -> dict[str, Any]:
    return {
        "city": itinerary.city,
        "summary": itinerary.summary,
        "days": [day_to_dict(day) for day in itinerary.days],
    }
