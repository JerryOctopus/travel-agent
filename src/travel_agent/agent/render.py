"""把结构化行程转换为前端可渲染的数据：高德地图 payload 与 A2UI 卡片。

A2UI 卡片是一组带 ``type`` 的简单 dict，前端按类型渲染（天气 / 摘要 / critic /
每日行程）。地图 payload 包含点位 markers 与按天分组的路线折线 routes。
"""

from __future__ import annotations

from typing import Any

_WEATHER_LABEL = {
    "sunny": "晴",
    "cloudy": "多云",
    "rain": "雨",
    "rainy": "雨",
    "unknown": "未知",
}

_CATEGORY_COLOR = {
    "scenic": "#2e7d32",
    "culture": "#6a1b9a",
    "museum": "#1565c0",
    "food": "#e65100",
    "shopping": "#ad1457",
}


def _is_rainy(condition: str | None) -> bool:
    if not condition:
        return False
    return "雨" in condition or condition.lower() in {"rain", "rainy"}


def build_itinerary_cards(payload: dict[str, Any], weather: dict[str, Any] | None) -> list[dict[str, Any]]:
    itinerary = payload["itinerary"]
    critic = payload.get("critic", {})
    cards: list[dict[str, Any]] = []

    if weather:
        condition = weather.get("condition")
        cards.append(
            {
                "type": "weather",
                "city": weather.get("city"),
                "condition": _WEATHER_LABEL.get(condition, condition),
                "temperature_c": weather.get("temperature_c"),
                "source": weather.get("source"),
                "rainy": _is_rainy(condition),
            }
        )

    cards.append(
        {
            "type": "summary",
            "city": itinerary.get("city"),
            "summary": itinerary.get("summary"),
            "day_count": len(itinerary.get("days", [])),
        }
    )

    cards.append(
        {
            "type": "critic",
            "passed": critic.get("passed", True),
            "original_issue_count": payload.get("original_issue_count", 0),
            "final_issue_count": payload.get("final_issue_count", len(critic.get("issues", []))),
            "iterations": payload.get("iterations", 0),
            "revision_notes": payload.get("revision_notes", []),
            "issues": critic.get("issues", []),
        }
    )

    for day in itinerary.get("days", []):
        stops = []
        for stop in day.get("stops", []):
            poi = stop["poi"]
            route = stop.get("route_from_previous")
            stops.append(
                {
                    "start_time": stop.get("start_time"),
                    "name": poi.get("name"),
                    "category": poi.get("category"),
                    "duration_min": stop.get("duration_min"),
                    "note": stop.get("note"),
                    "indoor": poi.get("indoor", False),
                    "rating": poi.get("rating"),
                    "route_from_previous": (
                        {
                            "duration_min": route.get("duration_min"),
                            "distance_km": route.get("distance_km"),
                            "mode": route.get("mode"),
                        }
                        if route
                        else None
                    ),
                }
            )
        cards.append(
            {
                "type": "day",
                "day_index": day.get("day_index"),
                "theme": day.get("theme"),
                "stops": stops,
            }
        )

    return cards


def build_map_payload(itinerary: dict[str, Any]) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    all_lng: list[float] = []
    all_lat: list[float] = []

    for day in itinerary.get("days", []):
        day_index = day.get("day_index")
        path: list[list[float]] = []
        for stop in day.get("stops", []):
            poi = stop["poi"]
            lng, lat = float(poi["lng"]), float(poi["lat"])
            all_lng.append(lng)
            all_lat.append(lat)
            path.append([lng, lat])
            markers.append(
                {
                    "name": poi.get("name"),
                    "lng": lng,
                    "lat": lat,
                    "day": day_index,
                    "category": poi.get("category"),
                    "color": _CATEGORY_COLOR.get(poi.get("category"), "#1976d2"),
                    "start_time": stop.get("start_time"),
                }
            )
        if len(path) >= 2:
            routes.append({"day": day_index, "path": path})

    center = (
        [sum(all_lng) / len(all_lng), sum(all_lat) / len(all_lat)]
        if all_lng
        else [116.397, 39.908]
    )
    return {
        "city": itinerary.get("city"),
        "center": center,
        "markers": markers,
        "routes": routes,
    }
