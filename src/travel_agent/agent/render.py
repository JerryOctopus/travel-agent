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

    return_plan = payload.get("return_plan")
    if isinstance(return_plan, dict) and return_plan.get("required"):
        cards.append({"type": "return_plan", **return_plan})

    lodging_plan = payload.get("lodging_plan")
    if isinstance(lodging_plan, dict) and lodging_plan.get("required"):
        cards.append({"type": "lodging_plan", **lodging_plan})

    budget_plan = payload.get("budget_plan")
    if isinstance(budget_plan, dict):
        cards.append({"type": "budget_plan", **budget_plan})

    fixed_event_plan = payload.get("fixed_event_plan")
    if isinstance(fixed_event_plan, dict):
        cards.append({"type": "fixed_event_plan", **fixed_event_plan})

    mobility_plan = payload.get("mobility_plan")
    if isinstance(mobility_plan, dict):
        cards.append({"type": "mobility_plan", **mobility_plan})

    candidate_verification = payload.get("candidate_verification")
    if isinstance(candidate_verification, dict):
        cards.append({"type": "candidate_verification", **candidate_verification})

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

    cards.append(
        {
            "type": "handoff",
            "title": "人工确认后导出",
            "summary": "Agent 只生成路线建议；用户确认或微调后，再导出高德路书草稿。",
            "provider": "amap",
            "safety": {
                "requires_user_confirmation": True,
                "auto_ordering": False,
                "auto_payment": False,
            },
            "roadbook": build_amap_roadbook_payload(itinerary),
        }
    )

    return cards


def build_supplement_cards(
    *,
    restaurants: dict[str, Any] | None = None,
    hotels: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build optional cards for tools that complement, but do not alter, the itinerary."""
    cards: list[dict[str, Any]] = []

    if restaurants:
        cards.append(
            {
                "type": "restaurants",
                "city": restaurants.get("city"),
                "cuisine": restaurants.get("cuisine"),
                "area": restaurants.get("area"),
                "budget_level": restaurants.get("budget_level"),
                "items": [
                    {
                        "name": item.get("name"),
                        "category": item.get("category"),
                        "rating": item.get("rating"),
                        "tags": list(item.get("tags") or [])[:4],
                        "price_level": item.get("price_level"),
                        "source": item.get("source"),
                    }
                    for item in restaurants.get("restaurants", [])[:6]
                ],
            }
        )

    if hotels:
        cards.append(
            {
                "type": "hotels",
                "city": hotels.get("city"),
                "area": hotels.get("area"),
                "budget_level": hotels.get("budget_level"),
                "items": [
                    {
                        "name": item.get("name"),
                        "area": item.get("area"),
                        "rating": item.get("rating"),
                        "price_per_night": item.get("price_per_night"),
                        "source": item.get("source"),
                    }
                    for item in hotels.get("hotels", [])[:5]
                ],
            }
        )

    if budget:
        cards.append(
            {
                "type": "budget",
                "city": budget.get("city"),
                "days": budget.get("days"),
                "companions": budget.get("companions"),
                "budget_level": budget.get("budget_level"),
                "hotel_required": budget.get("hotel_required"),
                "total_low": budget.get("total_low"),
                "total_high": budget.get("total_high"),
                "breakdown": {
                    "住宿": budget.get("hotel"),
                    "餐饮": budget.get("meals"),
                    "门票": budget.get("tickets"),
                    "市内交通": budget.get("inner_city_transport"),
                },
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


def build_amap_roadbook_payload(itinerary: dict[str, Any]) -> dict[str, Any]:
    """灰度上线用的高德路书草稿。

    第一版只在前端导出 JSON，不自动调用高德生产 API。
    """
    days = []
    for day in itinerary.get("days", []):
        stops = []
        for stop in day.get("stops", []):
            poi = stop.get("poi") or {}
            stops.append(
                {
                    "name": poi.get("name"),
                    "lng": poi.get("lng"),
                    "lat": poi.get("lat"),
                    "start_time": stop.get("start_time"),
                    "duration_min": stop.get("duration_min"),
                    "category": poi.get("category"),
                }
            )
        days.append({"day": day.get("day_index"), "stops": stops})
    return {
        "provider": "amap",
        "city": itinerary.get("city"),
        "summary": itinerary.get("summary"),
        "days": days,
        "requires_user_confirmation": True,
    }
