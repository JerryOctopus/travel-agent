from __future__ import annotations

from typing import Protocol

from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    RouteInfo,
    ScoredPOI,
    TravelProfile,
    TransportMode,
)


class RouteEstimator(Protocol):
    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        ...


START_TIMES = ["09:30", "13:30", "18:00"]


def build_simple_itinerary(
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    route_estimator: RouteEstimator | None = None,
) -> Itinerary:
    if not profile.destination:
        raise ValueError("destination is required to build itinerary")
    if not profile.days:
        raise ValueError("days is required to build itinerary")

    stops_per_day = _stops_per_day(profile)
    total_stops = profile.days * stops_per_day
    selected = ranked_pois[:total_stops]

    days: list[ItineraryDay] = []
    for day_index in range(1, profile.days + 1):
        start = (day_index - 1) * stops_per_day
        day_candidates = selected[start : start + stops_per_day]
        stops = _build_day_stops(day_candidates, profile, route_estimator)
        days.append(
            ItineraryDay(
                day_index=day_index,
                theme=_make_day_theme(stops),
                stops=stops,
            )
        )

    return Itinerary(
        city=profile.destination,
        days=days,
        summary=f"{profile.destination}{profile.days}天{_pace_label(profile)}行程草案",
    )


def _stops_per_day(profile: TravelProfile) -> int:
    if profile.pace == "relaxed":
        return 2
    if profile.pace == "intensive":
        return 3
    return 2


def _build_day_stops(
    day_candidates: list[ScoredPOI],
    profile: TravelProfile,
    route_estimator: RouteEstimator | None,
) -> list[ItineraryStop]:
    stops: list[ItineraryStop] = []
    for index, item in enumerate(day_candidates):
        route_from_previous = None
        if route_estimator and stops:
            route_from_previous = route_estimator.estimate_route(
                stops[-1].poi,
                item.poi,
                profile.transport_mode,
            )
        stops.append(
            ItineraryStop(
                poi=item.poi,
                start_time=START_TIMES[index],
                duration_min=item.poi.estimated_duration_min,
                note=_make_stop_note(item),
                route_from_previous=route_from_previous,
            )
        )
    return stops


def _make_stop_note(item: ScoredPOI) -> str:
    if item.reasons:
        return "；".join(item.reasons)
    return "作为行程候选点纳入规划"


def _make_day_theme(stops: list[ItineraryStop]) -> str:
    categories = [stop.poi.category for stop in stops]
    if "scenic" in categories and "food" in categories:
        return "自然风光与本地美食"
    if "culture" in categories or "museum" in categories:
        return "文化体验"
    if "food" in categories:
        return "本地美食"
    return "城市经典体验"


def _pace_label(profile: TravelProfile) -> str:
    return {
        "relaxed": "轻松",
        "standard": "标准",
        "intensive": "紧凑",
    }[profile.pace]
