"""Route evidence classification and endpoint-binding invariants."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from travel_agent.schemas import Itinerary, RouteInfo, TransportMode


SHORT_WALK_DISTANCE_KM = 0.8
LODGING_ACTIVITY_AREA_CONFLICT_KM = 20.0
HARD_ROUTE_CONTEXTS = frozenset(
    {"return_deadline", "fixed_appointment", "accessibility", "tight_transfer", "intercity"}
)
LOW_CONFIDENCE_ROUTE_STATUSES = frozenset({"haversine_estimate", "unavailable"})
CONTROLLED_ROUTE_STATUSES = frozenset(
    {"provider_verified", "deterministic_estimate", "haversine_estimate", "unavailable"}
)


def canonical_route_evidence_status(route: Mapping[str, Any] | RouteInfo | None) -> str:
    """Return a fail-closed status based on actual, endpoint-bound route evidence."""
    if isinstance(route, RouteInfo):
        route = vars(route)
    if not isinstance(route, Mapping) or not route:
        return "unavailable"
    declared = str(route.get("evidence_status") or "").strip()
    if declared == "unavailable":
        return "unavailable"
    inferred = evidence_status_for_source(str(route.get("source") or ""))
    status = declared if declared in CONTROLLED_ROUTE_STATUSES else inferred
    if status == "provider_verified":
        if inferred != "provider_verified" or not _route_payload_complete(route):
            return "unavailable"
        return "provider_verified"
    if status in {"deterministic_estimate", "haversine_estimate"}:
        return status if _route_payload_complete(route) else "unavailable"
    return "unavailable"


def route_evidence_reason_codes(route: Mapping[str, Any] | None) -> list[str]:
    """Explain contradictions without silently upgrading missing evidence."""
    if not isinstance(route, Mapping) or not route:
        return ["route_evidence_unavailable"]
    declared = str(route.get("evidence_status") or "").strip()
    canonical = canonical_route_evidence_status(route)
    codes: list[str] = []
    if declared == "provider_verified" and canonical != "provider_verified":
        codes.append("provider_verified_route_evidence_missing")
    if declared == "provider_verified" and str(route.get("source") or "") == "unavailable":
        codes.append("provider_verified_with_unavailable_source")
    if declared == "provider_verified" and not _route_payload_complete(route):
        codes.append("provider_verified_route_payload_incomplete")
    return list(dict.fromkeys(codes))


def _route_payload_complete(route: Mapping[str, Any]) -> bool:
    if not str(route.get("origin_poi_id") or "").strip():
        return False
    if not str(route.get("destination_poi_id") or "").strip():
        return False
    try:
        return float(route.get("duration_min") or 0) > 0 and float(route.get("distance_km") or 0) >= 0
    except (TypeError, ValueError):
        return False


def evidence_status_for_source(source: str | None) -> str:
    value = str(source or "").casefold()
    if not value or value == "unavailable":
        return "unavailable"
    if "haversine" in value or "fallback_estimate" in value or "recovery_estimate" in value:
        return "haversine_estimate"
    if "deterministic" in value:
        return "deterministic_estimate"
    if any(provider in value for provider in ("amap", "gaode", "baidu", "google", "mapbox", "provider")):
        return "provider_verified"
    return "unavailable"


def normalize_route_evidence(route: RouteInfo, *, provider_explicit: bool | None = None) -> RouteInfo:
    status = (
        route.evidence_status
        if route.evidence_status != "unavailable"
        else evidence_status_for_source(route.source)
    )
    if provider_explicit is True:
        status = "provider_verified"
    elif provider_explicit is False and status == "provider_verified":
        status = evidence_status_for_source(route.source)
    explicit = status == "provider_verified" if provider_explicit is None else provider_explicit
    mode: TransportMode = route.mode
    if 0 < route.distance_km <= SHORT_WALK_DISTANCE_KM and not explicit:
        mode = "walk"
    return replace(route, mode=mode, evidence_status=status)  # type: ignore[arg-type]


def route_supports_endpoints(route: RouteInfo, origin_poi_id: str, destination_poi_id: str) -> bool:
    return (
        route.origin_poi_id == origin_poi_id
        and route.destination_poi_id == destination_poi_id
    )


def route_can_prove_hard_feasibility(route: RouteInfo | None, context: str) -> bool:
    if context not in HARD_ROUTE_CONTEXTS:
        return route is not None and route.evidence_status != "unavailable"
    return route is not None and route.evidence_status not in LOW_CONFIDENCE_ROUTE_STATUSES


def itinerary_route_violations(itinerary: Itinerary) -> list[tuple[str, str]]:
    """Return deterministic (code, message) violations for final adjacent legs."""
    violations: list[tuple[str, str]] = []
    for day in itinerary.days:
        for index, stop in enumerate(day.stops):
            route = stop.route_from_previous
            # Legacy/manual RouteInfo without an evidence classification is
            # not admissible evidence and therefore cannot create a false
            # endpoint claim. Generated/provider routes are always classified.
            if route is not None and route.evidence_status == "unavailable":
                continue
            if index == 0:
                if route is not None and route.destination_poi_id != stop.poi.poi_id:
                    violations.append((
                        "route_endpoint_mismatch",
                        f"第{day.day_index}天首站路线终点 {route.destination_poi_id} 与实际站点 {stop.poi.poi_id} 不一致。",
                    ))
                continue
            previous = day.stops[index - 1]
            if route is None:
                violations.append((
                    "route_evidence_missing",
                    f"第{day.day_index}天 {previous.poi.name} → {stop.poi.name} 缺少路线证据。",
                ))
            elif not route_supports_endpoints(route, previous.poi.poi_id, stop.poi.poi_id):
                violations.append((
                    "route_endpoint_mismatch",
                    f"第{day.day_index}天路线端点 {route.origin_poi_id} → {route.destination_poi_id} 与相邻站点 {previous.poi.poi_id} → {stop.poi.poi_id} 不一致。",
                ))
    return violations
