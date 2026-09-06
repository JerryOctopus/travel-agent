from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MEAL_TIME_HOURS = {11, 12, 17, 18, 19}

PACE_MAX_STOPS = {
    "relaxed": 2,
    "standard": 3,
    "intensive": 4,
}

PACE_MAX_SINGLE_ROUTE_MIN = {
    "relaxed": 45,
    "standard": 60,
    "intensive": 75,
}

PACE_MAX_DAILY_ROUTE_MIN = {
    "relaxed": 75,
    "standard": 110,
    "intensive": 150,
}

INTEREST_TO_CATEGORIES = {
    "food": {"food"},
    "history": {"culture", "museum"},
    "culture": {"culture", "museum"},
    "museum": {"museum"},
    "nature": {"scenic"},
    "citywalk": {"scenic", "shopping", "culture"},
    "shopping": {"shopping"},
    "nightlife": {"food", "scenic"},
    "family": {"family", "museum", "scenic"},
    "couple": {"scenic", "culture", "food"},
}


@dataclass(frozen=True)
class PlanEvalResult:
    environment_pass: bool
    constraint_pass: bool
    preference_pass: bool
    final_pass: bool
    meal_time_valid: bool
    route_feasible: bool
    daily_load_valid: bool
    poi_city_valid: bool
    interest_coverage_valid: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_pass": self.environment_pass,
            "constraint_pass": self.constraint_pass,
            "preference_pass": self.preference_pass,
            "final_pass": self.final_pass,
            "meal_time_valid": self.meal_time_valid,
            "route_feasible": self.route_feasible,
            "daily_load_valid": self.daily_load_valid,
            "poi_city_valid": self.poi_city_valid,
            "interest_coverage_valid": self.interest_coverage_valid,
            "issues": list(self.issues),
        }


def evaluate_plan_artifact(
    artifact: dict[str, Any] | None,
    *,
    profile: dict[str, Any] | None = None,
    required_interests: list[str] | None = None,
    supporting_artifacts: dict[str, Any] | None = None,
) -> PlanEvalResult | None:
    if not artifact:
        return None
    itinerary = artifact.get("itinerary") or {}
    profile = profile or {}
    issues: list[str] = []

    poi_city_valid = _check_poi_city_valid(itinerary, profile, issues)
    meal_time_valid = _check_meal_time_valid(itinerary, issues)
    daily_load_valid = _check_daily_load_valid(itinerary, profile, issues)
    route_feasible = _check_route_feasible(itinerary, profile, issues)
    interest_coverage_valid = _check_interest_coverage(
        itinerary,
        required_interests or profile.get("interests") or [],
        issues,
        supporting_artifacts=supporting_artifacts,
    )

    environment_pass = poi_city_valid
    constraint_pass = meal_time_valid and daily_load_valid and route_feasible
    preference_pass = interest_coverage_valid
    final_pass = environment_pass and constraint_pass and preference_pass

    return PlanEvalResult(
        environment_pass=environment_pass,
        constraint_pass=constraint_pass,
        preference_pass=preference_pass,
        final_pass=final_pass,
        meal_time_valid=meal_time_valid,
        route_feasible=route_feasible,
        daily_load_valid=daily_load_valid,
        poi_city_valid=poi_city_valid,
        interest_coverage_valid=interest_coverage_valid,
        issues=issues,
    )


def _check_poi_city_valid(
    itinerary: dict[str, Any], profile: dict[str, Any], issues: list[str]
) -> bool:
    target_city = itinerary.get("city")
    state = profile.get("constraint_state") or {}
    explicit_cross_city_terms = [str(item) for item in (profile.get("must_visit") or [])]
    explicit_cross_city_terms.extend(
        str(event.get("location") or "")
        for event in (state.get("fixed_events") or [])
        if isinstance(event, dict)
    )
    allowed_cities = {str(item) for item in (state.get("destinations") or []) if item}
    ok = True
    for stop in _iter_stops(itinerary):
        poi = stop.get("poi") or {}
        city = poi.get("city")
        name = str(poi.get("name") or "")
        explicitly_required = any(
            term and (term in name or name in term) for term in explicit_cross_city_terms
        )
        if (
            target_city
            and city
            and _normalize_city(city) != _normalize_city(target_city)
            and _normalize_city(city)
            not in {_normalize_city(item) for item in allowed_cities}
            and not explicitly_required
        ):
            issues.append(f"poi_city_mismatch:{poi.get('name')}:{city}!={target_city}")
            ok = False
        if not _valid_coordinate(poi.get("lat"), poi.get("lng")):
            issues.append(f"invalid_coordinate:{poi.get('name')}")
            ok = False
    return ok


def _normalize_city(value: Any) -> str:
    text = str(value or "").strip()
    for suffix in ("特别行政区", "自治州", "地区", "盟", "市"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _check_meal_time_valid(itinerary: dict[str, Any], issues: list[str]) -> bool:
    ok = True
    for stop in _iter_stops(itinerary):
        poi = stop.get("poi") or {}
        if poi.get("category") != "food":
            continue
        hour = _start_hour(stop.get("start_time"))
        if hour not in MEAL_TIME_HOURS:
            issues.append(f"meal_time_invalid:{poi.get('name')}:{stop.get('start_time')}")
            ok = False
    return ok


def _check_daily_load_valid(
    itinerary: dict[str, Any],
    profile: dict[str, Any],
    issues: list[str],
) -> bool:
    pace = str(profile.get("pace") or "standard")
    max_stops = PACE_MAX_STOPS.get(pace, PACE_MAX_STOPS["standard"])
    ok = True
    for day in itinerary.get("days", []):
        count = len(day.get("stops", []))
        if count > max_stops:
            issues.append(f"daily_load_too_high:day{day.get('day_index')}:{count}>{max_stops}")
            ok = False
    return ok


def _check_route_feasible(
    itinerary: dict[str, Any],
    profile: dict[str, Any],
    issues: list[str],
) -> bool:
    pace = str(profile.get("pace") or "standard")
    max_single = PACE_MAX_SINGLE_ROUTE_MIN.get(pace, PACE_MAX_SINGLE_ROUTE_MIN["standard"])
    max_daily = PACE_MAX_DAILY_ROUTE_MIN.get(pace, PACE_MAX_DAILY_ROUTE_MIN["standard"])
    state = profile.get("constraint_state") or {}
    pace_is_explicit = state.get("pace") not in (None, "")
    try:
        walking_limit = (
            float(state.get("max_walking_km_per_day"))
            if state.get("max_walking_km_per_day") is not None
            else None
        )
    except (TypeError, ValueError):
        walking_limit = None
    ok = True
    for day in itinerary.get("days", []):
        total = 0
        walking_total = 0.0
        for stop in day.get("stops", []):
            route = stop.get("route_from_previous")
            if not route:
                continue
            duration = int(route.get("duration_min") or 0)
            total += duration
            if route.get("walking_distance_km") is not None:
                walking_total += float(route.get("walking_distance_km") or 0)
            elif route.get("mode") == "walk":
                walking_total += float(route.get("distance_km") or 0)
            if duration > max_single:
                issues.append(
                    f"route_too_long:day{day.get('day_index')}:{duration}>{max_single}"
                )
                if pace_is_explicit:
                    ok = False
        if total > max_daily:
            issues.append(
                f"daily_route_too_long:day{day.get('day_index')}:{total}>{max_daily}"
            )
            if pace_is_explicit:
                ok = False
        if walking_limit is not None and walking_total > walking_limit:
            issues.append(
                f"walking_distance_exceeded:day{day.get('day_index')}:"
                f"{walking_total:.1f}>{walking_limit:g}"
            )
            ok = False
    return ok


def _check_interest_coverage(
    itinerary: dict[str, Any],
    interests: list[str],
    issues: list[str],
    *,
    supporting_artifacts: dict[str, Any] | None = None,
) -> bool:
    normalized = [item for item in interests if item]
    if not normalized:
        return True
    tags = set()
    categories = set()
    for stop in _iter_stops(itinerary):
        poi = stop.get("poi") or {}
        categories.add(poi.get("category"))
        tags.update(poi.get("tags") or [])
    supporting_artifacts = supporting_artifacts or {}
    restaurants = (supporting_artifacts.get("restaurants") or {}).get("restaurants") or []
    if restaurants:
        categories.add("food")
        tags.add("food")
    ok = True
    for interest in normalized:
        expected_categories = INTEREST_TO_CATEGORIES.get(interest, set())
        if interest in tags or categories.intersection(expected_categories):
            continue
        issues.append(f"interest_not_covered:{interest}")
        ok = False
    return ok


def _iter_stops(itinerary: dict[str, Any]):
    for day in itinerary.get("days", []):
        for stop in day.get("stops", []):
            yield stop


def _start_hour(start_time: Any) -> int | None:
    if not isinstance(start_time, str) or ":" not in start_time:
        return None
    try:
        return int(start_time.split(":", 1)[0])
    except ValueError:
        return None


def _valid_coordinate(lat: Any, lng: Any) -> bool:
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return False
    return -90 <= lat_f <= 90 and -180 <= lng_f <= 180
