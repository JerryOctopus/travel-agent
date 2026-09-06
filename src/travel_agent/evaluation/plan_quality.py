from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


PACE_MAX_STOPS = {"relaxed": 2, "standard": 3, "intensive": 4}
PACE_MAX_SINGLE_ROUTE_MIN = {"relaxed": 45, "standard": 60, "intensive": 75}
PACE_MAX_DAILY_ROUTE_MIN = {"relaxed": 75, "standard": 110, "intensive": 150}
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
class PlanQualityRuleResult:
    hard_feasibility_pass: bool
    rule_quality_pass: bool
    total_stops: int
    grounded_stops: int
    grounded_poi_rate: float | None
    day_structure_valid: bool
    city_and_coordinates_valid: bool
    duplicate_poi_free: bool
    schedule_conflict_free: bool | None
    transfer_feasible: bool | None
    route_transition_count: int
    route_covered_count: int
    route_coverage_rate: float | None
    daily_load_valid: bool
    transport_mode_consistent: bool | None
    must_visit_coverage_rate: float | None
    avoid_compliance: bool | None
    budget_consistency: bool | None
    hotel_area_match: bool | None
    food_preference_match: bool | None
    interest_coverage_valid: bool | None
    weather_adaptation_valid: bool | None
    opening_hours_known_count: int
    opening_hours_valid_count: int
    opening_hours_coverage: float | None
    opening_hours_valid_rate: float | None
    unknown_checks: list[str] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_plan_quality(
    case: dict[str, Any],
    final_profile: dict[str, Any],
    final_artifacts: dict[str, Any],
) -> PlanQualityRuleResult | None:
    artifact = final_artifacts.get("itinerary") or {}
    itinerary = artifact.get("itinerary") if isinstance(artifact, dict) else None
    if not isinstance(itinerary, dict):
        return None

    issues: list[dict[str, Any]] = []
    unknown: list[str] = []
    days = itinerary.get("days") if isinstance(itinerary.get("days"), list) else []
    stops = [stop for day in days for stop in _as_list(day.get("stops"))]
    pois = [stop.get("poi") or {} for stop in stops]
    hard = dict(case.get("hard_constraints") or {})
    soft = dict(case.get("soft_preferences") or {})

    expected_city = hard.get("destination") or case.get("expected_city") or final_profile.get("destination")
    expected_days = hard.get("days") or case.get("expected_days") or final_profile.get("days")
    day_structure_valid = _day_structure_valid(days, expected_days, issues)
    itinerary_city_valid = not expected_city or _normalize_city(itinerary.get("city")) == _normalize_city(expected_city)
    if not itinerary_city_valid:
        _issue(
            issues,
            "itinerary_city_mismatch",
            "error",
            f"actual={itinerary.get('city')},expected={expected_city}",
        )
    fixed_events = (
        hard.get("fixed_events")
        or (final_profile.get("constraint_state") or {}).get("fixed_events")
        or []
    )
    cross_city_fixed_locations = {
        _normalize(event.get("location"))
        for event in fixed_events
        if isinstance(event, dict) and event.get("location")
    }
    city_coordinates_valid = itinerary_city_valid and _city_coordinates_valid(
        pois,
        expected_city,
        issues,
        allowed_cross_city_locations=cross_city_fixed_locations,
    )
    duplicate_poi_free = _duplicates_valid(pois, issues)

    source_entities = _source_entities(final_artifacts)
    grounded_count = sum(_is_grounded(poi, source_entities) for poi in pois)
    grounded_rate: float | None
    if not stops:
        grounded_rate = 0.0
    elif not source_entities:
        grounded_rate = None
        unknown.append("poi_grounding")
    else:
        grounded_rate = grounded_count / len(stops)
        if grounded_rate < 1:
            _issue(issues, "ungrounded_poi", "error", f"{len(stops) - grounded_count}个POI不在工具事实快照中")

    schedule_ok, transfer_ok, transitions, covered = _schedule_and_routes(days, issues)
    route_coverage = covered / transitions if transitions else None
    pace = str(soft.get("pace") or final_profile.get("pace") or "standard")
    daily_load_valid = _daily_load_valid(days, pace, issues)
    route_thresholds_valid = _route_thresholds_valid(days, pace, issues)

    transport_expected = hard.get("transport_mode") or soft.get("transport_mode")
    transport_consistent = _transport_consistency(days, transport_expected, issues)
    must_visit = _as_list(hard.get("must_visit") or final_profile.get("must_visit"))
    must_visit_rate = _must_visit_coverage(pois, must_visit, issues)
    avoid = _as_list(hard.get("avoid") or final_profile.get("avoid"))
    avoid_ok = _avoid_compliance(pois, avoid, issues)
    budget_ok = _budget_consistency(final_artifacts, hard.get("budget_level"), issues, unknown)
    hotel_ok = _hotel_area_match(final_artifacts, soft.get("hotel_area"), issues, unknown)
    food_ok = _food_preference_match(
        final_artifacts, soft.get("food_preference"), issues, unknown
    )
    interests = _as_list(soft.get("interests") or case.get("required_interests"))
    interest_ok = _interest_coverage(pois, interests, issues)
    weather_ok = _weather_adaptation(
        pois, final_artifacts.get("weather"), soft.get("weather_adaptation"), issues, unknown
    )
    known_opening, valid_opening = _opening_hours(pois, stops, issues, unknown)
    opening_coverage = known_opening / len(stops) if stops else None
    opening_valid_rate = valid_opening / known_opening if known_opening else None

    hard_checks: list[bool] = [
        day_structure_valid,
        city_coordinates_valid,
        duplicate_poi_free,
        daily_load_valid,
    ]
    if grounded_rate is not None:
        hard_checks.append(grounded_rate == 1.0)
    if schedule_ok is not None:
        hard_checks.append(schedule_ok)
    if transitions:
        hard_checks.extend([covered == transitions, bool(transfer_ok)])
    for optional in (must_visit_rate, avoid_ok, budget_ok):
        if optional is not None:
            hard_checks.append(optional is True or optional == 1.0)
    if hard.get("transport_mode") and transport_consistent is not None:
        hard_checks.append(transport_consistent)
    if known_opening:
        hard_checks.append(valid_opening == known_opening)
    hard_pass = all(hard_checks)

    quality_checks = [hard_pass, route_thresholds_valid]
    for optional in (hotel_ok, food_ok, interest_ok, weather_ok):
        if optional is not None:
            quality_checks.append(optional)
    rule_pass = all(quality_checks)
    return PlanQualityRuleResult(
        hard_feasibility_pass=hard_pass,
        rule_quality_pass=rule_pass,
        total_stops=len(stops),
        grounded_stops=grounded_count,
        grounded_poi_rate=grounded_rate,
        day_structure_valid=day_structure_valid,
        city_and_coordinates_valid=city_coordinates_valid,
        duplicate_poi_free=duplicate_poi_free,
        schedule_conflict_free=schedule_ok,
        transfer_feasible=transfer_ok,
        route_transition_count=transitions,
        route_covered_count=covered,
        route_coverage_rate=route_coverage,
        daily_load_valid=daily_load_valid,
        transport_mode_consistent=transport_consistent,
        must_visit_coverage_rate=must_visit_rate,
        avoid_compliance=avoid_ok,
        budget_consistency=budget_ok,
        hotel_area_match=hotel_ok,
        food_preference_match=food_ok,
        interest_coverage_valid=interest_ok,
        weather_adaptation_valid=weather_ok,
        opening_hours_known_count=known_opening,
        opening_hours_valid_count=valid_opening,
        opening_hours_coverage=opening_coverage,
        opening_hours_valid_rate=opening_valid_rate,
        unknown_checks=list(dict.fromkeys(unknown)),
        issues=issues,
    )


def aggregate_rule_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    available = [result for result in results if result]
    total_stops = sum(int(item.get("total_stops") or 0) for item in available)
    grounded = sum(int(item.get("grounded_stops") or 0) for item in available)
    transitions = sum(int(item.get("route_transition_count") or 0) for item in available)
    covered = sum(int(item.get("route_covered_count") or 0) for item in available)
    opening_known = sum(int(item.get("opening_hours_known_count") or 0) for item in available)
    opening_valid = sum(int(item.get("opening_hours_valid_count") or 0) for item in available)
    unknown_counts = Counter(
        check for item in available for check in item.get("unknown_checks") or []
    )
    return {
        "evaluated_itinerary_count": len(available),
        "grounded_poi_rate": grounded / total_stops if total_stops else None,
        "schedule_conflict_free_rate": _bool_rate(available, "schedule_conflict_free"),
        "transfer_feasible_rate": _bool_rate(available, "transfer_feasible"),
        "route_coverage_rate": covered / transitions if transitions else None,
        "must_visit_coverage_rate": _mean_optional(available, "must_visit_coverage_rate"),
        "avoid_compliance_rate": _bool_rate(available, "avoid_compliance"),
        "budget_consistency_rate": _bool_rate(available, "budget_consistency"),
        "opening_hours_coverage": opening_known / total_stops if total_stops else None,
        "opening_hours_valid_rate": opening_valid / opening_known if opening_known else None,
        "rule_quality_pass_rate": _bool_rate(available, "rule_quality_pass"),
        "unknown_check_counts": dict(sorted(unknown_counts.items())),
    }


def _day_structure_valid(
    days: list[dict[str, Any]], expected_days: Any, issues: list[dict[str, Any]]
) -> bool:
    indexes = [day.get("day_index") for day in days]
    expected_indexes = list(range(1, len(days) + 1))
    ok = bool(days) and indexes == expected_indexes and all(_as_list(day.get("stops")) for day in days)
    if expected_days is not None:
        ok = ok and len(days) == int(expected_days)
    if not ok:
        _issue(issues, "invalid_day_structure", "error", "天数、日期序号或每日停靠点不完整")
    return ok


def _city_coordinates_valid(
    pois: list[dict[str, Any]],
    expected_city: Any,
    issues: list[dict[str, Any]],
    *,
    allowed_cross_city_locations: set[str] | None = None,
) -> bool:
    ok = True
    allowed = allowed_cross_city_locations or set()
    for poi in pois:
        normalized_name = _normalize(poi.get("name"))
        is_fixed_event_venue = any(
            location and (
                location in normalized_name or normalized_name in location
            )
            for location in allowed
        )
        if (
            expected_city
            and poi.get("city")
            and _normalize_city(poi.get("city")) != _normalize_city(expected_city)
            and not is_fixed_event_venue
        ):
            ok = False
            _issue(issues, "poi_city_mismatch", "error", str(poi.get("name") or "unknown"))
        try:
            lat, lng = float(poi.get("lat")), float(poi.get("lng"))
            coordinate_ok = -90 <= lat <= 90 and -180 <= lng <= 180
        except (TypeError, ValueError):
            coordinate_ok = False
        if not coordinate_ok:
            ok = False
            _issue(issues, "invalid_coordinate", "error", str(poi.get("name") or "unknown"))
    return ok and bool(pois)


def _duplicates_valid(pois: list[dict[str, Any]], issues: list[dict[str, Any]]) -> bool:
    names = [_normalize(poi.get("name")) for poi in pois if poi.get("name")]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        _issue(issues, "duplicate_poi", "warning", ",".join(duplicates))
    return not duplicates


def _source_entities(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    entities.extend(_as_list((artifacts.get("candidates") or {}).get("pois")))
    for item in _as_list((artifacts.get("ranked") or {}).get("pois")):
        entities.append(item.get("poi") or item)
    entities.extend(_as_list((artifacts.get("restaurants") or {}).get("restaurants")))
    entities.extend(_as_list((artifacts.get("hotels") or {}).get("hotels")))
    return [item for item in entities if isinstance(item, dict)]


def _is_grounded(poi: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    poi_id = str(poi.get("poi_id") or poi.get("id") or "")
    name = _normalize(poi.get("name"))
    for source in sources:
        source_id = str(source.get("poi_id") or source.get("id") or "")
        if poi_id and source_id and poi_id == source_id:
            return True
        if name and name == _normalize(source.get("name") or source.get("position")):
            return True
    return False


def _schedule_and_routes(
    days: list[dict[str, Any]], issues: list[dict[str, Any]]
) -> tuple[bool | None, bool | None, int, int]:
    schedule_ok = True
    transfer_ok = True
    saw_schedule = False
    transitions = 0
    covered = 0
    for day in days:
        day_stops = _as_list(day.get("stops"))
        previous_end: int | None = None
        for index, stop in enumerate(day_stops):
            start = _time_minutes(stop.get("start_time"))
            duration = _safe_int(stop.get("duration_min"))
            if start is None or duration is None or duration <= 0:
                schedule_ok = False
                _issue(issues, "invalid_stop_time", "error", f"day={day.get('day_index')},stop={index + 1}")
                previous_end = None
                continue
            saw_schedule = True
            if previous_end is not None and start < previous_end:
                schedule_ok = False
                _issue(issues, "schedule_overlap", "error", f"day={day.get('day_index')},stop={index + 1}")
            if index > 0:
                transitions += 1
                route = stop.get("route_from_previous")
                if isinstance(route, dict) and _safe_int(route.get("duration_min")) is not None:
                    covered += 1
                    route_minutes = max(0, int(route.get("duration_min") or 0))
                    if previous_end is not None and start < previous_end + route_minutes + 15:
                        transfer_ok = False
                        _issue(issues, "transfer_time_conflict", "error", f"day={day.get('day_index')},stop={index + 1}")
                else:
                    transfer_ok = False
                    _issue(issues, "route_missing", "error", f"day={day.get('day_index')},stop={index + 1}")
            previous_end = start + duration
    return (
        schedule_ok if saw_schedule else None,
        transfer_ok if transitions else None,
        transitions,
        covered,
    )


def _daily_load_valid(
    days: list[dict[str, Any]], pace: str, issues: list[dict[str, Any]]
) -> bool:
    maximum = PACE_MAX_STOPS.get(pace, PACE_MAX_STOPS["standard"])
    ok = all(len(_as_list(day.get("stops"))) <= maximum for day in days)
    if not ok:
        _issue(issues, "daily_load_too_high", "warning", f"pace={pace},max={maximum}")
    return ok


def _route_thresholds_valid(
    days: list[dict[str, Any]], pace: str, issues: list[dict[str, Any]]
) -> bool:
    single_max = PACE_MAX_SINGLE_ROUTE_MIN.get(pace, 60)
    daily_max = PACE_MAX_DAILY_ROUTE_MIN.get(pace, 110)
    ok = True
    for day in days:
        durations = [
            int(route.get("duration_min") or 0)
            for stop in _as_list(day.get("stops"))
            if isinstance((route := stop.get("route_from_previous")), dict)
        ]
        if any(duration > single_max for duration in durations) or sum(durations) > daily_max:
            ok = False
            _issue(issues, "route_duration_exceeded", "warning", f"day={day.get('day_index')}")
    return ok


def _transport_consistency(
    days: list[dict[str, Any]], expected: Any, issues: list[dict[str, Any]]
) -> bool | None:
    if not expected:
        return None
    modes = [
        route.get("mode")
        for day in days
        for stop in _as_list(day.get("stops"))
        if isinstance((route := stop.get("route_from_previous")), dict)
    ]
    if not modes:
        return None
    ok = all(mode == expected for mode in modes)
    if not ok:
        _issue(issues, "transport_mode_mismatch", "warning", f"expected={expected}")
    return ok


def _must_visit_coverage(
    pois: list[dict[str, Any]], expected: list[Any], issues: list[dict[str, Any]]
) -> float | None:
    if not expected:
        return None
    hits = sum(any(_poi_covers_requirement(poi, item) for poi in pois) for item in expected)
    rate = hits / len(expected)
    if rate < 1:
        _issue(issues, "must_visit_missing", "error", f"coverage={rate:.2f}")
    return rate


def _poi_covers_requirement(poi: dict[str, Any], required: Any) -> bool:
    """Use the same canonical identity contract as planning and validation."""
    try:
        from travel_agent.agent.serde import poi_from_dict
        from travel_agent.poi_evidence import poi_covers_requirement

        return poi_covers_requirement(poi_from_dict(poi), str(required))
    except (KeyError, TypeError, ValueError):
        required_name = _normalize(required)
        names = [
            _normalize(value)
            for value in (
                poi.get("name"),
                poi.get("canonical_name"),
                *(poi.get("aliases") or []),
            )
            if value
        ]
        return bool(required_name and required_name in names)


def _avoid_compliance(
    pois: list[dict[str, Any]], avoid: list[Any], issues: list[dict[str, Any]]
) -> bool | None:
    if not avoid:
        return None
    haystacks = [
        _normalize(" ".join([str(poi.get("name") or ""), *map(str, _as_list(poi.get("tags")))]))
        for poi in pois
    ]
    matched = [item for item in avoid if any(_normalize(item) in text for text in haystacks)]
    if matched:
        _issue(issues, "avoid_term_included", "error", ",".join(map(str, matched)))
    return not matched


def _budget_consistency(
    artifacts: dict[str, Any], expected: Any, issues: list[dict[str, Any]], unknown: list[str]
) -> bool | None:
    if not expected:
        return None
    budget = artifacts.get("budget")
    if not isinstance(budget, dict):
        unknown.append("budget_consistency")
        return None
    ok = budget.get("budget_level") == expected and all(
        budget.get(key) is not None for key in ("total_low", "total_high")
    )
    if not ok:
        _issue(issues, "budget_configuration_mismatch", "error", f"expected={expected}")
    return ok


def _hotel_area_match(
    artifacts: dict[str, Any], expected: Any, issues: list[dict[str, Any]], unknown: list[str]
) -> bool | None:
    if not expected:
        return None
    hotels = _as_list((artifacts.get("hotels") or {}).get("hotels"))
    if not hotels:
        unknown.append("hotel_area_match")
        return None
    ok = any(_normalize(expected) in _normalize(hotel.get("area")) for hotel in hotels)
    if not ok:
        _issue(issues, "hotel_area_mismatch", "warning", str(expected))
    return ok


def _food_preference_match(
    artifacts: dict[str, Any], expected: Any, issues: list[dict[str, Any]], unknown: list[str]
) -> bool | None:
    preferences = _as_list(expected)
    if not preferences:
        return None
    restaurants = _as_list((artifacts.get("restaurants") or {}).get("restaurants"))
    if not restaurants:
        unknown.append("food_preference_match")
        return None
    text = _normalize(str(restaurants))
    ok = all(_normalize(item) in text for item in preferences)
    if not ok:
        _issue(issues, "food_preference_missing", "warning", ",".join(map(str, preferences)))
    return ok


def _interest_coverage(
    pois: list[dict[str, Any]], interests: list[Any], issues: list[dict[str, Any]]
) -> bool | None:
    if not interests:
        return None
    categories = {str(poi.get("category") or "") for poi in pois}
    tags = {str(tag) for poi in pois for tag in _as_list(poi.get("tags"))}
    missing = []
    for interest in map(str, interests):
        expected = INTEREST_TO_CATEGORIES.get(interest, set())
        if interest not in tags and not categories.intersection(expected):
            missing.append(interest)
    if missing:
        _issue(issues, "interest_not_covered", "warning", ",".join(missing))
    return not missing


def _weather_adaptation(
    pois: list[dict[str, Any]], weather: Any, expected: Any,
    issues: list[dict[str, Any]], unknown: list[str]
) -> bool | None:
    if not expected:
        return None
    if not isinstance(weather, dict):
        unknown.append("weather_adaptation")
        return None
    condition = str(weather.get("condition") or "").lower()
    if not any(term in condition for term in ("rain", "雨", "storm")):
        return True
    if not pois:
        return False
    indoor_ratio = sum(bool(poi.get("indoor")) for poi in pois) / len(pois)
    ok = indoor_ratio >= 0.5
    if not ok:
        _issue(issues, "rain_plan_insufficient", "warning", f"indoor_ratio={indoor_ratio:.2f}")
    return ok


def _opening_hours(
    pois: list[dict[str, Any]], stops: list[dict[str, Any]],
    issues: list[dict[str, Any]], unknown: list[str]
) -> tuple[int, int]:
    known = valid = 0
    for poi, stop in zip(pois, stops):
        hours = _parse_opening_hours(poi.get("opening_hours"))
        start = _time_minutes(stop.get("start_time"))
        duration = _safe_int(stop.get("duration_min"))
        if hours is None or start is None or duration is None:
            continue
        known += 1
        if any(window_start <= start and start + duration <= window_end for window_start, window_end in hours):
            valid += 1
        else:
            _issue(issues, "outside_opening_hours", "error", str(poi.get("name") or "unknown"))
    if stops and known == 0:
        unknown.append("opening_hours")
    return known, valid


def _parse_opening_hours(value: Any) -> list[tuple[int, int]] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in {"全天", "24小时", "24h", "00:00-24:00"}:
        return [(0, 24 * 60)]
    matches = re.findall(r"(\d{1,2}):(\d{2})\s*[-~至]\s*(\d{1,2}):(\d{2})", normalized)
    if not matches:
        return None
    windows = []
    for start_h, start_m, end_h, end_m in matches:
        start, end = int(start_h) * 60 + int(start_m), int(end_h) * 60 + int(end_m)
        if 0 <= start < end <= 24 * 60:
            windows.append((start, end))
    return windows or None


def _time_minutes(value: Any) -> int | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,2}:\d{2}", value.strip()):
        return None
    hour, minute = map(int, value.split(":"))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").lower(), flags=re.UNICODE)


def _normalize_city(value: Any) -> str:
    text = _normalize(value)
    for suffix in ("特别行政区", "自治州", "地区", "盟", "市"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _issue(
    issues: list[dict[str, Any]], code: str, severity: str, evidence: str
) -> None:
    issues.append({"code": code, "severity": severity, "evidence": evidence})


def _bool_rate(items: list[dict[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in items if isinstance(item.get(key), bool)]
    return sum(values) / len(values) if values else None


def _mean_optional(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    return sum(values) / len(values) if values else None
