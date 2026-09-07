from __future__ import annotations

from collections import Counter
from datetime import date
import math
import re
from typing import Any

from travel_agent.poi_evidence import (
    is_verified_plannable_poi,
    poi_avoid_match,
    poi_covers_requirement,
)
from travel_agent.route_evidence import (
    SHORT_WALK_DISTANCE_KM,
    itinerary_route_violations,
    route_can_prove_hard_feasibility,
)
from travel_agent.schemas import CriticIssue, CriticResult, Itinerary, POI, TravelProfile


INTEREST_TO_CATEGORIES = {
    "food": {"food"},
    "history": {"culture", "museum"},
    "culture": {"culture", "museum"},
    "museum": {"museum"},
    "nature": {"scenic"},
    "citywalk": {"scenic", "culture", "shopping"},
    "shopping": {"shopping"},
    "nightlife": {"food", "scenic"},
    "family": {"family", "museum", "scenic"},
    "couple": {"scenic", "culture", "food"},
    "历史景点": {"culture", "museum"},
}

INTEREST_ALIASES = {
    "海边": {"海边", "海湾", "海滨", "沙滩", "海滩", "环岛", "seaside", "beach"},
    "咖啡店": {"咖啡", "coffee", "cafe", "café"},
    "园林": {"园林", "花园", "庭园", "庭院", "garden"},
    "历史景点": {"历史", "古迹", "遗址", "故居", "博物馆", "history", "historic"},
}


def requested_interests(profile: TravelProfile) -> list[str]:
    """Return both compact and lossless structured interest requirements."""
    structured = (profile.constraint_state or {}).get("interests") or []
    if isinstance(structured, str):
        structured = [structured]
    return list(dict.fromkeys(
        str(item).strip()
        for item in [*profile.interests, *structured]
        if str(item).strip()
    ))


def poi_matches_interest(poi: Any, interest: str) -> bool:
    """Match a POI to an interest without treating broad categories as specifics."""
    term = str(interest or "").strip()
    category = str(getattr(poi, "category", "") or "")
    tags = {str(item).lower() for item in (getattr(poi, "tags", None) or [])}
    text = " ".join(
        [str(getattr(poi, "name", "") or ""), *tags]
    ).lower()
    aliases = INTEREST_ALIASES.get(term)
    if aliases is not None:
        return any(alias.lower() in text for alias in aliases)
    expected_categories = INTEREST_TO_CATEGORIES.get(term, set())
    return category in expected_categories or term.lower() in tags or term.lower() in text


def critique_itinerary(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> CriticResult:
    issues: list[CriticIssue] = []
    issues.extend(_check_required_fields(profile))
    issues.extend(_check_temporal_invariants(itinerary, profile))
    issues.extend(_check_day_load(itinerary, profile))
    issues.extend(_check_schedule_completeness(itinerary, profile))
    issues.extend(_check_route_feasibility(itinerary, profile))
    issues.extend(_check_poi_and_route_identity(itinerary, profile))
    issues.extend(_check_interest_coverage(itinerary, profile))
    issues.extend(_check_must_visit(itinerary, profile))
    issues.extend(_check_avoid_terms(itinerary, profile))
    issues.extend(_check_dietary_constraints(itinerary, profile))
    issues.extend(_check_structured_constraints(itinerary, profile))
    issues.extend(_check_diversity(itinerary))
    issues.extend(_check_duplicate_food_brands(itinerary))

    has_error = any(issue.severity == "error" for issue in issues)
    # Warnings are delivery annotations, not hard-gate failures.  Treating any
    # warning as a failed critic inverted the intended severity contract and
    # caused harmless pacing advice to block otherwise valid plans.
    return CriticResult(passed=not has_error, issues=issues)


def _check_temporal_invariants(itinerary: Itinerary, profile: TravelProfile) -> list[CriticIssue]:
    """The requested trip span is immutable; sparse evidence may yield empty days."""
    state = profile.constraint_state or {}
    expected = profile.days
    start = profile.start_date or state.get("date_start")
    end = state.get("date_end")
    duration_value = state.get("duration_days") or profile.days
    date_days = None
    if start and end:
        try:
            date_days = (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days + 1
            expected = date_days
        except ValueError:
            pass
    try:
        duration_days = int(duration_value) if duration_value is not None else None
    except (TypeError, ValueError):
        duration_days = None
    issues: list[CriticIssue] = []
    if date_days is not None and duration_days is not None and date_days != duration_days:
        issues.append(CriticIssue("date_duration_mismatch", f"日期区间为{date_days}天，但 duration_days 为{duration_days}天。", "error"))
    if expected is not None and len(itinerary.days) != expected:
        issues.append(CriticIssue("trip_day_count_mismatch", f"行程应为{expected}天，实际为{len(itinerary.days)}天。", "error"))
    indexes = [day.day_index for day in itinerary.days]
    if indexes != list(range(1, len(itinerary.days) + 1)):
        issues.append(CriticIssue("day_index_not_contiguous", "day_index 必须连续且无重复。", "error"))
    return issues


def _check_required_fields(profile: TravelProfile) -> list[CriticIssue]:
    missing = profile.missing_required_fields()
    if not missing:
        return []
    return [
        CriticIssue(
            code="missing_required_fields",
            message=f"缺少必要出行信息：{', '.join(missing)}",
            severity="error",
        )
    ]


def _check_day_load(itinerary: Itinerary, profile: TravelProfile) -> list[CriticIssue]:
    max_stops = {"relaxed": 2, "standard": 3, "intensive": 4}[profile.pace]
    issues = []
    for day in itinerary.days:
        if len(day.stops) > max_stops:
            issues.append(
                CriticIssue(
                    code="day_too_full",
                    message=(
                        f"第{day.day_index}天安排了{len(day.stops)}个地点，"
                        f"超过{profile.pace}节奏建议上限{max_stops}个。"
                    ),
                )
            )
        total_duration = sum(stop.duration_min for stop in day.stops)
        if profile.pace == "relaxed" and total_duration > 360:
            issues.append(
                CriticIssue(
                    code="duration_too_long",
                    message=f"第{day.day_index}天游览时长约{total_duration}分钟，不符合轻松节奏。",
                )
            )
    return issues


def _check_schedule_completeness(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    """Flag obviously partial day plans that previously passed the critic."""
    issues: list[CriticIssue] = []
    state = profile.constraint_state or {}
    # Dietary restrictions constrain meals that the itinerary actually
    # recommends.  Requiring a concrete restaurant on every day is a separate
    # product request, represented by the explicit food interest.
    meal_is_explicit = "food" in requested_interests(profile)
    for day in itinerary.days:
        fixed_event_day = any(
            isinstance(event, dict) and (
                str(event.get("day") or "") == str(day.day_index)
                or (
                    event.get("date") and profile.start_date
                    and (date.fromisoformat(str(event["date"])) - date.fromisoformat(str(profile.start_date))).days + 1 == day.day_index
                )
            )
            for event in (state.get("fixed_events") or [])
        )
        if len(day.stops) < 2:
            issues.append(
                CriticIssue(
                    code="day_too_sparse",
                    message=(f"第{day.day_index}天仅有固定活动/自由时段。" if fixed_event_day else f"第{day.day_index}天证据不足，保留自由活动时段。"),
                )
            )
        activity_count = sum(stop.poi.category != "food" for stop in day.stops)
        if profile.days > 1 and activity_count < 2 and not fixed_event_day:
            issues.append(
                CriticIssue(
                    code="daily_activity_sparse",
                    message=(
                        f"第{day.day_index}天只有{activity_count}个非餐饮活动，"
                        "多日行程的下午时段明显不完整。"
                    ),
                )
            )
        if (
            state.get("candidate_only") is True
            and activity_count >= 3
            and not any(stop.poi.category == "food" for stop in day.stops)
            and not _has_lunch_window(day)
        ):
            issues.append(
                CriticIssue(
                    code="meal_break_missing",
                    message=(
                        f"第{day.day_index}天候选全部排入后没有可执行的午餐窗口；"
                        "应按用户授权删减候选，而不是输出时间冲突。"
                    ),
                    severity="error",
                )
            )
        if not any(stop.poi.category == "food" for stop in day.stops) and meal_is_explicit:
            issues.append(
                CriticIssue(
                    code="daily_meal_missing",
                    message=f"第{day.day_index}天没有有证据支持的用餐安排。",
                    severity="error",
                )
            )
    return issues


def _has_lunch_window(day: ItineraryDay) -> bool:
    """Whether a one-hour lunch plus small transfer margins fits the schedule."""
    occupied: list[tuple[int, int]] = []
    for stop in day.stops:
        start = _clock_minutes(stop.start_time)
        inbound = max(0, int(stop.route_from_previous.duration_min)) if stop.route_from_previous else 0
        occupied.append((start - inbound, start + int(stop.duration_min)))
    for candidate in range(11 * 60 + 30, 14 * 60 + 1, 15):
        end = candidate + 60
        if not any(candidate < occupied_end + 15 and end > occupied_start - 15 for occupied_start, occupied_end in occupied):
            return True
    return False


def _check_route_feasibility(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    max_single_route_min = {
        "relaxed": 45,
        "standard": 60,
        "intensive": 75,
    }[profile.pace]
    max_daily_route_min = {
        "relaxed": 75,
        "standard": 110,
        "intensive": 150,
    }[profile.pace]
    issues = []
    for day in itinerary.days:
        routes = [
            stop.route_from_previous
            for stop in day.stops
            if stop.route_from_previous is not None
        ]
        for route in routes:
            if route.duration_min > max_single_route_min:
                issues.append(
                    CriticIssue(
                        code="route_too_long",
                        message=(
                            f"第{day.day_index}天存在单段通勤约{route.duration_min}分钟，"
                            f"超过{profile.pace}节奏建议上限{max_single_route_min}分钟。"
                        ),
                        severity=(
                            "error"
                            if route.mode == "walk"
                            and route.duration_min > max_single_route_min * 2
                            else "warning"
                        ),
                    )
                )
        for index in range(1, len(day.stops) - 1):
            inbound = day.stops[index].route_from_previous
            outbound = day.stops[index + 1].route_from_previous
            if (
                inbound is not None
                and outbound is not None
                and inbound.distance_km >= 20.0
                and outbound.distance_km >= 20.0
                and _poi_distance_km(day.stops[index - 1].poi, day.stops[index + 1].poi) <= 8.0
            ):
                issues.append(
                    CriticIssue(
                        code="route_backtracking",
                        message=(
                            f"第{day.day_index}天经由 `{day.stops[index].poi.name}` "
                            "形成两段跨区域往返，应替换或删减中间可选地点。"
                        ),
                        severity="warning",
                    )
                )
        total_route_min = sum(route.duration_min for route in routes)
        if total_route_min > max_daily_route_min:
            issues.append(
                CriticIssue(
                    code="daily_route_too_long",
                    message=(
                        f"第{day.day_index}天通勤合计约{total_route_min}分钟，"
                        f"超过{profile.pace}节奏建议上限{max_daily_route_min}分钟。"
                    ),
                    severity=(
                        "error"
                        if any(route.mode == "walk" for route in routes)
                        and total_route_min > max_daily_route_min * 2
                        else "warning"
                    ),
                )
            )
    return issues


def _poi_distance_km(left: POI, right: POI) -> float:
    radius = 6371.0
    lat1, lat2 = math.radians(left.lat), math.radians(right.lat)
    dlat = lat2 - lat1
    dlng = math.radians(right.lng - left.lng)
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    )
    return radius * 2 * math.asin(min(1.0, math.sqrt(value)))


def _check_interest_coverage(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    interests = requested_interests(profile)
    if not interests:
        return []
    issues = []
    pois = [stop.poi for day in itinerary.days for stop in day.stops]
    for interest in interests:
        if not any(poi_matches_interest(poi, interest) for poi in pois):
            issues.append(
                CriticIssue(
                    code="interest_not_covered",
                    message=f"用户偏好 `{interest}` 没有在当前行程中得到明显覆盖。",
                )
            )
    return issues


def _check_must_visit(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    if not profile.must_visit:
        return []
    missing = [
        term
        for term in profile.must_visit
        if not any(
            poi_matches_must_visit(stop.poi, term)
            for day in itinerary.days
            for stop in day.stops
        )
    ]
    if not missing:
        return []
    return [
        CriticIssue(
            code="must_visit_missing",
            message=f"行程没有包含必去地点：{', '.join(missing)}。",
            severity="error",
        )
    ]


def poi_matches_must_visit(poi: Any, term: str) -> bool:
    """Use canonical identity, provider aliases, or explicit parent coverage only."""
    return isinstance(poi, POI) and poi_covers_requirement(poi, term)


def _check_poi_and_route_identity(
    itinerary: Itinerary, profile: TravelProfile
) -> list[CriticIssue]:
    issues: list[CriticIssue] = []
    for day in itinerary.days:
        for stop in day.stops:
            if stop.poi.category != "food" and not is_verified_plannable_poi(stop.poi):
                reason = f"（{stop.poi.verification_reason}）" if stop.poi.verification_reason else ""
                issues.append(CriticIssue(
                    code="poi_not_verified",
                    message=(
                        f"第{day.day_index}天地点 {stop.poi.name} 未通过实体核验："
                        f"{stop.poi.verification_status}{reason}"
                    ),
                    severity="error",
                ))
    for code, message in itinerary_route_violations(itinerary):
        if code == "route_evidence_missing":
            continue
        issues.append(CriticIssue(
            code=code,
            message=message,
            severity="error",
        ))

    state = profile.constraint_state or {}
    if state.get("accessibility_priority") or state.get("wheelchair_user"):
        if any(
            not route_can_prove_hard_feasibility(stop.route_from_previous, "accessibility")
            for day in itinerary.days
            for stop in day.stops[1:]
        ):
            issues.append(CriticIssue(
                code="hard_route_evidence_insufficient",
                message="无障碍硬约束依赖的路线只有低可信估算或无证据，无法证明可达。",
                severity="error",
            ))

    events = [event for event in (state.get("fixed_events") or []) if isinstance(event, dict)]
    for event in events:
        location = str(event.get("location") or "").strip()
        if not location or location in {"自由活动", "休息", "自由时间"}:
            continue
        try:
            event_day = int(event.get("day") or 0)
        except (TypeError, ValueError):
            event_day = 0
        day = next((item for item in itinerary.days if not event_day or item.day_index == event_day), None)
        if day is None:
            continue
        index = next(
            (
                i for i, stop in enumerate(day.stops)
                if poi_covers_requirement(stop.poi, location)
                or location == stop.poi.name
                or location in str(stop.poi.address or "")
            ),
            None,
        )
        if index is not None and index > 0 and not route_can_prove_hard_feasibility(
            day.stops[index].route_from_previous, "fixed_appointment"
        ):
            issues.append(CriticIssue(
                code="hard_route_evidence_insufficient",
                message=f"固定预约 {location} 的前往路线只有低可信估算，无法证明准时到达。",
                severity="error",
            ))

    for day in itinerary.days:
        for stop in day.stops[1:]:
            route = stop.route_from_previous
            if (
                route is not None
                and route.distance_km <= SHORT_WALK_DISTANCE_KM
                and route.mode != "walk"
                and route.evidence_status != "provider_verified"
            ):
                issues.append(CriticIssue(
                    code="short_route_mode_warning",
                    message=f"第{day.day_index}天 {route.distance_km:g} 公里短距离路段未标为步行，建议复核交通方式。",
                    severity="warning",
                ))
    return issues


def _check_avoid_terms(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    matched = list(
        dict.fromkeys(
            match.term
            for day in itinerary.days
            for stop in day.stops
            if (match := poi_avoid_match(stop.poi, profile)) is not None
        )
    )
    if not matched:
        return []
    return [
        CriticIssue(
            code="avoid_term_included",
            message=f"行程包含用户希望避开的内容：{', '.join(matched)}。",
            severity="error",
        )
    ]


def poi_complies_with_dietary(poi: Any, profile: TravelProfile) -> bool:
    """Return whether an evidenced food POI is compatible with hard dietary rules."""
    if str(getattr(poi, "category", "")) != "food":
        return True
    state = profile.constraint_state or {}
    raw = state.get("dietary") or []
    rules = [str(raw)] if isinstance(raw, str) else [str(item) for item in raw]
    if not rules:
        return True
    name = str(getattr(poi, "name", "") or "")
    tags = {str(item).lower() for item in (getattr(poi, "tags", None) or [])}
    searchable = name.lower() + " " + " ".join(tags)
    if any("仅清真" in rule for rule in rules):
        if not any(marker in searchable for marker in ("清真", "halal", "muslim")):
            return False
    if any("不吃海鲜" in rule for rule in rules):
        if any(marker in searchable for marker in ("海鲜", "水产", "seafood", "蟹", "虾", "生蚝")):
            return False
        if not any(
            marker in searchable
            for marker in ("无海鲜", "不含海鲜", "不吃海鲜", "素食", "纯素", "vegetarian", "vegan")
        ):
            return False
    if any("不吃辣" in rule or "不太辣" in rule for rule in rules):
        if "辣" in searchable or "spicy" in searchable:
            return False
        if not any(
            marker in searchable
            for marker in ("不辣", "少辣", "微辣", "清淡", "mild", "non-spicy")
        ):
            return False
    return True


def _check_dietary_constraints(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    violations = [
        stop.poi.name
        for day in itinerary.days
        for stop in day.stops
        if stop.poi.category == "food" and not poi_complies_with_dietary(stop.poi, profile)
    ]
    if not violations:
        return []
    return [CriticIssue(
        code="dietary_constraint_violated",
        message=f"行程包含缺乏饮食约束合规证据的餐厅：{', '.join(dict.fromkeys(violations))}。",
        severity="error",
    )]


def _check_structured_constraints(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    """Validate hard schedule/transport constraints kept in constraint_state."""
    state = profile.constraint_state or {}
    issues: list[CriticIssue] = []
    raw_avoid = state.get("avoid") or profile.avoid or []
    avoid_values = [raw_avoid] if isinstance(raw_avoid, str) else list(raw_avoid)
    terrain_access_required = any(
        marker in str(value).casefold()
        for value in avoid_values
        for marker in (
            "爬坡", "上坡", "坡道", "楼梯", "台阶",
            "hill", "slope", "stairs", "steps",
        )
    )
    if state.get("accessibility_priority") or state.get("wheelchair_user"):
        evidenced = any(
            any(
                marker in " ".join(
                    [stop.poi.name, stop.poi.address or "", *stop.poi.tags]
                ).lower()
                for marker in ("无障碍", "轮椅", "barrier_free", "wheelchair", "accessible")
            )
            for day in itinerary.days
            for stop in day.stops
        )
        if not evidenced:
            issues.append(
                CriticIssue(
                    code="accessibility_evidence_missing",
                    message=(
                        "当前工具证据不足以确认无台阶/低坡度可达；请在出发前向场馆和交通运营方"
                        "核实无障碍入口、电梯及路面情况，无法确认时替换候选。"
                    ),
                    severity="error",
                )
            )
    if terrain_access_required:
        routes = [
            stop.route_from_previous
            for day in itinerary.days
            for stop in day.stops[1:]
        ]
        unsafe_routes = [
            route
            for route in routes
            if route is None
            or route.mode == "walk"
            or not route_can_prove_hard_feasibility(route, "accessibility")
        ]
        if not routes or unsafe_routes:
            issues.append(
                CriticIssue(
                    code="accessibility_evidence_missing",
                    message=(
                        "避免连续爬坡或长楼梯的约束需要有证据的非步行接驳；"
                        "当前至少一段路线仍无法证明。"
                    ),
                    severity="error",
                )
            )
    if state.get("self_driving_allowed") is False:
        driving = [
            route
            for day in itinerary.days
            for stop in day.stops
            if (route := stop.route_from_previous) is not None and route.mode == "drive"
        ]
        if driving:
            issues.append(
                CriticIssue(
                    code="self_driving_forbidden",
                    message="行程包含自驾路线，但用户明确要求不自驾。",
                    severity="error",
                )
            )

    conditional_avoid = state.get("conditional_avoid_window") or {}
    if (
        isinstance(conditional_avoid, dict)
        and conditional_avoid.get("avoid") == "long_outdoor_activity"
    ):
        try:
            avoid_start = _clock_minutes(str(conditional_avoid.get("start")))
            avoid_end = _clock_minutes(str(conditional_avoid.get("end")))
        except (TypeError, ValueError):
            avoid_start = avoid_end = 0
        if avoid_end > avoid_start:
            conflicts = [
                f"第{day.day_index}天 {stop.poi.name}"
                for day in itinerary.days
                for stop in day.stops
                if not stop.poi.indoor
                and stop.duration_min >= 90
                and _clock_minutes(stop.start_time) < avoid_end
                and _clock_minutes(stop.start_time) + stop.duration_min > avoid_start
            ]
            if conflicts:
                issues.append(
                    CriticIssue(
                        code="conditional_avoid_window_conflict",
                        message=(
                            f"条件性避让时段 {conditional_avoid.get('start')}-"
                            f"{conditional_avoid.get('end')} 内包含长时间户外活动："
                            f"{', '.join(conflicts)}。"
                        ),
                        severity="error",
                    )
                )

    walking_limit = state.get("max_walking_km_per_day")
    try:
        walking_limit_value = float(walking_limit) if walking_limit is not None else None
    except (TypeError, ValueError):
        walking_limit_value = None
    if walking_limit_value is not None:
        for day in itinerary.days:
            walking_km = sum(
                (
                    route.walking_distance_km
                    if route.walking_distance_km is not None
                    else (route.distance_km if route.mode == "walk" else 0.0)
                )
                for stop in day.stops
                if (route := stop.route_from_previous) is not None
            )
            if walking_km > walking_limit_value:
                issues.append(
                    CriticIssue(
                        code="walking_distance_exceeded",
                        message=(
                            f"第{day.day_index}天已知路线步行约{walking_km:.1f}公里，"
                            f"超过每日上限{walking_limit_value:g}公里。"
                        ),
                        severity="error",
                    )
                )

    deadline_text = state.get("activity_end_deadline") or state.get("return_deadline")
    try:
        deadline = _clock_minutes(str(deadline_text)) if deadline_text else None
    except (TypeError, ValueError):
        deadline = None
    if deadline is not None:
        late_days = [
            day.day_index
            for day in itinerary.days
            if any(
                _clock_minutes(stop.start_time) + stop.duration_min > deadline
                for stop in day.stops
            )
        ]
        if late_days:
            issues.append(
                CriticIssue(
                    code="end_deadline_exceeded",
                    message=f"第 {', '.join(map(str, late_days))} 天活动超过截止时间 {deadline_text}。",
                    severity="error",
                )
            )
        return_location = str(state.get("return_location") or "").strip()
        destination = str(profile.destination or "").strip()
        if state.get("return_deadline") and return_location and destination not in return_location:
            issues.append(
                CriticIssue(
                    code="return_leg_requires_verification",
                    message=(
                        f"返程目标为 {return_location}、截止 {state.get('return_deadline')}；"
                        "城际班次与进站余量未获实时库存证据，请预留返程缓冲并在购票平台复核。"
                    ),
                    severity="warning",
                )
            )

    raw_events = state.get("fixed_events") or []
    for event in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(event, dict):
            continue
        try:
            raw_day = event.get("day")
            if raw_day is not None:
                day_index = int(raw_day)
            elif event.get("date") and profile.start_date:
                day_index = (
                    date.fromisoformat(str(event.get("date")))
                    - date.fromisoformat(str(profile.start_date))
                ).days + 1
            else:
                continue
            expected_start = str(event.get("start"))
            expected_end = _clock_minutes(str(event.get("end")))
        except (TypeError, ValueError):
            continue
        location = str(event.get("location") or "").strip()
        unspecified_locations = {
            str(item).strip()
            for item in state.get("user_owned_unspecified_fixed_event_locations") or []
            if str(item).strip()
        }
        user_owned_unspecified_meal = location in unspecified_locations
        if location in {"自由活动", "休息", "自由时间"} or user_owned_unspecified_meal:
            # User-authored reserved blocks are represented by fixed_event_plan,
            # not fabricated POIs in the map itinerary.
            continue
        day = next((item for item in itinerary.days if item.day_index == day_index), None)
        matched = next(
            (
                stop
                for stop in (day.stops if day else [])
                if location
                and (
                    expected_start < "17:00" or stop.poi.category == "food"
                )
                and (
                    location in stop.poi.name
                    or stop.poi.name in location
                    or location in str(stop.poi.address or "")
                )
            ),
            None,
        )
        if (
            matched is None
            or matched.start_time != expected_start
            or _clock_minutes(matched.start_time) + matched.duration_min != expected_end
        ):
            issues.append(
                CriticIssue(
                    code="fixed_event_missing_or_conflicting",
                    message=f"第{day_index}天固定活动 {location} 未按 {expected_start} 安排。",
                    severity="error",
                )
            )
    return issues


def _clock_minutes(value: str) -> int:
    from travel_agent.constraint_events import clock_minutes

    return clock_minutes(value)


def _check_diversity(itinerary: Itinerary) -> list[CriticIssue]:
    categories = [
        stop.poi.category
        for day in itinerary.days
        for stop in day.stops
    ]
    if len(categories) < 3:
        return []
    counts = Counter(categories)
    top_category, top_count = counts.most_common(1)[0]
    if top_count / len(categories) <= 0.8:
        return []
    return [
        CriticIssue(
            code="low_category_diversity",
            message=f"行程中 `{top_category}` 类型占比过高，体验可能偏单一。",
        )
    ]


def _check_duplicate_food_brands(itinerary: Itinerary) -> list[CriticIssue]:
    def brand(name: str) -> str:
        value = re.sub(r"[（(][^）)]*[）)]", "", name)
        value = re.sub(r"(?:旗舰店|总店|分店|店)$", "", value)
        return re.sub(r"[\s·._-]", "", value)

    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for day in itinerary.days:
        for stop in day.stops:
            if stop.poi.category != "food":
                continue
            key = brand(stop.poi.name)
            if key and key in seen:
                duplicates.append(f"{seen[key]} / {stop.poi.name}")
            elif key:
                seen[key] = stop.poi.name
    if not duplicates:
        return []
    return [
        CriticIssue(
            code="duplicate_food_brand",
            message=f"行程重复安排同一餐饮品牌：{'；'.join(duplicates)}。",
        )
    ]


def _itinerary_categories(itinerary: Itinerary) -> set[str]:
    return {
        stop.poi.category
        for day in itinerary.days
        for stop in day.stops
    }


def _itinerary_tags(itinerary: Itinerary) -> set[str]:
    return {
        tag
        for day in itinerary.days
        for stop in day.stops
        for tag in stop.poi.tags
    }


def _itinerary_poi_names(itinerary: Itinerary) -> list[str]:
    return [
        stop.poi.name
        for day in itinerary.days
        for stop in day.stops
    ]
