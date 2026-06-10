from __future__ import annotations

from collections import Counter

from travel_agent.schemas import CriticIssue, CriticResult, Itinerary, TravelProfile


INTEREST_TO_CATEGORIES = {
    "food": {"food"},
    "history": {"culture", "museum"},
    "culture": {"culture", "museum"},
    "museum": {"museum"},
    "nature": {"scenic"},
}


def critique_itinerary(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> CriticResult:
    issues: list[CriticIssue] = []
    issues.extend(_check_required_fields(profile))
    issues.extend(_check_day_load(itinerary, profile))
    issues.extend(_check_route_feasibility(itinerary, profile))
    issues.extend(_check_interest_coverage(itinerary, profile))
    issues.extend(_check_must_visit(itinerary, profile))
    issues.extend(_check_avoid_terms(itinerary, profile))
    issues.extend(_check_diversity(itinerary))

    has_error = any(issue.severity == "error" for issue in issues)
    return CriticResult(passed=not has_error and not issues, issues=issues)


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
                )
            )
    return issues


def _check_interest_coverage(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    if not profile.interests:
        return []

    categories = _itinerary_categories(itinerary)
    tags = _itinerary_tags(itinerary)
    issues = []
    for interest in profile.interests:
        expected_categories = INTEREST_TO_CATEGORIES.get(interest)
        covered_by_category = bool(expected_categories and categories & expected_categories)
        covered_by_tag = interest in tags
        if not covered_by_category and not covered_by_tag:
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
    poi_names = _itinerary_poi_names(itinerary)
    missing = [
        name
        for name in profile.must_visit
        if not any(name in poi_name for poi_name in poi_names)
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


def _check_avoid_terms(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> list[CriticIssue]:
    if not profile.avoid:
        return []
    poi_names = _itinerary_poi_names(itinerary)
    matched = [
        term
        for term in profile.avoid
        if any(term in poi_name for poi_name in poi_names)
    ]
    if not matched:
        return []
    return [
        CriticIssue(
            code="avoid_term_included",
            message=f"行程包含用户希望避开的内容：{', '.join(matched)}。",
            severity="error",
        )
    ]


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
