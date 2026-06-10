from __future__ import annotations

from dataclasses import replace

from travel_agent.critic import INTEREST_TO_CATEGORIES, critique_itinerary
from travel_agent.planning import START_TIMES
from travel_agent.schemas import CriticResult, Itinerary, ItineraryDay, ItineraryStop, ScoredPOI, TravelProfile


def revise_itinerary(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    critic_result: CriticResult,
) -> tuple[Itinerary, CriticResult, list[str]]:
    if critic_result.passed:
        return itinerary, critic_result, []

    revised = itinerary
    notes: list[str] = []
    for issue in critic_result.issues:
        if issue.code == "avoid_term_included":
            revised, changed = _remove_avoided_pois(revised, ranked_pois, profile)
            notes.extend(changed)
        elif issue.code == "must_visit_missing":
            revised, changed = _ensure_must_visit(revised, ranked_pois, profile)
            notes.extend(changed)
        elif issue.code == "interest_not_covered":
            revised, changed = _ensure_interest_coverage(revised, ranked_pois, profile)
            notes.extend(changed)

    revised = _normalize_start_times(revised)
    revised_result = critique_itinerary(revised, profile)
    return revised, revised_result, notes


def _remove_avoided_pois(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    notes: list[str] = []
    used_ids = _used_poi_ids(itinerary)
    replacement_pool = [
        item
        for item in ranked_pois
        if item.poi.poi_id not in used_ids
        and not _matches_any_term(item.poi.name, profile.avoid)
    ]

    days: list[ItineraryDay] = []
    for day in itinerary.days:
        stops: list[ItineraryStop] = []
        for stop in day.stops:
            if not _matches_any_term(stop.poi.name, profile.avoid):
                stops.append(stop)
                continue
            replacement = _pop_first(replacement_pool)
            if replacement:
                stops.append(_stop_from_scored(replacement, stop.start_time))
                notes.append(f"移除了避开项 `{stop.poi.name}`，替换为 `{replacement.poi.name}`。")
            else:
                notes.append(f"移除了避开项 `{stop.poi.name}`，但暂无合适替代地点。")
        days.append(replace(day, stops=stops))
    return replace(itinerary, days=days), notes


def _ensure_must_visit(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    notes: list[str] = []
    current_names = _itinerary_poi_names(itinerary)
    missing_terms = [
        term
        for term in profile.must_visit
        if not any(term in name for name in current_names)
    ]
    revised = itinerary
    for term in missing_terms:
        candidate = _find_candidate(
            ranked_pois,
            used_ids=_used_poi_ids(revised),
            predicate=lambda item, term=term: term in item.poi.name,
        )
        if not candidate:
            notes.append(f"未找到必去地点 `{term}` 的候选 POI。")
            continue
        revised = _replace_lowest_priority_stop(revised, candidate)
        notes.append(f"补充必去地点 `{candidate.poi.name}`。")
    return revised, notes


def _ensure_interest_coverage(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    notes: list[str] = []
    revised = itinerary
    current_categories = _itinerary_categories(revised)
    current_tags = _itinerary_tags(revised)

    for interest in profile.interests:
        expected_categories = INTEREST_TO_CATEGORIES.get(interest, set())
        covered = bool(current_categories & expected_categories) or interest in current_tags
        if covered:
            continue
        candidate = _find_candidate(
            ranked_pois,
            used_ids=_used_poi_ids(revised),
            predicate=lambda item, interest=interest, expected_categories=expected_categories: (
                item.poi.category in expected_categories or interest in item.poi.tags
            ),
        )
        if not candidate:
            notes.append(f"未找到可覆盖 `{interest}` 偏好的候选 POI。")
            continue
        revised = _replace_lowest_priority_stop(revised, candidate)
        current_categories = _itinerary_categories(revised)
        current_tags = _itinerary_tags(revised)
        notes.append(f"为覆盖 `{interest}` 偏好，补充 `{candidate.poi.name}`。")
    return revised, notes


def _replace_lowest_priority_stop(
    itinerary: Itinerary,
    candidate: ScoredPOI,
) -> Itinerary:
    if not itinerary.days:
        return itinerary

    target_day = min(itinerary.days, key=lambda day: len(day.stops))
    if not target_day.stops:
        new_stops = [_stop_from_scored(candidate, START_TIMES[0])]
    else:
        new_stops = list(target_day.stops)
        new_stops[-1] = _stop_from_scored(candidate, new_stops[-1].start_time)

    days = [
        replace(day, stops=new_stops)
        if day.day_index == target_day.day_index
        else day
        for day in itinerary.days
    ]
    return replace(itinerary, days=days)


def _normalize_start_times(itinerary: Itinerary) -> Itinerary:
    days = []
    for day in itinerary.days:
        stops = [
            replace(stop, start_time=START_TIMES[index])
            for index, stop in enumerate(day.stops[: len(START_TIMES)])
        ]
        days.append(replace(day, stops=stops))
    return replace(itinerary, days=days)


def _stop_from_scored(item: ScoredPOI, start_time: str) -> ItineraryStop:
    note = "；".join(item.reasons) if item.reasons else "由 reviser 自动补充"
    return ItineraryStop(
        poi=item.poi,
        start_time=start_time,
        duration_min=item.poi.estimated_duration_min,
        note=note,
    )


def _find_candidate(
    ranked_pois: list[ScoredPOI],
    used_ids: set[str],
    predicate,
) -> ScoredPOI | None:
    for item in ranked_pois:
        if item.poi.poi_id in used_ids:
            continue
        if predicate(item):
            return item
    return None


def _pop_first(items: list[ScoredPOI]) -> ScoredPOI | None:
    if not items:
        return None
    return items.pop(0)


def _used_poi_ids(itinerary: Itinerary) -> set[str]:
    return {
        stop.poi.poi_id
        for day in itinerary.days
        for stop in day.stops
    }


def _itinerary_poi_names(itinerary: Itinerary) -> list[str]:
    return [
        stop.poi.name
        for day in itinerary.days
        for stop in day.stops
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


def _matches_any_term(value: str, terms: list[str]) -> bool:
    return any(term in value for term in terms)
