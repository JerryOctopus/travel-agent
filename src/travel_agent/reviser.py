from __future__ import annotations

from dataclasses import replace
import math

from travel_agent.critic import (
    INTEREST_TO_CATEGORIES,
    _has_lunch_window,
    critique_itinerary,
    poi_complies_with_dietary,
    poi_matches_interest,
    poi_matches_must_visit,
    requested_interests,
)
from travel_agent.planning import (
    ACTIVITY_TIMES, MEAL_TIMES, MEAL_TIME_WINDOWS, START_TIMES, _daily_opening_window,
    poi_open_on_trip_day,
)
from travel_agent.poi_evidence import is_verified_plannable_poi, poi_avoid_match
from travel_agent.schemas import CriticResult, Itinerary, ItineraryDay, ItineraryStop, ScoredPOI, TravelProfile


def revise_itinerary(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    critic_result: CriticResult,
) -> tuple[Itinerary, CriticResult, list[str]]:
    ranked_pois = [
        item for item in ranked_pois
        if (item.poi.category == "food" or is_verified_plannable_poi(item.poi))
        and poi_avoid_match(item.poi, profile) is None
    ]
    if not critic_result.issues:
        return itinerary, critic_result, []

    revised = itinerary
    notes: list[str] = []
    for issue in critic_result.issues:
        if issue.code == "avoid_term_included":
            revised, changed = _remove_avoided_pois(revised, ranked_pois, profile)
            notes.extend(changed)
        elif issue.code == "dietary_constraint_violated":
            revised, changed = _remove_noncompliant_food(revised, ranked_pois, profile)
            notes.extend(changed)
        elif issue.code == "must_visit_missing":
            revised, changed = _ensure_must_visit(revised, ranked_pois, profile)
            notes.extend(changed)
        elif issue.code == "interest_not_covered":
            revised, changed = _ensure_interest_coverage(revised, ranked_pois, profile)
            notes.extend(changed)
        elif issue.code == "meal_break_missing":
            revised, changed = _make_room_for_meal_window(revised, profile)
            notes.extend(changed)

    if any(
        issue.code in {"route_too_long", "daily_route_too_long", "walking_distance_exceeded"}
        for issue in critic_result.issues
    ):
        revised, changed = _repair_long_routes(revised, ranked_pois, profile)
        notes.extend(changed)

    # Completeness repair runs after route repair because the latter may remove
    # an outlying restaurant or activity and create a new gap.
    revised, changed = _fill_missing_meal_days(revised, ranked_pois, profile)
    notes.extend(changed)
    revised, changed = _fill_sparse_activity_days(revised, ranked_pois, profile)
    notes.extend(changed)

    # Route and completeness repairs can remove a venue that satisfied a hard
    # place or a stated interest. Re-apply coverage once to the final shape so
    # revision prose and the delivered itinerary cannot diverge.
    revised, changed = _ensure_must_visit(revised, ranked_pois, profile)
    notes.extend(changed)
    revised, changed = _ensure_interest_coverage(revised, ranked_pois, profile)
    notes.extend(changed)

    revised = _normalize_start_times(revised)
    revised_result = critique_itinerary(revised, profile)
    return revised, revised_result, _reconcile_revision_notes(notes, revised, profile)


def _make_room_for_meal_window(
    itinerary: Itinerary,
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    """Apply the user's explicit shortlist-pruning permission for a real meal."""
    state = profile.constraint_state or {}
    if state.get("candidate_only") is not True:
        return itinerary, []
    fixed_terms = {
        str(event.get("location") or "").strip()
        for event in state.get("fixed_events") or []
        if isinstance(event, dict) and str(event.get("location") or "").strip()
    }
    days: list[ItineraryDay] = []
    notes: list[str] = []
    for day in itinerary.days:
        stops = list(day.stops)
        if _has_lunch_window(day):
            days.append(day)
            continue
        removable = [
            index for index, stop in enumerate(stops)
            if stop.poi.category != "food"
            and not any(poi_matches_must_visit(stop.poi, term) for term in profile.must_visit)
            and not any(poi_matches_must_visit(stop.poi, term) for term in fixed_terms)
        ]
        selected: tuple[int, list[ItineraryStop]] | None = None
        for index in removable:
            candidate_stops = [stop for offset, stop in enumerate(stops) if offset != index]
            candidate_day = replace(day, stops=candidate_stops)
            if _has_lunch_window(candidate_day):
                selected = index, candidate_stops
                break
        if selected is None:
            days.append(day)
            continue
        removed_index, kept = selected
        removed = stops[removed_index]
        days.append(replace(day, stops=kept, theme=_make_day_theme(kept)))
        notes.append(
            f"为保留可执行用餐窗口，按候选取舍授权删减 `{removed.poi.name}`。"
        )
    return replace(itinerary, days=days), notes


def _repair_long_routes(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    """Replace the optional endpoint of the worst route with a nearby POI.

    The critic's route issues used to be terminal because the reviser handled
    only must/avoid/interest violations.  This bounded repair preserves hard
    must-visits and prefers dropping an explicitly optional-remove venue.
    Route estimates are recomputed by the planning subgraph after this step.
    """
    single_limit = {"relaxed": 45, "standard": 60, "intensive": 75}[profile.pace]
    daily_limit = {"relaxed": 75, "standard": 110, "intensive": 150}[profile.pace]
    explicit_relaxed_pace = (
        profile.pace == "relaxed"
        and (profile.constraint_state or {}).get("pace") == "relaxed"
    )
    optional_terms = [
        str(item)
        for item in ((profile.constraint_state or {}).get("optional_remove") or [])
        if str(item)
    ]
    used_ids = _used_poi_ids(itinerary)
    pool = [
        item
        for item in ranked_pois
        # A verified restaurant may truthfully be recommended on more than one
        # day. Reusing a nearby compatible meal is preferable to deleting the
        # day's only grounded meal when no distinct local alternative exists.
        if (item.poi.poi_id not in used_ids or item.poi.category == "food")
        and item.poi.category not in {"hotel", "transport"}
        and (item.poi.category != "food" or poi_complies_with_dietary(item.poi, profile))
        and poi_avoid_match(item.poi, profile) is None
        and not _matches_any_term(item.poi.name, optional_terms)
    ]
    notes: list[str] = []
    days: list[ItineraryDay] = []

    for day in itinerary.days:
        stops = list(day.stops)
        routes = [
            (index, stop.route_from_previous)
            for index, stop in enumerate(stops)
            if index > 0 and stop.route_from_previous is not None
        ]
        total = sum(route.duration_min for _index, route in routes)
        try:
            walking_limit = float(
                (profile.constraint_state or {}).get("max_walking_km_per_day")
            )
        except (TypeError, ValueError):
            walking_limit = None
        walking_total = sum(
            route.walking_distance_km
            if route.walking_distance_km is not None
            else (route.distance_km if route.mode == "walk" else 0.0)
            for _index, route in routes
        )
        walking_exceeded = walking_limit is not None and walking_total > walking_limit
        bad = [(index, route) for index, route in routes if route.duration_min > single_limit]
        if not bad and total <= daily_limit and not walking_exceeded:
            days.append(day)
            continue
        if not routes or len(stops) < 2:
            days.append(day)
            continue

        route_index, _route = max(
            routes if walking_exceeded else (bad or routes),
            key=(
                (lambda item: (
                    item[1].walking_distance_km
                    if item[1].walking_distance_km is not None
                    else (item[1].distance_km if item[1].mode == "walk" else 0.0)
                ))
                if walking_exceeded
                else (lambda item: item[1].duration_min)
            ),
        )
        endpoints = [route_index - 1, route_index]
        fixed_terms = [
            str(event.get("location") or "").strip()
            for event in ((profile.constraint_state or {}).get("fixed_events") or [])
            if isinstance(event, dict) and str(event.get("location") or "").strip()
        ]
        fixed_endpoint_route = any(
            _matches_any_term(stops[index].poi.name, fixed_terms)
            for index in endpoints
        )
        removable = [
            index
            for index in endpoints
            if not any(
                poi_matches_must_visit(stops[index].poi, term)
                for term in profile.must_visit
            )
        ]
        if not removable:
            days.append(day)
            continue
        optional_index = next(
            (
                index
                for index in removable
                if _matches_any_term(stops[index].poi.name, optional_terms)
            ),
            None,
        )
        culprit_index = (
            optional_index
            if optional_index is not None
            else max(
                removable,
                key=lambda index: (
                    _isolation_score(stops, index),
                    stops[index].poi.category == "food",
                    any(
                        item.poi.category == stops[index].poi.category
                        for item in pool
                    ),
                ),
            )
        )
        culprit = stops[culprit_index]
        anchors = [
            stops[index].poi
            for index in (culprit_index - 1, culprit_index + 1)
            if 0 <= index < len(stops)
        ]
        replacement_pool = (
            [
                item for item in pool
                if item.poi.category == "food"
                and item.poi.poi_id != culprit.poi.poi_id
                and all(
                    stop.poi.poi_id != item.poi.poi_id
                    for index, stop in enumerate(stops)
                    if index != culprit_index
                )
            ]
            if culprit.poi.category == "food"
            else [item for item in pool if item.poi.category != "food"]
        )
        replacement = min(
            replacement_pool,
            key=lambda item: max(
                (_haversine_km(item.poi, anchor) for anchor in anchors),
                default=0.0,
            ),
            default=None,
        )
        if replacement is not None and anchors:
            current_distance = max(
                (_haversine_km(culprit.poi, anchor) for anchor in anchors),
                default=0.0,
            )
            replacement_distance = max(
                (_haversine_km(replacement.poi, anchor) for anchor in anchors),
                default=0.0,
            )
            # Avoid oscillating among similarly remote alternatives across the
            # bounded critic loop.  A replacement must materially improve the
            # route; otherwise remove the optional stop and surface the gap.
            if current_distance > 0 and replacement_distance > current_distance * 0.75:
                replacement = None
        # A literal walking cap is a hard constraint.  Replacing one remote
        # optional stop with another can remain over the cap after the route
        # provider recomputes the path, so reduce load before optimizing variety.
        if (
            walking_exceeded
            or fixed_endpoint_route
            or (explicit_relaxed_pace and bool(bad))
        ) and len(stops) > 1:
            stops.pop(culprit_index)
            if walking_exceeded:
                notes.append(
                    f"为满足每日步行上限，删减可选地点 `{culprit.poi.name}`。"
                )
            else:
                reason = (
                    "为满足显式轻松节奏"
                    if explicit_relaxed_pace and not fixed_endpoint_route
                    else "为保障固定时段活动"
                )
                notes.append(f"{reason}，删减长通勤端点 `{culprit.poi.name}`。")
        elif replacement is not None:
            stops[culprit_index] = _stop_from_scored(
                replacement, culprit.start_time
            )
            pool.remove(replacement)
            notes.append(
                f"为修复超长通勤，将 `{culprit.poi.name}` 替换为附近的 `{replacement.poi.name}`。"
            )
        elif len(stops) > 1:
            stops.pop(culprit_index)
            notes.append(f"为修复超长通勤，删减可选地点 `{culprit.poi.name}`。")
        days.append(replace(day, stops=stops, theme=_make_day_theme(stops)))
    return replace(itinerary, days=days), notes


def _isolation_score(stops: list[ItineraryStop], index: int) -> float:
    poi = stops[index].poi
    others = [stop.poi for other_index, stop in enumerate(stops) if other_index != index]
    return sum(_haversine_km(poi, other) for other in others)


def _haversine_km(left, right) -> float:
    radius = 6371.0
    lat1 = math.radians(left.lat)
    lat2 = math.radians(right.lat)
    dlat = lat2 - lat1
    dlng = math.radians(right.lng - left.lng)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(value))


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
        and poi_avoid_match(item.poi, profile) is None
    ]

    days: list[ItineraryDay] = []
    for day in itinerary.days:
        stops: list[ItineraryStop] = []
        for stop in day.stops:
            if poi_avoid_match(stop.poi, profile) is None:
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


def _remove_noncompliant_food(
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
        and item.poi.category == "food"
        and poi_complies_with_dietary(item.poi, profile)
    ]
    days: list[ItineraryDay] = []
    for day in itinerary.days:
        stops: list[ItineraryStop] = []
        for stop in day.stops:
            if poi_complies_with_dietary(stop.poi, profile):
                stops.append(stop)
                continue
            replacement = _pop_first(replacement_pool)
            if replacement:
                stops.append(_stop_from_scored(replacement, stop.start_time))
                notes.append(
                    f"将缺乏饮食约束证据的 `{stop.poi.name}` 替换为 `{replacement.poi.name}`。"
                )
            else:
                notes.append(f"移除缺乏饮食约束证据的 `{stop.poi.name}`，暂无合规替代。")
        days.append(replace(day, stops=stops))
    return replace(itinerary, days=days), notes


def _ensure_must_visit(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    notes: list[str] = []
    missing_terms = [
        term
        for term in profile.must_visit
        if not any(
            poi_matches_must_visit(stop.poi, term)
            for day in itinerary.days
            for stop in day.stops
        )
    ]
    revised = itinerary
    for term in missing_terms:
        candidate = _find_candidate(
            ranked_pois,
            used_ids=_used_poi_ids(revised),
            predicate=lambda item, term=term: poi_matches_must_visit(item.poi, term),
            itinerary=revised,
        )
        if not candidate:
            notes.append(f"未找到必去地点 `{term}` 的候选 POI。")
            continue
        updated = _add_or_replace_stop(revised, candidate, profile)
        if candidate.poi.poi_id in _used_poi_ids(updated):
            notes.append(f"补充必去地点 `{candidate.poi.name}`。")
        revised = updated
    return revised, notes


def _ensure_interest_coverage(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    notes: list[str] = []
    revised = itinerary
    for interest in requested_interests(profile):
        covered = any(
            poi_matches_interest(stop.poi, interest)
            for day in revised.days
            for stop in day.stops
        )
        if covered:
            continue
        candidate = _find_candidate(
            ranked_pois,
            used_ids=_used_poi_ids(revised),
            predicate=lambda item, interest=interest: poi_matches_interest(item.poi, interest),
            itinerary=revised,
        )
        if not candidate:
            notes.append(f"未找到可覆盖 `{interest}` 偏好的候选 POI。")
            continue
        updated = _add_or_replace_stop(revised, candidate, profile)
        if candidate.poi.poi_id in _used_poi_ids(updated):
            notes.append(f"为覆盖 `{interest}` 偏好，补充 `{candidate.poi.name}`。")
        revised = updated
    return revised, notes


def _fill_sparse_activity_days(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    """Add one nearby evidenced activity to sparse days in multi-day plans."""
    used_ids = _used_poi_ids(itinerary)
    all_candidates = [
        item
        for item in ranked_pois
        if item.poi.category not in {"food", "hotel", "transport"}
        and not any(
            marker in item.poi.name
            for marker in ("纪念品", "旅游广场", "停车场", "售票处", "游客中心")
        )
        and poi_avoid_match(item.poi, profile) is None
    ]
    pool = [item for item in all_candidates if item.poi.poi_id not in used_ids]
    notes: list[str] = []
    days: list[ItineraryDay] = []
    fixed_days = {
        int(event.get("day"))
        for event in ((profile.constraint_state or {}).get("fixed_events") or [])
        if isinstance(event, dict)
        and str(event.get("day") or "").isdigit()
        and str(event.get("location") or "").strip() not in {"自由活动", "休息", "自由时间"}
    }
    for day in itinerary.days:
        stops = list(day.stops)
        activity_count = sum(stop.poi.category != "food" for stop in stops)
        if (
            day.day_index in fixed_days
            or activity_count >= 2
            or (
                activity_count >= 1
                and profile.pace == "relaxed"
                and (profile.constraint_state or {}).get("pace") == "relaxed"
            )
        ):
            days.append(day)
            continue
        anchors = [stop.poi for stop in stops]
        open_pool = [
            item for item in pool
            if poi_open_on_trip_day(item.poi, profile, day.day_index)
        ]
        if not open_pool:
            days.append(day)
            continue
        candidate = min(
            open_pool,
            key=lambda item: (
                max((_haversine_km(item.poi, anchor) for anchor in anchors), default=0.0),
                -item.score,
            ),
        )
        max_anchor_distance = max(
            (_haversine_km(candidate.poi, anchor) for anchor in anchors),
            default=0.0,
        )
        sparse_fill_radius_km = {
            "relaxed": 10.0,
            "standard": 15.0,
            "intensive": 25.0,
        }[profile.pace]
        if anchors and max_anchor_distance > sparse_fill_radius_km:
            # A completeness warning must not reintroduce an outlier that the
            # route repair just removed.  Preserve a truthful free period when
            # no geographically coherent evidenced activity is available.
            days.append(day)
            continue
        stops.append(_stop_from_scored(candidate, ACTIVITY_TIMES[-1]))
        if candidate in pool:
            pool.remove(candidate)
        used_ids.add(candidate.poi.poi_id)
        notes.append(f"为补全第{day.day_index}天下午行程，增加 `{candidate.poi.name}`。")
        days.append(replace(day, stops=stops, theme=_make_day_theme(stops)))
    return replace(itinerary, days=days), notes


def _fill_missing_meal_days(
    itinerary: Itinerary,
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
) -> tuple[Itinerary, list[str]]:
    """Restore one nearby, dietary-compliant meal after route repairs."""
    used_ids = _used_poi_ids(itinerary)
    candidates = [
        item
        for item in ranked_pois
        if item.poi.category == "food"
        and poi_complies_with_dietary(item.poi, profile)
    ]
    notes: list[str] = []
    days: list[ItineraryDay] = []
    for day in itinerary.days:
        stops = list(day.stops)
        if any(stop.poi.category == "food" for stop in stops) or not candidates:
            days.append(day)
            continue
        anchors = [stop.poi for stop in stops]
        meal_fill_radius_km = {
            "relaxed": 10.0,
            "standard": 15.0,
            "intensive": 25.0,
        }[profile.pace]
        day_candidates = [
            (
                item,
                min(
                    (_haversine_km(item.poi, anchor) for anchor in anchors),
                    default=0.0,
                ),
            )
            for item in candidates
            if poi_open_on_trip_day(item.poi, profile, day.day_index)
        ]
        viable = [
            (item, distance)
            for item, distance in day_candidates
            if not anchors or distance <= meal_fill_radius_km
        ]
        if not viable and "food" in requested_interests(profile):
            # An explicit daily food request is incomplete without a grounded
            # venue. Keep the nearest verified option even when it is outside
            # the clustering radius; final route rebinding can select a
            # provider-verified faster mode or reject the candidate honestly.
            viable = day_candidates
        if not viable:
            # A meal-completeness repair must not undo a prior long-route
            # repair. The renderer can truthfully reserve an unspecific meal
            # window when no dietary-compliant evidenced venue is nearby.
            days.append(day)
            continue
        candidate, _distance = min(
            viable,
            key=lambda pair: (
                pair[0].poi.poi_id in used_ids,
                pair[1],
                -pair[0].score,
            ),
        )
        reused = candidate.poi.poi_id in used_ids
        meal_stop = _stop_from_scored(candidate, MEAL_TIMES[0])
        if len(stops) < _max_stops_per_day(profile):
            stops.append(meal_stop)
        else:
            replaceable = [
                index
                for index, stop in enumerate(stops)
                if stop.poi.category != "food"
                and not any(
                    poi_matches_must_visit(stop.poi, term)
                    for term in profile.must_visit
                )
            ]
            if not replaceable:
                days.append(day)
                continue
            stops[replaceable[-1]] = meal_stop
        used_ids.add(candidate.poi.poi_id)
        action = "复用" if reused else "增加"
        notes.append(
            f"为补全第{day.day_index}天用餐，{action} `{candidate.poi.name}`。"
        )
        days.append(replace(day, stops=stops, theme=_make_day_theme(stops)))
    return replace(itinerary, days=days), notes


def _add_or_replace_stop(
    itinerary: Itinerary,
    candidate: ScoredPOI,
    profile: TravelProfile,
) -> Itinerary:
    if not itinerary.days:
        return itinerary

    target_day = min(itinerary.days, key=lambda day: len(day.stops))
    max_stops = _max_stops_per_day(profile)
    if not target_day.stops:
        new_stops = [_stop_from_scored(candidate, START_TIMES[0])]
    elif len(target_day.stops) < max_stops:
        start_time = START_TIMES[min(len(target_day.stops), len(START_TIMES) - 1)]
        new_stops = [*target_day.stops, _stop_from_scored(candidate, start_time)]
    else:
        new_stops = list(target_day.stops)
        replacement_index = _best_replacement_index(new_stops, candidate, profile)
        if replacement_index is None:
            return itinerary
        new_stops[replacement_index] = _stop_from_scored(
            candidate,
            new_stops[replacement_index].start_time,
        )

    days = [
        replace(day, stops=new_stops, theme=_make_day_theme(new_stops))
        if day.day_index == target_day.day_index
        else day
        for day in itinerary.days
    ]
    return replace(itinerary, days=days)


def _normalize_start_times(itinerary: Itinerary) -> Itinerary:
    days = []
    for day in itinerary.days:
        stops = _schedule_stops(day.stops[: len(START_TIMES)])
        days.append(replace(day, stops=stops))
    return replace(itinerary, days=days)


def _best_replacement_index(
    stops: list[ItineraryStop],
    candidate: ScoredPOI,
    profile: TravelProfile,
) -> int | None:
    candidate_coverage = _covered_interests(candidate.poi, profile)
    best_index: int | None = None
    best_lost_count = 10**6
    for index, stop in enumerate(stops):
        # Never fix one requirement by deleting another hard must-visit.  The
        # candidate may replace a stop only when all must-visit terms remain
        # represented by the candidate or another stop.
        remaining_pois = [candidate.poi] + [
            other.poi for other_index, other in enumerate(stops) if other_index != index
        ]
        if any(
            poi_matches_must_visit(stop.poi, term)
            and not any(poi_matches_must_visit(poi, term) for poi in remaining_pois)
            for term in profile.must_visit
        ):
            continue
        other_coverage = set(candidate_coverage)
        for other_index, other in enumerate(stops):
            if other_index != index:
                other_coverage.update(_covered_interests(other.poi, profile))
        lost = _covered_interests(stop.poi, profile) - other_coverage
        if len(lost) < best_lost_count:
            best_lost_count = len(lost)
            best_index = index
        if not lost:
            return index
    return best_index


def _covered_interests(poi, profile: TravelProfile) -> set[str]:
    return {
        interest
        for interest in requested_interests(profile)
        if poi_matches_interest(poi, interest)
    }


def _schedule_stops(stops: list[ItineraryStop]) -> list[ItineraryStop]:
    food_stops = [stop for stop in stops if stop.poi.category == "food"]
    activity_stops = [stop for stop in stops if stop.poi.category != "food"]
    if (
        food_stops
        and activity_stops
        and all(_is_evening_only(stop.poi) for stop in activity_stops)
    ):
        scheduled = [_retime_stop(food_stops[0], "17:30")]
        scheduled.extend(_retime_stop(stop, "18:00") for stop in activity_stops)
        scheduled.extend(_retime_stop(stop, "19:00") for stop in food_stops[1:])
        return scheduled
    scheduled: list[ItineraryStop] = []
    activity_index = 0
    food_index = 0

    for stop in activity_stops[:1]:
        scheduled.append(_retime_stop(stop, ACTIVITY_TIMES[activity_index]))
        activity_index += 1
    if food_stops:
        scheduled.append(_retime_stop(food_stops[0], MEAL_TIMES[food_index]))
        food_index += 1
    for stop in activity_stops[1:]:
        time = ACTIVITY_TIMES[min(activity_index, len(ACTIVITY_TIMES) - 1)]
        scheduled.append(_retime_stop(stop, time))
        activity_index += 1
    for stop in food_stops[1:]:
        time = MEAL_TIMES[min(food_index, len(MEAL_TIMES) - 1)]
        scheduled.append(_retime_stop(stop, time))
        food_index += 1

    ordered = sorted(scheduled, key=lambda stop: _time_sort_key(stop.start_time))
    retimed: list[ItineraryStop] = []
    for stop in ordered:
        start_minutes = _time_to_minutes(stop.start_time)
        if retimed:
            previous = retimed[-1]
            earliest = (
                _time_to_minutes(previous.start_time)
                + previous.duration_min
                + 15
            )
            start_minutes = max(start_minutes, _round_up_to_quarter(earliest))
        if stop.poi.category == "food":
            start_minutes = _next_meal_time_minutes(start_minutes)
        opening_window = _daily_opening_window(stop.poi.opening_hours)
        if opening_window is not None:
            start_minutes = max(start_minutes, opening_window[0])
        retimed.append(_retime_stop(stop, _minutes_to_time(start_minutes)))
    return retimed


def _is_evening_only(poi) -> bool:
    window = _daily_opening_window(poi.opening_hours)
    return window is not None and window[0] >= _time_to_minutes("17:00")


def _retime_stop(stop: ItineraryStop, start_time: str) -> ItineraryStop:
    return replace(stop, start_time=start_time, route_from_previous=None)


def _time_sort_key(value: str) -> tuple[int, int]:
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def _time_to_minutes(value: str) -> int:
    from travel_agent.constraint_events import clock_minutes

    return clock_minutes(value)


def _minutes_to_time(value: int) -> str:
    value = max(0, min(value, 23 * 60 + 59))
    return f"{value // 60:02d}:{value % 60:02d}"


def _next_meal_time_minutes(earliest: int) -> int:
    # Keep a delayed lunch in the lunch window.  Jumping directly from the
    # canonical 11:30 slot to 18:00 creates a false half-day gap whenever an
    # earlier activity plus transfer ends around noon.
    for slot in MEAL_TIME_WINDOWS:
        minutes = _time_to_minutes(slot)
        if minutes >= earliest:
            return minutes
    return _time_to_minutes("19:00")


def _round_up_to_quarter(value: int) -> int:
    remainder = value % 15
    if remainder == 0:
        return value
    return value + 15 - remainder


def _max_stops_per_day(profile: TravelProfile) -> int:
    if profile.pace == "relaxed":
        return 3 if int(profile.days or 1) > 1 else 2
    return {"standard": 3, "intensive": 4}[profile.pace]


def _make_day_theme(stops: list[ItineraryStop]) -> str:
    categories = [stop.poi.category for stop in stops]
    if "scenic" in categories and "food" in categories:
        return "自然风光与本地美食"
    if "culture" in categories or "museum" in categories:
        return "文化体验"
    if "food" in categories:
        return "本地美食"
    return "城市经典体验"


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
    itinerary: Itinerary | None = None,
) -> ScoredPOI | None:
    candidates = [
        item for item in ranked_pois
        if item.poi.poi_id not in used_ids and predicate(item)
    ]
    if not candidates:
        return None
    anchors = [
        stop.poi for day in (itinerary.days if itinerary else []) for stop in day.stops
    ]
    if not anchors:
        return candidates[0]
    rank = {item.poi.poi_id: index for index, item in enumerate(ranked_pois)}
    return min(
        candidates,
        key=lambda item: (
            min(_haversine_km(item.poi, anchor) for anchor in anchors),
            rank[item.poi.poi_id],
        ),
    )


def _reconcile_revision_notes(
    notes: list[str], itinerary: Itinerary, profile: TravelProfile
) -> list[str]:
    """Remove stale success/failure prose after later repair stages changed the plan."""
    final_pois = [stop.poi for day in itinerary.days for stop in day.stops]
    reconciled: list[str] = []
    for note in notes:
        quoted = [item.strip() for item in note.split("`")[1::2]]
        if any(marker in note for marker in ("补充", "增加")) and quoted:
            candidate_name = quoted[-1]
            if not any(poi.name == candidate_name for poi in final_pois):
                continue
        if any(marker in note for marker in ("移除", "删减")) and quoted:
            removed_name = quoted[-1]
            if any(poi.name == removed_name for poi in final_pois):
                continue
        if note.startswith("未找到可覆盖") and quoted:
            interest = quoted[0]
            if any(poi_matches_interest(poi, interest) for poi in final_pois):
                continue
        if note.startswith("未找到必去地点") and quoted:
            term = quoted[0]
            if any(poi_matches_must_visit(poi, term) for poi in final_pois):
                continue
        if note not in reconciled:
            reconciled.append(note)
    return reconciled


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
