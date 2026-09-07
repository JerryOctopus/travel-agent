from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import date, timedelta
from typing import Protocol

from travel_agent.critic import poi_complies_with_dietary, poi_matches_must_visit
from travel_agent.poi_evidence import (
    is_verified_plannable_poi,
    normalize_candidate_requirement,
    normalize_entity_name,
    poi_covers_requirement,
)
from travel_agent.route_evidence import (
    canonical_route_evidence_status,
    normalize_route_evidence,
    route_supports_endpoints,
)
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


START_TIMES = ["09:30", "11:30", "14:30", "18:00"]
ACTIVITY_TIMES = ["09:30", "14:30", "16:30"]
# Hard named venues must fit before an explicit same-day deadline.  Optional
# activities keep the more leisurely public schedule above, while required
# venues use compact, evenly spaced anchors and are still subject to route,
# opening-hours, and final deadline validation below.
REQUIRED_ACTIVITY_TIMES = ["09:00", "12:00", "15:00", "17:00"]
MEAL_TIMES = ["11:30", "18:00"]
MEAL_TIME_WINDOWS = ["11:30", "12:00", "12:30", "12:45", "17:30", "18:00", "19:00"]
TRANSFER_BUFFER_MIN = 15
QUALITATIVE_LOW_WALKING_LEG_KM = 1.0
MEAL_AREA_DIVERSITY_KM = 5.0


def build_simple_itinerary(
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    route_estimator: RouteEstimator | None = None,
    *,
    preserve_must_visit_capacity: bool = False,
) -> Itinerary:
    if not profile.destination:
        raise ValueError("destination is required to build itinerary")
    if not profile.days:
        raise ValueError("days is required to build itinerary")

    stops_per_day = _stops_per_day(profile)
    total_stops = profile.days * stops_per_day
    ranked_pois = [
        item for item in ranked_pois
        if item.poi.category == "food" or is_verified_plannable_poi(item.poi)
    ]
    selected = _select_plan_candidates(
        ranked_pois,
        profile,
        total_stops,
        preserve_must_visit_capacity=preserve_must_visit_capacity,
    )
    day_groups = _cluster_scored_pois_by_day(
        selected,
        profile.days,
        stops_per_day,
        profile,
    )

    days: list[ItineraryDay] = []
    for day_index, day_candidates in enumerate(day_groups, start=1):
        stops = _build_day_stops(day_candidates, profile, route_estimator)
        days.append(
            ItineraryDay(
                day_index=day_index,
                theme=_make_day_theme(stops),
                stops=stops,
            )
        )

    days = apply_structured_schedule_constraints(
        days,
        ranked_pois,
        profile,
        route_estimator,
    )
    return Itinerary(
        city=profile.destination,
        days=days,
        summary=f"{profile.destination}{profile.days}天{_pace_label(profile)}行程草案",
    )


def _select_plan_candidates(
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    total_stops: int,
    *,
    preserve_must_visit_capacity: bool = False,
) -> list[ScoredPOI]:
    """Reserve capacity for evidenced meals before score fill."""
    selected: list[ScoredPOI] = []

    def add(item: ScoredPOI) -> None:
        if item.poi.poi_id not in {entry.poi.poi_id for entry in selected}:
            selected.append(item)

    state = profile.constraint_state or {}
    unspecified_fixed_locations = {
        normalize_entity_name(item)
        for item in state.get("user_owned_unspecified_fixed_event_locations") or []
        if normalize_entity_name(item)
    }
    fixed_location_terms = list(dict.fromkeys(
        str(event.get("location") or "").strip()
        for event in state.get("fixed_events") or []
        if isinstance(event, dict)
        and str(event.get("location") or "").strip()
        and normalize_entity_name(event.get("location"))
        not in unspecified_fixed_locations
        and str(event.get("location") or "").strip()
        not in {"自由活动", "休息", "自由时间"}
    ))
    raw_candidate_terms = state.get("candidate_attractions") or []
    candidate_terms = (
        [raw_candidate_terms]
        if isinstance(raw_candidate_terms, str)
        else list(raw_candidate_terms)
    )
    eligible = [
        item
        for item in ranked_pois
        if item.poi.category != "food" or poi_complies_with_dietary(item.poi, profile)
        if _is_suitable_for_requested_weekday(item.poi, state.get("weekday"))
        if not profile.start_date or any(
            poi_open_on_trip_day(item.poi, profile, day_index)
            for day_index in range(1, max(1, int(profile.days or 1)) + 1)
        )
    ]
    eligible = _dedupe_semantic_venues(
        eligible,
        required_terms=[str(term) for term in candidate_terms],
    )
    # A named fixed appointment is a hard venue constraint even when the user
    # did not repeat it in must_visit.  Reserve the evidenced venue before
    # score fill so geographic clustering can keep a feasible nearby stop on
    # that day and the final schedule can prove the transfer.
    for term in fixed_location_terms:
        match = min(
            (
                item for item in eligible
                if poi_matches_must_visit(item.poi, term)
            ),
            key=lambda item: _named_venue_priority(item.poi, term),
            default=None,
        )
        if match is not None:
            add(match)
            event = next(
                (
                    item for item in state.get("fixed_events") or []
                    if isinstance(item, dict)
                    and normalize_entity_name(item.get("location"))
                    == normalize_entity_name(term)
                ),
                {},
            )
            try:
                event_start = _time_to_minutes(str(event.get("start")))
            except (TypeError, ValueError):
                event_start = 0
            if event_start >= _time_to_minutes("12:00"):
                nearby = min(
                    (
                        item for item in eligible
                        if item.poi.poi_id != match.poi.poi_id
                        and item.poi.category not in {"food", "hotel", "transport"}
                        and _haversine_km(
                            item.poi.lng,
                            item.poi.lat,
                            match.poi.lng,
                            match.poi.lat,
                        ) <= 15.0
                    ),
                    key=lambda item: (
                        _haversine_km(
                            item.poi.lng,
                            item.poi.lat,
                            match.poi.lng,
                            match.poi.lat,
                        ),
                        -item.score,
                    ),
                    default=None,
                )
                if nearby is not None:
                    add(nearby)
    # Hard must-visits own capacity before completeness preferences.  Previously
    # meals were reserved first, so a small one-day plan could consume its last
    # slot with food and silently drop an evidenced must-visit.
    if preserve_must_visit_capacity:
        for term in profile.must_visit:
            matches = [
                item
                for item in eligible
                if poi_matches_must_visit(item.poi, term)
            ]
            match = min(
                matches,
                key=lambda item: _named_venue_priority(item.poi, term),
                default=None,
            )
            if match is not None:
                add(match)

    # Named places that the user explicitly asked us to verify are softer than
    # must-visits, but they still take precedence over unrelated recommendations.
    # Only reserve a place when the bound evidence represents the venue itself
    # and does not explicitly say it is closed on the requested weekday.
    for term in candidate_terms:
        match = min(
            (
                item
                for item in eligible
                if _matches_candidate_attraction(item.poi, str(term))
                and _is_suitable_for_requested_weekday(item.poi, state.get("weekday"))
            ),
            key=lambda item: _named_venue_priority(item.poi, str(term)),
            default=None,
        )
        if match is not None:
            add(match)

    # Concrete restaurant stops are optional unless the user requested food or
    # supplied a dietary hard constraint. Generic itineraries reserve meal time
    # in prose/rendering instead of inventing a restaurant recommendation.
    selected_food_brands: set[str] = set()
    selected_food_pois: list[POI] = []
    food_candidates = [entry for entry in eligible if entry.poi.category == "food"]
    meal_anchors = [entry.poi for entry in selected if entry.poi.category != "food"]
    if meal_anchors:
        food_candidates.sort(
            key=lambda item: (
                min(
                    _haversine_km(
                        item.poi.lng,
                        item.poi.lat,
                        anchor.lng,
                        anchor.lat,
                    )
                    for anchor in meal_anchors
                ),
                -item.score,
            )
        )
    meal_requested = bool(
        profile.food_preference
        or "food" in profile.interests
        or state.get("specific_restaurant_recommendation")
        # Dietary-only wording does not trigger restaurant retrieval, but if
        # verified restaurant evidence is already bound, use compatible meals
        # instead of discarding that evidence.
        or (state.get("dietary") and food_candidates)
    )
    for item in food_candidates if meal_requested else []:
        if len(selected) >= total_stops:
            break
        brand = _food_brand_key(item.poi.name)
        if brand in selected_food_brands:
            continue
        if any(
            _haversine_km(
                item.poi.lng,
                item.poi.lat,
                selected_food.lng,
                selected_food.lat,
            ) < MEAL_AREA_DIVERSITY_KM
            for selected_food in selected_food_pois
        ):
            continue
        add(item)
        selected_food_brands.add(brand)
        selected_food_pois.append(item.poi)
        if sum(entry.poi.category == "food" for entry in selected) >= profile.days:
            break

    # If the provider only returned one brand, completeness is preferable to
    # omitting meals.  Duplicate branches remain visible to the critic.
    for item in food_candidates if meal_requested else []:
        if len(selected) >= total_stops or sum(
            entry.poi.category == "food" for entry in selected
        ) >= profile.days:
            break
        brand = _food_brand_key(item.poi.name)
        if brand in selected_food_brands:
            continue
        before = len(selected)
        add(item)
        if len(selected) > before:
            selected_food_brands.add(brand)
            selected_food_pois.append(item.poi)

    # Sparse offline/local catalogs may contain fewer distinct restaurants
    # than trip days. Reusing a grounded restaurant on another day is truthful
    # and preferable to either inventing a venue or leaving an explicit food
    # itinerary without a meal. Live catalogs normally satisfy the distinct
    # candidate path above.
    food_count = sum(entry.poi.category == "food" for entry in selected)
    repeat_index = 0
    while meal_requested and food_candidates and food_count < profile.days and len(selected) < total_stops:
        selected.append(food_candidates[repeat_index % len(food_candidates)])
        repeat_index += 1
        food_count += 1

    # Fill remaining capacity with activities before optional extra meals.  A
    # distant second restaurant must not displace a viable attraction after
    # the daily evidenced-meal requirement has already been satisfied.
    fill_order = [item for item in eligible if item.poi.category != "food"]
    mobility_text = " ".join(
        str(value or "")
        for value in [state.get("mobility"), *(state.get("avoid") or [])]
    ).casefold()
    mobility_sensitive = bool(
        state.get("elderly")
        or state.get("wheelchair_user")
        or state.get("accessibility_priority")
        or any(
            state.get(key) not in (None, "", [], {})
            for key in (
                "max_walking_km_per_day", "max_single_walk_km",
                "walking_time_max_min", "max_single_walk_min",
            )
        )
        or any(
            marker in mobility_text
            for marker in (
                "low_walking", "少步行", "不要安排太多步行", "避免步行",
                "wheelchair", "轮椅", "台阶", "楼梯", "爬坡",
            )
        )
    )
    activity_anchors = [
        item.poi for item in selected if item.poi.category != "food"
    ]
    if mobility_sensitive and activity_anchors:
        fill_order.sort(key=lambda item: (
            min(
                _haversine_km(
                    item.poi.lng, item.poi.lat, anchor.lng, anchor.lat
                )
                for anchor in activity_anchors
            ),
            -item.score,
        ))
    # Meal reservation above is capped at one concrete restaurant per trip
    # day. Do not use spare activity capacity for additional restaurants: two
    # adjacent meals can create an infeasible transfer and does not improve a
    # full itinerary's completeness.
    if meal_requested and profile.days == 1:
        # A one-day food-focused request may intentionally be a lunch/dinner
        # crawl; the multi-day cap above is about preventing uneven clustering.
        fill_order += [item for item in eligible if item.poi.category == "food"]
    for item in fill_order:
        if len(selected) >= total_stops:
            break
        add(item)
    return selected[:total_stops]


def apply_structured_schedule_constraints(
    days: list[ItineraryDay],
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    route_estimator: RouteEstimator | None,
) -> list[ItineraryDay]:
    """Apply literal fixed events and end-time limits without inventing POIs."""
    state = profile.constraint_state or {}
    conditional_avoid = state.get("conditional_avoid_window") or {}
    conditional_outdoor_window: tuple[int, int] | None = None
    if (
        isinstance(conditional_avoid, dict)
        and conditional_avoid.get("avoid") == "long_outdoor_activity"
    ):
        try:
            avoid_start = _time_to_minutes(str(conditional_avoid.get("start")))
            avoid_end = _time_to_minutes(str(conditional_avoid.get("end")))
        except (TypeError, ValueError):
            pass
        else:
            if avoid_end > avoid_start:
                conditional_outdoor_window = (avoid_start, avoid_end)
    raw_events = state.get("fixed_events") or []
    candidate_terms = state.get("candidate_attractions") or []
    if isinstance(candidate_terms, str):
        candidate_terms = [candidate_terms]
    events_by_day: dict[int, list[tuple[int, int, ScoredPOI]]] = {}
    reserved_windows_by_day: dict[int, list[tuple[int, int]]] = {}
    unspecified_locations = {
        normalize_entity_name(item)
        for item in state.get("user_owned_unspecified_fixed_event_locations") or []
        if normalize_entity_name(item)
    }
    for raw in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(raw, dict):
            continue
        try:
            raw_day = raw.get("day")
            if raw_day is not None:
                day_index = int(raw_day)
            elif raw.get("date") and profile.start_date:
                day_index = (
                    date.fromisoformat(str(raw.get("date")))
                    - date.fromisoformat(str(profile.start_date))
                ).days + 1
            else:
                continue
            start = _time_to_minutes(str(raw.get("start")))
            end = _time_to_minutes(str(raw.get("end")))
        except (TypeError, ValueError):
            continue
        location = str(raw.get("location") or "").strip()
        if day_index < 1 or day_index > len(days) or end <= start or not location:
            continue
        if normalize_entity_name(location) in unspecified_locations:
            reserved_windows_by_day.setdefault(day_index, []).append((start, end))
            continue
        location_matches = [
            item
            for item in ranked_pois
            if (
                location in item.poi.name
                or item.poi.name in location
                or location in str(item.poi.address or "")
            )
        ]
        match = min(
            location_matches,
            key=lambda item: _named_venue_priority(item.poi, location),
            default=None,
        )
        if match is not None:
            if start >= _time_to_minutes("17:00"):
                meal_match = next(
                    (
                        item for item in ranked_pois
                        if item.poi.category == "food"
                        and (
                            location in item.poi.name
                            or item.poi.name in location
                            or location in str(item.poi.address or "")
                        )
                    ),
                    None,
                )
                match = meal_match or match
            events_by_day.setdefault(day_index, []).append((start, end, match))
        else:
            reserved_windows_by_day.setdefault(day_index, []).append((start, end))

    fixed_poi_ids = {
        item.poi.poi_id
        for events in events_by_day.values()
        for _start, _end, item in events
    }
    fixed_locations = [
        str(raw.get("location") or "").strip()
        for raw in raw_events if isinstance(raw, dict)
        if str(raw.get("location") or "").strip()
    ]
    deadline_text = state.get("activity_end_deadline") or state.get("return_deadline")
    try:
        deadline = _time_to_minutes(str(deadline_text)) if deadline_text else None
    except (TypeError, ValueError):
        deadline = None
    target_text = state.get("activity_end_target")
    try:
        activity_end_target = (
            _time_to_minutes(str(target_text)) if target_text else None
        )
    except (TypeError, ValueError):
        activity_end_target = None
    # A return deadline is arrival at the return location, not the end of the
    # last attraction.  When that location is outside the destination city and
    # no inventory-backed timetable is available, reserve a conservative
    # intercity/terminal buffer instead of filling the schedule to the deadline.
    return_location = str(state.get("return_location") or "").strip()
    if (
        deadline is not None
        and state.get("return_deadline")
        and return_location
        and str(profile.destination or "").strip() not in return_location
    ):
        deadline = max(0, deadline - 180)

    constrained_days: list[ItineraryDay] = []
    for day in days:
        candidates: list[tuple[int, int | None, ItineraryStop, bool]] = []
        for stop in day.stops:
            if stop.poi.poi_id in fixed_poi_ids:
                continue
            if any(
                location in stop.poi.name or stop.poi.name in location
                for location in fixed_locations
            ):
                continue
            if not poi_open_on_trip_day(stop.poi, profile, day.day_index):
                continue
            requested_start = _time_to_minutes(stop.start_time)
            opening_window = _opening_window_for_trip_day(
                stop.poi, profile, day.day_index
            )
            if opening_window is not None:
                requested_start = max(requested_start, opening_window[0])
            candidates.append((requested_start, None, stop, False))
        for start, end, item in events_by_day.get(day.day_index, []):
            candidates.append(
                (
                    start,
                    end,
                    ItineraryStop(
                        poi=item.poi,
                        start_time=_minutes_to_time(start),
                        duration_min=end - start,
                        note="用户已确认的固定时段活动",
                    ),
                    True,
                )
            )
        if deadline is not None:
            required: list[tuple[int, int | None, ItineraryStop, bool]] = []
            optional: list[tuple[int, int | None, ItineraryStop, bool]] = []
            for entry in candidates:
                stop = entry[2]
                is_required = any(
                    poi_matches_must_visit(stop.poi, name)
                    for name in profile.must_visit
                )
                (required if is_required and not entry[3] else optional).append(entry)
            required = [
                (
                    _time_to_minutes(
                        REQUIRED_ACTIVITY_TIMES[
                            min(index, len(REQUIRED_ACTIVITY_TIMES) - 1)
                        ]
                    ),
                    end,
                    stop,
                    fixed,
                )
                for index, (_start, end, stop, fixed) in enumerate(required)
            ]
            candidates = required + optional

        # Required venues with an evidenced closing time must not inherit a
        # late generic slot and then get discarded. Schedule earlier-closing
        # must-visits first; later steps still enforce transfers and opening.
        required_flexible = [
            entry
            for entry in candidates
            if not entry[3]
            and any(
                poi_matches_must_visit(entry[2].poi, name)
                for name in profile.must_visit
            )
        ]
        if required_flexible:
            def required_order_fits_opening_hours() -> bool:
                for index, entry in enumerate(required_flexible):
                    stop = entry[2]
                    proposed = max(
                        _time_to_minutes(
                            REQUIRED_ACTIVITY_TIMES[
                                min(index, len(REQUIRED_ACTIVITY_TIMES) - 1)
                            ]
                        ),
                        _preferred_activity_start_minutes(stop.poi, 0),
                    )
                    window = _opening_window_for_trip_day(
                        stop.poi, profile, day.day_index
                    )
                    if window is not None:
                        proposed = max(proposed, window[0])
                        if proposed + stop.duration_min > window[1]:
                            return False
                    admission = _last_admission_for_trip_day(
                        stop.poi, profile, day.day_index
                    )
                    if admission is not None and proposed > admission:
                        return False
                return True

            # The incoming order is already the minimum-distance daily route.
            # Preserve it when compact required slots satisfy every verified
            # opening boundary; earliest-closing-first is only a fallback for
            # an order that would actually miss admission or closing time.
            if not required_order_fits_opening_hours():
                required_flexible.sort(
                    key=lambda entry: (
                        (_opening_window_for_trip_day(
                            entry[2].poi, profile, day.day_index
                        ) or (0, 24 * 60))[1],
                        entry[0],
                    )
                )
            retimed_required = {
                entry[2].poi.poi_id: max(
                    _time_to_minutes(
                        REQUIRED_ACTIVITY_TIMES[
                            min(index, len(REQUIRED_ACTIVITY_TIMES) - 1)
                        ]
                    ),
                    _preferred_activity_start_minutes(entry[2].poi, 0),
                )
                for index, entry in enumerate(required_flexible)
            }
            candidates = [
                (retimed_required.get(entry[2].poi.poi_id, entry[0]), *entry[1:])
                for entry in candidates
            ]

        fixed_count = sum(1 for entry in candidates if entry[3])
        flexible_limit = max(0, _stops_per_day(profile) - fixed_count)
        flexible_seen = 0
        bounded_candidates: list[tuple[int, int | None, ItineraryStop, bool]] = []
        for entry in sorted(candidates, key=lambda item: (item[0], not item[3])):
            if entry[3]:
                bounded_candidates.append(entry)
            elif flexible_seen < flexible_limit:
                bounded_candidates.append(entry)
                flexible_seen += 1
        candidates = bounded_candidates
        candidates.sort(key=lambda entry: (entry[0], not entry[3]))

        scheduled: list[ItineraryStop] = []
        for requested_start, fixed_end, stop, is_fixed in candidates:
            route = _estimate_stop_route(
                scheduled[-1].poi if scheduled else None,
                stop.poi,
                profile,
                route_estimator,
            )
            earliest = requested_start
            if scheduled:
                previous = scheduled[-1]
                earliest = max(
                    earliest,
                    _time_to_minutes(previous.start_time)
                    + previous.duration_min
                    + (route.duration_min if route else 0)
                    + TRANSFER_BUFFER_MIN,
                )
            if is_fixed and earliest > requested_start:
                # A flexible earlier stop may not push a user-confirmed event.
                while scheduled and earliest > requested_start:
                    scheduled.pop()
                    route = _estimate_stop_route(
                        scheduled[-1].poi if scheduled else None,
                        stop.poi,
                        profile,
                        route_estimator,
                    )
                    earliest = requested_start
                    if scheduled:
                        previous = scheduled[-1]
                        earliest = max(
                            earliest,
                            _time_to_minutes(previous.start_time)
                            + previous.duration_min
                            + (route.duration_min if route else 0)
                            + TRANSFER_BUFFER_MIN,
                        )
                if earliest > requested_start:
                    continue
            actual_start = requested_start if is_fixed else _round_up_to_quarter(earliest)
            is_hard_required = any(
                poi_matches_must_visit(stop.poi, term)
                for term in profile.must_visit
            )
            if not is_fixed and stop.poi.category == "food":
                actual_start = _next_meal_time_minutes(actual_start)
            elif (
                not is_fixed
                and not is_hard_required
                and _time_to_minutes("11:30") <= actual_start < _time_to_minutes("13:00")
            ):
                # Do not turn the only plausible lunch window into a second
                # major activity.  Meal metadata must describe a usable break,
                # not claim a 17:30 "lunch" after a continuous sightseeing day.
                actual_start = _time_to_minutes("13:00")
            duration = (fixed_end - requested_start) if is_fixed and fixed_end else stop.duration_min
            if (
                not is_fixed
                and conditional_outdoor_window is not None
                and not stop.poi.indoor
                and duration >= 90
            ):
                avoid_start, avoid_end = conditional_outdoor_window
                if actual_start < avoid_end and actual_start + duration > avoid_start:
                    # The condition may not be knowable for a future trip.  A
                    # conservative schedule that reserves the user's avoidance
                    # window works in both branches and does not invent weather.
                    actual_start = avoid_end
            if not is_fixed and any(
                actual_start < reserved_end
                and actual_start + duration + TRANSFER_BUFFER_MIN > reserved_start
                for reserved_start, reserved_end in reserved_windows_by_day.get(day.day_index, [])
            ):
                continue
            # Never clamp an overflowed activity to 23:59.  If the preceding
            # transfer pushes a flexible stop past the calendar day, the stop
            # is not schedulable and must be omitted for the critic/gate to
            # handle explicitly.
            if actual_start + duration > 24 * 60:
                continue
            if deadline is not None and actual_start + duration > deadline:
                continue
            opening_window = _opening_window_for_trip_day(
                stop.poi, profile, day.day_index
            )
            if opening_window is not None:
                opens, closes = opening_window
                actual_start = max(actual_start, opens)
                # A verified named candidate should displace an unrelated
                # earlier optional stop instead of silently disappearing.
                is_named_candidate = any(
                    _matches_candidate_attraction(stop.poi, str(term))
                    for term in candidate_terms
                )
                # Exact-closing schedules are brittle: a small queue or
                # provider rounding error makes an optional visit infeasible.
                # Reserve the same conservative buffer used between stops,
                # while preserving user-fixed, hard-required and explicitly
                # named candidate commitments at their evidenced boundary.
                closing_deadline = closes
                if (
                    not is_fixed
                    and not is_hard_required
                    and not is_named_candidate
                    and stop.poi.category not in {"food", "hotel", "transport"}
                ):
                    closing_deadline -= TRANSFER_BUFFER_MIN
                if actual_start + duration > closing_deadline:
                    if is_named_candidate and not is_fixed:
                        actual_start = opens
                        while scheduled and actual_start < (
                            _time_to_minutes(scheduled[-1].start_time)
                            + scheduled[-1].duration_min
                        ):
                            scheduled.pop()
                        route = _estimate_stop_route(
                            scheduled[-1].poi if scheduled else None,
                            stop.poi,
                            profile,
                            route_estimator,
                        )
                    if actual_start + duration > closing_deadline:
                        continue
            last_admission = _last_admission_for_trip_day(
                stop.poi, profile, day.day_index
            )
            if not is_fixed and last_admission is not None and actual_start > last_admission:
                # Admission cutoffs govern arrival, not the venue's closing
                # time.  Move a flexible visit earlier only when doing so still
                # preserves the preceding route and opening boundary.
                earliest_allowed = 0
                if scheduled:
                    previous = scheduled[-1]
                    earliest_allowed = _round_up_to_quarter(
                        _time_to_minutes(previous.start_time)
                        + previous.duration_min
                        + (route.duration_min if route else 0)
                        + TRANSFER_BUFFER_MIN
                    )
                if opening_window is not None:
                    earliest_allowed = max(earliest_allowed, opening_window[0])
                if earliest_allowed > last_admission:
                    continue
                actual_start = last_admission
            scheduled.append(
                ItineraryStop(
                    poi=stop.poi,
                    start_time=_minutes_to_time(actual_start),
                    duration_min=duration,
                    note=stop.note,
                    route_from_previous=route,
                )
            )
        if activity_end_target is not None and scheduled:
            last = scheduled[-1]
            current_start = _time_to_minutes(last.start_time)
            current_end = current_start + last.duration_min
            proposed_start = activity_end_target - last.duration_min
            route_duration = (
                last.route_from_previous.duration_min
                if last.route_from_previous is not None else 0
            )
            earliest = 0
            if len(scheduled) > 1:
                previous = scheduled[-2]
                earliest = (
                    _time_to_minutes(previous.start_time)
                    + previous.duration_min
                    + route_duration
                    + TRANSFER_BUFFER_MIN
                )
            opening_window = _opening_window_for_trip_day(
                last.poi, profile, day.day_index
            )
            opens_late_enough = (
                opening_window is None
                or (
                    proposed_start >= opening_window[0]
                    and activity_end_target <= opening_window[1]
                )
            )
            last_admission = _last_admission_for_trip_day(
                last.poi, profile, day.day_index
            )
            overlaps_reserved = any(
                proposed_start < reserved_end
                and activity_end_target + TRANSFER_BUFFER_MIN > reserved_start
                for reserved_start, reserved_end in reserved_windows_by_day.get(
                    day.day_index, []
                )
            )
            if (
                last.poi.poi_id not in fixed_poi_ids
                and last.poi.category != "food"
                and current_end < activity_end_target
                and proposed_start >= earliest
                and (deadline is None or activity_end_target <= deadline)
                and opens_late_enough
                and (last_admission is None or proposed_start <= last_admission)
                and not overlaps_reserved
            ):
                scheduled[-1] = replace(
                    last, start_time=_minutes_to_time(proposed_start)
                )
        constrained_days.append(
            ItineraryDay(
                day_index=day.day_index,
                theme=_make_day_theme(scheduled),
                stops=scheduled,
            )
        )
    return constrained_days


def _named_venue_priority(poi: POI, required_name: str) -> tuple[int, int]:
    """Prefer the requested venue itself over a loosely related child POI."""
    required = normalize_entity_name(required_name)
    names = [
        normalize_entity_name(value)
        for value in (poi.name, poi.canonical_name, *poi.aliases)
        if str(value or "").strip()
    ]
    city = normalize_entity_name(poi.city)
    city_prefixes = {
        prefix
        for prefix in (city, city.removesuffix("市"), city.removesuffix("地区"))
        if prefix
    }
    names.extend(
        name[len(prefix):]
        for name in list(names)
        for prefix in city_prefixes
        if name.startswith(prefix) and len(name) > len(prefix)
    )
    if required in names:
        return 0, len(normalize_entity_name(poi.name))
    venue_suffix_priority = {
        "文化旅游区": 1,
        "风景名胜区": 1,
        "风景区": 2,
        "景区": 2,
        "旅游区": 2,
        "公园": 3,
    }
    suffix_match = min(
        (
            priority
            for suffix, priority in venue_suffix_priority.items()
            for name in names
            if name == required + suffix
        ),
        default=None,
    )
    if suffix_match is not None:
        return suffix_match, len(normalize_entity_name(poi.name))
    if poi.coverage_relation == "child_covers_parent":
        return 3, len(normalize_entity_name(poi.name))
    return 2, len(normalize_entity_name(poi.name))


def _daily_opening_window(value: str | None) -> tuple[int, int] | None:
    """Return a simple same-day opening window when the evidence is explicit."""
    text = str(value or "").strip()
    if not text or text.lower() in {"all_day", "24h"}:
        return None
    match = re.search(r"(\d{1,2}):(\d{2})\s*[-—–至到]\s*(\d{1,2}):(\d{2})", text)
    if match is None:
        return None
    opens = int(match.group(1)) * 60 + int(match.group(2))
    closes = int(match.group(3)) * 60 + int(match.group(4))
    if opens == 0 and closes == 24 * 60:
        return None
    # Overnight businesses (for example 16:00-04:00) are usable from their
    # opening time through midnight on the itinerary day.  Preserve the next-
    # day close as >24h so daytime scheduling can honor the opening boundary.
    if closes <= opens:
        closes += 24 * 60
    return opens, closes


_WEEKDAY_INDEX = {name: index for index, name in enumerate("一二三四五六日")}


def _opening_window_for_trip_day(
    poi: POI,
    profile: TravelProfile,
    day_index: int,
) -> tuple[int, int] | None:
    """Return the opening window that applies to the itinerary weekday.

    Provider strings often contain several weekday clauses.  Reading the first
    clock range for every day can schedule a weekend visit using weekday hours.
    When a dated trip is available, select the matching semicolon-delimited
    weekday clause before parsing its time range.
    """
    text = str(poi.opening_hours or "").strip()
    if not text or not profile.start_date:
        return _daily_opening_window(text)
    try:
        trip_date = (
            date.fromisoformat(str(profile.start_date))
            + timedelta(days=day_index - 1)
        )
    except (TypeError, ValueError):
        return _daily_opening_window(text)
    weekday_index = trip_date.weekday()

    clauses = [clause.strip() for clause in re.split(r"[；;]", text) if clause.strip()]
    scoped_clauses = [
        clause for clause in clauses
        if _has_weekday_selector(clause)
        or _calendar_ranges(clause, trip_date.year)
    ]
    for clause in scoped_clauses:
        if (
            _weekday_clause_applies(clause, weekday_index)
            and _date_clause_applies(clause, trip_date)
        ):
            return _daily_opening_window(clause)
    # If the provider supplied weekday-specific clauses but none applies, the
    # venue is not evidenced open that day.  The separate closure predicate
    # handles explicit closed text; here None keeps the scheduler fail-closed
    # instead of borrowing another day's hours.
    if scoped_clauses:
        return None
    return _daily_opening_window(text)


def _last_admission_for_trip_day(
    poi: POI,
    profile: TravelProfile,
    day_index: int,
) -> int | None:
    """Return an evidenced last-entry cutoff applicable to the trip day."""
    text = str(poi.opening_hours or "").strip()
    if not text:
        return None
    applicable = text
    if profile.start_date:
        try:
            trip_date = (
                date.fromisoformat(str(profile.start_date))
                + timedelta(days=day_index - 1)
            )
        except (TypeError, ValueError):
            trip_date = None
        if trip_date is not None:
            clauses = [
                clause.strip() for clause in re.split(r"[；;]", text)
                if clause.strip()
            ]
            scoped = [
                clause for clause in clauses
                if _has_weekday_selector(clause)
                or _calendar_ranges(clause, trip_date.year)
            ]
            matching = [
                clause for clause in scoped
                if _weekday_clause_applies(clause, trip_date.weekday())
                and _date_clause_applies(clause, trip_date)
            ]
            if scoped and not matching:
                return None
            if matching:
                applicable = "；".join(matching)
    patterns = (
        r"最晚(?:进入|入园|入馆|入场|检票|售票)\s*(\d{1,2}):(\d{2})",
        r"(\d{1,2}):(\d{2})\s*停止(?:进入|入园|入馆|入场|检票|售票)",
    )
    values: list[int] = []
    for pattern in patterns:
        values.extend(
            int(hour) * 60 + int(minute)
            for hour, minute in re.findall(pattern, applicable)
        )
    return min(values) if values else None


def _weekday_clause_applies(clause: str, weekday_index: int) -> bool:
    selectors = _weekday_selector_text(clause)
    if not re.search(r"周[一二三四五六日]", selectors):
        return True
    selected: set[int] = set()
    for start, end in re.findall(
        r"周([一二三四五六日])\s*(?:至|到|[-—–~])\s*周?([一二三四五六日])",
        selectors,
    ):
        left, right = _WEEKDAY_INDEX[start], _WEEKDAY_INDEX[end]
        if left <= right:
            selected.update(range(left, right + 1))
        else:
            selected.update((*range(left, 7), *range(0, right + 1)))
    without_ranges = re.sub(
        r"周[一二三四五六日]\s*(?:至|到|[-—–~])\s*周?[一二三四五六日]",
        "",
        selectors,
    )
    selected.update(
        _WEEKDAY_INDEX[name]
        for name in re.findall(r"周([一二三四五六日])", without_ranges)
    )
    return weekday_index in selected


def _weekday_selector_text(clause: str) -> str:
    selectors = re.split(r"\d{1,2}:\d{2}", clause, maxsplit=1)[0].replace("、", ",")
    # A weekday in parentheses immediately following a concrete calendar date
    # is a date annotation (for example ``9月1日(周二)起``), not a recurrence
    # selector.  Treating it as a selector incorrectly closes the venue on all
    # other days after the effective date.
    return re.sub(
        r"(?<=日)\s*[（(]周[一二三四五六日](?:\s*(?:至|到|[-—–~])\s*周?[一二三四五六日])?[）)]",
        "",
        selectors,
    )


def _has_weekday_selector(clause: str) -> bool:
    return re.search(r"周[一二三四五六日]", _weekday_selector_text(clause)) is not None


def _calendar_ranges(clause: str, year: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    pattern = re.compile(
        r"(?<!\d)(\d{1,2})(?:[./-]|月)(\d{1,2})(?:日)?\s*"
        r"(?:至|到|[-—–~])\s*"
        r"(\d{1,2})(?:[./-]|月)(\d{1,2})(?:日)?"
    )
    for start_month, start_day, end_month, end_day in pattern.findall(clause):
        try:
            start = date(year, int(start_month), int(start_day))
            end = date(year, int(end_month), int(end_day))
        except ValueError:
            continue
        ranges.append((start, end))
    return ranges


def _date_clause_applies(clause: str, trip_date: date) -> bool:
    ranges = _calendar_ranges(clause, trip_date.year)
    if not ranges:
        return True
    for start, end in ranges:
        if start <= end and start <= trip_date <= end:
            return True
        if start > end and (trip_date >= start or trip_date <= end):
            return True
    return False


def _matches_candidate_attraction(poi: POI, term: str) -> bool:
    return poi_covers_requirement(poi, normalize_candidate_requirement(term))


def _is_suitable_for_requested_weekday(poi: POI, weekday: object) -> bool:
    requested = str(weekday or "").strip()
    hours = str(poi.opening_hours or "").replace(" ", "")
    if not requested or not hours:
        return True
    explicitly_closed = (
        f"{requested}全天不开放" in hours
        or f"{requested}全天关闭" in hours
        or f"{requested}闭馆" in hours
        or (requested == "周一" and "周一闭馆" in hours)
    )
    if explicitly_closed:
        return False
    weekday_name = requested.removeprefix("周")
    weekday_index = _WEEKDAY_INDEX.get(weekday_name)
    if weekday_index is None:
        return True
    recurring_open_clauses = [
        clause
        for clause in re.split(r"[；;，,]", hours)
        if _has_weekday_selector(clause)
        and _daily_opening_window(clause) is not None
        and not re.search(r"不开放|关闭|闭馆|休息|暂停营业|停止开放", clause)
    ]
    if recurring_open_clauses:
        return any(
            _weekday_clause_applies(clause, weekday_index)
            for clause in recurring_open_clauses
        )
    return True


def poi_open_on_trip_day(poi: POI, profile: TravelProfile, day_index: int) -> bool:
    if not profile.start_date:
        return True
    try:
        current = date.fromisoformat(str(profile.start_date)) + timedelta(days=day_index - 1)
    except (TypeError, ValueError):
        return True
    opening = str(poi.opening_hours or "").strip()
    if not opening:
        return True
    weekday_index = current.weekday()
    clauses = [
        clause.strip()
        for clause in re.split(r"[；;，,]", opening)
        if clause.strip()
    ]
    relevant_clauses = [
        clause
        for clause in clauses
        if not _calendar_ranges(clause, current.year)
        or _date_clause_applies(clause, current)
    ]
    timed_clauses = [
        clause for clause in clauses if _daily_opening_window(clause) is not None
    ]
    if (
        timed_clauses
        and all(_calendar_ranges(clause, current.year) for clause in timed_clauses)
        and not any(
            _date_clause_applies(clause, current) for clause in timed_clauses
        )
    ):
        return False
    closure_pattern = re.compile(
        r"(?:不开放|关闭|闭馆|休息|暂停营业|停止开放)"
    )
    for clause in relevant_clauses:
        if (
            _has_weekday_selector(clause)
            and _weekday_clause_applies(clause, weekday_index)
            and closure_pattern.search(clause)
        ):
            return False

    scoped_open = [
        clause
        for clause in relevant_clauses
        if _has_weekday_selector(clause)
        and _daily_opening_window(clause) is not None
        and not closure_pattern.search(clause)
    ]
    if any(
        _weekday_clause_applies(clause, weekday_index)
        for clause in scoped_open
    ):
        return True
    general_open = any(
        not _has_weekday_selector(clause)
        and _daily_opening_window(clause) is not None
        and not closure_pattern.search(clause)
        for clause in relevant_clauses
    )
    if general_open:
        return True
    # A provider-supplied open weekday range is positive evidence only for the
    # included days.  Treat excluded days as closed instead of silently
    # borrowing that range's hours.
    if scoped_open:
        return False
    weekday = "周" + "一二三四五六日"[weekday_index]
    return _is_suitable_for_requested_weekday(poi, weekday)


def _estimate_stop_route(
    origin: POI | None,
    destination: POI,
    profile: TravelProfile,
    route_estimator: RouteEstimator | None,
) -> RouteInfo | None:
    if origin is None or route_estimator is None:
        return None
    route: RouteInfo | None = None
    try:
        route = normalize_route_evidence(
            route_estimator.estimate_route(origin, destination, profile.transport_mode)
        )
    except Exception:
        route = None

    route_is_bound = route is not None and route_supports_endpoints(
        route, origin.poi_id, destination.poi_id
    )
    state = profile.constraint_state or {}
    has_walking_cap = any(
        state.get(key) is not None
        for key in ("max_walking_km_per_day", "max_single_walk_km")
    )
    raw_avoid = state.get("avoid") or []
    avoid_values = [raw_avoid] if isinstance(raw_avoid, str) else list(raw_avoid)
    mobility_text = " ".join(
        str(value or "")
        for value in [state.get("mobility"), *avoid_values]
    ).casefold()
    qualitative_low_walking = bool(
        state.get("elderly")
        or any(
            marker in mobility_text
            for marker in (
                "low_walking", "少步行", "太多步行", "避免步行",
                "行动不便", "mobility_limited",
            )
        )
    )
    transit_walk_is_unknown = bool(
        route_is_bound
        and route is not None
        and route.mode == "public_transport"
        and route.walking_distance_km is None
    )
    transit_walk_exceeds_qualitative_limit = bool(
        route_is_bound
        and route is not None
        and route.mode == "public_transport"
        and route.walking_distance_km is not None
        and route.walking_distance_km > QUALITATIVE_LOW_WALKING_LEG_KM
    )
    pace_limit = {
        "relaxed": 45,
        "standard": 60,
        "intensive": 75,
    }[profile.pace]
    transit_exceeds_pace = bool(
        route_is_bound
        and route is not None
        and route.mode == "public_transport"
        and route.duration_min > pace_limit
    )
    raw_transport_modes = state.get("transport_modes") or []
    if isinstance(raw_transport_modes, str):
        raw_transport_modes = [raw_transport_modes]
    explicit_taxi_backup = bool(
        state.get("taxi_backup") is True
        or str(state.get("fallback_transport") or "").lower() in {"taxi", "drive"}
        or any(
            str(mode).lower() in {"taxi", "drive"}
            for mode in raw_transport_modes
        )
    )
    taxi_fallback_allowed = bool(
        not state.get("public_transport_required") or explicit_taxi_backup
    )
    if (
        profile.transport_mode not in {"taxi", "drive"}
        and taxi_fallback_allowed
        and (
            (
                (has_walking_cap or qualitative_low_walking)
                and (
                    not route_is_bound
                    or canonical_route_evidence_status(route) != "provider_verified"
                    or transit_walk_is_unknown
                    or (
                        qualitative_low_walking
                        and transit_walk_exceeds_qualitative_limit
                    )
                )
            )
            or transit_exceeds_pace
        )
    ):
        try:
            taxi_route = normalize_route_evidence(
                route_estimator.estimate_route(origin, destination, "taxi")
            )
            if (
                route_supports_endpoints(taxi_route, origin.poi_id, destination.poi_id)
                and canonical_route_evidence_status(taxi_route) == "provider_verified"
                and (
                    not transit_exceeds_pace
                    or (
                        taxi_route.duration_min <= pace_limit
                        and route is not None
                        and taxi_route.duration_min < route.duration_min
                    )
                )
            ):
                return taxi_route
        except Exception:
            pass

    if route_is_bound:
        return route

    try:
        from travel_agent.tools import estimate_route_minutes

        distance_km, duration_min = estimate_route_minutes(
            origin,
            destination,
            profile.transport_mode,
        )
        return normalize_route_evidence(RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=distance_km,
            duration_min=duration_min,
            mode=profile.transport_mode,
            source="haversine_recovery_estimate",
        ), provider_explicit=False)
    except Exception:
        return None


def rebind_itinerary_routes(
    itinerary: Itinerary,
    profile: TravelProfile,
    route_estimator: RouteEstimator | None,
) -> Itinerary:
    """Invalidate every old adjacent leg and bind routes to final POI ids."""
    days: list[ItineraryDay] = []
    for day in itinerary.days:
        stops: list[ItineraryStop] = []
        for stop in day.stops:
            previous = stops[-1].poi if stops else None
            existing = stop.route_from_previous
            existing_exceeds_pace = bool(
                existing is not None
                and existing.mode == "public_transport"
                and existing.duration_min > {
                    "relaxed": 45,
                    "standard": 60,
                    "intensive": 75,
                }[profile.pace]
            )
            route = (
                normalize_route_evidence(existing)
                if previous is not None
                and existing is not None
                and route_supports_endpoints(existing, previous.poi_id, stop.poi.poi_id)
                and canonical_route_evidence_status(existing) == "provider_verified"
                and not existing_exceeds_pace
                else _estimate_stop_route(previous, stop.poi, profile, route_estimator)
            )
            stops.append(replace(stop, route_from_previous=route))
        days.append(replace(day, stops=stops))
    return replace(itinerary, days=days)


def _haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(a))


def _centroid(pois: list[ScoredPOI]) -> tuple[float, float]:
    lngs = [item.poi.lng for item in pois]
    lats = [item.poi.lat for item in pois]
    return sum(lngs) / len(lngs), sum(lats) / len(lats)


def _greedy_nearest(pois: list[ScoredPOI]) -> list[ScoredPOI]:
    if len(pois) <= 2:
        return pois
    def path_from(start_index: int) -> tuple[float, list[ScoredPOI]]:
        remaining = list(pois)
        ordered = [remaining.pop(start_index)]
        distance = 0.0
        while remaining:
            last = ordered[-1].poi
            nearest_idx = min(
                range(len(remaining)),
                key=lambda index: _haversine_km(
                    last.lng,
                    last.lat,
                    remaining[index].poi.lng,
                    remaining[index].poi.lat,
                ),
            )
            next_item = remaining.pop(nearest_idx)
            distance += _haversine_km(
                last.lng, last.lat, next_item.poi.lng, next_item.poi.lat
            )
            ordered.append(next_item)
        return distance, ordered

    # With at most four stops per day, trying every start is cheap and avoids
    # reversed routes caused by whichever candidate happened to sort first.
    ordered = min(
        (path_from(index) for index in range(len(pois))), key=lambda item: item[0]
    )[1]
    return _apply_known_visit_order(ordered)


def _apply_known_visit_order(pois: list[ScoredPOI]) -> list[ScoredPOI]:
    """Honor a small set of externally constrained, one-way venue sequences."""
    names = [item.poi.name for item in pois]
    if any("故宫" in name for name in names):
        priority = {"天安门": 0, "故宫": 1, "景山": 2}

        def key(item: ScoredPOI) -> int:
            return next(
                (value for marker, value in priority.items() if marker in item.poi.name),
                1,
            )

        return sorted(pois, key=key)
    return pois


def _food_brand_key(name: str) -> str:
    value = re.sub(r"[（(][^）)]*[）)]", "", name)
    value = re.sub(r"(?:旗舰店|总店|分店|店)$", "", value)
    return re.sub(r"[\s·._-]", "", value)


def _semantic_venue_key(name: str) -> str:
    value = re.sub(r"[（(][^）)]*[）)]", "", str(name or ""))
    value = re.sub(r"^(?:成都|天津|武汉|西安|北京|上海)", "", value)
    value = re.sub(r"(?:景区|公园|博物馆|博物院|文化旅游区)$", "", value)
    # Provider parent/child records commonly expose an ancestral-hall plaza
    # and the hall itself as separate POIs. They are one itinerary stop unless
    # the user explicitly asks for a subvenue-specific artifact.
    value = re.sub(r"(?<=祠)(?:堂|广场)$", "", value)
    return re.sub(r"[\s·._-]", "", value)


def _dedupe_semantic_venues(
    items: list[ScoredPOI],
    *,
    required_terms: list[str] | None = None,
) -> list[ScoredPOI]:
    required_terms = required_terms or []
    indexed = list(enumerate(items))

    def request_priority(entry: tuple[int, ScoredPOI]) -> tuple[int, tuple[int, int], int]:
        index, item = entry
        priorities = [
            _named_venue_priority(item.poi, term)
            for term in required_terms
            if _matches_candidate_attraction(item.poi, term)
        ]
        return (
            0 if priorities else 1,
            min(priorities, default=(99, 99)),
            index,
        )

    selected: list[ScoredPOI] = []
    for _index, item in sorted(indexed, key=request_priority):
        if item.poi.category in {"food", "hotel", "transport"}:
            selected.append(item)
            continue
        key = _semantic_venue_key(item.poi.name)
        duplicate = next(
            (
                prior for prior in selected
                if (
                    (
                        (
                        item.poi.parent_poi_id
                        and prior.poi.source_poi_id
                        and item.poi.parent_poi_id == prior.poi.source_poi_id
                        )
                        or (
                        prior.poi.parent_poi_id
                        and item.poi.source_poi_id
                        and prior.poi.parent_poi_id == item.poi.source_poi_id
                        )
                        or (
                        item.poi.parent_poi_id
                        and prior.poi.parent_poi_id
                        and item.poi.parent_poi_id == prior.poi.parent_poi_id
                        )
                    )
                    and (
                        item.poi.category == prior.poi.category
                        or (
                            key
                            and _semantic_venue_key(prior.poi.name)
                            and (
                                key in _semantic_venue_key(prior.poi.name)
                                or _semantic_venue_key(prior.poi.name) in key
                            )
                        )
                    )
                    or (
                        key
                        and (
                            key == _semantic_venue_key(prior.poi.name)
                            or key in _semantic_venue_key(prior.poi.name)
                            or _semantic_venue_key(prior.poi.name) in key
                        )
                        and _haversine_km(
                            item.poi.lng, item.poi.lat, prior.poi.lng, prior.poi.lat
                        ) < 1.0
                    )
                )
            ),
            None,
        )
        if duplicate is None:
            selected.append(item)
    return selected


def _cluster_scored_pois_by_day(
    pois: list[ScoredPOI],
    days: int,
    max_per_day: int,
    profile: TravelProfile,
) -> list[list[ScoredPOI]]:
    """按地理位置把 POI 分到各天（借鉴 smart_plan_itinerary 的 K-means 风格聚类）。"""
    if not pois:
        return [[] for _ in range(days)]

    # Cluster activities first, then distribute reserved restaurants.  When
    # food was clustered together with attractions, nearby restaurants often
    # landed on one day and left the other days without any meal.
    food_pois = [item for item in pois if item.poi.category == "food"]
    activity_pois = [item for item in pois if item.poi.category != "food"]
    sorted_pois = sorted(activity_pois, key=lambda item: item.poi.lng)
    if not sorted_pois:
        groups = [[] for _ in range(days)]
        for index, item in enumerate(food_pois):
            groups[index % days].append(item)
        return groups
    chunk = max(1, len(sorted_pois) // days)
    groups: list[list[ScoredPOI]] = []
    for index in range(days):
        start = index * chunk
        end = start + chunk if index < days - 1 else len(sorted_pois)
        groups.append(sorted_pois[start:end])

    for _ in range(10):
        centers = [_centroid(group) if group else (0.0, 0.0) for group in groups]
        new_groups: list[list[ScoredPOI]] = [[] for _ in range(days)]
        for poi in sorted_pois:
            distances = [
                _haversine_km(poi.poi.lng, poi.poi.lat, center[0], center[1]) for center in centers
            ]
            best = distances.index(min(distances))
            new_groups[best].append(poi)
        if new_groups == groups:
            break
        groups = new_groups

    groups = _spread_hard_must_visit_anchors(groups, profile)

    # K-means is allowed to produce 1/2/3 activity splits, but a complete
    # multi-day itinerary should not leave a half-empty day when the selected,
    # grounded supply is already sufficient for two activities per day.
    minimum_activities = 2
    if max_per_day >= minimum_activities and len(sorted_pois) >= days * minimum_activities:
        state = profile.constraint_state or {}
        fixed_terms = [
            str(event.get("location") or "").strip()
            for event in (state.get("fixed_events") or [])
            if isinstance(event, dict)
            and str(event.get("location") or "").strip()
            not in {"自由活动", "休息", "自由时间"}
        ]
        hard_terms = [*(profile.must_visit or []), *fixed_terms]

        def hard_required(item: ScoredPOI) -> bool:
            return any(
                poi_matches_must_visit(item.poi, term)
                for term in hard_terms
            )

        for target_index in sorted(range(days), key=lambda index: len(groups[index])):
            while len(groups[target_index]) < minimum_activities:
                donors = [
                    index
                    for index in range(days)
                    if index != target_index and len(groups[index]) > minimum_activities
                ]
                if not donors:
                    break
                target_center = _centroid(groups[target_index]) if groups[target_index] else None
                rebalance_radius_km = {
                    "relaxed": 20.0,
                    "standard": 30.0,
                    "intensive": 40.0,
                }[profile.pace]
                movable = [
                    (source_index, item)
                    for source_index in donors
                    for item in groups[source_index]
                    if target_center is None
                    or _haversine_km(
                        item.poi.lng,
                        item.poi.lat,
                        target_center[0],
                        target_center[1],
                    ) <= rebalance_radius_km
                ]
                if not movable:
                    # A sparse but coherent remote day is preferable to
                    # manufacturing completeness with a cross-city detour.
                    break
                donor_index, moved = min(
                    movable,
                    key=lambda pair: (
                        hard_required(pair[1]),
                        _haversine_km(
                            pair[1].poi.lng,
                            pair[1].poi.lat,
                            target_center[0],
                            target_center[1],
                        ) if target_center is not None else -pair[1].score,
                    ),
                )
                groups[donor_index].remove(moved)
                groups[target_index].append(moved)

    remaining_food = list(food_pois)
    if len(remaining_food) >= days:
        # Reserve one slot in every day before assigning meals.  K-means may
        # otherwise fill one cluster to capacity and force two restaurants
        # onto another day even though the overall plan has enough room.
        activity_limit = max(0, max_per_day - 1)
        meal_rebalance_radius_km = {
            "relaxed": 20.0,
            "standard": 30.0,
            "intensive": 40.0,
        }[profile.pace]
        for source_index in range(days):
            while len(groups[source_index]) > activity_limit:
                targets = [
                    index
                    for index in range(days)
                    if index != source_index
                    and len(groups[index]) < activity_limit
                ]
                movable = next(
                    (
                        item for item in reversed(groups[source_index])
                        if not any(
                            poi_matches_must_visit(item.poi, term)
                            for term in profile.must_visit
                        )
                    ),
                    None,
                )
                if movable is None:
                    break
                coherent_targets = [
                    index for index in targets
                    if not groups[index]
                    or _haversine_km(
                        movable.poi.lng,
                        movable.poi.lat,
                        *_centroid(groups[index]),
                    ) <= meal_rebalance_radius_km
                ]
                groups[source_index].remove(movable)
                if coherent_targets:
                    target_index = min(
                        coherent_targets,
                        key=lambda index: (
                            _haversine_km(
                                movable.poi.lng,
                                movable.poi.lat,
                                *_centroid(groups[index]),
                            )
                            if groups[index]
                            else 0.0
                        ),
                    )
                    groups[target_index].append(movable)
                # When every under-filled day belongs to a remote area, leave
                # the optional activity out so each day can still receive a
                # nearby meal without cross-city backtracking.
    for index in sorted(range(days), key=lambda value: len(groups[value])):
        if not remaining_food or len(groups[index]) >= max_per_day:
            continue
        center = _centroid(groups[index]) if groups[index] else None
        best = min(
            remaining_food,
            key=lambda item: (
                _haversine_km(item.poi.lng, item.poi.lat, center[0], center[1])
                if center is not None
                else -item.score
            ),
        )
        groups[index].append(best)
        remaining_food.remove(best)

    for item in remaining_food:
        available = [index for index in range(days) if len(groups[index]) < max_per_day]
        if not available:
            break
        index = min(available, key=lambda value: len(groups[value]))
        groups[index].append(item)

    for index in range(days):
        while len(groups[index]) > max_per_day:
            overflow = groups[index].pop()
            emptiest = min(range(days), key=lambda day_index: len(groups[day_index]))
            groups[emptiest].append(overflow)

    return [_greedy_nearest(group) for group in groups]


def _spread_hard_must_visit_anchors(
    groups: list[list[ScoredPOI]],
    profile: TravelProfile,
) -> list[list[ScoredPOI]]:
    """Spread distinct hard must-visits across available trip days.

    Geographic clustering can still put two distant mandatory venues in one
    group during capacity balancing.  Exchange an extra mandatory venue with
    a non-mandatory activity from an unanchored day, preserving group sizes.
    """
    state = profile.constraint_state or {}
    unspecified = {
        normalize_entity_name(item)
        for item in state.get("user_owned_unspecified_fixed_event_locations") or []
        if normalize_entity_name(item)
    }
    fixed_locations = [
        str(event.get("location") or "").strip()
        for event in state.get("fixed_events") or []
        if isinstance(event, dict)
        and str(event.get("location") or "").strip()
        and normalize_entity_name(event.get("location")) not in unspecified
    ]
    required_terms = [*(profile.must_visit or []), *fixed_locations]
    if len(groups) < 2 or not required_terms:
        return groups

    def required(item: ScoredPOI) -> bool:
        return any(
            poi_matches_must_visit(item.poi, term)
            for term in required_terms
        )

    result = [list(group) for group in groups]
    for source_index, source in enumerate(result):
        anchors = [item for item in source if required(item)]
        for anchor in anchors[1:]:
            targets = [
                index
                for index, group in enumerate(result)
                if index != source_index and not any(required(item) for item in group)
            ]
            if not targets:
                break
            target_index = min(
                targets,
                key=lambda index: (
                    _haversine_km(
                        anchor.poi.lng,
                        anchor.poi.lat,
                        *_centroid(result[index]),
                    )
                    if result[index]
                    else 0.0
                ),
            )
            target = result[target_index]
            replacement = min(
                (item for item in target if not required(item)),
                key=lambda item: (
                    _haversine_km(
                        item.poi.lng,
                        item.poi.lat,
                        *_centroid(source),
                    )
                    if source
                    else 0.0
                ),
                default=None,
            )
            source.remove(anchor)
            target.append(anchor)
            if replacement is not None:
                target.remove(replacement)
                source.append(replacement)
    return result


def _stops_per_day(profile: TravelProfile) -> int:
    if profile.pace == "relaxed":
        return 2
    if profile.pace == "intensive":
        return 4
    return 3


def _build_day_stops(
    day_candidates: list[ScoredPOI],
    profile: TravelProfile,
    route_estimator: RouteEstimator | None,
) -> list[ItineraryStop]:
    stops: list[ItineraryStop] = []
    scheduled = _schedule_day_candidates(day_candidates)
    for item, start_time in scheduled:
        route_from_previous = None
        earliest_minutes = _time_to_minutes(start_time)
        if route_estimator and stops:
            try:
                route_from_previous = normalize_route_evidence(route_estimator.estimate_route(
                    stops[-1].poi,
                    item.poi,
                    profile.transport_mode,
                ))
                if not route_supports_endpoints(
                    route_from_previous, stops[-1].poi.poi_id, item.poi.poi_id
                ):
                    route_from_previous = None
            except Exception:  # 路线服务异常时使用可核验的直线距离估算，避免整单失败
                from travel_agent.tools import estimate_route_minutes

                distance_km, duration_min = estimate_route_minutes(
                    stops[-1].poi,
                    item.poi,
                    profile.transport_mode,
                )
                route_from_previous = normalize_route_evidence(RouteInfo(
                    origin_poi_id=stops[-1].poi.poi_id,
                    destination_poi_id=item.poi.poi_id,
                    distance_km=distance_km,
                    duration_min=duration_min,
                    mode=profile.transport_mode,
                    source="haversine_recovery_estimate",
                ), provider_explicit=False)
        if stops:
            previous = stops[-1]
            previous_end = (
                _time_to_minutes(previous.start_time)
                + previous.duration_min
                + (route_from_previous.duration_min if route_from_previous else 0)
                + TRANSFER_BUFFER_MIN
            )
            earliest_minutes = max(earliest_minutes, _round_up_to_quarter(previous_end))
        if item.poi.category == "food":
            earliest_minutes = _next_meal_time_minutes(earliest_minutes)
        elif _time_to_minutes("11:30") <= earliest_minutes < _time_to_minutes("13:00"):
            earliest_minutes = _time_to_minutes("13:00")
        actual_start_time = _minutes_to_time(earliest_minutes)
        stops.append(
            ItineraryStop(
                poi=item.poi,
                start_time=actual_start_time,
                duration_min=item.poi.estimated_duration_min,
                note=_make_stop_note(item),
                route_from_previous=route_from_previous,
            )
        )
    return stops


def _schedule_day_candidates(day_candidates: list[ScoredPOI]) -> list[tuple[ScoredPOI, str]]:
    """按时间语义安排一天内 POI：餐饮放午/晚餐，景点放上午/下午。"""
    food_items = [item for item in day_candidates if item.poi.category == "food"]
    activity_items = [item for item in day_candidates if item.poi.category != "food"]

    # If the day's only activity explicitly opens in the evening, dining at
    # 11:30 creates an empty day and normalisation can push it past 19:00.
    # Put the meal immediately before the evening venue instead.
    if (
        food_items
        and activity_items
        and all(_is_evening_only(item.poi) for item in activity_items)
    ):
        scheduled = [(food_items[0], "17:30")]
        scheduled.extend((item, "18:00") for item in activity_items)
        scheduled.extend((item, "19:00") for item in food_items[1:])
        return scheduled

    scheduled: list[tuple[ScoredPOI, str]] = []
    activity_index = 0
    food_index = 0

    for item in activity_items[:1]:
        scheduled.append((
            item,
            _minutes_to_time(_preferred_activity_start_minutes(
                item.poi, _time_to_minutes(ACTIVITY_TIMES[activity_index])
            )),
        ))
        activity_index += 1

    if food_items:
        scheduled.append((food_items[0], MEAL_TIMES[food_index]))
        food_index += 1

    for item in activity_items[1:]:
        time = ACTIVITY_TIMES[min(activity_index, len(ACTIVITY_TIMES) - 1)]
        scheduled.append((
            item,
            _minutes_to_time(_preferred_activity_start_minutes(
                item.poi, _time_to_minutes(time)
            )),
        ))
        activity_index += 1

    for item in food_items[1:]:
        time = MEAL_TIMES[min(food_index, len(MEAL_TIMES) - 1)]
        scheduled.append((item, time))
        food_index += 1

    if not scheduled:
        return []
    return sorted(scheduled, key=lambda pair: _time_sort_key(pair[1]))


def _is_evening_only(poi: POI) -> bool:
    window = _daily_opening_window(poi.opening_hours)
    return window is not None and window[0] >= _time_to_minutes("17:00")


def _preferred_activity_start_minutes(poi: POI, default: int) -> int:
    """Respect explicit experience semantics even when a venue is open all day."""
    identity = " ".join(
        str(value or "")
        for value in (poi.name, poi.canonical_name, *poi.aliases, *poi.tags)
    ).casefold()
    night_markers = (
        "夜景", "灯光秀", "灯光表演", "夜游", "不夜城", "夜市",
        "light show", "night view", "night cruise", "night market",
    )
    if any(marker in identity for marker in night_markers):
        return max(default, _time_to_minutes("18:00"))
    return default


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
    for slot in MEAL_TIME_WINDOWS:
        minutes = _time_to_minutes(slot)
        if minutes >= earliest:
            return minutes
    return _time_to_minutes(MEAL_TIME_WINDOWS[-1])


def _round_up_to_quarter(value: int) -> int:
    remainder = value % 15
    if remainder == 0:
        return value
    return value + 15 - remainder


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
