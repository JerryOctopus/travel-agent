from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Protocol

from travel_agent.critic import poi_complies_with_dietary, poi_matches_must_visit
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
MEAL_TIMES = ["11:30", "18:00"]
MEAL_TIME_WINDOWS = ["11:30", "12:00", "12:30", "13:00", "17:30", "18:00", "19:00"]
TRANSFER_BUFFER_MIN = 15


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
        if not any(
            _matches_candidate_attraction(item.poi, str(term))
            and not _is_suitable_for_requested_weekday(item.poi, state.get("weekday"))
            for term in candidate_terms
        )
    ]
    eligible = _dedupe_semantic_venues(eligible)
    # Hard must-visits own capacity before completeness preferences.  Previously
    # meals were reserved first, so a small one-day plan could consume its last
    # slot with food and silently drop an evidenced must-visit.
    if preserve_must_visit_capacity:
        for term in profile.must_visit:
            match = next(
                (
                    item
                    for item in eligible
                    if poi_matches_must_visit(item.poi, term)
                ),
                None,
            )
            if match is not None:
                add(match)

    # Named places that the user explicitly asked us to verify are softer than
    # must-visits, but they still take precedence over unrelated recommendations.
    # Only reserve a place when the bound evidence represents the venue itself
    # and does not explicitly say it is closed on the requested weekday.
    for term in candidate_terms:
        match = next(
            (
                item
                for item in eligible
                if _matches_candidate_attraction(item.poi, str(term))
                and _is_suitable_for_requested_weekday(item.poi, state.get("weekday"))
            ),
            None,
        )
        if match is not None:
            add(match)

    # A deliverable day plan needs at least one evidenced meal when restaurant
    # candidates exist.  This is schedule completeness, not an inferred food
    # preference; dietary filtering above still fails closed.
    selected_food_brands: set[str] = set()
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
    for item in food_candidates:
        if len(selected) >= total_stops:
            break
        brand = _food_brand_key(item.poi.name)
        if brand in selected_food_brands:
            continue
        add(item)
        selected_food_brands.add(brand)
        if sum(entry.poi.category == "food" for entry in selected) >= profile.days:
            break

    # If the provider only returned one brand, completeness is preferable to
    # omitting meals.  Duplicate branches remain visible to the critic.
    for item in food_candidates:
        if len(selected) >= total_stops or sum(
            entry.poi.category == "food" for entry in selected
        ) >= profile.days:
            break
        add(item)

    # Sparse offline/local catalogs may contain fewer distinct restaurants
    # than trip days. Reusing a grounded restaurant on another day is truthful
    # and preferable to either inventing a venue or leaving an explicit food
    # itinerary without a meal. Live catalogs normally satisfy the distinct
    # candidate path above.
    food_count = sum(entry.poi.category == "food" for entry in selected)
    repeat_index = 0
    while food_candidates and food_count < profile.days and len(selected) < total_stops:
        selected.append(food_candidates[repeat_index % len(food_candidates)])
        repeat_index += 1
        food_count += 1

    # Fill remaining capacity with activities before optional extra meals.  A
    # distant second restaurant must not displace a viable attraction after
    # the daily evidenced-meal requirement has already been satisfied.
    fill_order = [item for item in eligible if item.poi.category != "food"] + [
        item for item in eligible if item.poi.category == "food"
    ]
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
    raw_events = state.get("fixed_events") or []
    candidate_terms = state.get("candidate_attractions") or []
    if isinstance(candidate_terms, str):
        candidate_terms = [candidate_terms]
    events_by_day: dict[int, list[tuple[int, int, ScoredPOI]]] = {}
    reserved_windows_by_day: dict[int, list[tuple[int, int]]] = {}
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
        match = next(
            (
                item
                for item in ranked_pois
                if (
                    location in item.poi.name
                    or item.poi.name in location
                    or location in str(item.poi.address or "")
                )
            ),
            None,
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
            opening_window = _daily_opening_window(stop.poi.opening_hours)
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
                    _time_to_minutes(ACTIVITY_TIMES[min(index, len(ACTIVITY_TIMES) - 1)]),
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
            required_flexible.sort(
                key=lambda entry: (
                    (_daily_opening_window(entry[2].poi.opening_hours) or (0, 24 * 60))[1],
                    entry[0],
                )
            )
            retimed_required = {
                entry[2].poi.poi_id: _time_to_minutes(
                    ACTIVITY_TIMES[min(index, len(ACTIVITY_TIMES) - 1)]
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
            duration = (fixed_end - requested_start) if is_fixed and fixed_end else stop.duration_min
            if not is_fixed and any(
                actual_start < reserved_end and actual_start + duration > reserved_start
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
            opening_window = _daily_opening_window(stop.poi.opening_hours)
            if opening_window is not None:
                opens, closes = opening_window
                actual_start = max(actual_start, opens)
                if actual_start + duration > closes:
                    # A verified named candidate should displace an unrelated
                    # earlier optional stop instead of silently disappearing.
                    is_named_candidate = any(
                        _matches_candidate_attraction(stop.poi, str(term))
                        for term in candidate_terms
                    )
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
                    if actual_start + duration > closes:
                        continue
            scheduled.append(
                ItineraryStop(
                    poi=stop.poi,
                    start_time=_minutes_to_time(actual_start),
                    duration_min=duration,
                    note=stop.note,
                    route_from_previous=route,
                )
            )
        constrained_days.append(
            ItineraryDay(
                day_index=day.day_index,
                theme=_make_day_theme(scheduled),
                stops=scheduled,
            )
        )
    return constrained_days


def _daily_opening_window(value: str | None) -> tuple[int, int] | None:
    """Return a simple same-day opening window when the evidence is explicit."""
    text = str(value or "").strip()
    if not text or text.lower() in {"all_day", "24h"} or "24:00" in text:
        return None
    match = re.search(r"(\d{1,2}):(\d{2})\s*[-至到]\s*(\d{1,2}):(\d{2})", text)
    if match is None:
        return None
    opens = int(match.group(1)) * 60 + int(match.group(2))
    closes = int(match.group(3)) * 60 + int(match.group(4))
    return (opens, closes) if closes > opens else None


def _matches_candidate_attraction(poi: POI, term: str) -> bool:
    name = str(poi.name or "").strip()
    target = str(term or "").strip()
    if not name or not target or poi.category in {"food", "hotel", "transport"}:
        return False
    if any(marker in name for marker in ("服务中心", "游客中心", "停车场", "售票处")):
        return False
    return target in name or name in target


def _is_suitable_for_requested_weekday(poi: POI, weekday: object) -> bool:
    requested = str(weekday or "").strip()
    hours = str(poi.opening_hours or "").replace(" ", "")
    if not requested or not hours:
        return True
    return not (
        f"{requested}全天不开放" in hours
        or f"{requested}闭馆" in hours
        or (requested == "周一" and "周一闭馆" in hours)
    )


def poi_open_on_trip_day(poi: POI, profile: TravelProfile, day_index: int) -> bool:
    if not profile.start_date:
        return True
    try:
        current = date.fromisoformat(str(profile.start_date)) + timedelta(days=day_index - 1)
    except (TypeError, ValueError):
        return True
    weekday = "周" + "一二三四五六日"[current.weekday()]
    return _is_suitable_for_requested_weekday(poi, weekday)


def _estimate_stop_route(
    origin: POI | None,
    destination: POI,
    profile: TravelProfile,
    route_estimator: RouteEstimator | None,
) -> RouteInfo | None:
    if origin is None or route_estimator is None:
        return None
    try:
        return route_estimator.estimate_route(origin, destination, profile.transport_mode)
    except Exception:
        from travel_agent.tools import estimate_route_minutes

        distance_km, duration_min = estimate_route_minutes(
            origin,
            destination,
            profile.transport_mode,
        )
        return RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=distance_km,
            duration_min=duration_min,
            mode=profile.transport_mode,
            source="haversine_recovery_estimate",
        )


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
    return re.sub(r"[\s·._-]", "", value)


def _dedupe_semantic_venues(items: list[ScoredPOI]) -> list[ScoredPOI]:
    selected: list[ScoredPOI] = []
    for item in items:
        if item.poi.category in {"food", "hotel", "transport"}:
            selected.append(item)
            continue
        key = _semantic_venue_key(item.poi.name)
        duplicate = next(
            (
                prior for prior in selected
                if key and (
                    key == _semantic_venue_key(prior.poi.name)
                    or key in _semantic_venue_key(prior.poi.name)
                    or _semantic_venue_key(prior.poi.name) in key
                )
                and _haversine_km(
                    item.poi.lng, item.poi.lat, prior.poi.lng, prior.poi.lat
                ) < 1.0
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

    remaining_food = list(food_pois)
    if len(remaining_food) >= days:
        # Reserve one slot in every day before assigning meals.  K-means may
        # otherwise fill one cluster to capacity and force two restaurants
        # onto another day even though the overall plan has enough room.
        activity_limit = max(0, max_per_day - 1)
        for source_index in range(days):
            while len(groups[source_index]) > activity_limit:
                targets = [
                    index
                    for index in range(days)
                    if index != source_index and len(groups[index]) < activity_limit
                ]
                if not targets:
                    break
                moved = groups[source_index].pop()
                target_index = min(
                    targets,
                    key=lambda index: (
                        _haversine_km(
                            moved.poi.lng,
                            moved.poi.lat,
                            *_centroid(groups[index]),
                        )
                        if groups[index]
                        else 0.0
                    ),
                )
                groups[target_index].append(moved)
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
    if len(groups) < 2 or not profile.must_visit:
        return groups

    def required(item: ScoredPOI) -> bool:
        return any(
            poi_matches_must_visit(item.poi, term)
            for term in profile.must_visit
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
        return 3 if int(profile.days or 1) > 1 else 2
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
                route_from_previous = route_estimator.estimate_route(
                    stops[-1].poi,
                    item.poi,
                    profile.transport_mode,
                )
            except Exception:  # 路线服务异常时使用可核验的直线距离估算，避免整单失败
                from travel_agent.tools import estimate_route_minutes

                distance_km, duration_min = estimate_route_minutes(
                    stops[-1].poi,
                    item.poi,
                    profile.transport_mode,
                )
                route_from_previous = RouteInfo(
                    origin_poi_id=stops[-1].poi.poi_id,
                    destination_poi_id=item.poi.poi_id,
                    distance_km=distance_km,
                    duration_min=duration_min,
                    mode=profile.transport_mode,
                    source="haversine_recovery_estimate",
                )
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
        scheduled.append((item, ACTIVITY_TIMES[activity_index]))
        activity_index += 1

    if food_items:
        scheduled.append((food_items[0], MEAL_TIMES[food_index]))
        food_index += 1

    for item in activity_items[1:]:
        time = ACTIVITY_TIMES[min(activity_index, len(ACTIVITY_TIMES) - 1)]
        scheduled.append((item, time))
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


def _time_sort_key(value: str) -> tuple[int, int]:
    hour, minute = value.split(":", 1)
    return int(hour), int(minute)


def _time_to_minutes(value: str) -> int:
    hour, minute = _time_sort_key(value)
    return hour * 60 + minute


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
