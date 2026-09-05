"""Deterministic, replayable multi-turn constraint state.

The public constraint state remains a plain mapping for planner compatibility,
while ``_constraint_events`` is its source of truth.  Replaying the log always
produces the same active state and preserves remove tombstones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import json
from typing import Any, Literal


Operation = Literal["set", "add", "remove", "replace", "confirm"]

SINGLE_VALUE_FIELDS = frozenset({
    "duration_days", "referenced_day_index", "traveler_count",
    "budget_max_cny", "prepaid_cost", "lodging_area", "return_deadline",
    "date_start", "date_end", "destination", "origin", "transport_mode",
    "self_driving_allowed", "public_transport_required", "mobility",
    "return_location", "return_day_index", "activity_end_deadline",
    "activity_end_target",
    "return_deadline_local_time", "return_deadline_day",
})
SET_VALUE_FIELDS = frozenset({
    "must_visit", "candidate_attractions", "removed", "interests", "dietary",
    "transport_modes", "avoid", "exclude", "optional_remove",
})
EVENT_LIST_FIELDS = frozenset({"fixed_events"})
META_KEY = "_constraint_events"


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _identity(field: str, value: Any) -> str:
    if field == "fixed_events" and isinstance(value, dict):
        return _stable({key: value.get(key) for key in ("day", "date", "start", "end", "location")})
    return _stable(value)


def _event_id(field: str, operation: str, value: Any, source_turn: int) -> str:
    raw = f"{source_turn}|{field}|{operation}|{_identity(field, value)}"
    return "ce_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ConstraintEvent:
    field: str
    operation: Operation
    value: Any
    source_turn: int
    explicitness: Literal["explicit", "soft", "inferred"] = "explicit"
    confidence: float = 1.0
    event_id: str = ""
    supersedes_event_id: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = _event_id(self.field, self.operation, self.value, self.source_turn)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_event(raw: dict[str, Any]) -> ConstraintEvent | None:
    try:
        operation = str(raw["operation"])
        if operation not in {"set", "add", "remove", "replace", "confirm"}:
            return None
        return ConstraintEvent(
            field=str(raw["field"]), operation=operation, value=raw.get("value"),
            source_turn=int(raw.get("source_turn") or 0),
            explicitness=str(raw.get("explicitness") or "explicit"),
            confidence=float(raw.get("confidence", 1.0)),
            event_id=str(raw.get("event_id") or ""),
            supersedes_event_id=raw.get("supersedes_event_id"),
            active=bool(raw.get("active", True)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _is_soft(explicitness: str) -> bool:
    return explicitness in {"soft", "inferred"}


def replay_events(raw_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild active constraint state using scalar/set/list-specific semantics."""
    events = [event for raw in raw_events if isinstance(raw, dict) if (event := _as_event(raw))]
    scalars: dict[str, ConstraintEvent] = {}
    sets: dict[str, dict[str, tuple[Any, ConstraintEvent]]] = {}
    tombstones: dict[str, dict[str, ConstraintEvent]] = {}

    for event in events:
        if not event.active:
            continue
        field = event.field
        ident = _identity(field, event.value)
        if event.operation == "confirm":
            # Confirmation is deliberately a no-op unless a matching value exists.
            continue
        if field not in SET_VALUE_FIELDS | EVENT_LIST_FIELDS:
            previous = scalars.get(field)
            if event.operation == "remove":
                if previous is not None:
                    scalars.pop(field, None)
                continue
            if previous is not None and _is_soft(event.explicitness) and previous.explicitness == "explicit":
                continue
            scalars[field] = event
            continue
        bucket = sets.setdefault(field, {})
        removed = tombstones.setdefault(field, {})
        if event.operation == "remove":
            bucket.pop(ident, None)
            removed[ident] = event
        elif event.operation == "replace":
            bucket.clear()
            removed.clear()
            bucket[ident] = (event.value, event)
        else:
            tombstone = removed.get(ident)
            if tombstone is not None and _is_soft(event.explicitness):
                continue
            if tombstone is not None:
                removed.pop(ident, None)
            bucket.setdefault(ident, (event.value, event))

    state: dict[str, Any] = {field: event.value for field, event in scalars.items()}
    for field, values in sets.items():
        if values:
            state[field] = [item[0] for item in values.values()]
    # Removed is both an ordinary active set and a cross-field tombstone.
    removed_values = state.get("removed") or []
    if removed_values:
        for field in ("must_visit", "candidate_attractions"):
            state[field] = [
                value for value in state.get(field, [])
                if not any(_entity_matches(value, removed) for removed in removed_values)
            ]
            if not state[field]:
                state.pop(field, None)
    return state


def _entity_matches(left: Any, right: Any) -> bool:
    a, b = str(left).strip(), str(right).strip()
    return bool(a and b and (a == b or a in b or b in a))


def merge_constraint_update(
    current: dict[str, Any], update: dict[str, Any], *, source_turn: int,
    explicitness: str = "explicit", confidence: float = 1.0,
    confirmations: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append idempotent events and replace ``current`` with replayed active state."""
    raw_events = list(current.get(META_KEY) or [])
    version_meta = {
        key: value for key, value in current.items()
        if key.startswith("_") and key != META_KEY
    }
    if not raw_events:
        # Migrate legacy state as turn-zero facts without inventing provenance.
        for field, value in current.items():
            if field.startswith("_") or value in (None, "", [], {}):
                continue
            values = value if field in SET_VALUE_FIELDS | EVENT_LIST_FIELDS and isinstance(value, list) else [value]
            op: Operation = "add" if field in SET_VALUE_FIELDS | EVENT_LIST_FIELDS else "set"
            raw_events.extend(ConstraintEvent(field, op, item, 0, "inferred", 0.5).to_dict() for item in values)

    active_before = replay_events(raw_events)
    new_events: list[ConstraintEvent] = []
    for field, value in update.items():
        if field.startswith("_") or value is None:
            continue
        if field == "removed":
            for item in value if isinstance(value, list) else [value]:
                new_events.append(ConstraintEvent("removed", "add", item, source_turn, explicitness, confidence))
                for target in ("must_visit", "candidate_attractions"):
                    new_events.append(ConstraintEvent(target, "remove", item, source_turn, explicitness, confidence))
            continue
        if field in SET_VALUE_FIELDS | EVENT_LIST_FIELDS:
            for item in value if isinstance(value, list) else [value]:
                if field in {"must_visit", "candidate_attractions"} and explicitness == "explicit":
                    # An explicit re-add is a deliberate override of the old
                    # tombstone; soft/inferred mentions never reach this path.
                    for raw in raw_events:
                        if (
                            raw.get("field") == "removed"
                            and raw.get("active", True)
                            and _entity_matches(raw.get("value"), item)
                        ):
                            raw["active"] = False
                new_events.append(ConstraintEvent(field, "add", item, source_turn, explicitness, confidence))
            continue
        operation: Operation = "replace" if field in active_before and active_before.get(field) != value else "set"
        new_events.append(ConstraintEvent(field, operation, value, source_turn, explicitness, confidence))
    for field, value in (confirmations or {}).items():
        if active_before.get(field) == value:
            new_events.append(ConstraintEvent(field, "confirm", value, source_turn, "explicit", confidence))

    existing_ids = {str(item.get("event_id")) for item in raw_events if isinstance(item, dict)}
    for event in new_events:
        if event.event_id in existing_ids:
            continue
        if event.field not in SET_VALUE_FIELDS | EVENT_LIST_FIELDS and event.operation in {"set", "replace"}:
            # A soft mention cannot supersede an explicit active scalar.
            previous_active = next((raw for raw in reversed(raw_events) if raw.get("field") == event.field and raw.get("active", True) and raw.get("operation") != "confirm"), None)
            if previous_active is not None and _is_soft(event.explicitness) and previous_active.get("explicitness") == "explicit":
                continue
            for raw in reversed(raw_events):
                if raw.get("field") == event.field and raw.get("active", True) and raw.get("operation") != "confirm":
                    raw["active"] = False
                    event.supersedes_event_id = raw.get("event_id")
                    break
        raw_events.append(event.to_dict())
        existing_ids.add(event.event_id)

    rebuilt = replay_events(raw_events)
    current.clear()
    current.update(rebuilt)
    current[META_KEY] = raw_events
    current.update(version_meta)
    return [event.to_dict() for event in new_events if event.event_id in existing_ids]


def validate_constraint_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return concrete deterministic conflicts; an empty list means Planner-safe."""
    conflicts: list[dict[str, Any]] = []
    days = state.get("duration_days")
    start, end = state.get("date_start"), state.get("date_end")
    if days and start and end:
        try:
            ranged = (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days + 1
            if int(days) != ranged:
                conflicts.append({"code": "duration_date_range_mismatch", "field": "duration_days", "values": [days, ranged]})
        except (TypeError, ValueError):
            conflicts.append({"code": "invalid_date_range", "field": "date_start/date_end", "values": [start, end]})
    for event in state.get("fixed_events") or []:
        if isinstance(event, dict) and days and event.get("day"):
            try:
                if int(event["day"]) > int(days):
                    conflicts.append({"code": "fixed_event_out_of_range", "field": "fixed_events", "values": [event, days]})
            except (TypeError, ValueError):
                conflicts.append({"code": "invalid_fixed_event_day", "field": "fixed_events", "values": [event]})
    for must in state.get("must_visit") or []:
        for removed in state.get("removed") or []:
            if _entity_matches(must, removed):
                conflicts.append({"code": "must_visit_removed_conflict", "field": "must_visit/removed", "values": [must, removed]})
    events = state.get(META_KEY) or []
    scalar_fields = {
        str(item.get("field")) for item in events
        if item.get("field") not in SET_VALUE_FIELDS | EVENT_LIST_FIELDS
    }
    for field in scalar_fields:
        active = [item for item in events if item.get("field") == field and item.get("active", True) and item.get("operation") != "confirm"]
        if len(active) > 1:
            conflicts.append({"code": "multiple_active_scalar_values", "field": field, "event_ids": [item.get("event_id") for item in active]})
    for field, value in state.items():
        if field.startswith("_") or value in (None, "", [], {}):
            continue
        if not any(item.get("field") == field and item.get("active", True) for item in events):
            conflicts.append({"code": "missing_source_event", "field": field})
    deadline = state.get("return_deadline")
    relative_last_day = bool(
        state.get("return_deadline_day") == "last_day"
        and state.get("return_deadline_local_time")
        and str(deadline) == str(state.get("return_deadline_local_time"))
    )
    if deadline and not _is_complete_timestamp(deadline) and not relative_last_day:
        conflicts.append({"code": "incomplete_return_deadline", "field": "return_deadline", "values": [deadline]})
    return conflicts


def _is_complete_timestamp(value: Any) -> bool:
    try:
        text = str(value).strip()
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return ("T" in text or " " in text) and parsed.tzinfo is not None
    except (TypeError, ValueError):
        return False


def normalize_return_deadline(
    state: dict[str, Any], *, start_date: str | None = None,
    duration_days: int | None = None, reference_datetime: str | None = None,
) -> str | None:
    """Resolve a return clock time to the trip's final dated, zoned timestamp."""
    raw = state.get("return_deadline")
    if raw in (None, "") and state.get("return_deadline_day") == "last_day":
        raw = state.get("return_deadline_local_time")
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    from travel_agent.datetime_semantics import normalize_datetime_value

    runtime_timezone = _session_timezone(reference_datetime)
    semantic_context = {
        **state,
        "date_start": state.get("date_start") or start_date,
        "duration_days": state.get("duration_days") or duration_days,
        "reference_datetime": reference_datetime,
        "timezone": runtime_timezone,
    }
    semantic = normalize_datetime_value(
        raw,
        context=semantic_context,
        prefer_trip_end=True,
    )
    state.setdefault("_datetime_normalizations", {})["return_deadline"] = semantic.to_dict()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if "T" in text or " " in text:
            zone = parsed.tzinfo or runtime_timezone
            if zone is None:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=zone)
            normalized = parsed.isoformat(timespec="minutes")
            state["return_deadline"] = normalized
            return normalized
    except ValueError:
        pass
    try:
        clock = time.fromisoformat(text)
    except ValueError:
        return None
    final_date = _trip_final_date(state, start_date=start_date, duration_days=duration_days)
    if final_date is None:
        # Preserve exactly what the user supplied without inventing a calendar
        # date. Planner/Reviewer compare this as an explicit last-day wall-clock
        # constraint; calendar-dependent facts remain unresolved separately.
        local_time = clock.isoformat(timespec="minutes")
        state["return_deadline"] = local_time
        state["return_deadline_local_time"] = local_time
        state["return_deadline_day"] = "last_day"
        return None
    zone = runtime_timezone
    if zone is None:
        local_time = clock.isoformat(timespec="minutes")
        state["return_deadline"] = local_time
        state["return_deadline_local_time"] = local_time
        state["return_deadline_day"] = "last_day"
        return None
    resolved = datetime.combine(final_date, clock, tzinfo=zone)
    normalized = resolved.isoformat(timespec="minutes")
    state["return_deadline"] = normalized
    return normalized


def clock_minutes(value: Any) -> int:
    """Read HH:MM or an ISO timestamp without discarding its calendar identity."""
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if "T" in text or " " in text:
            return parsed.hour * 60 + parsed.minute
    except ValueError:
        pass
    parsed_time = time.fromisoformat(text)
    return parsed_time.hour * 60 + parsed_time.minute


def _trip_final_date(
    state: dict[str, Any], *, start_date: str | None, duration_days: int | None,
) -> date | None:
    if state.get("date_end"):
        try:
            return date.fromisoformat(str(state["date_end"]))
        except ValueError:
            return None
    raw_start = state.get("date_start") or start_date
    raw_days = state.get("duration_days") or duration_days
    if raw_start and raw_days:
        try:
            return date.fromisoformat(str(raw_start)) + timedelta(days=int(raw_days) - 1)
        except (TypeError, ValueError):
            return None
    return None


def _session_timezone(reference_datetime: str | None):
    if reference_datetime:
        try:
            parsed = datetime.fromisoformat(str(reference_datetime).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.tzinfo
        except ValueError:
            pass
    # Runtime deployment timezone is deterministic process context. Evaluator
    # comparison never uses this fallback; it must receive case/constraint
    # timezone evidence explicitly or from a complete counterpart timestamp.
    return datetime.now().astimezone().tzinfo
