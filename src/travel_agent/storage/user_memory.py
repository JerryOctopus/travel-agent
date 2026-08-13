"""L3 长期记忆：稳定偏好证据与最近已完成行程。"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from travel_agent.schemas import TravelProfile

if TYPE_CHECKING:
    from travel_agent.settings import MemorySettings

logger = logging.getLogger(__name__)

Polarity = Literal["positive", "negative"]
MULTI_CATEGORIES = {"interests", "food_preference", "avoid"}
SINGLE_CATEGORIES = {"budget_level", "pace", "transport_mode"}
STABLE_CATEGORIES = MULTI_CATEGORIES | SINGLE_CATEGORIES
MAX_ACTIVE_VALUES = 20
MAX_PREFERENCE_EVENTS = 200
MAX_RECENT_TRIPS = 20
RETENTION_SECONDS = 2 * 365 * 24 * 60 * 60


@dataclass(frozen=True)
class PreferenceObservation:
    category: str
    value: str
    polarity: Polarity = "positive"


@dataclass
class PreferenceEvent:
    event_id: str
    session_id: str
    turn_index: int
    category: str
    value: str
    polarity: Polarity
    observed_at: float


@dataclass
class PreferenceEvidence:
    category: str
    value: str
    positive_count: int = 0
    negative_count: int = 0
    active: bool = False
    last_polarity: Polarity = "positive"
    last_seen_at: float = 0.0


@dataclass
class RecentTrip:
    session_id: str
    destination: str | None = None
    days: int | None = None
    start_date: str | None = None
    companions: str | None = None
    budget_level: str | None = None
    interests: list[str] = field(default_factory=list)
    pace: str = "standard"
    hotel_area: str | None = None
    food_preference: list[str] = field(default_factory=list)
    must_visit: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    transport_mode: str = "public_transport"
    itinerary_summary: str = ""
    poi_names: list[str] = field(default_factory=list)
    critic_passed: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class UserMemory:
    user_id: str
    schema_version: int = 2
    created_at: float = 0.0
    updated_at: float = 0.0
    preference_events: list[PreferenceEvent] = field(default_factory=list)
    preference_evidence: list[PreferenceEvidence] = field(default_factory=list)
    recent_trips: list[RecentTrip] = field(default_factory=list)


class UserMemoryRepository(Protocol):
    def load_memory(self, user_id: str) -> UserMemory: ...
    def record_preference_events(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        observations: list[PreferenceObservation],
        observed_at: float | None = None,
    ) -> UserMemory: ...
    def upsert_completed_trip(self, user_id: str, trip: RecentTrip) -> UserMemory: ...
    def list_recent_trips(self, user_id: str, limit: int = 10) -> list[RecentTrip]: ...
    def forget_preference(self, user_id: str, category: str, value: str | None = None) -> None: ...
    def clear_preferences(self, user_id: str) -> None: ...
    def delete_user_memory(self, user_id: str) -> None: ...
    def close(self) -> None: ...


class JsonUserMemoryRepository:
    """本地开发 adapter；兼容读取旧版 TravelProfile JSON。"""

    def __init__(self, profile_dir: Path | str) -> None:
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user_id)
        return self.profile_dir / f"{safe}.json"

    def load_memory(self, user_id: str) -> UserMemory:
        path = self._path(user_id)
        if not path.exists():
            return _empty_memory(user_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") == 2:
                return _memory_from_dict(data)
            return _legacy_memory_from_dict(user_id, data)
        except Exception as exc:
            logger.warning("failed to load user memory user_id=%s error=%s", user_id, exc)
            return _empty_memory(user_id)

    def record_preference_events(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        observations: list[PreferenceObservation],
        observed_at: float | None = None,
    ) -> UserMemory:
        memory = self.load_memory(user_id)
        now = observed_at or time.time()
        existing = {event.event_id for event in memory.preference_events}
        for observation in _normalize_observations(observations):
            event_id = preference_event_id(user_id, session_id, turn_index, observation)
            if event_id in existing:
                continue
            memory.preference_events.append(
                PreferenceEvent(
                    event_id=event_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    category=observation.category,
                    value=observation.value,
                    polarity=observation.polarity,
                    observed_at=now,
                )
            )
            existing.add(event_id)
        _prune_and_rebuild(memory, now)
        self._save(memory)
        return memory

    def upsert_completed_trip(self, user_id: str, trip: RecentTrip) -> UserMemory:
        memory = self.load_memory(user_id)
        now = trip.updated_at or time.time()
        previous = next((item for item in memory.recent_trips if item.session_id == trip.session_id), None)
        trip.created_at = previous.created_at if previous else (trip.created_at or now)
        trip.updated_at = now
        trip.poi_names = _unique(trip.poi_names)[:30]
        memory.recent_trips = [item for item in memory.recent_trips if item.session_id != trip.session_id]
        memory.recent_trips.append(trip)
        _prune_and_rebuild(memory, now)
        self._save(memory)
        return memory

    def list_recent_trips(self, user_id: str, limit: int = 10) -> list[RecentTrip]:
        trips = sorted(self.load_memory(user_id).recent_trips, key=lambda item: item.updated_at, reverse=True)
        return trips[: max(0, limit)]

    def forget_preference(self, user_id: str, category: str, value: str | None = None) -> None:
        memory = self.load_memory(user_id)
        memory.preference_events = [
            event
            for event in memory.preference_events
            if not (event.category == category and (value is None or event.value == value))
        ]
        _prune_and_rebuild(memory, time.time())
        self._save(memory)
        logger.info(
            "forgot user preference user_id=%s category=%s has_value=%s",
            user_id,
            category,
            value is not None,
        )

    def clear_preferences(self, user_id: str) -> None:
        memory = self.load_memory(user_id)
        memory.preference_events = []
        memory.preference_evidence = []
        memory.updated_at = time.time()
        self._save(memory)
        logger.info("cleared user preferences user_id=%s", user_id)

    def delete_user_memory(self, user_id: str) -> None:
        path = self._path(user_id)
        if path.exists():
            path.unlink()
        logger.info("deleted user memory user_id=%s", user_id)

    def close(self) -> None:
        return None

    def _save(self, memory: UserMemory) -> None:
        path = self._path(memory.user_id)
        memory.updated_at = memory.updated_at or time.time()
        payload = json.dumps(_memory_to_dict(memory), ensure_ascii=False, indent=2)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(path)


class UserMemoryService:
    def __init__(self, repository: UserMemoryRepository) -> None:
        self.repository = repository

    def load_stable_profile(self, user_id: str) -> TravelProfile:
        return stable_profile_from_memory(self.repository.load_memory(user_id))

    def record_preference_events(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        observations: list[PreferenceObservation],
    ) -> UserMemory:
        return self.repository.record_preference_events(
            user_id, session_id, turn_index, observations
        )

    def upsert_completed_trip(
        self,
        user_id: str,
        session_id: str,
        profile: TravelProfile,
        itinerary: dict,
    ) -> UserMemory:
        now = time.time()
        poi_names: list[str] = []
        for day in itinerary.get("days", []):
            for stop in day.get("stops", []):
                poi = stop.get("poi", stop)
                name = poi.get("name") if isinstance(poi, dict) else None
                if name:
                    poi_names.append(str(name))
        trip = RecentTrip(
            session_id=session_id,
            destination=profile.destination,
            days=profile.days,
            start_date=profile.start_date,
            companions=profile.companions,
            budget_level=profile.budget_level,
            interests=list(profile.interests),
            pace=profile.pace,
            hotel_area=profile.hotel_area,
            food_preference=list(profile.food_preference),
            must_visit=list(profile.must_visit),
            avoid=list(profile.avoid),
            transport_mode=profile.transport_mode,
            itinerary_summary=str(itinerary.get("summary", "")),
            poi_names=_unique(poi_names)[:30],
            critic_passed=True,
            created_at=now,
            updated_at=now,
        )
        return self.repository.upsert_completed_trip(user_id, trip)

    def list_recent_trips(self, user_id: str, limit: int = 10) -> list[RecentTrip]:
        return self.repository.list_recent_trips(user_id, limit)

    def forget_preference(self, user_id: str, category: str, value: str | None = None) -> None:
        self.repository.forget_preference(user_id, category, value)

    def clear_preferences(self, user_id: str) -> None:
        self.repository.clear_preferences(user_id)

    def delete_user_memory(self, user_id: str) -> None:
        self.repository.delete_user_memory(user_id)

    def close(self) -> None:
        self.repository.close()


_SERVICES: dict[tuple[str, str, str], UserMemoryService] = {}


def get_user_memory_service(settings: MemorySettings) -> UserMemoryService:
    backend = settings.backend.lower()
    key = (backend, settings.database_url or "", str(settings.profile_dir))
    if key in _SERVICES:
        return _SERVICES[key]
    if backend == "json":
        repository: UserMemoryRepository = JsonUserMemoryRepository(settings.profile_dir)
    elif backend == "postgres":
        if not settings.database_url:
            raise RuntimeError(
                "memory.backend=postgres requires TRAVEL_AGENT_DATABASE_URL"
            )
        from travel_agent.storage.postgres_user_memory import PostgresUserMemoryRepository

        repository = PostgresUserMemoryRepository(
            settings.database_url,
            legacy_profile_dir=settings.profile_dir,
        )
    else:
        raise ValueError(f"unsupported memory backend: {settings.backend}")
    service = UserMemoryService(repository)
    _SERVICES[key] = service
    return service


def close_user_memory_services() -> None:
    for service in _SERVICES.values():
        service.close()
    _SERVICES.clear()


def preference_event_id(
    user_id: str,
    session_id: str,
    turn_index: int,
    observation: PreferenceObservation,
) -> str:
    raw = "\0".join(
        (user_id, session_id, str(turn_index), observation.category, observation.value, observation.polarity)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_profile_from_memory(memory: UserMemory) -> TravelProfile:
    active = [item for item in memory.preference_evidence if item.active]
    grouped: dict[str, list[PreferenceEvidence]] = {}
    for item in active:
        grouped.setdefault(item.category, []).append(item)
    for values in grouped.values():
        values.sort(key=lambda item: (item.positive_count, item.last_seen_at), reverse=True)

    def multi(category: str) -> list[str]:
        return [item.value for item in grouped.get(category, [])[:MAX_ACTIVE_VALUES]]

    def single(category: str, default=None):
        values = sorted(grouped.get(category, []), key=lambda item: item.last_seen_at, reverse=True)
        return values[0].value if values else default

    return TravelProfile(
        budget_level=single("budget_level"),
        interests=multi("interests"),
        pace=single("pace", "standard"),
        food_preference=multi("food_preference"),
        avoid=multi("avoid"),
        transport_mode=single("transport_mode", "public_transport"),
    )


def _normalize_observations(observations: list[PreferenceObservation]) -> list[PreferenceObservation]:
    normalized: list[PreferenceObservation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in observations:
        category = item.category.strip()
        value = item.value.strip()
        if category not in STABLE_CATEGORIES or not value:
            continue
        key = (category, value, item.polarity)
        if key not in seen:
            normalized.append(PreferenceObservation(category, value, item.polarity))
            seen.add(key)
    return normalized


def _prune_and_rebuild(memory: UserMemory, now: float) -> None:
    old_event_count = len(memory.preference_events)
    old_trip_count = len(memory.recent_trips)
    cutoff = now - RETENTION_SECONDS
    events = [event for event in memory.preference_events if event.observed_at >= cutoff]
    events.sort(key=lambda event: (event.observed_at, event.event_id))
    memory.preference_events = events[-MAX_PREFERENCE_EVENTS:]
    memory.recent_trips = sorted(
        (trip for trip in memory.recent_trips if trip.updated_at >= cutoff),
        key=lambda trip: trip.updated_at,
        reverse=True,
    )[:MAX_RECENT_TRIPS]
    memory.preference_evidence = _build_evidence(memory.preference_events)
    memory.updated_at = now
    if old_event_count != len(memory.preference_events) or old_trip_count != len(memory.recent_trips):
        logger.info(
            "pruned user memory user_id=%s events_removed=%s trips_removed=%s",
            memory.user_id,
            old_event_count - len(memory.preference_events),
            old_trip_count - len(memory.recent_trips),
        )


def _build_evidence(events: list[PreferenceEvent]) -> list[PreferenceEvidence]:
    evidence: dict[tuple[str, str], PreferenceEvidence] = {}
    latest_single: dict[str, tuple[float, str, Polarity]] = {}
    for event in sorted(
        events,
        key=lambda item: (
            item.observed_at,
            0 if item.polarity == "positive" else 1,
            item.event_id,
        ),
    ):
        key = (event.category, event.value)
        item = evidence.setdefault(key, PreferenceEvidence(event.category, event.value))
        if event.polarity == "positive":
            item.positive_count += 1
        else:
            item.negative_count += 1
        item.last_polarity = event.polarity
        item.last_seen_at = event.observed_at
        item.active = event.polarity == "positive"
        if event.category in SINGLE_CATEGORIES:
            latest_single[event.category] = (event.observed_at, event.value, event.polarity)
    for item in evidence.values():
        if item.category in SINGLE_CATEGORIES:
            latest = latest_single[item.category]
            item.active = latest[1] == item.value and latest[2] == "positive"
    for category in MULTI_CATEGORIES:
        active_values = sorted(
            (
                item
                for item in evidence.values()
                if item.category == category and item.active
            ),
            key=lambda item: (item.positive_count, item.last_seen_at, item.value),
            reverse=True,
        )
        for item in active_values[MAX_ACTIVE_VALUES:]:
            item.active = False
    return sorted(evidence.values(), key=lambda item: (item.category, -item.last_seen_at, item.value))


def _empty_memory(user_id: str) -> UserMemory:
    now = time.time()
    return UserMemory(user_id=user_id, created_at=now, updated_at=now)


def _legacy_memory_from_dict(user_id: str, data: dict) -> UserMemory:
    updated_at = float(data.get("updated_at") or time.time())
    profile = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
    memory = UserMemory(user_id=user_id, created_at=updated_at, updated_at=updated_at)
    observations: list[PreferenceObservation] = []
    for category in MULTI_CATEGORIES:
        for value in profile.get(category, []) or []:
            observations.append(PreferenceObservation(category, str(value)))
    for category, default in (
        ("budget_level", None),
        ("pace", "standard"),
        ("transport_mode", "public_transport"),
    ):
        value = profile.get(category)
        if value and value != default:
            observations.append(PreferenceObservation(category, str(value)))
    for observation in observations:
        memory.preference_events.append(
            PreferenceEvent(
                event_id=preference_event_id(user_id, "legacy", 0, observation),
                session_id="legacy",
                turn_index=0,
                category=observation.category,
                value=observation.value,
                polarity=observation.polarity,
                observed_at=updated_at,
            )
        )
    if profile.get("destination") or profile.get("days"):
        legacy_id = hashlib.sha256(f"{user_id}\0{updated_at}".encode()).hexdigest()[:12]
        memory.recent_trips.append(
            RecentTrip(
                session_id=f"legacy-{legacy_id}",
                destination=profile.get("destination"),
                days=profile.get("days"),
                start_date=profile.get("start_date"),
                companions=profile.get("companions"),
                budget_level=profile.get("budget_level"),
                interests=list(profile.get("interests", [])),
                pace=profile.get("pace", "standard"),
                hotel_area=profile.get("hotel_area"),
                food_preference=list(profile.get("food_preference", [])),
                must_visit=list(profile.get("must_visit", [])),
                avoid=list(profile.get("avoid", [])),
                transport_mode=profile.get("transport_mode", "public_transport"),
                itinerary_summary="旧版画像迁移记录",
                created_at=updated_at,
                updated_at=updated_at,
            )
        )
    _prune_and_rebuild(memory, updated_at)
    return memory


def _memory_to_dict(memory: UserMemory) -> dict:
    return asdict(memory)


def _memory_from_dict(data: dict) -> UserMemory:
    return UserMemory(
        user_id=str(data.get("user_id", "default")),
        schema_version=2,
        created_at=float(data.get("created_at", 0.0)),
        updated_at=float(data.get("updated_at", 0.0)),
        preference_events=[PreferenceEvent(**item) for item in data.get("preference_events", [])],
        preference_evidence=[PreferenceEvidence(**item) for item in data.get("preference_evidence", [])],
        recent_trips=[RecentTrip(**item) for item in data.get("recent_trips", [])],
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
