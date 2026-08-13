"""PostgreSQL implementation of the L3 user-memory repository."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from travel_agent.storage.user_memory import (
    MAX_PREFERENCE_EVENTS,
    MAX_RECENT_TRIPS,
    RETENTION_SECONDS,
    JsonUserMemoryRepository,
    PreferenceEvidence,
    PreferenceEvent,
    PreferenceObservation,
    RecentTrip,
    UserMemory,
    _build_evidence,
    _empty_memory,
    _normalize_observations,
    preference_event_id,
)

logger = logging.getLogger(__name__)


class PostgresUserMemoryRepository:
    def __init__(self, database_url: str, legacy_profile_dir: Path | str | None = None) -> None:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:  # pragma: no cover - depends on deployment extras
            raise RuntimeError("PostgreSQL memory requires sqlalchemy and psycopg") from exc
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.legacy_profile_dir = Path(legacy_profile_dir) if legacy_profile_dir else None
        with self.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")

    def load_memory(self, user_id: str) -> UserMemory:
        with self.engine.begin() as connection:
            memory = self._load(connection, user_id)
        if memory is None and self.legacy_profile_dir:
            legacy = JsonUserMemoryRepository(self.legacy_profile_dir).load_memory(user_id)
            if legacy.preference_events or legacy.recent_trips:
                try:
                    self._import_memory(legacy)
                    logger.info("migrated legacy user memory user_id=%s", user_id)
                except Exception:
                    logger.exception("failed to migrate legacy user memory user_id=%s", user_id)
                    raise
                with self.engine.begin() as connection:
                    memory = self._load(connection, user_id)
        return memory or _empty_memory(user_id)

    def record_preference_events(
        self,
        user_id: str,
        session_id: str,
        turn_index: int,
        observations: list[PreferenceObservation],
        observed_at: float | None = None,
    ) -> UserMemory:
        from sqlalchemy import text

        now = observed_at or time.time()
        with self.engine.begin() as connection:
            self._ensure_user(connection, user_id, now)
            self._lock_user(connection, user_id)
            for observation in _normalize_observations(observations):
                result = connection.execute(
                    text(
                        """
                        INSERT INTO user_preference_events
                          (event_id, user_id, session_id, turn_index, category, value, polarity, observed_at)
                        VALUES
                          (:event_id, :user_id, :session_id, :turn_index, :category, :value, :polarity, :observed_at)
                        ON CONFLICT (event_id) DO NOTHING
                        """
                    ),
                    {
                        "event_id": preference_event_id(user_id, session_id, turn_index, observation),
                        "user_id": user_id,
                        "session_id": session_id,
                        "turn_index": turn_index,
                        "category": observation.category,
                        "value": observation.value,
                        "polarity": observation.polarity,
                        "observed_at": now,
                    },
                )
                if result.rowcount == 0:
                    logger.debug(
                        "ignored duplicate preference event user_id=%s event_id=%s",
                        user_id,
                        preference_event_id(user_id, session_id, turn_index, observation),
                    )
            self._prune_events(connection, user_id, now)
            self._rebuild_evidence(connection, user_id)
            connection.execute(
                text("UPDATE user_memories SET updated_at=:now WHERE user_id=:user_id"),
                {"now": now, "user_id": user_id},
            )
            memory = self._load(connection, user_id)
        return memory or _empty_memory(user_id)

    def upsert_completed_trip(self, user_id: str, trip: RecentTrip) -> UserMemory:
        from sqlalchemy import text

        now = trip.updated_at or time.time()
        trip.updated_at = now
        trip.created_at = trip.created_at or now
        trip.poi_names = list(dict.fromkeys(trip.poi_names))[:30]
        with self.engine.begin() as connection:
            self._ensure_user(connection, user_id, now)
            self._lock_user(connection, user_id)
            connection.execute(
                text(
                    """
                    INSERT INTO user_recent_trips (
                      user_id, session_id, destination, days, start_date, companions,
                      budget_level, interests, pace, hotel_area, food_preference,
                      must_visit, avoid, transport_mode, itinerary_summary, poi_names,
                      critic_passed, created_at, updated_at
                    ) VALUES (
                      :user_id, :session_id, :destination, :days, :start_date, :companions,
                      :budget_level, CAST(:interests AS JSONB), :pace, :hotel_area,
                      CAST(:food_preference AS JSONB), CAST(:must_visit AS JSONB),
                      CAST(:avoid AS JSONB), :transport_mode, :itinerary_summary,
                      CAST(:poi_names AS JSONB), :critic_passed, :created_at, :updated_at
                    )
                    ON CONFLICT (user_id, session_id) DO UPDATE SET
                      destination=EXCLUDED.destination, days=EXCLUDED.days,
                      start_date=EXCLUDED.start_date, companions=EXCLUDED.companions,
                      budget_level=EXCLUDED.budget_level, interests=EXCLUDED.interests,
                      pace=EXCLUDED.pace, hotel_area=EXCLUDED.hotel_area,
                      food_preference=EXCLUDED.food_preference, must_visit=EXCLUDED.must_visit,
                      avoid=EXCLUDED.avoid, transport_mode=EXCLUDED.transport_mode,
                      itinerary_summary=EXCLUDED.itinerary_summary, poi_names=EXCLUDED.poi_names,
                      critic_passed=EXCLUDED.critic_passed, updated_at=EXCLUDED.updated_at
                    """
                ),
                {
                    "user_id": user_id,
                    "session_id": trip.session_id,
                    "destination": trip.destination,
                    "days": trip.days,
                    "start_date": trip.start_date,
                    "companions": trip.companions,
                    "budget_level": trip.budget_level,
                    "interests": json.dumps(trip.interests, ensure_ascii=False),
                    "pace": trip.pace,
                    "hotel_area": trip.hotel_area,
                    "food_preference": json.dumps(trip.food_preference, ensure_ascii=False),
                    "must_visit": json.dumps(trip.must_visit, ensure_ascii=False),
                    "avoid": json.dumps(trip.avoid, ensure_ascii=False),
                    "transport_mode": trip.transport_mode,
                    "itinerary_summary": trip.itinerary_summary,
                    "poi_names": json.dumps(trip.poi_names, ensure_ascii=False),
                    "critic_passed": trip.critic_passed,
                    "created_at": trip.created_at,
                    "updated_at": trip.updated_at,
                },
            )
            self._prune_trips(connection, user_id, now)
            connection.execute(
                text("UPDATE user_memories SET updated_at=:now WHERE user_id=:user_id"),
                {"now": now, "user_id": user_id},
            )
            memory = self._load(connection, user_id)
        return memory or _empty_memory(user_id)

    def list_recent_trips(self, user_id: str, limit: int = 10) -> list[RecentTrip]:
        return self.load_memory(user_id).recent_trips[: max(0, limit)]

    def forget_preference(self, user_id: str, category: str, value: str | None = None) -> None:
        from sqlalchemy import text

        clause = "category=:category" + (" AND value=:value" if value is not None else "")
        params: dict[str, Any] = {"user_id": user_id, "category": category}
        if value is not None:
            params["value"] = value
        with self.engine.begin() as connection:
            self._lock_user(connection, user_id)
            connection.execute(
                text(f"DELETE FROM user_preference_events WHERE user_id=:user_id AND {clause}"),
                params,
            )
            self._rebuild_evidence(connection, user_id)
            connection.execute(
                text("UPDATE user_memories SET updated_at=:now WHERE user_id=:user_id"),
                {"now": time.time(), "user_id": user_id},
            )
        logger.info(
            "forgot user preference user_id=%s category=%s has_value=%s",
            user_id,
            category,
            value is not None,
        )

    def clear_preferences(self, user_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            self._lock_user(connection, user_id)
            connection.execute(text("DELETE FROM user_preference_events WHERE user_id=:user_id"), {"user_id": user_id})
            connection.execute(text("DELETE FROM user_preference_evidence WHERE user_id=:user_id"), {"user_id": user_id})
            connection.execute(
                text("UPDATE user_memories SET updated_at=:now WHERE user_id=:user_id"),
                {"now": time.time(), "user_id": user_id},
            )
        logger.info("cleared user preferences user_id=%s", user_id)

    def delete_user_memory(self, user_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM user_memories WHERE user_id=:user_id"), {"user_id": user_id})
        logger.info("deleted user memory user_id=%s", user_id)

    def close(self) -> None:
        self.engine.dispose()

    def _ensure_user(self, connection, user_id: str, now: float) -> None:
        from sqlalchemy import text

        connection.execute(
            text(
                """INSERT INTO user_memories (user_id, schema_version, created_at, updated_at)
                VALUES (:user_id, 2, :now, :now)
                ON CONFLICT (user_id) DO NOTHING"""
            ),
            {"user_id": user_id, "now": now},
        )

    def _lock_user(self, connection, user_id: str) -> None:
        from sqlalchemy import text

        connection.execute(
            text("SELECT user_id FROM user_memories WHERE user_id=:user_id FOR UPDATE"),
            {"user_id": user_id},
        )

    def _load(self, connection, user_id: str) -> UserMemory | None:
        from sqlalchemy import text

        user = connection.execute(
            text("SELECT user_id, schema_version, created_at, updated_at FROM user_memories WHERE user_id=:user_id"),
            {"user_id": user_id},
        ).mappings().first()
        if not user:
            return None
        events = connection.execute(
            text("SELECT * FROM user_preference_events WHERE user_id=:user_id ORDER BY observed_at, event_id"),
            {"user_id": user_id},
        ).mappings()
        evidence = connection.execute(
            text("SELECT category, value, positive_count, negative_count, active, last_polarity, last_seen_at FROM user_preference_evidence WHERE user_id=:user_id"),
            {"user_id": user_id},
        ).mappings()
        trips = connection.execute(
            text("SELECT * FROM user_recent_trips WHERE user_id=:user_id ORDER BY updated_at DESC"),
            {"user_id": user_id},
        ).mappings()
        return UserMemory(
            user_id=user_id,
            schema_version=int(user["schema_version"]),
            created_at=float(user["created_at"]),
            updated_at=float(user["updated_at"]),
            preference_events=[
                PreferenceEvent(
                    event_id=row["event_id"], session_id=row["session_id"],
                    turn_index=row["turn_index"], category=row["category"], value=row["value"],
                    polarity=row["polarity"], observed_at=float(row["observed_at"]),
                ) for row in events
            ],
            preference_evidence=[PreferenceEvidence(**dict(row)) for row in evidence],
            recent_trips=[self._trip_from_row(row) for row in trips],
        )

    def _trip_from_row(self, row) -> RecentTrip:
        data = dict(row)
        data.pop("user_id", None)
        for key in ("interests", "food_preference", "must_visit", "avoid", "poi_names"):
            data[key] = list(data.get(key) or [])
        return RecentTrip(**data)

    def _prune_events(self, connection, user_id: str, now: float) -> None:
        from sqlalchemy import text

        expired = connection.execute(
            text("DELETE FROM user_preference_events WHERE user_id=:user_id AND observed_at < :cutoff"),
            {"user_id": user_id, "cutoff": now - RETENTION_SECONDS},
        )
        overflow = connection.execute(
            text(
                """DELETE FROM user_preference_events WHERE event_id IN (
                  SELECT event_id FROM user_preference_events WHERE user_id=:user_id
                  ORDER BY observed_at DESC, event_id DESC OFFSET :max_events
                )"""
            ),
            {"user_id": user_id, "max_events": MAX_PREFERENCE_EVENTS},
        )
        if expired.rowcount or overflow.rowcount:
            logger.info(
                "pruned preference events user_id=%s expired=%s overflow=%s",
                user_id,
                max(0, expired.rowcount),
                max(0, overflow.rowcount),
            )

    def _prune_trips(self, connection, user_id: str, now: float) -> None:
        from sqlalchemy import text

        expired = connection.execute(
            text("DELETE FROM user_recent_trips WHERE user_id=:user_id AND updated_at < :cutoff"),
            {"user_id": user_id, "cutoff": now - RETENTION_SECONDS},
        )
        overflow = connection.execute(
            text(
                """DELETE FROM user_recent_trips WHERE (user_id, session_id) IN (
                  SELECT user_id, session_id FROM user_recent_trips WHERE user_id=:user_id
                  ORDER BY updated_at DESC OFFSET :max_trips
                )"""
            ),
            {"user_id": user_id, "max_trips": MAX_RECENT_TRIPS},
        )
        if expired.rowcount or overflow.rowcount:
            logger.info(
                "pruned recent trips user_id=%s expired=%s overflow=%s",
                user_id,
                max(0, expired.rowcount),
                max(0, overflow.rowcount),
            )

    def _rebuild_evidence(self, connection, user_id: str) -> None:
        from sqlalchemy import text

        rows = connection.execute(
            text("SELECT * FROM user_preference_events WHERE user_id=:user_id ORDER BY observed_at, event_id"),
            {"user_id": user_id},
        ).mappings()
        events = [
            PreferenceEvent(
                event_id=row["event_id"], session_id=row["session_id"], turn_index=row["turn_index"],
                category=row["category"], value=row["value"], polarity=row["polarity"],
                observed_at=float(row["observed_at"]),
            ) for row in rows
        ]
        connection.execute(text("DELETE FROM user_preference_evidence WHERE user_id=:user_id"), {"user_id": user_id})
        for item in _build_evidence(events):
            connection.execute(
                text(
                    """INSERT INTO user_preference_evidence
                    (user_id, category, value, positive_count, negative_count, active, last_polarity, last_seen_at)
                    VALUES (:user_id, :category, :value, :positive_count, :negative_count, :active, :last_polarity, :last_seen_at)"""
                ),
                {"user_id": user_id, **item.__dict__},
            )

    def _import_memory(self, memory: UserMemory) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            self._ensure_user(connection, memory.user_id, memory.updated_at or time.time())
            self._lock_user(connection, memory.user_id)
            for event in memory.preference_events:
                connection.execute(
                    text(
                        """INSERT INTO user_preference_events
                        (event_id,user_id,session_id,turn_index,category,value,polarity,observed_at)
                        VALUES (:event_id,:user_id,:session_id,:turn_index,:category,:value,:polarity,:observed_at)
                        ON CONFLICT (event_id) DO NOTHING"""
                    ),
                    {"user_id": memory.user_id, **event.__dict__},
                )
            self._rebuild_evidence(connection, memory.user_id)
        for trip in memory.recent_trips:
            self.upsert_completed_trip(memory.user_id, trip)
