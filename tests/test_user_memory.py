from __future__ import annotations

import json

from travel_agent.schemas import TravelProfile
from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.agent import toolkit
from travel_agent.storage.user_memory import (
    JsonUserMemoryRepository,
    PreferenceObservation,
    RecentTrip,
    RETENTION_SECONDS,
    UserMemoryService,
)


def test_preference_events_are_idempotent_and_support_negation(tmp_path) -> None:
    repo = JsonUserMemoryRepository(tmp_path)
    positive = [PreferenceObservation("interests", "museum")]
    repo.record_preference_events("u1", "s1", 1, positive, observed_at=1000)
    repo.record_preference_events("u1", "s1", 1, positive, observed_at=1000)

    profile = UserMemoryService(repo).load_stable_profile("u1")
    evidence = repo.load_memory("u1").preference_evidence[0]
    assert profile.interests == ["museum"]
    assert evidence.positive_count == 1

    repo.record_preference_events(
        "u1", "s1", 2, [PreferenceObservation("interests", "museum", "negative")], observed_at=1001
    )
    assert UserMemoryService(repo).load_stable_profile("u1").interests == []

    repo.record_preference_events("u1", "s2", 1, positive, observed_at=1002)
    assert UserMemoryService(repo).load_stable_profile("u1").interests == ["museum"]


def test_single_preferences_use_latest_explicit_value(tmp_path) -> None:
    repo = JsonUserMemoryRepository(tmp_path)
    repo.record_preference_events(
        "u1", "s1", 1, [PreferenceObservation("pace", "relaxed")], observed_at=1000
    )
    repo.record_preference_events(
        "u1", "s2", 1, [PreferenceObservation("pace", "intensive")], observed_at=1001
    )
    assert UserMemoryService(repo).load_stable_profile("u1").pace == "intensive"


def test_multi_preference_evidence_caps_active_values(tmp_path) -> None:
    repo = JsonUserMemoryRepository(tmp_path)
    repo.record_preference_events(
        "u1",
        "s1",
        1,
        [PreferenceObservation("interests", f"interest-{index}") for index in range(25)],
        observed_at=1000,
    )
    memory = repo.load_memory("u1")
    assert sum(item.active for item in memory.preference_evidence) == 20
    assert len(UserMemoryService(repo).load_stable_profile("u1").interests) == 20


def test_tool_can_explicitly_remove_single_preference() -> None:
    ctx = build_session(session_id="single-removal", persist=False)
    toolkit.update_travel_profile(ctx, pace="relaxed")
    toolkit.update_travel_profile(ctx, remove_pace="relaxed")
    assert ctx.profile.pace == "standard"
    assert {
        (item["category"], item["value"], item["polarity"])
        for item in ctx.pending_preference_observations
    } >= {("pace", "relaxed", "positive"), ("pace", "relaxed", "negative")}


def test_recent_trips_upsert_and_retention(tmp_path) -> None:
    repo = JsonUserMemoryRepository(tmp_path)
    now = 2 * RETENTION_SECONDS
    repo.upsert_completed_trip(
        "u1", RecentTrip(session_id="old", destination="旧城", updated_at=now - RETENTION_SECONDS - 1)
    )
    for index in range(22):
        repo.upsert_completed_trip(
            "u1",
            RecentTrip(
                session_id=f"s{index}",
                destination=f"城市{index}",
                poi_names=[f"景点{i}" for i in range(35)],
                updated_at=now + index,
            ),
        )
    repo.upsert_completed_trip(
        "u1", RecentTrip(session_id="s21", destination="更新城市", updated_at=now + 100)
    )

    trips = repo.list_recent_trips("u1", 30)
    assert len(trips) == 20
    assert trips[0].destination == "更新城市"
    assert len(trips[0].poi_names) == 0
    assert all(trip.session_id != "old" for trip in trips)


def test_legacy_json_migrates_stable_preferences_and_trip(tmp_path) -> None:
    (tmp_path / "u1.json").write_text(
        json.dumps(
            {
                "user_id": "u1",
                "updated_at": 1000,
                "profile": {
                    "destination": "杭州",
                    "days": 3,
                    "interests": ["food"],
                    "pace": "relaxed",
                    "avoid": ["早起"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repo = JsonUserMemoryRepository(tmp_path)
    memory = repo.load_memory("u1")

    assert UserMemoryService(repo).load_stable_profile("u1").destination is None
    assert UserMemoryService(repo).load_stable_profile("u1").interests == ["food"]
    assert memory.recent_trips[0].destination == "杭州"


def test_forget_clear_and_delete_are_user_scoped(tmp_path) -> None:
    repo = JsonUserMemoryRepository(tmp_path)
    for user_id in ("u1", "u2"):
        repo.record_preference_events(
            user_id, "s1", 1, [PreferenceObservation("interests", "food")], observed_at=1000
        )
    repo.forget_preference("u1", "interests", "food")
    assert UserMemoryService(repo).load_stable_profile("u1").interests == []
    assert UserMemoryService(repo).load_stable_profile("u2").interests == ["food"]
    repo.clear_preferences("u2")
    assert UserMemoryService(repo).load_stable_profile("u2").interests == []
    repo.delete_user_memory("u2")
    assert not (tmp_path / "u2.json").exists()


def test_service_compacts_completed_itinerary(tmp_path) -> None:
    repo = JsonUserMemoryRepository(tmp_path)
    service = UserMemoryService(repo)
    itinerary = {
        "summary": "杭州三日轻松游",
        "days": [{"stops": [{"poi": {"name": f"景点{i}"}} for i in range(35)]}],
    }
    service.upsert_completed_trip(
        "u1", "s1", TravelProfile(destination="杭州", days=3), itinerary
    )
    trip = service.list_recent_trips("u1", 1)[0]
    assert trip.itinerary_summary == "杭州三日轻松游"
    assert len(trip.poi_names) == 30


def test_runtime_records_completed_trip_and_explicit_negation(offline_settings) -> None:
    ctx = build_session(session_id="runtime-memory", persist=False)
    run_production_turn(
        "杭州三天，喜欢博物馆",
        ctx=ctx,
        settings=offline_settings,
        user_id="u-runtime",
    )
    service = UserMemoryService(JsonUserMemoryRepository(offline_settings.memory.profile_dir))
    assert service.list_recent_trips("u-runtime", 1)[0].destination == "杭州"
    assert "museum" in service.load_stable_profile("u-runtime").interests

    run_production_turn(
        "我不再喜欢博物馆",
        ctx=ctx,
        history=[("user", "杭州三天，喜欢博物馆"), ("assistant", "已规划")],
        settings=offline_settings,
        user_id="u-runtime",
    )
    assert "museum" not in service.load_stable_profile("u-runtime").interests


def test_runtime_does_not_resave_stale_itinerary_on_chat(offline_settings) -> None:
    ctx = build_session(session_id="stale-itinerary", persist=False)
    run_production_turn(
        "杭州三天，喜欢美食",
        ctx=ctx,
        settings=offline_settings,
        user_id="u-stale",
    )
    service = UserMemoryService(JsonUserMemoryRepository(offline_settings.memory.profile_dir))
    before = service.list_recent_trips("u-stale", 1)[0].updated_at
    run_production_turn(
        "你好",
        ctx=ctx,
        history=[("user", "杭州三天，喜欢美食"), ("assistant", "已规划")],
        settings=offline_settings,
        user_id="u-stale",
    )
    assert service.list_recent_trips("u-stale", 1)[0].updated_at == before
