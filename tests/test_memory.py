from __future__ import annotations

from travel_agent.agent.session import ArtifactStore, build_session
from travel_agent.schemas import TravelProfile
from travel_agent.settings import MemorySettings
from travel_agent.storage.memory_compressor import MemoryCompressor
from travel_agent.storage.memory_framework import choose_memory_mode, MemoryFramework
from travel_agent.storage.user_profile import UserProfileStore


def test_artifact_store_load_and_snapshot(tmp_path):
    store = ArtifactStore(session_id="s1", artifact_dir=tmp_path)
    store.put("weather", {"city": "杭州", "condition": "阴", "temperature_c": 22})
    store2 = ArtifactStore(session_id="s1", artifact_dir=tmp_path)
    assert store2.load_from_disk() == 1
    snap = store2.build_prompt_snapshot()
    assert "杭州" in snap and "阴" in snap


def test_memory_compressor_keeps_recent():
    history = [("user", f"msg{i}") for i in range(20)]
    compressor = MemoryCompressor(keep_recent_turns=2)
    recent, summary = compressor.compress(history)
    assert len(recent) == 4
    assert summary.startswith("历史对话摘要")


def test_choose_memory_mode():
    settings = MemorySettings(compress_message_threshold=4, profile_only_token_threshold=100)
    assert choose_memory_mode([], settings) == "full"
    assert choose_memory_mode([("user", "x")] * 4, settings) == "compressed"
    long_hist = [("user", "x" * 200)] * 10
    assert choose_memory_mode(long_hist, settings) == "profile_only"


def test_user_profile_store_roundtrip(tmp_path):
    store = UserProfileStore(tmp_path)
    profile = TravelProfile(destination="杭州", interests=["food"], pace="relaxed")
    store.save("u1", profile)
    loaded = store.load("u1")
    assert loaded.destination == "杭州"
    assert "food" in loaded.interests


def test_memory_framework_build(tmp_path):
    ctx = build_session(persist=False)
    ctx.store.put("candidates", {"city": "杭州", "pois": []})
    settings = MemorySettings(
        compress_message_threshold=2,
        profile_dir=tmp_path,
    )
    mem = MemoryFramework.build(ctx, [("user", "a"), ("assistant", "b")] * 3, "u1", settings, False)
    assert mem.mode == "compressed"
    assert "已检索 POI" in mem.l2_snapshot
