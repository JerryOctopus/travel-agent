from __future__ import annotations

from travel_agent.agent.session import ArtifactStore, build_session
from travel_agent.schemas import TravelProfile
from travel_agent.settings import LLMSettings, MemorySettings
from travel_agent.storage.memory_compressor import (
    MemoryCompressor,
    _llm_summary_blocked,
    reset_summarizer_guard,
)
from travel_agent.storage.memory_framework import (
    MemoryFramework,
    choose_memory_mode,
    effective_profile_only_threshold,
    estimate_history_tokens,
    estimate_model_context_window,
    estimate_text_tokens,
    _format_l3,
)
from travel_agent.storage.user_memory import RecentTrip
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


def test_token_estimate_handles_english_chinese_and_mixed_text():
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2
    assert estimate_text_tokens("杭州旅游规划") == 6
    assert estimate_text_tokens("abcd杭州") == 3
    assert estimate_history_tokens([("user", "abcd"), ("assistant", "杭州")]) == 3


def test_choose_memory_mode_uses_chinese_aware_token_estimate():
    settings = MemorySettings(
        compress_message_threshold=20,
        profile_only_token_threshold=6,
    )

    assert choose_memory_mode([("user", "杭州旅游规")], settings) == "full"
    assert choose_memory_mode([("user", "杭州旅游规划")], settings) == "profile_only"


def test_memory_settings_use_production_defaults():
    settings = MemorySettings()
    assert settings.compress_message_threshold == 20
    assert settings.profile_only_token_threshold == 64000
    assert settings.keep_recent_turns == 10


def test_user_profile_store_roundtrip(tmp_path):
    store = UserProfileStore(tmp_path)
    profile = TravelProfile(destination="杭州", interests=["food"], pace="relaxed")
    store.save("u1", profile)
    loaded = store.load("u1")
    assert loaded.destination is None
    assert "food" in loaded.interests
    assert store.service.list_recent_trips("u1", 1)[0].destination == "杭州"


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


def test_profile_only_keeps_recent_turns_and_summarizes_older_history(tmp_path):
    ctx = build_session(persist=False)
    history = [("user", f"msg{i}") for i in range(12)]
    settings = MemorySettings(
        compress_message_threshold=2,
        profile_only_token_threshold=1,
        keep_recent_turns=2,
        profile_dir=tmp_path,
    )

    mem = MemoryFramework.build(ctx, history, "u1", settings, False)

    assert mem.mode == "profile_only"
    assert mem.history_for_prompt == history[-4:]
    assert "msg0" in mem.compressed_summary
    assert "msg7" in mem.compressed_summary
    assert "msg8" not in mem.compressed_summary


def test_l3_snapshot_is_compact_and_includes_recent_trips():
    profile = TravelProfile(
        interests=[f"interest{i}" for i in range(20)],
        food_preference=[f"food{i}" for i in range(20)],
        avoid=[f"avoid{i}" for i in range(20)],
    )
    trips = [
        RecentTrip(
            session_id=f"s{i}",
            destination=f"城市{i}",
            days=i + 1,
            itinerary_summary="摘要" * 100,
        )
        for i in range(5)
    ]
    snapshot = _format_l3(profile, "u1", trips)

    assert "城市0" in snapshot
    assert "城市2" in snapshot
    assert "城市3" not in snapshot
    assert "interest7" in snapshot
    assert "interest8" not in snapshot
    assert len(snapshot) <= 2000


def test_compress_caches_llm_summary():
    reset_summarizer_guard()
    llm = LLMSettings(provider="openai", api_key="test", model="cache-test-model")
    compressor = MemoryCompressor(llm_settings=llm, keep_recent_turns=2)
    calls = {"n": 0}

    def fake_summarize(history):
        calls["n"] += 1
        return "LLM 摘要"

    compressor._llm_summarize = fake_summarize
    history = [("user", f"msg{i}") for i in range(10)]
    assert compressor.compress(history) == (history[-4:], "LLM 摘要")
    assert compressor.compress(history) == (history[-4:], "LLM 摘要")
    # 相同的旧历史只调一次 LLM，第二次命中缓存。
    assert calls["n"] == 1
    reset_summarizer_guard()


def test_summarizer_circuit_breaker_blocks_after_consecutive_failures():
    reset_summarizer_guard()
    llm = LLMSettings(provider="openai", api_key="test", model="breaker-test-model")
    compressor = MemoryCompressor(llm_settings=llm, keep_recent_turns=1)
    compressor._llm_summarize = lambda history: ""  # 模拟 LLM 持续失败

    for i in range(3):
        _, summary = compressor.compress([("user", f"fail{i}-{j}") for j in range(6)])
        assert summary.startswith("历史对话摘要")
    assert _llm_summary_blocked()

    compressor._llm_summarize = lambda history: "恢复后的摘要"
    _, summary = compressor.compress([("user", f"after-{j}") for j in range(6)])
    # 熔断冷却期内即使 LLM 可用也直接走规则压缩。
    assert summary.startswith("历史对话摘要")

    reset_summarizer_guard()
    _, summary = compressor.compress([("user", f"after-{j}") for j in range(6)])
    assert summary == "恢复后的摘要"


def test_estimate_model_context_window():
    assert estimate_model_context_window("qwen-turbo") == 128_000
    assert estimate_model_context_window("deepseek-chat") == 64_000
    assert estimate_model_context_window("glm-4.6") == 200_000
    assert estimate_model_context_window("unknown-model") == 128_000


def test_effective_threshold_uses_model_window():
    settings = MemorySettings(profile_only_token_threshold=64000)
    small = LLMSettings(provider="openai", api_key="k", model="small", context_window=30_000)
    assert effective_profile_only_threshold(settings, small) == 10_000
    big = LLMSettings(provider="openai", api_key="k", model="qwen-turbo")
    assert effective_profile_only_threshold(settings, big) == 64000
    assert effective_profile_only_threshold(settings, None) == 64000


def test_choose_memory_mode_respects_model_window():
    settings = MemorySettings(compress_message_threshold=100, profile_only_token_threshold=64000)
    llm = LLMSettings(provider="openai", api_key="k", model="small", context_window=20_100)
    hist = [("user", "x" * 400)]  # 约 100 token
    # 生效阈值 = 20100 - 20000 = 100，小窗口下提前进入强压缩。
    assert choose_memory_mode(hist, settings, llm) == "profile_only"
    assert choose_memory_mode(hist, settings) == "full"
