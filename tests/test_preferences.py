from __future__ import annotations

from travel_agent.agent.preferences import (
    build_preference_guide_text,
    needs_preference_guidance,
    turn_expresses_preferences,
    user_skips_preference_prompt,
)
from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.schemas import TravelProfile


def test_user_skips_preference_prompt() -> None:
    assert user_skips_preference_prompt("杭州三天随便")
    assert user_skips_preference_prompt("你推荐就行")
    assert not user_skips_preference_prompt("喜欢美食")


def test_turn_expresses_preferences() -> None:
    assert turn_expresses_preferences("玩两天，喜欢历史")
    assert turn_expresses_preferences("轻松一点，想去自然风景多的地方")
    assert not turn_expresses_preferences("玩两天")


def test_needs_preference_guidance() -> None:
    profile = TravelProfile(destination="杭州", days=3)
    assert not needs_preference_guidance(profile, "帮我规划杭州三天")
    assert not needs_preference_guidance(profile, "杭州三天，喜欢美食")
    assert not needs_preference_guidance(profile, "杭州三天随便")


def test_build_preference_guide_text_with_l3() -> None:
    profile = TravelProfile(destination="杭州", days=3)
    l3 = TravelProfile(interests=["food", "nature"], pace="relaxed")
    text = build_preference_guide_text(profile, l3)
    assert "杭州" in text
    assert "美食" in text or "自然" in text
    assert "随便" in text


def test_runtime_defaults_preferences_and_continues_planning(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("帮我规划杭州三天", ctx=ctx, settings=offline_settings)

    assert reply.clarification is False
    assert "request_preference_guide" not in reply.tool_trace
    assert "plan_and_critique" in reply.tool_trace


def test_runtime_inherits_l3_interests_on_sui_bian(offline_settings) -> None:
    from travel_agent.schemas import TravelProfile
    from travel_agent.storage.user_profile import UserProfileStore

    store = UserProfileStore(offline_settings.memory.profile_dir)
    store.save("default", TravelProfile(interests=["nature", "food"], pace="relaxed"))

    ctx = build_session(persist=False)
    reply = run_production_turn("杭州三天随便", ctx=ctx, settings=offline_settings)

    assert reply.clarification is False
    assert "nature" in reply.profile["interests"] or "food" in reply.profile["interests"]


def test_runtime_skips_preference_guide_on_sui_bian(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("杭州三天随便", ctx=ctx, settings=offline_settings)

    assert reply.clarification is False
    assert "plan_and_critique" in reply.tool_trace
