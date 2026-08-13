from __future__ import annotations

from travel_agent.agent.runtime import (
    _build_chat_model,
    _reply_from_outcome,
    run_production_turn,
)
from travel_agent.agent.session import build_session
from travel_agent.harness.faults import FaultInjectingProvider
from travel_agent.orchestration.multi_agent.engine import TurnOutcome
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_INCOMPLETE,
    ReviewIssue,
    ReviewResult,
    SubagentResult,
)
from travel_agent.settings import LLMSettings, Settings


def test_v4_flash_explicitly_disables_provider_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    settings = Settings(
        llm=LLMSettings(
            provider="siliconflow",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            model="deepseek-ai/DeepSeek-V4-Flash",
            thinking_enabled=False,
        )
    )

    _build_chat_model(settings)

    assert captured["extra_body"] == {"enable_thinking": False}


def test_v4_flash_thinking_mode_keeps_explicit_effort(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    settings = Settings(
        llm=LLMSettings(
            provider="siliconflow",
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
            model="deepseek-ai/DeepSeek-V4-Flash",
            thinking_enabled=True,
        )
    )

    _build_chat_model(settings)

    assert captured["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }


def test_offline_turn_produces_itinerary(offline_settings):
    ctx = build_session(persist=False)
    reply = run_production_turn(
        "帮我规划杭州三天，喜欢自然和美食，轻松一点",
        ctx=ctx,
        settings=offline_settings,
    )

    assert reply.used_real_agent is False
    assert reply.clarification is False
    assert "plan_and_critique" in reply.tool_trace
    assert reply.map_payload is not None
    assert any(card["type"] == "day" for card in reply.cards)
    assert reply.profile["destination"] == "杭州"
    assert reply.profile["days"] == 3


def test_incomplete_outcome_discloses_gate_failure() -> None:
    ctx = build_session(persist=False)
    outcome = TurnOutcome(
        status=STATUS_INCOMPLETE,
        reply="约束检查已通过。",
        review=ReviewResult(
            verdict="failed",
            issues=[
                ReviewIssue(
                    issue_type="closure",
                    severity="critical",
                    description="必去场馆当天闭馆",
                )
            ],
        ),
    )

    reply = _reply_from_outcome(ctx, outcome)

    assert "尚不可交付" in reply.text
    assert "必去场馆当天闭馆" in reply.text
    assert "约束检查已通过" not in reply.text


def test_incomplete_reply_omits_superseded_planner_failure() -> None:
    ctx = build_session(persist=False)
    outcome = TurnOutcome(
        status=STATUS_INCOMPLETE,
        reply="draft",
        review=ReviewResult(verdict="failed", error="review timeout"),
        results=[
            SubagentResult("req", "p1", "planner", "failed", error="bad artifact id"),
            SubagentResult("req", "p2", "planner", "completed"),
        ],
    )

    reply = _reply_from_outcome(ctx, outcome)

    assert "review timeout" in reply.text
    assert "bad artifact id" not in reply.text


def test_offline_turn_asks_when_missing(offline_settings):
    ctx = build_session(persist=False)
    reply = run_production_turn("我想去旅行", ctx=ctx, settings=offline_settings)

    assert reply.clarification is True
    assert reply.map_payload is None
    assert "request_travel_info" in reply.tool_trace


def test_destination_without_days_asks_before_planning(offline_settings):
    from travel_agent.schemas import TravelProfile
    from travel_agent.storage.user_profile import UserProfileStore

    # L3 有历史天数，但不应自动用于本轮规划
    store = UserProfileStore(offline_settings.memory.profile_dir)
    store.save("default", TravelProfile(destination="新西兰", days=5, interests=["nature"]))

    ctx = build_session(persist=False)
    reply = run_production_turn(
        "我想去日本",
        ctx=ctx,
        settings=offline_settings,
        user_id="default",
    )

    assert reply.clarification is True
    assert "plan_and_critique" not in reply.tool_trace
    assert reply.profile["destination"] == "日本"
    assert reply.profile["days"] is None
    assert "几天" in reply.text


def test_multi_turn_accumulates_profile(offline_settings):
    ctx = build_session(persist=False)
    first = run_production_turn("我想去北京", ctx=ctx, settings=offline_settings)
    assert first.clarification is True

    second = run_production_turn("玩两天，喜欢历史", ctx=ctx, settings=offline_settings)
    assert second.clarification is False
    assert second.profile["destination"] == "北京"
    assert second.profile["days"] == 2


def test_editing_existing_plan_skips_destination_days_ask(offline_settings):
    ctx = build_session(persist=False)
    init_reply = run_production_turn(
        "我想去南京，两天行程，主要逛历史景点",
        ctx=ctx,
        settings=offline_settings,
    )
    assert init_reply.clarification is False

    reply = run_production_turn(
        "把第二天的中山陵删掉，改成不去中山陵了",
        ctx=ctx,
        settings=offline_settings,
    )
    assert reply.clarification is False
    assert "request_travel_info" not in reply.tool_trace


def test_compare_hotel_areas_in_existing_plan_skips_clarification(offline_settings):
    ctx = build_session(persist=False)
    init_reply = run_production_turn(
        "帮我规划南京三天，喜欢历史，预算中等",
        ctx=ctx,
        settings=offline_settings,
    )
    assert init_reply.clarification is False

    reply = run_production_turn(
        "帮我比较新街口和夫子庙住哪里方便",
        ctx=ctx,
        settings=offline_settings,
    )
    assert reply.clarification is False
    assert "request_travel_info" not in reply.tool_trace


def test_offline_tool_timeout_recovers_without_blocking_itinerary(offline_settings):
    ctx = build_session(persist=False)
    ctx.provider = FaultInjectingProvider(
        ctx.provider,
        {"operation": "search_pois", "mode": "timeout"},
    )

    reply = run_production_turn("请规划杭州两天行程；如果查询异常请给出建议。", ctx=ctx, settings=offline_settings)

    assert reply.map_payload is not None
    assert reply.recovery_state == "recovered"
    assert "建议" in reply.text
