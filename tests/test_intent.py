from __future__ import annotations

from travel_agent.agent.intent import (
    MessageKind,
    classify_message,
    classify_message_rule_based,
    classify_message_with_llm,
    conversation_reply_text,
    has_travel_intent,
    is_pure_greeting,
)
from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.settings import LLMSettings, Settings


def test_is_pure_greeting() -> None:
    assert is_pure_greeting("你好")
    assert is_pure_greeting("Hello!")
    assert not is_pure_greeting("你好，帮我规划杭州三天")


def test_classify_message() -> None:
    assert classify_message("你好") == MessageKind.GREETING
    assert classify_message("在吗") == MessageKind.GREETING
    assert classify_message("1+1等于几") == MessageKind.OUT_OF_SCOPE
    assert classify_message("今天心情不好") == MessageKind.OUT_OF_SCOPE
    assert classify_message("你会写 Python 吗") == MessageKind.OUT_OF_SCOPE
    assert classify_message("帮我规划杭州三天") == MessageKind.TRAVEL
    assert classify_message("我想去北京玩两天") == MessageKind.TRAVEL


def test_rule_based_weak_signals_are_ambiguous() -> None:
    for text in [
        "我今天想吃火锅",
        "周末想找个地方放空一下",
        "情侣去哪比较好",
        "上海有哪些适合一个人散步拍照的地方",
    ]:
        decision = classify_message_rule_based(text)
        assert decision.kind == MessageKind.AMBIGUOUS
        assert decision.confidence == "uncertain"
        assert classify_message(text) == MessageKind.AMBIGUOUS


def test_rule_based_strong_non_travel_signals() -> None:
    for text in [
        "预算怎么做表格",
        "酒店行业分析",
        "情侣头像怎么选",
        "巴黎奥运会金牌榜",
    ]:
        decision = classify_message_rule_based(text)
        assert decision.kind == MessageKind.OUT_OF_SCOPE
        assert decision.confidence == "high"


def test_has_travel_intent() -> None:
    assert not has_travel_intent("你好")
    assert not has_travel_intent("你是谁")
    assert not has_travel_intent("巴黎奥运会金牌榜")
    assert has_travel_intent("帮我规划杭州三天")
    assert has_travel_intent("我想去旅行")


def test_hybrid_classifier_uses_llm_for_uncertain(monkeypatch) -> None:
    settings = Settings(llm=LLMSettings(provider="fake", api_key="key"))

    def fake_llm(user_message, settings, history=None):
        return MessageKind.TRAVEL

    monkeypatch.setattr("travel_agent.agent.intent._classify_message_llm", fake_llm)

    assert classify_message_with_llm("周末想找个地方放空一下", settings) == MessageKind.TRAVEL


def test_hybrid_classifier_keeps_ambiguous_on_llm_error(monkeypatch) -> None:
    settings = Settings(llm=LLMSettings(provider="fake", api_key="key"))

    def broken_llm(user_message, settings, history=None):
        raise ValueError("bad json")

    monkeypatch.setattr("travel_agent.agent.intent._classify_message_llm", broken_llm)

    assert classify_message_with_llm("周末想找个地方放空一下", settings) == MessageKind.AMBIGUOUS


def test_conversation_reply_off_topic() -> None:
    text = conversation_reply_text(MessageKind.OUT_OF_SCOPE, "1+1等于几")
    assert "旅行规划" in text
    assert "1+1" in text


def test_conversation_reply_ambiguous() -> None:
    text = conversation_reply_text(MessageKind.AMBIGUOUS, "周末想找个地方放空一下")
    assert "确认一下" in text
    assert "旅行" in text


def test_greeting_does_not_plan(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("你好", ctx=ctx, settings=offline_settings)

    assert reply.clarification is False  # 纯寒暄，非追问
    assert "plan_and_critique" not in reply.tool_trace
    assert "search_poi" not in reply.tool_trace
    assert reply.map_payload is None
    assert "旅行规划" in reply.text


def test_off_topic_does_not_plan(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("你会写 Python 吗", ctx=ctx, settings=offline_settings)

    assert "plan_and_critique" not in reply.tool_trace
    assert "search_poi" not in reply.tool_trace
    assert reply.map_payload is None
    assert "Python" in reply.text


def test_weak_signal_does_not_plan_without_llm(offline_settings) -> None:
    ctx = build_session(persist=False)
    reply = run_production_turn("我今天想吃火锅", ctx=ctx, settings=offline_settings)

    assert reply.clarification is True
    assert "plan_and_critique" not in reply.tool_trace
    assert "search_poi" not in reply.tool_trace
    assert reply.map_payload is None
    assert "旅行" in reply.text


def test_uncertain_signal_can_be_promoted_by_llm(monkeypatch, offline_settings) -> None:
    settings = Settings(
        llm=LLMSettings(provider="fake", api_key="key"),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )
    ctx = build_session(persist=False)

    def fake_analyze(user_message, ctx, settings, history, evaluation_trace):
        return {"kind": "travel", "task_type": "full_trip_plan", "slots": {}}

    def fail_react(*args, **kwargs):
        raise RuntimeError("skip real react in test")

    monkeypatch.setattr("travel_agent.agent.turn_analysis._analyze_with_llm", fake_analyze)
    monkeypatch.setattr("travel_agent.agent.runtime._run_react", fail_react)

    reply = run_production_turn("上海有哪些适合一个人散步拍照的地方", ctx=ctx, settings=settings)

    assert "request_travel_info" in reply.tool_trace
    assert "plan_and_critique" not in reply.tool_trace
    assert reply.map_payload is None


def test_llm_error_for_uncertain_signal_does_not_plan(monkeypatch, offline_settings) -> None:
    settings = Settings(
        llm=LLMSettings(provider="fake", api_key="key"),
        amap=offline_settings.amap,
        agent=offline_settings.agent,
        memory=offline_settings.memory,
        orchestration=offline_settings.orchestration,
        mcp=offline_settings.mcp,
        skills=offline_settings.skills,
    )
    ctx = build_session(persist=False)

    def broken_analyze(user_message, ctx, settings, history, evaluation_trace):
        raise ValueError("bad json")

    def fake_chat(*args, **kwargs):
        raise RuntimeError("skip chat in test")

    monkeypatch.setattr("travel_agent.agent.turn_analysis._analyze_with_llm", broken_analyze)
    monkeypatch.setattr("travel_agent.agent.runtime._run_conversation_chat", fake_chat)

    reply = run_production_turn("周末想找个地方放空一下", ctx=ctx, settings=settings)

    assert reply.clarification is True
    assert "plan_and_critique" not in reply.tool_trace
    assert "search_poi" not in reply.tool_trace
    assert reply.map_payload is None


def test_preference_followup_continues_travel_context(offline_settings) -> None:
    ctx = build_session(persist=False)
    first = run_production_turn("帮我规划杭州三天", ctx=ctx, settings=offline_settings)
    second = run_production_turn("随便，你推荐", ctx=ctx, settings=offline_settings)

    assert first.clarification is False
    assert "request_preference_guide" not in first.tool_trace
    assert "plan_and_critique" in first.tool_trace
    assert "plan_and_critique" in second.tool_trace
    assert second.map_payload is not None


def test_preference_change_followup_replans_with_new_interest(offline_settings) -> None:
    ctx = build_session(persist=False)
    first = run_production_turn("帮我规划杭州三天，喜欢美食", ctx=ctx, settings=offline_settings)
    second = run_production_turn("自然风景呢，结合吃饭", ctx=ctx, settings=offline_settings)

    assert "plan_and_critique" in first.tool_trace
    assert "plan_and_critique" in second.tool_trace
    assert {"food", "nature"}.issubset(set(ctx.profile.interests))
    assert second.map_payload is not None
    categories = {
        marker["category"]
        for marker in second.map_payload["markers"]
    }
    assert "scenic" in categories
    assert "food" in categories
