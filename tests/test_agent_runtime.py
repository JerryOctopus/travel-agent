from __future__ import annotations

from travel_agent.agent.runtime import run_turn
from travel_agent.agent.session import build_session


def test_offline_turn_produces_itinerary():
    ctx = build_session(persist=False)
    reply = run_turn("帮我规划杭州三天，喜欢自然和美食，轻松一点", ctx=ctx)

    assert reply.used_real_agent is False
    assert reply.clarification is False
    assert "plan_and_critique" in reply.tool_trace
    assert reply.map_payload is not None
    assert any(card["type"] == "day" for card in reply.cards)
    assert reply.profile["destination"] == "杭州"
    assert reply.profile["days"] == 3


def test_offline_turn_asks_when_missing():
    ctx = build_session(persist=False)
    reply = run_turn("我想去旅行", ctx=ctx)

    assert reply.clarification is True
    assert reply.map_payload is None
    assert "request_travel_info" in reply.tool_trace


def test_multi_turn_accumulates_profile():
    ctx = build_session(persist=False)
    first = run_turn("我想去北京", ctx=ctx)
    assert first.clarification is True

    second = run_turn("玩两天，喜欢历史", ctx=ctx)
    assert second.clarification is False
    assert second.profile["destination"] == "北京"
    assert second.profile["days"] == 2
