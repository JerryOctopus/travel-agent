from __future__ import annotations

from travel_agent.agent.response_summary import (
    build_plan_reply_text,
    looks_like_hallucinated_itinerary,
)
from travel_agent.agent.runtime import _finalize_reply_text
from travel_agent.agent.session import ArtifactStore, build_session
from travel_agent.schemas import TravelProfile


def test_looks_like_hallucinated_itinerary() -> None:
    bad = "⚠️ **Critic 检测到关键问题**\n我将**人工介入修正**\n#### Day 1｜外滩"
    assert looks_like_hallucinated_itinerary(bad)
    assert not looks_like_hallucinated_itinerary("上海3天行程已生成，请看右侧卡片。")


def test_build_plan_reply_text_is_short() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="上海", days=3, interests=["food", "nature"])
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {
                "summary": "上海3天轻松行程",
                "days": [
                    {
                        "day_index": 1,
                        "theme": "外滩",
                        "stops": [{"name": "外滩"}, {"name": "豫园"}],
                    }
                ],
            },
            "critic": {"passed": True},
            "original_issue_count": 1,
            "final_issue_count": 0,
            "revision_notes": ["调整了雨天安排"],
        },
    )
    text = build_plan_reply_text(ctx)
    assert "右侧面板" in text
    assert "第1天" in text
    assert "人工介入" not in text
    assert len(text) < 500


def test_plan_reply_hides_internal_reviewer_instruction() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="成都", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "成都一日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "revision_notes": [
                "已携带 Reviewer 定向要求重新规划：补充上午活动",
                "补充必去地点 `武侯祠`。",
            ],
        },
    )

    text = build_plan_reply_text(ctx)

    assert "Reviewer" not in text
    assert "补充必去地点" in text


def test_plan_reply_does_not_claim_delivery_when_budget_is_exceeded() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=3)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "杭州三日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "budget_plan": {
                "user_limit_cny": 3000,
                "total_expected_cny": 3240,
                "total_low_cny": 2800,
                "total_high_cny": 3700,
                "within_user_limit": False,
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "尚不可交付" in text


def test_finalize_prefers_artifact_over_llm() -> None:
    ctx = build_session(persist=False)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "杭州3天", "days": []},
            "critic": {"passed": True},
            "original_issue_count": 0,
            "final_issue_count": 0,
        },
    )
    llm = "#### Day 1｜" + "x" * 800
    out = _finalize_reply_text(ctx, ["plan_and_critique"], llm)
    assert "Day 1｜" not in out
    assert "右侧面板" in out or "杭州" in out


def test_plan_reply_surfaces_structured_return_plan_without_inventing_train() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "杭州一日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "return_plan": {
                "required": True,
                "activity_cutoff": "18:30",
                "arrival_deadline": "21:30",
                "to_location": "上海",
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "18:30" in text and "21:30" in text and "上海" in text
    assert "实时车次" in text


def test_plan_reply_surfaces_selected_lodging_and_budget_limit() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="北京", days=2)
    ctx.store.put(
        "itinerary",
        {
            "itinerary": {"summary": "北京两日游", "days": []},
            "critic": {"passed": True, "issues": []},
            "lodging_plan": {
                "status": "recommended_not_booked",
                "hotel": {"name": "王府井大饭店"},
                "rooms": 1,
                "nights": 1,
                "lodging_subtotal_cny": 585,
            },
            "budget_plan": {
                "total_low_cny": 1675.5,
                "total_expected_cny": 1965,
                "total_high_cny": 2254.5,
                "user_limit_cny": 4500,
                "within_user_limit": True,
            },
        },
    )

    text = build_plan_reply_text(ctx)

    assert "住宿推荐（未预订）：王府井大饭店" in text
    assert "不超过上限 ¥4500" in text
