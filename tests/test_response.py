from travel_agent.response import render_markdown_response
from travel_agent.schemas import TravelProfile
from travel_agent.workflow import run_mvp_workflow


def test_response_renders_clarification_question() -> None:
    result = run_mvp_workflow("我喜欢自然和美食，轻松一点")

    assert render_markdown_response(result) == "你想去哪个城市，计划玩几天？"


def test_response_renders_itinerary_markdown() -> None:
    result = run_mvp_workflow("帮我规划北京两天，喜欢历史和美食，不要太累")
    response = render_markdown_response(result)

    assert "# 北京2天轻松行程草案" in response
    assert "## 每日行程" in response
    assert "第1天" in response
    assert "出行上下文" in response
    assert "上一站通勤约" in response
    assert "约束检查" in response
    assert "已通过 critic 检查" in response


def test_response_renders_revision_notes() -> None:
    context = TravelProfile(
        destination="北京",
        days=1,
        interests=["food"],
        must_visit=["故宫博物院"],
        pace="relaxed",
    )
    result = run_mvp_workflow("", existing_profile=context)
    response = render_markdown_response(result)

    assert "## 自动修正" in response
    assert "故宫博物院" in response
    assert "已通过 critic 检查" in response
