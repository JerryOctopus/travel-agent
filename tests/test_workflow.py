from travel_agent.schemas import TravelProfile
from travel_agent.workflow import extract_profile_rule_based, merge_profile, run_mvp_workflow


def test_extract_profile_from_chinese_message() -> None:
    profile = extract_profile_rule_based("帮我规划杭州三天情侣旅行，喜欢自然和美食，轻松一点，预算中等")

    assert profile.destination == "杭州"
    assert profile.days == 3
    assert profile.companions == "couple"
    assert profile.pace == "relaxed"
    assert profile.budget_level == "mid"
    assert "nature" in profile.interests
    assert "food" in profile.interests


def test_workflow_returns_clarification_when_required_fields_missing() -> None:
    result = run_mvp_workflow("我想轻松一点，喜欢自然和美食")

    assert result.itinerary is None
    assert result.ranked_pois == []
    assert result.clarification_question == "你想去哪个城市，计划玩几天？"


def test_workflow_builds_itinerary_for_complete_request() -> None:
    result = run_mvp_workflow("帮我规划杭州三天情侣旅行，喜欢自然和美食，轻松一点，预算中等")

    assert result.clarification_question is None
    assert result.itinerary is not None
    assert result.itinerary.city == "杭州"
    assert len(result.itinerary.days) == 3
    assert result.ranked_pois[0].poi.name == "西湖"
    assert result.itinerary.days[0].stops
    assert result.itinerary.days[0].stops[1].route_from_previous is not None
    assert result.critic_result is not None
    assert result.knowledge_chunks
    assert result.weather is not None


def test_workflow_builds_beijing_itinerary() -> None:
    result = run_mvp_workflow("帮我规划北京两天，喜欢历史和美食，不要太累")

    assert result.clarification_question is None
    assert result.itinerary is not None
    assert result.itinerary.city == "北京"
    assert len(result.itinerary.days) == 2
    assert "故宫博物院" in [item.poi.name for item in result.ranked_pois]
    assert any(
        stop.poi.category == "food"
        for day in result.itinerary.days
        for stop in day.stops
    )
    assert result.critic_result is not None
    assert result.critic_result.passed is True


def test_workflow_builds_new_seed_city_itineraries() -> None:
    cases = [
        ("上海一天城市漫步，想看经典城市景观和吃点东西", "上海", 1),
        ("成都三天，喜欢美食和文化，节奏轻松一点", "成都", 3),
        ("西安两天历史文化游，不要太累", "西安", 2),
    ]

    for query, city, days in cases:
        result = run_mvp_workflow(query)

        assert result.clarification_question is None
        assert result.itinerary is not None
        assert result.profile.destination == city
        assert result.profile.days == days
        assert result.itinerary.days
        assert result.knowledge_chunks
        assert result.weather is not None


def test_merge_profile_keeps_existing_preferences_and_fills_missing_slots() -> None:
    base = TravelProfile(interests=["nature", "food"], pace="relaxed")
    update = extract_profile_rule_based("杭州三天")

    merged = merge_profile(base, update)

    assert merged.destination == "杭州"
    assert merged.days == 3
    assert merged.interests == ["nature", "food"]
    assert merged.pace == "relaxed"


def test_workflow_continues_with_existing_profile_context() -> None:
    first_turn = run_mvp_workflow("我想轻松一点，喜欢自然和美食")
    second_turn = run_mvp_workflow("杭州三天", existing_profile=first_turn.profile)

    assert first_turn.clarification_question == "你想去哪个城市，计划玩几天？"
    assert second_turn.clarification_question is None
    assert second_turn.itinerary is not None
    assert second_turn.profile.destination == "杭州"
    assert second_turn.profile.days == 3
    assert second_turn.profile.interests == ["nature", "food"]
    assert second_turn.profile.pace == "relaxed"


def test_workflow_revises_to_include_must_visit_from_context() -> None:
    context = TravelProfile(
        destination="北京",
        days=1,
        interests=["food"],
        must_visit=["故宫"],
        pace="relaxed",
    )

    result = run_mvp_workflow("", existing_profile=context)
    names = [
        stop.poi.name
        for day in result.itinerary.days
        for stop in day.stops
    ]

    assert result.revised is True
    assert any("故宫" in name for name in names)
    assert result.revision_notes
    assert "must_visit_missing" not in {
        issue.code for issue in result.critic_result.issues
    }
