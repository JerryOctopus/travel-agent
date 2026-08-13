from __future__ import annotations

from travel_agent.workflow import merge_profile
from travel_agent.workflow_rules import extract_profile_rule_based, normalize_interests
from travel_agent.schemas import TravelProfile


def test_normalize_interests_dedupes_chinese_and_english() -> None:
    assert normalize_interests(["自然", "美食", "nature", "food"]) == ["nature", "food"]


def test_merge_profile_normalizes_interests() -> None:
    base = TravelProfile(interests=["自然"])
    update = TravelProfile(interests=["food"])
    merged = merge_profile(base, update)
    assert merged.interests == ["nature", "food"]


def test_rule_extractor_covers_budget_party_hotel_and_food_constraints() -> None:
    profile = extract_profile_rule_based(
        "两个人去重庆三天，预算控制在3000元以内，住地铁站附近，不能吃辣。"
    )

    assert profile.destination == "重庆"
    assert profile.days == 3
    assert profile.budget_level == "low"
    assert profile.budget_limit == 3000
    assert profile.companions == "两个人"
    assert profile.party_size == 2
    assert profile.hotel_area == "地铁站附近"
    assert profile.food_preference == ["不辣"]


def test_rule_extractor_normalizes_date_and_must_visit() -> None:
    dated = extract_profile_rule_based("2026年10月11日去苏州两天，如果下雨就多安排室内地点。")
    must_visit = extract_profile_rule_based("安排杭州三天行程，西湖和灵隐寺必须去，其余就近推荐。")

    assert dated.start_date == "2026-10-11"
    assert must_visit.must_visit == ["西湖", "灵隐寺"]


def test_rule_extractor_preserves_explicit_followup_must_visit() -> None:
    profile = extract_profile_rule_based("预算改成3500元，但鼓浪屿仍然必须保留，住宿可以降档。")

    assert profile.must_visit == ["鼓浪屿"]


def test_compound_interest_aliases_are_canonical() -> None:
    profile = extract_profile_rule_based("想看自然风光、历史文化、海边、园林，也想找咖啡店。")

    assert set(profile.interests) == {"nature", "culture", "history", "food"}
    assert not set(profile.interests).intersection({"自然风光", "历史文化", "海边", "园林", "咖啡店"})


def test_arrange_trip_phrase_is_not_a_must_visit() -> None:
    profile = extract_profile_rule_based("帮我安排2026年10月10日至12日杭州三日游。")

    assert profile.must_visit == []
