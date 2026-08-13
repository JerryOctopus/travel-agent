from __future__ import annotations

from travel_agent.agent.preferences import build_preference_guide_text
from travel_agent.schemas import TravelProfile
from travel_agent.workflow_rules import format_interests_display


def test_format_interests_display_chinese() -> None:
    assert format_interests_display(["nature", "food"]) == "自然、美食"


def test_preference_guide_uses_chinese_labels() -> None:
    profile = TravelProfile(destination="上海", days=3)
    l3 = TravelProfile(interests=["nature", "food"], pace="relaxed")
    text = build_preference_guide_text(profile, l3)
    assert "自然、美食" in text
    assert "轻松" in text
    assert "nature" not in text
    assert "food" not in text
