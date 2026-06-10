from travel_agent.llm_extractor import (
    FakeLLMTravelProfileExtractor,
    travel_profile_from_payload,
)
from travel_agent.workflow import run_mvp_workflow


def test_fake_llm_extractor_can_drive_workflow() -> None:
    extractor = FakeLLMTravelProfileExtractor(
        payload={
            "destination": "北京",
            "days": 2,
            "interests": ["history", "food"],
            "pace": "relaxed",
        }
    )

    result = run_mvp_workflow("随便一句规则无法理解的话", extractor=extractor)

    assert result.itinerary is not None
    assert result.profile.destination == "北京"
    assert result.profile.days == 2
    assert result.profile.interests == ["history", "food"]
    assert result.critic_result is not None
    assert result.critic_result.passed is True


def test_profile_payload_parser_safely_coerces_invalid_values() -> None:
    profile = travel_profile_from_payload(
        {
            "destination": "杭州",
            "days": "bad",
            "budget_level": "very_high",
            "pace": "super_fast",
            "transport_mode": "rocket",
            "interests": ["nature"],
        }
    )

    assert profile.destination == "杭州"
    assert profile.days is None
    assert profile.budget_level is None
    assert profile.pace == "standard"
    assert profile.transport_mode == "public_transport"
    assert profile.interests == ["nature"]
