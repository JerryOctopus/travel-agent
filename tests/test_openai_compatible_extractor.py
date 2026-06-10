from travel_agent.config import LLMConfig
from travel_agent.openai_compatible_extractor import OpenAICompatibleTravelProfileExtractor


def test_openai_compatible_extractor_falls_back_when_disabled() -> None:
    extractor = OpenAICompatibleTravelProfileExtractor(
        LLMConfig(
            provider="deepseek",
            api_key=None,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            timeout_seconds=1,
        )
    )

    profile = extractor.extract("帮我规划北京两天，喜欢历史和美食，不要太累")

    assert profile.destination == "北京"
    assert profile.days == 2
    assert profile.pace == "relaxed"
    assert "history" in profile.interests
