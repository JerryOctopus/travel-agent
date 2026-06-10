from __future__ import annotations

from travel_agent.config import load_llm_config
from travel_agent.llm_extractor import RuleBasedTravelProfileExtractor
from travel_agent.openai_compatible_extractor import OpenAICompatibleTravelProfileExtractor


def build_extractor(use_llm: bool):
    if not use_llm:
        return RuleBasedTravelProfileExtractor()
    return OpenAICompatibleTravelProfileExtractor(load_llm_config())


def provider_label(extractor) -> str:
    if isinstance(extractor, OpenAICompatibleTravelProfileExtractor):
        status = "enabled" if extractor.config.enabled else "fallback(rule)"
        return f"extractor={extractor.__class__.__name__}:{status}"
    return f"extractor={extractor.__class__.__name__}"
