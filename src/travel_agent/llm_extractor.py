from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from travel_agent.schemas import BudgetLevel, Pace, TravelProfile, TransportMode
from travel_agent.workflow_rules import extract_profile_rule_based


class TravelProfilePayload(TypedDict, total=False):
    destination: str | None
    days: int | None
    start_date: str | None
    budget_level: BudgetLevel | None
    interests: list[str]
    companions: str | None
    pace: Pace
    hotel_area: str | None
    food_preference: list[str]
    must_visit: list[str]
    avoid: list[str]
    transport_mode: TransportMode


class TravelProfileExtractor(Protocol):
    def extract(self, user_message: str) -> TravelProfile:
        """从用户自然语言中抽取结构化 TravelProfile。"""


@dataclass(frozen=True)
class RuleBasedTravelProfileExtractor:
    def extract(self, user_message: str) -> TravelProfile:
        return extract_profile_rule_based(user_message)


@dataclass(frozen=True)
class FakeLLMTravelProfileExtractor:
    """测试用 LLM extractor，模拟 structured output 返回。"""

    payload: TravelProfilePayload

    def extract(self, user_message: str) -> TravelProfile:
        return travel_profile_from_payload(self.payload)


def travel_profile_from_payload(payload: TravelProfilePayload) -> TravelProfile:
    return TravelProfile(
        destination=payload.get("destination"),
        days=_coerce_days(payload.get("days")),
        start_date=payload.get("start_date"),
        budget_level=_coerce_literal(
            payload.get("budget_level"),
            allowed={"low", "mid", "high"},
        ),
        interests=list(payload.get("interests", [])),
        companions=payload.get("companions"),
        pace=_coerce_literal(
            payload.get("pace", "standard"),
            allowed={"relaxed", "standard", "intensive"},
            default="standard",
        ),
        hotel_area=payload.get("hotel_area"),
        food_preference=list(payload.get("food_preference", [])),
        must_visit=list(payload.get("must_visit", [])),
        avoid=list(payload.get("avoid", [])),
        transport_mode=_coerce_literal(
            payload.get("transport_mode", "public_transport"),
            allowed={"walk", "public_transport", "taxi", "drive"},
            default="public_transport",
        ),
    )


def _coerce_days(value: object) -> int | None:
    if value is None:
        return None
    try:
        days = int(value)
    except (TypeError, ValueError):
        return None
    return days if days > 0 else None


def _coerce_literal(
    value: object,
    allowed: set[str],
    default: str | None = None,
):
    if isinstance(value, str) and value in allowed:
        return value
    return default
