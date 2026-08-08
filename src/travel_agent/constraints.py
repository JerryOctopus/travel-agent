from __future__ import annotations

from dataclasses import dataclass, field

from travel_agent.schemas import BudgetLevel, Pace, TravelProfile, TransportMode


@dataclass(frozen=True)
class ConstraintSet:
    """可验证旅行约束的轻量结构。

    第一版作为 TravelProfile 和后续 ChinaTravel DSL 之间的桥，不改变现有主流程。
    """

    destination: str | None = None
    days: int | None = None
    budget_level: BudgetLevel | None = None
    budget_limit: float | None = None
    start_date: str | None = None
    companions: str | None = None
    party_size: int | None = None
    required_interests: list[str] = field(default_factory=list)
    cuisine_preferences: list[str] = field(default_factory=list)
    hotel_area: str | None = None
    transport_mode: TransportMode = "public_transport"
    must_visit: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    pace: Pace = "standard"

    @classmethod
    def from_profile(cls, profile: TravelProfile) -> "ConstraintSet":
        return cls(
            destination=profile.destination,
            days=profile.days,
            budget_level=profile.budget_level,
            budget_limit=profile.budget_limit,
            start_date=profile.start_date,
            companions=profile.companions,
            party_size=profile.party_size,
            required_interests=list(profile.interests),
            cuisine_preferences=list(profile.food_preference),
            hotel_area=profile.hotel_area,
            transport_mode=profile.transport_mode,
            must_visit=list(profile.must_visit),
            avoid=list(profile.avoid),
            pace=profile.pace,
        )

    def to_dict(self) -> dict:
        return {
            "destination": self.destination,
            "days": self.days,
            "budget_level": self.budget_level,
            "budget_limit": self.budget_limit,
            "start_date": self.start_date,
            "companions": self.companions,
            "party_size": self.party_size,
            "required_interests": list(self.required_interests),
            "cuisine_preferences": list(self.cuisine_preferences),
            "hotel_area": self.hotel_area,
            "transport_mode": self.transport_mode,
            "must_visit": list(self.must_visit),
            "avoid": list(self.avoid),
            "pace": self.pace,
        }
