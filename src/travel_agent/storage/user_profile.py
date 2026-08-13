"""旧 L3 API 的兼容门面；新代码应使用 user_memory repository/service。"""

from __future__ import annotations

from pathlib import Path

from travel_agent.schemas import TravelProfile
from travel_agent.storage.user_memory import (
    JsonUserMemoryRepository,
    PreferenceObservation,
    RecentTrip,
    UserMemoryService,
)

DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[3] / "data" / "profiles"


class UserProfileStore:
    """兼容旧调用；不会再把单次目的地/天数作为稳定画像返回。"""

    def __init__(self, profile_dir: Path | str = DEFAULT_PROFILE_DIR) -> None:
        self.repository = JsonUserMemoryRepository(profile_dir)
        self.service = UserMemoryService(self.repository)

    @property
    def profile_dir(self) -> Path:
        return self.repository.profile_dir

    def load(self, user_id: str) -> TravelProfile:
        return self.service.load_stable_profile(user_id)

    def save(self, user_id: str, profile: TravelProfile) -> None:
        """测试/迁移兼容入口：替换该用户数据后导入稳定字段与可选历史行程。"""
        self.repository.delete_user_memory(user_id)
        observations = _profile_observations(profile)
        if observations:
            self.repository.record_preference_events(user_id, "compat", 0, observations)
        if profile.destination or profile.days:
            self.repository.upsert_completed_trip(
                user_id,
                RecentTrip(
                    session_id="compat",
                    destination=profile.destination,
                    days=profile.days,
                    start_date=profile.start_date,
                    companions=profile.companions,
                    budget_level=profile.budget_level,
                    interests=list(profile.interests),
                    pace=profile.pace,
                    hotel_area=profile.hotel_area,
                    food_preference=list(profile.food_preference),
                    must_visit=list(profile.must_visit),
                    avoid=list(profile.avoid),
                    transport_mode=profile.transport_mode,
                    itinerary_summary="兼容接口导入",
                ),
            )

    def merge_and_save(self, user_id: str, update: TravelProfile) -> TravelProfile:
        """兼容旧写入；只合并稳定字段，调用方应迁移到 UserMemoryService。"""
        observations = _profile_observations(update)
        if observations:
            self.repository.record_preference_events(user_id, "compat", 0, observations)
        return self.load(user_id)


def _profile_observations(profile: TravelProfile) -> list[PreferenceObservation]:
    observations: list[PreferenceObservation] = []
    for category in ("interests", "food_preference", "avoid"):
        for value in getattr(profile, category):
            observations.append(PreferenceObservation(category, str(value)))
    if profile.budget_level:
        observations.append(PreferenceObservation("budget_level", profile.budget_level))
    if profile.pace != "standard":
        observations.append(PreferenceObservation("pace", profile.pace))
    if profile.transport_mode != "public_transport":
        observations.append(PreferenceObservation("transport_mode", profile.transport_mode))
    return observations


def _profile_to_dict(profile: TravelProfile) -> dict:
    """保留给 session_state 的旧序列化调用。"""
    from dataclasses import asdict

    return asdict(profile)


def _profile_from_dict(data: dict) -> TravelProfile:
    """保留给 session_state 与旧 JSON migration 的反序列化调用。"""
    return TravelProfile(
        destination=data.get("destination"),
        days=data.get("days"),
        start_date=data.get("start_date"),
        budget_level=data.get("budget_level"),
        budget_limit=data.get("budget_limit"),
        interests=list(data.get("interests", [])),
        companions=data.get("companions"),
        party_size=data.get("party_size"),
        pace=data.get("pace", "standard"),
        hotel_area=data.get("hotel_area"),
        food_preference=list(data.get("food_preference", [])),
        must_visit=list(data.get("must_visit", [])),
        avoid=list(data.get("avoid", [])),
        transport_mode=data.get("transport_mode", "public_transport"),
        constraint_state=dict(data.get("constraint_state") or {}),
    )
