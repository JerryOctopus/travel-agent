"""L3：跨会话用户偏好画像存储。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from travel_agent.schemas import TravelProfile
from travel_agent.workflow import merge_profile

DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[3] / "data" / "profiles"


class UserProfileStore:
    def __init__(self, profile_dir: Path | str = DEFAULT_PROFILE_DIR) -> None:
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in user_id)
        return self.profile_dir / f"{safe}.json"

    def load(self, user_id: str) -> TravelProfile:
        path = self._path(user_id)
        if not path.exists():
            return TravelProfile()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _profile_from_dict(data.get("profile", {}))
        except Exception:
            return TravelProfile()

    def save(self, user_id: str, profile: TravelProfile) -> None:
        path = self._path(user_id)
        record = {
            "user_id": user_id,
            "updated_at": time.time(),
            "profile": _profile_to_dict(profile),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def merge_and_save(self, user_id: str, update: TravelProfile) -> TravelProfile:
        merged = merge_profile(self.load(user_id), update)
        self.save(user_id, merged)
        return merged


def _profile_to_dict(profile: TravelProfile) -> dict:
    return asdict(profile)


def _profile_from_dict(data: dict) -> TravelProfile:
    return TravelProfile(
        destination=data.get("destination"),
        days=data.get("days"),
        start_date=data.get("start_date"),
        budget_level=data.get("budget_level"),
        interests=list(data.get("interests", [])),
        companions=data.get("companions"),
        pace=data.get("pace", "standard"),
        hotel_area=data.get("hotel_area"),
        food_preference=list(data.get("food_preference", [])),
        must_visit=list(data.get("must_visit", [])),
        avoid=list(data.get("avoid", [])),
        transport_mode=data.get("transport_mode", "public_transport"),
    )
