from __future__ import annotations

from travel_agent.schemas import TravelProfile
from travel_agent.storage.session_manager import SessionLifecycleManager
from travel_agent.storage.user_profile import UserProfileStore


def test_new_session_does_not_inject_l3_days(tmp_path) -> None:
    profile_dir = tmp_path / "profiles"
    artifact_dir = tmp_path / "artifacts"
    UserProfileStore(profile_dir).save(
        "user_a",
        TravelProfile(destination="新西兰", days=5, interests=["food"]),
    )

    mgr = SessionLifecycleManager(artifact_dir=artifact_dir, profile_dir=profile_dir)
    _, ctx, _ = mgr.get_or_create(None, user_id="user_a")

    assert ctx.profile.destination is None
    assert ctx.profile.days is None
    assert ctx.profile.interests == []
