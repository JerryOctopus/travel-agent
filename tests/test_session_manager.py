from __future__ import annotations

from travel_agent.agent import toolkit
from travel_agent.settings import load_settings
from travel_agent.storage.session_manager import SessionLifecycleManager


def test_session_restore_after_persist(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    profile_dir = tmp_path / "profiles"
    mgr = SessionLifecycleManager(artifact_dir=artifact_dir, profile_dir=profile_dir)

    sid1, ctx1, hist1 = mgr.get_or_create(None, user_id="user_a")
    toolkit.update_travel_profile(ctx1, destination="杭州", days=2, interests=["food"])
    toolkit.search_poi(ctx1)
    hist1.append(("user", "杭州两天"))
    hist1.append(("assistant", "好的"))
    mgr.persist_turn(sid1, ctx1, hist1, user_id="user_a")

    mgr2 = SessionLifecycleManager(artifact_dir=artifact_dir, profile_dir=profile_dir)
    sid2, ctx2, hist2 = mgr2.get_or_create(sid1, user_id="user_a")

    assert sid2 == sid1
    assert ctx2.profile.destination == "杭州"
    assert ctx2.store.latest("candidates") is not None
    assert len(hist2) == 2
