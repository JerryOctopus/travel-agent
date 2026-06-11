"""会话生命周期管理：隔离、持久化、重启恢复（替换 server 内联版）。"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from travel_agent.agent.serde import poi_from_dict, poi_to_dict
from travel_agent.agent.session import DEFAULT_ARTIFACT_DIR, SessionContext, build_session
from travel_agent.schemas import TravelProfile
from travel_agent.storage.user_profile import UserProfileStore, _profile_from_dict, _profile_to_dict

SESSION_STATE_FILE = "session_state.json"


class SessionLifecycleManager:
    def __init__(
        self,
        artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
        profile_dir: Path | str | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._profile_store = UserProfileStore(profile_dir) if profile_dir else UserProfileStore()

    def get_or_create(
        self,
        session_id: str | None,
        user_id: str = "default",
    ) -> tuple[str, SessionContext, list[tuple[str, str]]]:
        if session_id and session_id in self._sessions:
            entry = self._sessions[session_id]
            return session_id, entry["ctx"], entry["history"]

        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        restored = self._restore_from_disk(sid, user_id)
        if restored:
            ctx, history = restored
            self._sessions[sid] = {"ctx": ctx, "history": history, "user_id": user_id}
            return sid, ctx, history

        ctx = build_session(session_id=sid, persist=True)
        l3 = self._profile_store.load(user_id)
        if l3.destination or l3.interests or l3.budget_level:
            ctx.profile = _merge_profiles(l3, ctx.profile)
        history: list[tuple[str, str]] = []
        self._sessions[sid] = {"ctx": ctx, "history": history, "user_id": user_id}
        return sid, ctx, history

    def persist_turn(
        self,
        session_id: str,
        ctx: SessionContext,
        history: list[tuple[str, str]],
        user_id: str = "default",
    ) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["history"] = history
        self._profile_store.merge_and_save(user_id, ctx.profile)
        self._save_session_state(session_id, ctx, history, user_id)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        session_dir = self.artifact_dir / session_id
        state_path = session_dir / SESSION_STATE_FILE
        if state_path.exists():
            state_path.unlink()

    def _restore_from_disk(
        self,
        session_id: str,
        user_id: str,
    ) -> tuple[SessionContext, list[tuple[str, str]]] | None:
        state_path = self.artifact_dir / session_id / SESSION_STATE_FILE
        if not state_path.exists():
            loaded = self._try_restore_from_artifacts_only(session_id, user_id)
            return loaded

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._try_restore_from_artifacts_only(session_id, user_id)

        ctx = build_session(session_id=session_id, persist=True)
        ctx.store.load_from_disk()
        ctx.profile = _profile_from_dict(data.get("profile", {}))
        for poi_data in data.get("pois_by_id", {}).values():
            poi = poi_from_dict(poi_data)
            ctx.pois_by_id[poi.poi_id] = poi
        history = [tuple(pair) for pair in data.get("history", [])]
        l3 = self._profile_store.load(user_id)
        ctx.profile = _merge_profiles(l3, ctx.profile)
        return ctx, history

    def _try_restore_from_artifacts_only(
        self,
        session_id: str,
        user_id: str,
    ) -> tuple[SessionContext, list[tuple[str, str]]] | None:
        session_dir = self.artifact_dir / session_id
        if not session_dir.exists() or not any(session_dir.glob("*.json")):
            return None
        ctx = build_session(session_id=session_id, persist=True)
        count = ctx.store.load_from_disk()
        if count == 0:
            return None
        l3 = self._profile_store.load(user_id)
        ctx.profile = _merge_profiles(l3, ctx.profile)
        return ctx, []

    def _save_session_state(
        self,
        session_id: str,
        ctx: SessionContext,
        history: list[tuple[str, str]],
        user_id: str,
    ) -> None:
        session_dir = self.artifact_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "session_id": session_id,
            "user_id": user_id,
            "updated_at": time.time(),
            "profile": _profile_to_dict(ctx.profile),
            "pois_by_id": {pid: poi_to_dict(poi) for pid, poi in ctx.pois_by_id.items()},
            "history": history,
        }
        path = session_dir / SESSION_STATE_FILE
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_profiles(base: TravelProfile, update: TravelProfile) -> TravelProfile:
    from travel_agent.workflow import merge_profile

    return merge_profile(base, update)
