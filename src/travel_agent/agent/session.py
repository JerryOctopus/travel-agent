"""会话上下文与 artifact 存储。

每个对话 session 拥有一个 ``SessionContext``，它持有：

- ``provider``：真实高德优先、本地 seed 兜底的工具数据源；
- ``store``：本会话的 artifact 存储（工具结果落盘/落内存，供后续工具复用）；
- ``profile``：跨多轮累积的出行画像。

工具之间不通过 LLM 传递大对象（POI 列表等），而是把结果存进 ``store`` 并返回
紧凑摘要 + ``artifact_id``，让 LLM 只负责「编排决策」，符合 PROJECT_PLAN 中
「工具拆细、LLM 真正编排」的要求，同时避免 token 爆炸。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from travel_agent.agent.serde import poi_to_dict
from travel_agent.providers import TravelToolProvider, build_tool_provider
from travel_agent.schemas import POI, TravelProfile

DEFAULT_POI_PATH = Path(__file__).resolve().parents[3] / "data" / "seed" / "pois.json"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "data" / "artifacts"


@dataclass
class ArtifactStore:
    """会话级 artifact 存储：内存为主，可选落盘持久化。"""

    session_id: str
    artifact_dir: Path | None = None
    _items: dict[str, dict] = field(default_factory=dict)

    def put(self, kind: str, payload: dict) -> str:
        artifact_id = f"{kind}_{uuid.uuid4().hex[:8]}"
        record = {
            "artifact_id": artifact_id,
            "kind": kind,
            "session_id": self.session_id,
            "created_at": time.time(),
            "payload": payload,
        }
        self._items[artifact_id] = record
        self._persist(record)
        return artifact_id

    def get(self, artifact_id: str) -> dict | None:
        record = self._items.get(artifact_id)
        return record["payload"] if record else None

    def latest(self, kind: str) -> dict | None:
        records = [r for r in self._items.values() if r["kind"] == kind]
        if not records:
            return None
        newest = max(records, key=lambda r: r["created_at"])
        return newest["payload"]

    def latest_id(self, kind: str) -> str | None:
        records = [r for r in self._items.values() if r["kind"] == kind]
        if not records:
            return None
        return max(records, key=lambda r: r["created_at"])["artifact_id"]

    def load_from_disk(self) -> int:
        """从落盘目录恢复本会话 artifacts（M4 L2）。"""
        if self.artifact_dir is None:
            return 0
        session_dir = self.artifact_dir / self.session_id
        if not session_dir.exists():
            return 0
        loaded = 0
        for path in session_dir.glob("*.json"):
            if path.name == "session_state.json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            artifact_id = record.get("artifact_id")
            if artifact_id:
                self._items[artifact_id] = record
                loaded += 1
        return loaded

    def build_prompt_snapshot(self) -> str:
        """L2：把本会话工具结果压缩成 system prompt 可注入的快照。"""
        lines: list[str] = []
        weather = self.latest("weather")
        if weather:
            lines.append(
                f"- 天气：{weather.get('city')} {weather.get('condition')} "
                f"{weather.get('temperature_c')}°C"
            )
        candidates = self.latest("candidates")
        if candidates:
            lines.append(f"- 已检索 POI：{candidates.get('city')} 共 {len(candidates.get('pois', []))} 个")
        ranked = self.latest("ranked")
        if ranked:
            lines.append(f"- 已打分候选：{len(ranked.get('pois', []))} 个")
        itinerary_pack = self.latest("itinerary")
        if itinerary_pack:
            itin = itinerary_pack.get("itinerary", {})
            critic = itinerary_pack.get("critic", {})
            lines.append(
                f"- 已有行程：{itin.get('summary', '')}；critic通过={critic.get('passed')}"
            )
        if not lines:
            return "（本会话尚无工具结果快照）"
        return "\n".join(lines)

    def _persist(self, record: dict) -> None:
        if self.artifact_dir is None:
            return
        try:
            session_dir = self.artifact_dir / self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / f"{record['artifact_id']}.json"
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            # 持久化失败不应影响主流程（例如沙箱/只读环境）。
            pass


@dataclass
class SessionContext:
    session_id: str
    provider: TravelToolProvider
    store: ArtifactStore
    profile: TravelProfile = field(default_factory=TravelProfile)
    pois_by_id: dict[str, POI] = field(default_factory=dict)

    def remember_pois(self, pois: list[POI]) -> None:
        for poi in pois:
            self.pois_by_id[poi.poi_id] = poi

    def poi(self, poi_id: str) -> POI | None:
        return self.pois_by_id.get(poi_id)


def build_session(
    session_id: str | None = None,
    poi_path: Path | str = DEFAULT_POI_PATH,
    persist: bool = True,
) -> SessionContext:
    sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
    store = ArtifactStore(
        session_id=sid,
        artifact_dir=DEFAULT_ARTIFACT_DIR if persist else None,
    )
    provider = build_tool_provider(poi_path)
    return SessionContext(session_id=sid, provider=provider, store=store)


def poi_dicts(pois: list[POI]) -> list[dict]:
    return [poi_to_dict(poi) for poi in pois]
