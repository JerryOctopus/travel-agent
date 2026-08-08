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
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from travel_agent.agent.serde import poi_to_dict
from travel_agent.providers import TravelToolProvider, build_tool_provider
from travel_agent.schemas import POI, TravelProfile

# 当前 Subagent 任务元数据：由 SubagentRunner 在执行前 set，工具写入
# artifact 时未显式传元数据则自动补全，保证并发场景下每条 artifact 都能
# 关联到 request_id / task_id / agent（业务工具代码无需改动）。
_TASK_META: ContextVar[dict | None] = ContextVar("travel_agent_task_meta", default=None)


def set_current_task_meta(meta: dict | None):
    """设置当前任务元数据，返回 reset token（调用方负责在 finally 中复位）。"""
    return _TASK_META.set(meta)


def reset_task_meta(token) -> None:
    _TASK_META.reset(token)


def current_task_meta() -> dict:
    """返回当前 Subagent 任务元数据的只读快照。"""
    return dict(_TASK_META.get() or {})

DEFAULT_POI_PATH = Path(__file__).resolve().parents[3] / "data" / "seed" / "pois.json"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "data" / "artifacts"

# per-session 工具串行锁：LangGraph ToolNode 会并行执行同一 AIMessage 里的
# 多个 tool_call，而工具共享可变 ctx（store/profile/pois_by_id）。按 session
# 串行执行，fail-closed 防止并发写 artifact/画像竞态。
_SESSION_LOCKS: dict[str, threading.Lock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()

# 锁隔离白名单：这些工具内部会派生 Subagent，Subagent 的业务工具还要再次
# 获取同一把 session 锁；若把它们也包进 serialized()，不可重入的 threading.Lock
# 会同线程嵌套自锁 → 死锁。因此 dispatch 类工具全程不持 session 锁。
UNLOCKED_TOOL_NAMES: frozenset[str] = frozenset({"dispatch_subagent"})


def session_tool_lock(session_id: str) -> threading.Lock:
    """返回该 session 的工具串行锁；不同 session 之间互不阻塞。

    注意：返回的是**不可重入** ``threading.Lock``，同一线程嵌套 acquire 会死锁。
    """
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[session_id] = lock
        return lock


def should_serialize_tool(tool_name: str) -> bool:
    """判定工具是否需要 per-session 串行锁（dispatch 类工具永远不加锁）。"""
    return tool_name not in UNLOCKED_TOOL_NAMES


def holds_session_lock(session_id: str) -> bool:
    """非阻塞探测当前线程是否已持有该 session 锁（用于防嵌套断言）。"""
    lock = session_tool_lock(session_id)
    if lock.acquire(blocking=False):
        lock.release()
        return False
    return True


@dataclass
class ArtifactStore:
    """会话级 artifact 存储：内存为主，可选落盘持久化。

    并发契约（多 Agent 架构）：

    - 所有公开方法内部持 ``RLock`` 短暂加锁，读改写原子完成，外部无需再包锁；
    - 写入方可携带 ``request_id / task_id / agent / data_source`` 元数据，
      并发场景下按关联字段精确定位结果；
    - Planner 必须按明确 ``artifact_id`` 读取（``get`` / ``get_record``），
      不得依赖 ``latest()`` 获取并行领域结果。
    """

    session_id: str
    artifact_dir: Path | None = None
    _items: dict[str, dict] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def put(
        self,
        kind: str,
        payload: dict,
        *,
        request_id: str | None = None,
        task_id: str | None = None,
        agent: str | None = None,
        data_source: str | None = None,
    ) -> str:
        artifact_id = f"{kind}_{uuid.uuid4().hex[:8]}"
        record = {
            "artifact_id": artifact_id,
            "kind": kind,
            "session_id": self.session_id,
            "created_at": time.time(),
            "payload": payload,
        }
        if request_id is not None:
            record["request_id"] = request_id
        if task_id is not None:
            record["task_id"] = task_id
        if agent is not None:
            record["agent"] = agent
        if data_source is not None:
            record["data_source"] = data_source
        # 未显式传入的元数据从当前 Subagent 任务上下文自动补全。
        meta = _TASK_META.get()
        if meta:
            record.setdefault("request_id", meta.get("request_id"))
            record.setdefault("task_id", meta.get("task_id"))
            record.setdefault("agent", meta.get("agent"))
        with self._lock:
            self._items[artifact_id] = record
        self._persist(record)
        return artifact_id

    def get(self, artifact_id: str) -> dict | None:
        with self._lock:
            record = self._items.get(artifact_id)
        return record["payload"] if record else None

    def get_record(self, artifact_id: str) -> dict | None:
        """返回完整记录（含元数据），供 Planner/审计校验产出者与状态。"""
        with self._lock:
            record = self._items.get(artifact_id)
        return dict(record) if record else None

    def get_payloads(self, artifact_ids: list[str]) -> list[dict]:
        """按明确 artifact_id 列表批量取 payload，缺失的 id 跳过。"""
        payloads: list[dict] = []
        with self._lock:
            for artifact_id in artifact_ids:
                record = self._items.get(artifact_id)
                if record is not None:
                    payloads.append(record["payload"])
        return payloads

    def artifact_ids(self) -> set[str]:
        """当前全部 artifact id 快照（供执行前后 diff 归因新产出）。"""
        with self._lock:
            return set(self._items.keys())

    def artifact_ids_for_task(
        self,
        request_id: str,
        task_id: str,
        *,
        agent: str | None = None,
    ) -> list[str]:
        """按任务元数据返回该任务自己产出的 artifact，禁止全局 diff 串线。"""
        with self._lock:
            records = [
                record
                for record in self._items.values()
                if record.get("request_id") == request_id
                and record.get("task_id") == task_id
                and (agent is None or record.get("agent") == agent)
            ]
        records.sort(key=lambda record: (record["created_at"], record["artifact_id"]))
        return [str(record["artifact_id"]) for record in records]

    def latest(self, kind: str) -> dict | None:
        record = self._latest_record(kind)
        return record["payload"] if record else None

    def latest_id(self, kind: str) -> str | None:
        record = self._latest_record(kind)
        return record["artifact_id"] if record else None

    def _latest_record(self, kind: str) -> dict | None:
        with self._lock:
            records = [r for r in self._items.values() if r["kind"] == kind]
        if not records:
            return None
        return max(records, key=lambda r: (r["created_at"], r["artifact_id"]))

    def load_from_disk(self) -> int:
        """从落盘目录恢复本会话 artifacts（M4 L2）。"""
        if self.artifact_dir is None:
            return 0
        session_dir = self.artifact_dir / self.session_id
        if not session_dir.exists():
            return 0
        loaded = 0
        with self._lock:
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
        with self._lock:
            items = list(self._items.values())

        def _latest(kind: str) -> dict | None:
            records = [r for r in items if r["kind"] == kind]
            if not records:
                return None
            return max(records, key=lambda r: (r["created_at"], r["artifact_id"]))["payload"]

        weather = _latest("weather")
        if weather:
            lines.append(
                f"- 天气：{weather.get('city')} {weather.get('condition')} "
                f"{weather.get('temperature_c')}°C"
            )
        candidates = _latest("candidates")
        if candidates:
            lines.append(f"- 已检索 POI：{candidates.get('city')} 共 {len(candidates.get('pois', []))} 个")
        ranked = _latest("ranked")
        if ranked:
            lines.append(f"- 已打分候选：{len(ranked.get('pois', []))} 个")
        itinerary_pack = _latest("itinerary")
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
    pending_preference_observations: list[dict[str, str]] = field(default_factory=list)
    evaluation_trace_enabled: bool = False
    evaluation_trace: list[dict] = field(default_factory=list)

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
