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
import copy
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from travel_agent.agent.serde import poi_to_dict
from travel_agent.providers import TravelToolProvider, build_tool_provider
from travel_agent.schemas import POI, TravelProfile
from travel_agent.hybrid_planning.call_cache import HybridCallCache

# 当前 Subagent 任务元数据：由 SubagentRunner 在执行前 set，工具写入
# artifact 时未显式传元数据则自动补全，保证并发场景下每条 artifact 都能
# 关联到 request_id / task_id / agent（业务工具代码无需改动）。
_TASK_META: ContextVar[dict | None] = ContextVar("travel_agent_task_meta", default=None)


class RequestCancelledError(RuntimeError):
    """Raised when work tries to mutate a cancelled request snapshot."""


@dataclass
class RequestControl:
    """Thread-safe request lifetime marker shared by a request snapshot and its workers."""

    request_id: str
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def check_active(self) -> None:
        if self.cancelled:
            raise RequestCancelledError(f"request cancelled: {self.request_id}")


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
# 多个 tool_call。共享工具契约仅对显式声明为非 parallel_safe 的工具加锁；
# 查询型工具依赖 ArtifactStore/profile 自身的短锁，可继续并发执行。
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
    """Return the fail-closed concurrency decision for one logical tool."""
    if tool_name in UNLOCKED_TOOL_NAMES:
        return False
    from travel_agent.agent.tool_contract import tool_execution_policy

    return not tool_execution_policy(tool_name).parallel_safe


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
    request_control: RequestControl | None = field(default=None, repr=False, compare=False)
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
        if self.request_control is not None:
            self.request_control.check_active()
        meta = _TASK_META.get()
        if kind == "itinerary" and not isinstance(payload.get("state_version"), dict):
            task_version = (meta or {}).get("constraint_version")
            if isinstance(task_version, dict):
                payload = copy.deepcopy(payload)
                payload["state_version"] = {
                    "constraint_revision": task_version.get("revision"),
                    "constraint_hash": task_version.get("constraint_hash"),
                    "constraint_snapshot": copy.deepcopy(
                        task_version.get("constraint_snapshot") or {}
                    ),
                }
        artifact_id = f"{kind}_{uuid.uuid4().hex[:8]}"
        record = {
            "artifact_id": artifact_id,
            "kind": kind,
            "session_id": self.session_id,
            "created_at": time.time(),
            # Store owns its payload.  Neither the caller nor a reader may keep
            # a mutable alias to state protected by the store lock.
            "payload": copy.deepcopy(payload),
        }
        if kind == "itinerary":
            record["artifact_status"] = str(payload.get("artifact_status") or "current")
            version = payload.get("state_version") or {}
            record["constraint_revision"] = version.get("constraint_revision")
            record["constraint_hash"] = version.get("constraint_hash")
            record["constraint_snapshot"] = copy.deepcopy(version.get("constraint_snapshot") or {})
            record["parent_artifact_id"] = payload.get("parent_plan_artifact_id")
            record["revision_lineage"] = list(payload.get("revision_lineage") or [])
            validation = payload.get("validation_result") or {}
            record["validation_constraint_revision"] = validation.get(
                "validated_constraint_revision"
            )
            record["validation_constraint_hash"] = validation.get(
                "validated_constraint_hash"
            )
        elif payload.get("artifact_status"):
            # Read-only diagnostics participate in audit lookup but never in
            # the current deliverable lifecycle.
            record["artifact_status"] = str(payload["artifact_status"])
        if request_id is not None:
            record["request_id"] = request_id
        if task_id is not None:
            record["task_id"] = task_id
        if agent is not None:
            record["agent"] = agent
        if data_source is not None:
            record["data_source"] = data_source
        # 未显式传入的元数据从当前 Subagent 任务上下文自动补全。
        if meta:
            record.setdefault("request_id", meta.get("request_id"))
            record.setdefault("task_id", meta.get("task_id"))
            record.setdefault("agent", meta.get("agent"))
            state = meta.get("constraint_state")
            if isinstance(state, dict):
                from travel_agent.artifact_policy import constraint_basis, constraint_fingerprint

                record["constraint_basis"] = constraint_basis(kind, state, payload)
                record["constraint_fingerprint"] = constraint_fingerprint(kind, state, payload)
        with self._lock:
            if kind == "itinerary" and record["artifact_status"] == "current":
                for previous in self._items.values():
                    if (
                        previous.get("kind") == "itinerary"
                        and previous.get("artifact_status", "current") == "current"
                    ):
                        previous["artifact_status"] = "historical"
                        previous["payload"]["artifact_status"] = "historical"
                        self._persist(previous)
            self._items[artifact_id] = record
        self._persist(record)
        return artifact_id

    def snapshot_records(self) -> dict[str, dict]:
        """Return a deep snapshot suitable for isolated request/task execution."""
        with self._lock:
            return copy.deepcopy(self._items)

    def merge_records(self, records: dict[str, dict]) -> None:
        """Import records without changing their IDs; cancelled snapshots cannot commit."""
        if self.request_control is not None:
            self.request_control.check_active()
        with self._lock:
            for artifact_id, record in records.items():
                if artifact_id not in self._items:
                    copied = copy.deepcopy(record)
                    self._items[artifact_id] = copied
                    self._persist(copied)

    def get(self, artifact_id: str) -> dict | None:
        with self._lock:
            record = self._items.get(artifact_id)
            return copy.deepcopy(record["payload"]) if record else None

    def get_record(self, artifact_id: str) -> dict | None:
        """返回完整记录（含元数据），供 Planner/审计校验产出者与状态。"""
        with self._lock:
            record = self._items.get(artifact_id)
            return copy.deepcopy(record) if record else None

    def get_payloads(self, artifact_ids: list[str]) -> list[dict]:
        """按明确 artifact_id 列表批量取 payload，缺失的 id 跳过。"""
        payloads: list[dict] = []
        with self._lock:
            for artifact_id in artifact_ids:
                record = self._items.get(artifact_id)
                if record is not None:
                    payloads.append(copy.deepcopy(record["payload"]))
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

    def artifact_ids_for_request(self, request_id: str, *, kind: str | None = None) -> list[str]:
        """Return only artifacts owned by one request, ordered by occurrence."""
        with self._lock:
            records = [
                copy.deepcopy(record)
                for record in self._items.values()
                if record.get("request_id") == request_id
                and (kind is None or record.get("kind") == kind)
            ]
        records.sort(key=lambda record: (record["created_at"], record["artifact_id"]))
        return [str(record["artifact_id"]) for record in records]

    def latest_id_for_request(self, request_id: str, kind: str) -> str | None:
        ids = self.artifact_ids_for_request(request_id, kind=kind)
        return ids[-1] if ids else None

    def latest(self, kind: str) -> dict | None:
        record = self._latest_record(kind)
        return copy.deepcopy(record["payload"]) if record else None

    def latest_id(self, kind: str) -> str | None:
        record = self._latest_record(kind)
        return record["artifact_id"] if record else None

    def latest_current_id(self, kind: str) -> str | None:
        record = self._latest_record(kind, statuses={"current"})
        return record["artifact_id"] if record else None

    def latest_current(self, kind: str) -> dict | None:
        record = self._latest_record(kind, statuses={"current"})
        return copy.deepcopy(record["payload"]) if record else None

    def latest_revisable_id(self, kind: str) -> str | None:
        """Return the newest artifact that may be used only as revision context.

        A stale or historical itinerary is deliberately revisable but is never
        deliverable.  Validation failures are excluded because they never
        became a usable plan in the first place.
        """
        record = self._latest_record(kind, statuses={"current", "stale", "historical"})
        return record["artifact_id"] if record else None

    def latest_revisable(self, kind: str) -> dict | None:
        record = self._latest_record(kind, statuses={"current", "stale", "historical"})
        return copy.deepcopy(record["payload"]) if record else None

    def promote_itinerary(
        self,
        artifact_id: str,
        *,
        limitations: list[str] | None = None,
        review_advisories: list[str] | None = None,
        resolved_review_issues: list[str] | None = None,
        promotion_reason: str = "finalizer_acceptance",
    ) -> bool:
        """Atomically promote one validated candidate and supersede current.

        No current artifact is demoted until the selected candidate has passed
        the immutable revision/hash checks.  This keeps failed rework attempts
        from creating an empty or partially advanced final state.
        """
        with self._lock:
            record = self._items.get(artifact_id)
            if not record or record.get("kind") != "itinerary":
                return False
            status = str(record.get("artifact_status") or "")
            if status == "current":
                return True
            if status != "candidate":
                return False
            payload = record.get("payload") or {}
            validation = payload.get("validation_result") or {}
            state_version = payload.get("state_version") or {}
            if (
                validation.get("passed") is not True
                or (payload.get("critic") or {}).get("passed") is not True
                or state_version.get("constraint_revision")
                != validation.get("validated_constraint_revision")
                or state_version.get("constraint_hash")
                != validation.get("validated_constraint_hash")
            ):
                return False
            for previous in self._items.values():
                if (
                    previous.get("kind") == "itinerary"
                    and previous.get("artifact_id") != artifact_id
                    and previous.get("artifact_status", "current") == "current"
                ):
                    previous["artifact_status"] = "historical"
                    previous["payload"]["artifact_status"] = "historical"
                    previous["superseded_by"] = artifact_id
                    self._persist(previous)
            normalized_limitations = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in [
                        *list(payload.get("limitations") or []),
                        *list(limitations or []),
                    ]
                    if str(item).strip()
                )
            )
            if normalized_limitations:
                payload["limitations"] = normalized_limitations
            normalized_advisories = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in [
                        *list(payload.get("review_advisories") or []),
                        *list(review_advisories or []),
                    ]
                    if str(item).strip()
                )
            )
            if normalized_advisories:
                payload["review_advisories"] = normalized_advisories
            normalized_resolved = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in [
                        *list(payload.get("resolved_review_issues") or []),
                        *list(resolved_review_issues or []),
                    ]
                    if str(item).strip()
                )
            )
            if normalized_resolved:
                payload["resolved_review_issues"] = normalized_resolved
                payload["repair_verification_pending"] = []
            payload["artifact_status"] = "current"
            payload["promotion_reason"] = promotion_reason
            record["artifact_status"] = "current"
            record["promotion_reason"] = promotion_reason
            record["promoted_at"] = time.time()
            self._persist(record)
            return True

    def reject_itinerary_candidate(
        self,
        artifact_id: str,
        *,
        reason: str,
        status: str = "review_failed",
    ) -> bool:
        """Mark a non-current attempt rejected without touching current state."""
        with self._lock:
            record = self._items.get(artifact_id)
            if (
                not record
                or record.get("kind") != "itinerary"
                or record.get("artifact_status") == "current"
            ):
                return False
            record["artifact_status"] = status
            record["rejection_reason"] = reason
            record["payload"]["artifact_status"] = status
            record["payload"]["rejection_reason"] = reason
            self._persist(record)
            return True

    def invalidate_itineraries(
        self, *, constraint_revision: int, constraint_hash: str,
        changed_fields: list[str], reason: str = "material_constraint_change",
    ) -> list[str]:
        """Mark current plans stale and revoke validation against active constraints."""
        invalidated: list[str] = []
        with self._lock:
            for record in self._items.values():
                if record.get("kind") != "itinerary":
                    continue
                if record.get("artifact_status", "current") != "current":
                    continue
                record["artifact_status"] = "stale"
                record["stale_reason"] = reason
                record["stale_at_constraint_revision"] = constraint_revision
                record["stale_at_constraint_hash"] = constraint_hash
                record["validation_status"] = "stale"
                record["active_constraint_revision"] = constraint_revision
                record["active_constraint_hash"] = constraint_hash
                payload = record["payload"]
                payload["artifact_status"] = "stale"
                payload["stale_reason"] = reason
                payload["stale_changed_fields"] = list(changed_fields)
                prior_validation = copy.deepcopy(payload.get("validation_result") or {})
                payload["historical_validation_result"] = prior_validation
                payload["validation_result"] = {
                    "passed": False,
                    "status": "stale",
                    "issues": [{
                        "code": "stale_constraint_revision",
                        "message": "该计划基于旧约束，必须按当前状态重建并重新校验。",
                        "severity": "error",
                    }],
                    "validated_constraint_revision": (
                        prior_validation.get("validated_constraint_revision")
                        or (payload.get("state_version") or {}).get("constraint_revision")
                    ),
                    "active_constraint_revision": constraint_revision,
                    "active_constraint_hash": constraint_hash,
                }
                invalidated.append(str(record["artifact_id"]))
                self._persist(record)
        return invalidated

    def _latest_record(self, kind: str, statuses: set[str] | None = None) -> dict | None:
        with self._lock:
            records = [
                r for r in self._items.values()
                if r["kind"] == kind
                and (
                    statuses is None
                    or str(r.get("artifact_status", "current")) in statuses
                )
            ]
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
            records = [
                r for r in items
                if r["kind"] == kind
                and (
                    kind != "itinerary"
                    or r.get("artifact_status", "current") == "current"
                )
            ]
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
    reference_datetime: str | None = None
    runtime_settings: Any | None = field(default=None, repr=False, compare=False)
    hybrid_llm_client: Any | None = field(default=None, repr=False, compare=False)
    hybrid_call_cache: HybridCallCache = field(
        default_factory=HybridCallCache, repr=False, compare=False
    )
    hybrid_request_scope: str | None = field(default=None, repr=False, compare=False)
    active_task_type: str | None = field(default=None, repr=False, compare=False)
    active_delivery_intent: str | None = field(default=None, repr=False, compare=False)
    request_control: RequestControl | None = field(default=None, repr=False)
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def remember_pois(self, pois: list[POI]) -> None:
        with self._state_lock:
            for poi in pois:
                self.pois_by_id[poi.poi_id] = poi

    def poi(self, poi_id: str) -> POI | None:
        with self._state_lock:
            return self.pois_by_id.get(poi_id)

    def clone_isolated(self, control: RequestControl | None = None) -> "SessionContext":
        """Copy mutable request state while sharing only the provider implementation."""
        active_control = control or self.request_control
        return SessionContext(
            session_id=self.session_id,
            provider=self.provider,
            store=ArtifactStore(
                session_id=self.session_id,
                artifact_dir=None,
                _items=self.store.snapshot_records(),
                request_control=active_control,
            ),
            profile=copy.deepcopy(self.profile),
            pois_by_id=copy.deepcopy(self.pois_by_id),
            pending_preference_observations=copy.deepcopy(
                self.pending_preference_observations
            ),
            evaluation_trace_enabled=self.evaluation_trace_enabled,
            evaluation_trace=copy.deepcopy(self.evaluation_trace),
            reference_datetime=self.reference_datetime,
            runtime_settings=self.runtime_settings,
            hybrid_llm_client=self.hybrid_llm_client,
            hybrid_call_cache=self.hybrid_call_cache,
            hybrid_request_scope=self.hybrid_request_scope,
            active_task_type=self.active_task_type,
            active_delivery_intent=self.active_delivery_intent,
            request_control=active_control,
        )

    def commit_from(self, snapshot: "SessionContext") -> None:
        """Atomically publish a completed snapshot into this live session."""
        if snapshot.request_control is not None:
            snapshot.request_control.check_active()
        self.store.merge_records(snapshot.store.snapshot_records())
        with self._state_lock:
            self.profile = copy.deepcopy(snapshot.profile)
            self.pois_by_id = copy.deepcopy(snapshot.pois_by_id)
            self.pending_preference_observations = copy.deepcopy(
                snapshot.pending_preference_observations
            )
            self.evaluation_trace = copy.deepcopy(snapshot.evaluation_trace)

    def merge_task_from(self, snapshot: "SessionContext", base_ids: set[str]) -> None:
        """Publish only a successful isolated task's new artifacts and POI cache."""
        if snapshot.request_control is not None:
            snapshot.request_control.check_active()
        records = snapshot.store.snapshot_records()
        self.store.merge_records(
            {artifact_id: record for artifact_id, record in records.items() if artifact_id not in base_ids}
        )
        with self._state_lock:
            self.pois_by_id.update(copy.deepcopy(snapshot.pois_by_id))
            if snapshot.evaluation_trace_enabled:
                existing = len(self.evaluation_trace)
                self.evaluation_trace.extend(copy.deepcopy(snapshot.evaluation_trace[existing:]))


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
