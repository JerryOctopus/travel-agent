"""Step 2 并发、死锁与 Artifact 关联测试。

覆盖：

- ArtifactStore 并发 put 不丢记录、元数据完整、按 artifact_id 精确读取；
- Planner 按明确 artifact_id 列表读取（不依赖 latest），产出者校验；
- AgentTraceLog 并发追加不丢记录、禁止 latest→改→put 反模式；
- per-session 锁不可重入的事实确认 + dispatch 嵌套死锁复现与隔离验证；
- lc_tools serialized / MCP 包装对 dispatch 类工具的锁跳过。
"""

from __future__ import annotations

import threading
import time

from travel_agent.agent.session import (
    ArtifactStore,
    build_session,
    holds_session_lock,
    session_tool_lock,
    should_serialize_tool,
)
from travel_agent.orchestration.multi_agent import (
    AgentTraceLog,
    read_artifacts_for_planner,
)

THREADS = 8
PER_THREAD = 50


def _run_threads(target, n: int = THREADS) -> list[threading.Thread]:
    threads = [threading.Thread(target=target) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "线程未能在超时内结束（疑似死锁）"
    return threads


# --- ArtifactStore 并发 ------------------------------------------------------ #


def test_session_clone_waits_for_atomic_state_snapshot() -> None:
    """A worker clone must not copy POIs while another worker is merging them."""
    ctx = build_session(session_id="sess_clone_snapshot", persist=False)
    started = threading.Event()
    completed = threading.Event()

    def clone_worker() -> None:
        started.set()
        ctx.clone_isolated()
        completed.set()

    ctx._state_lock.acquire()
    try:
        thread = threading.Thread(target=clone_worker)
        thread.start()
        assert started.wait(timeout=1)
        assert completed.wait(timeout=0.05) is False
    finally:
        ctx._state_lock.release()
    thread.join(timeout=1)
    assert completed.is_set()


def test_artifact_store_concurrent_put_loses_nothing():
    store = ArtifactStore(session_id="sess_conc")
    ids: list[str] = []
    ids_guard = threading.Lock()

    def worker():
        for i in range(PER_THREAD):
            artifact_id = store.put("candidates", {"i": i})
            with ids_guard:
                ids.append(artifact_id)

    _run_threads(worker)
    assert len(ids) == THREADS * PER_THREAD
    assert len(set(ids)) == THREADS * PER_THREAD  # id 全局唯一
    for artifact_id in ids:
        assert store.get(artifact_id) is not None


def test_artifact_store_metadata_and_get_record():
    store = ArtifactStore(session_id="sess_meta")
    artifact_id = store.put(
        "candidates",
        {"pois": []},
        request_id="req_1",
        task_id="attraction-abc",
        agent="attraction",
        data_source="amap",
    )
    record = store.get_record(artifact_id)
    assert record is not None
    assert record["request_id"] == "req_1"
    assert record["task_id"] == "attraction-abc"
    assert record["agent"] == "attraction"
    assert record["data_source"] == "amap"
    # get_record 返回拷贝，外部篡改不影响存储
    record["payload"] = {"hacked": True}
    assert store.get(artifact_id) == {"pois": []}
    # 旧式调用不携带元数据也兼容
    legacy_id = store.put("weather", {"city": "杭州"})
    assert "agent" not in store.get_record(legacy_id)


def test_artifact_store_get_payloads_by_explicit_ids():
    store = ArtifactStore(session_id="sess_batch")
    a = store.put("candidates", {"n": 1}, agent="attraction")
    b = store.put("hotels", {"n": 2}, agent="hotel")
    payloads = store.get_payloads([a, b, "missing_id"])
    assert payloads == [{"n": 1}, {"n": 2}]  # 缺失 id 被跳过
    assert store.get_payloads([]) == []


def test_artifact_store_latest_is_deterministic_under_same_timestamp():
    store = ArtifactStore(session_id="sess_latest")
    ids = [store.put("candidates", {"n": i}) for i in range(5)]
    # 同一时刻写入的多条同 kind 记录，latest 必须稳定（artifact_id 兜底排序）
    assert store.latest_id("candidates") in ids
    assert store.latest_id("candidates") == store.latest_id("candidates")


def test_artifact_store_concurrent_mixed_read_write():
    store = ArtifactStore(session_id="sess_rw")
    stop = threading.Event()
    errors: list[Exception] = []

    def writer():
        for i in range(PER_THREAD):
            store.put("candidates", {"i": i}, agent="attraction")

    def reader():
        while not stop.is_set():
            try:
                store.latest("candidates")
                store.build_prompt_snapshot()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for thread in readers:
        thread.start()
    writers = [threading.Thread(target=writer) for _ in range(THREADS)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(timeout=10)
    stop.set()
    for thread in readers:
        thread.join(timeout=10)
    assert not errors


# --- Planner 按 artifact_id 读取 ---------------------------------------------- #


def test_planner_reads_by_explicit_artifact_ids_not_latest():
    store = ArtifactStore(session_id="sess_planner")
    old = store.put("candidates", {"gen": "old"}, agent="attraction")
    new = store.put("candidates", {"gen": "new"}, agent="attraction")

    # 显式指定旧 artifact：即使存在更新的 latest 也只读指定的
    result = read_artifacts_for_planner(store, [old], expected_agent="attraction")
    assert result.ok
    assert result.records[0]["payload"] == {"gen": "old"}

    result = read_artifacts_for_planner(store, [new, old], expected_agent="attraction")
    assert result.ok and len(result.records) == 2


def test_planner_read_reports_missing_and_mismatched():
    store = ArtifactStore(session_id="sess_planner2")
    attraction = store.put("candidates", {}, agent="attraction")
    hotel = store.put("hotels", {}, agent="hotel")

    result = read_artifacts_for_planner(
        store, [attraction, hotel, "ghost_id"], expected_agent="attraction"
    )
    assert not result.ok
    assert result.missing_ids == ["ghost_id"]
    assert result.mismatched == [{"artifact_id": hotel, "agent": "hotel"}]
    assert len(result.records) == 1


# --- AgentTraceLog ------------------------------------------------------------- #


def test_agent_trace_concurrent_append_loses_nothing():
    log = AgentTraceLog(request_id="req_trace")

    def make_worker(worker_idx: int):
        def worker():
            for i in range(PER_THREAD):
                log.append(
                    "subagent_done",
                    agent="attraction",
                    task_id=f"task_{worker_idx}_{i}",
                    status="completed",
                )

        return worker

    threads = [
        threading.Thread(target=make_worker(worker_idx)) for worker_idx in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "线程未能在超时内结束（疑似死锁）"
    snapshot = log.snapshot()
    assert len(snapshot) == THREADS * PER_THREAD
    assert len(log) == THREADS * PER_THREAD
    assert {entry["task_id"] for entry in snapshot} == {
        f"task_{w}_{i}" for w in range(THREADS) for i in range(PER_THREAD)
    }
    assert all(entry["request_id"] == "req_trace" for entry in snapshot)


def test_agent_trace_flush_writes_single_artifact_with_metadata():
    store = ArtifactStore(session_id="sess_trace")
    log = AgentTraceLog(request_id="req_flush")
    log.append("dispatch", agent="orchestrator", task_id="t1")
    log.append("subagent_done", agent="transport", task_id="t1")

    artifact_id = log.flush_to_store(store)
    record = store.get_record(artifact_id)
    assert record["kind"] == AgentTraceLog.KIND
    assert record["request_id"] == "req_flush"
    assert len(record["payload"]["items"]) == 2


def test_agent_trace_concurrent_requests_never_overwrite_each_other():
    """并发回合各自 flush，每条 trace artifact 带 request_id，无 latest→改→put 覆盖。"""
    store = ArtifactStore(session_id="sess_trace2")
    log_a = AgentTraceLog(request_id="req_a")
    log_b = AgentTraceLog(request_id="req_b")
    log_a.append("dispatch")
    log_b.append("dispatch")
    log_b.append("subagent_done")
    id_a = log_a.flush_to_store(store)
    id_b = log_b.flush_to_store(store)
    assert store.get_record(id_a)["request_id"] == "req_a"
    assert store.get_record(id_b)["request_id"] == "req_b"
    assert len(store.get_record(id_a)["payload"]["items"]) == 1
    assert len(store.get_record(id_b)["payload"]["items"]) == 2


# --- 锁与死锁 ------------------------------------------------------------------- #


def test_session_lock_is_non_reentrant():
    """事实确认：per-session 锁是不可重入 threading.Lock（死锁前提）。"""
    lock = session_tool_lock("sess_reentrant_probe")
    assert lock.acquire()
    try:
        assert lock.acquire(blocking=False) is False  # 同线程再拿不到 → 嵌套必死锁
    finally:
        lock.release()


def test_should_serialize_tool_excludes_dispatch():
    assert should_serialize_tool("dispatch_subagent") is False
    for name in ("search_poi", "search_hotel", "plan_route"):
        assert should_serialize_tool(name) is False
    for name in ("build_constraints", "plan_and_critique", "unknown_tool"):
        assert should_serialize_tool(name) is True


def test_holds_session_lock_probe():
    session_id = "sess_probe"
    lock = session_tool_lock(session_id)
    assert holds_session_lock(session_id) is False
    lock.acquire()
    try:
        assert holds_session_lock(session_id) is True
    finally:
        lock.release()


def test_nested_serialized_tools_deadlock_without_isolation():
    """复现风险：dispatch 若持锁 + Subagent 业务工具再取同锁 → 死锁。

    用后台线程模拟被卡死的嵌套获取，主线程释放外层锁后其才能完成。
    """
    session_id = "sess_deadlock_demo"
    lock = session_tool_lock(session_id)
    inner_acquired = threading.Event()

    def nested_inner():
        # 模拟 Subagent 内业务工具再次获取同一把锁
        with lock:
            inner_acquired.set()

    lock.acquire()  # 模拟被 serialized 包住的 dispatch 持锁执行
    thread = threading.Thread(target=nested_inner)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive()  # 内层被外层阻塞（死锁态）
    assert not inner_acquired.is_set()
    lock.release()  # 释放外层后内层才能完成 → 证明嵌套不可行
    thread.join(timeout=5)
    assert inner_acquired.is_set()


def test_dispatch_without_lock_allows_nested_business_tools():
    """隔离验证：dispatch 不持锁时，嵌套业务工具（持锁）正常完成。"""
    session_id = "sess_dispatch_ok"
    done = threading.Event()

    def business_tool():
        with session_tool_lock(session_id):
            time.sleep(0.01)

    def dispatch_simulation():
        # 与 runner 契约一致：dispatch 全程不获取 session 锁
        assert not holds_session_lock(session_id)
        business_tool()  # Subagent 业务工具自由取锁
        assert not holds_session_lock(session_id)
        done.set()

    dispatch_simulation()
    assert done.is_set()


def test_lc_tools_serialized_skips_dispatch_tools():
    """Legacy session lock no longer wraps provider/tool I/O."""
    from travel_agent.agent.lc_tools import build_tools
    from travel_agent.agent.session import build_session
    from travel_agent.settings import load_settings

    ctx = build_session(session_id="sess_lctools", persist=False)
    settings = load_settings()
    tools = build_tools(ctx, settings)
    by_name = {tool.name: tool for tool in tools}
    assert "search_poi" in by_name
    assert "dispatch_subagent" not in by_name  # Step 3 才注册，此处确认不越权

    lock = session_tool_lock("sess_lctools")
    lock.acquire()
    completed = threading.Event()

    def call_business_tool():
        by_name["search_poi"].invoke({"city": "杭州"})
        completed.set()

    thread = threading.Thread(target=call_business_tool)
    thread.start()
    thread.join(timeout=0.3)
    assert not thread.is_alive() and completed.is_set()
    lock.release()


def test_mcp_wrapper_skips_dispatch_tools():
    """MCP adapter injects context without holding the legacy lock."""
    from langchain_core.tools import StructuredTool

    from travel_agent.agent.tool_source import _wrap_mcp_tool_with_session

    def dispatch(session_id: str = "default", task_context: dict | None = None):
        return {"isError": False, "summary": "ok"}

    def business(session_id: str = "default", task_context: dict | None = None):
        return {"isError": False, "summary": "ok"}

    dispatch_tool = StructuredTool.from_function(
        func=dispatch, name="dispatch_subagent", description="d"
    )
    business_tool = StructuredTool.from_function(
        func=business, name="search_poi", description="b"
    )
    assert _wrap_mcp_tool_with_session(dispatch_tool, "sess_mcp") is not dispatch_tool
    wrapped = _wrap_mcp_tool_with_session(business_tool, "sess_mcp")
    assert wrapped is not business_tool and wrapped.name == "search_poi"
    lock = session_tool_lock("sess_mcp")
    lock.acquire()
    try:
        assert '"isError": false' in wrapped.invoke({})
    finally:
        lock.release()
