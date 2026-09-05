from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from travel_agent.agent.session import ArtifactStore, RequestControl, build_session
from travel_agent.agent.tool_source import (
    LOGICAL_TOOL_NAMES,
    ToolContractError,
    _wrap_mcp_tool_with_session,
    validate_tool_contract,
)
from travel_agent.orchestration.meter import (
    TurnBudgetExhausted,
    TurnMeter,
    current_turn_meter,
    turn_meter_scope,
    meter_callbacks,
)
from travel_agent.orchestration.multi_agent.engine import MultiAgentEngine
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_INCOMPLETE,
    EngineCapabilities,
)
from travel_agent.orchestration.multi_agent.runner import SubagentRunner
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog


def test_turn_meter_uses_one_contract_for_all_architecture_roles() -> None:
    meter = TurnMeter("req-meter", token_budget=100, input_cost_per_million=1, output_cost_per_million=2)
    with turn_meter_scope(meter):
        for role in (
            "v0_main",
            "orchestrator",
            "worker:attraction",
            "planner",
            "reviewer",
            "worker:transport",
        ):
            call = meter.begin_llm(role, 2)
            meter.finish_llm(call, input_tokens=2, output_tokens=1, total_tokens=3)
        meter.record_tool("v0_main")
        meter.record_tool("worker:attraction")
        meter.record_dispatch("attraction")
        meter.record_dispatch("planner")
    meter.finish()
    snapshot = meter.snapshot()
    assert {key: snapshot["totals"][key] for key in (
        "input_tokens", "output_tokens", "total_tokens", "llm_calls", "tool_calls", "dispatch_count"
    )} == {
        "input_tokens": 12,
        "output_tokens": 6,
        "total_tokens": 18,
        "llm_calls": 6,
        "tool_calls": 2,
        "dispatch_count": 2,
    }
    assert snapshot["roles"]["reviewer"]["llm_calls"] == 1
    assert snapshot["roles"]["worker:transport"]["llm_calls"] == 1


def test_turn_meter_blocks_before_call_and_charges_missing_usage_conservatively() -> None:
    meter = TurnMeter("req-budget", token_budget=4)
    first = meter.begin_llm("preflight", 3)
    meter.finish_llm(first, input_tokens=None, output_tokens=None, total_tokens=None)
    assert meter.total_tokens == 3
    with pytest.raises(TurnBudgetExhausted):
        meter.begin_llm("orchestrator", 2)
    assert meter.snapshot()["roles"].get("orchestrator") is None

    tool_meter = TurnMeter("req-tools", tool_call_budget=1)
    tool_meter.begin_tool("v0_main")
    with pytest.raises(TurnBudgetExhausted):
        tool_meter.begin_tool("worker:hotel")
    assert tool_meter.snapshot()["totals"]["tool_calls"] == 1


def test_turn_meter_closes_outer_timed_out_call_once() -> None:
    meter = TurnMeter("req-outer-timeout")
    call = meter.begin_llm("orchestrator", 7)

    assert meter.fail_pending_llm("orchestrator", "router timeout") == 1
    # A late provider callback must not mutate the already-final snapshot.
    meter.finish_llm(
        call,
        input_tokens=100,
        output_tokens=100,
        total_tokens=200,
    )
    snapshot = meter.snapshot()["roles"]["orchestrator"]
    assert snapshot["llm_calls"] == 1
    assert snapshot["total_tokens"] == 7
    assert snapshot["failures"] == 1


def test_langchain_callbacks_count_roles_and_block_provider_before_generate() -> None:
    class UsageModel(BaseChatModel):
        generated: int = 0

        @property
        def _llm_type(self) -> str:
            return "usage-test"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.generated += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="ok",
                            usage_metadata={
                                "input_tokens": 4,
                                "output_tokens": 2,
                                "total_tokens": 6,
                            },
                        )
                    )
                ]
            )

    model = UsageModel()
    meter = TurnMeter("req-callback")
    with turn_meter_scope(meter):
        for role in ("v0_main", "orchestrator", "worker:hotel", "planner", "reviewer"):
            model.invoke([HumanMessage(content="x")], config={"callbacks": meter_callbacks(role)})
    assert model.generated == 5
    assert meter.snapshot()["totals"]["total_tokens"] == 30
    assert all(
        meter.snapshot()["roles"][role]["llm_calls"] == 1
        for role in ("v0_main", "orchestrator", "worker:hotel", "planner", "reviewer")
    )

    blocked = UsageModel()
    hard = TurnMeter("req-hard", token_budget=1)
    with turn_meter_scope(hard), pytest.raises(TurnBudgetExhausted):
        blocked.invoke(
            [HumanMessage(content="this input is definitely larger than one token")],
            config={"callbacks": meter_callbacks("v0_main")},
        )
    assert blocked.generated == 0


def test_artifact_store_never_exposes_or_accepts_mutable_aliases() -> None:
    store = ArtifactStore("safe")
    source = {"nested": {"items": [1]}}
    artifact_id = store.put("unit", source)
    source["nested"]["items"].append(2)
    assert store.get(artifact_id) == {"nested": {"items": [1]}}

    payload = store.get(artifact_id)
    payload["nested"]["items"].append(3)
    record = store.get_record(artifact_id)
    record["payload"]["nested"]["items"].append(4)
    payloads = store.get_payloads([artifact_id])
    payloads[0]["nested"]["items"].append(5)
    assert store.get(artifact_id) == {"nested": {"items": [1]}}


def test_trace_snapshot_is_deep_and_records_real_event_time_and_failure() -> None:
    trace = AgentTraceLog("req-trace")
    detail = {"warnings": ["one"]}
    before = time.time()
    trace.append(
        "subagent",
        agent="hotel",
        task_id="hotel-1",
        attempt=1,
        status="failed",
        error="provider timeout",
        duration_ms=12.5,
        detail=detail,
    )
    after = time.time()
    detail["warnings"].append("mutated")
    item = trace.snapshot()[0]
    assert before <= item["created_at"] <= after
    assert item["request_id"] == "req-trace"
    assert item["error"] == "provider timeout"
    assert item["duration_ms"] == 12.5
    assert item["detail"]["warnings"] == ["one"]
    item["detail"]["warnings"].append("external")
    assert trace.snapshot()[0]["detail"]["warnings"] == ["one"]


def test_engine_rejects_runner_reuse_across_sessions() -> None:
    first = build_session(session_id="first", persist=False)
    second = build_session(session_id="second", persist=False)
    engine = MultiAgentEngine(runner=SubagentRunner(first, lambda *_: {}))
    with pytest.raises(RuntimeError, match="session-scoped"):
        engine._resolve_runner(second, SimpleNamespace())


def test_two_session_artifact_trace_and_profile_state_do_not_cross() -> None:
    contexts = [build_session(session_id=name, persist=False) for name in ("left", "right")]

    def write(ctx, destination):
        ctx.profile.destination = destination
        trace = AgentTraceLog(f"req-{destination}")
        artifact_id = ctx.store.put("unit", {"destination": destination}, request_id=trace.request_id)
        trace.append("subagent", agent="attraction", task_id="t", status="completed", attempt=1)
        trace.flush_to_store(ctx.store)
        return artifact_id

    threads = [
        threading.Thread(target=write, args=(contexts[0], "杭州")),
        threading.Thread(target=write, args=(contexts[1], "上海")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert contexts[0].profile.destination == "杭州"
    assert contexts[1].profile.destination == "上海"
    assert contexts[0].store.latest("unit")["destination"] == "杭州"
    assert contexts[1].store.latest("unit")["destination"] == "上海"
    assert contexts[0].store.latest("agent_trace")["request_id"] == "req-杭州"
    assert contexts[1].store.latest("agent_trace")["request_id"] == "req-上海"


def test_mcp_wrapper_hides_transport_fields_mirrors_artifact_and_counts_tool() -> None:
    def remote(
        city: str,
        session_id: str = "default",
        task_context: dict | None = None,
    ) -> dict:
        assert session_id == "mcp-client"
        assert (task_context or {}).get("agent") == "attraction"
        return {
            "isError": False,
            "summary": city,
            "artifact_id": "candidates_remote",
            "_artifact_record": {
                "artifact_id": "candidates_remote",
                "kind": "candidates",
                "session_id": session_id,
                "created_at": time.time(),
                "payload": {"city": city},
                "request_id": "req-mcp",
                "task_id": "task-mcp",
                "agent": "attraction",
            },
        }

    from travel_agent.agent.session import reset_task_meta, set_current_task_meta

    ctx = build_session(session_id="mcp-client", persist=False)
    raw = StructuredTool.from_function(remote, name="search_poi", description="remote")
    wrapped = _wrap_mcp_tool_with_session(raw, ctx)
    assert set(wrapped.args_schema.model_fields) == {"city"}
    meter = TurnMeter("req-mcp")
    token = set_current_task_meta(
        {"request_id": "req-mcp", "task_id": "task-mcp", "agent": "attraction"}
    )
    try:
        with turn_meter_scope(meter):
            result = json.loads(wrapped.invoke({"city": "杭州"}))
    finally:
        reset_task_meta(token)
    assert "_artifact_record" not in result
    assert ctx.store.get("candidates_remote") == {"city": "杭州"}
    assert meter.snapshot()["roles"]["attraction"]["tool_calls"] == 1


def test_mcp_contract_is_fail_closed() -> None:
    one = StructuredTool.from_function(
        lambda: {"isError": False, "summary": "ok"},
        name="search_poi",
        description="one",
    )
    with pytest.raises(ToolContractError, match="missing="):
        validate_tool_contract([one], source="mcp")
    assert "dispatch_subagent" not in LOGICAL_TOOL_NAMES


def test_loopback_mcp_client_ignores_environment_proxy(monkeypatch) -> None:
    from travel_agent.agent.tool_source import _loopback_httpx_client

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9999")
    client = _loopback_httpx_client()
    try:
        assert client._trust_env is False
        assert client.follow_redirects is True
    finally:
        asyncio_run = __import__("asyncio").run
        asyncio_run(client.aclose())


def test_cancelled_mcp_result_cannot_merge_into_live_snapshot() -> None:
    def remote(session_id: str = "default", task_context: dict | None = None) -> dict:
        control.cancel()
        return {
            "isError": False,
            "summary": "late",
            "artifact_id": "late_artifact",
            "_artifact_record": {
                "artifact_id": "late_artifact",
                "kind": "unit",
                "session_id": session_id,
                "created_at": time.time(),
                "payload": {"late": True},
            },
        }

    control = RequestControl("cancelled")
    ctx = build_session(session_id="cancelled", persist=False).clone_isolated(control)
    wrapped = _wrap_mcp_tool_with_session(
        StructuredTool.from_function(remote, name="check_weather", description="remote"), ctx
    )
    with pytest.raises(Exception, match="request cancelled"):
        wrapped.invoke({})
    assert ctx.store.get("late_artifact") is None


def test_local_and_mcp_publish_identical_logical_names_and_public_fields(offline_settings) -> None:
    import asyncio

    from travel_agent.agent.tool_source import resolve_tools_async
    from travel_agent.mcp_server import mcp

    local, source = asyncio.run(
        resolve_tools_async(build_session(session_id="contract", persist=False), offline_settings)
    )
    remote = asyncio.run(mcp.list_tools())
    assert source == "local"
    assert {tool.name for tool in local} == LOGICAL_TOOL_NAMES
    assert {tool.name for tool in remote} == LOGICAL_TOOL_NAMES
    local_fields = {
        tool.name: set(tool.args_schema.model_fields)
        for tool in local
    }
    remote_fields = {
        tool.name: set((tool.parameters.get("properties") or {}))
        - {"session_id", "task_context"}
        for tool in remote
    }
    assert local_fields == remote_fields
    local_schemas = {tool.name: tool.args_schema.model_json_schema() for tool in local}
    for tool in remote:
        remote_schema = tool.parameters
        remote_properties = dict(remote_schema.get("properties") or {})
        remote_properties.pop("session_id", None)
        remote_properties.pop("task_context", None)
        local_properties = local_schemas[tool.name].get("properties") or {}
        assert {
            name: {key: value for key, value in schema.items() if key not in {"title", "description"}}
            for name, schema in local_properties.items()
        } == {
            name: {key: value for key, value in schema.items() if key not in {"title", "description"}}
            for name, schema in remote_properties.items()
        }
        assert set(local_schemas[tool.name].get("required") or []) == (
            set(remote_schema.get("required") or []) - {"session_id", "task_context"}
        )


def test_fastmcp_tool_result_normalizes_to_shared_envelope() -> None:
    import asyncio

    from travel_agent.agent.tool_source import _normalize_envelope
    from travel_agent.mcp_server import mcp

    result = asyncio.run(
        mcp.call_tool(
            "request_travel_info",
            {"session_id": "direct-contract", "missing_fields": ["destination"]},
        )
    )
    payload = json.loads(_normalize_envelope(result, "request_travel_info"))
    assert payload["isError"] is False
    assert payload["summary"]
    assert "_profile_snapshot" not in payload


def test_local_tools_do_not_hold_session_lock_across_io(monkeypatch, offline_settings) -> None:
    from travel_agent.agent import toolkit
    from travel_agent.agent.lc_tools import build_tools

    def slow_search(_ctx, city=None, interests=None, category=None, max_results=30):
        time.sleep(0.15)
        return {"isError": False, "summary": city or "ok"}

    monkeypatch.setattr(toolkit, "search_poi", slow_search)
    ctx = build_session(session_id="parallel", persist=False)
    tool = next(item for item in build_tools(ctx, offline_settings) if item.name == "search_poi")
    started = time.perf_counter()
    threads = [threading.Thread(target=tool.invoke, args=({"city": city},)) for city in ("杭州", "上海")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert time.perf_counter() - started < 0.27


def test_repair_with_failed_hard_critic_is_incomplete() -> None:
    from travel_agent.agent.turn_analysis import TaskType
    ctx = build_session(session_id="repair-gate", persist=False)

    class Executor:
        def __init__(self):
            self.planner_calls = 0

        def __call__(self, _definition, task, isolated):
            if task.agent != "planner":
                isolated.store.put("candidates", {"pois": []})
                return {"summary": "domain"}
            self.planner_calls += 1
            passed = self.planner_calls == 1
            from travel_agent.artifact_policy import constraint_version
            from travel_agent.plan_invariants import validate_plan_artifact

            version = constraint_version(isolated.profile)
            payload = {
                "artifact_status": "candidate",
                "state_version": {
                    "constraint_revision": version["revision"],
                    "constraint_hash": version["constraint_hash"],
                    "constraint_snapshot": version["constraint_snapshot"],
                },
                "itinerary": {"city": "杭州", "summary": "plan", "days": []},
                "source_artifact_ids": list(task.inputs.get("artifact_ids") or []),
                "critic": {
                    "passed": passed,
                    "issues": [] if passed else [
                        {"severity": "critical", "message": "bad"}
                    ],
                },
            }
            payload["validation_result"] = validate_plan_artifact(
                payload, isolated.profile
            )
            isolated.store.put("itinerary", payload)
            return {"summary": "planner"}

    review = {
        "verdict": "rework",
        "issues": [
            {
                "issue_type": "pace",
                "severity": "recoverable",
                "description": "too dense",
                "evidence": ["itinerary.day2"],
                "repair_target": "planner",
                "repair_instruction": "relax",
            }
        ],
    }
    capabilities = EngineCapabilities(
        mode="orchestrated", dispatch="fixed", reviewer_enabled=True, max_rework=1
    )
    engine = MultiAgentEngine(
        capabilities,
        runner=SubagentRunner(ctx, Executor()),
        review_callable=lambda _context: review,
    )
    outcome = engine.run_turn(
        ctx,
        SimpleNamespace(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-repair-gate",
    )
    assert outcome.rework_used == 1
    assert outcome.status == STATUS_COMPLETED_WITH_WARNINGS
    assert outcome.plan_artifact_id == ctx.store.latest_current_id("itinerary")
