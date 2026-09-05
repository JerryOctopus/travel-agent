from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType
from travel_agent.agent.turn_lifecycle import prepare_turn
from travel_agent.orchestration.meter import TurnMeter, turn_meter_scope
from travel_agent.orchestration.multi_agent.engine import V0_CONFIG
from travel_agent.orchestration.multi_agent.orchestrator import DispatchLedger, build_dispatch_tool
from travel_agent.orchestration.multi_agent.registry import SUBAGENT_REGISTRY
from travel_agent.orchestration.multi_agent.runner import SubagentRunner
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_COMPLETED,
    SubagentTask,
)


def _settings(*, llm_enabled: bool = False, timeout: float = 0.05):
    return SimpleNamespace(
        llm=SimpleNamespace(enabled=llm_enabled),
        agent=SimpleNamespace(request_timeout_seconds=timeout),
    )


def _plan(city: str = "杭州") -> dict:
    return {
        "itinerary": {"city": city, "days": [{"day_index": 1, "stops": []}, {"day_index": 2, "stops": []}]},
        "critic": {"passed": True, "issues": []},
        "source_artifact_ids": [],
        "domain_inputs": {},
    }


def test_revision_preflight_freezes_existing_plan_and_directives() -> None:
    ctx = build_session(session_id="revision-preflight", persist=False)
    source_id = ctx.store.put("hotels", {"hotels": [{"hotel_id": "h1"}]}, agent="hotel")
    payload = _plan()
    payload["source_artifact_ids"] = [source_id]
    plan_id = ctx.store.put("itinerary", payload, agent="planner")

    prepared = prepare_turn(
        "第二天换室内，改轻松一点，酒店别换了",
        ctx,
        _settings(),
        [("user", "帮我做杭州两日行程"), ("assistant", "已生成")],
    )

    # The day-level target is local, but the same request also changes global
    # pace.  It must therefore rebuild the full itinerary while keeping the
    # stale plan only as revision context.
    assert prepared.analysis.task_type == TaskType.FULL_ITINERARY
    assert prepared.existing_plan_artifact_id is None
    assert prepared.revisable_parent_artifact_id == plan_id
    assert prepared.turn_inputs["plan_artifact_id"] is None
    assert prepared.turn_inputs["parent_plan_artifact_id"] == plan_id
    # Legacy ancestors without a current constraint fingerprint are not
    # silently rebound into a revision.
    assert prepared.turn_inputs["artifact_ids"] == [plan_id]
    assert prepared.turn_inputs["revision_directives"] == {
        "indoor_days": [2],
        "preserve_hotel": True,
        "pace": "relaxed",
    }
    assert ctx.profile.destination == "杭州"
    assert ctx.profile.days == 2
    assert ctx.profile.pace == "relaxed"


def test_offline_multiturn_revision_changes_day_and_preserves_hotel() -> None:
    from travel_agent.agent.runtime import run_production_turn
    from travel_agent.settings import LLMSettings, Settings

    ctx = build_session(session_id="revision-integration", persist=False)
    ctx.profile.hotel_area = "西湖附近"
    settings = Settings(llm=LLMSettings(provider="rule"))
    first = run_production_turn(
        "帮我规划杭州两天，住西湖附近",
        ctx=ctx,
        settings=settings,
    )
    second = run_production_turn(
        "第二天换室内，改轻松一点，酒店别换了",
        ctx=ctx,
        history=[("user", "帮我规划杭州两天，住西湖附近"), ("assistant", first.text)],
        settings=settings,
    )

    first_payload = ctx.store.get(first.plan_artifact_id)
    second_payload = ctx.store.get(second.plan_artifact_id)
    day_two = second_payload["itinerary"]["days"][1]
    first_hotels = first_payload["domain_inputs"]["hotels"][-1]["payload"]
    second_hotels = second_payload["domain_inputs"]["hotels"][-1]["payload"]

    assert second.status == STATUS_COMPLETED
    assert day_two["stops"] and all(stop["poi"]["indoor"] for stop in day_two["stops"])
    assert second_payload["revision_directives"]["preserve_hotel"] is True
    assert second_hotels == first_hotels


class _RecordingRunner:
    def __init__(self) -> None:
        self.tasks: list[SubagentTask] = []

    def run_subagent(self, task: SubagentTask):
        from travel_agent.orchestration.multi_agent.schemas import SubagentResult

        self.tasks.append(task)
        return SubagentResult(
            request_id=task.request_id,
            task_id=task.task_id,
            agent=task.agent,
            status=STATUS_COMPLETED,
        )


def test_dispatch_ledger_rejects_duplicates_dependencies_limits_and_post_planner() -> None:
    runner = _RecordingRunner()
    ledger = DispatchLedger(max_total=3, per_agent_limits={"attraction": 1, "planner": 1})
    dispatch = build_dispatch_tool(runner, "req-ledger", ledger=ledger)

    first = json.loads(dispatch("attraction", "搜索 西湖"))
    duplicate = json.loads(dispatch("attraction", "  搜索   西湖 "))
    per_agent = json.loads(dispatch("attraction", "搜索 灵隐寺"))
    missing = json.loads(dispatch("planner", "规划", depends_on=["missing-task"]))
    planner = json.loads(dispatch("planner", "规划", depends_on=[first["task_id"]]))
    after_terminal = json.loads(dispatch("attraction", "搜索 灵隐寺"))

    assert first["status"] == STATUS_COMPLETED
    assert duplicate["status"] == "failed" and "duplicate" in duplicate["error"]
    assert per_agent["status"] == "failed" and "per-agent" in per_agent["error"]
    assert missing["status"] == "failed" and "unresolved" in missing["error"]
    assert planner["status"] == STATUS_COMPLETED
    assert after_terminal["status"] == "failed" and "terminal" in after_terminal["error"]
    assert len(runner.tasks) == 2

    total_runner = _RecordingRunner()
    total_dispatch = build_dispatch_tool(
        total_runner,
        "req-total",
        ledger=DispatchLedger(
            max_total=1,
            per_agent_limits={"attraction": 2, "hotel": 2},
        ),
    )
    assert json.loads(total_dispatch("attraction", "a"))["status"] == STATUS_COMPLETED
    total_rejected = json.loads(total_dispatch("hotel", "b"))
    assert "global dispatch limit" in total_rejected["error"]


def test_default_dispatch_ledger_allows_bounded_second_dispatch_per_domain() -> None:
    ledger = DispatchLedger()

    assert ledger.authorize("transport", "first route task") is None
    assert ledger.authorize("transport", "different route task") is None
    assert ledger.authorize("transport", "third route task") == (
        "per-agent dispatch limit exhausted: transport=2"
    )


def test_rejected_dynamic_dispatch_is_not_counted_as_executed() -> None:
    runner = _RecordingRunner()
    dispatch = build_dispatch_tool(runner, "req-meter")
    meter = TurnMeter("req-meter")

    with turn_meter_scope(meter):
        assert json.loads(dispatch("hotel", "first"))["status"] == STATUS_COMPLETED
        assert json.loads(dispatch("hotel", "second"))["status"] == STATUS_COMPLETED
        rejected = json.loads(dispatch("hotel", "third"))

    assert "per-agent dispatch limit" in rejected["error"]
    assert meter.snapshot()["totals"]["dispatch_count"] == 2


def test_subagent_timeout_discards_late_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = build_session(session_id="subagent-timeout", persist=False)
    original = SUBAGENT_REGISTRY["hotel"]
    monkeypatch.setitem(
        SUBAGENT_REGISTRY,
        "hotel",
        replace(original, timeout_seconds=0.01),
    )

    def slow_executor(_definition, _task, isolated_ctx):
        time.sleep(0.05)
        isolated_ctx.store.put("hotels", {"hotels": [{"hotel_id": "late"}]})
        return {"summary": "late"}

    result = SubagentRunner(ctx, slow_executor).run_subagent(
        SubagentTask("req-timeout", "hotel-timeout", "hotel", "slow")
    )
    time.sleep(0.08)

    assert result.status == STATUS_BUDGET_EXHAUSTED
    assert ctx.store.latest("hotels") is None


def test_max_steps_is_passed_as_hard_graph_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from travel_agent.orchestration.multi_agent.executor import build_subagent_executor
    from travel_agent.orchestration.multi_agent.registry import SubagentDefinition

    observed: dict[str, int] = {}

    class GraphRecursionError(RuntimeError):
        pass

    class FakeAgent:
        def invoke(self, _state, config):
            observed["limit"] = config["recursion_limit"]
            raise GraphRecursionError("limit")

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.create_agent",
        lambda _model, _tools, **_kwargs: FakeAgent(),
    )
    monkeypatch.setattr("travel_agent.agent.lc_tools.build_tools", lambda _ctx, _settings: [])
    settings = SimpleNamespace(llm=SimpleNamespace(enabled=True, model="fake"))
    definition = SubagentDefinition("unit", "", "", (), max_steps=3, max_tool_calls=0)
    raw = build_subagent_executor(settings, model=object())(
        definition,
        SubagentTask("req", "unit-1", "unit", "x"),
        build_session(persist=False),
    )

    assert observed["limit"] == 3
    assert raw["status"] == STATUS_BUDGET_EXHAUSTED


def test_http_timeout_never_commits_late_profile_or_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    from travel_agent import server

    live = build_session(session_id="http-timeout", persist=False)

    def slow_turn(_message, snapshot, *_args):
        time.sleep(0.05)
        snapshot.profile.destination = "不应提交"
        snapshot.store.put("candidates", {"city": "不应提交", "pois": []})
        return SimpleNamespace(text="late")

    monkeypatch.setattr(server, "run_production_turn", slow_turn)
    async def exercise() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await server._run_isolated_request("x", live, [], _settings(timeout=0.01), "u")
        await asyncio.sleep(0.08)

    asyncio.run(exercise())

    assert live.profile.destination is None
    assert live.store.latest("candidates") is None


def test_v0_uses_common_trace_and_renderer_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from travel_agent.agent.runtime import AgentReply
    from travel_agent.orchestration.multi_agent.engine import MultiAgentEngine

    ctx = build_session(session_id="v0-gate", persist=False)

    def fake_react(_message, active_ctx, _history, _settings):
        active_ctx.store.put("itinerary", _plan())
        return AgentReply(text="ok")

    monkeypatch.setattr("travel_agent.agent.runtime._run_react", fake_react)
    outcome = MultiAgentEngine(V0_CONFIG).run_turn(
        ctx,
        SimpleNamespace(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-v0-gate",
    )

    assert outcome.status == STATUS_COMPLETED
    assert outcome.gate_status == "rendered"
    assert outcome.plan_artifact_id
    assert ctx.store.get_record(outcome.plan_artifact_id)["agent"] == "single_agent"
    assert outcome.trace_artifact_id


def test_route_and_clarification_architecture_policy_do_not_require_planner() -> None:
    from travel_agent.harness.cases import HarnessCase
    from travel_agent.harness.production_evaluators import evaluate_production_case
    from travel_agent.harness.result import HarnessCaseResult, HarnessTurnResult

    turn = HarnessTurnResult(
        user_message="机场到酒店怎么走",
        reply_text="建议乘地铁。",
        tool_trace=["plan_route"],
        used_real_agent=True,
        clarification=False,
        profile={},
        artifacts={},
        status="completed",
    )
    result = HarnessCaseResult("route", [turn], {}, {})
    case = HarnessCase("route", [turn.user_message], gold_outcome="full_plan")

    scored = evaluate_production_case(case, result, variant="V3")

    assert scored["actual_outcome"] == "full_plan"
    assert scored["architecture_policy"]["passed"] is True
