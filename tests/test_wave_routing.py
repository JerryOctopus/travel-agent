from __future__ import annotations

import threading
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent.deadlines import DeadlineConfig, TurnDeadline
from travel_agent.orchestration.multi_agent.engine import (
    FULL_CONFIG,
    V2_CONFIG,
    MultiAgentEngine,
)
from travel_agent.orchestration.multi_agent.orchestrator_agent import (
    ROUTER_MAX_OUTPUT_TOKENS,
    RoutingDecision,
    RoutingTask,
    route_wave,
    routing_policy_hash,
)
from travel_agent.orchestration.multi_agent.runner import SubagentRunner
from travel_agent.orchestration.multi_agent.schemas import STATUS_COMPLETED, STATUS_INCOMPLETE
from travel_agent.orchestration.meter import TurnMeter, turn_meter_scope
from travel_agent.settings import Settings


def test_preplanner_timeout_preserves_planner_reserve_and_guard() -> None:
    deadline = TurnDeadline(
        started_at=100.0,
        config=DeadlineConfig(
            turn_hard_cap=240,
            base_timeout=120,
            planner_reserve=30,
            reviewer_reserve=45,
            admission_guard=5,
        ),
    )

    assert deadline.remaining_base(now=180.0) == 40
    assert deadline.usable_preplanner_time(now=180.0) == 5
    assert deadline.effective_preplanner_timeout(20, now=180.0) == 5
    assert deadline.effective_preplanner_timeout(20, now=185.0) == 0
    assert deadline.admits_planner(now=185.0)
    assert not deadline.admits_recovery_wave(now=180.0)
    assert deadline.reviewer_timeout(now=330.0) == 5


def test_turn_deadline_includes_outer_preflight_time() -> None:
    meter = TurnMeter(request_id="req-deadline-preflight")

    with turn_meter_scope(meter):
        deadline = TurnDeadline.start(Settings())

    assert deadline.started_at == meter.started_monotonic


def test_v3_candidate_budget_preserves_planner_reviewer_and_finalization() -> None:
    deadline = TurnDeadline(
        started_at=0.0,
        config=DeadlineConfig(),
    )

    # Base closes at 210s because the 300s turn cap preserves the 85s
    # Reviewer reserve plus the independent 5s finalization guard.
    assert deadline.base_deadline == 210
    assert deadline.admits_router(now=40)
    assert deadline.router_timeout_preserving_worker(now=40) == 80
    assert deadline.admits_planner(now=180)
    assert deadline.planner_timeout(60, now=145) == 60
    assert deadline.reviewer_timeout(now=210) == 85
    assert deadline.reviewer_timeout(now=295) == 0


def test_recovery_wave_uses_minimum_useful_windows_not_full_transport_caps() -> None:
    deadline = TurnDeadline(started_at=0.0, config=DeadlineConfig())

    # At t=120, usable pre-Planner time is 60s: enough for the calibrated
    # Router-useful 35s + recovery-worker-useful 25s admission threshold.
    assert deadline.usable_preplanner_time(now=120) == 60
    assert deadline.admits_recovery_wave(now=120)
    assert deadline.admits_router(now=120)
    assert deadline.router_timeout_preserving_worker(now=120) == 35
    assert not deadline.admits_recovery_wave(now=121)
    assert not deadline.admits_router(now=121)


def test_router_effective_timeout_cannot_consume_recovery_worker_window() -> None:
    deadline = TurnDeadline(started_at=0.0, config=DeadlineConfig())

    # The calibrated Router transport cap is 80s, but at this admission point
    # only 65s are usable. Keep 25s for the worker, so Router receives 40s.
    assert deadline.usable_preplanner_time(now=115) == 65
    assert deadline.router_timeout_preserving_worker(now=115) == 40


class _JsonRouterModel(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "json-router-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=(
                            '{"ready":false,"tasks":['
                            '{"agent":"attraction","instruction":"检索景点",'
                            '"objective":"景点候选","depends_on":[]},'
                            '{"agent":"planner","instruction":"写计划",'
                            '"objective":"计划","depends_on":[]}]}'
                        )
                    )
                )
            ]
        )


def test_route_wave_is_one_structured_call_and_rejects_planner() -> None:
    model = _JsonRouterModel()
    decision = route_wave(
        None,
        Settings(),
        "杭州两日游",
        "req-router-json",
        wave=1,
        model=model,
        task_type=TaskType.FULL_TRIP_PLAN,
        timeout_seconds=1,
        max_tasks=4,
    )

    assert model.calls == 1
    assert [task.agent for task in decision.tasks] == ["attraction"]


def test_route_wave_binds_bounded_json_output() -> None:
    captured = {}

    class Model:
        def bind(self, **kwargs):
            captured.update(kwargs)
            return self

        def invoke(self, messages, config=None):
            return AIMessage(content='{"ready":true,"tasks":[]}')

    decision = route_wave(
        None,
        Settings(),
        "杭州两日游",
        "req-router-bounded-json",
        wave=1,
        model=Model(),
        task_type=TaskType.FULL_TRIP_PLAN,
        timeout_seconds=1,
        max_tasks=4,
    )

    assert decision.ready is True
    assert captured == {
        "max_tokens": ROUTER_MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }


def test_routing_policy_hash_records_output_contract(monkeypatch) -> None:
    original = routing_policy_hash(Settings())
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.ROUTER_MAX_OUTPUT_TOKENS",
        ROUTER_MAX_OUTPUT_TOKENS + 1,
    )

    assert routing_policy_hash(Settings()) != original


class _SlowRouterModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "slow-router-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(0.05)
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content='{"ready":true,"tasks":[]}'))
            ]
        )


def test_router_outer_timeout_is_finalized_in_turn_meter() -> None:
    meter = TurnMeter(request_id="req-router-timeout-meter")
    with turn_meter_scope(meter):
        decision = route_wave(
            None,
            Settings(),
            "杭州两日游",
            meter.request_id,
            wave=1,
            model=_SlowRouterModel(),
            task_type=TaskType.FULL_TRIP_PLAN,
            timeout_seconds=0.01,
            max_tasks=4,
        )

    assert "router timeout" in str(decision.error)
    usage = meter.snapshot()["roles"]["orchestrator"]
    assert usage["llm_calls"] == 1
    assert usage["failures"] == 1
    assert usage["latency_ms"] >= 5


class _WaveExecutor:
    def __init__(self) -> None:
        self.tasks = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, definition, task, ctx):
        self.tasks.append(task)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if task.agent != "planner":
                time.sleep(0.02)
            if task.agent == "planner":
                ctx.store.put(
                    "itinerary",
                    {
                        "itinerary": {"city": "杭州", "summary": "两日游", "days": []},
                        "critic": {"passed": True, "issues": []},
                        "source_artifact_ids": list(task.inputs.get("artifact_ids") or []),
                    },
                )
            else:
                kind = {
                    "attraction": "candidates",
                    "transport": "routes",
                    "hotel": "hotels",
                    "restaurant": "restaurants",
                }[task.agent]
                ctx.store.put(kind, {"city": "杭州", "items": []})
                if task.agent == "attraction":
                    ctx.store.put(
                        "restaurants",
                        {"items": [
                            {"poi_id": "meal-1", "name": "餐厅一"},
                            {"poi_id": "meal-2", "name": "餐厅二"},
                        ]},
                    )
            return {"status": STATUS_COMPLETED, "summary": f"{task.agent} done"}
        finally:
            with self.lock:
                self.active -= 1


def _engine(capabilities):
    ctx = build_session(persist=False)
    executor = _WaveExecutor()
    return MultiAgentEngine(
        capabilities,
        runner=SubagentRunner(ctx, executor),
        review_callable=lambda _ctx: {"verdict": "pass", "issues": []},
    ), ctx, executor


def test_wave1_ready_goes_directly_to_planner(monkeypatch) -> None:
    calls = []

    def route(*args, **kwargs):
        calls.append(kwargs["wave"])
        return RoutingDecision(
            tasks=(
                RoutingTask("attraction", "检索景点", "景点候选"),
                RoutingTask("transport", "验证路线", "路线可行性"),
            )
        )

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave", route
    )
    engine, ctx, executor = _engine(V2_CONFIG)
    outcome = engine.run_turn(
        ctx,
        Settings(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-wave1",
    )

    assert outcome.status == STATUS_COMPLETED
    assert calls == [1]
    assert executor.max_active == 2
    assert [task.agent for task in executor.tasks].count("planner") == 1


def test_wave2_only_runs_for_missing_hard_evidence(monkeypatch) -> None:
    calls = []

    def route(*args, **kwargs):
        wave = kwargs["wave"]
        calls.append(wave)
        if wave == 1:
            return RoutingDecision(
                tasks=(RoutingTask("attraction", "检索景点", "景点候选"),)
            )
        return RoutingDecision(
            tasks=(RoutingTask("transport", "补齐路线", "路线可行性"),)
        )

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave", route
    )
    engine, ctx, _executor = _engine(V2_CONFIG)
    outcome = engine.run_turn(
        ctx,
        Settings(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-wave2",
    )

    assert outcome.status == STATUS_COMPLETED
    assert calls == [1, 2]


def test_missing_hard_evidence_does_not_force_planner(monkeypatch) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        lambda *args, **kwargs: RoutingDecision(ready=True),
    )
    engine, ctx, executor = _engine(V2_CONFIG)
    outcome = engine.run_turn(
        ctx,
        Settings(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-no-plan",
    )

    assert outcome.status == STATUS_INCOMPLETE
    assert not any(task.agent == "planner" for task in executor.tasks)
    assert ctx.store.latest_id("itinerary") is None


def test_v2_v3_execute_independent_bases_with_same_policy(monkeypatch) -> None:
    calls = []

    def route(*args, **kwargs):
        calls.append(args[3])  # request_id
        return RoutingDecision(
            tasks=(
                RoutingTask("attraction", "检索景点", "景点候选"),
                RoutingTask("transport", "验证路线", "路线可行性"),
            )
        )

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave", route
    )
    v2, ctx2, _ = _engine(V2_CONFIG)
    v3, ctx3, _ = _engine(FULL_CONFIG)
    out2 = v2.run_turn(
        ctx2,
        Settings(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-v2-independent",
    )
    out3 = v3.run_turn(
        ctx3,
        Settings(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req-v3-independent",
    )

    assert calls == ["req-v2-independent", "req-v3-independent"]
    assert out2.routing_policy_hash == out3.routing_policy_hash
    assert out2.plan_artifact_id != out3.plan_artifact_id
