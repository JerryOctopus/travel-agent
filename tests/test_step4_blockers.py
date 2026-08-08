"""第一批阻塞修复的跨模块集成测试。"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent.dispatch_rules import build_fixed_tasks
from travel_agent.orchestration.multi_agent.engine import V1_CONFIG, V2_CONFIG, MultiAgentEngine
from travel_agent.orchestration.multi_agent.orchestrator import build_dispatch_tool
from travel_agent.orchestration.multi_agent.review import ReviewContext, run_semantic_review
from travel_agent.orchestration.multi_agent.runner import SubagentRunner
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    SubagentTask,
)


PLAN_PAYLOAD = {
    "itinerary": {"city": "杭州", "summary": "杭州两日游", "days": []},
    "critic": {"passed": True, "issues": []},
}


def test_runner_claims_only_artifacts_from_its_own_request_and_task():
    ctx = build_session(session_id="sess_exact_owner", persist=False)
    barrier = threading.Barrier(2)

    def executor(_definition, task, ctx_):
        barrier.wait(timeout=5)
        ctx_.store.put("candidates", {"owner": task.task_id})
        barrier.wait(timeout=5)
        return {"summary": task.task_id}

    runner = SubagentRunner(ctx, executor)
    tasks = [
        SubagentTask("req_a", "attraction-a", "attraction", "a"),
        SubagentTask("req_b", "attraction-b", "attraction", "b"),
    ]
    results = [None, None]

    def run(index: int) -> None:
        results[index] = runner.run_subagent(tasks[index])

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    for task, result in zip(tasks, results):
        assert result is not None
        assert len(result.evidence) == 1
        record = ctx.store.get_record(result.evidence[0]["artifact_id"])
        assert record["request_id"] == task.request_id
        assert record["task_id"] == task.task_id
        assert record["payload"]["owner"] == task.task_id


def test_dynamic_depends_on_is_resolved_to_artifact_ids():
    ctx = build_session(session_id="sess_dynamic_dep", persist=False)
    seen: list[SubagentTask] = []

    def executor(_definition, task, ctx_):
        seen.append(task)
        if task.agent == "attraction":
            ctx_.store.put("candidates", {"pois": []})
        else:
            ctx_.store.put("itinerary", dict(PLAN_PAYLOAD))
        return {"summary": "ok"}

    dispatch = build_dispatch_tool(SubagentRunner(ctx, executor), request_id="req_dep")
    attraction = json.loads(dispatch("attraction", "查景点"))
    planner = json.loads(
        dispatch("planner", "做计划", depends_on=[attraction["task_id"]])
    )

    assert planner["status"] == STATUS_COMPLETED
    assert seen[-1].agent == "planner"
    assert seen[-1].depends_on == [attraction["task_id"]]
    assert seen[-1].inputs["artifact_ids"] == [attraction["evidence"][0]["artifact_id"]]


def test_dynamic_unknown_dependency_fails_without_running_subagent():
    ctx = build_session(session_id="sess_dynamic_missing_dep", persist=False)
    calls: list[SubagentTask] = []
    dispatch = build_dispatch_tool(
        SubagentRunner(ctx, lambda _definition, task, _ctx: calls.append(task) or {}),
        request_id="req_missing",
    )

    result = json.loads(dispatch("planner", "做计划", depends_on=["missing-task"]))

    assert result["status"] == "failed"
    assert "unresolved dependencies" in result["error"]
    assert calls == []


class _RealToolkitExecutor:
    """执行真实领域工具与 Planner 工具，不经过 LLM。"""

    def __init__(self) -> None:
        self.tasks: list[SubagentTask] = []
        self.foreign_artifact_id: str | None = None

    def __call__(self, _definition, task, ctx):
        self.tasks.append(task)
        if task.agent == "attraction":
            toolkit.search_poi(ctx)
            toolkit.check_weather(ctx)
        elif task.agent == "hotel":
            toolkit.search_hotel(ctx)
        elif task.agent == "restaurant":
            toolkit.search_restaurant(ctx)
            toolkit.estimate_budget(ctx)
        elif task.agent == "transport":
            poi_ids = list(ctx.pois_by_id)
            if len(poi_ids) >= 2:
                toolkit.plan_route(ctx, poi_ids[0], poi_ids[1])
            else:
                ctx.store.put("routes", {"duration_min": 20})
            # 并发/其他请求写入的同 kind 最新结果不得被 Planner 消费。
            self.foreign_artifact_id = ctx.store.put(
                "candidates",
                {"city": "错误城市", "pois": []},
                request_id="req_foreign",
                task_id="attraction-foreign",
                agent="attraction",
            )
        else:
            ranked = toolkit.recommend_candidates(ctx)
            if ranked.get("isError"):
                return {"status": "failed", "summary": ranked["summary"]}
            plan = toolkit.plan_and_critique(ctx)
            if plan.get("isError"):
                return {"status": "failed", "summary": plan["summary"]}
        return {"summary": f"{task.agent} ok"}


def test_v1_real_toolkit_planner_consumes_all_bound_domains_not_latest():
    ctx = build_session(session_id="sess_real_planner", persist=False)
    ctx.profile.destination = "杭州"
    ctx.profile.days = 2
    executor = _RealToolkitExecutor()
    engine = MultiAgentEngine(V1_CONFIG, runner=SubagentRunner(ctx, executor))

    outcome = engine.run_turn(
        ctx,
        SimpleNamespace(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req_real_planner",
    )

    assert outcome.status == STATUS_COMPLETED
    assert outcome.gate_status == "rendered"
    plan = ctx.store.get(outcome.plan_artifact_id)
    assert executor.foreign_artifact_id not in plan["source_artifact_ids"]
    domain_inputs = plan["domain_inputs"]
    for key in ("attractions", "hotels", "restaurants", "transport", "weather", "budgets"):
        assert domain_inputs[key], key
    planned_categories = {
        stop["poi"]["category"]
        for day in plan["itinerary"]["days"]
        for stop in day["stops"]
    }
    assert "food" in planned_categories
    transport_task = next(task for task in executor.tasks if task.agent == "transport")
    compact_inputs = transport_task.inputs["artifact_inputs"]
    assert any(item["kind"] == "candidates" and item["payload"]["pois"] for item in compact_inputs)


def test_dynamic_full_plan_without_planner_is_incomplete(monkeypatch):
    ctx = build_session(session_id="sess_no_planner", persist=False)
    engine = MultiAgentEngine(V2_CONFIG, runner=SubagentRunner(ctx, lambda *_args: {}))
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.run_orchestrator",
        lambda *_args, **_kwargs: {
            "reply": "这里是一份模型手写行程",
            "results": [],
            "clarification": False,
        },
    )

    outcome = engine.run_turn(
        ctx,
        SimpleNamespace(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req_no_planner",
    )

    assert outcome.status == STATUS_INCOMPLETE
    assert outcome.plan_artifact_id is None
    assert outcome.gate_status == "skipped"


def test_renderer_incomplete_overrides_top_level_completed(monkeypatch):
    ctx = build_session(session_id="sess_gate_override", persist=False)
    task_id = "planner-gate"
    plan_id = ctx.store.put(
        "itinerary",
        {
            "itinerary": {"city": "杭州", "summary": "不完整计划", "days": []},
            "critic": {"passed": False, "issues": [{"severity": "critical"}]},
        },
        request_id="req_gate_override",
        task_id=task_id,
        agent="planner",
    )
    engine = MultiAgentEngine(V2_CONFIG, runner=SubagentRunner(ctx, lambda *_args: {}))
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.run_orchestrator",
        lambda *_args, **_kwargs: {
            "reply": "done",
            "results": [
                {
                    "request_id": "req_gate_override",
                    "task_id": task_id,
                    "agent": "planner",
                    "status": STATUS_COMPLETED,
                    "evidence": [{"artifact_id": plan_id}],
                }
            ],
            "clarification": False,
        },
    )

    outcome = engine.run_turn(
        ctx,
        SimpleNamespace(),
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req_gate_override",
    )

    assert outcome.gate_status == "rendered_incomplete"
    assert outcome.status == STATUS_INCOMPLETE


def test_reviewer_invalid_schema_fails_closed():
    for raw in (
        {},
        {"verdict": "pass", "issues": "none"},
        {
            "verdict": "rework",
            "issues": [
                {
                    "issue_type": "pace",
                    "severity": "typo",
                    "description": "too busy",
                }
            ],
        },
    ):
        result = run_semantic_review(
            ReviewContext(request_id="req_review", plan={}),
            review_callable=lambda _ctx, value=raw: value,
        )
        assert result.verdict == "failed"
        assert result.error and "invalid reviewer output" in result.error


def test_v1_poi_transport_depends_on_upstream_poi_tasks():
    tasks = build_fixed_tasks("req_poi", TaskType.POI_ADVICE, task_brief="住哪方便")
    assert isinstance(tasks, list)
    attraction, hotel, transport = tasks
    assert [task.agent for task in tasks] == ["attraction", "hotel", "transport"]
    assert transport.depends_on == [attraction.task_id, hotel.task_id]


def test_v1_revision_binds_existing_itinerary_artifact():
    ctx = build_session(session_id="sess_revision", persist=False)
    old_plan_id = ctx.store.put("itinerary", dict(PLAN_PAYLOAD), agent="planner")
    seen: list[SubagentTask] = []

    def executor(_definition, task, ctx_):
        seen.append(task)
        ctx_.store.put("itinerary", dict(PLAN_PAYLOAD))
        return {"summary": "revised"}

    engine = MultiAgentEngine(V1_CONFIG, runner=SubagentRunner(ctx, executor))
    outcome = engine.run_turn(
        ctx,
        SimpleNamespace(),
        "修改第二天",
        task_type=TaskType.ITINERARY_REVISION,
        request_id="req_revision",
    )

    assert outcome.status == STATUS_COMPLETED
    assert seen[0].agent == "planner"
    assert seen[0].inputs["artifact_ids"] == [old_plan_id]


def test_v1_revision_without_existing_itinerary_asks_for_one():
    ctx = build_session(session_id="sess_revision_missing", persist=False)
    engine = MultiAgentEngine(V1_CONFIG, runner=SubagentRunner(ctx, lambda *_args: {}))
    outcome = engine.run_turn(
        ctx,
        SimpleNamespace(),
        "修改第二天",
        task_type=TaskType.ITINERARY_REVISION,
    )
    assert outcome.status == STATUS_CLARIFICATION_REQUIRED
