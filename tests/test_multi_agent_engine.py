"""multi_agent Step 3 单元测试：真实编排链路的离线验证。

覆盖：

- V1 规则派工全链路（真实 SubagentRunner + Fake executor + Renderer Gate）；
- V2 动态 Orchestrator（Scripted Fake LLM 驱动 create_agent）与
  Engine 对 clarification / 结果的聚合；
- V3 Reviewer 一次调用 + 最多一次修复周期（recoverable 修复 / critical 禁交付 /
  Reviewer 不可用 fail-closed）；
- Renderer Artifact Gate（无 plan 跳过 / 状态拦截 / 产出者校验 / incomplete 渲染）；
- executor 白名单与预算守卫；review JSON 提取与真实 callable 构建。
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
import json
import threading
import time
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from travel_agent.agent.session import build_session, reset_task_meta, set_current_task_meta
from travel_agent.agent import toolkit
from travel_agent.agent.turn_analysis import DeliveryIntent, TaskType
from travel_agent.orchestration.meter import TurnMeter, turn_meter_scope
from travel_agent.providers import LocalToolProvider
from travel_agent.schemas import POI, TravelProfile
from travel_agent.orchestration.multi_agent import (
    SubagentRunner,
    render_plan_outcome,
)
from travel_agent.orchestration.multi_agent.engine import (
    V1_CONFIG,
    V2_CONFIG,
    MultiAgentEngine,
    _all_evidence_ids,
    _plan_required_for_turn,
    _repair_route_pairs_for_plan,
    _repairable_critical_route_targets,
    _required_evidence,
    _select_lightweight_candidates,
)
from travel_agent.orchestration.multi_agent.executor import (
    _BUDGET_SENTINEL,
    _max_output_tokens_middleware,
    _budget_guard,
    _enforce_transport_route_postcondition,
    _enforce_comparison_search_postcondition,
    _enforce_indoor_backup_postcondition,
    _enforce_required_poi_postcondition,
    _enforce_restaurant_search_postcondition,
    _enforce_hotel_search_postcondition,
    _planner_route_estimator,
    _task_has_nonempty_domain_evidence,
    _promote_recovered_worker_status,
    build_subagent_executor,
    ToolCallBudgetExceeded,
)
from travel_agent.orchestration.multi_agent.registry import SUBAGENT_REGISTRY
from travel_agent.orchestration.multi_agent.review import (
    REVIEWER_MAX_OUTPUT_TOKENS,
    REVIEWER_TIMEOUT_SECONDS,
    ReviewContext,
    build_review_callable,
    extract_json_payload,
    resolve_delivery_status,
    run_semantic_review,
)
from travel_agent.orchestration.multi_agent.orchestrator_agent import (
    RoutingDecision,
    RoutingTask,
)
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    EngineCapabilities,
    ReviewIssue,
    ReviewResult,
)
from travel_agent.delivery_contract import (
    CANDIDATE_REJECTED,
    DELIVERABLE_CURRENT,
    PARTIAL_CURRENT_WITH_LIMITATIONS,
    REBUILD_PENDING,
    resolve_delivery_snapshot,
)

V3_FIXED = EngineCapabilities(
    mode="orchestrated", dispatch="fixed", reviewer_enabled=True, max_rework=1
)

PLAN_PAYLOAD: dict[str, Any] = {
    "itinerary": {"city": "杭州", "summary": "杭州两日游", "days": []},
    "critic": {"passed": True, "issues": []},
}

DOMAIN_KINDS = {
    "attraction": "pois",
    "hotel": "hotels",
    "restaurant": "restaurants",
    "transport": "routes",
}


def test_planner_required_is_derived_from_delivery_intent() -> None:
    assert not _plan_required_for_turn(
        "dynamic", TaskType.FULL_ITINERARY, DeliveryIntent.STATE_UPDATE_ONLY
    )
    assert _plan_required_for_turn(
        "dynamic", TaskType.FULL_ITINERARY, DeliveryIntent.REBUILD_NOW
    )


# --- Fake 协作者 ------------------------------------------------------------- #


class ScriptedChatModel(BaseChatModel):
    """按脚本逐条返回 AIMessage 的 Fake LLM（支持 tool_calls，不 bind 工具）。"""

    messages: list
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools, **kwargs):  # create_agent 需要
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.messages[min(self.i, len(self.messages) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeDomainExecutor:
    """Fake Subagent executor：领域 agent 写 artifact，planner 写行程。"""

    def __init__(self) -> None:
        self.tasks: list[Any] = []

    def __call__(self, definition, task, ctx):
        self.tasks.append(task)
        if task.agent == "planner":
            from travel_agent.artifact_policy import constraint_version
            from travel_agent.plan_invariants import validate_plan_artifact

            version = constraint_version(ctx.profile)
            payload = {
                **PLAN_PAYLOAD,
                "artifact_status": "candidate",
                "state_version": {
                    "constraint_revision": version["revision"],
                    "constraint_hash": version["constraint_hash"],
                    "constraint_snapshot": version["constraint_snapshot"],
                },
            }
            payload["validation_result"] = validate_plan_artifact(payload, ctx.profile)
            ctx.store.put("itinerary", payload)
            return {"summary": "行程已生成并通过自检。"}
        kind = DOMAIN_KINDS[task.agent]
        ctx.store.put(kind, {"city": "杭州", "items": [], "restaurants": []})
        return {"summary": f"{task.agent} 调研完成。"}


def make_engine(capabilities, executor=None, **kwargs) -> tuple[MultiAgentEngine, Any, Any]:
    ctx = build_session(session_id=f"sess_{id(capabilities)}", persist=False)
    runner = SubagentRunner(ctx, executor or FakeDomainExecutor())
    return MultiAgentEngine(capabilities, runner=runner, **kwargs), ctx, runner


# --- V1：规则派工全链路 -------------------------------------------------------- #


def test_v1_full_trip_chain_dispatch_reviewless_render():
    engine, ctx, runner = make_engine(V1_CONFIG)
    outcome = engine.run_turn(
        ctx,
        None,
        "帮我规划杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        task_brief="杭州两日游",
        request_id="req_v1",
        turn_inputs={
            "artifact_reuse_audit": [
                {"artifact_id": "candidate_old", "reason": "constraint_changed"}
            ]
        },
    )
    assert outcome.status == STATUS_COMPLETED
    assert outcome.plan_artifact_id == ctx.store.latest_current_id("itinerary")
    assert outcome.gate_status == "rendered"
    assert any(card.get("type") == "summary" for card in outcome.cards)
    assert outcome.map_payload is None
    assert outcome.rework_used == 0
    finalization = next(
        item for item in ctx.store.latest("agent_trace")["items"]
        if item["kind"] == "finalization"
    )
    assert finalization["detail"]["delivery_status"] == DELIVERABLE_CURRENT
    assert finalization["detail"]["artifact_id"] == outcome.plan_artifact_id

    agents = [task.agent for task in runner._executor.tasks]
    assert agents == ["attraction", "transport", "planner"]
    # planner 只按 artifact_id 消费上游证据（inputs 注入了上游 artifact_ids）
    planner_task = runner._executor.tasks[-1]
    upstream_ids = set(planner_task.inputs.get("artifact_ids") or [])
    assert len(upstream_ids) == 2
    assert all(ctx.store.get_record(aid) is not None for aid in upstream_ids)
    # itinerary 记录元数据完整（Renderer Gate 依赖产出者校验）
    record = ctx.store.get_record(outcome.plan_artifact_id)
    assert record["agent"] == "planner" and record["request_id"] == "req_v1"
    contract = next(
        item for item in ctx.store.latest("agent_trace")["items"]
        if item["kind"] == "turn_contract"
    )
    assert contract["detail"] == {
        "task_type": "full_itinerary",
        "planner_required": True,
        "artifact_reuse_audit": [
            {"artifact_id": "candidate_old", "reason": "constraint_changed"}
        ],
    }


def test_v1_unknown_task_type_asks_clarification_without_dispatch():
    engine, ctx, runner = make_engine(V1_CONFIG)
    outcome = engine.run_turn(ctx, None, "随便聊聊", task_type=None)
    assert outcome.status == STATUS_CLARIFICATION_REQUIRED
    assert runner._executor.tasks == []
    assert ctx.store.latest_id("itinerary") is None


def test_v1_lightweight_task_without_planner_skips_render():
    engine, ctx, runner = make_engine(V1_CONFIG)
    outcome = engine.run_turn(ctx, None, "西湖到灵隐寺怎么走", task_type=TaskType.ROUTE_QUERY)
    assert outcome.status == STATUS_COMPLETED
    assert [task.agent for task in runner._executor.tasks] == ["transport"]
    assert outcome.plan_artifact_id is None
    assert outcome.gate_status == "skipped" and outcome.cards == []


# --- V2：动态 Orchestrator ------------------------------------------------------ #


def test_v2_dynamic_engine_aggregates_dispatch_results(monkeypatch):
    engine, ctx, _ = make_engine(V2_CONFIG)

    def fake_route(*args, **kwargs):
        return RoutingDecision(
            tasks=(
                RoutingTask(
                    agent="transport",
                    instruction="计算西湖到灵隐寺路线",
                    objective="西湖到灵隐寺路线",
                ),
            )
        )

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        fake_route,
    )
    outcome = engine.run_turn(
        ctx,
        None,
        "西湖到灵隐寺怎么走",
        request_id="req_v2",
        task_type=TaskType.ROUTE_QUERY,
    )
    assert outcome.status == STATUS_COMPLETED
    assert outcome.results[0].agent == "transport"
    assert outcome.results[0].evidence[0]["kind"] == "routes"
    # 无 planner 产出 → 不渲染、不 Review
    assert outcome.plan_artifact_id is None and outcome.gate_status == "skipped"


def test_v2_clarification_short_circuits(monkeypatch):
    engine, ctx, runner = make_engine(V2_CONFIG)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        lambda *args, **kwargs: RoutingDecision(
            clarification=True,
            reply="请告诉我目的地和天数。",
        ),
    )
    outcome = engine.run_turn(ctx, None, "帮我规划行程")
    assert outcome.status == STATUS_CLARIFICATION_REQUIRED
    assert "目的地" in outcome.reply


def test_v2_clarification_without_router_reply_uses_generic_fallback(monkeypatch):
    engine, ctx, runner = make_engine(V2_CONFIG)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        lambda *args, **kwargs: RoutingDecision(clarification=True),
    )

    outcome = engine.run_turn(
        ctx,
        None,
        "帮我处理一下这个旅行需求",
        task_type=TaskType.POI_ADVICE,
    )

    assert outcome.status == STATUS_CLARIFICATION_REQUIRED
    assert outcome.reply == "当前信息不足以安全执行，请补充具体的旅行目标或相关条件。"
    assert "目的地/天数" not in outcome.reply
    assert runner._executor.tasks == []


def test_v2_route_with_anchored_endpoints_overrides_router_clarification(monkeypatch):
    engine, ctx, runner = make_engine(V2_CONFIG)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        lambda *args, **kwargs: RoutingDecision(
            clarification=True,
            reply="请用户自己选择具体分店。",
        ),
    )
    profile = {
        "destination": "示例城",
        "constraint_state": {
            "origin": "示例城火车站",
            "destination_name": "中心商场",
            "need_disambiguation": True,
        },
    }

    outcome = engine.run_turn(
        ctx,
        None,
        "从火车站去中心商场，请先确认具体地点再给路线。",
        task_type=TaskType.ROUTE_PLAN,
        turn_inputs={"profile": profile},
    )

    assert outcome.status != STATUS_CLARIFICATION_REQUIRED
    assert [task.agent for task in runner._executor.tasks] == ["transport"]


def test_v2_full_plan_with_required_slots_overrides_soft_router_clarification(monkeypatch):
    engine, ctx, runner = make_engine(V2_CONFIG)
    ctx.profile.destination = "示例城"
    ctx.profile.days = 4
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        lambda *args, **kwargs: RoutingDecision(
            clarification=True,
            reply="请补充预算、兴趣和同行者信息。",
        ),
    )

    outcome = engine.run_turn(
        ctx,
        None,
        "示例城四天，三个人。",
        task_type=TaskType.FULL_ITINERARY,
        turn_inputs={
            "profile": {
                "destination": "示例城",
                "days": 4,
                "constraint_state": {
                    "destination_city": "示例城",
                    "duration_days": 4,
                    "traveler_count": 3,
                },
            }
        },
    )

    assert outcome.status != STATUS_CLARIFICATION_REQUIRED
    assert {task.agent for task in runner._executor.tasks} >= {"attraction", "transport"}


def test_orchestrator_agent_one_shot_wave_with_fake_llm(offline_settings):
    """Compatibility adapter consumes one Router response without a ReAct loop."""
    from travel_agent.orchestration.multi_agent.orchestrator_agent import run_orchestrator

    ctx = build_session(session_id="sess_orch", persist=False)
    runner = SubagentRunner(ctx, FakeDomainExecutor())
    model = ScriptedChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dispatch_subagent",
                        "args": {"agent": "transport", "instruction": "西湖到灵隐寺"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="已为您调研交通路线。"),
        ]
    )
    result = run_orchestrator(
        ctx, offline_settings, "西湖到灵隐寺怎么走", "req_orch", runner=runner, model=model
    )
    assert result["clarification"] is False
    assert result["reply"] == "领域调研已完成。"
    assert result["tool_trace"] == ["dispatch_subagent"]
    assert len(result["results"]) == 1
    assert result["results"][0]["agent"] == "transport"
    assert result["results"][0]["status"] == STATUS_COMPLETED


def test_nested_worker_callbacks_do_not_double_count_as_orchestrator(offline_settings):
    from travel_agent.orchestration.multi_agent.orchestrator_agent import run_orchestrator

    ctx = build_session(session_id="sess_nested_meter", persist=False)
    worker_model = ScriptedChatModel(messages=[AIMessage(content="住宿调研完成。")])
    runner = SubagentRunner(
        ctx,
        build_subagent_executor(offline_settings, model=worker_model),
    )
    router_model = ScriptedChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dispatch_subagent",
                        "args": {"agent": "hotel", "instruction": "调研住宿"},
                        "id": "call_nested",
                    }
                ],
            ),
            AIMessage(content="住宿调研完成。"),
        ]
    )
    meter = TurnMeter("req_nested_meter")

    with turn_meter_scope(meter):
        run_orchestrator(
            ctx,
            offline_settings,
            "杭州住哪里",
            "req_nested_meter",
            runner=runner,
            model=router_model,
        )

    roles = meter.snapshot()["roles"]
    assert roles["orchestrator"]["llm_calls"] == 1
    assert roles["worker:hotel"]["llm_calls"] == 1


def test_orchestrator_runs_independent_dispatches_from_one_turn_concurrently(
    offline_settings,
):
    from travel_agent.orchestration.multi_agent.orchestrator_agent import run_orchestrator
    from travel_agent.orchestration.multi_agent.schemas import SubagentResult

    class ConcurrentRunner:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def run_subagent(self, task):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return SubagentResult(
                request_id=task.request_id,
                task_id=task.task_id,
                agent=task.agent,
                status=STATUS_COMPLETED,
            )

    runner = ConcurrentRunner()
    ctx = build_session(session_id="sess_parallel_dispatch", persist=False)
    model = ScriptedChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dispatch_subagent",
                        "args": {"agent": "hotel", "instruction": "调研住宿"},
                        "id": "call_hotel",
                    },
                    {
                        "name": "dispatch_subagent",
                        "args": {"agent": "restaurant", "instruction": "调研餐饮"},
                        "id": "call_restaurant",
                    },
                ],
            ),
            AIMessage(content="调研完成。"),
        ]
    )

    result = run_orchestrator(
        ctx,
        offline_settings,
        "杭州住宿和餐饮建议",
        "req_parallel_dispatch",
        runner=runner,
        model=model,
    )

    assert runner.max_active == 2
    assert {item["agent"] for item in result["results"]} == {"hotel", "restaurant"}
    assert result["dispatch_ledger"].counts == {"hotel": 1, "restaurant": 1}


def test_orchestrator_tools_exclude_render_and_collect_sink(offline_settings):
    from travel_agent.orchestration.multi_agent.orchestrator_agent import (
        build_orchestrator_tools,
    )

    ctx = build_session(session_id="sess_orch_tools", persist=False)
    runner = SubagentRunner(ctx, FakeDomainExecutor())
    tools, sink = build_orchestrator_tools(runner, ctx, offline_settings, "req_tools")
    names = {tool.name for tool in tools}
    assert "dispatch_subagent" in names
    assert "request_travel_info" in names
    assert not {"render_itinerary", "render_map"} & names
    assert not {"search_poi", "plan_and_critique"} & names

    dispatch = next(tool for tool in tools if tool.name == "dispatch_subagent")
    wire_result = json.loads(dispatch.invoke({"agent": "hotel", "instruction": "住哪"}))
    assert len(sink) == 1 and sink[0]["agent"] == "hotel"
    assert sink[0]["payload"]
    assert "payload" not in wire_result
    assert wire_result["evidence"] == sink[0]["evidence"]


# --- V3：Reviewer + 最多一次修复周期 --------------------------------------------- #


def _counting_review(scripted: list[dict]):
    calls: list[ReviewContext] = []

    def callable_(review_ctx: ReviewContext) -> dict:
        calls.append(review_ctx)
        return scripted[min(len(calls) - 1, len(scripted) - 1)]

    return callable_, calls


def test_v3_review_pass_renders_completed():
    review_callable, calls = _counting_review([{"verdict": "pass", "issues": []}])
    engine, ctx, _ = make_engine(V3_FIXED, review_callable=review_callable)
    outcome = engine.run_turn(
        ctx, None, "杭州两日游", task_type=TaskType.FULL_TRIP_PLAN, request_id="req_v3"
    )
    assert outcome.status == STATUS_COMPLETED
    assert outcome.gate_status == "rendered"
    assert outcome.review is not None and outcome.review.verdict == "pass"
    assert len(calls) == 1  # Reviewer 只执行一次
    assert outcome.rework_used == 0


def test_v3_noncritical_reviewer_warning_remains_deliverable():
    review_callable, calls = _counting_review([{
        "verdict": "pass",
        "issues": [{
            "issue_type": "live_hours",
            "severity": "noncritical",
            "description": "出发前复核实时开放时间",
            "evidence": ["itinerary.day1"],
        }],
    }])
    engine, ctx, _ = make_engine(V3_FIXED, review_callable=review_callable)

    outcome = engine.run_turn(
        ctx, None, "杭州两日游", task_type=TaskType.FULL_ITINERARY,
        request_id="req_v3_warning",
    )

    assert outcome.status == STATUS_COMPLETED_WITH_WARNINGS
    assert outcome.gate_status == "rendered"
    assert outcome.delivery_status == DELIVERABLE_CURRENT
    assert outcome.rework_used == 0
    assert len(calls) == 1
    plan = ctx.store.get(outcome.plan_artifact_id)
    assert plan["review_advisories"] == ["出发前复核实时开放时间"]
    assert not plan.get("limitations")


def test_v3_recoverable_triggers_single_repair_cycle_without_second_review():
    rework = {
        "verdict": "rework",
        "issues": [
            {
                "issue_type": "pace",
                "severity": "recoverable",
                "description": "第二天景点过多",
                "evidence": ["day2 含 5 个景点"],
                "repair_target": "attraction",
                "repair_instruction": "为第二天补充低强度景点",
            }
        ],
    }
    review_callable, calls = _counting_review([rework])
    engine, ctx, runner = make_engine(V3_FIXED, review_callable=review_callable)
    outcome = engine.run_turn(
        ctx, None, "杭州两日游", task_type=TaskType.FULL_TRIP_PLAN, request_id="req_fix"
    )
    assert outcome.status == STATUS_COMPLETED_WITH_WARNINGS
    assert outcome.rework_used == 1
    assert len(calls) == 1  # 修复后不再二轮 Review
    assert outcome.gate_status == "rendered_incomplete"
    assert outcome.delivery_status == PARTIAL_CURRENT_WITH_LIMITATIONS

    tasks = runner._executor.tasks
    rework_tasks = [task for task in tasks if task.attempt == 2]
    assert [task.agent for task in rework_tasks] == ["attraction", "planner"]
    # 定向重派使用 Reviewer 的修复指令
    assert "低强度" in rework_tasks[0].instruction
    # planner 重规划只消费修复周期新产出的 artifact_id
    repair_attraction = outcome.results[-2]
    new_poi_ids = _evidence_artifact_ids(repair_attraction)
    assert new_poi_ids and rework_tasks[1].inputs.get("artifact_ids") == new_poi_ids
    assert "低强度" in rework_tasks[1].instruction
    # failed rework 不得覆盖 finalizer 选择的 current Artifact。
    assert outcome.plan_artifact_id == ctx.store.latest_current_id("itinerary")
    assert ctx.store.get(outcome.plan_artifact_id)["limitations"] == ["第二天景点过多"]


def test_v3_planner_only_repair_propagates_reviewer_directives():
    rework = {
        "verdict": "rework",
        "issues": [
            {
                "issue_type": "transport",
                "severity": "recoverable",
                "description": "缺少返程交通",
                "evidence": ["return_deadline=21:30"],
                "repair_target": "planner",
                "repair_instruction": "补充返程交通并校验21:30截止时间",
            }
        ],
    }
    review_callable, _calls = _counting_review([rework])
    engine, ctx, runner = make_engine(V3_FIXED, review_callable=review_callable)
    outcome = engine.run_turn(
        ctx, None, "杭州一日游，21:30前返回", task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req_planner_repair",
    )

    planner_repair = next(
        task for task in runner._executor.tasks if task.agent == "planner" and task.attempt == 2
    )
    assert "补充返程交通" in planner_repair.instruction
    assert planner_repair.inputs["revision_directives"]["reviewer_issue_types"] == [
        "transport"
    ]


def test_v3_critical_blocks_delivery_and_repair():
    critical = {
        "verdict": "rework",
        "issues": [
            {
                "issue_type": "budget",
                "severity": "critical",
                "description": "预算严重超支",
                "evidence": ["validation_result.issues[0]: budget_hard_limit_exceeded"],
                "repair_target": "hotel",
                "repair_instruction": "换低价酒店",
            }
        ],
    }
    review_callable, calls = _counting_review([critical])
    engine, ctx, runner = make_engine(V3_FIXED, review_callable=review_callable)
    outcome = engine.run_turn(
        ctx, None, "杭州两日游", task_type=TaskType.FULL_TRIP_PLAN, request_id="req_crit"
    )
    assert outcome.status == STATUS_INCOMPLETE
    assert outcome.rework_used == 0
    assert not any(task.attempt == 2 for task in runner._executor.tasks)
    # 无 current Artifact 时不得渲染 sidebar/map。
    assert outcome.gate_status == "skipped"
    assert outcome.cards == [] and outcome.map_payload is None
    assert len(calls) == 1


def test_only_deterministically_bound_critical_route_gap_is_repairable():
    route_review = ReviewResult(
        verdict="failed",
        issues=[ReviewIssue(
            issue_type="return_route_evidence_insufficient",
            severity="critical",
            description="最后一站到返程车站缺少可信路线。",
            evidence=["validation_result.issues[0].code=return_route_evidence_insufficient"],
            repair_target="transport",
            repair_instruction="重新取得该端点对的路线。",
        )],
    )
    route_plan = {
        "validation_result": {
            "passed": False,
            "issues": [{
                "code": "return_route_evidence_insufficient",
                "severity": "error",
            }],
        },
        "critic": {
            "passed": False,
            "issues": [{
                "code": "return_route_evidence_insufficient",
                "severity": "error",
            }],
        },
        "required_route_anchors": {
            "last_stop_to_return_location": {
                "origin_poi_id": "poi-last",
                "destination_poi_id": "poi-station",
                "evidence_status": "haversine_estimate",
            },
        },
    }

    assert _repairable_critical_route_targets(route_review, route_plan) == ["transport"]

    budget_review = ReviewResult(
        verdict="failed",
        issues=[ReviewIssue(
            issue_type="budget",
            severity="critical",
            description="预算严重超支。",
            evidence=["validation_result.issues[0].code=budget_hard_limit_exceeded"],
            repair_target="hotel",
            repair_instruction="换低价酒店。",
        )],
    )
    assert _repairable_critical_route_targets(budget_review, route_plan) == []


def test_route_repair_covers_return_fallbacks_and_intra_day_adjacency():
    plan = {
        "itinerary": {"days": [
            {"day_index": 1, "stops": [{"poi": {"poi_id": "day-one"}}]},
            {"day_index": 2, "stops": [
                {"poi": {"poi_id": "last-a"}},
                {"poi": {"poi_id": "last-b"}},
            ]},
        ]},
        "required_route_anchors": {
            "last_stop_to_return_location": {
                "origin_poi_id": "last-b",
                "destination_poi_id": "station",
                "evidence_status": "haversine_estimate",
            },
            "legs": [],
        },
    }

    assert _repair_route_pairs_for_plan(plan) == [
        ["last-b", "station"],
        ["last-a", "station"],
        ["last-a", "last-b"],
    ]


def test_route_repair_covers_each_intra_day_leg_without_cross_day_guessing():
    plan = {
        "itinerary": {"days": [
            {"day_index": 1, "stops": [
                {"poi": {"poi_id": "day1-a"}},
                {"poi": {"poi_id": "day1-b"}},
                {"poi": {"poi_id": "day1-c"}},
            ]},
            {"day_index": 2, "stops": [
                {"poi": {"poi_id": "day2-a"}},
                {"poi": {"poi_id": "day2-b"}},
            ]},
        ]},
        "required_route_anchors": {"legs": []},
    }

    assert _repair_route_pairs_for_plan(plan) == [
        ["day1-a", "day1-b"],
        ["day1-b", "day1-c"],
        ["day2-a", "day2-b"],
    ]


def test_return_route_rework_preserves_the_reviewed_terminal_endpoint():
    """A route-only rebuild must not invalidate the endpoint it just repaired."""
    from travel_agent.agent.toolkit import _apply_revision_directives
    from travel_agent.schemas import (
        CriticResult,
        Itinerary,
        ItineraryDay,
        ItineraryStop,
        ScoredPOI,
    )
    from travel_agent.planning_subgraph import PlanAndCritiqueResult

    terminal = POI(
        poi_id="reviewed-terminal",
        name="丝绸博物馆",
        city="测试城",
        category="museum",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["museum"],
        estimated_duration_min=90,
        price_level="mid",
    )
    other = replace(terminal, poi_id="other", name="围棋博物馆")
    itinerary = Itinerary(
        city="测试城",
        summary="两日行程",
        days=[
            ItineraryDay(1, "文化", [ItineraryStop(terminal, "09:00", 90, "参观")]),
            ItineraryDay(2, "文化", [ItineraryStop(other, "14:00", 90, "参观")]),
        ],
    )
    result = PlanAndCritiqueResult(
        itinerary=itinerary,
        original_itinerary=itinerary,
        critic_result=CriticResult(passed=True, issues=[]),
    )

    ctx = build_session(session_id="sess_preserve_terminal", persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=2)
    ranked = [ScoredPOI(other, 0.9, [])]
    repaired = _apply_revision_directives(
        result,
        ranked,
        ctx,
        {
            "reviewer_issue_types": ["return_route_evidence_insufficient"],
            "preserve_terminal_poi_id": "reviewed-terminal",
        },
    )

    assert repaired.itinerary.days[-1].stops[-1].poi.poi_id == "reviewed-terminal"
    assert {
        stop.poi.poi_id
        for day in repaired.itinerary.days
        for stop in day.stops
    } == {"reviewed-terminal", "other"}
    assert {item.poi.poi_id for item in ranked} == {"reviewed-terminal", "other"}


def test_return_route_rework_restores_a_valid_terminal_dropped_by_rebuild():
    from travel_agent.agent.toolkit import _apply_revision_directives
    from travel_agent.planning_subgraph import PlanAndCritiqueResult
    from travel_agent.schemas import (
        CriticResult,
        Itinerary,
        ItineraryDay,
        ItineraryStop,
        ScoredPOI,
    )

    terminal = POI(
        poi_id="reviewed-terminal",
        name="丝绸博物馆",
        city="测试城",
        category="museum",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["museum"],
        estimated_duration_min=90,
        price_level="mid",
        source="provider",
        verification_status="verified",
    )
    optional = replace(terminal, poi_id="optional", name="围棋博物馆")
    itinerary = Itinerary(
        city="测试城",
        summary="一日行程",
        days=[ItineraryDay(1, "文化", [ItineraryStop(optional, "14:00", 90, "参观")])],
    )
    result = PlanAndCritiqueResult(
        itinerary=itinerary,
        original_itinerary=itinerary,
        critic_result=CriticResult(passed=True, issues=[]),
    )
    ctx = build_session(session_id="sess_restore_terminal", persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=1)
    parent_id = ctx.store.put("itinerary", {
        "itinerary": {
            "city": "测试城",
            "days": [{
                "day_index": 1,
                "theme": "文化",
                "stops": [{
                    "poi": dataclasses.asdict(terminal),
                    "start_time": "14:00",
                    "duration_min": 90,
                    "note": "参观",
                    "route_from_previous": None,
                }],
            }],
            "summary": "父版本",
        }
    })
    ctx.remember_pois([optional])

    repaired = _apply_revision_directives(
        result,
        [ScoredPOI(optional, 0.9, [])],
        ctx,
        {
            "reviewer_issue_types": ["return_route_evidence_insufficient"],
            "preserve_terminal_poi_id": "reviewed-terminal",
            "preserve_terminal_poi": dataclasses.asdict(terminal),
            "parent_plan_artifact_id": parent_id,
        },
    )

    assert repaired.itinerary.days[-1].stops[-1].poi.poi_id == "reviewed-terminal"


def test_v3_failed_rework_preserves_same_revision_original_candidate():
    from travel_agent.artifact_policy import constraint_version, update_constraint_version
    from travel_agent.plan_invariants import validate_plan_artifact

    class FailingReworkExecutor(FakeDomainExecutor):
        def __call__(self, definition, task, ctx):
            self.tasks.append(task)
            if task.agent != "planner":
                kind = DOMAIN_KINDS[task.agent]
                ctx.store.put(kind, {"city": "杭州", "items": [{"poi_id": "e1"}]})
                return {"summary": f"{task.agent} complete"}
            if task.attempt == 2:
                return {"summary": "targeted rework failed before producing a plan"}
            version = constraint_version(ctx.profile)
            payload = {
                "artifact_status": "candidate",
                "state_version": {
                    "constraint_revision": version["revision"],
                    "constraint_hash": version["constraint_hash"],
                    "constraint_snapshot": version["constraint_snapshot"],
                },
                "itinerary": {"city": "杭州", "days": [{
                    "day_index": 1,
                    "stops": [{"poi": {
                        "poi_id": "p1", "name": "有效候选", "canonical_name": "有效候选",
                        "source": "provider", "verification_status": "verified",
                        "lng": 120.15, "lat": 30.25,
                    }}],
                }]},
                "critic": {"passed": True, "issues": []},
            }
            payload["validation_result"] = validate_plan_artifact(payload, ctx.profile)
            ctx.store.put("itinerary", payload)
            return {"summary": "candidate produced"}

    reviewer, _calls = _counting_review([{
        "verdict": "rework",
        "issues": [{
            "issue_type": "pace",
            "severity": "recoverable",
            "description": "可优化节奏",
            "evidence": ["itinerary.day1"],
            "repair_target": "planner",
            "repair_instruction": "尝试调整节奏",
        }],
    }])
    executor = FailingReworkExecutor()
    engine, ctx, _runner = make_engine(
        V3_FIXED, executor=executor, review_callable=reviewer
    )
    ctx.profile = TravelProfile(destination="杭州", days=1)
    update_constraint_version(ctx.profile, {})

    outcome = engine.run_turn(
        ctx, None, "杭州一日游", task_type=TaskType.FULL_ITINERARY,
        request_id="req_preserve_candidate",
    )

    assert outcome.status == STATUS_COMPLETED_WITH_WARNINGS
    assert outcome.rework_used == 1
    assert outcome.plan_artifact_id == ctx.store.latest_current_id("itinerary")
    assert ctx.store.get_record(outcome.plan_artifact_id)["artifact_status"] == "current"
    assert ctx.store.get(outcome.plan_artifact_id)["limitations"] == ["可优化节奏"]
    assert outcome.delivery_status == PARTIAL_CURRENT_WITH_LIMITATIONS
    trace = ctx.store.latest("agent_trace")["items"]
    assert any(item["kind"] == "candidate_preservation" for item in trace)


def test_reviewer_false_budget_omission_is_nonblocking_when_budget_plan_exists():
    review = run_semantic_review(
        ReviewContext(
            request_id="req_budget_composite",
            plan={
                "itinerary": {"days": []},
                "budget_plan": {"expected_total": 3000, "within_user_limit": True},
                "validation_result": {"passed": True, "issues": []},
                "critic": {"passed": True, "issues": []},
            },
        ),
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "budget",
                "severity": "recoverable",
                "description": "行程中缺少预算信息",
                "evidence": ["itinerary.days"],
                "repair_target": "planner",
                "repair_instruction": "补预算",
            }],
        },
    )

    status, repair = resolve_delivery_status(
        review, reviewer_enabled=True, max_rework=1, rework_used=0
    )
    assert status == STATUS_COMPLETED_WITH_WARNINGS
    assert repair is False


def test_v3_reviewer_unavailable_fails_closed(offline_settings):
    engine, ctx, _ = make_engine(V3_FIXED)  # 未注入 review_callable 且 LLM 离线
    outcome = engine.run_turn(
        ctx,
        offline_settings,
        "杭州两日游",
        task_type=TaskType.FULL_TRIP_PLAN,
        request_id="req_nor",
    )
    assert outcome.status == STATUS_INCOMPLETE
    assert outcome.gate_status == "skipped"
    assert outcome.review is None


def test_v3_lightweight_task_skips_review():
    review_callable, calls = _counting_review([{"verdict": "pass", "issues": []}])
    engine, ctx, _ = make_engine(V3_FIXED, review_callable=review_callable)
    outcome = engine.run_turn(ctx, None, "西湖怎么走", task_type=TaskType.ROUTE_QUERY)
    assert outcome.status == STATUS_COMPLETED
    assert calls == []  # 非完整行程任务不触发 Reviewer


def test_lightweight_candidate_selection_prefers_routed_and_diverse_entities():
    candidates = [
        {"poi_id": "mall-east", "name": "中心商场东门", "category": "shopping", "rating": 4.9},
        {"poi_id": "mall-west", "name": "中心商场西门", "category": "shopping", "rating": 4.8},
        {"poi_id": "museum", "name": "城市博物馆", "category": "museum", "rating": 4.7},
        {"poi_id": "park", "name": "人民公园", "category": "scenic", "rating": 4.6},
    ]
    routes = [
        {"destination_poi_id": "museum", "duration_min": 12},
        {"destination_poi_id": "park", "duration_min": 8},
    ]

    selected = _select_lightweight_candidates(
        candidates, routes, {"diversity_required": True}
    )

    assert [item["poi_id"] for item in selected[:2]] == ["park", "museum"]
    assert len([item for item in selected if "中心商场" in item["name"]]) == 1


def test_lightweight_readiness_requires_transport_for_location_distance():
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {"location": "成都春熙路附近"}
    from travel_agent.orchestration.multi_agent.schemas import SubagentResult

    prior = SubagentResult(
        request_id="req_light",
        task_id="attraction-1",
        agent="attraction",
        status=STATUS_COMPLETED,
        evidence=[{"artifact_id": "candidates_1", "kind": "candidates"}],
    )
    hard, _soft = _required_evidence(
        ctx,
        TaskType.POI_ADVICE,
        [prior],
        {},
    )
    assert hard == ["与指定位置的距离"]


def test_lightweight_readiness_rejects_empty_candidate_artifact_even_with_route():
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {
        "location": "测试中心附近",
        "walking_time_max_min": 15,
    }
    candidate_id = ctx.store.put("candidates", {"pois": []})
    route_id = ctx.store.put("routes", {"routes": []})

    hard, _soft = _required_evidence(
        ctx,
        TaskType.POI_ADVICE,
        [],
        {"artifact_ids": [candidate_id, route_id]},
    )

    assert hard == ["候选地点"]


def test_lightweight_readiness_accepts_nonempty_candidate_and_bound_route():
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {"location": "测试中心", "walking_time_max_min": 15}
    candidate_id = ctx.store.put(
        "candidates", {"pois": [{"poi_id": "candidate-1", "name": "候选甲"}]}
    )
    route_id = ctx.store.put("routes", {"routes": []})

    hard, _soft = _required_evidence(
        ctx,
        TaskType.POI_ADVICE,
        [],
        {"artifact_ids": [candidate_id, route_id]},
    )

    assert hard == []


def test_full_plan_readiness_does_not_request_restaurant_evidence_without_concrete_output():
    ctx = build_session(persist=False)
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})

    hard, soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id]},
    )

    assert hard == []
    assert soft == ["住宿证据", "预算证据"]


def test_full_plan_readiness_treats_lodging_downgrade_as_explicit_hotel_evidence() -> None:
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {"lodging_flexibility": "can_downgrade"}
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})

    hard, soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {
            "artifact_ids": [candidate_id, route_id],
            "task_brief": "预算降低，住宿可以降档，请重新生成完整行程",
        },
    )

    assert hard == ["住宿证据"]
    assert soft == ["预算证据"]


def test_full_plan_readiness_does_not_retrigger_hotel_from_historical_downgrade() -> None:
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {"lodging_flexibility": "can_downgrade"}
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})

    hard, soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {
            "artifact_ids": [candidate_id, route_id],
            "task_brief": "再增加一个室内备选并重新生成完整行程",
        },
    )

    assert hard == []
    assert soft == ["住宿证据", "预算证据"]


def test_full_plan_readiness_ignores_empty_restaurant_artifact_when_food_is_implicit():
    ctx = build_session(persist=False)
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put("restaurants", {"restaurants": []})

    hard, soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id]},
    )

    assert hard == []
    assert "餐饮证据" not in soft


def test_full_plan_readiness_accepts_nonempty_restaurant_artifact():
    ctx = build_session(persist=False)
    ctx.profile.days = 1
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put(
        "restaurants",
        {"restaurants": [{"poi_id": "meal-1", "name": "清真餐厅"}]},
    )

    hard, _soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id]},
    )

    assert hard == []


def test_full_plan_readiness_accepts_grounded_restaurant_without_one_per_day_rule():
    ctx = build_session(persist=False)
    ctx.profile.days = 2
    ctx.profile.food_preference = ["清真"]
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put(
        "restaurants",
        {"restaurants": [{"poi_id": "meal-1", "name": "餐厅一"}]},
    )

    hard, _soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id]},
    )

    assert hard == []


def test_full_plan_readiness_does_not_require_specific_hotel_for_area_only():
    ctx = build_session(persist=False)
    ctx.profile.hotel_area = "王府井附近"
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put("restaurants", {"restaurants": [{"name": "餐厅"}]})
    hotel_id = ctx.store.put("hotels", {"hotels": []})

    hard, _soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id, hotel_id]},
    )

    assert hard == []


def test_full_plan_readiness_requires_budget_artifact_for_explicit_limit():
    ctx = build_session(persist=False)
    ctx.profile.budget_limit = 4500
    candidate_id = ctx.store.put("candidates", {"pois": [{"poi_id": "sight-1"}]})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put("restaurants", {"restaurants": [{"name": "餐厅"}]})

    hard, _soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id]},
    )

    assert hard == ["预算证据"]


# --- Renderer Gate -------------------------------------------------------------- #


def test_render_gate_skips_without_plan_or_unrenderable_status():
    ctx = build_session(session_id="sess_gate1", persist=False)
    result = render_plan_outcome(ctx, None, STATUS_COMPLETED)
    assert result["rendered"] is False and result["gate_status"] == "skipped"

    plan_id = ctx.store.put("itinerary", dict(PLAN_PAYLOAD), agent="planner")
    result = render_plan_outcome(ctx, plan_id, STATUS_FAILED)
    assert result["rendered"] is False and result["gate_status"] == "skipped"


def test_render_gate_rejects_non_planner_artifact():
    ctx = build_session(session_id="sess_gate2", persist=False)
    fake_id = ctx.store.put("itinerary", dict(PLAN_PAYLOAD), agent="attraction")
    result = render_plan_outcome(ctx, fake_id, STATUS_COMPLETED)
    assert result["rendered"] is False and result["gate_status"] == "rejected"
    assert "planner" in result["reason"]


def test_render_gate_partial_current_marks_incomplete_without_unverified_map():
    ctx = build_session(session_id="sess_gate3", persist=False)
    plan_id = ctx.store.put(
        "itinerary", {**PLAN_PAYLOAD, "limitations": ["路线待补证"]}, agent="planner"
    )
    result = render_plan_outcome(ctx, plan_id, STATUS_INCOMPLETE)
    assert result["rendered"] is True
    assert result["gate_status"] == "rendered_incomplete"
    assert result["map_payload"] is None
    assert result["delivery_status"] == PARTIAL_CURRENT_WITH_LIMITATIONS


def test_render_gate_rejected_candidate_without_current_emits_no_sidebar_or_map():
    ctx = build_session(session_id="sess_gate_rejected", persist=False)
    candidate = ctx.store.put(
        "itinerary",
        {**PLAN_PAYLOAD, "artifact_status": "validation_failure"},
        agent="planner",
    )

    result = render_plan_outcome(ctx, candidate, STATUS_INCOMPLETE)

    assert result["rendered"] is False
    assert result["cards"] == []
    assert result["map_payload"] is None
    assert result["delivery_status"] == CANDIDATE_REJECTED


def test_delivery_snapshot_uses_final_store_not_reviewer_text():
    ctx = build_session(session_id="sess_delivery_source", persist=False)
    current = ctx.store.put("itinerary", dict(PLAN_PAYLOAD), agent="planner")
    snapshot = resolve_delivery_snapshot(ctx, attempted_artifact_id=current)
    assert snapshot.status == DELIVERABLE_CURRENT
    assert snapshot.artifact_id == current

    partial_ctx = build_session(session_id="sess_delivery_partial", persist=False)
    partial = partial_ctx.store.put(
        "itinerary",
        {**PLAN_PAYLOAD, "limitations": ["实时路线待复核"]},
        agent="planner",
    )
    snapshot = resolve_delivery_snapshot(partial_ctx, attempted_artifact_id=partial)
    assert snapshot.status == PARTIAL_CURRENT_WITH_LIMITATIONS


def test_delivery_snapshot_reports_rebuild_pending_without_current():
    ctx = build_session(session_id="sess_rebuild_pending", persist=False)
    ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"
    snapshot = resolve_delivery_snapshot(ctx)
    assert snapshot.status == REBUILD_PENDING
    assert snapshot.artifact_id is None


def test_render_gate_marks_required_lodging_without_evidence_incomplete():
    ctx = build_session(session_id="sess_gate_lodging", persist=False)
    from travel_agent.artifact_policy import constraint_version

    version = constraint_version(ctx.profile)
    payload = {
        **PLAN_PAYLOAD,
        "state_version": {
            "constraint_revision": version["revision"],
            "constraint_hash": version["constraint_hash"],
            "constraint_snapshot": version["constraint_snapshot"],
        },
        "lodging_plan": {
            "required": True,
            "explicit_requirement": True,
            "status": "evidence_unavailable",
        },
    }
    plan_id = ctx.store.put("itinerary", payload, agent="planner")
    from travel_agent.plan_invariants import validate_plan_artifact

    assert validate_plan_artifact(payload, ctx.profile)["issues"] == []

    result = render_plan_outcome(ctx, plan_id, STATUS_COMPLETED)

    assert result["rendered"] is True
    assert result["gate_status"] == "rendered_incomplete"
    assert result["delivery_status"] == PARTIAL_CURRENT_WITH_LIMITATIONS


def test_render_gate_accepts_user_owned_prepaid_lodging_without_provider_hotel():
    ctx = build_session(session_id="sess_gate_prepaid_lodging", persist=False)
    ctx.profile.constraint_state["prepaid_lodging_cny"] = 600.0
    from travel_agent.artifact_policy import update_constraint_version

    version = update_constraint_version(ctx.profile)
    payload = {
        **PLAN_PAYLOAD,
        "state_version": {
            "constraint_revision": version["revision"],
            "constraint_hash": version["constraint_hash"],
            "constraint_snapshot": version["constraint_snapshot"],
        },
        "lodging_plan": {
            "required": True,
            "explicit_requirement": False,
            "status": "user_owned_prepaid",
            "nights": 0,
            "prepaid_lodging_cny": 600.0,
            "evidence_status": "user_provided",
        },
        "budget_plan": {
            "constraint_revision": version["revision"],
            "constraint_hash": version["constraint_hash"],
            "people": 1,
            "days": 1,
            "breakdown_cny": {
                "lodging": 0.0,
                "transport": 0.0,
                "tickets": 0.0,
                "meals": 0.0,
                "fixed_event_cost": 0.0,
                "contingency": 0.0,
            },
            "unknown_items": [],
            "expected_total": 0.0,
            "high_total": 0.0,
        },
    }
    plan_id = ctx.store.put("itinerary", payload, agent="planner")
    from travel_agent.plan_invariants import validate_plan_artifact

    assert validate_plan_artifact(payload, ctx.profile)["issues"] == []

    result = render_plan_outcome(ctx, plan_id, STATUS_COMPLETED)

    assert result["gate_status"] == "rendered"
    assert result["delivery_status"] == DELIVERABLE_CURRENT


# --- executor：白名单与预算守卫 ---------------------------------------------------- #


def test_agent_middleware_applies_max_tokens_to_model_request():
    from langchain.agents.middleware import ModelRequest

    request = ModelRequest(model=ScriptedChatModel(messages=[]), messages=[])
    middleware = _max_output_tokens_middleware(2048)
    captured = {}

    def handler(updated):
        captured.update(updated.model_settings)
        return AIMessage(content="ok")

    middleware.wrap_model_call(request, handler)
    assert captured == {"max_tokens": 2048}


def test_executor_runs_react_loop_with_fake_model(offline_settings):
    ctx = build_session(session_id="sess_exec", persist=False)
    model = ScriptedChatModel(messages=[AIMessage(content="景点调研完成。")])
    executor = build_subagent_executor(offline_settings, model=model)
    result = executor(SUBAGENT_REGISTRY["attraction"], _task_for("attraction"), ctx)
    assert result["status"] == STATUS_COMPLETED
    assert result["summary"] == "景点调研完成。"
    assert result["tool_trace"] == []


def test_planner_executor_runs_mandatory_tool_chain_without_model(offline_settings):
    from travel_agent.agent import toolkit

    class ForbiddenModel:
        def bind_tools(self, *_args, **_kwargs):
            raise AssertionError("deterministic planner must not invoke the model")

    ctx = build_session(session_id="sess_deterministic_planner", persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1, interests=["自然"])
    candidates = toolkit.search_poi(
        ctx, city="杭州", interests=["自然"], max_results=6
    )
    task = _task_for("planner")
    task.inputs["artifact_ids"] = [candidates["artifact_id"]]

    result = SubagentRunner(
        ctx, build_subagent_executor(offline_settings, model=ForbiddenModel())
    ).run_subagent(task)

    assert result.status == STATUS_COMPLETED
    assert result.tool_trace == [
        "build_constraints",
        "recommend_candidates",
        "plan_and_critique",
    ]
    assert any(item["kind"] == "itinerary" for item in result.evidence)


def test_planner_executor_never_calls_live_route_provider(offline_settings):
    from travel_agent.agent import toolkit

    ctx = build_session(session_id="sess_planner_no_live_route", persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1, interests=["自然"])
    candidates = toolkit.search_poi(
        ctx, city="杭州", interests=["自然"], max_results=6
    )

    def forbidden_route(*_args, **_kwargs):
        raise AssertionError("planner must reuse evidence or local geometry")

    ctx.provider.estimate_route = forbidden_route
    task = _task_for("planner")
    task.inputs["artifact_ids"] = [candidates["artifact_id"]]

    result = SubagentRunner(
        ctx, build_subagent_executor(offline_settings, model=object())
    ).run_subagent(task, timeout_seconds=1)

    assert result.status == STATUS_COMPLETED
    assert any(item["kind"] == "itinerary" for item in result.evidence)


def test_attraction_executor_searches_missing_required_poi(offline_settings):
    required = POI(
        poi_id="gx_sxd",
        name="三星堆博物馆",
        city="成都",
        category="museum",
        lat=31.0,
        lng=104.2,
        rating=4.9,
        popularity=1.0,
        tags=["history"],
        estimated_duration_min=150,
        price_level="mid",
    )
    ctx = build_session(session_id="sess_required_poi", persist=False)
    ctx.profile = TravelProfile(destination="成都", days=2, must_visit=["三星堆"])
    required = replace(required, aliases=["三星堆"])
    ctx.provider = LocalToolProvider([required])
    model = ScriptedChatModel(messages=[AIMessage(content="景点调研完成。")])

    task = _task_for("attraction")
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = build_subagent_executor(offline_settings, model=model)(
            SUBAGENT_REGISTRY["attraction"], task, ctx
        )
    finally:
        reset_task_meta(token)

    assert result["status"] == STATUS_COMPLETED
    assert result["tool_trace"] == ["search_poi"]
    assert any(
        item["name"] == "三星堆博物馆"
        for artifact_id in ctx.store.artifact_ids()
        for item in ((ctx.store.get(artifact_id) or {}).get("pois") or [])
    )


def test_attraction_postcondition_searches_named_fixed_event_without_must_visit():
    venue = POI(
        poi_id="fixed-venue",
        name="城市科技馆",
        city="测试城",
        category="museum",
        lat=30.0,
        lng=120.0,
        rating=4.8,
        popularity=0.9,
        tags=["science"],
        estimated_duration_min=120,
        price_level="mid",
        entity_type="museum",
    )
    ctx = build_session(session_id="sess_fixed_event_poi", persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        must_visit=[],
        constraint_state={
            "fixed_events": [{
                "day": 1,
                "start": "14:00",
                "end": "16:00",
                "location": "城市科技馆",
            }],
        },
    )
    ctx.provider = LocalToolProvider([venue])
    task = _task_for("attraction")
    task.inputs.update({
        "task_type": "full_itinerary",
        "profile": {"constraint_state": {"duration_days": 1}},
    })
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = _enforce_required_poi_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert result["status"] == STATUS_COMPLETED
    assert "search_poi" in result["tool_trace"]
    assert any(
        item["name"] == "城市科技馆"
        for artifact_id in ctx.store.artifact_ids()
        for item in ((ctx.store.get(artifact_id) or {}).get("pois") or [])
    )


def test_required_museum_search_uses_general_museum_name_variants():
    museum = POI(
        poi_id="province-museum",
        name="测试博物院",
        city="测试城",
        category="museum",
        lat=30.0,
        lng=120.0,
        rating=4.9,
        popularity=1.0,
        tags=["history"],
        estimated_duration_min=120,
        price_level="mid",
        entity_type="museum",
    )

    class Provider:
        queries: list[str] = []

        def search_pois(self, city, query_tags=None, category=None, max_results=20):
            query = str((query_tags or [""])[0])
            self.queries.append(query)
            return [museum] if query == "测试省博物馆" else []

    ctx = build_session(session_id="sess_museum_alias", persist=False)
    ctx.profile = TravelProfile(
        destination="测试城", days=1, must_visit=["测试博物院"]
    )
    provider = Provider()
    ctx.provider = provider
    task = _task_for("attraction")
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = _enforce_required_poi_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert "测试博物院" in provider.queries
    assert "测试博物馆" in provider.queries
    assert "测试省博物馆" in provider.queries
    assert result["status"] == STATUS_COMPLETED


def test_full_itinerary_attraction_postcondition_broadens_sparse_named_supply():
    pois = [
        POI(
            poi_id=f"poi-{index}",
            name=name,
            city="测试城",
            category="museum" if index % 2 else "scenic",
            lat=30 + index / 100,
            lng=120 + index / 100,
            rating=4.5,
            popularity=0.8,
            tags=["culture" if index % 2 else "nature"],
            estimated_duration_min=90,
            price_level="mid",
            entity_type="museum" if index % 2 else "attraction",
        )
        for index, name in enumerate(
            ["湖畔公园", "城市博物馆", "山景台", "艺术馆", "古城公园", "历史馆"],
            start=1,
        )
    ]
    ctx = build_session(session_id="sess_sparse_named_supply", persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=3, must_visit=["湖畔公园"])
    ctx.provider = LocalToolProvider(pois)
    task = _task_for("attraction")
    task.inputs.update({
        "task_type": "full_itinerary",
        "profile": {"constraint_state": {"duration_days": 3}},
    })
    ctx.store.put(
        "candidates",
        {"pois": [dataclasses.asdict(pois[0])]},
        request_id=task.request_id,
        task_id=task.task_id,
        agent="attraction",
    )
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = _enforce_required_poi_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert result["tool_trace"] == ["search_poi"]
    assert sum(
        len((ctx.store.get(artifact_id) or {}).get("pois") or [])
        for artifact_id in ctx.store.artifact_ids_for_task(
            task.request_id, task.task_id, agent="attraction"
        )
    ) >= 6
    latest = ctx.store.latest("candidates")
    assert latest is not None
    assert {item["name"] for item in latest["pois"]} == {poi.name for poi in pois}
    assert latest["pois"][0]["name"] == "湖畔公园"


def test_attraction_postcondition_ignores_dispatch_snapshot_interests_not_in_profile(
    monkeypatch,
):
    ctx = build_session(session_id="sess_snapshot_interests", persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=1, interests=[])
    task = _task_for("attraction")
    task.inputs.update({
        "task_type": "full_itinerary",
        "profile": {"constraint_state": {"duration_days": 1, "interests": ["寺庙"]}},
    })
    calls: list[list[str]] = []

    def search_poi(_ctx, city=None, interests=None, category=None, max_results=30):
        calls.append(list(interests or []))
        return {"isError": False, "artifact_id": "candidate"}

    monkeypatch.setattr(toolkit, "search_poi", search_poi)
    result = _enforce_required_poi_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    assert ["寺庙"] not in calls
    assert result["status"] == STATUS_COMPLETED


def test_attraction_postcondition_clears_stale_missing_required_poi_error():
    required = POI(
        poi_id="museum-main",
        name="甲乙博物馆(本馆)",
        city="测试城",
        category="museum",
        lat=30.01,
        lng=120.01,
        rating=4.8,
        popularity=0.9,
        tags=["history"],
        estimated_duration_min=90,
        price_level="unknown",
        source="provider",
        canonical_name="甲乙博物馆(本馆)",
        entity_type="museum",
    )
    ctx = build_session(session_id="sess_stale_required_error", persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=1, must_visit=["甲乙博物馆"])
    ctx.provider = LocalToolProvider([required])
    task = _task_for("attraction")
    task.inputs.update({
        "task_type": "full_itinerary",
        "profile": {"constraint_state": {"duration_days": 1}},
    })
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = _enforce_required_poi_postcondition(
            task,
            ctx,
            {
                "status": STATUS_FAILED,
                "tool_trace": [],
                "unresolved": ["未检索到必去地点：甲乙博物馆"],
            },
        )
    finally:
        reset_task_meta(token)

    assert result["status"] == STATUS_COMPLETED
    assert result["unresolved"] == []
    assert _task_has_nonempty_domain_evidence(ctx, task)


def test_full_itinerary_supply_counts_one_named_attraction_family_once():
    names = [
        "湖畔公园主园区",
        "湖畔公园北门",
        "湖畔公园游客中心",
        "湖畔公园游船码头",
        "湖畔公园观景台",
        "湖畔公园文化广场",
        "城市博物馆",
        "山景台",
    ]
    pois = [
        POI(
            poi_id=f"family-{index}",
            name=name,
            city="测试城",
            category="museum" if "博物馆" in name else "scenic",
            lat=30 + index / 100,
            lng=120 + index / 100,
            rating=4.5,
            popularity=0.8,
            tags=["culture"],
            estimated_duration_min=90,
            price_level="mid",
            entity_type="museum" if "博物馆" in name else "attraction",
        )
        for index, name in enumerate(names, start=1)
    ]
    ctx = build_session(session_id="sess_one_named_family", persist=False)
    ctx.profile = TravelProfile(destination="测试城", days=3, must_visit=["湖畔公园"])
    ctx.provider = LocalToolProvider(pois)
    task = _task_for("attraction")
    task.inputs.update({
        "task_type": "full_itinerary",
        "profile": {"constraint_state": {"duration_days": 3}},
    })
    ctx.store.put(
        "candidates",
        {"pois": [dataclasses.asdict(poi) for poi in pois[:6]]},
        request_id=task.request_id,
        task_id=task.task_id,
        agent="attraction",
    )
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = _enforce_required_poi_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert result["tool_trace"] == ["search_poi"]
    latest = ctx.store.latest("candidates")
    assert latest is not None
    assert {item["name"] for item in latest["pois"]} == set(names)


def test_attraction_executor_searches_lossless_structured_interest(offline_settings):
    cafe = POI(
        poi_id="xm_cafe",
        name="自然地物·Camp Cafe",
        city="厦门",
        category="food",
        lat=24.48,
        lng=118.08,
        rating=4.8,
        popularity=0.9,
        tags=["coffee", "cafe"],
        estimated_duration_min=60,
        price_level="mid",
    )
    ctx = build_session(session_id="sess_structured_interest", persist=False)
    ctx.profile = TravelProfile(
        destination="厦门",
        days=2,
        interests=["food"],
        constraint_state={"interests": ["咖啡店"]},
    )
    ctx.provider = LocalToolProvider([cafe])
    model = ScriptedChatModel(messages=[AIMessage(content="景点调研完成。")])
    task = _task_for("attraction")
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = build_subagent_executor(offline_settings, model=model)(
            SUBAGENT_REGISTRY["attraction"], task, ctx
        )
    finally:
        reset_task_meta(token)

    assert result["status"] == STATUS_COMPLETED
    assert result["tool_trace"] == ["search_poi"]
    assert any(
        item["name"] == cafe.name
        for artifact_id in ctx.store.artifact_ids()
        for item in ((ctx.store.get(artifact_id) or {}).get("pois") or [])
    )


def test_executor_rejects_unsatisfiable_whitelist(offline_settings):
    ctx = build_session(session_id="sess_exec2", persist=False)
    model = ScriptedChatModel(messages=[AIMessage(content="x")])
    executor = build_subagent_executor(offline_settings, model=model)
    broken = dataclasses.replace(
        SUBAGENT_REGISTRY["hotel"], tool_names=("search_hotel", "no_such_tool")
    )
    with pytest.raises(RuntimeError, match="whitelist"):
        executor(broken, _task_for("hotel"), ctx)


def test_budget_guard_blocks_calls_over_limit():
    def ping() -> str:
        return "ok"

    tool = StructuredTool.from_function(ping, name="ping", description="")
    guarded = _budget_guard(tool, max_calls=1, counter={"n": 0})
    assert guarded.func() == "ok"
    with pytest.raises(ToolCallBudgetExceeded):
        guarded.func()


def test_budget_guard_records_only_executed_tool_names():
    counter = {"n": 0}
    tool = StructuredTool.from_function(lambda: "ok", name="ping", description="")
    guarded = _budget_guard(tool, max_calls=1, counter=counter)

    assert guarded.func() == "ok"
    with pytest.raises(ToolCallBudgetExceeded):
        guarded.func()
    assert counter["names"] == ["ping"]


def test_budget_guard_enforces_per_tool_limit_before_global_limit():
    counter = {"n": 0}
    tool = StructuredTool.from_function(lambda: "ok", name="search_poi", description="")
    guarded = _budget_guard(tool, max_calls=6, counter=counter, per_tool_max=2)

    assert guarded.func() == "ok"
    assert guarded.func() == "ok"
    with pytest.raises(ToolCallBudgetExceeded, match="search_poi call limit"):
        guarded.func()
    assert counter["n"] == 2


def test_budget_guard_rejects_retry_after_non_retryable_error():
    tool = StructuredTool.from_function(
        lambda: json.dumps(
            {
                "isError": True,
                "summary": "bad input",
                "error_code": "INVALID_INPUT",
                "retryable": False,
            }
        ),
        name="search_poi",
        description="",
    )
    guarded = _budget_guard(tool, max_calls=4, counter={"n": 0}, per_tool_max=2)

    guarded.func()
    with pytest.raises(ToolCallBudgetExceeded, match="non-retryable"):
        guarded.func()


def test_budget_guard_allows_exactly_one_retryable_retry():
    calls = 0

    def flaky() -> str:
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "isError": calls == 1,
                "summary": "temporary" if calls == 1 else "ok",
                "error_code": "UPSTREAM_UNAVAILABLE" if calls == 1 else None,
                "retryable": calls == 1,
            }
        )

    tool = StructuredTool.from_function(flaky, name="search_poi", description="")
    guarded = _budget_guard(tool, max_calls=4, counter={"n": 0}, per_tool_max=2)

    guarded.func()
    guarded.func()
    with pytest.raises(ToolCallBudgetExceeded):
        guarded.func()
    assert calls == 2


def test_transport_postcondition_builds_route_from_resolved_candidate_ids():
    from travel_agent.agent import toolkit

    ctx = build_session(session_id="sess_transport_postcondition", persist=False)
    ctx.profile.destination = "杭州"
    search = toolkit.search_poi(ctx, city="杭州", interests=["自然"], max_results=3)
    task = _task_for("transport")
    task.inputs["artifact_ids"] = [search["artifact_id"]]

    result = _enforce_transport_route_postcondition(
        task,
        ctx,
        {"status": STATUS_COMPLETED, "tool_trace": ["search_poi"]},
    )

    assert result["status"] == STATUS_COMPLETED
    assert result["tool_trace"][-1] == "plan_route"
    assert "transport route postcondition applied deterministically" in result["warnings"]
    assert ctx.store.latest("routes") is not None


def test_transport_postcondition_materializes_public_and_taxi_options():
    from travel_agent.agent import toolkit

    ctx = build_session(session_id="sess_transport_modes", persist=False)
    ctx.profile.destination = "杭州"
    ctx.profile.constraint_state["transport_modes"] = ["public_transport", "taxi"]
    search = toolkit.search_poi(ctx, city="杭州", interests=["自然"], max_results=3)
    task = _task_for("transport")
    task.inputs["artifact_ids"] = [search["artifact_id"]]

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": ["search_poi"]}
    )

    modes = {
        (ctx.store.get(artifact_id) or {}).get("mode")
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
    }
    assert {"public_transport", "taxi"} <= modes


def test_transport_postcondition_resolves_explicit_route_plan_endpoints():
    ctx = build_session(session_id="sess_explicit_route_plan", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "destination_city": "测试城",
        "origin": "中央车站",
        "destination_name": "城市展馆",
        "transport_modes": ["public_transport", "taxi"],
    }
    pois = [
        POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category="transport" if "站" in name else "museum",
            lat=30.0 + offset,
            lng=120.0 + offset,
            rating=4.5,
            popularity=0.8,
            tags=[name],
            estimated_duration_min=60,
            price_level="mid",
            entity_type="transport" if "站" in name else "museum",
        )
        for poi_id, name, offset in (
            ("origin", "中央车站", 0.0),
            ("destination", "城市展馆", 0.02),
        )
    ]
    ctx.provider = LocalToolProvider(pois)
    task = _task_for("transport")
    task.inputs.update({
        "task_type": "route_plan",
        "profile": {"constraint_state": dict(ctx.profile.constraint_state)},
    })

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    routes = [
        ctx.store.get(artifact_id)
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
    ]
    assert {
        (route["origin_poi_id"], route["destination_poi_id"], route["mode"])
        for route in routes
    } == {
        ("origin", "destination", "public_transport"),
        ("origin", "destination", "taxi"),
    }


def test_transport_postcondition_reuses_existing_ordered_route_endpoints():
    from travel_agent.agent import toolkit

    ctx = build_session(session_id="sess_transport_endpoint_binding", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state["transport_modes"] = ["public_transport", "taxi"]
    pois = [
        POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category="scenic",
            lat=30.0 + offset,
            lng=120.0 + offset,
            rating=4.5,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=90,
            price_level="mid",
        )
        for poi_id, name, offset in (
            ("endpoint-a", "端点甲", 0.0),
            ("distractor", "干扰点", 0.01),
            ("endpoint-b", "端点乙", 0.02),
        )
    ]
    ctx.remember_pois(pois)
    ctx.provider = LocalToolProvider(pois)
    candidate_id = ctx.store.put(
        "candidates", {"pois": [dataclasses.asdict(poi) for poi in pois]}
    )
    origin_id, destination_id = pois[0].poi_id, pois[-1].poi_id
    route = toolkit.plan_route(ctx, origin_id, destination_id, mode="public_transport")
    task = _task_for("transport")
    task.inputs["artifact_ids"] = [candidate_id, route["artifact_id"]]

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    taxi_routes = [
        ctx.store.get(artifact_id)
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
        and (ctx.store.get(artifact_id) or {}).get("mode") == "taxi"
    ]
    assert len(taxi_routes) == 1
    assert (
        taxi_routes[0]["origin_poi_id"],
        taxi_routes[0]["destination_poi_id"],
    ) == (origin_id, destination_id)


def test_transport_budget_only_contract_completes_without_model_round_trip():
    ctx = build_session(session_id="sess_transport_budget_only", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.days = 3
    ctx.profile.party_size = 2
    ctx.profile.budget_limit = 3000
    ctx.profile.constraint_state = {
        "destination_city": "测试城",
        "duration_days": 3,
        "traveler_count": 2,
        "budget_max_cny": 3000,
    }
    task = _task_for("transport")
    task.inputs["task_type"] = "full_itinerary"

    result = _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    assert result["tool_trace"] == ["estimate_budget"]
    assert result["_transport_contract_satisfied"] is True
    assert any(
        record.get("kind") == "budget"
        for record in ctx.store.snapshot_records().values()
    )


def test_transport_rework_adds_taxi_for_existing_return_deadline_pair():
    from travel_agent.agent import toolkit

    ctx = build_session(session_id="sess_return_rework_taxi", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "return_location": "南站",
        "return_deadline": "18:00",
    }
    sight = POI(
        poi_id="sight",
        name="湖畔公园",
        city="测试城",
        category="scenic",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["nature"],
        estimated_duration_min=90,
        price_level="mid",
    )
    station = replace(
        sight,
        poi_id="station",
        name="南站",
        category="transport",
        entity_type="transport",
    )
    ctx.provider = LocalToolProvider([sight, station])
    ctx.remember_pois([sight, station])
    route = toolkit.plan_route(ctx, "sight", "station", mode="public_transport")
    task = _task_for("transport")
    task.inputs["artifact_ids"] = [route["artifact_id"]]
    task.inputs["repair_route_pairs"] = [["sight", "station"]]

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    modes = {
        payload["mode"]
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
        if (payload := ctx.store.get(artifact_id))
    }
    assert modes == {"public_transport", "taxi"}


def test_transport_rework_does_not_treat_reverse_route_as_endpoint_evidence():
    from travel_agent.agent import toolkit

    ctx = build_session(session_id="sess_return_rework_direction", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "return_location": "南站",
        "return_deadline": "18:00",
    }
    sight = POI(
        poi_id="sight",
        name="湖畔公园",
        city="测试城",
        category="scenic",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["nature"],
        estimated_duration_min=90,
        price_level="mid",
    )
    station = replace(
        sight,
        poi_id="station",
        name="南站",
        category="transport",
        entity_type="transport",
    )
    ctx.provider = LocalToolProvider([sight, station])
    ctx.remember_pois([sight, station])
    reverse = toolkit.plan_route(
        ctx, "station", "sight", mode="public_transport"
    )
    task = _task_for("transport")
    task.inputs["artifact_ids"] = [reverse["artifact_id"]]
    task.inputs["repair_route_pairs"] = [["sight", "station"]]

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    directed_modes = {
        payload["mode"]
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
        if (payload := ctx.store.get(artifact_id))
        if payload["origin_poi_id"] == "sight"
        and payload["destination_poi_id"] == "station"
    }
    assert directed_modes == {"public_transport", "taxi"}


def test_transport_rework_replaces_unverified_existing_route_evidence():
    ctx = build_session(session_id="sess_return_rework_unverified", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "return_location": "南站",
        "return_deadline": "18:00",
    }
    sight = POI(
        poi_id="sight",
        name="湖畔公园",
        city="测试城",
        category="scenic",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["nature"],
        estimated_duration_min=90,
        price_level="mid",
    )
    station = replace(
        sight,
        poi_id="station",
        name="南站",
        category="transport",
        entity_type="transport",
    )
    ctx.provider = LocalToolProvider([sight, station])
    ctx.remember_pois([sight, station])
    unverified_id = ctx.store.put("routes", {
        "origin_poi_id": "sight",
        "destination_poi_id": "station",
        "origin_name": "湖畔公园",
        "destination_name": "南站",
        "distance_km": 2.0,
        "duration_min": 20,
        "mode": "public_transport",
        "source": "haversine_recovery_estimate",
        "evidence_status": "fallback_estimate",
    })
    task = _task_for("transport")
    task.inputs["artifact_ids"] = [unverified_id]
    task.inputs["repair_route_pairs"] = [["sight", "station"]]

    result = _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    assert result["tool_trace"] == ["plan_route", "plan_route"]
    route_modes = [
        payload["mode"]
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
        if (payload := ctx.store.get(artifact_id))
    ]
    assert route_modes.count("public_transport") == 2
    assert route_modes.count("taxi") == 1


def test_transport_postcondition_materializes_explicit_budget_evidence():
    ctx = build_session(session_id="sess_transport_budget", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.days = 3
    ctx.profile.budget_limit = 4200
    ctx.profile.constraint_state["budget_max_cny"] = 4200
    task = _task_for("transport")

    result = _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    assert result["tool_trace"] == ["estimate_budget"]
    assert "transport budget postcondition applied deterministically" in result["warnings"]
    assert ctx.store.latest("budget") is not None


def test_budget_artifact_survives_transport_route_budget_exhaustion() -> None:
    from travel_agent.orchestration.multi_agent.schemas import SubagentResult

    ctx = build_session(session_id="sess_partial_budget_evidence", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.days = 2
    ctx.profile.budget_limit = 3000
    ctx.profile.constraint_state = {"budget_max_cny": 3000}
    budget_id = ctx.store.put(
        "budget",
        {"city": "测试城", "days": 2, "total_low": 1000, "total_high": 1800},
        request_id="req",
        task_id="transport-task",
        agent="transport",
    )
    result = SubagentResult(
        request_id="req",
        task_id="transport-task",
        agent="transport",
        status="budget_exhausted",
        evidence=[{"artifact_id": budget_id, "kind": "budget"}],
    )

    missing, _soft = _required_evidence(
        ctx,
        TaskType.FULL_ITINERARY,
        [result],
        {"profile": {"constraint_state": dict(ctx.profile.constraint_state)}},
    )

    assert "预算证据" not in missing
    assert "路线可行性" in missing
    assert _all_evidence_ids([result]) == [budget_id]


def test_transport_postcondition_fans_out_routes_for_fixed_event() -> None:
    ctx = build_session(session_id="sess_transport_fixed_event", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "destination_city": "测试城",
        "fixed_events": [{
            "day": 2,
            "start": "14:00",
            "end": "16:00",
            "location": "城市展馆",
        }],
    }
    pois = [
        POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category=category,
            lat=30.0 + offset,
            lng=120.0 + offset,
            rating=4.5,
            popularity=0.8,
            tags=[name],
            estimated_duration_min=90,
            price_level="mid",
            entity_type=category,
        )
        for poi_id, name, category, offset in (
            ("candidate-a", "古城公园", "scenic", 0.01),
            ("candidate-b", "历史街区", "culture", 0.02),
            ("event", "城市展馆", "museum", 0.03),
        )
    ]
    ctx.provider = LocalToolProvider(pois)
    ctx.remember_pois(pois)
    candidates_id = ctx.store.put(
        "candidates", {"pois": [dataclasses.asdict(poi) for poi in pois[:2]]}
    )
    task = _task_for("transport")
    task.inputs.update({
        "artifact_ids": [candidates_id],
        "task_type": "full_itinerary",
        "profile": {"constraint_state": dict(ctx.profile.constraint_state)},
    })

    _enforce_transport_route_postcondition(
        task, ctx, {"status": "budget_exhausted", "tool_trace": []}
    )

    routes = [
        ctx.store.get(artifact_id)
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
    ]
    assert {
        (route["origin_poi_id"], route["destination_poi_id"])
        for route in routes
    } == {("candidate-a", "event"), ("candidate-b", "event")}
    assert {route["mode"] for route in routes} == {"public_transport", "taxi"}


def test_planner_route_estimator_binds_task_local_provider_route() -> None:
    ctx = build_session(session_id="sess_planner_task_route", persist=False)
    origin = POI(
        "origin", "上午景点", "测试城", "scenic", 30.0, 120.0,
        4.5, 0.8, [], 90, "mid",
    )
    event = POI(
        "event", "固定展馆", "测试城", "museum", 30.1, 120.1,
        4.5, 0.8, [], 90, "mid",
    )
    ctx.remember_pois([origin, event])
    route_id = ctx.store.put("routes", {
        "origin_poi_id": "origin", "destination_poi_id": "event",
        "distance_km": 12.0, "duration_min": 42,
        "mode": "public_transport", "source": "amap",
        "evidence_status": "provider_verified",
    })

    estimator = _planner_route_estimator(ctx, [route_id])
    route = estimator.estimate_route(origin, event, "public_transport")

    assert route.duration_min == 42
    assert route.source == "amap"
    assert route.evidence_status == "provider_verified"


def test_transport_postcondition_builds_adjacent_chain_for_ordinary_full_plan() -> None:
    ctx = build_session(session_id="sess_transport_full_chain", persist=False)
    ctx.profile.destination = "测试城"
    pois = [
        POI(
            poi_id=f"candidate-{index}",
            name=f"候选{index}",
            city="测试城",
            category="scenic",
            lat=30.0 + index / 100,
            lng=120.0 + index / 100,
            rating=4.5,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=90,
            price_level="mid",
        )
        for index in range(3)
    ]
    ctx.provider = LocalToolProvider(pois)
    ctx.remember_pois(pois)
    candidates_id = ctx.store.put(
        "candidates", {"pois": [dataclasses.asdict(poi) for poi in pois]}
    )
    task = _task_for("transport")
    task.inputs.update({
        "artifact_ids": [candidates_id],
        "task_type": "full_itinerary",
        "profile": {"constraint_state": {"destination_city": "测试城"}},
    })

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    routes = [
        ctx.store.get(artifact_id)
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
    ]
    assert {
        (route["origin_poi_id"], route["destination_poi_id"])
        for route in routes
    } == {("candidate-0", "candidate-1"), ("candidate-1", "candidate-2")}


def test_transport_postcondition_covers_explicit_origin_and_return_endpoints() -> None:
    ctx = build_session(session_id="sess_transport_trip_endpoints", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "destination_city": "测试城",
        "origin": "中央车站",
        "return_location": "南站",
        "return_deadline": "18:00",
    }
    pois = [
        POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category="transport" if "站" in name else "scenic",
            lat=30.0 + offset,
            lng=120.0 + offset,
            rating=4.5,
            popularity=0.8,
            tags=[name],
            estimated_duration_min=60,
            price_level="mid",
            entity_type="transport" if "站" in name else "attraction",
        )
        for poi_id, name, offset in (
            ("origin", "中央车站", 0.0),
            ("candidate-a", "湖畔公园", 0.01),
            ("candidate-b", "城市博物馆", 0.02),
            ("return", "南站", 0.03),
        )
    ]
    ctx.provider = LocalToolProvider(pois)
    ctx.remember_pois(pois)
    candidates_id = ctx.store.put(
        "candidates", {"pois": [dataclasses.asdict(poi) for poi in pois[1:3]]}
    )
    task = _task_for("transport")
    task.inputs.update({
        "artifact_ids": [candidates_id],
        "task_type": "full_itinerary",
        "profile": {"constraint_state": dict(ctx.profile.constraint_state)},
    })

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    pairs = {
        (payload["origin_poi_id"], payload["destination_poi_id"])
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
        if (payload := ctx.store.get(artifact_id))
    }
    assert pairs == {
        ("origin", "candidate-a"),
        ("origin", "candidate-b"),
        ("candidate-a", "return"),
        ("candidate-b", "return"),
    }


def test_transport_postcondition_covers_more_than_four_trip_candidates() -> None:
    ctx = build_session(session_id="sess_transport_many_endpoints", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "destination_city": "测试城",
        "origin": "中央车站",
        "return_location": "南站",
        "return_deadline": "18:00",
    }
    candidates = [
        POI(
            poi_id=f"candidate-{index}",
            name=f"景点{index}",
            city="测试城",
            category="scenic",
            lat=30.01 + index / 100,
            lng=120.01 + index / 100,
            rating=4.5,
            popularity=0.8,
            tags=["nature"],
            estimated_duration_min=60,
            price_level="mid",
            entity_type="attraction",
        )
        for index in range(6)
    ]
    endpoints = [
        POI(
            poi_id="origin",
            name="中央车站",
            city="测试城",
            category="transport",
            lat=30.0,
            lng=120.0,
            rating=4.5,
            popularity=0.8,
            tags=["station"],
            estimated_duration_min=30,
            price_level="mid",
            entity_type="transport",
        ),
        POI(
            poi_id="return",
            name="南站",
            city="测试城",
            category="transport",
            lat=30.09,
            lng=120.09,
            rating=4.5,
            popularity=0.8,
            tags=["station"],
            estimated_duration_min=30,
            price_level="mid",
            entity_type="transport",
        ),
    ]
    ctx.provider = LocalToolProvider([*endpoints, *candidates])
    ctx.remember_pois([*endpoints, *candidates])
    candidates_id = ctx.store.put(
        "candidates", {"pois": [dataclasses.asdict(poi) for poi in candidates]}
    )
    task = _task_for("transport")
    task.inputs.update({
        "artifact_ids": [candidates_id],
        "task_type": "full_itinerary",
        "profile": {"constraint_state": dict(ctx.profile.constraint_state)},
    })

    _enforce_transport_route_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    pairs = {
        (payload["origin_poi_id"], payload["destination_poi_id"])
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
        if (payload := ctx.store.get(artifact_id))
    }
    assert pairs == {
        pair
        for candidate in candidates
        for pair in (("origin", candidate.poi_id), (candidate.poi_id, "return"))
    }


def test_indoor_backup_postcondition_uses_turn_local_constraint_state():
    ctx = build_session(session_id="sess_indoor_backup", persist=False)
    ctx.profile.destination = "测试城"
    museum = POI(
        poi_id="indoor-museum",
        name="城市历史博物馆",
        city="测试城",
        category="museum",
        lat=30.0,
        lng=120.0,
        rating=4.6,
        popularity=0.8,
        tags=["博物馆", "室内"],
        estimated_duration_min=90,
        price_level="free",
        entity_type="museum",
    )
    ctx.provider = LocalToolProvider([museum])
    task = _task_for("attraction")
    task.inputs["profile"] = {
        "constraint_state": {
            "destination_city": "测试城",
            "need_indoor_backup": True,
        }
    }
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "attraction"}
    )
    try:
        result = _enforce_indoor_backup_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert result["tool_trace"] == ["search_poi"]
    assert "indoor backup evidence postcondition applied" in result["warnings"]
    artifact_ids = ctx.store.artifact_ids_for_task(
        task.request_id, task.task_id, agent="attraction"
    )
    assert any((ctx.store.get_record(aid) or {}).get("kind") == "candidates" for aid in artifact_ids)


def test_transport_postcondition_routes_discovered_restaurants_to_local_anchor():
    ctx = build_session(session_id="sess_restaurant_anchor_routes", persist=False)
    ctx.profile.destination = "测试城"
    pois = [
        POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category=category,
            lat=30.0 + offset,
            lng=120.0 + offset,
            rating=4.5,
            popularity=0.8,
            tags=tags,
            estimated_duration_min=60,
            price_level="mid",
        )
        for poi_id, name, category, tags, offset in (
            ("meal-a", "青禾餐厅", "food", ["晚餐"], 0.01),
            ("meal-b", "松风餐厅", "food", ["晚餐"], 0.02),
            ("anchor", "湖畔商务中心", "scenic", ["湖畔商务中心"], 0.03),
        )
    ]
    ctx.provider = LocalToolProvider(pois)
    ctx.remember_pois(pois[:2])
    restaurants_id = ctx.store.put(
        "restaurants", {"restaurants": [dataclasses.asdict(poi) for poi in pois[:2]]}
    )
    task = _task_for("transport")
    task.inputs.update({
        "artifact_ids": [restaurants_id],
        "profile": {
            "constraint_state": {
                "destination_city": "测试城",
                "location_anchor": "湖畔商务中心",
                "walking_time_max_min": 10,
                "top_n": 2,
            }
        },
    })
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "transport"}
    )
    try:
        _enforce_transport_route_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    routes = [
        ctx.store.get(artifact_id)
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
    ]
    assert len(routes) == 2
    assert {route["origin_poi_id"] for route in routes} == {"meal-a", "meal-b"}
    assert {route["destination_poi_id"] for route in routes} == {"anchor"}
    assert {route["mode"] for route in routes} == {"walk"}


def test_restaurant_postcondition_replaces_empty_artifact_with_real_candidates():
    ctx = build_session(session_id="sess_empty_restaurant_recovery", persist=False)
    ctx.profile.destination = "测试城"
    restaurant = POI(
        poi_id="meal-a",
        name="青禾餐厅",
        city="测试城",
        category="food",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["餐厅", "聚餐"],
        estimated_duration_min=60,
        price_level="mid",
        entity_type="restaurant",
    )
    ctx.provider = LocalToolProvider([restaurant])
    task = _task_for("restaurant")
    task.inputs["profile"] = {
        "constraint_state": {
            "destination_city": "测试城",
            "location_anchor": "湖畔商务中心",
            "top_n": 3,
        }
    }
    ctx.store.put(
        "restaurants",
        {"restaurants": []},
        request_id=task.request_id,
        task_id=task.task_id,
        agent="restaurant",
    )
    assert not _task_has_nonempty_domain_evidence(ctx, task)

    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "restaurant"}
    )
    try:
        result = _enforce_restaurant_search_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert result["tool_trace"] in (
        ["search_restaurant"],
        ["search_restaurant", "search_restaurant"],
    )
    assert _task_has_nonempty_domain_evidence(ctx, task)


def test_restaurant_postcondition_searches_near_hard_anchors_for_multiday_dietary_trip(monkeypatch):
    ctx = build_session(session_id="sess_anchor_meal_recovery", persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=3,
        must_visit=["远郊遗址", "中心城墙"],
        constraint_state={
            "destination_city": "测试城",
            "duration_days": 3,
            "dietary": ["仅清真餐厅"],
            "must_visit": ["远郊遗址", "中心城墙"],
        },
    )
    task = _task_for("restaurant")
    task.inputs["profile"] = {"constraint_state": dict(ctx.profile.constraint_state)}
    searched_areas = []

    def fake_search(_ctx, **kwargs):
        searched_areas.append(kwargs.get("area"))
        return {"isError": False, "count": 3, "artifact_id": f"meal-{len(searched_areas)}"}

    monkeypatch.setattr(toolkit, "search_restaurant", fake_search)

    result = _enforce_restaurant_search_postcondition(
        task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
    )

    assert searched_areas == [None, "远郊遗址", "中心城墙"]
    assert result["tool_trace"] == [
        "search_restaurant", "search_restaurant", "search_restaurant",
    ]


def test_hotel_postcondition_materializes_grounded_candidates():
    ctx = build_session(session_id="sess_empty_hotel_recovery", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.hotel_area = "湖畔区"
    hotel = POI(
        poi_id="hotel-a",
        name="湖畔区旅馆",
        city="测试城",
        category="hotel",
        lat=30.01,
        lng=120.01,
        rating=4.5,
        popularity=0.8,
        tags=["酒店"],
        estimated_duration_min=0,
        price_level="mid",
        address="测试城湖畔区一号",
        entity_type="hotel",
    )
    ctx.provider = LocalToolProvider([hotel])
    task = _task_for("hotel")
    task.inputs["profile"] = {
        "constraint_state": {
            "destination_city": "测试城",
            "lodging_area": "湖畔区",
        }
    }
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "hotel"}
    )
    try:
        result = _enforce_hotel_search_postcondition(
            task, ctx, {"status": "budget_exhausted", "tool_trace": []}
        )
    finally:
        reset_task_meta(token)

    assert result["tool_trace"] == ["search_hotel"]
    assert _task_has_nonempty_domain_evidence(ctx, task)


def test_transport_postcondition_evidence_recovers_budget_exhausted_status():
    ctx = build_session(session_id="sess_transport_status_recovery", persist=False)
    task = _task_for("transport")
    ctx.store.put(
        "routes",
        {"origin_poi_id": "a", "destination_poi_id": "b", "duration_min": 12},
        request_id=task.request_id,
        task_id=task.task_id,
        agent="transport",
    )

    result = _promote_recovered_worker_status(
        task,
        ctx,
        {"status": "budget_exhausted", "warnings": ["redundant call rejected"]},
    )

    assert result["status"] == STATUS_COMPLETED


def test_unanchored_comparison_transit_entities_recover_transport_worker_status():
    ctx = build_session(session_id="sess_unanchored_comparison_recovery", persist=False)
    task = _task_for("transport")
    task.inputs.update({
        "task_type": "candidate_comparison",
        "profile": {
            "constraint_state": {
                "comparison_candidates": ["甲区", "乙区"],
                "comparison_dimensions": ["accessibility"],
            },
        },
    })
    ctx.store.put(
        "candidates",
        {"pois": [{"poi_id": "station", "name": "甲区地铁站"}]},
        request_id=task.request_id,
        task_id=task.task_id,
        agent="transport",
    )

    result = _promote_recovered_worker_status(
        task,
        ctx,
        {"status": "budget_exhausted", "warnings": ["redundant call rejected"]},
    )

    assert result["status"] == STATUS_COMPLETED


def test_transport_comparison_postcondition_covers_every_candidate_to_anchor():
    ctx = build_session(session_id="sess_transport_comparison_coverage", persist=False)
    ctx.profile.destination = "测试城"
    ctx.profile.constraint_state = {
        "comparison_candidates": ["甲区", "乙区", "丙区"],
        "comparison_dimensions": ["accessibility"],
        "target_anchor": "中央车站",
    }
    pois = [
        POI(
            poi_id=poi_id,
            name=name,
            city="测试城",
            category="scenic",
            lat=30.0 + offset,
            lng=120.0 + offset,
            rating=4.5,
            popularity=0.8,
            tags=[term],
            estimated_duration_min=60,
            price_level="mid",
            entity_type="district",
        )
        for poi_id, name, term, offset in (
            ("area-a", "甲区生活圈", "甲区", 0.01),
            ("area-b", "乙区生活圈", "乙区", 0.02),
            ("area-c", "丙区生活圈", "丙区", 0.03),
            ("anchor", "中央车站", "中央车站", 0.05),
        )
    ]
    ctx.provider = LocalToolProvider(pois)
    task = _task_for("transport")
    task.inputs["profile"] = {"constraint_state": dict(ctx.profile.constraint_state)}
    token = set_current_task_meta(
        {"request_id": task.request_id, "task_id": task.task_id, "agent": "transport"}
    )
    try:
        result = _enforce_comparison_search_postcondition(
            task, ctx, {"status": STATUS_COMPLETED, "tool_trace": []}
        )
        _enforce_transport_route_postcondition(task, ctx, result)
    finally:
        reset_task_meta(token)

    routes = [
        ctx.store.get(artifact_id)
        for artifact_id, record in ctx.store.snapshot_records().items()
        if record.get("kind") == "routes"
    ]
    assert len(routes) == 3
    assert {route["origin_name"] for route in routes} == {
        "甲区生活圈", "乙区生活圈", "丙区生活圈"
    }
    assert {route["destination_name"] for route in routes} == {"中央车站"}


def _task_for(agent: str):
    from travel_agent.orchestration.multi_agent.schemas import SubagentTask, new_task_id

    return SubagentTask(
        request_id="req_exec", task_id=new_task_id(agent), agent=agent, instruction="测试任务"
    )


def _evidence_artifact_ids(result) -> list[str]:
    return [
        item["artifact_id"]
        for item in result.evidence
        if isinstance(item, dict) and item.get("artifact_id")
    ]


# --- review：JSON 提取与真实 callable --------------------------------------------- #


def test_extract_json_payload_tolerant():
    assert extract_json_payload('{"verdict": "pass"}') == {"verdict": "pass"}
    fenced = '```json\n{"verdict": "rework", "issues": []}\n```'
    assert extract_json_payload(fenced)["verdict"] == "rework"
    wrapped = '结论如下 {"verdict": "failed"} 以上'
    assert extract_json_payload(wrapped)["verdict"] == "failed"
    assert extract_json_payload("没有 JSON") == {}


def test_build_review_callable_invokes_model_once_without_tools(offline_settings):
    model = ScriptedChatModel(
        messages=[AIMessage(content='{"verdict": "pass", "issues": []}')]
    )
    callable_ = build_review_callable(offline_settings, model=model)
    result = callable_(ReviewContext(request_id="req_r", plan={"days": []}))
    assert result == {"verdict": "pass", "issues": []}
    assert model.i == 1  # 单层、只调一次


def test_build_review_callable_retries_one_invalid_shape(offline_settings):
    model = ScriptedChatModel(messages=[
        AIMessage(content=""),
        AIMessage(content='{"verdict": "pass", "issues": []}'),
    ])

    callable_ = build_review_callable(offline_settings, model=model)
    result = callable_(ReviewContext(request_id="req_retry", plan={"days": []}))

    assert result == {"verdict": "pass", "issues": []}
    assert model.i == 2


def test_build_review_callable_retries_rework_without_recoverable_issue(offline_settings):
    model = ScriptedChatModel(messages=[
        AIMessage(content='{"verdict": "rework", "issues": []}'),
        AIMessage(content='{"verdict": "pass", "issues": []}'),
    ])

    callable_ = build_review_callable(offline_settings, model=model)
    result = callable_(ReviewContext(request_id="req_retry_rework", plan={"days": []}))

    assert result == {"verdict": "pass", "issues": []}
    assert model.i == 2


def test_build_review_callable_retries_issue_with_invalid_repair_target(offline_settings):
    model = ScriptedChatModel(messages=[
        AIMessage(content=json.dumps({
            "verdict": "rework",
            "issues": [{
                "issue_type": "budget_gap",
                "severity": "recoverable",
                "description": "预算需要调整",
                "evidence": ["budget_plan.expected_total"],
                "repair_target": "budget_agent",
                "repair_instruction": "降低可选活动支出",
            }],
        }, ensure_ascii=False)),
        AIMessage(content='{"verdict": "pass", "issues": []}'),
    ])

    callable_ = build_review_callable(offline_settings, model=model)
    result = callable_(ReviewContext(request_id="req_retry_bad_target", plan={"days": []}))

    assert result == {"verdict": "pass", "issues": []}
    assert model.i == 2


def test_build_review_callable_does_not_retry_valid_failed_verdict(offline_settings):
    model = ScriptedChatModel(messages=[AIMessage(content=json.dumps({
        "verdict": "failed",
        "issues": [{
            "issue_type": "hard_conflict",
            "severity": "critical",
            "description": "明确冲突",
            "evidence": ["trace"],
            "repair_target": "planner",
            "repair_instruction": "修复",
        }],
    }, ensure_ascii=False))])

    callable_ = build_review_callable(offline_settings, model=model)
    result = callable_(ReviewContext(request_id="req_failed", plan={"days": []}))

    assert result["verdict"] == "failed"
    assert model.i == 1


def test_review_context_compacts_map_only_poi_fields() -> None:
    ctx = ReviewContext(
        request_id="req_r",
        plan={
            "itinerary": {
                "city": "杭州",
                "days": [
                    {
                        "day_index": 1,
                        "stops": [
                            {
                                "poi": {
                                    "name": "西湖",
                                    "lat": 30.2,
                                    "lng": 120.1,
                                    "opening_hours": "周一至周日 00:00-24:00",
                                },
                                "start_time": "09:00",
                            }
                        ],
                    }
                ],
            },
            "budget_plan": {"expected_total": 3200, "within_user_limit": True},
            "lodging_plan": {"status": "grounded", "area_requirement": "西湖附近"},
            "required_route_anchors": [{"origin": "酒店", "destination": "西湖"}],
            "meal_strategy": {"dietary_constraints": ["清淡"]},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={"constraint_state": {"budget_max_cny": 5000}},
    )

    payload = json.loads(ctx.to_prompt_text())
    poi = payload["plan"]["itinerary"]["days"][0]["stops"][0]["poi"]
    assert poi == {"name": "西湖", "opening_hours": "周一至周日 00:00-24:00"}
    assert payload["plan"]["budget_plan"]["expected_total"] == 3200
    assert payload["plan"]["lodging_plan"]["status"] == "grounded"
    assert payload["plan"]["route_evidence"]["required_route_anchors"]
    assert payload["plan"]["meal_strategy"]["dietary_constraints"] == ["清淡"]
    assert payload["plan"]["validation_result"]["passed"] is True
    assert payload["plan"]["active_constraints"]["budget_max_cny"] == 5000
    assert REVIEWER_MAX_OUTPUT_TOKENS == 1024
    assert REVIEWER_TIMEOUT_SECONDS == 90


def test_build_review_callable_returns_none_when_llm_disabled(offline_settings):
    assert offline_settings.llm.enabled is False
    assert build_review_callable(offline_settings) is None
