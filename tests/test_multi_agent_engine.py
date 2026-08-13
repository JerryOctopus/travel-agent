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
from travel_agent.agent.turn_analysis import TaskType
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
    _required_evidence,
    _select_lightweight_candidates,
)
from travel_agent.orchestration.multi_agent.executor import (
    _BUDGET_SENTINEL,
    _max_output_tokens_middleware,
    _budget_guard,
    _enforce_transport_route_postcondition,
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
            ctx.store.put("itinerary", dict(PLAN_PAYLOAD))
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
    )
    assert outcome.status == STATUS_COMPLETED
    assert outcome.plan_artifact_id == ctx.store.latest_id("itinerary")
    assert outcome.gate_status == "rendered"
    assert any(card.get("type") == "summary" for card in outcome.cards)
    assert outcome.map_payload is not None
    assert outcome.rework_used == 0

    agents = [task.agent for task in runner._executor.tasks]
    assert set(agents[:3]) == {"attraction", "hotel", "restaurant"}
    assert agents[3:] == ["transport", "planner"]
    # planner 只按 artifact_id 消费上游证据（inputs 注入了上游 artifact_ids）
    planner_task = runner._executor.tasks[-1]
    upstream_ids = set(planner_task.inputs.get("artifact_ids") or [])
    assert len(upstream_ids) == 4
    assert all(ctx.store.get_record(aid) is not None for aid in upstream_ids)
    # itinerary 记录元数据完整（Renderer Gate 依赖产出者校验）
    record = ctx.store.get_record(outcome.plan_artifact_id)
    assert record["agent"] == "planner" and record["request_id"] == "req_v1"


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
    assert outcome.gate_status == "rendered"

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
    # plan_artifact_id 更新为重规划产物
    assert outcome.plan_artifact_id == ctx.store.latest_id("itinerary")


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
    # critical 只能渲染 incomplete 状态
    assert outcome.gate_status == "rendered_incomplete"
    assert len(calls) == 1


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
    assert outcome.gate_status == "rendered_incomplete"
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


def test_lightweight_readiness_accepts_bound_route_artifact_without_duplicate_label():
    ctx = build_session(persist=False)
    ctx.profile.constraint_state = {
        "location": "成都春熙路附近",
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

    assert hard == []


def test_full_plan_readiness_treats_meal_evidence_as_soft_without_food_preference():
    ctx = build_session(persist=False)
    candidate_id = ctx.store.put("candidates", {"pois": []})
    route_id = ctx.store.put("routes", {"routes": []})

    hard, soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id]},
    )

    assert hard == []
    assert soft == ["住宿证据", "预算证据", "餐饮证据"]


def test_full_plan_readiness_allows_empty_restaurant_artifact_when_food_is_implicit():
    ctx = build_session(persist=False)
    candidate_id = ctx.store.put("candidates", {"pois": []})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put("restaurants", {"restaurants": []})

    hard, soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id]},
    )

    assert hard == []
    assert "餐饮证据" in soft


def test_full_plan_readiness_accepts_nonempty_restaurant_artifact():
    ctx = build_session(persist=False)
    ctx.profile.days = 1
    candidate_id = ctx.store.put("candidates", {"pois": []})
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


def test_full_plan_readiness_requires_one_distinct_restaurant_per_day():
    ctx = build_session(persist=False)
    ctx.profile.days = 2
    ctx.profile.food_preference = ["清真"]
    candidate_id = ctx.store.put("candidates", {"pois": []})
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

    assert hard == ["餐饮证据"]


def test_full_plan_readiness_rejects_empty_explicit_hotel_artifact():
    ctx = build_session(persist=False)
    ctx.profile.hotel_area = "王府井附近"
    candidate_id = ctx.store.put("candidates", {"pois": []})
    route_id = ctx.store.put("routes", {"routes": []})
    restaurant_id = ctx.store.put("restaurants", {"restaurants": [{"name": "餐厅"}]})
    hotel_id = ctx.store.put("hotels", {"hotels": []})

    hard, _soft = _required_evidence(
        ctx,
        TaskType.FULL_TRIP_PLAN,
        [],
        {"artifact_ids": [candidate_id, route_id, restaurant_id, hotel_id]},
    )

    assert hard == ["住宿证据"]


def test_full_plan_readiness_requires_budget_artifact_for_explicit_limit():
    ctx = build_session(persist=False)
    ctx.profile.budget_limit = 4500
    candidate_id = ctx.store.put("candidates", {"pois": []})
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


def test_render_gate_incomplete_marks_incomplete():
    ctx = build_session(session_id="sess_gate3", persist=False)
    plan_id = ctx.store.put("itinerary", dict(PLAN_PAYLOAD), agent="planner")
    result = render_plan_outcome(ctx, plan_id, STATUS_INCOMPLETE)
    assert result["rendered"] is True
    assert result["gate_status"] == "rendered_incomplete"
    assert result["map_payload"] is not None


def test_render_gate_marks_required_lodging_without_evidence_incomplete():
    ctx = build_session(session_id="sess_gate_lodging", persist=False)
    payload = {
        **PLAN_PAYLOAD,
        "lodging_plan": {
            "required": True,
            "explicit_requirement": True,
            "status": "evidence_unavailable",
        },
    }
    plan_id = ctx.store.put("itinerary", payload, agent="planner")

    result = render_plan_outcome(ctx, plan_id, STATUS_COMPLETED)

    assert result["rendered"] is True
    assert result["gate_status"] == "rendered_incomplete"


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
            }
        },
    )

    payload = json.loads(ctx.to_prompt_text())
    poi = payload["plan"]["itinerary"]["days"][0]["stops"][0]["poi"]
    assert poi == {"name": "西湖", "opening_hours": "周一至周日 00:00-24:00"}
    assert REVIEWER_MAX_OUTPUT_TOKENS == 1024
    assert REVIEWER_TIMEOUT_SECONDS == 90


def test_build_review_callable_returns_none_when_llm_disabled(offline_settings):
    assert offline_settings.llm.enabled is False
    assert build_review_callable(offline_settings) is None
