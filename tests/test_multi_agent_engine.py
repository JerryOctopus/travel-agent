"""multi_agent Step 3 单元测试：真实编排链路的离线验证。

覆盖：

- V1 规则派工全链路（真实 SubagentRunner + Fake executor + Renderer Gate）；
- V2 动态 Orchestrator（Scripted Fake LLM 驱动 create_react_agent）与
  Engine 对 clarification / 结果的聚合；
- V3 Reviewer 一次调用 + 最多一次修复周期（recoverable 修复 / critical 禁交付 /
  Reviewer 不可用 fail-closed）；
- Renderer Artifact Gate（无 plan 跳过 / 状态拦截 / 产出者校验 / incomplete 渲染）；
- executor 白名单与预算守卫；review JSON 提取与真实 callable 构建。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent import (
    SubagentRunner,
    render_plan_outcome,
)
from travel_agent.orchestration.multi_agent.engine import (
    V1_CONFIG,
    V2_CONFIG,
    MultiAgentEngine,
)
from travel_agent.orchestration.multi_agent.executor import (
    _BUDGET_SENTINEL,
    _budget_guard,
    build_subagent_executor,
    ToolCallBudgetExceeded,
)
from travel_agent.orchestration.multi_agent.registry import SUBAGENT_REGISTRY
from travel_agent.orchestration.multi_agent.review import (
    ReviewContext,
    build_review_callable,
    extract_json_payload,
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

    def bind_tools(self, tools, **kwargs):  # create_react_agent 需要
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
    assert agents == ["attraction", "hotel", "restaurant", "transport", "planner"]
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

    def fake_orchestrator(ctx_, settings_, message, request_id, *, runner, model=None, results_sink=None):
        return {
            "reply": "已安排交通调研。",
            "tool_trace": ["dispatch_subagent"],
            "results": [
                {
                    "request_id": request_id,
                    "task_id": "transport-1",
                    "agent": "transport",
                    "status": STATUS_COMPLETED,
                    "summary": "路线已计算",
                    "payload": {"route": {"minutes": 30}},
                }
            ],
            "clarification": False,
        }

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.run_orchestrator",
        fake_orchestrator,
    )
    outcome = engine.run_turn(
        ctx,
        None,
        "西湖到灵隐寺怎么走",
        request_id="req_v2",
        task_type=TaskType.ROUTE_QUERY,
    )
    assert outcome.status == STATUS_COMPLETED
    assert outcome.reply == "已安排交通调研。"
    assert outcome.results[0].agent == "transport"
    assert outcome.results[0].payload["route"]["minutes"] == 30
    # 无 planner 产出 → 不渲染、不 Review
    assert outcome.plan_artifact_id is None and outcome.gate_status == "skipped"


def test_v2_clarification_short_circuits(monkeypatch):
    engine, ctx, runner = make_engine(V2_CONFIG)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.run_orchestrator",
        lambda *args, **kwargs: {
            "reply": "请告诉我目的地和天数。",
            "tool_trace": ["request_travel_info"],
            "results": [],
            "clarification": True,
        },
    )
    outcome = engine.run_turn(ctx, None, "帮我规划行程")
    assert outcome.status == STATUS_CLARIFICATION_REQUIRED
    assert "目的地" in outcome.reply


def test_orchestrator_agent_real_react_loop_with_fake_llm(offline_settings):
    """run_orchestrator 用 Scripted LLM 走真实 create_react_agent + dispatch 工具。"""
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
    assert result["reply"] == "已为您调研交通路线。"
    assert result["tool_trace"] == ["dispatch_subagent"]
    assert len(result["results"]) == 1
    assert result["results"][0]["agent"] == "transport"
    assert result["results"][0]["status"] == STATUS_COMPLETED


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
    dispatch.invoke({"agent": "hotel", "instruction": "住哪"})
    assert len(sink) == 1 and sink[0]["agent"] == "hotel"


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
    # plan_artifact_id 更新为重规划产物
    assert outcome.plan_artifact_id == ctx.store.latest_id("itinerary")


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


# --- executor：白名单与预算守卫 ---------------------------------------------------- #


def test_executor_runs_react_loop_with_fake_model(offline_settings):
    ctx = build_session(session_id="sess_exec", persist=False)
    model = ScriptedChatModel(messages=[AIMessage(content="景点调研完成。")])
    executor = build_subagent_executor(offline_settings, model=model)
    result = executor(SUBAGENT_REGISTRY["attraction"], _task_for("attraction"), ctx)
    assert result["status"] == STATUS_COMPLETED
    assert result["summary"] == "景点调研完成。"
    assert result["tool_trace"] == []


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


def test_build_review_callable_returns_none_when_llm_disabled(offline_settings):
    assert offline_settings.llm.enabled is False
    assert build_review_callable(offline_settings) is None
