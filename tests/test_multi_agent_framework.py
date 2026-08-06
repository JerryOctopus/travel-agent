"""multi_agent 基础框架 Step 1 单元测试。

覆盖验收标准：Registry 只含 5 个执行型 Subagent、Reviewer 不在 Registry、
Schema 字段完整、V1 固定派工映射、FULL/V3/PRODUCTION 配置同一性、
Runner Mock 结构化返回与异常包装、Orchestrator 工具面约束、
Reviewer 修复周期判定。
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent import (
    FULL_CONFIG,
    PRODUCTION_CONFIG,
    SUBAGENT_REGISTRY,
    V0_CONFIG,
    V1_CONFIG,
    V2_CONFIG,
    V3_CONFIG,
    MultiAgentEngine,
    SubagentRunner,
    SubagentTask,
    capabilities_for_variant,
    new_request_id,
    new_task_id,
    requires_semantic_review,
)
from travel_agent.orchestration.multi_agent.dispatch_rules import (
    TASK_TYPE_SUBAGENT_MAP,
    build_fixed_tasks,
    needs_clarification,
    resolve_task_batches,
)
from travel_agent.orchestration.multi_agent.orchestrator import (
    BUSINESS_HEAVY_TOOLS,
    ORCHESTRATOR_ALLOWED_TOOLS,
    assert_orchestrator_toolset,
    build_dispatch_tool,
)
from travel_agent.orchestration.multi_agent.review import (
    ReviewContext,
    repair_targets,
    resolve_delivery_status,
    reviewer_prompt,
    run_semantic_review,
)
from travel_agent.orchestration.multi_agent.runner import assert_no_dispatch_leak
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    ReviewResult,
)

# --- Registry --------------------------------------------------------------- #


def test_registry_contains_exactly_five_executing_subagents():
    assert set(SUBAGENT_REGISTRY) == {"attraction", "hotel", "restaurant", "transport", "planner"}


def test_reviewer_not_in_registry():
    assert "reviewer" not in SUBAGENT_REGISTRY
    assert not any("review" in name for name in SUBAGENT_REGISTRY)


def test_subagent_whitelists_never_contain_dispatch_or_render_tools():
    for definition in SUBAGENT_REGISTRY.values():
        assert "dispatch_subagent" not in definition.tool_names
        assert_no_dispatch_leak(definition.tool_names)
        # 每个定义都有独立 prompt 与预算约束
        assert definition.system_prompt
        assert definition.max_steps > 0
        assert definition.max_tool_calls > 0
        assert definition.timeout_seconds > 0


def test_planner_is_the_only_plan_and_critique_holder():
    holders = [
        name
        for name, definition in SUBAGENT_REGISTRY.items()
        if "plan_and_critique" in definition.tool_names
    ]
    assert holders == ["planner"]


# --- Schema 字段完整性 -------------------------------------------------------- #


def test_subagent_task_schema_fields():
    names = {f.name for f in dataclasses.fields(SubagentTask)}
    assert {
        "request_id",
        "task_id",
        "agent",
        "instruction",
        "inputs",
        "constraints",
        "depends_on",
        "attempt",
    } <= names


def test_subagent_result_schema_fields():
    from travel_agent.orchestration.multi_agent.schemas import SubagentResult

    names = {f.name for f in dataclasses.fields(SubagentResult)}
    assert {
        "request_id",
        "task_id",
        "agent",
        "status",
        "attempt",
        "summary",
        "payload",
        "evidence",
        "constraints_used",
        "warnings",
        "unresolved",
        "tool_trace",
        "token_usage",
        "duration_ms",
        "error",
    } <= names


def test_ids_are_unique_and_prefixed():
    rid = new_request_id()
    tid = new_task_id("attraction")
    assert rid.startswith("req_")
    assert tid.startswith("attraction-")
    assert new_task_id("attraction") != tid


# --- 配置别名与预设 ------------------------------------------------------------- #


def test_full_v3_production_are_the_same_object():
    assert V3_CONFIG is FULL_CONFIG
    assert PRODUCTION_CONFIG is FULL_CONFIG
    assert FULL_CONFIG.reviewer_enabled is True
    assert FULL_CONFIG.max_rework == 1
    assert FULL_CONFIG.dispatch == "dynamic"
    assert FULL_CONFIG.mode == "orchestrated"


def test_variant_presets_degrade_from_same_engine_model():
    assert capabilities_for_variant("v0") is V0_CONFIG
    assert capabilities_for_variant("v1") is V1_CONFIG
    assert capabilities_for_variant("v2") is V2_CONFIG
    assert capabilities_for_variant("v3") is FULL_CONFIG
    # V0 单 Agent：无派工、无 Reviewer
    assert V0_CONFIG.mode == "single" and V0_CONFIG.dispatch == "none"
    # V1 规则派工、V2 动态派工，均无 Reviewer
    assert V1_CONFIG.dispatch == "fixed" and not V1_CONFIG.reviewer_enabled
    assert V2_CONFIG.dispatch == "dynamic" and not V2_CONFIG.reviewer_enabled
    with pytest.raises(ValueError):
        capabilities_for_variant("v9")


def test_engine_default_is_production_and_run_not_wired_yet():
    engine = MultiAgentEngine()
    assert engine.is_production
    with pytest.raises(NotImplementedError):
        engine.run()


# --- V1 确定性派工 -------------------------------------------------------------- #


def test_fixed_dispatch_map_covers_all_task_types_without_full_fallback():
    # 每个映射目标都是小集合，绝不允许“未识别任务统一回退 full_itinerary”
    assert resolve_task_batches(TaskType.ROUTE_QUERY) == (("transport",),)
    assert resolve_task_batches(TaskType.DAY_ADVICE) == (("attraction",),)
    assert resolve_task_batches(None) is None
    assert needs_clarification(None)
    full = resolve_task_batches(TaskType.FULL_TRIP_PLAN)
    assert full == (("attraction", "hotel", "restaurant"), ("transport",), ("planner",))


def test_build_fixed_tasks_expands_batches_with_dependencies():
    tasks = build_fixed_tasks("req_1", TaskType.FULL_TRIP_PLAN, task_brief="杭州两日游")
    assert isinstance(tasks, list) and len(tasks) == 5
    first_batch = [task for task in tasks if not task.depends_on]
    assert {task.agent for task in first_batch} == {"attraction", "hotel", "restaurant"}
    transport = next(task for task in tasks if task.agent == "transport")
    assert set(transport.depends_on) == {task.task_id for task in first_batch}
    planner = next(task for task in tasks if task.agent == "planner")
    assert planner.depends_on == [transport.task_id]
    assert all(task.request_id == "req_1" for task in tasks)


def test_build_fixed_tasks_unknown_type_returns_clarification_sentinel():
    assert build_fixed_tasks("req_1", None) == STATUS_CLARIFICATION_REQUIRED


def test_dispatch_map_only_uses_registered_subagents():
    for batches in TASK_TYPE_SUBAGENT_MAP.values():
        for batch in batches:
            for agent in batch:
                assert agent in SUBAGENT_REGISTRY


# --- Runner ------------------------------------------------------------------ #


def _task(agent: str = "attraction") -> SubagentTask:
    return SubagentTask(
        request_id="req_t",
        task_id=new_task_id(agent),
        agent=agent,
        instruction="调研景点",
    )


def test_runner_mock_executor_returns_structured_result():
    def executor(definition, task, ctx):
        return {
            "summary": "找到 5 个景点",
            "payload": {"candidates": [{"poi_id": "p1"}]},
            "evidence": [{"artifact_id": "art_1", "data_source": "amap"}],
            "tool_trace": ["search_poi"],
            "token_usage": {"total_tokens": 120},
        }

    result = SubagentRunner(ctx=None, executor=executor).run_subagent(_task())
    assert result.status == STATUS_COMPLETED
    assert result.agent == "attraction"
    assert result.request_id == "req_t"
    assert result.payload["candidates"][0]["poi_id"] == "p1"
    assert result.evidence[0]["artifact_id"] == "art_1"
    assert result.duration_ms >= 0
    assert result.error is None


def test_runner_wraps_exceptions_as_failed_without_raising():
    def executor(definition, task, ctx):
        raise RuntimeError("boom")

    result = SubagentRunner(ctx=None, executor=executor).run_subagent(_task())
    assert result.status == STATUS_FAILED
    assert "RuntimeError" in (result.error or "")


def test_runner_unknown_agent_and_missing_executor_return_failed():
    result = SubagentRunner(ctx=None, executor=lambda *args: {}).run_subagent(_task("ghost"))
    assert result.status == STATUS_FAILED and "unknown subagent" in (result.error or "")

    result = SubagentRunner(ctx=None).run_subagent(_task())
    assert result.status == STATUS_FAILED and "executor" in (result.error or "")


# --- Orchestrator 工具面与 dispatch 工具 ---------------------------------------- #


def test_orchestrator_toolset_excludes_business_heavy_tools():
    assert_orchestrator_toolset(ORCHESTRATOR_ALLOWED_TOOLS)  # 允许清单自身合法
    assert not (ORCHESTRATOR_ALLOWED_TOOLS & BUSINESS_HEAVY_TOOLS)
    with pytest.raises(ValueError):
        assert_orchestrator_toolset({"dispatch_subagent", "search_poi"})
    with pytest.raises(ValueError):
        assert_orchestrator_toolset({"some_random_tool"})


def test_dispatch_tool_returns_structured_json_result():
    def executor(definition, task, ctx):
        return {"summary": "ok", "payload": {"route": {"minutes": 30}}}

    runner = SubagentRunner(ctx=None, executor=executor)
    dispatch = build_dispatch_tool(runner, request_id="req_d")
    raw = dispatch("transport", "西湖到灵隐寺怎么走", {"from": "西湖"}, [])
    data = json.loads(raw)
    assert data["status"] == STATUS_COMPLETED
    assert data["request_id"] == "req_d"
    assert data["task_id"].startswith("transport-")
    assert data["payload"]["route"]["minutes"] == 30
    # 失败同样以结构化结果回流，不上抛
    def failing(definition, task, ctx):
        raise KeyError("missing")

    dispatch_failed = build_dispatch_tool(SubagentRunner(ctx=None, executor=failing))
    data = json.loads(dispatch_failed("hotel", "住哪", None, None))
    assert data["status"] == STATUS_FAILED


# --- Reviewer 判定 -------------------------------------------------------------- #


def _review(verdict: str, issues: list[dict]) -> ReviewResult:
    return run_semantic_review(
        ReviewContext(request_id="req_r", plan={"days": []}),
        review_callable=lambda ctx: {"verdict": verdict, "issues": issues},
    )


def test_review_parses_structured_issues():
    review = _review(
        "rework",
        [
            {
                "issue_type": "pace",
                "severity": "recoverable",
                "description": "第二天节奏过紧",
                "evidence": ["day2 含 5 个景点"],
                "repair_target": "planner",
                "repair_instruction": "削减 day2 至 3 个景点",
            }
        ],
    )
    assert review.verdict == "rework"
    assert review.issues[0].repair_target == "planner"
    assert review.error is None


def test_review_failure_wrapped_not_raised():
    def broken(ctx):
        raise ValueError("bad")

    review = run_semantic_review(
        ReviewContext(request_id="req_r", plan={}), review_callable=broken
    )
    assert review.verdict == "failed" and "ValueError" in (review.error or "")

    review = run_semantic_review(ReviewContext(request_id="req_r", plan={}))
    assert review.verdict == "failed" and review.error


def test_delivery_status_rules_by_severity():
    no_reviewer = ReviewResult(verdict="pass")
    assert resolve_delivery_status(
        no_reviewer, reviewer_enabled=False, max_rework=1, rework_used=0
    ) == (STATUS_COMPLETED, False)

    critical = _review(
        "failed",
        [{"issue_type": "budget", "severity": "critical", "description": "预算冲突"}],
    )
    status, repair = resolve_delivery_status(
        critical, reviewer_enabled=True, max_rework=1, rework_used=0
    )
    assert status == STATUS_INCOMPLETE and repair is False  # critical 禁止自动交付/修复

    rework = _review(
        "rework",
        [{"issue_type": "pace", "severity": "recoverable", "description": "节奏"}],
    )
    status, repair = resolve_delivery_status(
        rework, reviewer_enabled=True, max_rework=1, rework_used=0
    )
    assert status == STATUS_INCOMPLETE and repair is True  # 触发唯一一次修复周期
    status, repair = resolve_delivery_status(
        rework, reviewer_enabled=True, max_rework=1, rework_used=1
    )
    assert repair is False  # 已用完修复预算

    minor = _review(
        "pass",
        [{"issue_type": "style", "severity": "noncritical", "description": "措辞"}],
    )
    status, repair = resolve_delivery_status(
        minor, reviewer_enabled=True, max_rework=1, rework_used=0
    )
    assert status == STATUS_COMPLETED_WITH_WARNINGS and repair is False


def test_repair_targets_ordered_with_planner_last():
    review = _review(
        "rework",
        [
            {"severity": "recoverable", "repair_target": "planner", "issue_type": "a"},
            {"severity": "recoverable", "repair_target": "transport", "issue_type": "b"},
            {"severity": "critical", "repair_target": "hotel", "issue_type": "c"},
        ],
    )
    assert repair_targets(review) == ["transport", "planner"]


def test_reviewer_prompt_contains_context_and_no_tools():
    ctx = ReviewContext(request_id="req_r", plan={"days": [1]}, task_brief="杭州两日")
    text = reviewer_prompt(ctx)
    assert "req_r" in text and "杭州两日" in text
    assert "不拥有任何工具" in text


# --- Reviewer 触发判定 ----------------------------------------------------------- #


def test_requires_semantic_review_only_for_full_plan_tasks():
    assert requires_semantic_review(FULL_CONFIG, TaskType.FULL_TRIP_PLAN)
    assert requires_semantic_review(FULL_CONFIG, TaskType.ITINERARY_REVISION)
    assert not requires_semantic_review(FULL_CONFIG, TaskType.ROUTE_QUERY)
    assert not requires_semantic_review(FULL_CONFIG, TaskType.POI_ADVICE)
    assert not requires_semantic_review(V2_CONFIG, TaskType.FULL_TRIP_PLAN)
    assert not requires_semantic_review(FULL_CONFIG, None)
