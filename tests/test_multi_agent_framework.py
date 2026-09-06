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


def test_engine_default_is_production():
    engine = MultiAgentEngine()
    assert engine.is_production
    assert engine.capabilities is PRODUCTION_CONFIG


# --- V1 确定性派工 -------------------------------------------------------------- #


def test_fixed_dispatch_map_covers_all_task_types_without_full_fallback():
    # 每个映射目标都是小集合，绝不允许“未识别任务统一回退 full_itinerary”
    assert resolve_task_batches(TaskType.ROUTE_QUERY) == (("transport",),)
    assert resolve_task_batches(TaskType.DAY_ADVICE) == (("attraction",), ("transport",))
    assert resolve_task_batches(None) is None
    assert resolve_task_batches(TaskType.UNKNOWN) is None
    assert needs_clarification(None)
    assert needs_clarification(TaskType.UNKNOWN)
    full = resolve_task_batches(TaskType.FULL_TRIP_PLAN)
    assert full == (("attraction",), ("transport",), ("planner",))


def test_build_fixed_tasks_expands_batches_with_dependencies():
    tasks = build_fixed_tasks("req_1", TaskType.FULL_TRIP_PLAN, task_brief="杭州两日游")
    assert isinstance(tasks, list) and len(tasks) == 3
    first_batch = [task for task in tasks if not task.depends_on]
    assert {task.agent for task in first_batch} == {"attraction"}
    transport = next(task for task in tasks if task.agent == "transport")
    assert set(transport.depends_on) == {task.task_id for task in first_batch}
    planner = next(task for task in tasks if task.agent == "planner")
    # 累积依赖：planner 等待全部上游批次，从而拿到所有领域 evidence
    assert set(planner.depends_on) == {task.task_id for task in tasks if task.agent != "planner"}
    assert all(task.request_id == "req_1" for task in tasks)


def test_build_fixed_tasks_unknown_type_returns_clarification_sentinel():
    assert build_fixed_tasks("req_1", None) == STATUS_CLARIFICATION_REQUIRED


def test_fixed_dispatch_adds_hotel_and_restaurant_only_when_explicit() -> None:
    tasks = build_fixed_tasks(
        "req_explicit",
        TaskType.FULL_ITINERARY,
        task_brief="规划两日行程，并推荐酒店；全程只吃清真餐厅",
        inputs={"profile": {"constraint_state": {"dietary": ["仅清真餐厅"]}}},
    )
    assert isinstance(tasks, list)
    assert {task.agent for task in tasks} == {
        "attraction", "hotel", "restaurant", "transport", "planner"
    }


def test_fixed_dispatch_respects_self_arranged_and_excluded_domains() -> None:
    tasks = build_fixed_tasks(
        "req_arranged",
        TaskType.FULL_ITINERARY,
        task_brief="酒店已经订好，用餐自行安排，不要推荐餐厅",
        inputs={"profile": {"constraint_state": {"exclude": ["餐厅推荐"]}}},
    )
    assert isinstance(tasks, list)
    assert {task.agent for task in tasks} == {"attraction", "transport", "planner"}


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
        [{
            "issue_type": "budget",
            "severity": "critical",
            "description": "预算冲突",
            "evidence": ["validation_result.issues[0]: budget_hard_limit_exceeded"],
        }],
    )
    status, repair = resolve_delivery_status(
        critical, reviewer_enabled=True, max_rework=1, rework_used=0
    )
    assert status == STATUS_INCOMPLETE and repair is False  # critical 禁止自动交付/修复

    rework = _review(
        "rework",
        [
            {
                "issue_type": "pace",
                "severity": "recoverable",
                    "description": "节奏",
                    "evidence": ["itinerary.day2"],
                    "repair_target": "planner",
                "repair_instruction": "降低行程强度",
            }
        ],
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
            {
                "severity": "recoverable",
                "repair_target": "planner",
                "repair_instruction": "修复 a",
                "issue_type": "a",
                "description": "a issue",
                "evidence": ["itinerary.day2"],
            },
            {
                "severity": "recoverable",
                "repair_target": "transport",
                "repair_instruction": "修复 b",
                "issue_type": "b",
                "description": "b issue",
                "evidence": ["return_deadline=21:30"],
            },
            {
                "severity": "critical",
                "repair_target": "hotel",
                "issue_type": "c",
                "description": "c issue",
                "evidence": ["validation_result.issues[0]: critical"],
            },
        ],
    )
    assert repair_targets(review) == ["transport", "planner"]


def test_reviewer_prompt_contains_context_and_no_tools():
    ctx = ReviewContext(request_id="req_r", plan={"days": [1]}, task_brief="杭州两日")
    text = reviewer_prompt(ctx)
    assert "req_r" in text and "杭州两日" in text
    assert "不拥有任何工具" in text
    assert "不要求把酒店塞进 itinerary.days[].stops" in text
    assert "candidate_attractions 是待核验/比较的候选集合" in text
    assert "不要求每种方式都必须出现在最终路线" in text
    assert "用户自有但未指定具体场所" in text


def test_reviewer_downgrades_unsupported_candidate_omission():
    ctx = ReviewContext(
        request_id="req_candidate",
        plan={"critic": {"passed": True}},
        profile_brief={
            "must_visit": [],
            "constraint_state": {"candidate_attractions": ["甲博物馆"]},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "failed",
            "issues": [{
                "issue_type": "candidate_missing",
                "severity": "critical",
                "description": "候选甲博物馆未安排，可能体验不完整。",
                "evidence": ["缺乏明确证据"],
                "repair_target": "planner",
                "repair_instruction": "加入候选",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"
    assert resolve_delivery_status(
        review, reviewer_enabled=True, max_rework=1, rework_used=0
    ) == ("completed_with_warnings", False)


def test_reviewer_downgrades_explained_closed_candidate_substitution() -> None:
    ctx = ReviewContext(
        request_id="req_candidate_substitution",
        plan={"critic": {"passed": True}, "validation_result": {"passed": True}},
        profile_brief={
            "must_visit": [],
            "constraint_state": {"candidate_attractions": ["甲博物馆"]},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "attraction_substitution",
                "severity": "recoverable",
                "description": "候选甲博物馆因闭馆被省略，替代为开放的乙公园。",
                "evidence": ["candidate_verification.results[0].status='closed'"],
                "repair_target": "planner",
                "repair_instruction": "补充替代说明",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_inferred_weekday_and_preserved_fixed_event_claims():
    ctx = ReviewContext(
        request_id="req_fixed",
        plan={"critic": {"passed": True}},
        profile_brief={
            "constraint_state": {
                "fixed_events": [{"day": 2, "start": "18:00", "end": "20:00", "location": "陆家嘴"}],
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "failed",
            "issues": [
                {
                    "issue_type": "weekday_guess",
                    "severity": "critical",
                    "description": "根据第二天活动推断第一天为周二。",
                    "evidence": ["推断的星期"],
                    "repair_target": "planner",
                    "repair_instruction": "换日",
                },
                {
                    "issue_type": "appointment_missing",
                    "severity": "recoverable",
                    "description": "晚饭预约未安排。",
                    "evidence": ["未包含晚饭预约"],
                    "repair_target": "planner",
                    "repair_instruction": "补预约",
                },
            ],
        },
    )

    assert review.verdict == "pass"
    assert {issue.severity for issue in review.issues} == {"noncritical"}


def test_reviewer_does_not_treat_missing_meal_as_dietary_violation():
    ctx = ReviewContext(
        request_id="req_dietary",
        plan={"critic": {"passed": True}},
        profile_brief={"constraint_state": {"dietary": ["仅清真餐厅"]}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "failed",
            "issues": [{
                "issue_type": "dietary",
                "severity": "critical",
                "description": "第三天未安排任何餐饮，没有任何清真餐厅。",
                "evidence": ["当天没有包含任何餐厅"],
                "repair_target": "planner",
                "repair_instruction": "补餐厅",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_unverified_dietary_guess_when_plan_is_fail_closed():
    ctx = ReviewContext(
        request_id="req_dietary_uncertainty",
        plan={
            "meal_strategy": {
                "dietary_constraints": ["不吃海鲜"],
                "dietary_policy": {
                    "mode": "confirm_or_replace",
                    "requirements": ["不吃海鲜"],
                    "instruction": "下单前确认；无法确认则更换。",
                },
            },
            "critic": {"passed": True},
            "validation_result": {"passed": True},
        },
        profile_brief={"constraint_state": {"dietary": ["不吃海鲜"]}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "dietary_compliance",
                "severity": "recoverable",
                "description": "候选是否包含海鲜未提供任何证据，需要确认。",
                "evidence": ["meal_strategy.dietary_policy.mode=confirm_or_replace"],
                "repair_target": "restaurant",
                "repair_instruction": "替换餐厅",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_self_contradictory_correct_lodging_gap():
    ctx = ReviewContext(
        request_id="req_correct_lodging",
        plan={
            "lodging_plan": {"nights": 1},
            "critic": {"passed": True},
            "validation_result": {"passed": True},
        },
        profile_brief={
            "constraint_state": {
                "date_start": "2026-10-17",
                "date_end": "2026-10-18",
                "lodging_area": "市中心",
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "lodging_plan_gap",
                "severity": "recoverable",
                "description": "两天行程住宿一晚，nights=1是正确的，但需要确认住宿日期。",
                "evidence": ["lodging_plan.nights=1"],
                "repair_target": "planner",
                "repair_instruction": "确认日期",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_soft_gap_and_unrequested_lodging_omission():
    ctx = ReviewContext(
        request_id="req_soft_quality",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True},
        },
        profile_brief={"constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [
                {
                    "issue_type": "pace",
                    "severity": "recoverable",
                    "description": "There is a six-hour unaccounted gap in day 1.",
                    "evidence": ["itinerary.day1"],
                    "repair_target": "planner",
                    "repair_instruction": "Fill the gap.",
                },
                {
                    "issue_type": "lodging",
                    "severity": "recoverable",
                    "description": "No accommodation evidence was provided.",
                    "evidence": ["lodging_plan: evidence_unavailable"],
                    "repair_target": "hotel",
                    "repair_instruction": "Find a hotel.",
                },
            ],
        },
    )

    assert review.verdict == "pass"
    assert {issue.severity for issue in review.issues} == {"noncritical"}


def test_reviewer_keeps_lodging_gap_recoverable_when_area_is_explicit():
    ctx = ReviewContext(
        request_id="req_required_lodging",
        plan={"lodging_plan": {"status": "evidence_unavailable"}},
        profile_brief={"constraint_state": {"lodging_area": "湖畔区"}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "lodging",
                "severity": "recoverable",
                "description": "No accommodation evidence was provided.",
                "evidence": ["lodging_plan: evidence_unavailable"],
                "repair_target": "hotel",
                "repair_instruction": "Find area evidence.",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_downgrades_soft_route_limit_and_budget_risk_band():
    ctx = ReviewContext(
        request_id="req_soft_risk",
        plan={
            "critic": {"passed": True, "issues": [{"code": "route_too_long", "severity": "warning"}]},
            "budget_plan": {
                "expected_total": 5305,
                "user_limit_cny": 6000,
                "within_user_limit": True,
                "risk_high_exceeds_limit": True,
            },
        },
        profile_brief={"pace": "standard", "constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [
                {
                    "issue_type": "schedule_feasibility",
                    "severity": "recoverable",
                    "description": "A route exceeds the standard pace suggested limit.",
                    "evidence": ["critic.issues[0].code=route_too_long"],
                    "repair_target": "planner",
                    "repair_instruction": "Shorten it.",
                },
                {
                    "issue_type": "budget_consistency",
                    "severity": "recoverable",
                    "description": "The high risk band exceeds the limit.",
                    "evidence": ["budget_plan.risk_high_exceeds_limit=true"],
                    "repair_target": "planner",
                    "repair_instruction": "Reduce uncertainty.",
                },
            ],
        },
    )

    assert review.verdict == "pass"
    assert {issue.severity for issue in review.issues} == {"noncritical"}


def test_reviewer_downgrades_implicit_pace_route_issue_with_hard_transport_rules():
    ctx = ReviewContext(
        request_id="req_implicit_pace",
        plan={
            "critic": {
                "passed": True,
                "issues": [{"code": "route_too_long", "severity": "warning"}],
            },
            "validation_result": {"passed": True, "checks_run": []},
        },
        profile_brief={
            "pace": "standard",
            "constraint_state": {
                "public_transport_required": True,
                "max_single_walk_min": 20,
            },
        },
    )

    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "route_duration_exceeds_pace",
                "severity": "recoverable",
                "description": "公共交通通勤耗时62分钟，超过standard节奏建议上限60分钟。",
                "evidence": ["itinerary.days[0].stops[1].route_from_previous.duration_min = 62"],
                "repair_target": "planner",
                "repair_instruction": "调整顺序。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_keeps_explicit_pace_route_issue_recoverable():
    ctx = ReviewContext(
        request_id="req_explicit_pace",
        plan={
            "critic": {
                "passed": True,
                "issues": [{"code": "route_too_long", "severity": "warning"}],
            },
            "validation_result": {"passed": True, "checks_run": []},
        },
        profile_brief={
            "pace": "relaxed",
            "constraint_state": {"pace": "relaxed"},
        },
    )

    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "route_duration_exceeds_pace",
                "severity": "recoverable",
                "description": "通勤耗时56分钟，超过relaxed节奏建议上限45分钟。",
                "evidence": ["itinerary.days[1].stops[1].route_from_previous.duration_min = 56"],
                "repair_target": "planner",
                "repair_instruction": "调整顺序。",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_does_not_treat_default_transport_as_user_taxi_prohibition():
    ctx = ReviewContext(
        request_id="req_default_transport_fallback",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True},
        },
        profile_brief={
            "transport_mode": "public_transport",
            "constraint_state": {"budget_max_cny": 1800},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "transport_mode_conflict",
                "severity": "recoverable",
                "description": (
                    "行程使用 taxi，但 profile.transport_mode=public_transport，"
                    "且没有 taxi_backup，违反用户明确公共交通要求。"
                ),
                "evidence": ["itinerary.days[0]"],
                "repair_target": "planner",
                "repair_instruction": "改用公共交通。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_does_not_invent_explicit_transport_requirement_from_profile_default():
    ctx = ReviewContext(
        request_id="req_default_transport_explicitness",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={
            "transport_mode": "public_transport",
            "constraint_state": {"budget_max_cny": 2400},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "transport_mode_compliance",
                "severity": "recoverable",
                "description": (
                    "行程使用 taxi，但用户明确要求 public_transport，"
                    "且没有 taxi_backup 说明。"
                ),
                "evidence": ["profile.transport_mode=public_transport"],
                "repair_target": "planner",
                "repair_instruction": "全部改用公共交通。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_treats_unrequested_internal_accessibility_as_advisory():
    ctx = ReviewContext(
        request_id="req_bounded_mobility",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "mobility_plan": {
                "required": True,
                "status": "bounded_with_taxi_fallback",
                "max_walking_km_per_day": 6.0,
                "days": [{
                    "day_index": 1,
                    "known_walking_km": 0.0,
                    "unknown_walking_legs": ["上一站→下一站"],
                    "taxi_fallback_required": True,
                }],
            },
        },
        profile_brief={
            "constraint_state": {
                "elderly": True,
                "max_walking_km_per_day": 6.0,
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [
                {
                    "issue_type": "elderly_accessibility",
                    "severity": "recoverable",
                    "description": "景区内部步行距离未知，无法确认是否超过老人每日限制。",
                    "evidence": ["mobility_plan.days[0].known_walking_km=0.0"],
                    "repair_target": "planner",
                    "repair_instruction": "补充内部步行距离。",
                },
                {
                    "issue_type": "route_evidence_gap",
                    "severity": "recoverable",
                    "description": "普通景点间路线证据缺失，仅有直线距离估算。",
                    "evidence": ["itinerary.days[0].stops route_evidence 缺失"],
                    "repair_target": "transport",
                    "repair_instruction": "补充路线证据。",
                },
            ],
        },
    )

    assert review.verdict == "pass"
    assert any(
        issue.issue_type == "elderly_accessibility" and issue.severity == "noncritical"
        for issue in review.issues
    )
    assert any(
        issue.issue_type == "route_evidence_gap" and issue.severity == "noncritical"
        for issue in review.issues
    )


def test_reviewer_accepts_bounded_unknown_walking_legs_with_taxi_fallback() -> None:
    ctx = ReviewContext(
        request_id="req_bounded_walking",
        plan={
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True},
            "mobility_plan": {
                "status": "bounded_with_taxi_fallback",
                "max_walking_km_per_day": 6.0,
                "days": [{
                    "day_index": 1,
                    "known_walking_km": 0.0,
                    "unknown_walking_legs": ["上一站→下一站"],
                    "taxi_fallback_required": True,
                }],
            },
        },
        profile_brief={
            "constraint_state": {"elderly": True, "max_walking_km_per_day": 6.0},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "walking_distance_evidence_gap",
                "severity": "recoverable",
                "description": "每日 known_walking_km 为 0，步行距离未知，无法确认。",
                "evidence": ["mobility_plan.days[0].unknown_walking_legs"],
                "repair_target": "transport",
                "repair_instruction": "补充步行证据",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_keeps_internal_accessibility_gap_for_explicit_stair_avoidance():
    ctx = ReviewContext(
        request_id="req_explicit_stairs",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
        },
        profile_brief={
            "constraint_state": {"elderly": True, "avoid": ["长楼梯"]},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "elderly_accessibility",
                "severity": "recoverable",
                "description": "景区内部台阶信息未知，不能证明符合长楼梯避让要求。",
                "evidence": ["constraint_state.avoid: 长楼梯"],
                "repair_target": "attraction",
                "repair_instruction": "补充无台阶证据或替换候选。",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_keeps_unknown_route_recoverable_for_hard_timed_event():
    ctx = ReviewContext(
        request_id="req_hard_timed_route",
        plan={
            "critic": {"passed": True, "issues": []},
            "required_route_anchors": {
                "fixed_event_transfer": {"evidence_status": "unavailable"},
            },
            "mobility_plan": {
                "required": True,
                "status": "bounded_with_taxi_fallback",
                "max_walking_km_per_day": 6.0,
                "days": [{"day_index": 1, "taxi_fallback_required": True}],
            },
        },
        profile_brief={
            "constraint_state": {
                "max_walking_km_per_day": 6.0,
                "fixed_events": [
                    {"day": 1, "start": "18:00", "end": "20:00", "location": "会场"}
                ],
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "route_evidence_gap",
                "severity": "recoverable",
                "description": "到固定预约会场的路线证据缺失。",
                "evidence": ["required_route_anchors.fixed_event_transfer: unavailable"],
                "repair_target": "transport",
                "repair_instruction": "补充固定预约路线。",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_accepts_bound_internal_routes_when_summary_says_no_route_evidence():
    ctx = ReviewContext(
        request_id="req_bound_internal_routes",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": [
                {"poi": {"poi_id": "a", "name": "甲馆"}},
                {
                    "poi": {"poi_id": "b", "name": "乙馆"},
                    "route_from_previous": {
                        "origin_poi_id": "a",
                        "destination_poi_id": "b",
                        "evidence_status": "provider_verified",
                        "source": "amap",
                    },
                },
            ]}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={
            "constraint_state": {
                "fixed_events": [
                    {"day": 2, "start": "15:00", "end": "17:00", "location": "远郊展馆"}
                ]
            }
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "route_evidence_gap",
                "severity": "recoverable",
                "description": (
                    "Only the fixed_event_transfer is summarized separately; "
                    "there is no route evidence between consecutive stops "
                    "within the same day."
                ),
                "evidence": ["itinerary.days[0].stops"],
                "repair_target": "transport",
                "repair_instruction": "Add route evidence.",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_bounded_lodging_route_unrelated_to_return_deadline():
    ctx = ReviewContext(
        request_id="req_bounded_lodging_route",
        plan={
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
            "required_route_anchors": {
                "last_stop_to_return_location": {
                    "evidence_status": "provider_verified",
                    "recommended_latest_departure": "15:20",
                    "route": {"hard_feasibility_proven": True},
                },
            },
            "mobility_plan": {
                "required": True,
                "status": "bounded_with_taxi_fallback",
                "max_walking_km_per_day": 6.0,
                "days": [{
                    "day_index": 1,
                    "known_walking_km": 0.0,
                    "unknown_walking_legs": ["酒店→首站"],
                    "taxi_fallback_required": True,
                }],
            },
        },
        profile_brief={
            "constraint_state": {
                "elderly": True,
                "max_walking_km_per_day": 6.0,
                "lodging_area": "河畔商圈",
                "return_deadline": "17:00",
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "lodging_route_evidence",
                "severity": "recoverable",
                "description": "酒店到每日首站只有距离估算，公共交通可达性需复核。",
                "evidence": [
                    "route_evidence.lodging_route_anchors.daily_routes[0].evidence_status=haversine_estimate"
                ],
                "repair_target": "planner",
                "repair_instruction": "补充住宿接驳路线。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_schedule_tightness_after_deterministic_validation():
    ctx = ReviewContext(
        request_id="req_schedule_advisory",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={"constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_feasibility",
                "severity": "recoverable",
                "description": "景点衔接较紧，建议增加午餐缓冲。",
                "evidence": ["itinerary.days[0].stops"],
                "repair_target": "planner",
                "repair_instruction": "调整普通游览节奏。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_soft_schedule_rhythm_after_deterministic_validation():
    ctx = ReviewContext(
        request_id="req_schedule_rhythm_advisory",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={
            "constraint_state": {
                "fixed_events": [
                    {"day": 1, "start": "14:00", "end": "16:00", "location": "展馆"}
                ]
            }
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_rhythm",
                "severity": "recoverable",
                "description": "当天仅包含固定活动，上午安排较稀疏，与普通节奏不符。",
                "evidence": ["itinerary.days[0].stops"],
                "repair_target": "planner",
                "repair_instruction": "增加可选活动。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_redundant_area_advisory_after_validation():
    ctx = ReviewContext(
        request_id="req_redundant_area_advisory",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={"constraint_state": {"must_visit": ["历史街区"]}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "constraint_conflict",
                "severity": "recoverable",
                "description": (
                    "同一历史街区被拆成两个相邻游览点，可能造成重复安排，"
                    "但必去地点已经覆盖。"
                ),
                "evidence": ["itinerary.days[0].stops"],
                "repair_target": "planner",
                "repair_instruction": "合并相邻游览点。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_downgrades_nonfixed_schedule_conflict_after_validation():
    ctx = ReviewContext(
        request_id="req_schedule_boundary",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={"constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_conflict",
                "severity": "recoverable",
                "description": "餐段结束时间恰好等于下一活动开始时间，被误判为重叠。",
                "evidence": ["itinerary.days[0].stops"],
                "repair_target": "planner",
                "repair_instruction": "调整普通游览节奏。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_treats_source_backed_budget_estimate_as_advisory_without_contrary_evidence():
    ctx = ReviewContext(
        request_id="req_budget_estimate_advisory",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
            "budget_plan": {
                "tickets": 240.0,
                "expected_total": 1600.0,
                "within_user_limit": None,
                "source_artifact_id": "budget_synthetic",
            },
        },
        profile_brief={"constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "budget_inconsistency",
                "severity": "recoverable",
                "description": "门票估算缺乏依据，景点没有提供实时票价信息。",
                "evidence": ["budget_plan.tickets=240; itinerary.days[0] 没有票价字段"],
                "repair_target": "planner",
                "repair_instruction": "删除门票估算。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_keeps_schedule_conflict_recoverable_with_fixed_event():
    ctx = ReviewContext(
        request_id="req_schedule_fixed_event",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={
            "constraint_state": {
                "fixed_events": [
                    {"day": 1, "start": "14:00", "end": "15:00", "location": "会场"}
                ],
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_conflict",
                "severity": "recoverable",
                "description": "活动与固定预约时间发生冲突。",
                "evidence": ["itinerary.days[0].stops"],
                "repair_target": "planner",
                "repair_instruction": "调整预约前活动。",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_defers_to_verified_return_anchor_for_return_schedule_claim():
    ctx = ReviewContext(
        request_id="req_verified_return_schedule",
        plan={
            "itinerary": {
                "days": [{
                    "day_index": 1,
                    "stops": [{
                        "start_time": "10:00",
                        "duration_min": 90,
                        "poi": {"name": "丝绸博物馆", "category": "museum"},
                    }],
                }],
            },
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
            "required_route_anchors": {
                "last_stop_to_return_location": {
                    "evidence_status": "provider_verified",
                    "recommended_latest_departure": "16:10",
                    "route": {
                        "hard_feasibility_proven": True,
                        "duration_min": 20,
                    },
                },
            },
        },
        profile_brief={
            "constraint_state": {
                "return_location": "中央车站",
                "return_deadline": "17:00",
            },
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_feasibility",
                "severity": "recoverable",
                "description": "酒店到博物馆需要47分钟，因此可能无法在17:00前返回中央车站。",
                "evidence": [
                    "required_route_anchors.last_stop_to_return_location.route.duration_min=20"
                ],
                "repair_target": "planner",
                "repair_instruction": "删除博物馆。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_defers_to_verified_return_anchor_for_route_feasibility_wording():
    ctx = ReviewContext(
        request_id="req_verified_return_route_feasibility",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
            "required_route_anchors": {
                "last_stop_to_return_location": {
                    "evidence_status": "provider_verified",
                    "recommended_latest_departure": "2027-06-01T17:36+08:00",
                    "route": {"hard_feasibility_proven": True, "duration_min": 54},
                },
            },
        },
        profile_brief={
            "constraint_state": {
                "return_location": "南站",
                "return_deadline": "2027-06-01T19:00+08:00",
            }
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "return_route_feasibility",
                "severity": "recoverable",
                "description": "16:30离开可在截止前到站，但仍需确认96分钟缓冲是否足够。",
                "evidence": [
                    "required_route_anchors.last_stop_to_return_location.recommended_latest_departure="
                    "2027-06-01T17:36+08:00"
                ],
                "repair_target": "planner",
                "repair_instruction": "再次确认返程缓冲。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_accepts_taxi_when_explicit_transport_set_and_backup_allow_it():
    ctx = ReviewContext(
        request_id="req_explicit_taxi_backup",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={
            "constraint_state": {
                "public_transport_required": True,
                "transport_modes": ["public_transport", "taxi"],
                "taxi_backup": True,
            }
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "transport_mode_conflict",
                "severity": "recoverable",
                "description": "两段使用 taxi，缺少对应公共交通路线证据。",
                "evidence": [
                    "active_constraints.transport_modes=['public_transport','taxi']; "
                    "active_constraints.taxi_backup=true"
                ],
                "repair_target": "transport",
                "repair_instruction": "全部替换为公共交通。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_keeps_taxi_conflict_when_public_transport_is_the_only_allowed_mode():
    ctx = ReviewContext(
        request_id="req_public_transport_only",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
        },
        profile_brief={
            "constraint_state": {
                "public_transport_required": True,
                "transport_modes": ["public_transport"],
                "taxi_backup": False,
            }
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "transport_mode_conflict",
                "severity": "recoverable",
                "description": "路线使用 taxi，违反仅公共交通的明确约束。",
                "evidence": ["active_constraints.transport_modes=['public_transport']"],
                "repair_target": "transport",
                "repair_instruction": "改为公共交通。",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_treats_missing_hours_on_verified_optional_poi_as_advisory():
    ctx = ReviewContext(
        request_id="req_verified_optional_missing_hours",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": [{
                "poi": {
                    "poi_id": "verified-optional",
                    "name": "城市旧址",
                    "source": "amap",
                    "verification_status": "verified",
                    "entity_type": "attraction",
                }
            }]}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {
                "passed": True,
                "issues": [],
                "checks_run": ["applicable_opening_hours"],
            },
        },
        profile_brief={"constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "attraction_quality_concern",
                "severity": "recoverable",
                "description": "该候选缺少 opening_hours，需核实游览价值和开放状态。",
                "evidence": ["itinerary.days[0].stops[0].poi.opening_hours is missing"],
                "repair_target": "attraction",
                "repair_instruction": "替换景点。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_keeps_known_closure_conflict_recoverable():
    ctx = ReviewContext(
        request_id="req_known_closure",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": []}]},
            "critic": {"passed": False, "issues": [{"code": "poi_closed"}]},
            "validation_result": {
                "passed": False,
                "issues": [{"code": "applicable_opening_hours"}],
                "checks_run": ["applicable_opening_hours"],
            },
        },
        profile_brief={"constraint_state": {}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "attraction_quality_concern",
                "severity": "recoverable",
                "description": "该场馆在行程日期明确闭馆，开放状态冲突。",
                "evidence": ["validation_result.issues[0].code=applicable_opening_hours"],
                "repair_target": "attraction",
                "repair_instruction": "替换为开放场馆。",
            }],
        },
    )

    assert review.verdict == "rework"
    assert review.issues[0].severity == "recoverable"


def test_reviewer_defers_to_validated_route_bindings_for_schedule_gap_claim() -> None:
    ctx = ReviewContext(
        request_id="req_validated_route_bindings",
        plan={
            "itinerary": {"days": [{"day_index": 1, "stops": [
                {
                    "start_time": "09:30", "duration_min": 120,
                    "poi": {"poi_id": "first", "name": "第一站", "category": "scenic"},
                },
                {
                    "start_time": "14:30", "duration_min": 120,
                    "poi": {"poi_id": "final", "name": "最终站", "category": "museum"},
                    "route_from_previous": {
                        "origin_poi_id": "first", "destination_poi_id": "final",
                        "duration_min": 25, "evidence_status": "provider_verified",
                    },
                },
            ]}]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
            "required_route_anchors": {
                "trip_origin_to_first_stop": {
                    "evidence_status": "provider_verified",
                    "origin_poi_id": "station", "destination_poi_id": "first",
                    "route": {"origin_poi_id": "station", "destination_poi_id": "first"},
                },
                "last_stop_to_return_location": {
                    "evidence_status": "provider_verified",
                    "origin_poi_id": "final", "destination_poi_id": "station",
                    "recommended_latest_departure": "17:30",
                    "route": {"hard_feasibility_proven": True, "duration_min": 30},
                },
            },
        },
        profile_brief={"constraint_state": {
            "origin": "测试城南站",
            "return_location": "测试城南站",
            "return_deadline": "19:00",
        }},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_gap",
                "severity": "recoverable",
                "description": "缺少从车站到第一站、第一站到最终站，以及最终站返回车站的路线证据。",
                "evidence": ["itinerary.days[0]"],
                "repair_target": "planner",
                "repair_instruction": "补充路线。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_scopes_hard_timing_to_the_day_named_by_schedule_issue() -> None:
    ctx = ReviewContext(
        request_id="req_cross_day_schedule_gap",
        plan={
            "itinerary": {"days": [
                {"day_index": 1, "stops": [{
                    "start_time": "09:00", "duration_min": 90,
                    "poi": {"name": "第一天景点", "category": "scenic"},
                }]},
                {"day_index": 2, "stops": [{
                    "start_time": "14:30", "duration_min": 150,
                    "poi": {"name": "第二天景点", "category": "museum"},
                }]},
            ]},
            "critic": {"passed": True, "issues": []},
            "validation_result": {"passed": True, "issues": []},
            "fixed_event_plan": {"events": [{
                "day": 1,
                "start": "18:00",
                "location": "第一天会场",
                "route_evidence_status": "provider_verified",
            }]},
        },
        profile_brief={"constraint_state": {"fixed_events": [{
            "day": 1, "start": "18:00", "location": "第一天会场",
        }]}},
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "schedule_gap",
                "severity": "recoverable",
                "description": "第2天下午行程结束较早，可以增加晚间活动。",
                "evidence": ["itinerary.days[1].stops[0]"],
                "repair_target": "planner",
                "repair_instruction": "增加第二天晚间活动。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


def test_reviewer_does_not_turn_uncovered_soft_preference_into_rework():
    ctx = ReviewContext(
        request_id="req_soft_preference",
        plan={"critic": {"passed": True}},
        profile_brief={
            "interests": ["咖啡店"],
            "must_visit": [],
            "constraint_state": {"interests": ["咖啡店"]},
        },
    )
    review = run_semantic_review(
        ctx,
        review_callable=lambda _ctx: {
            "verdict": "rework",
            "issues": [{
                "issue_type": "soft_preference_mismatch",
                "severity": "recoverable",
                "description": "用户偏好咖啡店，但行程未覆盖该偏好。",
                "evidence": ["profile.interests: 咖啡店"],
                "repair_target": "planner",
                "repair_instruction": "补充咖啡店。",
            }],
        },
    )

    assert review.verdict == "pass"
    assert review.issues[0].severity == "noncritical"


# --- Reviewer 触发判定 ----------------------------------------------------------- #


def test_requires_semantic_review_only_for_full_plan_tasks():
    assert requires_semantic_review(FULL_CONFIG, TaskType.FULL_TRIP_PLAN)
    assert not requires_semantic_review(FULL_CONFIG, TaskType.ITINERARY_REVISION)
    assert not requires_semantic_review(FULL_CONFIG, TaskType.ROUTE_QUERY)
    assert not requires_semantic_review(FULL_CONFIG, TaskType.POI_ADVICE)
    assert not requires_semantic_review(V2_CONFIG, TaskType.FULL_TRIP_PLAN)
    assert not requires_semantic_review(FULL_CONFIG, None)
