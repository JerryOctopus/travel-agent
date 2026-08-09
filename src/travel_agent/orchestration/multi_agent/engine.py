"""Multi-Agent Engine：统一引擎（Step 3 实现）。

核心约定（严格执行）：

- **Production = Full = V3**：三个名字是**同一配置对象的别名**；
- V0–V3 共用同一个 Engine，由 ``EngineCapabilities`` 降级产生：

  - V0：mode=single（复用 runtime._run_react 单 Agent 全工具链路，薄适配）；
  - V1：mode=orchestrated + dispatch=fixed（task_type 查表派工，无 LLM 路由）；
  - V2：mode=orchestrated + dispatch=dynamic（LLM Orchestrator 动态派工）；
  - V3/Full/Production：V2 + Semantic Reviewer + 最多一次修复周期。

- 生产入口固定使用 ``PRODUCTION_CONFIG``，**不读取** settings 里的 variant；
  评测入口才允许显式选择 ``VARIANT_PRESETS`` 中的消融配置；
- Reviewer 启停只由 ``EngineCapabilities.reviewer_enabled`` 决定，且仅对
  "生成/修改完整 TravelPlan"的任务生效；Reviewer 不属于 SubagentRegistry、
  不经过 SubagentRunner，是 Engine 直调的一次无工具 LLM 调用；
- 修复周期最多一次：定向重派领域 Subagent（可跳过）→ Planner 重规划 →
  不再二轮 Review；整个周期只计一次 rework；
- 渲染统一经 Renderer Gate（render_gate.render_plan_outcome）；
- 可观测性统一为 agent_trace：每轮执行在回合边界一次性落 artifact。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    EngineCapabilities,
    ReviewResult,
    SubagentResult,
    SubagentTask,
    new_request_id,
    new_task_id,
)
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog

# --- V0–V3 能力预设 --------------------------------------------------------- #

V0_CONFIG = EngineCapabilities(
    mode="single",
    dispatch="none",
    reviewer_enabled=False,
    max_rework=0,
)

V1_CONFIG = EngineCapabilities(
    mode="orchestrated",
    dispatch="fixed",
    reviewer_enabled=False,
    max_rework=0,
)

V2_CONFIG = EngineCapabilities(
    mode="orchestrated",
    dispatch="dynamic",
    reviewer_enabled=False,
    max_rework=0,
)

FULL_CONFIG = EngineCapabilities(
    mode="orchestrated",
    dispatch="dynamic",
    reviewer_enabled=True,
    max_rework=1,
)

# 对象别名（不是拷贝、不是 is 表达式赋值）：Production = Full = V3。
V3_CONFIG = FULL_CONFIG
PRODUCTION_CONFIG = FULL_CONFIG

# 评测入口显式选择消融配置用的映射（生产入口不得使用）。
VARIANT_PRESETS: dict[str, EngineCapabilities] = {
    "v0": V0_CONFIG,
    "v1": V1_CONFIG,
    "v2": V2_CONFIG,
    "v3": V3_CONFIG,
}

# 只有生成/修改完整 TravelPlan 的任务才触发 Semantic Reviewer（V3）。
REVIEW_REQUIRED_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.FULL_TRIP_PLAN, TaskType.ITINERARY_REVISION}
)


def capabilities_for_variant(variant: str) -> EngineCapabilities:
    """评测入口专用：variant 名 → 能力预设。非法名字直接报错，不做隐式回退。"""
    key = variant.strip().lower()
    if key not in VARIANT_PRESETS:
        raise ValueError(f"unknown variant: {variant!r} (allowed: {sorted(VARIANT_PRESETS)})")
    return VARIANT_PRESETS[key]


def requires_semantic_review(capabilities: EngineCapabilities, task_type: TaskType | None) -> bool:
    """Engine 判定是否执行 Reviewer：能力开启 且 任务会生成/修改完整 TravelPlan。"""
    if not capabilities.reviewer_enabled:
        return False
    return task_type in REVIEW_REQUIRED_TASK_TYPES


@dataclass
class TurnOutcome:
    """一轮执行的最终交付状态。"""

    status: str
    reply: str = ""
    plan_artifact_id: str | None = None
    rework_used: int = 0
    results: list[Any] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    map_payload: dict[str, Any] | None = None
    gate_status: str = "skipped"
    review: ReviewResult | None = None
    trace_artifact_id: str | None = None


class MultiAgentEngine:
    """统一 Engine：V0–V3 与生产链路共用。

    协作者均可注入（测试用）：``runner``（含 executor）、``review_callable``、
    ``orchestrator_model`` / ``subagent_model``（Fake LLM）。
    """

    def __init__(
        self,
        capabilities: EngineCapabilities = PRODUCTION_CONFIG,
        *,
        runner: Any | None = None,
        review_callable: Any | None = None,
        orchestrator_model: Any | None = None,
        subagent_model: Any | None = None,
    ) -> None:
        self.capabilities = capabilities
        self._runner = runner
        self._review_callable = review_callable
        self._orchestrator_model = orchestrator_model
        self._subagent_model = subagent_model

    @property
    def is_production(self) -> bool:
        """是否为生产 Full 配置（对象别名同一性判定）。"""
        return self.capabilities is PRODUCTION_CONFIG

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def run_turn(
        self,
        ctx: Any,
        settings: Any,
        user_message: str,
        *,
        task_type: TaskType | None = None,
        task_brief: str = "",
        request_id: str | None = None,
        history: list[tuple[str, str]] | None = None,
        existing_plan_artifact_id: str | None = None,
        turn_inputs: dict[str, Any] | None = None,
    ) -> TurnOutcome:
        rid = request_id or new_request_id()
        from travel_agent.orchestration.multi_agent.trace import (
            AgentTraceLog,
            reset_current_trace,
            set_current_trace,
        )

        trace = AgentTraceLog(rid)
        token = set_current_trace(trace)
        try:
            return self._run_turn(
                ctx,
                settings,
                user_message,
                task_type=task_type,
                task_brief=task_brief,
                request_id=rid,
                history=history,
                existing_plan_artifact_id=existing_plan_artifact_id,
                turn_inputs=turn_inputs,
            )
        except Exception as exc:
            trace.append(
                "orchestration",
                agent="engine",
                status=STATUS_FAILED,
                error=f"{type(exc).__name__}: {exc}",
                detail={
                    "mode": self.capabilities.mode,
                    "dispatch": self.capabilities.dispatch,
                },
            )
            store = getattr(ctx, "store", None)
            if store is not None:
                try:
                    trace.flush_to_store(store)
                except Exception:
                    pass
            raise
        finally:
            reset_current_trace(token)

    def _run_turn(
        self,
        ctx: Any,
        settings: Any,
        user_message: str,
        *,
        task_type: TaskType | None = None,
        task_brief: str = "",
        request_id: str | None = None,
        history: list[tuple[str, str]] | None = None,
        existing_plan_artifact_id: str | None = None,
        turn_inputs: dict[str, Any] | None = None,
    ) -> TurnOutcome:
        rid = request_id or new_request_id()
        if self.capabilities.mode == "single":
            outcome = self._run_v0(ctx, settings, user_message, history or [], rid)
        else:
            runner = self._resolve_runner(ctx, settings)
            if self.capabilities.dispatch == "fixed":
                outcome = self._run_fixed(
                    ctx,
                    runner,
                    rid,
                    task_type,
                    task_brief or user_message,
                    existing_plan_artifact_id,
                    turn_inputs or {},
                )
            else:
                outcome = self._run_dynamic(
                    ctx,
                    settings,
                    runner,
                    rid,
                    user_message,
                    task_brief,
                    task_type,
                    history or [],
                    existing_plan_artifact_id,
                    turn_inputs or {},
                )

        if outcome.status == STATUS_CLARIFICATION_REQUIRED:
            self._flush_trace(ctx, rid, outcome)
            return outcome

        if self.capabilities.mode != "single":
            outcome.plan_artifact_id = _plan_artifact_id_from_results(
                ctx,
                outcome.results,
                rid,
            )
        plan_required = _plan_required_for_turn(self.capabilities.dispatch, task_type)
        if plan_required and not outcome.plan_artifact_id:
            outcome.status = STATUS_INCOMPLETE
            if not outcome.reply:
                outcome.reply = "Planner 未成功产出本轮 TravelPlan，当前结果不可交付。"
        elif outcome.plan_artifact_id:
            self._apply_review_and_render(
                ctx, settings, outcome, task_type, task_brief or user_message, rid
            )
        self._flush_trace(ctx, rid, outcome)
        return outcome

    def _flush_trace(self, ctx: Any, request_id: str, outcome: TurnOutcome) -> None:
        """回合边界一次性落 agent_trace（含派工/Review/渲染结论）。"""
        store = getattr(ctx, "store", None)
        if store is None:
            return
        from travel_agent.orchestration.multi_agent.trace import current_trace

        trace = current_trace() or AgentTraceLog(request_id)
        trace.append(
            "orchestration",
            agent="engine",
            status=outcome.status,
            detail={
                "mode": self.capabilities.mode,
                "dispatch": self.capabilities.dispatch,
                "rework_used": outcome.rework_used,
                "plan_artifact_id": outcome.plan_artifact_id,
            },
        )
        if outcome.plan_artifact_id:
            trace.append(
                "render",
                agent="engine",
                status=outcome.gate_status,
                detail={"plan_artifact_id": outcome.plan_artifact_id},
            )
        try:
            outcome.trace_artifact_id = trace.flush_to_store(store)
        except Exception:  # noqa: BLE001 — 轨迹落盘失败不影响交付
            outcome.trace_artifact_id = None

    # ------------------------------------------------------------------ #
    # V0：单 Agent 全工具（薄适配既有 runtime._run_react）
    # ------------------------------------------------------------------ #
    def _run_v0(
        self,
        ctx: Any,
        settings: Any,
        user_message: str,
        history: list[tuple[str, str]],
        request_id: str,
    ) -> TurnOutcome:
        from travel_agent.agent.runtime import _run_react
        from travel_agent.agent.session import reset_task_meta, set_current_task_meta

        task_id = new_task_id("single_agent")
        token = set_current_task_meta(
            {"request_id": request_id, "task_id": task_id, "agent": "single_agent"}
        )
        try:
            reply = _run_react(user_message, ctx, history, settings)
        except Exception as exc:  # noqa: BLE001
            return TurnOutcome(status=STATUS_FAILED, reply=f"执行失败：{exc}")
        finally:
            reset_task_meta(token)
        if getattr(reply, "clarification", False):
            return TurnOutcome(status=STATUS_CLARIFICATION_REQUIRED, reply=reply.text)
        plan_id = next(
            (
                artifact_id
                for artifact_id in reversed(
                    ctx.store.artifact_ids_for_task(request_id, task_id, agent="single_agent")
                )
                if (ctx.store.get_record(artifact_id) or {}).get("kind") == "itinerary"
            ),
            None,
        )
        critic_passed = bool(
            ((ctx.store.get(plan_id) if plan_id else {}) or {}).get("critic", {}).get("passed")
            is True
        )
        status = (
            STATUS_COMPLETED
            if critic_passed or plan_id is None
            else STATUS_INCOMPLETE
        )
        return TurnOutcome(
            status=status,
            reply=reply.text,
            plan_artifact_id=plan_id,
        )

    # ------------------------------------------------------------------ #
    # V1：规则派工（无 LLM 路由）
    # ------------------------------------------------------------------ #
    def _run_fixed(
        self,
        ctx: Any,
        runner: Any,
        request_id: str,
        task_type: TaskType | None,
        task_brief: str,
        existing_plan_artifact_id: str | None,
        turn_inputs: dict[str, Any],
    ) -> TurnOutcome:
        from travel_agent.orchestration.multi_agent.fixed_dispatch import run_fixed_dispatch

        if task_type == TaskType.ITINERARY_REVISION and not existing_plan_artifact_id:
            return TurnOutcome(
                status=STATUS_CLARIFICATION_REQUIRED,
                reply="当前会话没有可修改的既有行程，请先生成或提供一份行程。",
            )
        status, results = run_fixed_dispatch(
            runner,
            request_id,
            task_type,
            task_brief=task_brief,
            inputs=_fixed_turn_inputs(
                task_type, existing_plan_artifact_id, turn_inputs
            ),
        )
        if status == STATUS_CLARIFICATION_REQUIRED:
            return TurnOutcome(
                status=STATUS_CLARIFICATION_REQUIRED,
                reply="任务类型不明或信息不足，需要先补充目的地/天数等关键信息。",
            )
        delivery = _fixed_delivery_status(results)
        return TurnOutcome(status=delivery, reply=_compose_fixed_reply(results), results=results)

    # ------------------------------------------------------------------ #
    # V2/V3：动态 Orchestrator
    # ------------------------------------------------------------------ #
    def _run_dynamic(
        self,
        ctx: Any,
        settings: Any,
        runner: Any,
        request_id: str,
        user_message: str,
        task_brief: str,
        task_type: TaskType | None,
        history: list[tuple[str, str]],
        existing_plan_artifact_id: str | None,
        turn_inputs: dict[str, Any],
    ) -> TurnOutcome:
        from travel_agent.orchestration.multi_agent.orchestrator_agent import run_orchestrator

        import inspect

        if "history" in inspect.signature(run_orchestrator).parameters:
            orch = run_orchestrator(
                ctx,
                settings,
                user_message,
                request_id,
                runner=runner,
                model=self._orchestrator_model,
                history=history,
                task_type=task_type,
                turn_inputs=_fixed_turn_inputs(
                    task_type, existing_plan_artifact_id, turn_inputs
                ),
            )
        else:
            # Compatibility for injected legacy adapters.
            orch = run_orchestrator(
                ctx,
                settings,
                user_message,
                request_id,
                runner=runner,
                model=self._orchestrator_model,
            )
        results = [_dict_to_result(item) for item in orch.get("results", [])]
        if orch.get("clarification"):
            return TurnOutcome(
                status=STATUS_CLARIFICATION_REQUIRED,
                reply=orch.get("reply") or "需要补充目的地/天数等关键信息。",
                results=results,
            )
        status = STATUS_COMPLETED
        if _plan_required_for_turn("dynamic", task_type):
            planner = next((result for result in reversed(results) if result.agent == "planner"), None)
            if planner is None or planner.status not in {
                STATUS_COMPLETED,
                STATUS_COMPLETED_WITH_WARNINGS,
            }:
                status = STATUS_INCOMPLETE
        return TurnOutcome(
            status=status,
            reply=orch.get("reply") or "",
            results=results,
        )

    # ------------------------------------------------------------------ #
    # Reviewer + 修复周期 + Renderer Gate（Engine 确定性控制）
    # ------------------------------------------------------------------ #
    def _apply_review_and_render(
        self,
        ctx: Any,
        settings: Any,
        outcome: TurnOutcome,
        task_type: TaskType | None,
        task_brief: str,
        request_id: str,
    ) -> None:
        from travel_agent.orchestration.multi_agent.render_gate import render_plan_outcome

        delivery = outcome.status
        if requires_semantic_review(self.capabilities, task_type):
            delivery = self._run_review_cycle(ctx, settings, outcome, task_brief, request_id)

        allowed_agents = (
            frozenset({"single_agent"})
            if self.capabilities.mode == "single"
            else frozenset({"planner"})
        )
        render = render_plan_outcome(
            ctx, outcome.plan_artifact_id, delivery, allowed_agents=allowed_agents
        )
        outcome.cards = render.get("cards") or []
        outcome.map_payload = render.get("map_payload")
        outcome.gate_status = render.get("gate_status") or "skipped"
        if outcome.gate_status in {"rejected", "rendered_incomplete"}:
            outcome.status = STATUS_INCOMPLETE
        elif not render.get("rendered"):
            outcome.status = delivery if delivery in {STATUS_FAILED, STATUS_INCOMPLETE} else STATUS_INCOMPLETE
        else:
            outcome.status = delivery

    def _run_review_cycle(
        self,
        ctx: Any,
        settings: Any,
        outcome: TurnOutcome,
        task_brief: str,
        request_id: str,
    ) -> str:
        from travel_agent.orchestration.multi_agent.review import (
            ReviewContext,
            build_review_callable,
            repair_targets,
            resolve_delivery_status,
            run_semantic_review,
        )

        runner = self._resolve_runner(ctx, settings)
        plan_payload = ctx.store.get(outcome.plan_artifact_id) or {}
        review_callable = self._review_callable or build_review_callable(settings)
        if review_callable is None:
            # fail-closed：V3 固定要 Review，Reviewer 不可用时不得声称完成。
            outcome.status = STATUS_INCOMPLETE
            return STATUS_INCOMPLETE

        review_ctx = ReviewContext(
            request_id=request_id,
            plan=plan_payload,
            profile_brief=_profile_brief(ctx),
            subagent_results=[r for r in outcome.results if isinstance(r, SubagentResult)],
            task_brief=task_brief,
        )
        review = run_semantic_review(review_ctx, review_callable=review_callable)
        outcome.review = review
        from travel_agent.orchestration.multi_agent.trace import current_trace

        trace = current_trace()
        if trace is not None:
            trace.append(
                "review",
                agent="reviewer",
                status=review.verdict,
                attempt=1,
                error=review.error,
                duration_ms=review.duration_ms,
                detail={
                    "issues": [
                        {"issue_type": issue.issue_type, "severity": issue.severity}
                        for issue in review.issues
                    ]
                },
            )
        delivery, start_repair = resolve_delivery_status(
            review,
            reviewer_enabled=True,
            max_rework=self.capabilities.max_rework,
            rework_used=outcome.rework_used,
        )
        if start_repair:
            # 整个修复周期只计一次 rework；修复后不再进行第二轮 Reviewer。
            delivery = self._run_repair_cycle(
                ctx, runner, request_id, review, outcome, repair_targets(review)
            )
            outcome.rework_used += 1
        return delivery

    def _run_repair_cycle(
        self,
        ctx: Any,
        runner: Any,
        request_id: str,
        review: ReviewResult,
        outcome: TurnOutcome,
        targets: list[str],
    ) -> str:
        domain_targets = [name for name in targets if name != "planner"]
        instructions = _repair_instructions_by_agent(review)
        new_artifact_ids: list[str] = []
        previous_plan = ctx.store.get(outcome.plan_artifact_id) or {}
        planner_input_ids = list(previous_plan.get("source_artifact_ids") or [])
        for agent in domain_targets:
            task = SubagentTask(
                request_id=request_id,
                task_id=new_task_id(agent),
                agent=agent,
                instruction=instructions.get(agent) or "按 Reviewer 意见定向补充本领域结果。",
                attempt=2,
            )
            result = runner.run_subagent(task)
            outcome.results.append(result)
            new_artifact_ids.extend(_evidence_ids(result))
        for artifact_id in new_artifact_ids:
            if artifact_id not in planner_input_ids:
                planner_input_ids.append(artifact_id)

        planner_task = SubagentTask(
            request_id=request_id,
            task_id=new_task_id("planner"),
            agent="planner",
            instruction="使用修复周期更新后的领域证据重新规划，并执行 plan_and_critique。",
            inputs={"artifact_ids": planner_input_ids},
            attempt=2,
        )
        planner_result = runner.run_subagent(planner_task)
        outcome.results.append(planner_result)
        outcome.plan_artifact_id = _plan_artifact_id_from_results(
            ctx,
            [planner_result],
            request_id,
        )
        if planner_result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
            if not outcome.plan_artifact_id:
                return STATUS_INCOMPLETE
            # No second semantic review, but deterministic acceptance is mandatory.
            from travel_agent.agent.toolkit import _validate_plan_gate

            gate_error, _record = _validate_plan_gate(ctx, outcome.plan_artifact_id)
            repaired = ctx.store.get(outcome.plan_artifact_id) or {}
            critic = repaired.get("critic") or {}
            critical_issues = [
                issue
                for issue in (critic.get("issues") or [])
                if str(issue.get("severity") or "").lower() in {"error", "critical"}
            ]
            if (
                gate_error
                or critic.get("passed") is not True
                or critical_issues
                or planner_result.unresolved
            ):
                return STATUS_INCOMPLETE
            # 修复后不再 Review；通过 hard critic/Gate 后按带告警交付。
            return STATUS_COMPLETED_WITH_WARNINGS
        # A failed repair is a non-deliverable turn, not a transport/runtime
        # success.  Keep the top-level contract fail-closed.
        return STATUS_INCOMPLETE

    # ------------------------------------------------------------------ #
    def _resolve_runner(self, ctx: Any, settings: Any) -> Any:
        if self._runner is not None:
            bound_ctx = getattr(self._runner, "context", None)
            if bound_ctx is not None and bound_ctx is not ctx:
                raise RuntimeError(
                    "injected SubagentRunner is session-scoped and cannot be reused across SessionContext"
                )
            return self._runner
        from travel_agent.orchestration.multi_agent.executor import build_subagent_executor
        from travel_agent.orchestration.multi_agent.runner import SubagentRunner

        executor = build_subagent_executor(settings, model=self._subagent_model)
        # Auto-created runners are per run/session. Never cache the first ctx.
        return SubagentRunner(ctx, executor)


# --- 辅助 ------------------------------------------------------------------- #


def _fixed_delivery_status(results: list[SubagentResult]) -> str:
    planner = next((r for r in reversed(results) if r.agent == "planner"), None)
    if planner is None:
        # 无 planner 的轻量任务：全部成功才算完成。
        return (
            STATUS_COMPLETED
            if all(r.status == STATUS_COMPLETED for r in results)
            else STATUS_FAILED
        )
    if planner.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        return planner.status
    return STATUS_FAILED


def _compose_fixed_reply(results: list[SubagentResult]) -> str:
    planner = next((r for r in reversed(results) if r.agent == "planner"), None)
    if planner is not None:
        base = planner.summary or "行程规划已完成。"
    else:
        base = "；".join(r.summary for r in results if r.summary) or "调研已完成。"
    warnings = [warning for r in results for warning in r.warnings]
    if warnings:
        base += "（注意：" + "；".join(dict.fromkeys(warnings)) + "）"
    return base


def _dict_to_result(item: Any) -> Any:
    if isinstance(item, SubagentResult):
        return item
    if not isinstance(item, dict):
        return item
    return SubagentResult(
        request_id=str(item.get("request_id") or ""),
        task_id=str(item.get("task_id") or ""),
        agent=str(item.get("agent") or ""),
        status=str(item.get("status") or STATUS_FAILED),
        attempt=int(item.get("attempt") or 1),
        summary=str(item.get("summary") or ""),
        payload=dict(item.get("payload") or {}),
        evidence=list(item.get("evidence") or []),
        constraints_used=list(item.get("constraints_used") or []),
        warnings=list(item.get("warnings") or []),
        unresolved=list(item.get("unresolved") or []),
        tool_trace=list(item.get("tool_trace") or []),
        token_usage=dict(item.get("token_usage") or {}),
        duration_ms=int(item.get("duration_ms") or 0),
        error=item.get("error"),
    )


def _evidence_ids(result: SubagentResult) -> list[str]:
    ids: list[str] = []
    for item in result.evidence:
        artifact_id = item.get("artifact_id") if isinstance(item, dict) else None
        if artifact_id and artifact_id not in ids:
            ids.append(artifact_id)
    return ids


def _plan_required_for_turn(dispatch: str, task_type: TaskType | None) -> bool:
    """完整规划/修改必须有 Planner；动态未知任务也按 fail-closed 处理。"""
    if task_type in {TaskType.ROUTE_QUERY, TaskType.POI_ADVICE, TaskType.DAY_ADVICE}:
        return False
    return task_type in REVIEW_REQUIRED_TASK_TYPES or dispatch == "dynamic"


def _fixed_turn_inputs(
    task_type: TaskType | None,
    existing_plan_artifact_id: str | None,
    turn_inputs: dict[str, Any],
) -> dict[str, Any]:
    inputs = dict(turn_inputs)
    if task_type == TaskType.ITINERARY_REVISION and existing_plan_artifact_id:
        ids = [str(item) for item in (inputs.get("artifact_ids") or []) if item]
        if existing_plan_artifact_id not in ids:
            ids.insert(0, existing_plan_artifact_id)
        inputs["artifact_ids"] = ids
        inputs["plan_artifact_id"] = existing_plan_artifact_id
    return inputs


def _plan_artifact_id_from_results(
    ctx: Any,
    results: list[Any],
    request_id: str,
) -> str | None:
    """只接受成功 Planner 在本 request/task 中明确回报的 itinerary artifact。"""
    for result in reversed(results):
        if not isinstance(result, SubagentResult):
            continue
        if result.agent != "planner" or result.status not in {
            STATUS_COMPLETED,
            STATUS_COMPLETED_WITH_WARNINGS,
        }:
            continue
        for artifact_id in reversed(_evidence_ids(result)):
            record = ctx.store.get_record(artifact_id)
            if record is None:
                continue
            if (
                record.get("kind") == "itinerary"
                and record.get("request_id") == request_id
                and record.get("task_id") == result.task_id
                and record.get("agent") == "planner"
            ):
                return artifact_id
    return None


def _repair_instructions_by_agent(review: ReviewResult) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for issue in review.issues:
        if issue.repair_target and issue.repair_instruction:
            grouped.setdefault(issue.repair_target, []).append(issue.repair_instruction)
    return {agent: "\n".join(instructions) for agent, instructions in grouped.items()}


def _profile_brief(ctx: Any) -> dict[str, Any]:
    try:
        from travel_agent.agent.toolkit import _profile_brief as brief_fn

        return brief_fn(ctx.profile)
    except Exception:  # noqa: BLE001
        return {}
