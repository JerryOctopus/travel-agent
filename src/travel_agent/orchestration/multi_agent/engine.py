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

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from typing import Any

from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent.schemas import (
    SEVERITY_RECOVERABLE,
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
    routing_policy_hash: str = ""


@dataclass(frozen=True)
class DynamicBaseOutcome:
    """Immutable V2/V3 base result; production variants execute it independently."""

    status: str
    reply: str
    results: tuple[SubagentResult, ...] = ()
    plan_artifact_id: str | None = None
    routing_policy_hash: str = ""

    def to_turn_outcome(self) -> TurnOutcome:
        return TurnOutcome(
            status=self.status,
            reply=self.reply,
            results=list(self.results),
            plan_artifact_id=self.plan_artifact_id,
            routing_policy_hash=self.routing_policy_hash,
        )


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
        from travel_agent.orchestration.multi_agent.deadlines import TurnDeadline

        deadline = TurnDeadline.start(settings)
        if self.capabilities.mode == "single":
            outcome = self._run_v0(ctx, settings, user_message, history or [], rid)
        else:
            runner = self._resolve_runner(ctx, settings)
            if self.capabilities.dispatch == "fixed":
                outcome = self._run_fixed(
                    ctx,
                    runner,
                    deadline,
                    rid,
                    task_type,
                    task_brief or user_message,
                    existing_plan_artifact_id,
                    turn_inputs or {},
                )
            else:
                outcome = self.run_dynamic_base(
                    ctx,
                    settings,
                    runner,
                    deadline,
                    rid,
                    user_message,
                    task_brief,
                    task_type,
                    history or [],
                    existing_plan_artifact_id,
                    turn_inputs or {},
                ).to_turn_outcome()

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
                ctx,
                settings,
                outcome,
                task_type,
                task_brief or user_message,
                rid,
                deadline,
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
                "routing_policy_hash": outcome.routing_policy_hash,
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
        deadline: Any,
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
            deadline=deadline,
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
    def run_dynamic_base(
        self,
        ctx: Any,
        settings: Any,
        runner: Any,
        deadline: Any,
        request_id: str,
        user_message: str,
        task_brief: str,
        task_type: TaskType | None,
        history: list[tuple[str, str]],
        existing_plan_artifact_id: str | None,
        turn_inputs: dict[str, Any],
    ) -> DynamicBaseOutcome:
        """Shared V2/V3 wave state machine. It never inspects reviewer capability."""
        from travel_agent.orchestration.multi_agent.orchestrator import DispatchLedger
        from travel_agent.orchestration.multi_agent.orchestrator_agent import (
            route_wave,
            routing_policy_hash,
        )
        from travel_agent.orchestration.multi_agent.trace import current_trace

        orchestration = getattr(settings, "orchestration", None)
        max_waves = max(1, min(3, int(getattr(orchestration, "routing_max_waves", 3) or 3)))
        max_calls = max(1, min(3, int(getattr(orchestration, "routing_max_calls", 3) or 3)))
        max_dispatches = max(
            1, int(getattr(orchestration, "routing_max_dispatches", 6) or 6)
        )
        wave1_max = max(
            1, min(4, int(getattr(orchestration, "routing_wave1_max_tasks", 4) or 4))
        )
        # 给“当前 Router 路由策略”生成一个稳定指纹
        policy_hash = routing_policy_hash(settings)
        # 限制任务调度次数
        ledger = DispatchLedger(
            max_total=max_dispatches,
            per_agent_limits={
                "attraction": 2,
                "hotel": 2,
                "restaurant": 2,
                "transport": 2,
                "planner": 0,
            },
        )
        base_inputs = _fixed_turn_inputs(
            task_type, existing_plan_artifact_id, turn_inputs
        )
        results: list[SubagentResult] = []
        attempted: list[str] = [] # 保存已经尝试过的派工目标
        router_calls = 0
        last_wave_results: list[SubagentResult] = [] # 上一波派工的结果
        stop_reason = "routing_exhausted" # 默认停止原因
        trace = current_trace()

        for wave in range(1, max_waves + 1):
            missing_hard, _missing_soft = _required_evidence(
                ctx, task_type, results, base_inputs
            )
            if wave > 1 and not missing_hard:
                stop_reason = "hard_evidence_ready"
                break
            if router_calls >= max_calls or ledger.attempts >= max_dispatches:
                stop_reason = "routing_budget_exhausted"
                break
            if wave > 1 and not deadline.admits_recovery_wave():
                stop_reason = "recovery_admission_denied"
                break
            if wave == 3 and not _wave3_recovery_allowed(
                last_wave_results, missing_hard
            ):
                stop_reason = "wave3_conditions_not_met"
                break

            if not deadline.admits_router():
                stop_reason = "router_admission_denied"
                break
            router_admission = deadline.trace_detail()
            router_timeout = deadline.router_timeout_preserving_worker()
            if router_timeout <= 0:
                stop_reason = "no_usable_preplanner_time"
                break
            remaining_dispatch = max_dispatches - ledger.attempts
            max_tasks = min(wave1_max if wave == 1 else 2, remaining_dispatch)
            decision = route_wave(
                ctx,
                settings,
                user_message,
                request_id,
                wave=wave,
                model=self._orchestrator_model,
                history=history,
                task_type=task_type,
                turn_inputs=base_inputs,
                compact_results=_compact_router_results(results),
                attempted_objectives=attempted,
                missing_evidence=missing_hard,
                timeout_seconds=router_timeout,
                max_tasks=max_tasks,
            )
            router_calls += 1
            if trace is not None:
                trace.append(
                    "routing",
                    agent="orchestrator",
                    status="failed" if decision.error else "completed",
                    attempt=wave,
                    error=decision.error,
                    duration_ms=None,
                    detail={
                        "wave": wave,
                        "routing_policy_hash": policy_hash,
                        "routing_manifest_hash": _routing_manifest_hash(decision),
                        "readiness": {
                            "router_ready": decision.ready,
                            "missing_hard": missing_hard,
                        },
                        "objective_key": [task.objective_key for task in decision.tasks],
                        "admission_required_ms": int(
                            (
                                deadline.config.router_useful
                                + deadline.config.recovery_worker_useful
                                + deadline.config.planner_reserve
                                + deadline.config.admission_guard
                            )
                            * 1000
                        ),
                        "effective_timeout_ms": int(router_timeout * 1000),
                        "admission": router_admission,
                        **deadline.trace_detail(),
                    },
                )
            if decision.clarification and not results:
                return DynamicBaseOutcome(
                    status=STATUS_CLARIFICATION_REQUIRED,
                    reply=decision.reply or "需要补充目的地/天数等关键信息。",
                    routing_policy_hash=policy_hash,
                )
            if decision.error:
                stop_reason = "router_failed"
                break

            tasks: list[SubagentTask] = []
            objective_by_task: dict[str, str] = {}
            known_task_ids = {result.task_id for result in results}
            successful_task_ids = {
                result.task_id
                for result in results
                if result.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}
            }
            for routed in decision.tasks:
                if routed.objective_key in attempted:
                    continue
                if any(dep not in known_task_ids for dep in routed.depends_on):
                    continue
                if any(dep not in successful_task_ids for dep in routed.depends_on):
                    continue
                limit_error = ledger.authorize(routed.agent, routed.objective)
                if limit_error:
                    continue
                attempted.append(routed.objective_key)
                inputs = dict(base_inputs)
                artifact_ids = list(
                    dict.fromkeys(
                        [str(value) for value in (inputs.get("artifact_ids") or []) if value]
                        + _all_evidence_ids(results)
                    )
                )
                if artifact_ids:
                    inputs["artifact_ids"] = artifact_ids
                task = SubagentTask(
                    request_id=request_id,
                    task_id=new_task_id(routed.agent),
                    agent=routed.agent,
                    instruction=routed.instruction,
                    inputs=inputs,
                    depends_on=list(routed.depends_on),
                    attempt=ledger.counts.get(routed.agent, 1),
                )
                tasks.append(task)
                objective_by_task[task.task_id] = routed.objective_key
            if not tasks:
                stop_reason = "router_ready" if decision.ready else "no_valid_delta_tasks"
                break

            worker_admission = deadline.trace_detail()
            worker_timeout = deadline.effective_preplanner_timeout(
                deadline.config.worker_timeout
            )
            if worker_timeout <= 0 or (
                _task_needs_plan(task_type)
                and deadline.usable_preplanner_time()
                < deadline.config.recovery_worker_useful
            ):
                stop_reason = "worker_admission_denied"
                break
            last_wave_results = _execute_parallel_wave(
                runner, tasks, timeout_seconds=worker_timeout
            )
            for result in last_wave_results:
                ledger.record(result)
            results.extend(last_wave_results)
            if trace is not None:
                trace.append(
                    "wave_execution",
                    agent="engine",
                    status="completed",
                    attempt=wave,
                    detail={
                        "wave": wave,
                        "tasks": [
                            {
                                "task_id": result.task_id,
                                "agent": result.agent,
                                "objective_key": objective_by_task.get(result.task_id, ""),
                                "status": result.status,
                            }
                            for result in last_wave_results
                        ],
                        "effective_timeout_ms": int(worker_timeout * 1000),
                        "admission": worker_admission,
                        **deadline.trace_detail(),
                    },
                )

        missing_hard, missing_soft = _required_evidence(
            ctx, task_type, results, base_inputs
        )
        if _task_needs_plan(task_type):
            # Route evidence remains a hard routing target so recovery waves
            # still dispatch transport. Once all waves are exhausted, the
            # planner's own estimator plus final critic can safely recover it.
            if task_type == TaskType.FULL_TRIP_PLAN:
                missing_hard = [
                    item for item in missing_hard if item != "路线可行性"
                ]
            if missing_hard:
                _trace_stop(trace, deadline, stop_reason, missing_hard, policy_hash)
                return DynamicBaseOutcome(
                    status=STATUS_INCOMPLETE,
                    reply=_incomplete_evidence_reply(missing_hard),
                    results=tuple(results),
                    routing_policy_hash=policy_hash,
                )
            if not deadline.admits_planner():
                _trace_stop(
                    trace,
                    deadline,
                    "planner_admission_denied",
                    [],
                    policy_hash,
                )
                return DynamicBaseOutcome(
                    status=STATUS_INCOMPLETE,
                    reply="领域证据已返回，但剩余时间不足以安全启动 Planner。",
                    results=tuple(results),
                    routing_policy_hash=policy_hash,
                )
            artifact_ids = list(
                dict.fromkeys(
                    [str(value) for value in (base_inputs.get("artifact_ids") or []) if value]
                    + _all_evidence_ids(results)
                )
            )
            if not artifact_ids:
                return DynamicBaseOutcome(
                    status=STATUS_INCOMPLETE,
                    reply="当前没有可绑定给 Planner 的本轮证据 artifact。",
                    results=tuple(results),
                    routing_policy_hash=policy_hash,
                )
            planner_inputs = dict(base_inputs)
            planner_inputs["artifact_ids"] = artifact_ids
            if existing_plan_artifact_id:
                planner_inputs["plan_artifact_id"] = existing_plan_artifact_id
            planner_task = SubagentTask(
                request_id=request_id,
                task_id=new_task_id("planner"),
                agent="planner",
                instruction=(task_brief or user_message)
                + "\n基于明确绑定的 artifacts 生成行程；soft evidence 缺失时标记 unresolved。",
                inputs=planner_inputs,
            )
            if trace is not None:
                trace.append(
                    "admission",
                    agent="planner",
                    status="admitted",
                    detail={
                        "effective_timeout_ms": int(
                            deadline.planner_timeout(deadline.config.planner_timeout)
                            * 1000
                        ),
                        **deadline.trace_detail(),
                    },
                )
            planner_timeout = deadline.planner_timeout(
                deadline.config.planner_timeout
            )
            planner_result = _run_with_optional_timeout(
                runner, planner_task, planner_timeout
            )
            results.append(planner_result)
            plan_id = _plan_artifact_id_from_results(ctx, [planner_result], request_id)
            status = _planner_delivery_status(ctx, planner_result, plan_id)
            reply = planner_result.summary or (
                "Planner 已生成行程。" if plan_id else "Planner 未产出可交付行程。"
            )
            if missing_soft and status != STATUS_COMPLETED:
                reply += " 未完全覆盖：" + "、".join(missing_soft)
            return DynamicBaseOutcome(
                status=status,
                reply=reply,
                results=tuple(results),
                plan_artifact_id=plan_id,
                routing_policy_hash=policy_hash,
            )

        reply = _render_lightweight_evidence_reply(
            ctx, task_type, results, missing_hard
        )
        return DynamicBaseOutcome(
            status=STATUS_COMPLETED if not missing_hard else STATUS_INCOMPLETE,
            reply=reply,
            results=tuple(results),
            routing_policy_hash=policy_hash,
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
        deadline: Any,
    ) -> None:
        from travel_agent.orchestration.multi_agent.render_gate import render_plan_outcome

        delivery = outcome.status
        if requires_semantic_review(self.capabilities, task_type):
            delivery = self._run_review_cycle(
                ctx, settings, outcome, task_brief, request_id, deadline
            )

        allowed_agents = (
            frozenset({"single_agent"})
            if self.capabilities.mode == "single"
            else frozenset({"planner"})
        )
        render_admission = deadline.trace_detail()
        render_started = time.monotonic()
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
        from travel_agent.orchestration.multi_agent.trace import current_trace

        trace = current_trace()
        if trace is not None:
            trace.append(
                "finalization",
                agent="renderer",
                status=outcome.gate_status,
                duration_ms=round((time.monotonic() - render_started) * 1000, 2),
                detail={
                    "admission": render_admission,
                    **deadline.trace_detail(),
                },
            )

    def _run_review_cycle(
        self,
        ctx: Any,
        settings: Any,
        outcome: TurnOutcome,
        task_brief: str,
        request_id: str,
        deadline: Any,
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
        review_admission = deadline.trace_detail()
        review_timeout = deadline.reviewer_timeout()
        if review_timeout <= 0:
            outcome.status = STATUS_INCOMPLETE
            return STATUS_INCOMPLETE
        review_callable = self._review_callable or build_review_callable(
            settings,
            timeout_seconds=review_timeout,
            evaluation_trace=(
                ctx.evaluation_trace
                if getattr(ctx, "evaluation_trace_enabled", False)
                else None
            ),
        )
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
        review = run_semantic_review(
            review_ctx,
            review_callable=review_callable,
            timeout_seconds=review_timeout,
        )
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
                    "effective_timeout_ms": int(review_timeout * 1000),
                    "admission": review_admission,
                    **deadline.trace_detail(),
                    "issues": [
                        {
                            "issue_type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "evidence": list(issue.evidence),
                            "repair_target": issue.repair_target,
                        }
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
            targets = repair_targets(review)
            domain_repair = any(target != "planner" for target in targets)
            if not deadline.admits_repair(domain_worker=domain_repair):
                if trace is not None:
                    trace.append(
                        "repair",
                        agent="engine",
                        status=STATUS_INCOMPLETE,
                        detail={
                            "stop_reason": "repair_admission_denied",
                            "targets": targets,
                            **deadline.trace_detail(),
                        },
                    )
                return STATUS_INCOMPLETE
            # 整个修复周期只计一次 rework；修复后不再进行第二轮 Reviewer。
            delivery = self._run_repair_cycle(
                ctx, runner, request_id, review, outcome, targets, deadline
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
        deadline: Any,
    ) -> str:
        # One repair cycle owns at most one domain worker, then one Planner.
        domain_targets = [name for name in targets if name != "planner"][:1]
        instructions = _repair_instructions_by_agent(review)
        new_artifact_ids: list[str] = []
        previous_plan = ctx.store.get(outcome.plan_artifact_id) or {}
        planner_input_ids = list(previous_plan.get("source_artifact_ids") or [])
        if not domain_targets and outcome.plan_artifact_id and not planner_input_ids:
            planner_input_ids.append(outcome.plan_artifact_id)
        for agent in domain_targets:
            task = SubagentTask(
                request_id=request_id,
                task_id=new_task_id(agent),
                agent=agent,
                instruction=instructions.get(agent) or "按 Reviewer 意见定向补充本领域结果。",
                inputs={"artifact_ids": list(planner_input_ids)},
                attempt=2,
            )
            worker_timeout = deadline.repair_worker_timeout()
            if worker_timeout <= 0:
                return STATUS_INCOMPLETE
            from travel_agent.orchestration.meter import current_turn_meter

            meter = current_turn_meter()
            if meter is not None:
                meter.record_dispatch(agent)
            result = _run_with_optional_timeout(runner, task, worker_timeout)
            outcome.results.append(result)
            new_artifact_ids.extend(_evidence_ids(result))
        for artifact_id in new_artifact_ids:
            if artifact_id not in planner_input_ids:
                planner_input_ids.append(artifact_id)

        planner_review_instructions = "\n".join(
            issue.repair_instruction
            for issue in review.issues
            if issue.severity == SEVERITY_RECOVERABLE and issue.repair_instruction
        )
        planner_review_issue_types = [
            issue.issue_type
            for issue in review.issues
            if issue.severity == SEVERITY_RECOVERABLE
        ]

        planner_task = SubagentTask(
            request_id=request_id,
            task_id=new_task_id("planner"),
            agent="planner",
            instruction=(
                "使用修复周期更新后的领域证据重新规划，并执行 plan_and_critique。"
                + (
                    "\nReviewer 定向修复要求：\n" + planner_review_instructions
                    if planner_review_instructions
                    else ""
                )
            ),
            inputs={
                "artifact_ids": planner_input_ids,
                "revision_directives": {
                    "reviewer_instructions": planner_review_instructions,
                    "reviewer_issue_types": planner_review_issue_types,
                },
            },
            attempt=2,
        )
        planner_timeout = deadline.repair_planner_timeout()
        if planner_timeout <= 0:
            return STATUS_INCOMPLETE
        planner_result = _run_with_optional_timeout(
            runner, planner_task, planner_timeout
        )
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


def _component_timeout(agent: str) -> float:
    from travel_agent.orchestration.multi_agent.registry import SUBAGENT_REGISTRY

    definition = SUBAGENT_REGISTRY.get(agent)
    return float(definition.timeout_seconds if definition is not None else 0.0)


def _run_with_optional_timeout(
    runner: Any,
    task: SubagentTask,
    timeout_seconds: float,
) -> SubagentResult:
    try:
        return runner.run_subagent(task, timeout_seconds=timeout_seconds)
    except TypeError as exc:
        if "timeout_seconds" not in str(exc):
            raise
        return runner.run_subagent(task)


def _execute_parallel_wave(
    runner: Any,
    tasks: list[SubagentTask],
    *,
    timeout_seconds: float,
) -> list[SubagentResult]:
    """Execute a wave concurrently; timeout is one shared wall-clock window."""
    if not tasks:
        return []
    from travel_agent.orchestration.meter import current_turn_meter

    futures: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="dynamic-wave") as pool:
        for task in tasks:
            meter = current_turn_meter()
            if meter is not None:
                meter.record_dispatch(task.agent)
            context = contextvars.copy_context()
            futures[task.task_id] = pool.submit(
                context.run,
                _run_with_optional_timeout,
                runner,
                task,
                timeout_seconds,
            )
        return [futures[task.task_id].result() for task in tasks]


def _all_evidence_ids(results: list[SubagentResult]) -> list[str]:
    return list(
        dict.fromkeys(
            artifact_id
            for result in results
            if result.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}
            for artifact_id in _evidence_ids(result)
        )
    )


def _compact_router_results(results: list[SubagentResult]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for result in results:
        compact.append(
            {
                "task_id": result.task_id,
                "agent": result.agent,
                "status": result.status,
                "attempt": result.attempt,
                "summary": result.summary[:400],
                "evidence": [
                    {
                        "artifact_id": item.get("artifact_id"),
                        "kind": item.get("kind"),
                    }
                    for item in result.evidence
                    if isinstance(item, dict) and item.get("artifact_id")
                ],
                "warnings": list(result.warnings)[:6],
                "unresolved": list(result.unresolved)[:6],
                "error": result.error,
            }
        )
    return compact


def _task_needs_plan(task_type: TaskType | None) -> bool:
    return task_type in REVIEW_REQUIRED_TASK_TYPES


def _required_evidence(
    ctx: Any,
    task_type: TaskType | None,
    results: list[SubagentResult],
    base_inputs: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return deterministic (hard_missing, soft_missing) evidence labels."""
    kinds: set[str] = set()
    nonempty_kinds: set[str] = set()
    restaurant_keys: set[str] = set()

    def register_record(record: dict[str, Any] | None) -> None:
        if not record or not record.get("kind"):
            return
        kind = str(record["kind"])
        kinds.add(kind)
        payload = record.get("payload") or {}
        if kind in {"restaurants", "hotels"}:
            collection = payload.get(kind) or payload.get("items")
            if collection:
                nonempty_kinds.add(kind)
                if kind == "restaurants":
                    for item in collection:
                        if isinstance(item, dict):
                            key = item.get("poi_id") or item.get("name")
                            if key:
                                restaurant_keys.add(str(key))
        else:
            nonempty_kinds.add(kind)

    for result in results:
        if result.status not in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
            continue
        for item in result.evidence:
            if isinstance(item, dict) and item.get("kind"):
                kind = str(item["kind"])
                record = ctx.store.get_record(str(item.get("artifact_id") or ""))
                if record is not None:
                    register_record(record)
                else:
                    kinds.add(kind)
                    # Restaurant readiness is deliberately payload-backed;
                    # legacy metadata-only evidence remains valid elsewhere.
                    if kind != "restaurants":
                        nonempty_kinds.add(kind)
    for artifact_id in base_inputs.get("artifact_ids") or []:
        register_record(ctx.store.get_record(str(artifact_id)))
    if "pois" in kinds:
        kinds.add("candidates")
    if "pois" in nonempty_kinds:
        nonempty_kinds.add("candidates")

    # 当前用户已经明确提出的结构化约束
    state = dict(getattr(getattr(ctx, "profile", None), "constraint_state", {}) or {})
    hard: list[str] = []
    soft: list[str] = []

    def require(kind: str, label: str, *, hard_required: bool = True) -> None:
        if kind in nonempty_kinds:
            return
        target = hard if hard_required else soft
        if label not in target:
            target.append(label)

    if task_type == TaskType.FULL_TRIP_PLAN:
        require("candidates", "景点候选")
        require("routes", "路线可行性")
        hotel_explicit = bool(
            getattr(getattr(ctx, "profile", None), "hotel_area", None)
            or any(
                state.get(key) not in (None, "", [], {})
                for key in (
                    "lodging_area",
                    "compare_lodging_areas",
                    "hotel_budget_per_night_cny",
                    "prepaid_lodging_cny",
                )
            )
        )
        require("hotels", "住宿证据", hard_required=hotel_explicit)
        budget_explicit = bool(
            getattr(getattr(ctx, "profile", None), "budget_limit", None)
            or any(
                state.get(key) not in (None, "", [], {})
                for key in (
                    "budget_max_cny",
                    "budget_total_cny",
                    "budget_per_person_cny",
                    "budget_remaining_cny",
                    "hotel_budget_per_night_cny",
                )
            )
        )
        require("budget", "预算证据", hard_required=budget_explicit)
        raw_dietary = state.get("dietary") or []
        food_explicit = bool(
            getattr(getattr(ctx, "profile", None), "food_preference", None)
            or raw_dietary
            or "food" in (getattr(getattr(ctx, "profile", None), "interests", None) or [])
        )
        # A missing meal is a completeness warning for a generic itinerary,
        # but only explicit cuisine/dietary requirements justify blocking the
        # planner. This keeps evidence strict without turning provider sparsity
        # into a total delivery failure.
        require("restaurants", "餐饮证据", hard_required=food_explicit)
        required_meals = max(1, int(getattr(getattr(ctx, "profile", None), "days", 1) or 1))
        if len(restaurant_keys) < required_meals:
            target = hard if food_explicit else soft
            if "餐饮证据" not in target:
                target.append("餐饮证据")
    elif task_type == TaskType.ITINERARY_REVISION:
        require("itinerary", "既有行程")
    elif task_type == TaskType.ROUTE_QUERY:
        require("routes", "路线耗时/距离")
    elif task_type == TaskType.POI_ADVICE:
        if not kinds.intersection({"candidates", "restaurants", "hotels"}):
            hard.append("候选地点")
        if state.get("walking_time_max_min") is not None:
            require("routes", "步行时间")
        elif state.get("location"):
            require("routes", "与指定位置的距离")
        extra = _lightweight_missing_payload_fields(results, ctx.profile)
        for label in extra:
            if label not in hard:
                hard.append(label)
    elif task_type == TaskType.DAY_ADVICE:
        require("weather", "天气")
        require("candidates", "活动候选")
    elif task_type is None or task_type == TaskType.UNKNOWN:
        hard.append("可识别的任务目标")
    return hard, soft


def _wave3_recovery_allowed(
    wave2_results: list[SubagentResult],
    missing_hard: list[str],
) -> bool:
    if not missing_hard or not wave2_results:
        return False
    retryable_failure = any(
        result.status in {STATUS_FAILED, "budget_exhausted"}
        for result in wave2_results
    )
    new_unlocking_artifact = any(
        result.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}
        and bool(_evidence_ids(result))
        for result in wave2_results
    )
    return retryable_failure or new_unlocking_artifact


def _routing_manifest_hash(decision: Any) -> str:
    payload = decision.to_dict() if hasattr(decision, "to_dict") else decision
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _trace_stop(
    trace: Any,
    deadline: Any,
    reason: str,
    missing_hard: list[str],
    policy_hash: str,
) -> None:
    if trace is None:
        return
    trace.append(
        "routing_stop",
        agent="engine",
        status=STATUS_INCOMPLETE if missing_hard else STATUS_COMPLETED,
        detail={
            "stop_reason": reason,
            "missing_hard": list(missing_hard),
            "routing_policy_hash": policy_hash,
            **deadline.trace_detail(),
        },
    )


def _incomplete_evidence_reply(missing_hard: list[str]) -> str:
    return (
        "当前缺少安全规划所需的关键证据："
        + "、".join(missing_hard)
        + "。未强行启动 Planner，也未生成行程。"
    )


def _planner_delivery_status(
    ctx: Any,
    planner_result: SubagentResult,
    plan_artifact_id: str | None,
) -> str:
    if planner_result.status not in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
        return STATUS_INCOMPLETE
    if not plan_artifact_id:
        return STATUS_INCOMPLETE
    from travel_agent.agent.toolkit import _validate_plan_gate

    gate_error, record = _validate_plan_gate(ctx, plan_artifact_id)
    payload = (record or {}).get("payload") or {}
    critic = payload.get("critic") or {}
    critical_issues = [
        issue
        for issue in (critic.get("issues") or [])
        if str(issue.get("severity") or "").lower() in {"error", "critical"}
    ]
    if gate_error or critic.get("passed") is not True or critical_issues:
        return STATUS_INCOMPLETE
    return STATUS_COMPLETED


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


def _lightweight_missing_payload_fields(
    results: list[SubagentResult],
    profile: Any,
) -> list[str]:
    """Return missing POI-advice fields not covered by artifact-kind checks."""
    successful_results = [
        result
        for result in results
        if result.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}
    ]
    state = getattr(profile, "constraint_state", {}) or {}
    missing: list[str] = []
    if state.get("budget_per_person_cny") is not None and not _results_have_field(
        successful_results, {"average_cost", "cost", "price_cny"}
    ):
        missing.append("人均消费")
    if state.get("parking_preferred") is True and not _results_have_field(
        successful_results, {"parking_type", "parking_available"}
    ):
        missing.append("停车条件")
    return missing


def _results_have_field(results: list[SubagentResult], keys: set[str]) -> bool:
    def has(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                (key in keys and item not in (None, "", [], {})) or has(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(has(item) for item in value)
        return False

    return any(has(result.payload) for result in results)


def _render_lightweight_evidence_reply(
    ctx: Any,
    task_type: TaskType | None,
    results: list[SubagentResult],
    missing: list[str],
) -> str:
    """Render lightweight answers from artifacts only; never trust free-form worker prose."""
    records: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for result in results:
        for item in result.evidence:
            if not isinstance(item, dict):
                continue
            artifact_id = str(item.get("artifact_id") or "")
            if not artifact_id or artifact_id in seen_ids:
                continue
            payload = ctx.store.get(artifact_id)
            if isinstance(payload, dict):
                records.append((str(item.get("kind") or ""), payload))
                seen_ids.add(artifact_id)

    lines = ["已按工具证据整理如下："]
    weather = next((payload for kind, payload in records if kind == "weather"), None)
    if weather:
        lines.append(
            f"- 天气：{weather.get('city') or ''} {weather.get('condition') or '未知'}，"
            f"{weather.get('temperature_c')}°C（来源：{weather.get('source') or '未标注'}）"
        )

    routes = [payload for kind, payload in records if kind == "routes"]
    route_seen: set[tuple[Any, ...]] = set()
    for payload in routes:
        key = (
            payload.get("origin_name"),
            payload.get("destination_name"),
            payload.get("mode"),
        )
        if key in route_seen:
            continue
        route_seen.add(key)
        lines.append(
            f"- 路线：{payload.get('origin_name') or payload.get('origin_poi_id')} → "
            f"{payload.get('destination_name') or payload.get('destination_poi_id')}，"
            f"{payload.get('mode')}，约 {payload.get('duration_min')} 分钟 / "
            f"{payload.get('distance_km')} 公里（来源：{payload.get('source') or '未标注'}）"
        )

    candidates: list[dict[str, Any]] = []
    if task_type != TaskType.ROUTE_QUERY:
        for kind, payload in records:
            if kind == "candidates":
                candidates.extend(item for item in (payload.get("pois") or []) if isinstance(item, dict))
            elif kind == "restaurants":
                candidates.extend(
                    item for item in (payload.get("restaurants") or payload.get("items") or [])
                    if isinstance(item, dict)
                )
            elif kind == "hotels":
                candidates.extend(item for item in (payload.get("hotels") or []) if isinstance(item, dict))
    state = getattr(ctx.profile, "constraint_state", {}) or {}
    candidates = _select_lightweight_candidates(candidates, routes, state)
    limit = int(state.get("top_n") or state.get("max_selected") or 5)
    candidate_seen: set[str] = set()
    rendered = 0
    for item in candidates:
        identity = str(item.get("poi_id") or item.get("hotel_id") or item.get("name") or "")
        if not identity or identity in candidate_seen:
            continue
        candidate_seen.add(identity)
        details = []
        if item.get("category"):
            details.append(str(item["category"]))
        if item.get("rating") is not None:
            details.append(f"评分 {item['rating']}")
        if item.get("opening_hours"):
            details.append(f"营业/开放时间 {item['opening_hours']}")
        if item.get("average_cost") is not None:
            details.append(f"人均约 {item['average_cost']} 元")
        if item.get("parking_type"):
            details.append(f"停车 {item['parking_type']}")
        if item.get("address"):
            details.append(f"地址 {item['address']}")
        if item.get("source"):
            details.append(f"来源 {item['source']}")
        lines.append(f"- 候选：{item.get('name') or identity}" + (f"（{'；'.join(details)}）" if details else ""))
        rendered += 1
        if rendered >= max(1, min(limit, 8)):
            break

    if missing:
        lines.append(
            "- 未核实："
            + "、".join(missing)
            + "。当前不能确认这些条件已满足，请以地图或官方渠道复核。"
        )
    if len(lines) == 1:
        lines.append("- 当前没有可交付的结构化工具证据。")
    return "\n".join(lines)


def _select_lightweight_candidates(
    candidates: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Entity-dedupe, distance-rank and diversify lightweight POI results."""
    route_rank: dict[str, float] = {}
    for route in routes:
        try:
            duration = float(route.get("duration_min"))
        except (TypeError, ValueError):
            continue
        for key in (route.get("destination_poi_id"), route.get("destination_name")):
            if key:
                route_rank[str(key)] = min(duration, route_rank.get(str(key), duration))

    unique: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for item in candidates:
        if state.get("diversity_required") and str(item.get("category") or "") in {
            "hotel",
            "transport",
        }:
            continue
        name = str(item.get("name") or item.get("poi_id") or item.get("hotel_id") or "").strip()
        entity = _canonical_candidate_entity(name)
        if not entity or entity in seen_entities:
            continue
        seen_entities.add(entity)
        unique.append(item)

    def rank(item: dict[str, Any]) -> tuple[int, float, float]:
        keys = [str(item.get("poi_id") or ""), str(item.get("name") or "")]
        durations = [route_rank[key] for key in keys if key in route_rank]
        return (
            0 if durations else 1,
            min(durations) if durations else float("inf"),
            -float(item.get("rating") or 0),
        )

    unique.sort(key=rank)
    if not state.get("diversity_required"):
        return unique
    diverse: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for item in unique:
        category = str(item.get("category") or "unknown")
        if category not in seen_categories:
            diverse.append(item)
            seen_categories.add(category)
        else:
            deferred.append(item)
    return diverse + deferred


def _canonical_candidate_entity(name: str) -> str:
    value = re.sub(r"[（(][^）)]*[）)]", "", name).strip().lower()
    value = re.sub(
        r"(?:景区)?(?:东|西|南|北)?(?:门|入口|出口|游客中心|售票处)$",
        "",
        value,
    )
    return re.sub(r"\s+", "", value)


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
