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

from travel_agent.agent.turn_analysis import DeliveryIntent, TaskType
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
    {TaskType.FULL_ITINERARY}
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
    delivery_artifact_id: str | None = None
    rework_used: int = 0
    results: list[Any] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    map_payload: dict[str, Any] | None = None
    gate_status: str = "skipped"
    review: ReviewResult | None = None
    trace_artifact_id: str | None = None
    routing_policy_hash: str = ""
    delivery_status: str = "no_deliverable"
    review_repair_succeeded: bool = False
    review_repair_preserved: bool = False


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
            current_trace,
            reset_current_trace,
            set_current_trace,
        )

        inherited_trace = current_trace()
        owns_trace = inherited_trace is None or inherited_trace.request_id != rid
        trace = AgentTraceLog(rid) if owns_trace else inherited_trace
        token = set_current_trace(trace) if owns_trace else None
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
            if token is not None:
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
        from travel_agent.orchestration.multi_agent.trace import current_trace

        deadline = TurnDeadline.start(settings)
        trace = current_trace()
        delivery_intent = _delivery_intent_from_inputs(turn_inputs, task_type)
        if trace is not None:
            contract_detail = {
                "task_type": task_type.value if task_type is not None else None,
                "planner_required": _plan_required_for_turn(
                    self.capabilities.dispatch, task_type, delivery_intent
                ),
                "artifact_reuse_audit": list(
                    (turn_inputs or {}).get("artifact_reuse_audit") or []
                ),
            }
            if (turn_inputs or {}).get("delivery_intent"):
                contract_detail["delivery_intent"] = delivery_intent.value
            trace.append(
                "turn_contract",
                agent="engine",
                status="declared",
                detail=contract_detail,
            )
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
            outcome.delivery_artifact_id = _specialized_artifact_id_for_request(
                ctx, task_type, rid
            )
            if outcome.delivery_artifact_id:
                specialized = ctx.store.get(outcome.delivery_artifact_id) or {}
                outcome.delivery_status = (
                    "partial_specialized_with_limitations"
                    if specialized.get("limitations")
                    else "deliverable_specialized"
                )
        plan_required = _plan_required_for_turn(
            self.capabilities.dispatch, task_type, delivery_intent
        )
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
                "delivery_status": outcome.delivery_status,
                "delivery_artifact_id": (
                    outcome.delivery_artifact_id or outcome.plan_artifact_id
                    if outcome.delivery_status
                    in {
                        "deliverable_current",
                        "partial_current_with_limitations",
                        "deliverable_specialized",
                        "partial_specialized_with_limitations",
                    }
                    else None
                ),
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
        from travel_agent.artifact_policy import constraint_version

        token = set_current_task_meta({
            "request_id": request_id, "task_id": task_id, "agent": "single_agent",
            "constraint_version": constraint_version(ctx.profile),
        })
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

        if task_type == TaskType.ITINERARY_PATCH and not existing_plan_artifact_id:
            return TurnOutcome(
                status=STATUS_CLARIFICATION_REQUIRED,
                reply="当前会话没有可修改的既有行程，请先生成或提供一份行程。",
            )
        fixed_inputs = _fixed_turn_inputs(
            task_type, existing_plan_artifact_id, turn_inputs
        )
        fixed_inputs["task_brief"] = task_brief
        status, results = run_fixed_dispatch(
            runner,
            request_id,
            task_type,
            task_brief=task_brief,
            inputs=fixed_inputs,
            deadline=deadline,
        )
        if status == STATUS_CLARIFICATION_REQUIRED:
            return TurnOutcome(
                status=STATUS_CLARIFICATION_REQUIRED,
                reply="任务类型不明或信息不足，需要先补充目的地/天数等关键信息。",
            )
        missing_hard, _missing_soft = _required_evidence(
            ctx, task_type, results, fixed_inputs
        )
        delivery = _fixed_delivery_status(results)
        reply = _materialize_lightweight_artifact(
            ctx, task_type, results, missing_hard, request_id=request_id,
            existing_plan_artifact_id=existing_plan_artifact_id,
            effective_state=(fixed_inputs.get("profile") or {}).get("constraint_state"),
        ) if task_type != TaskType.FULL_ITINERARY else _compose_fixed_reply(results)
        if missing_hard and task_type != TaskType.FULL_ITINERARY:
            delivery = STATUS_INCOMPLETE
        return TurnOutcome(status=delivery, reply=reply, results=results)

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
        base_inputs["task_brief"] = task_brief or user_message
        base_inputs["execution_budget"] = deadline.trace_detail()
        from travel_agent.orchestration.multi_agent.dispatch_rules import agents_required_for_turn

        allowed_agents = set(
            agents_required_for_turn(task_type, task_brief=task_brief, inputs=base_inputs)
        )
        allowed_agents.discard("planner")
        from travel_agent.candidate_comparison import comparison_candidates

        comparison_state = dict(
            (base_inputs.get("profile") or {}).get("constraint_state") or {}
        )
        if task_type == TaskType.CANDIDATE_COMPARISON and (
            comparison_candidates(comparison_state)
            or comparison_state.get("specific_restaurant_recommendation")
            or comparison_state.get("compare_lodging_areas")
        ):
            return _run_candidate_comparison_base(
                ctx,
                runner,
                deadline,
                request_id,
                task_brief or user_message,
                base_inputs,
                allowed_agents,
                trace=current_trace(),
                routing_policy_hash=policy_hash,
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
            if deadline.usable_preplanner_time() < 2 * deadline.config.recovery_worker_useful:
                max_tasks = min(max_tasks, 1)
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
            # A named route endpoint may be ambiguous, but resolving that
            # ambiguity is tool work when both route anchors are already in
            # state.  Do not let a stochastic Router send the task back to the
            # user before the deterministic transport/POI evidence fallback
            # has had a chance to run.
            anchored_route_request = bool(
                task_type == TaskType.ROUTE_PLAN
                and _route_request_has_anchored_endpoints(base_inputs)
            )
            full_plan_ready_for_tool_work = bool(
                task_type == TaskType.FULL_ITINERARY
                and _full_itinerary_required_slots_ready(ctx, base_inputs)
            )
            if (
                decision.clarification
                and not results
                and not anchored_route_request
                and not full_plan_ready_for_tool_work
            ):
                return DynamicBaseOutcome(
                    status=STATUS_CLARIFICATION_REQUIRED,
                    reply=(
                        decision.reply
                        or "当前信息不足以安全执行，请补充具体的旅行目标或相关条件。"
                    ),
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
            # Router 返回的 decision.tasks 再次校验，并转换成真正可以交给 SubagentRunner 的 SubagentTask
            for routed in decision.tasks:
                if routed.agent not in allowed_agents:
                    continue
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
            fallback_dispatches = _missing_evidence_dispatches(
                missing_hard,
                allowed_agents=allowed_agents,
                excluded_agents={task.agent for task in tasks},
            )
            for agent, labels in fallback_dispatches:
                if len(tasks) >= max_tasks:
                    break
                objective = f"补齐硬证据：{'、'.join(labels)}"
                limit_error = ledger.authorize(agent, objective)
                if limit_error:
                    continue
                objective_key = f"{agent}:{objective.casefold()}"
                attempted.append(objective_key)
                inputs = dict(base_inputs)
                artifact_ids = list(
                    dict.fromkeys(
                        [
                            str(value)
                            for value in (inputs.get("artifact_ids") or [])
                            if value
                        ]
                        + _all_evidence_ids(results)
                    )
                )
                if artifact_ids:
                    inputs["artifact_ids"] = artifact_ids
                task = SubagentTask(
                    request_id=request_id,
                    task_id=new_task_id(agent),
                    agent=agent,
                    instruction=(
                        f"只补齐当前缺失的硬证据：{'、'.join(labels)}。"
                        "必须使用本领域真实工具并返回结构化 artifact；空结果不得算完成。"
                    ),
                    inputs=inputs,
                    attempt=ledger.counts.get(agent, 1),
                )
                tasks.append(task)
                objective_by_task[task.task_id] = objective_key
                if trace is not None:
                    trace.append(
                        "deterministic_dispatch_postcondition",
                        agent="engine",
                        status="applied",
                        attempt=wave,
                        detail={
                            "target_agent": agent,
                            "missing_evidence": labels,
                            "planner_dispatched": False,
                        },
                    )
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
                                "error": result.error,
                            }
                            for result in last_wave_results
                        ],
                        "effective_timeout_ms": int(worker_timeout * 1000),
                        "admission": worker_admission,
                        "failed_components": [
                            result.agent for result in last_wave_results
                            if result.status not in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}
                        ],
                        **deadline.trace_detail(),
                    },
                )

        missing_hard, missing_soft = _required_evidence(
            ctx, task_type, results, base_inputs
        )
        # 是否需要planner
        if _task_needs_plan(task_type):
            # Route evidence remains a hard routing target so recovery waves
            # still dispatch transport. Once all waves are exhausted, the
            # planner's own estimator plus final critic can safely recover it.
            # 没有真实路线证据，Planner 本地估算恢复，标记告警，建议出发前用地图复核
            # 但其他关键证据不能这样放宽
            if task_type == TaskType.FULL_TRIP_PLAN:
                missing_hard = [
                    item for item in missing_hard if item != "路线可行性"
                ]
            """
            所有 Wave 完成、并且移除可由 Planner 恢复的“路线可行性”后，如果仍然缺少其他硬证据，就停止执行，不进入 Planner
            """
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
            # 在启动 Planner 前，收集所有允许绑定给 Planner 的 Artifact ID
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
            # 创建plnner的任务
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

        reply = _materialize_lightweight_artifact(
            ctx, task_type, results, missing_hard, request_id=request_id,
            existing_plan_artifact_id=existing_plan_artifact_id,
            effective_state=(base_inputs.get("profile") or {}).get("constraint_state"),
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
        allowed_agents = (
            frozenset({"single_agent"})
            if self.capabilities.mode == "single"
            else frozenset({"planner"})
        )
        if requires_semantic_review(self.capabilities, task_type):
            delivery = self._run_review_cycle(
                ctx, settings, outcome, task_brief, request_id, deadline
            )

        if delivery in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
            review_notes = _review_limitations(outcome.review)
            limitations = review_notes if outcome.review_repair_preserved else []
            review_advisories = (
                review_notes
                if review_notes
                and not outcome.review_repair_preserved
                and not outcome.review_repair_succeeded
                else []
            )
            resolved_review_issues = (
                review_notes if outcome.review_repair_succeeded else []
            )
            if not _promote_selected_candidate(
                ctx,
                outcome.plan_artifact_id,
                limitations=limitations,
                review_advisories=review_advisories,
                resolved_review_issues=resolved_review_issues,
                allowed_agents=allowed_agents,
                reason=(
                    "semantic_review_warning"
                    if delivery == STATUS_COMPLETED_WITH_WARNINGS
                    else "semantic_review_pass"
                    if requires_semantic_review(self.capabilities, task_type)
                    else "deterministic_finalizer_pass"
                ),
            ):
                delivery = STATUS_INCOMPLETE
                _mark_rebuild_pending(ctx)
        elif outcome.plan_artifact_id:
            ctx.store.reject_itinerary_candidate(
                outcome.plan_artifact_id,
                reason="finalizer_delivery_blocked",
            )
            _mark_rebuild_pending(ctx)

        render_admission = deadline.trace_detail()
        render_started = time.monotonic()
        render = render_plan_outcome(
            ctx, outcome.plan_artifact_id, delivery, allowed_agents=allowed_agents
        )
        outcome.delivery_status = str(render.get("delivery_status") or "no_deliverable")
        if render.get("artifact_id"):
            outcome.plan_artifact_id = str(render["artifact_id"])
        outcome.cards = render.get("cards") or []
        outcome.map_payload = render.get("map_payload")
        outcome.gate_status = render.get("gate_status") or "skipped"
        if outcome.gate_status == "rejected":
            outcome.status = STATUS_INCOMPLETE
        elif (
            outcome.gate_status == "rendered_incomplete"
            and outcome.delivery_status == "partial_current_with_limitations"
            and render.get("partial_safe") is True
        ):
            outcome.status = STATUS_COMPLETED_WITH_WARNINGS
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
                    "delivery_status": outcome.delivery_status,
                    "artifact_id": outcome.plan_artifact_id,
                    "route_evidence_status": render.get("route_evidence_status"),
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
        critical_route_targets = _repairable_critical_route_targets(
            review, plan_payload
        )
        if (
            not start_repair
            and critical_route_targets
            and outcome.rework_used < self.capabilities.max_rework
        ):
            # A deterministic return-route gate has already identified exact
            # endpoint IDs.  One Transport retry can therefore repair the
            # evidence gap without guessing or weakening the hard constraint;
            # the repaired candidate must still pass every deterministic gate.
            delivery = STATUS_INCOMPLETE
            start_repair = True
        if start_repair:
            original_candidate_id = outcome.plan_artifact_id
            targets = critical_route_targets or repair_targets(review)
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
                return _preserve_original_candidate(
                    ctx,
                    outcome,
                    original_candidate_id,
                    review,
                    reason="repair_admission_denied",
                )
            # 整个修复周期只计一次 rework；修复后不再进行第二轮 Reviewer。
            delivery = self._run_repair_cycle(
                ctx, runner, request_id, review, outcome, targets, deadline
            )
            outcome.rework_used += 1
            repaired_candidate_id = outcome.plan_artifact_id
            if delivery not in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}:
                if repaired_candidate_id and repaired_candidate_id != original_candidate_id:
                    ctx.store.reject_itinerary_candidate(
                        repaired_candidate_id,
                        reason="targeted_rework_failed",
                    )
                return _preserve_original_candidate(
                    ctx,
                    outcome,
                    original_candidate_id,
                    review,
                    reason="targeted_rework_failed",
                )
            if _candidate_quality_worse(ctx, repaired_candidate_id, original_candidate_id):
                if repaired_candidate_id:
                    ctx.store.reject_itinerary_candidate(
                        repaired_candidate_id,
                        reason="targeted_rework_quality_regression",
                    )
                return _preserve_original_candidate(
                    ctx,
                    outcome,
                    original_candidate_id,
                    review,
                    reason="targeted_rework_quality_regression",
                )
            outcome.review_repair_succeeded = True
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
        parent_plan_artifact_id = outcome.plan_artifact_id
        planner_input_ids = list(previous_plan.get("source_artifact_ids") or [])
        if not domain_targets and outcome.plan_artifact_id and not planner_input_ids:
            planner_input_ids.append(outcome.plan_artifact_id)
        for agent in domain_targets:
            repair_inputs: dict[str, Any] = {
                "artifact_ids": list(planner_input_ids)
            }
            if agent == "transport":
                route_pairs = _repair_route_pairs_for_plan(previous_plan)
                if route_pairs:
                    repair_inputs["repair_route_pairs"] = route_pairs
            task = SubagentTask(
                request_id=request_id,
                task_id=new_task_id(agent),
                agent=agent,
                instruction=instructions.get(agent) or "按 Reviewer 意见定向补充本领域结果。",
                inputs=repair_inputs,
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

        repair_issues = [
            issue for issue in review.issues
            if issue.severity == SEVERITY_RECOVERABLE
            or _is_critical_route_issue(issue)
        ]
        planner_review_instructions = "\n".join(
            issue.repair_instruction
            for issue in repair_issues
            if issue.repair_instruction
        )
        planner_review_issue_types = [
            issue.issue_type
            for issue in repair_issues
        ]
        return_anchor = (previous_plan.get("required_route_anchors") or {}).get(
            "last_stop_to_return_location"
        ) or {}
        preserve_terminal_poi_id = ""
        preserve_terminal_poi: dict[str, Any] | None = None
        if any(
            "return" in str(issue_type or "").lower()
            or "deadline" in str(issue_type or "").lower()
            for issue_type in planner_review_issue_types
        ):
            preserve_terminal_poi_id = str(
                return_anchor.get("origin_poi_id") or ""
            ).strip()
            preserve_terminal_poi = next((
                stop.get("poi")
                for day in (previous_plan.get("itinerary") or {}).get("days") or []
                for stop in day.get("stops") or []
                if isinstance(stop, dict)
                and str((stop.get("poi") or {}).get("poi_id") or "")
                == preserve_terminal_poi_id
            ), None)
        if (
            preserve_terminal_poi_id
            and parent_plan_artifact_id
            and parent_plan_artifact_id not in planner_input_ids
        ):
            # A route-only revision needs the exact reviewed terminal POI from
            # its parent candidate. The final artifact removes itinerary
            # inputs from source evidence and records only lineage, so this
            # does not promote the failed parent or bypass stale-state gates.
            planner_input_ids.append(parent_plan_artifact_id)

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
                    "parent_plan_artifact_id": parent_plan_artifact_id,
                    "repair_targets": list(targets),
                    "preserve_terminal_poi_id": preserve_terminal_poi_id or None,
                    "preserve_terminal_poi": preserve_terminal_poi,
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

            gate_error, _record = _validate_plan_gate(
                ctx, outcome.plan_artifact_id, allow_candidate=True
            )
            repaired = ctx.store.get(outcome.plan_artifact_id) or {}
            critic = repaired.get("critic") or {}
            critical_issues = [
                issue
                for issue in (critic.get("issues") or [])
                if str(issue.get("severity") or "").lower() in {"error", "critical"}
            ]
            unresolved_repairs = _unresolved_review_repairs(
                ctx,
                review,
                previous_plan,
                repaired,
            )
            if (
                gate_error
                or critic.get("passed") is not True
                or critical_issues
                or planner_result.unresolved
                or unresolved_repairs
                or (
                    isinstance(repaired.get("validation_result"), dict)
                    and (
                        repaired["validation_result"].get("passed") is not True
                        or repaired.get("parent_plan_artifact_id") != parent_plan_artifact_id
                        or list(repaired.get("unresolved_changes") or [])
                    )
                )
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


# --- Finalizer candidate selection ----------------------------------------- #


def _review_limitations(review: ReviewResult | None) -> list[str]:
    if review is None:
        return []
    return list(
        dict.fromkeys(
            issue.description.strip()
            for issue in review.issues
            if issue.description.strip()
        )
    )


def _candidate_deterministically_valid(
    ctx: Any,
    artifact_id: str | None,
    *,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
) -> bool:
    if not artifact_id:
        return False
    from travel_agent.agent.toolkit import _validate_plan_gate

    error, record = _validate_plan_gate(
        ctx,
        artifact_id,
        allowed_agents=allowed_agents,
        allow_candidate=True,
        allow_earlier_attempt=True,
    )
    if error or not record:
        return False
    payload = record.get("payload") or {}
    critic = payload.get("critic") or {}
    validation = payload.get("validation_result") or {}
    if record.get("artifact_status") == "current":
        return critic.get("passed") is True
    return bool(
        critic.get("passed") is True
        and validation.get("passed") is True
        and not payload.get("unresolved_changes")
    )


def _promote_selected_candidate(
    ctx: Any,
    artifact_id: str | None,
    *,
    limitations: list[str],
    review_advisories: list[str] | None = None,
    resolved_review_issues: list[str] | None = None,
    reason: str,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
) -> bool:
    if not artifact_id or not _candidate_deterministically_valid(
        ctx, artifact_id, allowed_agents=allowed_agents
    ):
        return False
    promoted = ctx.store.promote_itinerary(
        artifact_id,
        limitations=limitations,
        review_advisories=review_advisories,
        resolved_review_issues=resolved_review_issues,
        promotion_reason=reason,
    )
    if promoted:
        state = getattr(ctx.profile, "constraint_state", {}) or {}
        state["_plan_status"] = "current"
        state["_revisable_parent_plan_artifact_id"] = artifact_id
    return promoted


def _mark_rebuild_pending(ctx: Any) -> None:
    state = getattr(ctx.profile, "constraint_state", None)
    if not isinstance(state, dict):
        state = {}
        ctx.profile.constraint_state = state
    state["_plan_status"] = "rebuild_pending"


def _preserve_original_candidate(
    ctx: Any,
    outcome: TurnOutcome,
    original_candidate_id: str | None,
    review: ReviewResult,
    *,
    reason: str,
) -> str:
    """Rollback selection to the same-revision candidate after failed rework."""
    if not _candidate_deterministically_valid(ctx, original_candidate_id):
        return STATUS_INCOMPLETE
    outcome.plan_artifact_id = original_candidate_id
    outcome.review_repair_preserved = True
    from travel_agent.orchestration.multi_agent.trace import current_trace

    trace = current_trace()
    if trace is not None:
        trace.append(
            "candidate_preservation",
            agent="finalizer",
            status=STATUS_COMPLETED_WITH_WARNINGS,
            detail={
                "artifact_id": original_candidate_id,
                "reason": reason,
                "same_revision_preserved": True,
            },
        )
    return STATUS_COMPLETED_WITH_WARNINGS


def _candidate_quality_worse(
    ctx: Any,
    repaired_id: str | None,
    original_id: str | None,
) -> bool:
    if not _candidate_deterministically_valid(ctx, repaired_id):
        return True
    if not original_id or not repaired_id:
        return False
    original = ctx.store.get(original_id) or {}
    repaired = ctx.store.get(repaired_id) or {}

    def issue_score(payload: dict[str, Any]) -> tuple[int, int]:
        validation_issues = len((payload.get("validation_result") or {}).get("issues") or [])
        critic_issues = len((payload.get("critic") or {}).get("issues") or [])
        return validation_issues, critic_issues

    return issue_score(repaired) > issue_score(original)


def _unresolved_review_repairs(
    ctx: Any,
    review: ReviewResult,
    parent: dict[str, Any],
    repaired: dict[str, Any],
) -> list[str]:
    """Verify recoverable Reviewer requests from artifact deltas, not prose.

    Deterministic hard gates still run separately.  This check only proves that
    the targeted repair made a relevant, measurable change before the new
    candidate can replace its parent atomically.
    """
    unresolved: list[str] = []
    for issue in review.issues:
        if issue.severity != SEVERITY_RECOVERABLE and not _is_critical_route_issue(issue):
            continue
        if not _review_issue_repair_resolved(
            ctx, str(issue.issue_type or ""), parent, repaired
        ):
            unresolved.append(issue.description or issue.issue_type)
    return unresolved


_REPAIRABLE_CRITICAL_ROUTE_CODES = frozenset({
    "return_route_missing",
    "return_route_evidence_insufficient",
})


def _repair_route_pairs_for_plan(plan: dict[str, Any]) -> list[list[str]]:
    """Bind repair evidence for exact anchors and scheduled adjacent legs.

    A return-deadline closure may remove a late optional final stop.  Include
    every stop on the last scheduled day against the same return endpoint so
    the next actual terminal is not left with stale or missing evidence.  The
    Reviewer can also identify an unverified route between two already chosen
    stops.  Include each same-day adjacent pair so the one bounded Transport
    repair can materialize those exact legs rather than re-searching or
    guessing endpoints.  Never infer a route across the overnight day break.
    """
    anchors = plan.get("required_route_anchors") or {}
    return_leg = anchors.get("last_stop_to_return_location") or {}
    legs = [
        return_leg,
        anchors.get("fixed_event_transfer"),
        *(anchors.get("legs") or []),
    ]
    pairs: list[list[str]] = []

    def add(origin: Any, destination: Any) -> None:
        origin_id = str(origin or "").strip()
        destination_id = str(destination or "").strip()
        pair = [origin_id, destination_id]
        if (
            origin_id
            and destination_id
            and origin_id != destination_id
            and pair not in pairs
        ):
            pairs.append(pair)

    for leg in legs:
        if not isinstance(leg, dict):
            continue
        add(
            leg.get("origin_poi_id"),
            leg.get("destination_poi_id") or leg.get("event_poi_id"),
        )

    return_id = str(return_leg.get("destination_poi_id") or "").strip()
    scheduled_days = [
        day
        for day in (plan.get("itinerary") or {}).get("days") or []
        if isinstance(day, dict) and day.get("stops")
    ]
    if return_id and scheduled_days:
        for stop in reversed(scheduled_days[-1].get("stops") or []):
            if isinstance(stop, dict):
                add((stop.get("poi") or {}).get("poi_id"), return_id)

    for day in scheduled_days:
        stops = [stop for stop in day.get("stops") or [] if isinstance(stop, dict)]
        for previous, current in zip(stops, stops[1:]):
            add(
                (previous.get("poi") or {}).get("poi_id"),
                (current.get("poi") or {}).get("poi_id"),
            )
    return pairs[:8]


def _is_critical_route_issue(issue: Any) -> bool:
    kind = str(getattr(issue, "issue_type", "") or "").strip().lower()
    return bool(
        getattr(issue, "severity", "") == "critical"
        and getattr(issue, "repair_target", "") in {"transport", "planner"}
        and getattr(issue, "repair_instruction", "")
        and any(token in kind for token in ("return_route", "return", "deadline"))
    )


def _repairable_critical_route_targets(
    review: ReviewResult,
    plan: dict[str, Any],
) -> list[str]:
    """Allow one exact-endpoint Transport retry for a proven return gap only."""
    critical = list(review.critical_issues())
    if not critical or not all(_is_critical_route_issue(issue) for issue in critical):
        return []
    error_codes = {
        str(item.get("code") or "")
        for section in (plan.get("validation_result") or {}, plan.get("critic") or {})
        for item in section.get("issues") or []
        if isinstance(item, dict)
        and str(item.get("severity") or "error").lower() in {"error", "critical"}
        and str(item.get("code") or "")
    }
    if not error_codes or not error_codes.issubset(_REPAIRABLE_CRITICAL_ROUTE_CODES):
        return []
    anchor = (plan.get("required_route_anchors") or {}).get(
        "last_stop_to_return_location"
    ) or {}
    if not (
        anchor.get("origin_poi_id")
        and anchor.get("destination_poi_id")
        and str(anchor.get("evidence_status") or "") != "provider_verified"
    ):
        return []
    return ["transport"]


def _review_issue_repair_resolved(
    ctx: Any,
    issue_type: str,
    parent: dict[str, Any],
    repaired: dict[str, Any],
) -> bool:
    kind = issue_type.strip().lower()
    parent_itinerary = parent.get("itinerary") or {}
    repaired_itinerary = repaired.get("itinerary") or {}

    if any(token in kind for token in ("transport", "route", "walking")):
        return _route_repair_score(repaired) > _route_repair_score(parent)

    if "return" in kind or "deadline" in kind:
        return _return_repair_score(repaired) > _return_repair_score(parent)

    if "lodging" in kind or "hotel" in kind:
        return _lodging_repair_score(repaired) > _lodging_repair_score(parent)

    if "budget" in kind:
        return _budget_repair_score(repaired) > _budget_repair_score(parent)

    if any(token in kind for token in ("accessibility", "mobility", "elderly")):
        return _accessibility_repair_score(repaired) > _accessibility_repair_score(parent)

    if "must_visit" in kind:
        from travel_agent.plan_invariants import validate_plan_artifact

        validation = validate_plan_artifact(repaired, ctx.profile)
        return validation.get("passed") is True and not any(
            "must_visit" in str(item.get("code") or "")
            for item in validation.get("issues") or []
            if isinstance(item, dict)
        )

    if any(token in kind for token in ("interest", "category", "activity_sparsity")):
        return _activity_repair_score(repaired_itinerary) > _activity_repair_score(
            parent_itinerary
        )

    if any(token in kind for token in ("schedule", "pace", "meal")):
        return _schedule_signature(repaired_itinerary) != _schedule_signature(
            parent_itinerary
        )

    return repaired_itinerary != parent_itinerary


def _iter_plan_routes(payload: dict[str, Any]):
    for day in (payload.get("itinerary") or {}).get("days") or []:
        for stop in day.get("stops") or []:
            route = stop.get("route_from_previous")
            if isinstance(route, dict) and route:
                yield route
    anchors = payload.get("required_route_anchors") or {}
    for leg in anchors.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        if isinstance(leg.get("route"), dict):
            yield leg["route"]
        for route in leg.get("routes") or []:
            if isinstance(route, dict):
                yield route


def _route_repair_score(payload: dict[str, Any]) -> tuple[int, int, int]:
    routes = list(_iter_plan_routes(payload))
    provider = sum(
        str(route.get("evidence_status") or "") == "provider_verified"
        or str(route.get("source") or "").lower() in {"amap", "provider"}
        for route in routes
    )
    walk_known = sum(route.get("walking_distance_km") is not None for route in routes)
    return provider, walk_known, len(routes)


def _return_repair_score(payload: dict[str, Any]) -> tuple[int, int]:
    plan = payload.get("return_plan") or {}
    segment = plan.get("intercity_segment") or {}
    verified = int(
        str(plan.get("status") or "").lower() in {"verified", "provider_verified"}
        or str(segment.get("status") or "").lower() == "verified_route"
    )
    evidence = int(bool(segment.get("route_evidence") or plan.get("route_evidence")))
    return verified, evidence


def _lodging_repair_score(payload: dict[str, Any]) -> tuple[int, float]:
    plan = payload.get("lodging_plan") or {}
    status = str(plan.get("status") or "").lower()
    verified = int(status not in {"", "evidence_unavailable", "unavailable", "missing"})
    anchors = payload.get("lodging_route_anchors") or {}
    distance = anchors.get("average_outbound_distance_km")
    proximity = -float(distance) if distance is not None else float("-inf")
    return verified, proximity


def _budget_repair_score(payload: dict[str, Any]) -> tuple[int, float]:
    plan = payload.get("budget_plan") or {}
    within = int(plan.get("within_user_limit") is True)
    total = plan.get("expected_total")
    affordability = -float(total) if total is not None else float("-inf")
    return within, affordability


def _accessibility_repair_score(payload: dict[str, Any]) -> tuple[int, int]:
    verification = payload.get("candidate_verification") or {}
    mobility = payload.get("mobility_plan") or {}
    text = json.dumps([verification, mobility], ensure_ascii=False).lower()
    evidence_terms = sum(
        text.count(token)
        for token in ("verified", "wheelchair", "accessible", "无障碍", "电梯")
    )
    unknown_terms = text.count("unknown") + text.count("unavailable")
    return evidence_terms, -unknown_terms


def _activity_repair_score(itinerary: dict[str, Any]) -> tuple[int, int]:
    stops = [
        stop
        for day in itinerary.get("days") or []
        for stop in day.get("stops") or []
        if isinstance(stop, dict)
    ]
    categories = {
        str((stop.get("poi") or {}).get("category") or "").strip().lower()
        for stop in stops
        if str((stop.get("poi") or {}).get("category") or "").strip()
    }
    return len(stops), len(categories)


def _schedule_signature(itinerary: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            day.get("day_index"),
            tuple(
                (
                    (stop.get("poi") or {}).get("poi_id"),
                    stop.get("start_time"),
                    stop.get("duration_min"),
                )
                for stop in day.get("stops") or []
                if isinstance(stop, dict)
            ),
        )
        for day in itinerary.get("days") or []
        if isinstance(day, dict)
    )


# --- 辅助 ------------------------------------------------------------------- #


def _run_candidate_comparison_base(
    ctx: Any,
    runner: Any,
    deadline: Any,
    request_id: str,
    task_brief: str,
    base_inputs: dict[str, Any],
    allowed_agents: set[str],
    *,
    trace: Any,
    routing_policy_hash: str,
) -> DynamicBaseOutcome:
    """Execute the deterministic evidence contract for candidate comparison.

    Candidate and dimension extraction is already complete before this point.
    A model Router may not declare the turn ready without evidence, so this
    specialized production path dispatches only the workers required by the
    declared dimensions.  Planner is structurally absent.
    """
    from travel_agent.candidate_comparison import (
        collect_comparison_evidence,
        comparison_candidates,
        comparison_dimensions,
    )

    state = dict((base_inputs.get("profile") or {}).get("constraint_state") or {})
    # Accessibility is relative to a target. Normalize the common local-area
    # wording before dispatch as well as during materialization, otherwise the
    # evidence gate can retain a stale unanchored limitation after route
    # evidence has already covered every candidate.
    if not state.get("target_anchor") and (
        state.get("walking_time_max_min") is not None
        or state.get("location_anchor")
    ):
        state["target_anchor"] = state.get("location_anchor") or state.get("location")
        profile_inputs = dict(base_inputs.get("profile") or {})
        profile_inputs["constraint_state"] = state
        base_inputs = {**base_inputs, "profile": profile_inputs}
    candidates = comparison_candidates(state)
    dimensions = comparison_dimensions(state)
    ordered_agents = [
        agent
        for agent in ("attraction", "hotel", "restaurant", "transport")
        if agent in allowed_agents
    ]
    results: list[SubagentResult] = []
    # Recommendation requests such as "find restaurants near X" discover
    # their candidates from a worker rather than declaring names up front.
    # Run exactly one bounded discovery worker first, then bind route/cost
    # evidence to the discovered entities in the normal candidate matrix.
    if not candidates:
        discovery_agent = next(
            (
                agent
                for agent in ("restaurant", "hotel", "attraction")
                if agent in ordered_agents
            ),
            None,
        )
        if discovery_agent is not None:
            timeout = deadline.effective_preplanner_timeout(
                _component_timeout(discovery_agent)
            )
            if timeout > 0:
                discovery_task = SubagentTask(
                    request_id=request_id,
                    task_id=new_task_id(discovery_agent),
                    agent=discovery_agent,
                    instruction=(
                        "发现满足当前地点、预算、场景和硬约束的候选；"
                        "必须使用本领域真实工具并返回结构化 artifact。"
                    ),
                    inputs=dict(base_inputs),
                )
                discovery = _run_with_optional_timeout(
                    runner, discovery_task, timeout
                )
                results.append(discovery)
                records: list[tuple[str, str, dict[str, Any]]] = []
                for item in discovery.evidence:
                    artifact_id = str(item.get("artifact_id") or "")
                    record = ctx.store.get_record(artifact_id) or {}
                    payload = record.get("payload")
                    if artifact_id and isinstance(payload, dict):
                        records.append(
                            (artifact_id, str(record.get("kind") or ""), payload)
                        )
                candidates = _discovered_candidate_names(records, state)
                if state.get("top_n") is not None:
                    try:
                        candidates = candidates[: max(1, int(state["top_n"]))]
                    except (TypeError, ValueError):
                        pass
                if candidates:
                    state["comparison_candidates"] = candidates
                    profile_inputs = dict(base_inputs.get("profile") or {})
                    profile_inputs["constraint_state"] = state
                    base_inputs = {**base_inputs, "profile": profile_inputs}
    # One bounded task owns one candidate and one worker-specific set of
    # dimensions.  After every result, recompute the shared coverage matrix
    # and skip any cell already covered by grounded evidence, regardless of
    # which eligible evidence worker produced it.
    dimension_owner: dict[str, str] = {}
    for dimension in dimensions:
        preferred = (
            ("transport", "attraction", "hotel", "restaurant")
            if dimension in {"accessibility", "accessibility_needs"}
            else ("hotel", "restaurant", "attraction", "transport")
            if dimension == "cost"
            else ("attraction", "hotel", "restaurant", "transport")
        )
        owner = next((agent for agent in preferred if agent in ordered_agents), None)
        if owner:
            dimension_owner[dimension] = owner
    dimensions_by_agent = {
        agent: [dimension for dimension in dimensions if dimension_owner.get(dimension) == agent]
        for agent in ordered_agents
    }
    max_tasks = max(1, min(16, len(candidates) * max(1, len(ordered_agents))))
    dispatched = len(results)
    # Non-transport workers run first so endpoint/area artifacts can be bound
    # into the transport task when a fixed anchor comparison needs routes.
    stages = [
        [agent for agent in ordered_agents if agent != "transport"],
        ["transport"] if "transport" in ordered_agents else [],
    ]
    for stage_index, agents in enumerate(stages, start=1):
        if not agents:
            continue
        for agent in agents:
            owned_dimensions = dimensions_by_agent.get(agent) or []
            if not owned_dimensions:
                continue
            for candidate_index, candidate in enumerate(candidates):
                current = collect_comparison_evidence(
                    ctx.store, results, state, request_id=request_id
                )
                covered_cells = {
                    (item["candidate"], item["dimension"])
                    for item in current.evidence
                }
                missing_dimensions = [
                    dimension
                    for dimension in owned_dimensions
                    if (candidate, dimension) not in covered_cells
                ]
                if not missing_dimensions:
                    continue
                if dispatched >= max_tasks:
                    break
                timeout = deadline.effective_preplanner_timeout(_component_timeout(agent))
                if timeout <= 0:
                    break
                inputs = dict(base_inputs)
                artifact_ids = list(dict.fromkeys(
                    [str(value) for value in (inputs.get("artifact_ids") or []) if value]
                    + _all_evidence_ids(results)
                ))
                if artifact_ids:
                    inputs["artifact_ids"] = artifact_ids
                inputs["comparison_candidate"] = candidate
                inputs["comparison_dimensions"] = missing_dimensions
                inputs["comparison_finalize_routes"] = not any(
                    (later_candidate, dimension) not in covered_cells
                    for later_candidate in candidates[candidate_index + 1:]
                    for dimension in owned_dimensions
                )
                task = SubagentTask(
                    request_id=request_id,
                    task_id=new_task_id(agent),
                    agent=agent,
                    instruction=(
                        f"为 Candidate Comparison 独立调研候选「{candidate}」；"
                        f"只补缺失维度 {missing_dimensions}。每条证据必须来自本任务真实工具 artifact，"
                        "并绑定到该候选；空结果、无关实体或模型文字不能算证据。"
                        + (
                            " 用户明确不需要具体酒店库存，不得推荐或检索具体酒店库存。"
                            if state.get("no_live_inventory_required")
                            else ""
                        )
                    ),
                    inputs=inputs,
                )
                result = _run_with_optional_timeout(runner, task, timeout)
                result.payload = dict(result.payload or {})
                result.payload["comparison_scope"] = {
                    "candidate": candidate,
                    "dimensions": missing_dimensions,
                }
                results.append(result)
                dispatched += 1
                if trace is not None:
                    trace.append(
                        "comparison_dispatch",
                        agent="engine",
                        status="completed",
                        attempt=stage_index,
                        detail={
                            "candidate": candidate,
                            "dimensions": missing_dimensions,
                            "agent": agent,
                            "planner_dispatched": False,
                            "result": {"agent": result.agent, "status": result.status},
                            "dispatch_count": dispatched,
                            "dispatch_limit": max_tasks,
                        },
                    )
            if dispatched >= max_tasks:
                break

    missing, _soft = _required_evidence(ctx, TaskType.CANDIDATE_COMPARISON, results, base_inputs)
    reply = _materialize_lightweight_artifact(
        ctx,
        TaskType.CANDIDATE_COMPARISON,
        results,
        missing,
        request_id=request_id,
        effective_state=(base_inputs.get("profile") or {}).get("constraint_state"),
    )
    return DynamicBaseOutcome(
        status=STATUS_COMPLETED if not missing else STATUS_INCOMPLETE,
        reply=reply,
        results=tuple(results),
        routing_policy_hash=routing_policy_hash,
    )


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
            for item in result.evidence
            if isinstance(item, dict)
            and item.get("artifact_id")
            and (
                result.status in {STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS}
                # A deterministic budget artifact remains independently usable
                # when the Transport worker later exhausts its route/search
                # steps.  This does not promote that worker or any candidate;
                # route readiness is still evaluated separately.
                or str(item.get("kind") or "") == "budget"
            )
            for artifact_id in [str(item["artifact_id"])]
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


def _route_request_has_anchored_endpoints(base_inputs: dict[str, Any]) -> bool:
    profile = base_inputs.get("profile") or {}
    state = profile.get("constraint_state") or {}
    origin = state.get("origin") or state.get("start_location")
    destination = next(
        (
            state.get(key)
            for key in (
                "destination_name", "destination_area", "destination",
                "return_location", "target_anchor", "location_anchor",
            )
            if state.get(key) not in (None, "", [], {})
        ),
        None,
    )
    return bool(origin and destination)


def _full_itinerary_required_slots_ready(ctx, base_inputs: dict[str, Any]) -> bool:
    """Ignore Router requests for optional preferences after slot gating passed."""
    profile = dict(base_inputs.get("profile") or {})
    state = dict(profile.get("constraint_state") or {})
    destination = (
        profile.get("destination")
        or state.get("destination_city")
        or state.get("destination")
        or (state.get("destinations") or [None])[0]
        or getattr(getattr(ctx, "profile", None), "destination", None)
    )
    days = (
        profile.get("days")
        or state.get("duration_days")
        or getattr(getattr(ctx, "profile", None), "days", None)
    )
    try:
        valid_days = int(days) > 0
    except (TypeError, ValueError):
        valid_days = False
    return bool(destination and valid_days)


def _missing_evidence_dispatches(
    missing_hard: list[str],
    *,
    allowed_agents: set[str],
    excluded_agents: set[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """Map hard-evidence labels to bounded domain workers, never Planner."""
    excluded = excluded_agents or set()
    grouped: dict[str, list[str]] = {}
    for label in missing_hard:
        text = str(label)
        if any(marker in text for marker in ("餐厅", "用餐", "饮食")):
            agent = "restaurant"
        elif any(marker in text for marker in ("住宿", "酒店")):
            agent = "hotel"
        elif any(
            marker in text.casefold()
            for marker in (
                "路线",
                "预算",
                "步行",
                "距离",
                "可达",
                "交通",
                "accessibility",
            )
        ):
            agent = "transport"
        elif any(marker in text for marker in ("景点", "候选", "室内备选", "poi")):
            agent = "attraction"
        elif len(allowed_agents) == 1:
            agent = next(iter(allowed_agents))
        else:
            continue
        if agent not in allowed_agents or agent in excluded:
            continue
        grouped.setdefault(agent, []).append(text)
    return [(agent, labels) for agent, labels in grouped.items()]


def _required_evidence(
    ctx: Any,
    task_type: TaskType | None,
    results: list[SubagentResult],
    base_inputs: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return deterministic (hard_missing, soft_missing) evidence labels."""
    kinds: set[str] = set()
    nonempty_kinds: set[str] = set()

    def register_record(record: dict[str, Any] | None) -> None:
        if not record or not record.get("kind"):
            return
        kind = str(record["kind"])
        kinds.add(kind)
        payload = record.get("payload") or {}
        if kind in {"candidates", "pois", "restaurants", "hotels"}:
            collection = (
                payload.get("pois")
                if kind in {"candidates", "pois"}
                else payload.get(kind)
            ) or payload.get("items")
            if collection:
                nonempty_kinds.add(kind)
        else:
            nonempty_kinds.add(kind)

    for result in results:
        for item in result.evidence:
            if isinstance(item, dict) and item.get("kind"):
                kind = str(item["kind"])
                if (
                    result.status not in {
                        STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS,
                    }
                    and kind != "budget"
                ):
                    continue
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
    state = dict(
        (base_inputs.get("profile") or {}).get("constraint_state")
        or getattr(getattr(ctx, "profile", None), "constraint_state", {})
        or {}
    )
    hard: list[str] = []
    soft: list[str] = []

    def require(kind: str, label: str, *, hard_required: bool = True) -> None:
        if kind in nonempty_kinds:
            return
        target = hard if hard_required else soft
        if label not in target:
            target.append(label)

    if task_type == TaskType.FULL_ITINERARY:
        require("candidates", "景点候选")
        require("routes", "路线可行性")
        task_brief = str(base_inputs.get("task_brief") or "")
        profile_payload = dict(base_inputs.get("profile") or {})
        duration_value = (
            state.get("duration_days")
            or profile_payload.get("days")
            or getattr(getattr(ctx, "profile", None), "days", None)
        )
        try:
            multiday = int(duration_value) > 1
        except (TypeError, ValueError):
            multiday = False
        self_arranged_hotel = bool(
            re.search(
                r"(?:酒店|住宿).{0,8}(?:已订|订好|自己安排|自行安排)",
                task_brief,
            )
        )
        hotel_recommendation_excluded = "酒店推荐" in " ".join(
            str(item) for item in (state.get("exclude") or [])
        )
        hotel_explicit = bool(
            re.search(
                r"(?:酒店|住宿).{0,8}(?:降档|经济|便宜|省预算)",
                task_brief.lower(),
            )
            or
            any(
                state.get(key) not in (None, "", [], {})
                for key in (
                    "compare_lodging_areas",
                    "hotel_budget_per_night_cny",
                    "lodging_area",
                )
            )
        )
        lodging_evidence_required = (
            (hotel_explicit or multiday)
            and not self_arranged_hotel
            and not hotel_recommendation_excluded
        )
        require(
            "hotels",
            "住宿证据",
            hard_required=lodging_evidence_required,
        )
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
        from travel_agent.orchestration.multi_agent.dispatch_rules import (
            specific_restaurant_recommendation_requested,
        )

        food_explicit = specific_restaurant_recommendation_requested(
            str(base_inputs.get("task_brief") or ""), profile_payload, state
        )
        if food_explicit:
            require("restaurants", "具体餐厅证据")
    elif task_type == TaskType.ITINERARY_PATCH:
        require("itinerary", "既有行程")
    elif task_type == TaskType.LOCAL_ADJUSTMENT_ADVICE:
        if state.get("date_start") or state.get("resolved_date") or state.get("weather_condition"):
            require("weather", "天气")
        if state.get("need_indoor_backup"):
            require("candidates", "室内备选")
    elif task_type == TaskType.ROUTE_PLAN:
        require("routes", "路线耗时/距离")
    elif task_type == TaskType.CANDIDATE_COMPARISON:
        from travel_agent.candidate_comparison import (
            collect_comparison_evidence,
            comparison_candidates,
            comparison_hard_missing,
        )

        if comparison_candidates(state):
            current_request_id = next(
                (str(result.request_id) for result in results if result.request_id),
                None,
            )
            comparison = collect_comparison_evidence(
                ctx.store,
                results,
                state,
                request_id=current_request_id,
            )
            hard.extend(comparison_hard_missing(state, comparison))
        else:
            if not nonempty_kinds.intersection({"candidates", "restaurants", "hotels"}):
                hard.append("候选地点")
            if state.get("walking_time_max_min") is not None:
                require("routes", "步行时间")
            elif state.get("location"):
                require("routes", "与指定位置的距离")
            extra = _lightweight_missing_payload_fields(results, ctx.profile)
            for label in extra:
                if label not in hard:
                    hard.append(label)
    elif task_type is None or task_type == TaskType.CLARIFICATION:
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

    gate_error, record = _validate_plan_gate(
        ctx, plan_artifact_id, allow_candidate=True
    )
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


def _delivery_intent_from_inputs(
    turn_inputs: dict[str, Any] | None,
    task_type: TaskType | None,
) -> DeliveryIntent:
    raw = (turn_inputs or {}).get("delivery_intent")
    try:
        return raw if isinstance(raw, DeliveryIntent) else DeliveryIntent(str(raw))
    except ValueError:
        return (
            DeliveryIntent.REBUILD_NOW
            if task_type == TaskType.FULL_ITINERARY
            else DeliveryIntent.LOCAL_PATCH
            if task_type == TaskType.ITINERARY_PATCH
            else DeliveryIntent.LIGHTWEIGHT_ADVICE
        )


def _plan_required_for_turn(
    dispatch: str,
    task_type: TaskType | None,
    delivery_intent: DeliveryIntent | str | None = None,
) -> bool:
    """Planner admission is a delivery decision; task type is compatibility fallback."""
    del dispatch
    if delivery_intent is not None:
        try:
            intent = (
                delivery_intent
                if isinstance(delivery_intent, DeliveryIntent)
                else DeliveryIntent(str(delivery_intent))
            )
            return intent == DeliveryIntent.REBUILD_NOW
        except ValueError:
            pass
    return task_type == TaskType.FULL_ITINERARY


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


def _discovered_candidate_names(
    records: list[tuple[str, str, dict[str, Any]]],
    state: dict[str, Any],
) -> list[str]:
    """Promote grounded retrieval rows into a bounded candidate set.

    Discovery requests intentionally have no names before tool execution.
    Candidate identity therefore comes only from current-turn artifacts.  The
    selector performs entity-level deduplication and, when requested, spreads
    picks across semantic categories before filling remaining slots.
    """
    rows: list[dict[str, Any]] = []
    restaurant_request = bool(state.get("specific_restaurant_recommendation"))
    for _artifact_id, kind, payload in records:
        if restaurant_request and kind != "restaurants":
            continue
        if not restaurant_request and kind not in {"candidates", "pois", "hotels"}:
            continue
        keys = (
            ("restaurants", "items")
            if kind == "restaurants"
            else ("hotels", "items")
            if kind == "hotels"
            else ("pois", "items")
        )
        for key in keys:
            for item in payload.get(key) or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("canonical_name") or item.get("name") or "").strip()
                if not name:
                    continue
                verification = str(item.get("verification_status") or "verified")
                if verification not in {"", "verified"}:
                    continue
                rows.append({**item, "_candidate_name": name})

    def identity(item: dict[str, Any]) -> str:
        parent = item.get("parent_poi_id") or item.get("parent_canonical_name")
        if parent:
            return "parent:" + str(parent).strip().casefold()
        value = str(item.get("_candidate_name") or "").casefold()
        value = re.sub(r"[\s\-—–_·•:：,，。/\\()（）\[\]【】]+", "", value)
        value = re.sub(
            r"(?:[东南西北上下内外\dA-Za-z一二三四五六七八九十]+号?)?(?:门|入口|出口)$",
            "",
            value,
        )
        return value

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rows:
        key = identity(item)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    try:
        requested = int(state.get("top_n") or 0)
    except (TypeError, ValueError):
        requested = 0
    limit = max(1, min(8, requested or 5))

    def semantic_bucket(item: dict[str, Any]) -> str:
        text = " ".join(
            str(value or "")
            for value in (
                item.get("entity_type"),
                item.get("category"),
                item.get("_candidate_name"),
                " ".join(str(tag) for tag in item.get("tags") or []),
            )
        ).casefold()
        patterns = (
            ("museum", r"museum|博物馆|展览|美术馆|科技馆"),
            ("park", r"park|公园|绿地|植物园|动物园"),
            ("historic", r"historic|历史街区|古镇|古城|遗址|寺|祠"),
            ("nature", r"nature|自然|山|湖|湿地|海滩|森林"),
            ("retail", r"shopping|retail|商场|购物"),
            ("food", r"food|restaurant|餐厅|饭店|咖啡"),
        )
        return next((label for label, pattern in patterns if re.search(pattern, text)), text.split(" ", 1)[0])

    if not state.get("diversity_required"):
        selected = unique[:limit]
    else:
        selected = []
        deferred: list[dict[str, Any]] = []
        buckets: set[str] = set()
        for item in unique:
            bucket = semantic_bucket(item)
            if bucket and bucket not in buckets:
                selected.append(item)
                buckets.add(bucket)
            else:
                deferred.append(item)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            selected.extend(deferred[: limit - len(selected)])
    return [str(item["_candidate_name"]) for item in selected]


def _materialize_lightweight_artifact(
    ctx: Any,
    task_type: TaskType | None,
    results: list[SubagentResult],
    missing: list[str],
    *,
    request_id: str,
    existing_plan_artifact_id: str | None = None,
    effective_state: dict[str, Any] | None = None,
) -> str:
    """Build the requested non-itinerary deliverable from bound evidence."""
    records: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for result in results:
        for item in result.evidence:
            artifact_id = str(item.get("artifact_id") or "") if isinstance(item, dict) else ""
            if not artifact_id or artifact_id in seen:
                continue
            record = ctx.store.get_record(artifact_id) or {}
            payload = record.get("payload")
            if isinstance(payload, dict):
                records.append((artifact_id, str(record.get("kind") or item.get("kind") or ""), payload))
                seen.add(artifact_id)

    state = dict(
        effective_state
        or getattr(ctx.profile, "constraint_state", {})
        or {}
    )
    evidence = [
        {"artifact_id": artifact_id, "kind": kind}
        for artifact_id, kind, _payload in records
    ]
    limitations = list(dict.fromkeys(str(item) for item in missing if item))

    if task_type == TaskType.CANDIDATE_COMPARISON:
        from travel_agent.candidate_comparison import (
            accessibility_contract,
            accessibility_mode,
            collect_comparison_evidence,
            comparison_candidates,
            comparison_core_dimensions,
            comparison_dimensions,
            supported_recommendation,
        )

        names = comparison_candidates(state) or _discovered_candidate_names(records, state)
        comparison_state = dict(state)
        if names and not comparison_candidates(comparison_state):
            comparison_state["comparison_candidates"] = names
        if not comparison_state.get("target_anchor") and (
            comparison_state.get("walking_time_max_min") is not None
            or comparison_state.get("location_anchor")
        ):
            comparison_state["target_anchor"] = (
                comparison_state.get("location_anchor")
                or comparison_state.get("location")
            )
        target = str(comparison_state.get("target_anchor") or "").strip()
        dimensions = comparison_dimensions(comparison_state)
        core_dimensions = comparison_core_dimensions(comparison_state)
        comparison = collect_comparison_evidence(
            ctx.store,
            results,
            comparison_state,
            request_id=request_id,
        )
        covered_cells = comparison.covered_cells()
        limitations.extend(item for item in comparison.missing if item not in limitations)
        recommendation = supported_recommendation(
            names,
            dimensions,
            comparison,
            core_dimensions=core_dimensions,
        )
        accessibility = accessibility_contract(
            names,
            comparison,
            target_anchor=target,
        )
        if (
            "accessibility_needs" in dimensions
            and "accessibility_needs" not in comparison.complete_dimensions(names, dimensions)
        ):
            limitations.append(
                "accessibility_needs 未覆盖全部候选，不据此声称适老或无障碍优势"
            )
        artifact = {
            "artifact_type": "candidate_comparison",
            "subject": {
                "destination_city": state.get("destination_city") or ctx.profile.destination,
                "target_anchor": target or None,
                "accessibility_mode": accessibility_mode(target),
            },
            "candidates": [{"name": name} for name in names],
            "comparison_dimensions": dimensions,
            "core_dimensions": core_dimensions,
            "evidence": comparison.evidence,
            "recommendation": recommendation,
            "limitations": list(dict.fromkeys(limitations)),
            "accessibility_contract": accessibility,
            "coverage_matrix": {
                name: {
                    dimension: (name, dimension) in covered_cells
                    for dimension in dimensions
                }
                for name in names
            },
            "coverage": {
                "required": len(names) * len(dimensions),
                "covered": len({
                    (item["candidate"], item["dimension"])
                    for item in comparison.evidence
                }),
                "complete": comparison.complete,
            },
        }
        ctx.store.put("candidate_comparison", artifact, request_id=request_id, task_id="artifact", agent="engine")
        return _render_specialized_artifact(artifact)

    if task_type in {TaskType.ITINERARY_PATCH, TaskType.LOCAL_ADJUSTMENT_ADVICE}:
        subject = state.get("conditional_activity") or (
            f"第{state['referenced_day_index']}天" if state.get("referenced_day_index") else state.get("location_anchor")
        )
        candidate_names = [
            str(item.get("name") or "").strip()
            for _artifact_id, kind, payload in records
            if kind in {"candidates", "restaurants", "hotels"}
            for item in (
                payload.get("pois") or payload.get("restaurants")
                or payload.get("hotels") or payload.get("items") or []
            )
            if isinstance(item, dict) and str(item.get("name") or "").strip()
            and (
                not state.get("need_indoor_backup")
                or _is_indoor_candidate(item)
            )
        ]
        alternatives = [name for name in dict.fromkeys(candidate_names) if not _entity_matches(str(subject or ""), name)]
        weather_payloads = [payload for _aid, kind, payload in records if kind == "weather"]
        adverse = any(
            re.search(r"雨|雪|暴|大风|高温", str(payload.get("condition") or ""))
            for payload in weather_payloads
        )
        recommendation = (
            "替换，并优先采用室内备选" if adverse and alternatives
            else "保留，但出发前复核实时条件" if not adverse
            else "条件不利，但尚无已核验备选"
        )
        if task_type == TaskType.LOCAL_ADJUSTMENT_ADVICE:
            artifact = {
                "artifact_type": "local_adjustment_advice",
                "subject": subject,
                "recommendation": recommendation,
                "alternatives": alternatives[:5],
                "evidence": evidence,
                "limitations": list(dict.fromkeys(limitations)),
                "apply_status": "advice_only_no_itinerary_modified",
            }
            ctx.store.put("local_adjustment_advice", artifact, request_id=request_id, task_id="artifact", agent="engine")
            return _render_specialized_artifact(artifact)

        plan = ctx.store.get(existing_plan_artifact_id) if existing_plan_artifact_id else None
        itinerary = (plan or {}).get("itinerary") or {}
        days = list(itinerary.get("days") or [])
        day_index = int(state.get("referenced_day_index") or 0)
        before = days[day_index - 1] if day_index and day_index <= len(days) else {"subject": subject}
        after = dict(before) if isinstance(before, dict) else {"subject": subject}
        after["local_adjustment"] = {
            "recommendation": recommendation,
            "replacement": alternatives[0] if adverse and alternatives else None,
        }
        artifact = {
            "artifact_type": "itinerary_patch",
            "subject": subject or "局部行程时段",
            "before": before,
            "after": after,
            "affected_periods": [day_index if day_index else str(subject or "local_segment")],
            "constraint_checks": [
                {"constraint": "original_itinerary_present", "passed": bool(days)},
                {"constraint": "scope_is_local", "passed": True},
            ],
            "evidence": evidence,
            "limitations": list(dict.fromkeys(limitations)),
            "apply_status": "proposed_patch",
            "source_itinerary_artifact_id": existing_plan_artifact_id,
        }
        ctx.store.put("itinerary_patch", artifact, request_id=request_id, task_id="artifact", agent="engine")
        return _render_specialized_artifact(artifact)

    if task_type == TaskType.ROUTE_PLAN:
        routes = [payload for _aid, kind, payload in records if kind == "routes"]
        if routes:
            route = routes[0]
            artifact = {
                "artifact_type": "route_plan",
                "origin": route.get("origin_name") or route.get("origin_poi_id"),
                "destination": route.get("destination_name") or route.get("destination_poi_id"),
                "evidence": evidence,
                "limitations": list(dict.fromkeys(limitations)),
                "routes": routes,
            }
            ctx.store.put(
                "route_plan",
                artifact,
                request_id=request_id,
                task_id="artifact",
                agent="engine",
            )
            return _render_specialized_artifact(artifact)
    return _render_lightweight_evidence_reply(ctx, task_type, results, missing)


def _is_indoor_candidate(item: dict[str, Any]) -> bool:
    description = " ".join(str(value or "") for value in (
        item.get("name"),
        item.get("category"),
        item.get("entity_type"),
        " ".join(str(tag) for tag in item.get("tags") or []),
    )).casefold()
    return any(marker in description for marker in (
        "museum", "gallery", "indoor", "aquarium",
        "博物馆", "美术馆", "展览馆", "室内", "科技馆", "海洋馆",
    ))


def _entity_matches(expected: str, actual: str) -> bool:
    left = _canonical_candidate_entity(expected)
    right = _canonical_candidate_entity(actual)
    return bool(left and right and (left in right or right in left))


def _render_specialized_artifact(artifact: dict[str, Any]) -> str:
    kind = str(artifact.get("artifact_type") or "artifact")
    lines = [f"Artifact：{kind}", f"- subject：{artifact.get('subject')}"]
    for field in (
        "candidates", "comparison_dimensions", "before", "after", "affected_periods",
        "constraint_checks", "recommendation", "alternatives", "evidence", "limitations", "apply_status",
    ):
        if field in artifact:
            lines.append(f"- {field}：{artifact.get(field)}")
    return "\n".join(lines)


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
    if task_type is not None:
        inputs["task_type"] = task_type.value
    if task_type == TaskType.ITINERARY_PATCH and existing_plan_artifact_id:
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


def _specialized_artifact_id_for_request(
    ctx: Any,
    task_type: TaskType | None,
    request_id: str,
) -> str | None:
    """Return only a valid structured artifact produced for this exact request."""
    kinds = {
        TaskType.ROUTE_PLAN: "route_plan",
        TaskType.CANDIDATE_COMPARISON: "candidate_comparison",
        TaskType.LOCAL_ADJUSTMENT_ADVICE: "local_adjustment_advice",
        TaskType.ITINERARY_PATCH: "itinerary_patch",
    }
    kind = kinds.get(task_type)
    if kind is None:
        return None
    artifact_id = ctx.store.latest_id_for_request(request_id, kind)
    if not artifact_id:
        return None
    payload = ctx.store.get(artifact_id)
    from travel_agent.evaluation.artifact_contract import artifact_content_valid

    if not isinstance(payload, dict) or not artifact_content_valid(kind, payload):
        return None
    return artifact_id


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
