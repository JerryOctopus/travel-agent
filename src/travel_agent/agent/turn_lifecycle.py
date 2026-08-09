"""Shared outer turn lifecycle for production and V0--V3 evaluation variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from travel_agent.agent import toolkit
from travel_agent.agent.intent import MessageKind, conversation_reply_text
from travel_agent.agent.preferences import turn_expresses_preferences, user_skips_preference_prompt
from travel_agent.agent.session import SessionContext
from travel_agent.agent.turn_analysis import REQUIRED_SLOTS, TaskType, TurnAnalysis, analyze_travel_turn
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    new_request_id,
    new_task_id,
)
from travel_agent.profile_patch import patches_to_payload


@dataclass(frozen=True)
class PreparedTurn:
    analysis: TurnAnalysis
    history: list[tuple[str, str]]
    existing_plan_artifact_id: str | None
    turn_inputs: dict[str, Any]
    early_reply: Any | None = None


def prepare_turn(
    user_message: str,
    ctx: SessionContext,
    settings: Any,
    history: list[tuple[str, str]] | None,
) -> PreparedTurn:
    """Run the one shared preflight and freeze all context passed to the engine."""
    normalized_history = list(history or [])
    analysis = analyze_travel_turn(
        user_message,
        ctx,
        settings,
        normalized_history,
        ctx.evaluation_trace if ctx.evaluation_trace_enabled else None,
    )
    from travel_agent.agent.runtime import AgentReply

    contextual_followup = bool(ctx.store.latest_id("itinerary")) and (
        user_skips_preference_prompt(user_message)
        or turn_expresses_preferences(user_message)
    )
    if (
        analysis.kind != MessageKind.TRAVEL
        and analysis.task_type == TaskType.FULL_TRIP_PLAN
        and not contextual_followup
    ):
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=None,
            turn_inputs={},
            early_reply=AgentReply(
                text=conversation_reply_text(analysis.kind, user_message),
                clarification=analysis.kind == MessageKind.AMBIGUOUS,
                profile=toolkit._profile_brief(ctx.profile),
                status=(
                    STATUS_CLARIFICATION_REQUIRED
                    if analysis.kind == MessageKind.AMBIGUOUS
                    else STATUS_COMPLETED
                ),
            ),
        )

    toolkit.apply_profile_patches(ctx, patches=patches_to_payload(analysis.patches))
    if analysis.revision_directives.get("pace") == "relaxed":
        toolkit.update_travel_profile(ctx, pace="relaxed")

    existing_plan_id = None
    artifact_ids: list[str] = []
    if analysis.task_type == TaskType.ITINERARY_REVISION:
        existing_plan_id = ctx.store.latest_id("itinerary")
        if existing_plan_id:
            _hydrate_profile_from_plan(ctx, existing_plan_id)
            artifact_ids = _revision_source_ids(ctx, existing_plan_id)
        else:
            return PreparedTurn(
                analysis=analysis,
                history=normalized_history,
                existing_plan_artifact_id=None,
                turn_inputs={"revision_directives": analysis.revision_directives},
                early_reply=AgentReply(
                    text="当前会话没有可修改的既有行程，请先生成或提供一份行程。",
                    clarification=True,
                    profile=toolkit._profile_brief(ctx.profile),
                    status=STATUS_CLARIFICATION_REQUIRED,
                ),
            )

    missing = [
        slot
        for slot in REQUIRED_SLOTS.get(analysis.task_type, ())
        if not getattr(ctx.profile, slot, None)
    ]
    if missing:
        info = toolkit.request_travel_info(ctx, missing)
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=existing_plan_id,
            turn_inputs={},
            early_reply=AgentReply(
                text=f"为确保我能给出可执行方案，请先补齐：{', '.join(missing[:2])}。\n"
                + info["question"],
                tool_trace=["request_travel_info"],
                clarification=True,
                profile=toolkit._profile_brief(ctx.profile),
                status=STATUS_CLARIFICATION_REQUIRED,
                planner_status="clarification",
                failure_reason="missing_slot",
            ),
        )

    return PreparedTurn(
        analysis=analysis,
        history=normalized_history,
        existing_plan_artifact_id=existing_plan_id,
        turn_inputs={
            "artifact_ids": artifact_ids,
            "plan_artifact_id": existing_plan_id,
            "revision_directives": dict(analysis.revision_directives),
            "profile": toolkit._profile_brief(ctx.profile),
        },
    )


def run_turn_lifecycle(
    capabilities: Any,
    user_message: str,
    ctx: SessionContext,
    settings: Any,
    history: list[tuple[str, str]] | None = None,
    user_id: str = "default",
) -> Any:
    """Execute the shared lifecycle; capabilities only select engine strategy."""
    del user_id  # persistence policy is intentionally outside architecture variants
    from travel_agent.agent.runtime import AgentReply, _reply_from_outcome
    from travel_agent.orchestration.meter import (
        TurnBudgetExhausted,
        TurnMeter,
        turn_meter_scope,
    )
    from travel_agent.orchestration.multi_agent import MultiAgentEngine

    request_id = new_request_id()
    llm_settings = settings.llm
    orchestration_settings = getattr(settings, "orchestration", None)
    meter = TurnMeter(
        request_id=request_id,
        token_budget=int(
            getattr(orchestration_settings, "variant_token_budget", 0) or 0
        ),
        llm_call_budget=int(
            getattr(orchestration_settings, "variant_llm_call_budget", 0) or 0
        ),
        tool_call_budget=int(
            getattr(orchestration_settings, "variant_tool_call_budget", 0) or 0
        ),
        input_cost_per_million=float(
            getattr(llm_settings, "input_cost_per_million", 0.0) or 0.0
        ),
        output_cost_per_million=float(
            getattr(llm_settings, "output_cost_per_million", 0.0) or 0.0
        ),
    )
    reply: Any
    try:
        with turn_meter_scope(meter):
            prepared = prepare_turn(user_message, ctx, settings, history)
            if prepared.early_reply is not None:
                reply = prepared.early_reply
            elif not settings.llm.enabled:
                reply = _run_offline_fallback(ctx, prepared, request_id=request_id)
            else:
                engine = MultiAgentEngine(capabilities)
                try:
                    outcome = engine.run_turn(
                        ctx,
                        settings,
                        user_message,
                        task_type=prepared.analysis.task_type,
                        task_brief=user_message,
                        request_id=request_id,
                        history=prepared.history,
                        existing_plan_artifact_id=prepared.existing_plan_artifact_id,
                        turn_inputs=prepared.turn_inputs,
                    )
                    reply = _reply_from_outcome(ctx, outcome)
                except TurnBudgetExhausted as exc:
                    reply = AgentReply(
                        text="本轮调用预算已耗尽，未继续发起新的模型调用。",
                        profile=toolkit._profile_brief(ctx.profile),
                        status=STATUS_INCOMPLETE,
                        failure_reason="budget_exhausted",
                    )
                    setattr(reply, "raw_failure", f"{type(exc).__name__}: {exc}")
                except Exception as exc:  # every variant shares identical fallback policy
                    reply = _run_offline_fallback(ctx, prepared, request_id=request_id)
                    reply.text = f"（多 Agent 执行失败，已降级到离线兜底：{exc}）\n\n" + reply.text
                    setattr(reply, "raw_failure", f"{type(exc).__name__}: {exc}")
                    setattr(reply, "fallback_triggered", True)
    except TurnBudgetExhausted as exc:
        reply = AgentReply(
            text="本轮调用预算已耗尽，未继续发起新的模型或工具调用。",
            profile=toolkit._profile_brief(ctx.profile),
            status=STATUS_INCOMPLETE,
            failure_reason="budget_exhausted",
        )
        setattr(reply, "raw_failure", f"{type(exc).__name__}: {exc}")
    finally:
        meter.finish()
    setattr(reply, "request_id", request_id)
    setattr(reply, "turn_metrics", meter.snapshot())
    if not getattr(reply, "agent_trace", None):
        trace_id = ctx.store.latest_id_for_request(request_id, "agent_trace")
        if trace_id:
            setattr(
                reply,
                "agent_trace",
                list((ctx.store.get(trace_id) or {}).get("items") or []),
            )
    if not hasattr(reply, "raw_failure"):
        setattr(reply, "raw_failure", None)
    if not hasattr(reply, "fallback_triggered"):
        setattr(reply, "fallback_triggered", False)
    setattr(reply, "final_outcome", getattr(reply, "status", None))
    return reply


def _hydrate_profile_from_plan(ctx: SessionContext, plan_id: str) -> None:
    payload = ctx.store.get(plan_id) or {}
    itinerary = payload.get("itinerary") or {}
    if not ctx.profile.destination and itinerary.get("city"):
        ctx.profile.destination = itinerary["city"]
    if not ctx.profile.days and isinstance(itinerary.get("days"), list):
        ctx.profile.days = len(itinerary["days"])


def _revision_source_ids(ctx: SessionContext, plan_id: str) -> list[str]:
    payload = ctx.store.get(plan_id) or {}
    ids = [plan_id]
    for artifact_id in payload.get("source_artifact_ids") or []:
        if ctx.store.get_record(str(artifact_id)) is not None and artifact_id not in ids:
            ids.append(str(artifact_id))
    return ids


def _run_offline_fallback(
    ctx: SessionContext,
    prepared: PreparedTurn,
    *,
    request_id: str | None = None,
) -> Any:
    """Shared deterministic fallback with explicit plan ID and Renderer Gate."""
    from travel_agent.agent.runtime import AgentReply
    from travel_agent.agent.session import reset_task_meta, set_current_task_meta
    from travel_agent.orchestration.multi_agent.render_gate import render_plan_outcome

    if prepared.analysis.task_type not in {
        TaskType.FULL_TRIP_PLAN,
        TaskType.ITINERARY_REVISION,
    }:
        return AgentReply(
            text="当前离线兜底无法可靠完成该轻量查询，请补充更明确的信息或稍后重试。",
            profile=toolkit._profile_brief(ctx.profile),
            status=STATUS_INCOMPLETE,
            failure_reason="offline_capability_unavailable",
        )

    request_id = request_id or new_request_id()
    task_id = new_task_id("fallback")
    token = set_current_task_meta(
        {
            "request_id": request_id,
            "task_id": task_id,
            "agent": "fallback",
        }
    )
    trace: list[str] = []

    def call_tool(name: str, fn, *args, **kwargs):
        from travel_agent.orchestration.meter import current_turn_meter

        meter = current_turn_meter()
        started = meter.begin_tool("fallback") if meter is not None else None
        failed = True
        try:
            value = fn(*args, **kwargs)
            failed = bool(isinstance(value, dict) and value.get("isError"))
            trace.append(name)
            return value
        finally:
            if meter is not None and started is not None:
                meter.finish_tool("fallback", started, error=failed)

    try:
        if prepared.analysis.task_type == TaskType.FULL_TRIP_PLAN:
            attraction = call_tool("search_poi", toolkit.search_poi, ctx)
            weather = call_tool("check_weather", toolkit.check_weather, ctx)
            input_ids = [
                item.get("artifact_id")
                for item in (attraction, weather)
                if item.get("artifact_id")
            ]
            if ctx.profile.hotel_area:
                hotel = call_tool("search_hotel", toolkit.search_hotel, ctx)
                if hotel.get("artifact_id"):
                    input_ids.append(hotel["artifact_id"])
            if ctx.profile.food_preference or "food" in ctx.profile.interests:
                restaurant = call_tool("search_restaurant", toolkit.search_restaurant, ctx)
                if restaurant.get("artifact_id"):
                    input_ids.append(restaurant["artifact_id"])
            budget = call_tool("estimate_budget", toolkit.estimate_budget, ctx)
            if budget.get("artifact_id"):
                input_ids.append(budget["artifact_id"])
        else:
            input_ids = list(prepared.turn_inputs.get("artifact_ids") or [])
        ranked = call_tool(
            "recommend_candidates",
            toolkit.recommend_candidates,
            ctx,
            artifact_ids=input_ids,
        )
        if ranked.get("artifact_id"):
            input_ids.append(ranked["artifact_id"])
    finally:
        reset_task_meta(token)

    planner_task_id = new_task_id("planner")
    token = set_current_task_meta(
        {
            "request_id": request_id,
            "task_id": planner_task_id,
            "agent": "planner",
            "artifact_ids": input_ids,
            "revision_directives": dict(
                prepared.turn_inputs.get("revision_directives") or {}
            ),
        }
    )
    try:
        plan = call_tool(
            "plan_and_critique", toolkit.plan_and_critique, ctx, artifact_ids=input_ids
        )
    finally:
        reset_task_meta(token)

    plan_id = plan.get("artifact_id") if isinstance(plan, dict) else None
    delivery = STATUS_COMPLETED if plan_id else STATUS_INCOMPLETE
    rendered = render_plan_outcome(ctx, plan_id, delivery)
    recovery = ctx.store.latest("recovery")
    text = plan.get("summary") or "离线规划未能完成。"
    if recovery:
        text += "\n\n建议：" + str(recovery.get("suggestion") or recovery.get("reason") or "出发前复核实时信息。")
    return AgentReply(
        text=text,
        cards=rendered.get("cards") or [],
        map_payload=rendered.get("map_payload"),
        tool_trace=trace,
        used_real_agent=False,
        profile=toolkit._profile_brief(ctx.profile),
        status=(delivery if rendered.get("rendered") else STATUS_INCOMPLETE),
        plan_artifact_id=plan_id,
        failure_reason=None if plan_id else "planner_failed",
        planner_status="planned" if plan_id else "planner_failed",
        recovery_state=("recovered" if recovery else "not_applicable"),
        critical_slots_matched=[
            slot
            for slot in ("destination", "days")
            if getattr(ctx.profile, slot, None)
        ],
    )
