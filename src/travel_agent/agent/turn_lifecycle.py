"""Shared outer turn lifecycle for production and V0--V3 evaluation variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any

from travel_agent.agent import toolkit
from travel_agent.agent.intent import MessageKind, conversation_reply_text
from travel_agent.agent.preferences import turn_expresses_preferences, user_skips_preference_prompt
from travel_agent.agent.session import RequestControl, SessionContext
from travel_agent.agent.turn_analysis import (
    REQUIRED_SLOTS,
    DeliveryIntent,
    TaskType,
    TurnAnalysis,
    analyze_travel_turn,
    required_slot_satisfied,
)
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    new_request_id,
    new_task_id,
)
from travel_agent.profile_patch import patches_to_payload
from travel_agent.providers import ProviderRateLimitError


@dataclass(frozen=True)
class PreparedTurn:
    analysis: TurnAnalysis
    history: list[tuple[str, str]]
    existing_plan_artifact_id: str | None
    turn_inputs: dict[str, Any]
    early_reply: Any | None = None
    revisable_parent_artifact_id: str | None = None


def _mark_failed_rebuild_pending(
    ctx: SessionContext,
    analysis: Any,
    reply: Any,
) -> bool:
    """Keep a failed rebuild from being retried by every constraint-only turn."""
    if getattr(analysis, "delivery_intent", None) != DeliveryIntent.REBUILD_NOW:
        return False
    if getattr(reply, "plan_artifact_id", None):
        return False
    if ctx.store.latest_current_id("itinerary"):
        return False
    ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"
    return True


def _merge_constraint_state(
    current: dict[str, Any],
    update: dict[str, Any],
    *,
    source_turn: int = 0,
    explicitness: str = "explicit",
    confirmations: dict[str, Any] | None = None,
) -> None:
    """Normalize entities, append constraint events, and rebuild active state."""
    normalized_update = dict(update)
    incoming_removed = [str(item) for item in normalized_update.get("removed") or []]
    if incoming_removed:
        known_entities = [
            str(item)
            for key in ("must_visit", "candidate_attractions")
            for item in (current.get(key) or [])
        ]
        normalized_update["removed"] = [
            next(
                (
                    known
                    for known in known_entities
                    if term in known or known in term
                ),
                term,
            )
            for term in incoming_removed
        ]
    from travel_agent.constraint_events import merge_constraint_update

    merge_constraint_update(
        current,
        normalized_update,
        source_turn=source_turn,
        explicitness=explicitness,
        confirmations=confirmations,
    )
    if (
        "budget_max_cny" in normalized_update
        and "budget_total_cny" not in normalized_update
    ):
        current.pop("budget_total_cny", None)


_GENERIC_INTEREST_REQUIREMENTS = {
    "海边",
    "园林",
    "咖啡店",
    "历史景点",
    "主要历史景点",
    "自然风光",
    "历史文化",
    "自由活动",
}


def _is_named_poi_requirement(value: Any) -> bool:
    text = str(value or "").strip()
    if len(text) < 2 or text in _GENERIC_INTEREST_REQUIREMENTS:
        return False
    if re.search(r"\d{4}年|(?:\d+|[一二两三四五六七])天(?:行程|游)", text):
        return False
    if any(term in text for term in ("预算", "冲突", "步行以内", "露天路段")):
        return False
    if text.endswith(("行程", "项目", "要求")) or text in {"午餐", "晚餐", "三天行程"}:
        return False
    return True


def _sync_profile_constraints(profile: Any) -> None:
    """Keep compact profile fields and the explicit constraint tree coherent."""
    from travel_agent.workflow_rules import normalize_interests

    state = profile.constraint_state or {}
    removed = [str(item) for item in state.get("removed") or [] if str(item).strip()]
    state_avoid_terms = [
        str(item).strip()
        for item in list(state.get("avoid") or [])
        if str(item).strip()
    ]
    avoid_terms = [
        str(item).strip()
        for item in [*list(profile.avoid or []), *state_avoid_terms]
        if str(item).strip()
    ]
    requested = [
        *list(profile.must_visit or []),
        *list(state.get("must_visit") or []),
    ]
    profile.must_visit = list(
        dict.fromkeys(
            value
            for value in (str(item).strip() for item in requested)
            if _is_named_poi_requirement(value)
            and not any(term in value or value in term for term in removed)
            # Only trip-scoped/current state avoidance can override a current
            # must-visit. Stable-profile dislikes from an older trip must not
            # erase an explicit requirement made earlier in this conversation.
            and not any(term in value or value in term for term in state_avoid_terms)
        )
    )
    if isinstance(state.get("must_visit"), list):
        state["must_visit"] = [
            value
            for value in (str(item).strip() for item in state["must_visit"])
            if _is_named_poi_requirement(value)
            and not any(term in value or value in term for term in removed)
            and not any(term in value or value in term for term in state_avoid_terms)
        ]
    profile.interests = normalize_interests(
        list(dict.fromkeys([*list(profile.interests or []), *list(state.get("interests") or [])]))
    )
    raw_dietary = state.get("dietary") or []
    dietary = [str(raw_dietary)] if isinstance(raw_dietary, str) else [
        str(item) for item in raw_dietary
    ]
    excluded_foods = {
        cuisine
        for cuisine in ("海鲜", "辣", "清真")
        if any(re.search(rf"(?:不吃|不能吃|不要吃|忌).{{0,2}}{cuisine}", item) for item in dietary)
    }
    if excluded_foods:
        profile.food_preference = [
            item for item in profile.food_preference if str(item) not in excluded_foods
        ]
    if state.get("lodging_area"):
        profile.hotel_area = str(state["lodging_area"])
    if state.get("budget_max_cny") is not None:
        profile.budget_limit = float(state["budget_max_cny"])
        people = int(profile.party_size or state.get("traveler_count") or 1)
        days = int(profile.days or state.get("duration_days") or 1)
        if float(state["budget_max_cny"]) / max(1, people * days) < 600:
            profile.budget_level = "low"
    if avoid_terms or removed:
        profile.avoid = list(dict.fromkeys([*avoid_terms, *removed]))


def _apply_hybrid_intent_normalization(
    user_message: str,
    analysis: TurnAnalysis,
    ctx: SessionContext,
    settings: Any,
    *,
    source_turn: int,
) -> None:
    """Attach only bounded soft-interest output to deterministic preflight."""
    if analysis.task_type != TaskType.FULL_ITINERARY:
        return
    hybrid = getattr(settings, "hybrid_planning", None)
    enabled = bool(getattr(hybrid, "enable_llm_intent_normalizer", False))
    client = getattr(ctx, "hybrid_llm_client", None)
    if enabled and client is None and getattr(settings.llm, "enabled", False):
        from travel_agent.hybrid_planning.structured_client import (
            ProductionStructuredLLMClient,
        )

        client = ProductionStructuredLLMClient(settings, phase="intent_normalizer")
    from travel_agent.hybrid_planning.intent_normalizer import IntentNormalizer

    try:
        from travel_agent.orchestration.multi_agent.trace import current_trace

        active_trace = current_trace()
        request_scope = active_trace.request_id if active_trace is not None else f"turn_{source_turn}"
    except Exception:
        request_scope = f"turn_{source_turn}:{id(analysis)}"
    ctx.hybrid_request_scope = request_scope
    normalizer = IntentNormalizer(
        llm_client=client,
        enabled=enabled,
        cache=ctx.hybrid_call_cache,
        request_scope=request_scope,
    )
    result = normalizer.normalize(
        user_message,
        source_turn=f"turn_{source_turn}",
        request_type=analysis.task_type,
        delivery_intent=analysis.delivery_intent,
    )
    if not enabled:
        return
    # Enabling the flag is diagnostic-only unless the module actually returns
    # accepted LLM output. Deterministic/skipped/fallback paths must be
    # business-identical to the disabled path.
    accepted_llm = [
        item for item in result.normalized_interests if item.basis == "llm"
    ]
    if not accepted_llm:
        return
    preferred = [
        item.label for item in accepted_llm
        if item.polarity == "prefer"
    ]
    avoided = [
        item.label for item in accepted_llm
        if item.polarity == "avoid"
    ]
    if preferred:
        from travel_agent.profile_patch import PatchOp, SlotPatch

        existing = analysis.patches.get("interests")
        existing_values = (
            list(existing.value or [])
            if existing is not None and existing.op == PatchOp.SET
            else []
        )
        analysis.patches["interests"] = SlotPatch(
            PatchOp.SET,
            list(dict.fromkeys([*existing_values, *preferred])),
        )
        analysis.constraint_state["interests"] = list(
            dict.fromkeys([*list(analysis.constraint_state.get("interests") or []), *preferred])
        )
    analysis.constraint_state["normalized_interest_context"] = {
        **result.to_dict(),
        "normalized_interests": [
            item for item in result.to_dict()["normalized_interests"]
            if item.get("basis") == "llm"
        ],
    }
    if avoided:
        # This remains soft policy context and never enters profile.avoid,
        # which is consumed as a hard candidate filter.
        analysis.constraint_state["soft_avoid_interests"] = avoided


def prepare_turn(
    user_message: str,
    ctx: SessionContext,
    settings: Any,
    history: list[tuple[str, str]] | None,
) -> PreparedTurn:
    """Run the one shared preflight and freeze all context passed to the engine."""
    normalized_history = list(history or [])
    from travel_agent.artifact_policy import active_constraint_snapshot

    constraint_snapshot_before = active_constraint_snapshot(ctx.profile)
    previous_current_plan_id = ctx.store.latest_current_id("itinerary")
    stored_parent_id = str(
        (ctx.profile.constraint_state or {}).get("_revisable_parent_plan_artifact_id")
        or ""
    ) or None
    stored_parent_record = ctx.store.get_record(stored_parent_id) if stored_parent_id else None
    if not stored_parent_record or stored_parent_record.get("artifact_status") not in {
        "current", "stale", "historical",
    }:
        stored_parent_id = None
    previous_revisable_plan_id = (
        stored_parent_id or ctx.store.latest_revisable_id("itinerary")
    )
    """
    kind: 4 种 (TRAVEL, AMBIGUOUS, GREETING, OUT_OF_SCOPE)
    task_type: 6 种（UNKNOWN + 5 种受支持的旅行任务）
    source: 2 种（"rule" 或 "llm"）
    """
    analysis = analyze_travel_turn(
        user_message,
        ctx,
        settings,
        normalized_history,
        ctx.evaluation_trace if ctx.evaluation_trace_enabled else None,
    )
    ctx.active_task_type = str(analysis.task_type.value)
    ctx.active_delivery_intent = str(analysis.delivery_intent.value)
    _apply_hybrid_intent_normalization(
        user_message,
        analysis,
        ctx,
        settings,
        source_turn=sum(1 for role, _ in normalized_history if role == "user") + 1,
    )
    from travel_agent.agent.runtime import AgentReply

    contextual_followup = bool(ctx.store.latest_id("itinerary")) and (
        user_skips_preference_prompt(user_message)
        or turn_expresses_preferences(
            user_message,
            patches=analysis.patches,
        ) # 包含偏好信号
    )
    if (
        analysis.kind != MessageKind.TRAVEL
        and analysis.task_type in {TaskType.UNKNOWN, TaskType.FULL_TRIP_PLAN}
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

    if analysis.task_type == TaskType.UNKNOWN:
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=None,
            turn_inputs={},
            early_reply=AgentReply(
                text=(
                    "我还不能确定要执行哪类旅行任务。当前支持完整行程规划、路线查询、"
                    "地点/酒店/餐厅建议、单日建议和既有行程修改；请把目标说具体一些。"
                ),
                clarification=True,
                profile=toolkit._profile_brief(ctx.profile),
                status=STATUS_CLARIFICATION_REQUIRED,
                failure_reason="unknown_task_type",
            ),
        )

    if analysis.task_type == TaskType.SAFE_DECLINE:
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=None,
            turn_inputs={},
            early_reply=AgentReply(
                text="我不能代你执行预订、支付、取消或联系商家；可以帮你核对选项、整理操作步骤和风险提示。",
                clarification=False,
                profile=toolkit._profile_brief(ctx.profile),
                status=STATUS_COMPLETED,
                failure_reason="safe_decline",
            ),
        )

    if analysis.task_type == TaskType.CONSTRAINT_NEGOTIATION:
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=None,
            turn_inputs={},
            early_reply=AgentReply(
                text="这些硬约束目前无法同时可靠满足，需要先确认可放宽哪一项；我会保留其余明确要求。",
                clarification=True,
                profile=toolkit._profile_brief(ctx.profile),
                status=STATUS_CLARIFICATION_REQUIRED,
                failure_reason="constraint_negotiation",
            ),
        )

    from travel_agent.agent.turn_analysis import is_pure_weather_advice_request

    pure_weather_advice = is_pure_weather_advice_request(
        user_message,
        analysis.constraint_state,
    )
    durable_patches = {} if pure_weather_advice else analysis.patches
    toolkit.apply_profile_patches(ctx, patches=patches_to_payload(durable_patches))
    destination_city = analysis.constraint_state.get("destination_city")
    if destination_city and not ctx.profile.destination and not pure_weather_advice:
        # A local area is an anchor inside the city, never a replacement for
        # the trip destination slot.
        toolkit.update_travel_profile(ctx, destination=str(destination_city))
    constraint_update = dict(analysis.constraint_state)
    if pure_weather_advice:
        # City/date words scope this evidence query; they are not durable trip
        # edits, even when no current itinerary artifact exists yet.
        constraint_update.clear()
    elif analysis.delivery_intent == DeliveryIntent.LIGHTWEIGHT_ADVICE:
        # Weather/advice query parameters are turn-local evidence selectors,
        # not new plan constraints.  Keeping them out of active state also
        # preserves the pending revision/hash exactly across an interjection.
        for key in (
            "weather_condition", "resolved_date", "need_indoor_backup",
            "advice_topic",
        ):
            constraint_update.pop(key, None)
    if constraint_update:
        from travel_agent.constraint_events import normalize_return_deadline

        # Normalize before event ingestion so an unresolved last-day clock is
        # preserved as an auditable structured constraint, not ad-hoc metadata.
        normalize_return_deadline(
            constraint_update,
            start_date=ctx.profile.start_date,
            duration_days=ctx.profile.days,
            reference_datetime=ctx.reference_datetime,
        )
        source_turn = sum(1 for role, _ in normalized_history if role == "user") + 1
        soft = bool(re.search(r"(?:有人|朋友|同事).{0,6}(?:提议|建议)|如果.{0,12}(?:算了|就算)", user_message))
        confirmations: dict[str, Any] = {}
        if re.search(r"住宿.{0,4}(?:不要改|别改|保持不变|不变)", user_message):
            lodging = (ctx.profile.constraint_state or {}).get("lodging_area") or ctx.profile.hotel_area
            if lodging:
                confirmations["lodging_area"] = lodging
            analysis.constraint_state.pop("lodging_area", None)
            constraint_update.pop("lodging_area", None)
        _merge_constraint_state(
            ctx.profile.constraint_state,
            constraint_update,
            source_turn=source_turn,
            explicitness="soft" if soft else "explicit",
            confirmations=confirmations,
        )
        if (
            constraint_update.get("self_driving_allowed") is False
            or constraint_update.get("public_transport_required") is True
        ):
            ctx.profile.transport_mode = "public_transport"
    if analysis.revision_directives.get("pace") == "relaxed":
        toolkit.update_travel_profile(ctx, pace="relaxed")
    _sync_profile_constraints(ctx.profile)
    from travel_agent.constraint_events import normalize_return_deadline

    normalize_return_deadline(
        ctx.profile.constraint_state,
        start_date=ctx.profile.start_date,
        duration_days=ctx.profile.days,
        reference_datetime=ctx.reference_datetime,
    )
    from travel_agent.artifact_policy import update_constraint_version

    version = update_constraint_version(ctx.profile, constraint_snapshot_before)
    stale_ids: list[str] = []
    if version["material_changed"]:
        stale_ids = ctx.store.invalidate_itineraries(
            constraint_revision=version["revision"],
            constraint_hash=version["constraint_hash"],
            changed_fields=version["changed_fields"],
        )
        if stale_ids:
            previous_revisable_plan_id = previous_current_plan_id or stale_ids[-1]
            ctx.profile.constraint_state[
                "_revisable_parent_plan_artifact_id"
            ] = previous_revisable_plan_id
            ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"
        elif previous_revisable_plan_id and not ctx.store.latest_current_id("itinerary"):
            # Further constraint accumulation must retain the original rebuild
            # parent after the current artifact has already been invalidated.
            ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"
    rebuild_pending = (
        ctx.profile.constraint_state.get("_plan_status") == "rebuild_pending"
    )
    parent_for_rebuild = previous_revisable_plan_id if (
        rebuild_pending
        or analysis.task_type == TaskType.FULL_ITINERARY
    ) else None
    if analysis.delivery_intent == DeliveryIntent.STATE_UPDATE_ONLY:
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=None,
            turn_inputs={
                "profile": toolkit._profile_brief(ctx.profile),
                "stale_plan_artifact_ids": stale_ids,
                "constraint_version": version,
                "parent_plan_artifact_id": parent_for_rebuild,
                "delivery_intent": analysis.delivery_intent.value,
            },
            early_reply=AgentReply(
                text=(
                    "已更新当前约束；旧行程已标记为 stale，当前处于尚未完成重建状态。"
                    "本轮按你的要求不启动 Planner，需要完整行程时我会基于当前约束重建并重新校验。"
                    if stale_ids
                    else "已更新并保留当前约束；本轮不重新搜索或启动 Planner。"
                ),
                profile=toolkit._profile_brief(ctx.profile),
                status=STATUS_COMPLETED,
                planner_status="deferred_state_update",
            ),
            revisable_parent_artifact_id=parent_for_rebuild,
        )

    if analysis.task_type == TaskType.LOCAL_ADJUSTMENT_ADVICE:
        state = ctx.profile.constraint_state or {}
        has_local_subject = bool(
            state.get("conditional_activity")
            or state.get("referenced_day_index")
            or state.get("fixed_events")
            or state.get("location_anchor")
        )
        has_decision_condition = bool(
            state.get("date_start")
            or state.get("resolved_date")
            or state.get("weather_condition")
            or re.search(r"天气|下雨|高温|大风|替换|保留|取消", user_message)
        )
        if not (has_local_subject and has_decision_condition):
            return PreparedTurn(
                analysis=analysis,
                history=normalized_history,
                existing_plan_artifact_id=None,
                turn_inputs={},
                early_reply=AgentReply(
                    text="请粘贴需要判断的日期和对应活动（或相关行程片段）；不需要提供目的地、总天数或整份行程。",
                    clarification=True,
                    profile=toolkit._profile_brief(ctx.profile),
                    status=STATUS_CLARIFICATION_REQUIRED,
                    planner_status="clarification",
                    failure_reason="missing_local_adjustment_context",
                ),
            )

    if (
        analysis.task_type == TaskType.CANDIDATE_COMPARISON
        and re.search(r"比较|对比|哪个|哪一个|vs\.?", user_message, re.IGNORECASE)
        and not (
            (ctx.profile.constraint_state or {}).get("comparison_candidates")
            or (ctx.profile.constraint_state or {}).get("compare_lodging_areas")
            or (ctx.profile.constraint_state or {}).get("candidate_attractions")
        )
    ):
        return PreparedTurn(
            analysis=analysis,
            history=normalized_history,
            existing_plan_artifact_id=None,
            turn_inputs={},
            early_reply=AgentReply(
                text="请只补充要比较的候选项名称；如果比较到固定地点的通勤，再附上该目标锚点。",
                clarification=True,
                profile=toolkit._profile_brief(ctx.profile),
                status=STATUS_CLARIFICATION_REQUIRED,
                planner_status="clarification",
                failure_reason="missing_comparison_candidates",
            ),
        )

    existing_plan_id = None
    from travel_agent.artifact_policy import reusable_artifact_ids

    artifact_ids, reuse_audit = reusable_artifact_ids(
        ctx.store, ctx.profile.constraint_state
    )
    if analysis.task_type == TaskType.ITINERARY_REVISION:
        existing_plan_id = previous_current_plan_id or ctx.store.latest_current_id("itinerary")
        if existing_plan_id:
            _hydrate_profile_from_plan(ctx, existing_plan_id)
            # The previous plan is revision context. Its ancestors are reused
            # only when their constraint fingerprints still match active state.
            artifact_ids = list(dict.fromkeys([existing_plan_id, *artifact_ids]))
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

    if analysis.task_type == TaskType.FULL_ITINERARY and parent_for_rebuild:
        _hydrate_profile_from_plan(ctx, parent_for_rebuild)
        artifact_ids = list(dict.fromkeys([parent_for_rebuild, *artifact_ids]))
        parent_record = ctx.store.get_record(parent_for_rebuild) or {}
        parent_status = str(parent_record.get("artifact_status") or "")
        if parent_status in {"stale", "historical"}:
            reuse_audit.append({
                "artifact_id": parent_for_rebuild,
                "reason": "stale_parent_rebuild",
                "artifact_status": parent_status,
            })

    missing = [
        slot
        for slot in REQUIRED_SLOTS.get(analysis.task_type, ())
        if not required_slot_satisfied(ctx.profile, analysis.task_type, slot)
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

    if analysis.task_type == TaskType.FULL_ITINERARY:
        from travel_agent.constraint_events import validate_constraint_state

        state = ctx.profile.constraint_state or {}
        conflicts = validate_constraint_state(ctx.profile.constraint_state)
        if conflicts:
            return PreparedTurn(
                analysis=analysis,
                history=normalized_history,
                existing_plan_artifact_id=existing_plan_id,
                turn_inputs={"constraint_conflicts": conflicts},
                early_reply=AgentReply(
                    text="启动 Planner 前发现约束冲突：" + "; ".join(
                        f"{item['code']}({item['field']})" for item in conflicts
                    ),
                    clarification=True,
                    profile=toolkit._profile_brief(ctx.profile),
                    status=STATUS_CLARIFICATION_REQUIRED,
                    failure_reason="constraint_state_conflict",
                    planner_status="blocked_by_constraint_gate",
                ),
                revisable_parent_artifact_id=parent_for_rebuild,
            )

    revision_directives = dict(analysis.revision_directives)
    turn_profile = toolkit._profile_brief(ctx.profile)
    if analysis.delivery_intent == DeliveryIntent.LIGHTWEIGHT_ADVICE:
        # Weather/date selectors and a requested fallback are scoped to this
        # delivery.  They must reach workers and artifact validation without
        # mutating the durable itinerary constraint hash (a pure weather
        # interjection must leave a pending rebuild byte-for-byte unchanged).
        effective_state = dict(turn_profile.get("constraint_state") or {})
        effective_state.update({
            key: value
            for key, value in analysis.constraint_state.items()
            if value not in (None, "", [], {})
        })
        turn_profile["constraint_state"] = effective_state
    return PreparedTurn(
        analysis=analysis,
        history=normalized_history,
        existing_plan_artifact_id=existing_plan_id,
        turn_inputs={
            "artifact_ids": artifact_ids,
            "plan_artifact_id": existing_plan_id,
            "parent_plan_artifact_id": parent_for_rebuild,
            "revision_directives": revision_directives,
            "profile": turn_profile,
            "artifact_reuse_audit": reuse_audit,
            "constraint_version": version,
            "stale_plan_artifact_ids": stale_ids,
            "delivery_intent": (
                analysis.delivery_intent.value
                if analysis.delivery_intent is not None
                else None
            ),
        },
        revisable_parent_artifact_id=parent_for_rebuild,
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
    owns_request_control = ctx.request_control is None
    if owns_request_control:
        ctx.request_control = RequestControl(request_id)
    ctx.runtime_settings = settings
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
    from travel_agent.orchestration.multi_agent.trace import (
        AgentTraceLog,
        reset_current_trace,
        set_current_trace,
    )

    request_trace = AgentTraceLog(request_id)
    trace_token = set_current_trace(request_trace)
    reply: Any = None
    prepared: PreparedTurn | None = None
    try:
        with turn_meter_scope(meter):
            # 前置规则判断
            prepared = prepare_turn(user_message, ctx, settings, history)
            if prepared.early_reply is not None:
                reply = prepared.early_reply
            elif _is_weather_advice_turn(prepared):
                reply = _run_weather_advice(ctx, prepared)
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
                except ProviderRateLimitError:
                    # Never convert a configured-tool quota failure into an
                    # offline fallback: that fallback may call the same exhausted
                    # provider and can hide the environment failure from eval.
                    raise
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
                if _mark_failed_rebuild_pending(ctx, prepared.analysis, reply):
                    request_trace.append(
                        "rebuild_state",
                        agent="lifecycle",
                        status="pending",
                        detail={"reason": "rebuild_attempt_without_current_artifact"},
                    )
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
        try:
            if len(request_trace) == 0:
                analysis = prepared.analysis if prepared is not None else None
                delivery_intent = getattr(analysis, "delivery_intent", None)
                request_trace.append(
                    "turn_contract",
                    agent="lifecycle",
                    status="declared",
                    detail={
                        "task_type": (
                            analysis.task_type.value if analysis is not None else None
                        ),
                        "delivery_intent": (
                            delivery_intent.value if delivery_intent is not None else None
                        ),
                        "planner_required": delivery_intent == DeliveryIntent.REBUILD_NOW,
                        "planner_admitted": False,
                        "early_reply": bool(
                            prepared is not None and prepared.early_reply is not None
                        ),
                    },
                )
                request_trace.append(
                    "orchestration",
                    agent="lifecycle",
                    status=str(getattr(reply, "status", "")),
                    detail={"planner_status": getattr(reply, "planner_status", None)},
                )
            if ctx.store.latest_id_for_request(request_id, "agent_trace") is None:
                request_trace.flush_to_store(ctx.store)
        finally:
            reset_current_trace(trace_token)
            if owns_request_control:
                ctx.request_control = None
    if (
        prepared is not None
        and prepared.analysis.delivery_intent == DeliveryIntent.LIGHTWEIGHT_ADVICE
        and isinstance(prepared.turn_inputs.get("profile"), dict)
    ):
        # The response snapshot includes this turn's ephemeral selectors for
        # auditing/scoring.  The session profile itself stays untouched, so a
        # weather/advice interjection cannot invalidate a current itinerary.
        reply.profile = copy.deepcopy(prepared.turn_inputs["profile"])
    setattr(reply, "request_id", request_id)
    setattr(reply, "turn_metrics", meter.snapshot())
    entries = [
        item
        for item in (getattr(reply, "agent_trace", None) or [])
        if isinstance(item, dict) and item.get("request_id") == request_id
    ]
    if not entries:
        trace_id = ctx.store.latest_id_for_request(request_id, "agent_trace")
        if trace_id:
            entries = [
                item
                for item in ((ctx.store.get(trace_id) or {}).get("items") or [])
                if isinstance(item, dict) and item.get("request_id") == request_id
            ]
    if not entries:
        from travel_agent.orchestration.multi_agent.trace import AgentTraceLog

        trace = AgentTraceLog(request_id)
        analysis = prepared.analysis if prepared is not None else None
        delivery_intent = getattr(analysis, "delivery_intent", None)
        trace.append(
            "turn_contract",
            agent="lifecycle",
            status="declared",
            detail={
                "task_type": (
                    analysis.task_type.value if analysis is not None else None
                ),
                "delivery_intent": (
                    delivery_intent.value if delivery_intent is not None else None
                ),
                "planner_required": delivery_intent == DeliveryIntent.REBUILD_NOW,
                "planner_admitted": False,
                "early_reply": bool(
                    prepared is not None and prepared.early_reply is not None
                ),
            },
        )
        trace.append(
            "orchestration",
            agent="lifecycle",
            status=str(getattr(reply, "status", "")),
            detail={"planner_status": getattr(reply, "planner_status", None)},
        )
        trace_id = trace.flush_to_store(ctx.store)
        entries = list((ctx.store.get(trace_id) or {}).get("items") or [])
    setattr(reply, "agent_trace", entries)
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


def _is_weather_advice_turn(prepared: PreparedTurn) -> bool:
    state = prepared.analysis.constraint_state or {}
    # Only a pure weather/clothing interjection may bypass orchestration.
    # Weather-conditioned restaurant discovery and local replacement requests
    # still owe their task-specific artifact and supporting tool evidence.
    requires_domain_delivery = any(
        state.get(field) not in (None, "", [], {})
        for field in (
            "comparison_candidates",
            "candidate_attractions",
            "compare_lodging_areas",
            "conditional_activity",
            "referenced_day_index",
            "need_indoor_backup",
            "specific_restaurant_recommendation",
            "top_n",
            "location",
            "location_anchor",
        )
    )
    return bool(
        prepared.analysis.delivery_intent == DeliveryIntent.LIGHTWEIGHT_ADVICE
        and state.get("weather_condition")
        and not requires_domain_delivery
        and prepared.analysis.task_type in {
            TaskType.CANDIDATE_COMPARISON,
            TaskType.LOCAL_ADJUSTMENT_ADVICE,
        }
    )


def _run_weather_advice(ctx: SessionContext, prepared: PreparedTurn) -> Any:
    """Answer a weather/clothing interjection without mutating plan state."""
    from travel_agent.agent.runtime import AgentReply

    state = (
        (prepared.turn_inputs.get("profile") or {}).get("constraint_state")
        or prepared.analysis.constraint_state
        or {}
    )
    city = (
        state.get("destination_city")
        or next(iter(state.get("destinations") or []), None)
        or state.get("destination")
        or ctx.profile.destination
    )
    weather = toolkit.check_weather(ctx, str(city) if city else None)
    if weather.get("isError"):
        return AgentReply(
            text=str(weather.get("summary") or "天气信息暂不可用。"),
            tool_trace=["check_weather"],
            profile=toolkit._profile_brief(ctx.profile),
            status=STATUS_INCOMPLETE,
            planner_status="not_required",
            failure_reason="weather_evidence_unavailable",
        )
    condition = str(weather.get("condition") or "天气待复核")
    temperature = weather.get("temperature_c")
    try:
        degrees = int(temperature)
    except (TypeError, ValueError):
        degrees = None
    topic = str((prepared.analysis.constraint_state or {}).get("advice_topic") or "weather")
    adverse = any(term in condition for term in ("雨", "雪", "暴", "大风", "高温"))
    if degrees is None:
        clothing = "建议采用可增减的分层穿法，并带轻便防雨层。"
    elif degrees <= 8:
        clothing = "建议穿保暖外套并做好手脚防寒。"
    elif degrees <= 17:
        clothing = "建议长袖加薄外套，早晚可再加一层。"
    elif degrees <= 27:
        clothing = "建议轻薄透气衣物，随身带一件薄外套。"
    else:
        clothing = "建议短袖等透气衣物，并注意遮阳、补水和防晒。"
    if adverse:
        clothing += " 当前条件下再带雨具或防风层，鞋子优先防滑。"
    activity = (
        "户外安排宜缩短，并保留室内或短距离交通备选。"
        if adverse or (prepared.analysis.constraint_state or {}).get("need_indoor_backup")
        else "通常可按原计划活动，出发前再复核临近预报。"
    )
    temperature_text = f"，约 {degrees}°C" if degrees is not None else ""
    if topic == "climate":
        lead = "工具提供的是当前天气而非长期气候统计；可先按当前条件准备"
    elif topic == "clothing":
        lead = "穿衣建议"
    else:
        lead = "天气与出行建议"
    return AgentReply(
        text=f"{lead}：{condition}{temperature_text}。{clothing}{activity}",
        tool_trace=["check_weather"],
        profile=toolkit._profile_brief(ctx.profile),
        status=STATUS_COMPLETED,
        planner_status="not_required",
    )


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

    state = {
        **(ctx.profile.constraint_state or {}),
        **(prepared.analysis.constraint_state or {}),
    }
    if _is_weather_advice_turn(prepared):
        return _run_weather_advice(ctx, prepared)

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
            "constraint_state": dict(ctx.profile.constraint_state or {}),
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
            "parent_plan_artifact_id": prepared.turn_inputs.get("parent_plan_artifact_id")
            or prepared.turn_inputs.get("plan_artifact_id"),
            "constraint_state": dict(ctx.profile.constraint_state or {}),
        }
    )
    try:
        plan = call_tool(
            "plan_and_critique", toolkit.plan_and_critique, ctx, artifact_ids=input_ids
        )
    finally:
        reset_task_meta(token)

    plan_id = plan.get("artifact_id") if isinstance(plan, dict) else None
    if plan_id:
        from travel_agent.orchestration.multi_agent.engine import _promote_selected_candidate

        if not _promote_selected_candidate(
            ctx,
            plan_id,
            limitations=[],
            reason="offline_deterministic_finalizer_pass",
        ):
            plan_id = None
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
