"""Agent 运行时：真实 ReAct 路径 + 离线确定性兜底。

- 有 LLM key 时：用 ``create_agent`` 构建真正自主 tool calling 的 agent，
  LLM 自主决定调用哪些工具、是否追问、何时规划（对应 PROJECT_PLAN M0）。
- 无 LLM key 时：退化为确定性兜底，复用同一套工具函数按固定顺序跑通，
  保证项目可离线演示（叙事上明确这是 fallback，不冒充「真 agent」）。

两条路径产出同样的 ``AgentReply``（文本 + A2UI 卡片 + 地图数据 + 工具轨迹），
前端无需区分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import Any

from travel_agent.agent import toolkit
from travel_agent.agent.intent import (
    MessageKind,
    classify_message,
    conversation_reply_text,
)
from travel_agent.agent.turn_analysis import (
    REQUIRED_SLOTS,
    TaskType,
    TurnAnalysis,
    analyze_travel_turn,
    is_plan_revision_followup,
    required_slot_satisfied,
)
from travel_agent.profile_patch import patches_to_payload
from travel_agent.agent.preferences import (
    needs_preference_guidance,
    turn_expresses_preferences,
    user_confirms_l3_preferences,
    user_skips_preference_prompt,
)
from travel_agent.agent.render import (
    build_itinerary_cards,
    build_map_payload,
    build_supplement_cards,
)
from travel_agent.agent.response_summary import (
    build_plan_reply_text,
    fallback_no_artifact_message,
    looks_like_hallucinated_itinerary,
)
from travel_agent.agent.session import SessionContext, build_session
from travel_agent.schemas import TravelProfile
from travel_agent.settings import Settings, get_settings
from travel_agent.storage.user_memory import (
    PreferenceObservation,
    get_user_memory_service,
)
from travel_agent.workflow import merge_l3_preferences
from travel_agent.workflow_rules import (
    extract_preference_removals,
    format_interests_display,
    format_pace_display,
)


PLANNER_STATUS = {
    "WAITING": "waiting",
    "PLAN_SUCCESS": "planned",
    "PLAN_FAILURE": "planner_failed",
    "PLAN_FALLBACK": "fallback_used",
    "CLARIFICATION": "clarification",
}


RECOVERY_STATUS = {
    "NA": "not_applicable",
    "OK": "recovered",
    "FAILED": "failed",
    "NOT_NEEDED": "not_needed",
}


@dataclass
class AgentReply:
    text: str
    cards: list[dict[str, Any]] = field(default_factory=list)
    map_payload: dict[str, Any] | None = None
    tool_trace: list[str] = field(default_factory=list)
    used_real_agent: bool = False
    clarification: bool = False
    planner_status: str | None = None
    recovery_state: str | None = None
    critical_slots_matched: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    status: str | None = None
    plan_artifact_id: str | None = None
    delivery_artifact_id: str | None = None
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    request_id: str | None = None
    turn_metrics: dict[str, Any] = field(default_factory=dict)
    raw_failure: str | None = None
    fallback_triggered: bool = False
    final_outcome: str | None = None
    delivery_status: str | None = None


def run_production_turn(
    user_message: str,
    ctx: SessionContext | None = None,
    history: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    user_id: str = "default",
) -> AgentReply:
    """生产固定入口：固定 ``PRODUCTION_CONFIG = FULL_CONFIG = V3_CONFIG``。

    不读取任何 variant 配置；评测/消融只能通过
    ``orchestration.variants.run_variant_turn`` 显式选择 V0–V3。
    统一持久化本轮明确偏好与已完成行程。
    """
    from travel_agent.orchestration.multi_agent import PRODUCTION_CONFIG

    return run_architecture_turn(
        PRODUCTION_CONFIG, user_message, ctx, history, settings, user_id
    )


def run_architecture_turn(
    capabilities: Any,
    user_message: str,
    ctx: SessionContext | None = None,
    history: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    user_id: str = "default",
) -> AgentReply:
    """Shared production/ablation outer lifecycle; only capabilities vary."""
    settings = settings or get_settings()
    ctx = ctx or build_session()
    history = history or []
    previous_itinerary_id = ctx.store.latest_id("itinerary")
    ctx.pending_preference_observations = []
    # 规则提取否定
    for value in extract_preference_removals(user_message):
        ctx.profile.interests = [item for item in ctx.profile.interests if item != value]
        ctx.pending_preference_observations.append(
            {"category": "interests", "value": value, "polarity": "negative"}
        )
    _apply_l3_preferences(ctx, user_id, settings)
    _maybe_inherit_l3_interests(ctx, user_id, settings, user_message)
    from travel_agent.agent.turn_lifecycle import run_turn_lifecycle

    reply = run_turn_lifecycle(
        capabilities, user_message, ctx, settings, history, user_id
    )
    if ctx.request_control is not None:
        ctx.request_control.check_active()
    _persist_user_memory_turn(
        user_message,
        ctx,
        history,
        settings,
        user_id,
        reply,
        previous_itinerary_id,
    )
    return reply


def _run_production_turn_impl(
    user_message: str,
    ctx: SessionContext | None = None,
    history: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    user_id: str = "default",
) -> AgentReply:
    """生产一轮对话；编排固定为 Multi-Agent Full（V3）。"""
    settings = settings or get_settings()
    ctx = ctx or build_session()
    history = history or []

    analysis = analyze_travel_turn(
        user_message,
        ctx,
        settings,
        history,
        ctx.evaluation_trace if ctx.evaluation_trace_enabled else None,
    )
    kind = analysis.kind
    if kind != MessageKind.TRAVEL and not _is_travel_followup(user_message, ctx):
        return _run_conversation_only(
            user_message,
            ctx,
            history,
            user_id,
            settings,
            preclassified_kind=kind,
        )
    # 先按任务类型补齐必填槽位（而非全局 destination+days），让用户补全信息后再规划，防止错规划
    clarify_reply = _prepare_travel_turn(user_message, ctx, user_id, settings, analysis)
    if clarify_reply is not None:
        return clarify_reply

    if settings.llm.enabled:
        try:
            return _run_multi_agent(user_message, ctx, history, settings, analysis)
        except Exception as exc:  # 真实路径失败时优雅降级，保证可用
            if ctx.evaluation_trace_enabled and not any(
                item.get("status") == "error" and item.get("phase") == "multi_agent"
                for item in ctx.evaluation_trace[-2:]
            ):
                ctx.evaluation_trace.append(
                    {
                        "kind": "model",
                        "phase": "multi_agent",
                        "model": settings.llm.model,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            reply = _run_fallback(
                user_message,
                ctx,
                settings=settings,
                history=history,
                user_id=user_id,
                preclassified_kind=kind,
                analysis=analysis,
            )
            reply.text = f"（{_format_react_failure(exc)}，已降级到离线兑底）\n\n" + reply.text
            return reply
    return _run_fallback(
        user_message,
        ctx,
        settings=settings,
        history=history,
        user_id=user_id,
        preclassified_kind=kind,
        analysis=analysis,
    )


def _run_multi_agent(
    user_message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Settings,
    analysis: TurnAnalysis,
) -> AgentReply:
    """生产 Multi-Agent Full（V3）链路；不读取 variant。"""
    from travel_agent.orchestration.multi_agent import PRODUCTION_CONFIG, MultiAgentEngine

    engine = MultiAgentEngine(PRODUCTION_CONFIG)
    outcome = engine.run_turn(
        ctx,
        settings,
        user_message,
        task_type=analysis.task_type,
        task_brief=user_message,
        history=history,
    )
    return _reply_from_outcome(ctx, outcome)


def _reply_from_outcome(ctx: SessionContext, outcome: Any) -> AgentReply:
    """把 Engine TurnOutcome 转为产品 AgentReply。"""
    from travel_agent.orchestration.multi_agent.schemas import (
        STATUS_CLARIFICATION_REQUIRED,
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    )

    clarification = outcome.status == STATUS_CLARIFICATION_REQUIRED
    tool_trace = [
        name
        for result in outcome.results
        for name in (getattr(result, "tool_trace", None) or [])
    ]
    if clarification:
        planner_status = PLANNER_STATUS["CLARIFICATION"]
        failure_reason = "missing_slot"
    elif outcome.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        planner_status = PLANNER_STATUS["PLAN_SUCCESS"]
        failure_reason = None
    else:
        planner_status = PLANNER_STATUS["PLAN_FAILURE"]
        failure_reason = "planner_failed"
    from travel_agent.delivery_contract import resolve_delivery_snapshot

    delivery = resolve_delivery_snapshot(
        ctx,
        attempted_artifact_id=outcome.plan_artifact_id,
        specialized_artifact_id=outcome.delivery_artifact_id,
        clarification=clarification,
    )
    outcome.delivery_status = delivery.status
    text = outcome.reply
    deterministic_preplanner_reason = text if not outcome.plan_artifact_id else ""
    if delivery.artifact_id and outcome.delivery_artifact_id:
        # A valid specialized artifact is the requested deliverable.  Preserve
        # its type-specific renderer even when it is explicitly partial.
        outcome.cards = []
        outcome.map_payload = None
    elif delivery.artifact_id and outcome.plan_artifact_id:
        outcome.plan_artifact_id = delivery.artifact_id
        text = build_plan_reply_text(ctx, delivery.artifact_id)
    elif outcome.status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        outcome.cards = []
        outcome.map_payload = None
        text = (
            "当前候选未通过最终校验，ArtifactStore 中没有可交付的 current 行程。"
            if delivery.status == "candidate_rejected"
            else "当前行程正在等待按最新约束重建，尚无可交付的 current 行程。"
            if delivery.status == "rebuild_pending"
            else "当前没有可交付的 current 行程。"
        )
        if deterministic_preplanner_reason.strip():
            text += "\n\n" + deterministic_preplanner_reason.strip()
    if not clarification and outcome.status not in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        text = _render_incomplete_outcome_reply(ctx, outcome, text)
    return AgentReply(
        text=text,
        cards=list(outcome.cards),
        map_payload=outcome.map_payload,
        tool_trace=tool_trace,
        used_real_agent=True,
        clarification=clarification,
        planner_status=planner_status,
        recovery_state=RECOVERY_STATUS["NA"],
        critical_slots_matched=_critical_slots_matched(ctx),
        failure_reason=failure_reason,
        profile=toolkit._profile_brief(ctx.profile),
        status=outcome.status,
        plan_artifact_id=outcome.plan_artifact_id,
        delivery_artifact_id=delivery.artifact_id,
        agent_trace=list(
            ((ctx.store.get(outcome.trace_artifact_id) or {}).get("items") or [])
            if outcome.trace_artifact_id
            else []
        ),
        delivery_status=delivery.status,
    )


def _render_incomplete_outcome_reply(ctx: SessionContext, outcome: Any, text: str) -> str:
    """Make a fail-closed Engine/Gate status explicit in the user-facing reply."""
    reasons: list[str] = []
    review = getattr(outcome, "review", None)
    if review is not None:
        if getattr(review, "error", None):
            reasons.append(f"语义复核不可用：{review.error}")
        reasons.extend(
            str(getattr(issue, "description", "") or "").strip()
            for issue in (getattr(review, "issues", None) or [])
            if str(getattr(issue, "description", "") or "").strip()
        )
    elif getattr(outcome, "plan_artifact_id", None):
        reasons.append("语义复核未完成")

    plan_id = getattr(outcome, "plan_artifact_id", None)
    plan_payload = ctx.store.get(plan_id) if plan_id else None
    if isinstance(plan_payload, dict):
        reasons.extend(
            str(issue.get("message") or "").strip()
            for issue in ((plan_payload.get("critic") or {}).get("issues") or [])
            if isinstance(issue, dict)
            and str(issue.get("severity") or "").lower() in {"error", "critical"}
            and str(issue.get("message") or "").strip()
        )
    latest_by_agent: dict[str, Any] = {}
    for result in getattr(outcome, "results", None) or []:
        latest_by_agent[str(getattr(result, "agent", "") or "unknown")] = result
    for result in latest_by_agent.values():
        # Do not disclose an intermediate failure that the bounded workflow
        # repair superseded with a later successful result for the same role.
        if getattr(result, "status", None) in {"completed", "completed_with_warnings"}:
            continue
        if getattr(result, "error", None):
            reasons.append(str(result.error))
        reasons.extend(str(item) for item in (getattr(result, "unresolved", None) or []) if item)

    unique_reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    detail = "；".join(unique_reasons[:3]) or "交付 Gate 未通过，仍有未解决问题"
    # build_plan_reply_text describes deterministic critic state and can say
    # “约束检查已通过” even when the semantic Reviewer/Gate failed.  Remove
    # that contradictory line while retaining the useful itinerary summary.
    safe_body = "\n".join(
        line for line in (text or "").splitlines() if not line.strip().startswith("约束检查")
    ).strip()
    header = f"当前方案尚不可交付，仅供审阅。未解决：{detail}。"
    return f"{header}\n\n{safe_body}" if safe_body else header


def _is_travel_followup(user_message: str, ctx: SessionContext) -> bool:
    """已有出行上下文时，允许「随便/沿用上次」这类短回复继续规划。"""
    if _is_plan_revision_followup(user_message, ctx):
        return True
    if ctx.profile.missing_required_fields():
        return False
    return (
        user_skips_preference_prompt(user_message)
        or user_confirms_l3_preferences(user_message)
        or turn_expresses_preferences(user_message)
    )


_is_plan_revision_followup = is_plan_revision_followup  # 兼容旧引用


def _hydrate_profile_from_latest_itinerary(profile: TravelProfile, ctx: SessionContext) -> None:
    """修改既有行程时，缺 destination/days 从历史行程补齐以避免重复追问。"""
    latest_pack = ctx.store.latest("itinerary")
    if not latest_pack:
        return
    itinerary = latest_pack.get("itinerary") if isinstance(latest_pack, dict) else None
    if not isinstance(itinerary, dict):
        return
    if not profile.destination:
        city = itinerary.get("city")
        if isinstance(city, str) and city:
            profile.destination = city
    if not profile.days:
        days = itinerary.get("days")
        if isinstance(days, list):
            profile.days = len(days)


def _format_react_failure(exc: Exception) -> str:
    msg = str(exc)
    if "AllocationQuota.FreeTierOnly" in msg or "free tier" in msg.lower():
        return (
            "千问 API 免费额度已用尽：请在阿里云 DashScope 控制台关闭「仅使用免费额度」，"
            "或充值 / 改用 qwen-turbo 后重试"
        )
    if "403" in msg and ("quota" in msg.lower() or "额度" in msg):
        return "千问 API 配额或权限不足，请检查 DashScope 账户"
    return f"ReAct agent 调用失败：{msg[:200]}"


def _apply_l3_preferences(ctx: SessionContext, user_id: str, settings: Settings) -> None:
    l3 = get_user_memory_service(settings.memory).load_stable_profile(user_id)
    if not l3.interests:
        from travel_agent.storage.user_profile import UserProfileStore
        # 从本地持久化存储补一次
        l3 = UserProfileStore(settings.memory.profile_dir).load(user_id)
    ctx.profile = merge_l3_preferences(ctx.profile, l3)


def _prepare_travel_turn(
    user_message: str,
    ctx: SessionContext,
    user_id: str,
    settings: Settings,
    analysis: TurnAnalysis,
) -> AgentReply | None:
    """出行轮次：合并 L3 偏好 + 按 patch 语义应用本轮抽取；缺任务必填槽位则先追问。"""
    _apply_l3_preferences(ctx, user_id, settings)
    _call_toolkit(
        ctx,
        "apply_profile_patches",
        toolkit.apply_profile_patches,
        patches=patches_to_payload(analysis.patches),
    )
    if analysis.task_type == TaskType.ITINERARY_REVISION:
        _hydrate_profile_from_latest_itinerary(ctx.profile, ctx)
    missing = _missing_required_slots(ctx.profile, analysis.task_type)
    if missing:
        return _ask_for_missing(ctx, missing, user_id, settings)
    _maybe_inherit_l3_interests(ctx, user_id, settings, user_message)
    return None


def _maybe_inherit_l3_interests(
    ctx: SessionContext,
    user_id: str,
    settings: Settings,
    user_message: str,
) -> None:
    """本轮未写明兴趣时，仅在用户说「随便」或确认沿用 L3 后才继承历史偏好。"""
    if ctx.profile.interests:
        return
    l3 = get_user_memory_service(settings.memory).load_stable_profile(user_id)
    if not l3.interests:
        return
    if user_skips_preference_prompt(user_message) or user_confirms_l3_preferences(user_message):
        toolkit.update_travel_profile(
            ctx,
            interests=list(l3.interests),
            _record_preferences=False,
        )


def _missing_required_slots(
    profile: TravelProfile,
    task_type: TaskType,
) -> list[str]:
    """按任务类型动态计算缺失的必填槽位；路线问答/行程修改不强制追问。"""
    return [
        slot
        for slot in REQUIRED_SLOTS.get(task_type, ())
        if not required_slot_satisfied(profile, task_type, slot)
    ]


def _contains_date_hint(text: str) -> bool:
    return any(keyword in text for keyword in ("周", "号", "月", "日", "周末", "明天", "后天", "今天", "星期", "周五", "周六", "周日"))


def _contains_budget_hint(text: str) -> bool:
    return any(keyword in text for keyword in ("预算", "预算大概", "预算范围", "多少钱", "花费", "价位", "贵", "省钱", "性价比"))


def _contains_companion_hint(text: str) -> bool:
    return any(keyword in text for keyword in ("两人", "三人", "4人", "5人", "家庭", "家人", "朋友", "情侣", "夫妻", "和我", "我和", "亲子", "独自", "一个人"))


def _contains_transport_hint(text: str) -> bool:
    return any(keyword in text for keyword in ("地铁", "公交", "打车", "步行", "开车", "公交车", "地铁站"))


def _ask_for_preferences(
    ctx: SessionContext,
    user_id: str,
    settings: Settings,
) -> AgentReply:
    l3 = get_user_memory_service(settings.memory).load_stable_profile(user_id)
    info = toolkit.request_preference_guide(ctx, l3)
    return AgentReply(
        text=info["question"],
        tool_trace=["request_preference_guide"],
        used_real_agent=False,
        clarification=True,
        planner_status=PLANNER_STATUS["CLARIFICATION"],
        recovery_state=RECOVERY_STATUS["NOT_NEEDED"],
        profile=toolkit._profile_brief(ctx.profile),
    )


def _ask_for_missing(
    ctx: SessionContext,
    missing: list[str],
    user_id: str,
    settings: Settings,
) -> AgentReply:
    info = toolkit.request_travel_info(ctx, missing)
    text = info["question"]
    recent_trips = get_user_memory_service(settings.memory).list_recent_trips(user_id, 1)
    if "days" in missing and recent_trips and recent_trips[0].days:
        text += f" 你最近一次玩了 {recent_trips[0].days} 天，也可以直接告诉我这次的天数。"
    ask_missing = missing[:2]  # 控制单轮最多2个关键槽位，减少问答发散
    return AgentReply(
        text=f"为确保我能给出可执行方案，请先补齐：{', '.join(ask_missing)}。\n" + text,
        tool_trace=["request_travel_info"],
        used_real_agent=False,
        clarification=True,
        planner_status=PLANNER_STATUS["CLARIFICATION"],
        recovery_state=RECOVERY_STATUS["NOT_NEEDED"],
        critical_slots_matched=[],
        failure_reason="missing_slot",
        profile=toolkit._profile_brief(ctx.profile),
    )


def _l3_hint_text(user_id: str, settings: Settings) -> str | None:
    service = get_user_memory_service(settings.memory)
    l3 = service.load_stable_profile(user_id)
    parts: list[str] = []
    if l3.interests:
        parts.append(f"偏好 {format_interests_display(l3.interests)}")
    if l3.pace != "standard":
        parts.append(f"节奏 {format_pace_display(l3.pace)}")
    return "；".join(parts) if parts else None


def _persist_user_memory_turn(
    user_message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Settings,
    user_id: str,
    reply: AgentReply,
    previous_itinerary_id: str | None,
) -> None:
    del user_message
    service = get_user_memory_service(settings.memory)
    observations: list[PreferenceObservation] = []
    for item in ctx.pending_preference_observations:
        value = str(item.get("value") or "").strip()
        category = str(item.get("category") or "").strip()
        if category == "interests":
            from travel_agent.workflow_rules import normalize_interest

            value = normalize_interest(value)
        if value:
            observations.append(
                PreferenceObservation(
                    category=category,
                    value=value,
                    polarity="negative" if item.get("polarity") == "negative" else "positive",
                )
            )
    turn_index = sum(1 for role, _ in history if role == "user") + 1
    if observations:
        service.record_preference_events(
            user_id,
            ctx.session_id,
            turn_index,
            observations,
        )
    pack = ctx.store.latest("itinerary")
    if (
        pack
        and ctx.store.latest_id("itinerary") != previous_itinerary_id
        and not reply.clarification
        and pack.get("critic", {}).get("passed") is True
        and isinstance(pack.get("itinerary"), dict)
    ):
        service.upsert_completed_trip(
            user_id,
            ctx.session_id,
            ctx.profile,
            pack["itinerary"],
        )


def _run_conversation_only(
    user_message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    user_id: str,
    settings: Settings,
    preclassified_kind: MessageKind | None = None,
) -> AgentReply:
    """非出行意图：寒暄/闲聊/无关话题，不跑规划工具链。"""
    kind = preclassified_kind or classify_message(user_message)
    hint = _l3_hint_text(user_id, settings)

    if settings.llm.enabled:
        try:
            return _run_conversation_chat(user_message, ctx, history, settings, hint, kind)
        except Exception:
            pass

    text = conversation_reply_text(kind, user_message, hint)
    return AgentReply(
        text=text,
        tool_trace=[],
        used_real_agent=False,
        clarification=kind == MessageKind.AMBIGUOUS,
        planner_status=PLANNER_STATUS["WAITING"],
        recovery_state=RECOVERY_STATUS["NOT_NEEDED"],
        profile=toolkit._profile_brief(ctx.profile),
    )


def _run_conversation_chat(
    user_message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Settings,
    l3_hint: str | None,
    kind: MessageKind,
) -> AgentReply:
    """有 LLM 时：用纯对话（无工具）回应非出行输入。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from travel_agent.agent.prompts import CONVERSATION_SYSTEM

    model = _build_chat_model(settings)
    from travel_agent.orchestration.meter import meter_callbacks

    callbacks = meter_callbacks("conversation")
    if ctx.evaluation_trace_enabled:
        from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

        callbacks.append(
            EvaluationTraceCallback(
                ctx.evaluation_trace,
                model=settings.llm.model,
                phase="conversation",
            )
        )
    system = CONVERSATION_SYSTEM
    if l3_hint:
        system += f"\n\n用户历史偏好（仅供参考，勿自动开规划）：{l3_hint}"

    messages: list[Any] = [SystemMessage(content=system)]
    for role, content in history[-6:]:
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    response = model.invoke(messages, config={"callbacks": callbacks} if callbacks else None)
    text = response.content if isinstance(response.content, str) else str(response.content)
    return AgentReply(
        text=text.strip() or "你好，我是旅行规划助手，有出行计划可以告诉我。",
        tool_trace=[],
        used_real_agent=True,
        clarification=kind == MessageKind.AMBIGUOUS,
        planner_status=PLANNER_STATUS["WAITING"],
        recovery_state=RECOVERY_STATUS["NOT_NEEDED"],
        profile=toolkit._profile_brief(ctx.profile),
    )


# --------------------------------------------------------------------------- #
# 真实 ReAct 路径
# --------------------------------------------------------------------------- #
def _build_chat_model(
    settings: Settings,
    *,
    timeout_seconds: float | None = None,
    max_retries: int = 2,
):
    from langchain_openai import ChatOpenAI

    extra_body = None
    if settings.llm.model.startswith("qwen3"):
        extra_body = {"enable_thinking": settings.llm.thinking_enabled}
    elif settings.llm.model.startswith("glm-"):
        extra_body = {
            "thinking": {
                "type": "enabled" if settings.llm.thinking_enabled else "disabled"
            }
        }
    elif settings.llm.model.lower() == "deepseek-ai/deepseek-v4-flash":
        # SiliconFlow defaults V4-Flash to a reasoning mode unless the switch
        # is sent explicitly.  ``max_tokens`` only limits the visible answer,
        # so omitting this flag can silently add thousands of reasoning tokens
        # and consume the whole per-stage timeout even when thinking is
        # disabled in our settings.
        extra_body = {"enable_thinking": settings.llm.thinking_enabled}
        if settings.llm.thinking_enabled:
            extra_body["reasoning_effort"] = "high"

    rate_limiter = _shared_rate_limiter(
        settings.llm.provider,
        settings.llm.model,
        settings.llm.requests_per_second,
    )

    return ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        timeout=(
            settings.llm.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        ),
        max_retries=max_retries,
        extra_body=extra_body,
        rate_limiter=rate_limiter,
    )


@lru_cache(maxsize=16)
def _shared_rate_limiter(provider: str, model: str, requests_per_second: float):
    del provider, model
    if requests_per_second <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=min(0.5, 1 / requests_per_second),
        max_bucket_size=1,
    )


def _run_react(
    user_message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Settings,
    user_id: str = "default",
) -> AgentReply:
    import asyncio

    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage

    from travel_agent.agent.prompts import (
        build_system_prompt_sections,
        record_system_prompt_trace,
        sections_to_text,
    )
    from travel_agent.agent.tool_source import resolve_tools_async
    from travel_agent.storage.memory_framework import MemoryFramework
    memory = MemoryFramework.build(
        ctx, history, user_id, settings.memory, settings.llm.enabled
    )
    skills_section = ""
    # Architecture ablations share the stable logical ToolSource contract.
    # Do not advertise optional skill tools that are intentionally filtered out.

    tools, _ = asyncio.run(resolve_tools_async(ctx, settings, user_id=user_id))
    model = _build_chat_model(settings)
    sections = build_system_prompt_sections(ctx.profile, memory, skills_section)
    agent = create_agent(model, tools, system_prompt=sections_to_text(sections))
    record_system_prompt_trace(ctx, sections, phase="react")
    messages: list[Any] = []
    for role, content in memory.history_for_prompt:
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    from travel_agent.orchestration.meter import meter_callbacks

    callbacks = meter_callbacks("v0_main")
    if ctx.evaluation_trace_enabled:
        from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

        callbacks.append(
            EvaluationTraceCallback(
                ctx.evaluation_trace,
                model=settings.llm.model,
                phase="react",
            )
        )
    state = agent.invoke(
        {"messages": messages},
        config={
            "recursion_limit": settings.agent.recursion_limit,
            **({"callbacks": callbacks} if callbacks else {}),
        },
    )

    out_messages = state["messages"]
    tool_trace = [
        call["name"]
        for msg in out_messages
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
    ]
    clarification = (
        "request_travel_info" in tool_trace
        and "plan_and_critique" not in tool_trace
    )
    recovered_tools: list[str] = []
    if not clarification and ctx.store.latest_current("itinerary") is None:
        recovered_tools = _complete_required_plan(ctx, user_message)
        tool_trace.extend(recovered_tools)
    final_text = ""
    for msg in reversed(out_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            final_text = msg.content.strip()
            break

    final_text = _finalize_reply_text(ctx, tool_trace, final_text)
    plan_itinerary = ctx.store.latest_current("itinerary")
    critic_passed = bool(
        plan_itinerary
        and isinstance(plan_itinerary, dict)
        and plan_itinerary.get("critic", {}).get("passed") is True
    )
    reply = _reply_from_store(
        ctx,
        final_text or "（无文本输出）",
        tool_trace,
        used_real_agent=True,
        planner_status=(
            PLANNER_STATUS["PLAN_SUCCESS"]
            if not clarification and critic_passed
            else PLANNER_STATUS["CLARIFICATION"]
            if clarification
            else PLANNER_STATUS["PLAN_FAILURE"]
        ),
        recovery_state=(RECOVERY_STATUS["OK"] if recovered_tools and critic_passed else RECOVERY_STATUS["NA"]),
        critical_slots_matched=_critical_slots_matched(ctx),
        failure_reason=("clarification_loop" if clarification else "planner_failed" if not critic_passed else None),
    )
    reply.clarification = clarification
    return reply


def _complete_required_plan(ctx: SessionContext, user_message: str) -> list[str]:
    """补齐 ReAct 漏掉的强制尾链；这是 Agent 恢复，不是完整离线 fallback。"""
    trace: list[str] = []
    if ctx.store.latest("candidates") is None:
        result = _call_toolkit(ctx, "search_poi", toolkit.search_poi)
        trace.append("search_poi")
        if result.get("isError"):
            return trace
    if ctx.store.latest("weather") is None:
        _call_toolkit(ctx, "check_weather", toolkit.check_weather)
        trace.append("check_weather")
    if _wants_restaurants(user_message, ctx) and ctx.store.latest("restaurants") is None:
        _call_toolkit(ctx, "search_restaurant", toolkit.search_restaurant)
        trace.append("search_restaurant")
    if _wants_hotels(user_message, ctx) and ctx.store.latest("hotels") is None:
        _call_toolkit(ctx, "search_hotel", toolkit.search_hotel)
        trace.append("search_hotel")
    if _wants_budget(user_message, ctx) and ctx.store.latest("budget") is None:
        _call_toolkit(ctx, "estimate_budget", toolkit.estimate_budget)
        trace.append("estimate_budget")
    if ctx.store.latest("ranked") is None:
        result = _call_toolkit(ctx, "recommend_candidates", toolkit.recommend_candidates)
        trace.append("recommend_candidates")
        if result.get("isError"):
            return trace
    _call_toolkit(ctx, "plan_and_critique", toolkit.plan_and_critique)
    trace.append("plan_and_critique")
    return trace


def _call_toolkit(ctx: SessionContext, name: str, fn, **kwargs: Any) -> dict[str, Any]:
    """调用确定性工具并记录真实参数，供恢复诊断和评测使用。"""
    started = time.perf_counter()
    record: dict[str, Any] = {"kind": "tool", "name": name, "arguments": dict(kwargs)}
    try:
        result = fn(ctx, **kwargs)
        record["status"] = "error" if result.get("isError") else "ok"
        if result.get("error_code") is not None:
            record["error_code"] = result.get("error_code")
        return result
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return {"isError": True, "summary": f"{name} 调用失败：{exc}", "error": record["error"]}
    finally:
        record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if ctx.evaluation_trace_enabled:
            ctx.evaluation_trace.append(record)


# --------------------------------------------------------------------------- #
# 离线确定性兜底
# --------------------------------------------------------------------------- #
def _run_fallback(
    user_message: str,
    ctx: SessionContext,
    settings: Settings | None = None,
    history: list[tuple[str, str]] | None = None,
    user_id: str = "default",
    preclassified_kind: MessageKind | None = None,
    analysis: TurnAnalysis | None = None,
) -> AgentReply:
    settings = settings or get_settings()
    history = history or []
    trace: list[str] = []
    if analysis is None:
        # 离线兜底路径：主流程已有 analysis 则复用，否则规则直出（LLM 失败自动回退规则）
        analysis = analyze_travel_turn(user_message, ctx, settings, history)
    kind = preclassified_kind or analysis.kind
    if kind != MessageKind.TRAVEL and not _is_travel_followup(user_message, ctx):
        return _run_conversation_only(
            user_message,
            ctx,
            history,
            user_id,
            settings,
            preclassified_kind=kind,
        )

    # 按 patch 语义应用本轮抽取（SET/CLEAR/UNCHANGED），写回画像
    _call_toolkit(
        ctx,
        "apply_profile_patches",
        toolkit.apply_profile_patches,
        patches=patches_to_payload(analysis.patches),
    )
    trace.append("apply_profile_patches")

    missing = _missing_required_slots(ctx.profile, analysis.task_type)  # 按任务类型检查必填槽位
    if missing:  # 有缺失时
        info = _call_toolkit(
            ctx,
            "request_travel_info",
            toolkit.request_travel_info,
            missing_fields=missing,
        ) #生成追问问题
        trace.append("request_travel_info")
        return AgentReply(
            text=f"离线兜底模式先补齐关键信息：{', '.join(missing[:2])}。\n" + info["question"],
            tool_trace=trace,
            used_real_agent=False,
            clarification=True,
            planner_status=PLANNER_STATUS["CLARIFICATION"],
            recovery_state=RECOVERY_STATUS["NOT_NEEDED"],
            critical_slots_matched=[],
            failure_reason="missing_slot",
            profile=toolkit._profile_brief(ctx.profile),
        )
    #没缺失时强制工具链
    search_result = _call_toolkit(ctx, "search_poi", toolkit.search_poi)
    trace.append("search_poi")
    if search_result.get("isError"):
        return AgentReply(
            text=search_result["summary"],
            tool_trace=trace,
            used_real_agent=False,
            planner_status=PLANNER_STATUS["PLAN_FAILURE"],
            recovery_state=RECOVERY_STATUS["FAILED"],
            critical_slots_matched=_critical_slots_matched(ctx),
            failure_reason="tool_or_data",
            profile=toolkit._profile_brief(ctx.profile),
        )
    _call_toolkit(ctx, "check_weather", toolkit.check_weather)
    trace.append("check_weather")
    if _wants_restaurants(user_message, ctx):
        _call_toolkit(ctx, "search_restaurant", toolkit.search_restaurant)
        trace.append("search_restaurant")
    if _wants_hotels(user_message, ctx):
        _call_toolkit(ctx, "search_hotel", toolkit.search_hotel)
        trace.append("search_hotel")
    if _wants_budget(user_message, ctx):
        _call_toolkit(ctx, "estimate_budget", toolkit.estimate_budget)
        trace.append("estimate_budget")
    _call_toolkit(ctx, "recommend_candidates", toolkit.recommend_candidates)
    trace.append("recommend_candidates")
    plan_result = _call_toolkit(ctx, "plan_and_critique", toolkit.plan_and_critique)
    trace.append("plan_and_critique")

    if plan_result.get("isError"): #规划失败
        return AgentReply(
            text=plan_result["summary"],
            tool_trace=trace,
            used_real_agent=False,
            planner_status=PLANNER_STATUS["PLAN_FAILURE"],
            recovery_state=_derive_fallback_recovery(plan_result),
            critical_slots_matched=_critical_slots_matched(ctx),
            failure_reason=_derive_fallback_failure_reason(plan_result),
            profile=toolkit._profile_brief(ctx.profile),
        )

    text = build_plan_reply_text(ctx) #面向用户的文本
    return _reply_from_store(
        ctx,
        text,
        trace,
        used_real_agent=False,
        planner_status=PLANNER_STATUS["PLAN_FALLBACK"],
        recovery_state=RECOVERY_STATUS["NA"],
        critical_slots_matched=_critical_slots_matched(ctx),
        failure_reason=None,
    ) #最终回复


def _finalize_reply_text(ctx: SessionContext, tool_trace: list[str], llm_text: str) -> str:
    """规划成功时用 artifact 摘要替代模型长文；拦截手写行程幻觉。"""
    has_itinerary = ctx.store.latest_current("itinerary") is not None
    if "plan_and_critique" in tool_trace and has_itinerary:
        return build_plan_reply_text(ctx)
    if looks_like_hallucinated_itinerary(llm_text):
        return build_plan_reply_text(ctx) if has_itinerary else fallback_no_artifact_message()
    return llm_text


# --------------------------------------------------------------------------- #
# 公共：从 store 拼装 reply
# --------------------------------------------------------------------------- #
def _reply_from_store(
    ctx: SessionContext,
    text: str,
    tool_trace: list[str],
    used_real_agent: bool,
    planner_status: str,
    recovery_state: str,
    critical_slots_matched: list[str] | None = None,
    failure_reason: str | None = None,
) -> AgentReply:
    payload = ctx.store.latest_current("itinerary")
    recovery = ctx.store.latest("recovery")
    if recovery:
        reason = recovery.get("reason") or "外部查询异常"
        suggestion = recovery.get("suggestion") or "建议出发前复核实时信息。"
        text = f"查询服务出现异常（{reason}），已使用受控降级结果继续规划。{suggestion}\n\n{text}"
        if recovery_state == RECOVERY_STATUS["NA"]:
            recovery_state = RECOVERY_STATUS["OK"]
    cards: list[dict[str, Any]] = []
    map_payload: dict[str, Any] | None = None
    if payload:
        weather = ctx.store.latest("weather")
        cards = build_itinerary_cards(payload, weather)
        cards.extend(
            build_supplement_cards(
                restaurants=ctx.store.latest("restaurants"),
                hotels=ctx.store.latest("hotels"),
                budget=ctx.store.latest("budget"),
            )
        )
        map_payload = build_map_payload(payload["itinerary"])
    return AgentReply(
        text=text,
        cards=cards,
        map_payload=map_payload,
        tool_trace=tool_trace,
        used_real_agent=used_real_agent,
        planner_status=planner_status,
        recovery_state=recovery_state,
        critical_slots_matched=critical_slots_matched or [],
        failure_reason=failure_reason,
        profile=toolkit._profile_brief(ctx.profile),
    )


def _critical_slots_matched(ctx: SessionContext) -> list[str]:
    slots = ["destination", "days", "start_date", "companions", "budget_level"]
    return [slot for slot in slots if bool(getattr(ctx.profile, slot, None))]


def _derive_fallback_recovery(plan_result: dict[str, Any]) -> str:
    if plan_result.get("isError"):
        return RECOVERY_STATUS["FAILED"]
    return RECOVERY_STATUS["NOT_NEEDED"]


def _derive_fallback_failure_reason(plan_result: dict[str, Any]) -> str:
    if plan_result.get("isError"):
        return "planner_failed"
    return "tool_or_data"


def _wants_restaurants(user_message: str, ctx: SessionContext) -> bool:
    from travel_agent.orchestration.multi_agent.dispatch_rules import (
        specific_restaurant_recommendation_requested,
    )

    return specific_restaurant_recommendation_requested(
        user_message,
        toolkit._profile_brief(ctx.profile),
        ctx.profile.constraint_state,
    )


def _wants_hotels(user_message: str, ctx: SessionContext) -> bool:
    text = user_message.lower()
    return bool(
        ctx.profile.hotel_area
        or any(keyword in text for keyword in ("住", "住宿", "酒店", "民宿", "hotel"))
    )


def _wants_budget(user_message: str, ctx: SessionContext) -> bool:
    text = user_message.lower()
    return bool(
        ctx.profile.budget_level
        or ctx.profile.companions
        or any(keyword in text for keyword in ("预算", "费用", "花多少钱", "多少钱", "省钱", "便宜", "人均"))
    )
