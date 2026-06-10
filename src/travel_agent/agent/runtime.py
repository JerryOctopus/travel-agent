"""Agent 运行时：真实 ReAct 路径 + 离线确定性兜底。

- 有 LLM key 时：用 ``create_react_agent`` 构建真正自主 tool calling 的 agent，
  LLM 自主决定调用哪些工具、是否追问、何时规划（对应 PROJECT_PLAN M0）。
- 无 LLM key 时：退化为确定性兜底，复用同一套工具函数按固定顺序跑通，
  保证项目可离线演示（叙事上明确这是 fallback，不冒充「真 agent」）。

两条路径产出同样的 ``AgentReply``（文本 + A2UI 卡片 + 地图数据 + 工具轨迹），
前端无需区分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from travel_agent.agent import toolkit
from travel_agent.agent.render import build_itinerary_cards, build_map_payload
from travel_agent.agent.session import SessionContext, build_session
from travel_agent.settings import Settings, get_settings
from travel_agent.workflow_rules import extract_profile_rule_based


@dataclass
class AgentReply:
    text: str
    cards: list[dict[str, Any]] = field(default_factory=list)
    map_payload: dict[str, Any] | None = None
    tool_trace: list[str] = field(default_factory=list)
    used_real_agent: bool = False
    clarification: bool = False
    profile: dict[str, Any] = field(default_factory=dict)


def run_turn(
    user_message: str,
    ctx: SessionContext | None = None,
    history: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
) -> AgentReply:
    """运行一轮对话。``ctx`` 复用同一会话以支持多轮。"""
    settings = settings or get_settings()
    ctx = ctx or build_session()

    if settings.llm.enabled:
        try:
            return _run_react(user_message, ctx, history or [], settings)
        except Exception as exc:  # 真实路径失败时优雅降级，保证可用
            reply = _run_fallback(user_message, ctx)
            reply.text = f"（ReAct agent 调用失败，已降级到离线兜底：{exc}）\n\n" + reply.text
            return reply
    return _run_fallback(user_message, ctx)


# --------------------------------------------------------------------------- #
# 真实 ReAct 路径
# --------------------------------------------------------------------------- #
def _build_chat_model(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=settings.llm.temperature,
        timeout=settings.llm.timeout_seconds,
    )


def _run_react(
    user_message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Settings,
) -> AgentReply:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    from travel_agent.agent.lc_tools import build_tools
    from travel_agent.agent.prompts import build_system_prompt

    tools = build_tools(ctx)
    model = _build_chat_model(settings)
    agent = create_react_agent(model, tools)

    messages: list[Any] = [SystemMessage(content=build_system_prompt(ctx.profile))]
    for role, content in history:
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))

    state = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": settings.agent.recursion_limit},
    )

    out_messages = state["messages"]
    tool_trace = [
        call["name"]
        for msg in out_messages
        if isinstance(msg, AIMessage)
        for call in (msg.tool_calls or [])
    ]
    final_text = ""
    for msg in reversed(out_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            final_text = msg.content.strip()
            break

    clarification = "request_travel_info" in tool_trace and "plan_and_critique" not in tool_trace
    reply = _reply_from_store(ctx, final_text or "（无文本输出）", tool_trace, used_real_agent=True)
    reply.clarification = clarification
    return reply


# --------------------------------------------------------------------------- #
# 离线确定性兜底
# --------------------------------------------------------------------------- #
def _run_fallback(user_message: str, ctx: SessionContext) -> AgentReply:
    trace: list[str] = []

    extracted = extract_profile_rule_based(user_message)
    toolkit.update_travel_profile(
        ctx,
        destination=extracted.destination,
        days=extracted.days,
        interests=extracted.interests,
        budget_level=extracted.budget_level,
        pace=extracted.pace,
        companions=extracted.companions,
    )
    trace.append("update_travel_profile")

    missing = ctx.profile.missing_required_fields()
    if missing:
        info = toolkit.request_travel_info(ctx, missing)
        trace.append("request_travel_info")
        return AgentReply(
            text=info["question"],
            tool_trace=trace,
            used_real_agent=False,
            clarification=True,
            profile=toolkit._profile_brief(ctx.profile),
        )

    toolkit.search_poi(ctx)
    trace.append("search_poi")
    toolkit.check_weather(ctx)
    trace.append("check_weather")
    toolkit.recommend_candidates(ctx)
    trace.append("recommend_candidates")
    plan_result = toolkit.plan_and_critique(ctx)
    trace.append("plan_and_critique")

    if plan_result.get("isError"):
        return AgentReply(
            text=plan_result["summary"],
            tool_trace=trace,
            used_real_agent=False,
            profile=toolkit._profile_brief(ctx.profile),
        )

    text = _fallback_summary_text(ctx, plan_result)
    return _reply_from_store(ctx, text, trace, used_real_agent=False)


def _fallback_summary_text(ctx: SessionContext, plan_result: dict[str, Any]) -> str:
    payload = ctx.store.latest("itinerary") or {}
    itinerary = payload.get("itinerary", {})
    lines = [f"# {itinerary.get('summary', '行程草案')}", ""]
    profile = ctx.profile
    meta = []
    if profile.destination:
        meta.append(f"目的地：{profile.destination}")
    if profile.days:
        meta.append(f"天数：{profile.days}天")
    if profile.interests:
        meta.append(f"偏好：{', '.join(profile.interests)}")
    if meta:
        lines.append("；".join(meta))
        lines.append("")
    lines.append(
        f"约束检查：违规项 {plan_result['original_issue_count']} → "
        f"{plan_result['final_issue_count']}，"
        f"{'已通过 critic' if plan_result['passed'] else '仍有可解释告警'}。"
    )
    if plan_result.get("revision_notes"):
        lines.append("")
        lines.append("自动修正：")
        lines.extend(f"- {note}" for note in plan_result["revision_notes"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 公共：从 store 拼装 reply
# --------------------------------------------------------------------------- #
def _reply_from_store(
    ctx: SessionContext,
    text: str,
    tool_trace: list[str],
    used_real_agent: bool,
) -> AgentReply:
    payload = ctx.store.latest("itinerary")
    cards: list[dict[str, Any]] = []
    map_payload: dict[str, Any] | None = None
    if payload:
        weather = ctx.store.latest("weather")
        cards = build_itinerary_cards(payload, weather)
        map_payload = build_map_payload(payload["itinerary"])
    return AgentReply(
        text=text,
        cards=cards,
        map_payload=map_payload,
        tool_trace=tool_trace,
        used_real_agent=used_real_agent,
        profile=toolkit._profile_brief(ctx.profile),
    )
