"""V2/V3 动态 Main Orchestrator（Step 3）。

- Orchestrator 是 ReAct agent，工具面只有 ``ORCHESTRATOR_ALLOWED_TOOLS``
  中的编排/交互工具 + ``dispatch_subagent``（业务重工具物理不可见）；
- ``dispatch_subagent`` 不加 session 锁（Step 2 锁隔离机制自动生效）；
- 每次派工的结构化 ``SubagentResult`` 由闭包收集，Engine 聚合消费
  payload / evidence，而非只拼接 summary；
- 渲染不在 Orchestrator 内发生：由 Engine 在 Reviewer 之后经 Renderer
  Gate 统一执行（见 render_gate.py）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from travel_agent.orchestration.multi_agent.orchestrator import (
    DispatchLedger,
    ORCHESTRATOR_ALLOWED_TOOLS,
    build_dispatch_tool,
    filter_orchestrator_tools,
)
from travel_agent.orchestration.multi_agent.registry import list_subagents
from travel_agent.orchestration.multi_agent.runner import SubagentRunner

_ORCHESTRATOR_PROMPT_TEMPLATE = """你是旅行规划 Main Orchestrator，只做编排，不做具体调研与规划。
职责：需求澄清、任务拆分、动态派工、聚合结构化结果、给出最终回复。

可用 Subagent（通过 dispatch_subagent 派工）：
{subagents}

派工纪律：
1. 路线问题 → transport；餐厅问题 → restaurant；酒店区域比较 → hotel + transport；
   完整行程 → 先派必要领域 Subagent，再派 planner；不要派不需要的 Subagent；
2. 有依赖的任务等上游返回后再派，并在 inputs.artifact_ids 里带上上游证据的 artifact_id；
3. 信息不足（目的地/天数缺失）时调用 request_travel_info 向用户追问，不要猜测派工；
4. 消费 Subagent 返回的 payload / evidence / warnings / unresolved 做决策，
   不要只依赖自然语言 summary；
5. planner 成功返回后直接总结回复；渲染由系统自动完成，不要描述卡片细节。
禁止：编造景点/酒店/路线事实；跳过 planner 直接给出完整行程。"""


def build_orchestrator_prompt() -> str:
    lines = "\n".join(
        f"- {definition.name}: {definition.description}" for definition in list_subagents()
    )
    return _ORCHESTRATOR_PROMPT_TEMPLATE.format(subagents=lines)


def build_orchestrator_tools(
    runner: SubagentRunner,
    ctx: Any,
    settings: Any,
    request_id: str,
    results_sink: list | None = None,
    ledger: DispatchLedger | None = None,
    base_inputs: dict[str, Any] | None = None,
) -> tuple[list[Any], list]:
    """构建 Orchestrator 工具面；返回 (tools, results_sink)。

    results_sink 收集每次 dispatch 的 SubagentResult，供 Engine 聚合。
    """
    from travel_agent.agent.tool_source import resolve_tools

    sink = results_sink if results_sink is not None else []
    all_tools, _source = resolve_tools(ctx, settings)
    # render 工具不暴露给 Orchestrator：渲染必须经 Engine 的 Renderer Gate。
    allowed = ORCHESTRATOR_ALLOWED_TOOLS - {"render_itinerary", "render_map"}
    base_tools = [tool for tool in filter_orchestrator_tools(all_tools) if tool.name in allowed]

    dispatch_fn = build_dispatch_tool(
        runner,
        request_id=request_id,
        ledger=ledger,
        base_inputs=base_inputs,
    )

    def _collecting_dispatch(
        agent: str,
        instruction: str,
        inputs: dict | None = None,
        depends_on: list | None = None,
    ) -> str:
        raw = dispatch_fn(agent, instruction, inputs, depends_on)
        import json

        try:
            sink.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
        return raw

    dispatch_tool = StructuredTool.from_function(
        _collecting_dispatch,
        name="dispatch_subagent",
        description=(
            "派工给领域 Subagent。参数：agent（attraction|hotel|restaurant|transport|planner）、"
            "instruction（任务描述）、inputs（可选，含 artifact_ids 上游证据）、"
            "depends_on（可选上游 task_id 列表）。返回结构化 JSON 结果。"
        ),
    )
    return base_tools + [dispatch_tool], sink


def run_orchestrator(
    ctx: Any,
    settings: Any,
    user_message: str,
    request_id: str,
    *,
    runner: SubagentRunner,
    model: Any | None = None,
    results_sink: list | None = None,
    history: list[tuple[str, str]] | None = None,
    task_type: Any | None = None,
    turn_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行动态 Orchestrator 一轮，返回 {reply, tool_trace, results, clarification}。"""
    active_model = model
    if active_model is None:
        if not settings.llm.enabled:
            raise RuntimeError("LLM disabled: orchestrator unavailable")
        from travel_agent.agent.runtime import _build_chat_model

        active_model = _build_chat_model(settings)

    ledger = DispatchLedger()
    tools, sink = build_orchestrator_tools(
        runner,
        ctx,
        settings,
        request_id,
        results_sink,
        ledger=ledger,
        base_inputs=turn_inputs,
    )
    agent = create_react_agent(active_model, tools)
    context_lines = [f"本轮任务类型：{getattr(task_type, 'value', task_type) or 'unknown'}"]
    if turn_inputs:
        import json

        context_lines.append(
            "本轮显式输入：" + json.dumps(turn_inputs, ensure_ascii=False, default=str)
        )
    messages: list[Any] = [SystemMessage(content=build_orchestrator_prompt())]
    for role, content in (history or [])[-8:]:
        messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
    messages.append(HumanMessage(content="\n".join(context_lines) + "\n当前请求：" + user_message))
    from travel_agent.orchestration.meter import meter_callbacks

    state = agent.invoke(
        {"messages": messages},
        config={
            "recursion_limit": max(settings.agent.recursion_limit, 24),
            "callbacks": meter_callbacks("orchestrator"),
        },
    )
    out_messages = state.get("messages", [])
    tool_trace = [
        call["name"]
        for msg in out_messages
        if isinstance(msg, AIMessage)
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    reply = ""
    for msg in reversed(out_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            reply = msg.content.strip()
            break
    clarification = "request_travel_info" in tool_trace
    return {
        "reply": reply,
        "tool_trace": tool_trace,
        "results": sink,
        "clarification": clarification,
        "dispatch_ledger": ledger,
    }
