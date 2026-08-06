"""真实 LLM 的受限 ReAct Subagent 执行器（Step 3）。

- 按 ``SubagentDefinition.tool_names`` 白名单过滤 ``build_tools`` 全量工具；
- ``create_react_agent`` 受限执行（recursion_limit = max_steps * 2）；
- ``max_tool_calls`` 超限后拒绝后续工具调用（结果带 budget_exhausted 告警）；
- token_usage 由执行区间的 EvaluationTraceCallback 汇总；
- 不持任何 session 锁；异常由 SubagentRunner 统一包装为 failed。

注意：本模块只构建执行器闭包，供 ``SubagentRunner`` 注入；
Orchestrator / fixed dispatch 的编排在 engine 层。
"""

from __future__ import annotations

import json
from functools import wraps
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from travel_agent.orchestration.multi_agent.registry import SubagentDefinition
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_COMPLETED,
    SubagentTask,
)

_BUDGET_SENTINEL = "__SUBAGENT_TOOL_BUDGET_EXHAUSTED__"


def _budget_guard(tool: Any, max_calls: int, counter: dict[str, int]):
    """工具调用计数守卫：超限后返回哨兵错误，不再真正执行工具。"""
    original_coroutine = getattr(tool, "coroutine", None)

    def _exceeded() -> bool:
        counter["n"] += 1
        return counter["n"] > max_calls

    if tool.func is not None:
        original_fn = tool.func

        @wraps(original_fn)
        def guarded(*args, **kwargs):
            if _exceeded():
                return json.dumps(
                    {"isError": True, "summary": _BUDGET_SENTINEL}, ensure_ascii=False
                )
            return original_fn(*args, **kwargs)

        tool.func = guarded

    if original_coroutine is not None:

        @wraps(original_coroutine)
        async def guarded_async(*args, **kwargs):
            if _exceeded():
                return json.dumps(
                    {"isError": True, "summary": _BUDGET_SENTINEL}, ensure_ascii=False
                )
            return await original_coroutine(*args, **kwargs)

        tool.coroutine = guarded_async
    return tool


def build_subagent_executor(settings: Any, model: Any | None = None):
    """构建真实执行器闭包。``model`` 可注入（测试用 Fake LLM）。"""

    def executor(definition: SubagentDefinition, task: SubagentTask, ctx: Any) -> dict[str, Any]:
        from travel_agent.agent.lc_tools import build_tools

        active_model = model
        if active_model is None:
            if not settings.llm.enabled:
                raise RuntimeError("LLM disabled: subagent executor unavailable")
            from travel_agent.agent.runtime import _build_chat_model

            active_model = _build_chat_model(settings)

        all_tools = build_tools(ctx, settings)
        allowed = set(definition.tool_names)
        counter: dict[str, int] = {"n": 0}
        tools = [
            _budget_guard(tool, definition.max_tool_calls, counter)
            for tool in all_tools
            if tool.name in allowed
        ]
        missing = allowed - {tool.name for tool in tools}
        if missing:
            raise RuntimeError(f"tool whitelist unsatisfied: {sorted(missing)}")

        agent = create_react_agent(active_model, tools)
        messages: list[Any] = [
            SystemMessage(content=definition.system_prompt),
            HumanMessage(content=task.prompt_text()),
        ]

        callbacks = None
        local_trace: list[dict] = []
        if ctx is not None and getattr(ctx, "evaluation_trace_enabled", False):
            from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

            callbacks = [
                EvaluationTraceCallback(
                    ctx.evaluation_trace,
                    model=settings.llm.model,
                    phase=f"subagent_{definition.name}",
                )
            ]
        else:
            from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

            callbacks = [
                EvaluationTraceCallback(
                    local_trace, model=getattr(settings.llm, "model", ""), phase=f"subagent_{definition.name}"
                )
            ]

        state = agent.invoke(
            {"messages": messages},
            config={
                "recursion_limit": max(definition.max_steps * 2, 8),
                "callbacks": callbacks,
            },
        )
        return _extract(definition, state, local_trace)

    return executor


def _extract(definition: SubagentDefinition, state: dict, trace: list[dict]) -> dict[str, Any]:
    out_messages = state.get("messages", [])
    tool_trace = [
        call["name"]
        for msg in out_messages
        if isinstance(msg, AIMessage)
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    budget_exhausted = any(
        isinstance(msg, ToolMessage) and _BUDGET_SENTINEL in str(msg.content)
        for msg in out_messages
    )
    summary = ""
    for msg in reversed(out_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            summary = msg.content.strip()
            break

    total_tokens = 0
    for record in trace:
        usage = record.get("usage") or {}
        if isinstance(usage, dict) and usage.get("total_tokens"):
            total_tokens += int(usage["total_tokens"])

    result: dict[str, Any] = {
        "status": STATUS_BUDGET_EXHAUSTED if budget_exhausted else STATUS_COMPLETED,
        "summary": summary,
        "tool_trace": tool_trace,
        "token_usage": {"total_tokens": total_tokens} if total_tokens else {},
    }
    if budget_exhausted:
        result["warnings"] = ["工具调用次数达到上限，后续调用被拒绝"]
    return result
