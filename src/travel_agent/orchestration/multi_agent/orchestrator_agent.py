"""Bounded, one-call-per-wave dynamic Router for V2/V3."""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from travel_agent.orchestration.multi_agent.orchestrator import (
    DispatchLedger,
    ORCHESTRATOR_ALLOWED_TOOLS,
    build_dispatch_tool,
    filter_orchestrator_tools,
)
from travel_agent.orchestration.multi_agent.registry import list_subagents
from travel_agent.orchestration.multi_agent.schemas import SubagentResult

DOMAIN_AGENTS = frozenset({"attraction", "hotel", "restaurant", "transport"})
ROUTER_MAX_OUTPUT_TOKENS = 768

_ROUTER_PROMPT_TEMPLATE = """你是旅行规划 Router。你每次只决定一个执行 wave，不做调研、不写行程。

可派发领域：
{subagents}

规则：
1. 只能派 attraction、hotel、restaurant、transport；绝不能派 planner/reviewer/repair。
2. tasks 必须是当前 wave 可并行启动的任务；依赖既有结果时填写 depends_on task_id。
   完整行程的 transport 若需要景点候选 poi_id，不能与 attraction 在同一 wave 猜测执行，
   应等待候选 artifact 后作为后续 delta task；只有起终点已明确时才可独立并行。
3. 不重复 attempted_objectives 中已完成的目标；已有 evidence 足够时返回 ready=true、tasks=[]。
4. Wave 1 最多四个独立任务；后续 wave 只返回缺失 hard evidence 所需的 delta tasks。
5. 信息不足以安全派工时 clarification=true，不猜测目的地或天数。
6. 只输出 JSON，不输出 markdown。格式：
{{"ready": false, "clarification": false, "reply": "", "missing_evidence": [],
  "reason": "", "tasks": [{{"agent": "attraction", "instruction": "...",
  "objective": "...", "depends_on": []}}]}}
"""


@dataclass(frozen=True)
class RoutingTask:
    agent: str
    instruction: str
    objective: str
    depends_on: tuple[str, ...] = ()

    @property
    def objective_key(self) -> str:
        normalized = re.sub(r"\s+", " ", self.objective or self.instruction).strip().casefold()
        return f"{self.agent}:{normalized}"


@dataclass(frozen=True)
class RoutingDecision:
    tasks: tuple[RoutingTask, ...] = ()
    ready: bool = False
    clarification: bool = False
    missing_evidence: tuple[str, ...] = ()
    reason: str = ""
    reply: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "tasks": [asdict(task) for task in self.tasks],
        }


def build_orchestrator_prompt() -> str:
    lines = "\n".join(
        f"- {definition.name}: {definition.description}"
        for definition in list_subagents()
        if definition.name in DOMAIN_AGENTS
    )
    return _ROUTER_PROMPT_TEMPLATE.format(subagents=lines)


def routing_policy_hash(settings: Any) -> str:
    llm = getattr(settings, "llm", None)
    orch = getattr(settings, "orchestration", None)
    policy = {
        "prompt": build_orchestrator_prompt(),
        "schema": {
            "agents": sorted(DOMAIN_AGENTS),
            "response_format": "json_object",
            "max_output_tokens": ROUTER_MAX_OUTPUT_TOKENS,
            "fields": [
                "ready",
                "clarification",
                "reply",
                "missing_evidence",
                "reason",
                "tasks.agent",
                "tasks.instruction",
                "tasks.objective",
                "tasks.depends_on",
            ],
        },
        "model": {
            "provider": getattr(llm, "provider", ""),
            "model": getattr(llm, "model", ""),
            "temperature": getattr(llm, "temperature", None),
            "thinking_enabled": getattr(llm, "thinking_enabled", None),
        },
        "budgets": {
            name: getattr(orch, name, None)
            for name in (
                "variant_token_budget",
                "variant_llm_call_budget",
                "variant_tool_call_budget",
                "base_timeout_seconds",
                "planner_reserve_seconds",
                "reviewer_reserve_seconds",
                "router_timeout_seconds",
                "router_useful_seconds",
                "worker_timeout_seconds",
                "planner_timeout_seconds",
                "worker_max_output_tokens",
                "planner_max_output_tokens",
                "reviewer_max_output_tokens",
                "recovery_worker_useful_seconds",
                "admission_guard_seconds",
                "routing_max_waves",
                "routing_max_calls",
                "routing_max_dispatches",
                "routing_wave1_max_tasks",
            )
        },
    }
    encoded = json.dumps(policy, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def route_wave(
    ctx: Any,
    settings: Any,
    user_message: str,
    request_id: str,
    *,
    wave: int,
    model: Any | None = None,
    history: list[tuple[str, str]] | None = None,
    task_type: Any | None = None,
    turn_inputs: dict[str, Any] | None = None,
    compact_results: list[dict[str, Any]] | None = None,
    attempted_objectives: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    timeout_seconds: float,
    max_tasks: int,
) -> RoutingDecision:
    """Make exactly one bounded model call and return a validated wave decision."""
    del request_id  # identity remains in trace; Router receives no mutable store handle
    if timeout_seconds <= 0:
        return RoutingDecision(error="router admission denied: no usable pre-planner time")

    active_model = model
    if active_model is None:
        llm = getattr(settings, "llm", None)
        if llm is None or not getattr(llm, "enabled", False):
            return RoutingDecision(error="LLM disabled: router unavailable")
        from travel_agent.agent.runtime import _build_chat_model

        active_model = _build_chat_model(
            settings,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )

    snapshot = {
        "wave": wave,
        "task_type": getattr(task_type, "value", task_type) or "unknown",
        "request": user_message,
        "turn_inputs": turn_inputs or {},
        "history": list(history or [])[-4:],
        "compact_results": compact_results or [],
        "attempted_objectives": attempted_objectives or [],
        "engine_missing_hard_evidence": missing_evidence or [],
        "max_tasks": max_tasks,
    }
    messages = [
        SystemMessage(content=build_orchestrator_prompt()),
        HumanMessage(
            content="本轮 Router snapshot（JSON）：\n"
            + json.dumps(snapshot, ensure_ascii=False, default=str)
        ),
    ]
    from travel_agent.orchestration.meter import meter_callbacks

    def invoke() -> Any:
        bounded = active_model.bind(
            max_tokens=ROUTER_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )
        callbacks = meter_callbacks("orchestrator")
        if ctx is not None and getattr(ctx, "evaluation_trace_enabled", False):
            from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

            callbacks.append(
                EvaluationTraceCallback(
                    ctx.evaluation_trace,
                    model=getattr(settings.llm, "model", ""),
                    phase=f"router_wave_{wave}",
                )
            )
        return bounded.invoke(messages, config={"callbacks": callbacks})

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"router-wave-{wave}")
    future = pool.submit(contextvars.copy_context().run, invoke)
    try:
        response = future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        from travel_agent.orchestration.meter import current_turn_meter

        meter = current_turn_meter()
        if meter is not None:
            meter.fail_pending_llm(
                "orchestrator", f"router timeout after {timeout_seconds:g}s"
            )
        return RoutingDecision(error=f"router timeout after {timeout_seconds:g}s")
    except Exception as exc:  # noqa: BLE001
        return RoutingDecision(error=f"{type(exc).__name__}: {exc}")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    raw = _response_payload(response)
    return _parse_decision(raw, max_tasks=max_tasks)


def _response_payload(response: Any) -> dict[str, Any]:
    content = getattr(response, "content", "")
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    parsed = _extract_json(str(content or ""))
    if parsed:
        return parsed
    # Compatibility with scripted tool-call fakes: consume only this single
    # response; never enter a ReAct loop.
    tool_calls = getattr(response, "tool_calls", None) or []
    tasks = []
    clarification = False
    for call in tool_calls:
        if call.get("name") == "request_travel_info":
            clarification = True
            continue
        if call.get("name") != "dispatch_subagent":
            continue
        args = call.get("args") or {}
        tasks.append(
            {
                "agent": args.get("agent"),
                "instruction": args.get("instruction"),
                "objective": args.get("objective") or args.get("instruction"),
                "depends_on": args.get("depends_on") or [],
            }
        )
    return {"tasks": tasks, "clarification": clarification, "ready": not tasks}


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if "```" in candidate:
        for part in candidate.split("```"):
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                candidate = stripped
                break
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_decision(raw: dict[str, Any], *, max_tasks: int) -> RoutingDecision:
    if not raw:
        return RoutingDecision(error="router returned no JSON decision")
    tasks: list[RoutingTask] = []
    seen: set[str] = set()
    for item in raw.get("tasks") or []:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "").strip().lower()
        instruction = str(item.get("instruction") or "").strip()
        objective = str(item.get("objective") or instruction).strip()
        if agent not in DOMAIN_AGENTS or not instruction or not objective:
            continue
        task = RoutingTask(
            agent=agent,
            instruction=instruction,
            objective=objective,
            depends_on=tuple(str(value) for value in (item.get("depends_on") or []) if value),
        )
        if task.objective_key in seen:
            continue
        seen.add(task.objective_key)
        tasks.append(task)
        if len(tasks) >= max(0, max_tasks):
            break
    return RoutingDecision(
        tasks=tuple(tasks),
        ready=bool(raw.get("ready")),
        clarification=bool(raw.get("clarification")),
        missing_evidence=tuple(str(value) for value in (raw.get("missing_evidence") or []) if value),
        reason=str(raw.get("reason") or ""),
        reply=str(raw.get("reply") or ""),
    )


# Compatibility helpers for local unit tests and external imports. Production
# dynamic execution uses route_wave + Engine-owned wave execution.
def build_orchestrator_tools(
    runner: Any,
    ctx: Any,
    settings: Any,
    request_id: str,
    results_sink: list | None = None,
    ledger: DispatchLedger | None = None,
    base_inputs: dict[str, Any] | None = None,
    task_type: Any | None = None,
) -> tuple[list[Any], list]:
    from travel_agent.agent.tool_source import resolve_tools

    del task_type
    sink = results_sink if results_sink is not None else []
    all_tools, _source = resolve_tools(ctx, settings)
    allowed = ORCHESTRATOR_ALLOWED_TOOLS - {"render_itinerary", "render_map"}
    base_tools = [tool for tool in filter_orchestrator_tools(all_tools) if tool.name in allowed]
    dispatch = build_dispatch_tool(
        runner,
        request_id=request_id,
        ledger=ledger,
        base_inputs=base_inputs,
    )

    def collect(agent: str, instruction: str, inputs=None, depends_on=None) -> str:
        parsed = json.loads(dispatch(agent, instruction, inputs, depends_on))
        sink.append(parsed)
        compact = {
            key: parsed.get(key)
            for key in (
                "request_id", "task_id", "agent", "status", "attempt", "summary",
                "evidence", "warnings", "unresolved", "error",
            )
            if parsed.get(key) not in (None, "", [], {})
        }
        return json.dumps(compact, ensure_ascii=False, default=str)

    return base_tools + [
        StructuredTool.from_function(
            collect,
            name="dispatch_subagent",
            description="Compatibility one-shot domain dispatch tool.",
        )
    ], sink


def run_orchestrator(
    ctx: Any,
    settings: Any,
    user_message: str,
    request_id: str,
    *,
    runner: Any,
    model: Any | None = None,
    results_sink: list | None = None,
    history: list[tuple[str, str]] | None = None,
    task_type: Any | None = None,
    turn_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deprecated one-wave adapter; it never performs iterative routing."""
    timeout = float(getattr(getattr(settings, "orchestration", None), "router_timeout_seconds", 20) or 20)
    decision = route_wave(
        ctx,
        settings,
        user_message,
        request_id,
        wave=1,
        model=model,
        history=history,
        task_type=task_type,
        turn_inputs=turn_inputs,
        timeout_seconds=timeout,
        max_tasks=4,
    )
    sink = results_sink if results_sink is not None else []
    ledger = DispatchLedger()
    if decision.tasks:
        dispatch = build_dispatch_tool(
            runner, request_id=request_id, ledger=ledger, base_inputs=turn_inputs
        )
        with ThreadPoolExecutor(max_workers=len(decision.tasks)) as pool:
            futures = [
                pool.submit(
                    contextvars.copy_context().run,
                    dispatch,
                    task.agent,
                    task.instruction,
                    None,
                    list(task.depends_on),
                )
                for task in decision.tasks
            ]
            sink.extend(json.loads(future.result()) for future in futures)
    return {
        "reply": decision.reply or ("领域调研已完成。" if sink else ""),
        "tool_trace": ["dispatch_subagent"] if sink else [],
        "results": sink,
        "clarification": decision.clarification,
        "dispatch_ledger": ledger,
        "routing_decision": decision.to_dict(),
    }
