"""真实 LLM 的受限 ReAct Subagent 执行器（Step 3）。

- 按 ``SubagentDefinition.tool_names`` 白名单过滤 ``build_tools`` 全量工具；
- ``create_agent`` 受限执行（recursion_limit = max_steps * 2）；
- ``max_tool_calls`` 超限后拒绝后续工具调用（结果带 budget_exhausted 告警）；
- token_usage 由执行区间的 EvaluationTraceCallback 汇总；
- 不持任何 session 锁；异常由 SubagentRunner 统一包装为 failed。

注意：本模块只构建执行器闭包，供 ``SubagentRunner`` 注入；
Orchestrator / fixed dispatch 的编排在 engine 层。
"""

from __future__ import annotations

from functools import wraps
import time
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp

from travel_agent.orchestration.multi_agent.registry import SubagentDefinition
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    SubagentTask,
)

_BUDGET_SENTINEL = "__SUBAGENT_TOOL_BUDGET_EXHAUSTED__"  # backward-compatible audit marker

_PER_TOOL_CALL_LIMITS: dict[str, dict[str, int]] = {
    "attraction": {"search_poi": 1, "check_weather": 1},
    "hotel": {"search_hotel": 2},
    "restaurant": {"search_restaurant": 1, "estimate_budget": 1},
    "transport": {"search_poi": 2, "estimate_budget": 1},
    "planner": {
        "build_constraints": 1,
        "recommend_candidates": 1,
        "plan_and_critique": 1,
    },
}


class ToolCallBudgetExceeded(GraphBubbleUp):
    """Stop the graph before a tool call that would exceed the hard budget."""


def _budget_guard(
    tool: Any,
    max_calls: int,
    counter: dict[str, Any],
    per_tool_max: int | None = None,
):
    """工具调用计数守卫：在执行下一次超额调用前立即终止。"""
    original_coroutine = getattr(tool, "coroutine", None)

    def _claim_call() -> None:
        tool_name = str(getattr(tool, "name", ""))
        per_tool_counts = counter.setdefault("by_name", {})
        if per_tool_max is not None and per_tool_counts.get(tool_name, 0) >= per_tool_max:
            raise ToolCallBudgetExceeded(
                f"{tool_name} call limit exhausted before call "
                f"{per_tool_counts.get(tool_name, 0) + 1}"
            )
        if counter["n"] >= max_calls:
            raise ToolCallBudgetExceeded(
                f"max_tool_calls exhausted before call {counter['n'] + 1}"
            )
        counter["n"] += 1
        per_tool_counts[tool_name] = per_tool_counts.get(tool_name, 0) + 1
        counter.setdefault("names", []).append(tool_name)

    if tool.func is not None:
        original_fn = tool.func

        @wraps(original_fn)
        def guarded(*args, **kwargs):
            _claim_call()
            return original_fn(*args, **kwargs)

        tool.func = guarded

    if original_coroutine is not None:

        @wraps(original_coroutine)
        async def guarded_async(*args, **kwargs):
            _claim_call()
            return await original_coroutine(*args, **kwargs)

        tool.coroutine = guarded_async
    return tool


def build_subagent_executor(settings: Any, model: Any | None = None):
    """构建真实执行器闭包。``model`` 可注入（测试用 Fake LLM）。"""

    def executor(definition: SubagentDefinition, task: SubagentTask, ctx: Any) -> dict[str, Any]:
        # Planner has a closed, mandatory three-tool contract. Asking the LLM
        # to rediscover that fixed order adds latency and can time out before
        # the first tool call, leaving otherwise complete domain evidence
        # without an itinerary. Execute the contract directly; the semantic
        # Reviewer remains the independent LLM quality gate afterwards.
        if definition.name == "planner":
            return _enforce_planner_postcondition(
                task,
                ctx,
                {
                    "status": STATUS_COMPLETED,
                    "summary": "Planner 已按绑定 artifacts 执行确定性规划工具链。",
                    "tool_trace": [],
                },
            )

        from travel_agent.agent.tool_source import resolve_tools

        active_model = model
        if active_model is None:
            if not settings.llm.enabled:
                raise RuntimeError("LLM disabled: subagent executor unavailable")
            from travel_agent.agent.runtime import _build_chat_model

            active_model = _build_chat_model(settings, max_retries=0)

        orchestration = getattr(settings, "orchestration", None)
        max_output_tokens = int(
            getattr(
                orchestration,
                (
                    "planner_max_output_tokens"
                    if definition.name == "planner"
                    else "worker_max_output_tokens"
                ),
                1024,
            )
            or 1024
        )
        allowed = set(definition.tool_names)
        all_tools, _source = resolve_tools(ctx, settings) if allowed else ([], "local")
        counter: dict[str, int] = {"n": 0}
        tools = [
            _budget_guard(
                tool,
                definition.max_tool_calls,
                counter,
                _PER_TOOL_CALL_LIMITS.get(definition.name, {}).get(tool.name),
            )
            for tool in all_tools
            if tool.name in allowed
        ]
        missing = allowed - {tool.name for tool in tools}
        if missing:
            raise RuntimeError(f"tool whitelist unsatisfied: {sorted(missing)}")

        agent = create_agent(
            active_model,
            tools,
            system_prompt=definition.system_prompt,
            middleware=[_max_output_tokens_middleware(max_output_tokens)],
        )
        messages: list[Any] = [
            HumanMessage(content=task.prompt_text()),
        ]

        from travel_agent.orchestration.meter import meter_callbacks

        callbacks = meter_callbacks(
            "planner" if definition.name == "planner" else f"worker:{definition.name}"
        )
        local_trace: list[dict] = []
        if ctx is not None and getattr(ctx, "evaluation_trace_enabled", False):
            from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

            callbacks.append(
                EvaluationTraceCallback(
                    ctx.evaluation_trace,
                    model=settings.llm.model,
                    phase=f"subagent_{definition.name}",
                )
            )
        else:
            from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

            callbacks.append(
                EvaluationTraceCallback(
                    local_trace, model=getattr(settings.llm, "model", ""), phase=f"subagent_{definition.name}"
                )
            )

        try:
            state = agent.invoke(
                {"messages": messages},
                config={
                    # LangGraph's recursion counter is the graph-step budget.  Do
                    # not silently widen it: max_steps is a hard upper bound.
                    "recursion_limit": definition.max_steps,
                    "callbacks": callbacks,
                },
            )
        except ToolCallBudgetExceeded as exc:
            # A worker commonly asks for one redundant tool call after it has
            # already persisted useful evidence.  Reject that call, but do not
            # discard the completed work or make Reviewer repair an otherwise
            # healthy domain result.
            has_evidence = _task_has_artifacts(ctx, task)
            result = {
                "status": STATUS_COMPLETED if has_evidence else STATUS_BUDGET_EXHAUSTED,
                "summary": (
                    "领域证据已完成；冗余工具调用已被硬预算拒绝。"
                    if has_evidence
                    else str(exc)
                ),
                "warnings": ["额外工具调用已被硬预算拒绝"],
                "tool_trace": list(counter.get("names") or []),
            }
            if definition.name == "attraction":
                result = _enforce_required_poi_postcondition(task, ctx, result)
            elif definition.name == "transport":
                result = _enforce_transport_route_postcondition(task, ctx, result)
            elif definition.name == "planner":
                result = _enforce_planner_postcondition(task, ctx, result)
            return result
        except Exception as exc:
            if type(exc).__name__ in {"GraphRecursionError", "RecursionError"}:
                return {
                    "status": STATUS_BUDGET_EXHAUSTED,
                    "summary": f"max_steps exhausted: {definition.max_steps}",
                    "warnings": ["执行步骤达到 max_steps 硬上限"],
                    "tool_trace": [],
                }
            raise
        result = _extract(definition, state, local_trace)
        if definition.name == "attraction":
            result = _enforce_required_poi_postcondition(task, ctx, result)
        elif definition.name == "transport":
            result = _enforce_transport_route_postcondition(task, ctx, result)
        elif definition.name == "planner":
            result = _enforce_planner_postcondition(task, ctx, result)
        return result

    return executor


def _max_output_tokens_middleware(max_output_tokens: int) -> Any:
    """Apply the output cap through the LangChain 1.x agent model request."""

    @wrap_model_call
    def apply_output_cap(request, handler):
        model_settings = dict(request.model_settings or {})
        model_settings["max_tokens"] = max_output_tokens
        return handler(request.override(model_settings=model_settings))

    return apply_output_cap


def _task_has_artifacts(ctx: Any, task: SubagentTask) -> bool:
    store = getattr(ctx, "store", None)
    if store is None:
        return False
    return bool(
        store.artifact_ids_for_task(task.request_id, task.task_id, agent=task.agent)
    )


def _enforce_required_poi_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Ensure every hard must/fixed venue has an explicit search artifact.

    Broad interest searches are stochastic and often omit a named venue even
    though it is a hard constraint.  The Attraction worker therefore gets a
    deterministic, bounded search for each still-missing required POI.  This
    is keyed only by the current profile, never by evaluation case IDs.
    """
    from travel_agent.agent import toolkit
    from travel_agent.agent.serde import poi_from_dict
    from travel_agent.critic import poi_matches_interest, poi_matches_must_visit

    required = [str(item).strip() for item in getattr(ctx.profile, "must_visit", []) if str(item).strip()]
    raw_interests = (getattr(ctx.profile, "constraint_state", {}) or {}).get("interests") or []
    if isinstance(raw_interests, str):
        raw_interests = [raw_interests]
    structured_interests = [str(item).strip() for item in raw_interests if str(item).strip()]
    if not required and not structured_interests:
        return result

    def task_pois() -> list[Any]:
        pois: list[Any] = []
        store = getattr(ctx, "store", None)
        if store is None:
            return pois
        for artifact_id in store.artifact_ids_for_task(
            task.request_id, task.task_id, agent="attraction"
        ):
            record = store.get_record(artifact_id) or {}
            if record.get("kind") != "candidates":
                continue
            payload = store.get(artifact_id) or {}
            for raw in payload.get("pois") or []:
                if not isinstance(raw, dict):
                    continue
                try:
                    pois.append(poi_from_dict(raw))
                except Exception:  # noqa: BLE001 - malformed provider item is not evidence
                    continue
        return pois

    repaired: list[str] = []
    for term in required:
        if any(poi_matches_must_visit(poi, term) for poi in task_pois()):
            continue
        search = toolkit.search_poi(
            ctx,
            city=getattr(ctx.profile, "destination", None),
            interests=[term],
            max_results=10,
        )
        repaired.append("search_poi")
        if search.get("isError"):
            result.setdefault("warnings", []).append(
                f"required POI search failed: {term}"
            )

    for interest in structured_interests:
        if any(poi_matches_interest(poi, interest) for poi in task_pois()):
            continue
        search = toolkit.search_poi(
            ctx,
            city=getattr(ctx.profile, "destination", None),
            interests=[interest],
            max_results=10,
        )
        repaired.append("search_poi")
        if search.get("isError"):
            result.setdefault("warnings", []).append(
                f"structured interest search failed: {interest}"
            )

    unresolved = [
        term
        for term in required
        if not any(poi_matches_must_visit(poi, term) for poi in task_pois())
    ]
    result.setdefault("tool_trace", []).extend(repaired)
    if repaired:
        result.setdefault("warnings", []).append(
            "required POI/interest search postcondition applied deterministically"
        )
    if unresolved:
        result["status"] = STATUS_FAILED
        result.setdefault("unresolved", []).extend(
            f"未检索到必去地点：{term}" for term in unresolved
        )
        result["summary"] = "required POI postcondition failed: " + ", ".join(unresolved)
    return result


def _enforce_transport_route_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Materialize one route when Transport resolved endpoints but omitted it.

    A bounded ReAct worker can spend its search allowance resolving two POIs
    and then ask for a redundant search instead of ``plan_route``.  Once two
    referenced POI ids exist, route evidence is deterministic, so enforce the
    domain contract without widening model or tool budgets.
    """
    store = getattr(ctx, "store", None)
    if store is None:
        return result

    artifact_ids = list(
        dict.fromkeys(
            [str(item) for item in (task.inputs.get("artifact_ids") or []) if item]
            + store.artifact_ids_for_task(task.request_id, task.task_id, agent="transport")
        )
    )
    if any(
        (store.get_record(artifact_id) or {}).get("kind") == "routes"
        for artifact_id in artifact_ids
    ):
        return result

    poi_ids: list[str] = []
    for artifact_id in artifact_ids:
        if (store.get_record(artifact_id) or {}).get("kind") not in {"candidates", "pois"}:
            continue
        payload = store.get(artifact_id) or {}
        for raw in payload.get("pois") or []:
            if not isinstance(raw, dict):
                continue
            poi_id = str(raw.get("poi_id") or "").strip()
            if poi_id and poi_id not in poi_ids and ctx.poi(poi_id) is not None:
                poi_ids.append(poi_id)
            if len(poi_ids) >= 2:
                break
        if len(poi_ids) >= 2:
            break
    if len(poi_ids) < 2:
        return result

    from travel_agent.agent import toolkit

    route = toolkit.plan_route(ctx, poi_ids[0], poi_ids[1])
    result.setdefault("tool_trace", []).append("plan_route")
    if route.get("isError"):
        result.setdefault("warnings", []).append(
            "transport route postcondition failed after endpoint resolution"
        )
        return result
    result.setdefault("warnings", []).append(
        "transport route postcondition applied deterministically"
    )
    return result


def _enforce_planner_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """A completed Planner must own an itinerary, regardless of LLM tool order."""
    store = getattr(ctx, "store", None)
    if store is None:
        result["status"] = STATUS_FAILED
        result["summary"] = "planner postcondition failed: artifact store unavailable"
        return result

    def task_ids(kind: str | None = None) -> list[str]:
        ids = store.artifact_ids_for_task(task.request_id, task.task_id, agent="planner")
        if kind is None:
            return ids
        return [
            artifact_id
            for artifact_id in ids
            if (store.get_record(artifact_id) or {}).get("kind") == kind
        ]

    if task_ids("itinerary"):
        return result

    from travel_agent.agent import toolkit
    from travel_agent.orchestration.meter import current_turn_meter

    repaired: list[str] = []
    bound_ids = [str(item) for item in (task.inputs.get("artifact_ids") or []) if item]

    def call(name: str, fn, **kwargs: Any) -> dict[str, Any]:
        meter = current_turn_meter()
        meter_started = meter.begin_tool("planner_repair") if meter is not None else None
        started = time.perf_counter()
        failed = True
        try:
            value = fn(ctx, **kwargs)
            failed = bool(value.get("isError"))
            return value
        finally:
            if getattr(ctx, "evaluation_trace_enabled", False):
                ctx.evaluation_trace.append(
                    {
                        "kind": "tool",
                        "name": name,
                        "arguments": dict(kwargs),
                        "status": "error" if failed else "ok",
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "phase": "planner_postcondition_repair",
                    }
                )
            if meter is not None and meter_started is not None:
                meter.finish_tool("planner_repair", meter_started, error=failed)
            repaired.append(name)

    try:
        if not task_ids("constraints"):
            call("build_constraints", toolkit.build_constraints)
        if not task_ids("ranked"):
            ranked = call(
                "recommend_candidates",
                toolkit.recommend_candidates,
                # recommend_candidates may read only the task's explicitly
                # bound domain evidence. Planner-produced constraints are not
                # candidate inputs and are intentionally excluded here.
                artifact_ids=bound_ids,
            )
            if ranked.get("isError"):
                raise RuntimeError(str(ranked.get("summary") or "candidate ranking failed"))
        planned = call(
            "plan_and_critique",
            toolkit.plan_and_critique,
            artifact_ids=bound_ids + task_ids(),
            route_estimator=_planner_route_estimator(ctx, bound_ids),
        )
        if planned.get("isError") or not task_ids("itinerary"):
            raise RuntimeError(str(planned.get("summary") or "itinerary artifact missing"))
    except Exception as exc:  # noqa: BLE001 - convert the postcondition to a result status
        result["status"] = STATUS_FAILED
        result["summary"] = f"planner postcondition failed: {type(exc).__name__}: {exc}"
    else:
        result["status"] = STATUS_COMPLETED
        result.setdefault("warnings", []).append(
            "planner tool-order postcondition repaired deterministically"
        )
    result.setdefault("tool_trace", []).extend(repaired)
    return result


def _planner_route_estimator(ctx: Any, artifact_ids: list[str]) -> Any:
    """Build a no-network estimator from bound route evidence plus local geometry."""
    from travel_agent.providers import LocalToolProvider
    from travel_agent.schemas import RouteInfo

    cached: dict[tuple[str, str, str], RouteInfo] = {}
    store = getattr(ctx, "store", None)
    if store is not None:
        for artifact_id in artifact_ids:
            record = store.get_record(artifact_id) or {}
            if record.get("kind") != "routes":
                continue
            payload = store.get(artifact_id) or {}
            try:
                route = RouteInfo(
                    origin_poi_id=str(payload["origin_poi_id"]),
                    destination_poi_id=str(payload["destination_poi_id"]),
                    distance_km=float(payload["distance_km"]),
                    duration_min=int(payload["duration_min"]),
                    mode=str(payload.get("mode") or "public_transport"),
                    source=str(payload.get("source") or "bound_route_artifact"),
                    walking_distance_km=(
                        float(payload["walking_distance_km"])
                        if payload.get("walking_distance_km") is not None
                        else None
                    ),
                )
            except (KeyError, TypeError, ValueError):
                continue
            cached[(route.origin_poi_id, route.destination_poi_id, route.mode)] = route

    local = LocalToolProvider(list(getattr(ctx, "pois_by_id", {}).values()))

    class BoundRouteEstimator:
        def estimate_route(self, origin, destination, mode="public_transport"):
            key = (origin.poi_id, destination.poi_id, mode)
            if key in cached:
                return cached[key]
            reverse = cached.get((destination.poi_id, origin.poi_id, mode))
            if reverse is not None:
                return RouteInfo(
                    origin_poi_id=origin.poi_id,
                    destination_poi_id=destination.poi_id,
                    distance_km=reverse.distance_km,
                    duration_min=reverse.duration_min,
                    mode=mode,
                    source=f"{reverse.source}_reversed",
                    walking_distance_km=reverse.walking_distance_km,
                )
            fallback = local.estimate_route(origin, destination, mode)
            return RouteInfo(
                origin_poi_id=fallback.origin_poi_id,
                destination_poi_id=fallback.destination_poi_id,
                distance_km=fallback.distance_km,
                duration_min=fallback.duration_min,
                mode=fallback.mode,
                source="haversine_recovery_estimate",
                walking_distance_km=fallback.walking_distance_km,
            )

    return BoundRouteEstimator()


def _extract(definition: SubagentDefinition, state: dict, trace: list[dict]) -> dict[str, Any]:
    out_messages = state.get("messages", [])
    tool_trace = [
        call["name"]
        for msg in out_messages
        if isinstance(msg, AIMessage)
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
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
        "status": STATUS_COMPLETED,
        "summary": summary,
        "tool_trace": tool_trace,
        "token_usage": {"total_tokens": total_tokens} if total_tokens else {},
    }
    return result
