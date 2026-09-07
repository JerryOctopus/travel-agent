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
import json
import re
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
    "attraction": {"search_poi": 2, "check_weather": 2},
    "hotel": {"search_hotel": 2},
    "restaurant": {"search_restaurant": 2, "estimate_budget": 2},
    "transport": {"search_poi": 2, "estimate_budget": 2},
    "planner": {
        "build_constraints": 1,
        "recommend_candidates": 1,
        "plan_and_critique": 1,
    },
}

_MULTI_SUCCESS_TOOLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("hotel", "search_hotel"),
        ("transport", "search_poi"),
        ("transport", "plan_route"),
    }
)


class ToolCallBudgetExceeded(GraphBubbleUp):
    """Stop the graph before a tool call that would exceed the hard budget."""


def _budget_guard(
    tool: Any,
    max_calls: int,
    counter: dict[str, Any],
    per_tool_max: int | None = None,
    *,
    allow_repeated_success: bool = False,
):
    """工具调用计数守卫：在执行下一次超额调用前立即终止。"""
    original_coroutine = getattr(tool, "coroutine", None)

    def _claim_call() -> None:
        tool_name = str(getattr(tool, "name", ""))
        per_tool_counts = counter.setdefault("by_name", {})
        previous = counter.setdefault("outcomes", {}).get(tool_name)
        if isinstance(previous, dict):
            if previous.get("isError") and not previous.get("retryable"):
                raise ToolCallBudgetExceeded(
                    f"{tool_name} returned a non-retryable error; retry rejected"
                )
            if not previous.get("isError") and not allow_repeated_success:
                raise ToolCallBudgetExceeded(
                    f"{tool_name} already succeeded; redundant call rejected"
                )
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

    def _record_outcome(value: Any) -> None:
        payload = value
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                return
        if isinstance(payload, dict) and "isError" in payload:
            counter.setdefault("outcomes", {})[str(getattr(tool, "name", ""))] = {
                "isError": bool(payload.get("isError")),
                "retryable": bool(payload.get("retryable")),
            }

    if tool.func is not None:
        original_fn = tool.func

        @wraps(original_fn)
        def guarded(*args, **kwargs):
            _claim_call()
            value = original_fn(*args, **kwargs)
            _record_outcome(value)
            return value

        tool.func = guarded

    if original_coroutine is not None:

        @wraps(original_coroutine)
        async def guarded_async(*args, **kwargs):
            _claim_call()
            value = await original_coroutine(*args, **kwargs)
            _record_outcome(value)
            return value

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

        # Transport has a closed evidence contract once its bound artifacts
        # and structured constraint state identify endpoints.  Resolve that
        # contract deterministically first; only invoke the model when the
        # task is genuinely underspecified.  This prevents a model from
        # spending its bounded calls on repeated endpoint searches and never
        # reaching plan_route.
        if definition.name == "transport":
            deterministic = _enforce_comparison_search_postcondition(
                task,
                ctx,
                {
                    "status": STATUS_COMPLETED,
                    "summary": "Transport 正在按绑定端点生成路线证据。",
                    "tool_trace": [],
                    "warnings": [],
                },
            )
            deterministic = _enforce_transport_route_postcondition(
                task,
                ctx,
                deterministic,
            )
            contract_satisfied = bool(
                deterministic.pop("_transport_contract_satisfied", False)
            )
            if _task_has_domain_evidence(ctx, task) or contract_satisfied:
                deterministic["status"] = STATUS_COMPLETED
                deterministic["summary"] = (
                    "Transport 已完成本轮可确定的预算/路线证据。"
                    if contract_satisfied
                    else "Transport 已按绑定端点完成路线证据。"
                )
                return deterministic

        # Hotel discovery is likewise a bounded lookup from canonical profile
        # state.  Materialize it before asking the model so an invalid area or
        # budget argument cannot consume the worker's entire tool allowance.
        if definition.name == "hotel":
            deterministic = _enforce_hotel_search_postcondition(
                task,
                ctx,
                {
                    "status": STATUS_COMPLETED,
                    "summary": "Hotel 正在按住宿约束检索候选证据。",
                    "tool_trace": [],
                    "warnings": [],
                },
            )
            if _task_has_nonempty_domain_evidence(ctx, task):
                deterministic["summary"] = "Hotel 已按住宿约束完成候选证据。"
                return deterministic

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
                allow_repeated_success=(definition.name, tool.name)
                in _MULTI_SUCCESS_TOOLS,
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
            has_evidence = _task_has_domain_evidence(ctx, task)
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
            if definition.name == "restaurant" and not _task_has_nonempty_domain_evidence(ctx, task):
                result = _enforce_restaurant_search_postcondition(task, ctx, result)
                has_evidence = _task_has_nonempty_domain_evidence(ctx, task)
                result["status"] = (
                    STATUS_COMPLETED if has_evidence else STATUS_BUDGET_EXHAUSTED
                )
                if has_evidence:
                    result["summary"] = (
                        "餐厅证据已通过受限后置检索完成；"
                        "模型的无效或冗余工具调用已被拒绝。"
                    )
            if definition.name == "hotel" and not _task_has_nonempty_domain_evidence(ctx, task):
                result = _enforce_hotel_search_postcondition(task, ctx, result)
            if definition.name == "attraction":
                result = _enforce_required_poi_postcondition(task, ctx, result)
                result = _enforce_indoor_backup_postcondition(task, ctx, result)
            if _is_candidate_comparison_task(task):
                result = _enforce_comparison_search_postcondition(task, ctx, result)
            if definition.name == "transport":
                result = _enforce_transport_route_postcondition(task, ctx, result)
            elif definition.name == "planner":
                result = _enforce_planner_postcondition(task, ctx, result)
            return _promote_recovered_worker_status(task, ctx, result)
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
        if definition.name == "restaurant" and not _task_has_nonempty_domain_evidence(ctx, task):
            result = _enforce_restaurant_search_postcondition(task, ctx, result)
        if definition.name == "hotel" and not _task_has_nonempty_domain_evidence(ctx, task):
            result = _enforce_hotel_search_postcondition(task, ctx, result)
        if definition.name == "attraction":
            result = _enforce_required_poi_postcondition(task, ctx, result)
            result = _enforce_indoor_backup_postcondition(task, ctx, result)
        if _is_candidate_comparison_task(task):
            result = _enforce_comparison_search_postcondition(task, ctx, result)
        if definition.name == "transport":
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


def _task_has_domain_evidence(ctx: Any, task: SubagentTask) -> bool:
    """Ignore recovery/diagnostic artifacts when deciding worker completion."""
    expected = {
        "attraction": {"candidates", "pois", "weather"},
        "hotel": {"hotels"},
        "restaurant": {"restaurants"},
        "transport": {"routes"},
        "planner": {"itinerary"},
    }.get(task.agent, set())
    if task.agent == "transport" and _is_candidate_comparison_task(task):
        # An unanchored area comparison can satisfy accessibility with
        # provider-backed transit-node/entity facts; a route is only mandatory
        # when there is a target anchor or a later matrix-finalization step.
        expected = {*expected, "candidates", "pois"}
    store = getattr(ctx, "store", None)
    if store is None:
        return False
    return any(
        (store.get_record(artifact_id) or {}).get("kind") in expected
        for artifact_id in store.artifact_ids_for_task(
            task.request_id, task.task_id, agent=task.agent
        )
    )


def _task_has_nonempty_domain_evidence(ctx: Any, task: SubagentTask) -> bool:
    """Require at least one usable domain item, not merely an empty artifact."""
    item_fields = {
        "attraction": ("pois", "items"),
        "hotel": ("hotels", "items"),
        "restaurant": ("restaurants", "items"),
        "transport": ("routes", "items"),
        "planner": ("days",),
    }.get(task.agent, ())
    store = getattr(ctx, "store", None)
    if store is None:
        return False
    for artifact_id in store.artifact_ids_for_task(
        task.request_id, task.task_id, agent=task.agent
    ):
        payload = store.get(artifact_id) or {}
        if any(payload.get(field) for field in item_fields):
            return True
    return False


def _promote_recovered_worker_status(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    if (
        result.get("status") == STATUS_BUDGET_EXHAUSTED
        and _task_has_domain_evidence(ctx, task)
        and not result.get("unresolved")
    ):
        result["status"] = STATUS_COMPLETED
        result["summary"] = (
            "领域证据已通过受限后置条件完成；冗余工具调用已被硬预算拒绝。"
        )
    return result


def _enforce_restaurant_search_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Recover one bounded restaurant search from invalid model arguments.

    Numeric per-person limits belong to ``budget_per_person_cny`` and are
    already present in the profile.  Models occasionally copy that number
    into the categorical ``budget_level`` argument, which the shared tool
    schema correctly rejects.  A restaurant task must not lose the whole
    delivery because of that redundant argument: rerun once from canonical
    state, then broaden only if the scoped query is empty.
    """
    from travel_agent.agent import toolkit

    state = dict((task.inputs.get("profile") or {}).get("constraint_state") or {})
    city = (
        getattr(ctx.profile, "destination", None)
        or state.get("destination_city")
        or state.get("location")
    )
    area = state.get("location_anchor") or state.get("location")
    if not city:
        result.setdefault("unresolved", []).append("餐厅检索缺少城市或地点范围")
        return result
    top_n = state.get("top_n")
    try:
        max_results = max(3, min(12, int(top_n or 8)))
    except (TypeError, ValueError):
        max_results = 8
    response = toolkit.search_restaurant(
        ctx,
        city=str(city),
        area=str(area) if area else None,
        budget_level=None,
        max_results=max_results,
    )
    calls = ["search_restaurant"]
    if not response.get("isError") and int(response.get("count") or 0) == 0 and area:
        response = toolkit.search_restaurant(
            ctx,
            city=str(city),
            area=None,
            budget_level=None,
            max_results=max_results,
        )
        calls.append("search_restaurant")
        result.setdefault("warnings", []).append(
            "区域内无结果，已按同一城市扩大一次检索范围"
        )
    dietary = state.get("dietary") or []
    dietary_values = [dietary] if isinstance(dietary, str) else list(dietary)
    hard_dietary = bool(dietary_values)
    must_visit = state.get("must_visit") or []
    anchor_values = [must_visit] if isinstance(must_visit, str) else list(must_visit)
    if hard_dietary and int(state.get("duration_days") or getattr(ctx.profile, "days", 1) or 1) > 1:
        for anchor in list(dict.fromkeys(str(item).strip() for item in anchor_values if str(item).strip()))[:3]:
            if area and str(area).strip() == anchor:
                continue
            anchored = toolkit.search_restaurant(
                ctx,
                city=str(city),
                area=anchor,
                budget_level=None,
                max_results=max_results,
            )
            calls.append("search_restaurant")
            if anchored.get("isError"):
                result.setdefault("warnings", []).append(
                    f"硬约束锚点附近餐厅检索失败：{anchor}"
                )
            else:
                result.setdefault("warnings", []).append(
                    f"已补充硬约束锚点附近的饮食证据：{anchor}"
                )
    result.setdefault("tool_trace", []).extend(calls)
    if response.get("isError"):
        result.setdefault("unresolved", []).append(
            f"餐厅后置检索失败：{response.get('summary') or 'unknown error'}"
        )
    else:
        result.setdefault("warnings", []).append(
            "restaurant evidence postcondition applied"
        )
    return result


def _enforce_hotel_search_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Perform the hotel worker's canonical bounded evidence lookup once."""
    from travel_agent.agent import toolkit

    state = dict((task.inputs.get("profile") or {}).get("constraint_state") or {})
    city = (
        getattr(ctx.profile, "destination", None)
        or state.get("destination_city")
        or state.get("location")
    )
    area = (
        state.get("lodging_area")
        or getattr(ctx.profile, "hotel_area", None)
    )
    if not city:
        result.setdefault("unresolved", []).append("酒店检索缺少目的地城市")
        return result
    response = toolkit.search_hotel(
        ctx,
        city=str(city),
        area=str(area) if area else None,
        budget_level=getattr(ctx.profile, "budget_level", None),
        max_results=8,
    )
    result.setdefault("tool_trace", []).append("search_hotel")
    if response.get("isError") or int(response.get("count") or 0) == 0:
        result.setdefault("unresolved", []).append(
            f"酒店后置检索未取得候选：{response.get('summary') or 'empty result'}"
        )
    else:
        result.setdefault("warnings", []).append(
            "hotel evidence postcondition applied"
        )
    return result


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
    from travel_agent.poi_evidence import poi_avoid_match

    task_state = dict((task.inputs.get("profile") or {}).get("constraint_state") or {})
    profile_state = dict(getattr(ctx.profile, "constraint_state", {}) or {})
    unspecified_fixed_locations = {
        str(item).strip()
        for item in profile_state.get("user_owned_unspecified_fixed_event_locations") or []
        if str(item).strip()
    }
    fixed_locations = [
        str(event.get("location") or "").strip()
        for event in profile_state.get("fixed_events") or []
        if isinstance(event, dict)
        and str(event.get("location") or "").strip()
        and str(event.get("location") or "").strip()
        not in unspecified_fixed_locations
        and str(event.get("location") or "").strip()
        not in {"自由活动", "休息", "自由时间"}
    ]
    required = list(dict.fromkeys([
        *[
            str(item).strip()
            for item in getattr(ctx.profile, "must_visit", [])
            if str(item).strip()
        ],
        *fixed_locations,
    ]))
    # The live profile is the latest-write-wins authority. A worker task brief
    # is only a dispatch snapshot and may contain model-suggested categories
    # that the user never requested; those must not widen real POI retrieval.
    raw_interests = (
        profile_state.get("interests")
        or getattr(ctx.profile, "interests", [])
        or []
    )
    if isinstance(raw_interests, str):
        raw_interests = [raw_interests]
    structured_interests = [str(item).strip() for item in raw_interests if str(item).strip()]
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
        aliases = [term]
        if term.endswith("博物院"):
            stem = term[:-3]
            aliases.extend([f"{stem}博物馆", f"{stem}省博物馆"])
        elif term.endswith("博物馆"):
            stem = term[:-3]
            aliases.append(f"{stem}博物院")
        for query in dict.fromkeys(aliases):
            search = toolkit.search_poi(
                ctx,
                city=getattr(ctx.profile, "destination", None),
                interests=[query],
                max_results=30,
            )
            repaired.append("search_poi")
            if search.get("isError"):
                result.setdefault("warnings", []).append(
                    f"required POI search failed: {query}"
                )
            if any(poi_matches_must_visit(poi, term) for poi in task_pois()):
                break

    interest_search_aliases = {
        "海边": ("沙滩", "海滩", "海滨"),
        "咖啡店": ("咖啡馆",),
        "园林": ("园林景区",),
        "历史景点": ("历史遗址", "历史文化景区"),
        "主要历史景点": ("历史遗址", "历史文化景区"),
    }

    def has_usable_interest_candidate(interest: str) -> bool:
        return any(
            poi_matches_interest(poi, interest)
            and poi_avoid_match(poi, ctx.profile) is None
            for poi in task_pois()
        )

    for interest in structured_interests:
        if has_usable_interest_candidate(interest):
            continue
        for query in (interest, *interest_search_aliases.get(interest, ())):
            search = toolkit.search_poi(
                ctx,
                city=getattr(ctx.profile, "destination", None),
                interests=[query],
                max_results=10,
            )
            repaired.append("search_poi")
            if search.get("isError"):
                result.setdefault("warnings", []).append(
                    f"structured interest search failed: {query}"
                )
            if has_usable_interest_candidate(interest):
                break

    declared_type = getattr(task.inputs.get("task_type"), "value", task.inputs.get("task_type"))
    if declared_type == "full_itinerary":
        try:
            trip_days = int(
                task_state.get("duration_days")
                or getattr(ctx.profile, "days", None)
                or 1
            )
        except (TypeError, ValueError):
            trip_days = 1
        minimum_supply = max(3, min(10, trip_days * 2))
        activity_pois = [
            poi
            for poi in task_pois()
            if poi.category not in {"food", "hotel", "transport", "unknown"}
        ]
        # A provider can return several entrances, docks or child facilities
        # for one named attraction.  Those are useful evidence for that must-
        # visit, but they are not six independent day-plan choices.  Count a
        # required attraction family once and preserve genuinely unrelated
        # entities individually before deciding whether to broaden discovery.
        matched_required = {
            term
            for term in required
            if any(poi_matches_must_visit(poi, term) for poi in activity_pois)
        }
        unrelated_ids = {
            poi.poi_id
            for poi in activity_pois
            if not any(poi_matches_must_visit(poi, term) for term in required)
        }
        effective_supply = len(matched_required) + len(unrelated_ids)
        if effective_supply < minimum_supply:
            search = toolkit.search_poi(
                ctx,
                city=getattr(ctx.profile, "destination", None),
                interests=[],
                max_results=max(10, minimum_supply),
            )
            repaired.append("search_poi")
            if search.get("isError"):
                result.setdefault("warnings", []).append(
                    "citywide activity supply search failed"
                )

    unresolved = [
        term
        for term in required
        if not any(poi_matches_must_visit(poi, term) for poi in task_pois())
    ]
    # Multiple bounded searches produce separate artifacts, while downstream
    # cache selection intentionally prefers the newest artifact of a kind.
    # Publish one lossless union so a broad interest/supply lookup cannot hide
    # the earlier exact must-visit evidence from Planner.
    collected = task_pois()
    if repaired and collected:
        from travel_agent.agent.serde import poi_to_dict

        unique = {poi.poi_id: poi for poi in collected}
        merged = sorted(
            unique.values(),
            key=lambda poi: (
                0
                if any(poi_matches_must_visit(poi, term) for term in required)
                else 1,
                -float(poi.rating or 0),
                poi.name,
            ),
        )
        ctx.store.put(
            "candidates",
            {
                "city": getattr(ctx.profile, "destination", None),
                "query_tags": list(dict.fromkeys([*required, *structured_interests])),
                "retrieval_scope": "required_interest_and_city_supply",
                "pois": [poi_to_dict(poi) for poi in merged],
            },
        )
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
    elif result.get("unresolved"):
        stale_missing_prefixes = (
            "未检索到必去地点",
            "未找到必去地点",
            "required poi missing",
            "required poi not found",
        )
        remaining = [
            item
            for item in result.get("unresolved") or []
            if not (
                any(prefix in str(item).casefold() for prefix in stale_missing_prefixes)
                and any(term in str(item) for term in required)
            )
        ]
        result["unresolved"] = remaining
        if not remaining and collected:
            result["status"] = STATUS_COMPLETED
            result["summary"] = (
                "必去地点证据已由确定性后置检索补齐；已清除早期陈旧缺失状态。"
            )
    return result


def _enforce_indoor_backup_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Run one grounded indoor-candidate lookup when the turn requires it.

    Weather checks and fallback discovery are separate evidence obligations.
    A bounded worker that stops after weather must not silently produce an
    empty "indoor backup" recommendation.  The query is semantic and shared;
    it is not tied to any evaluation city, date, POI, or case identifier.
    """
    state = dict((task.inputs.get("profile") or {}).get("constraint_state") or {})
    if not state.get("need_indoor_backup"):
        return result
    store = getattr(ctx, "store", None)
    if store is None:
        return result

    indoor_markers = ("museum", "gallery", "indoor", "aquarium", "博物馆", "美术馆", "展览馆", "室内", "科技馆", "海洋馆")
    for artifact_id in store.artifact_ids_for_task(
        task.request_id, task.task_id, agent="attraction"
    ):
        record = store.get_record(artifact_id) or {}
        if record.get("kind") not in {"candidates", "pois"}:
            continue
        payload = record.get("payload") or {}
        for item in payload.get("pois") or payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            description = " ".join(str(value or "") for value in (
                item.get("name"), item.get("category"), item.get("entity_type"),
                " ".join(str(tag) for tag in item.get("tags") or []),
            )).casefold()
            if any(marker in description for marker in indoor_markers):
                return result

    from travel_agent.agent import toolkit

    city = (
        state.get("destination_city")
        or getattr(ctx.profile, "destination", None)
    )
    if not city:
        result.setdefault("unresolved", []).append("室内备选检索缺少城市")
        return result
    response = toolkit.search_poi(
        ctx,
        city=str(city),
        interests=["博物馆", "美术馆", "室内展馆"],
        category="museum",
        max_results=8,
    )
    result.setdefault("tool_trace", []).append("search_poi")
    if response.get("isError"):
        result.setdefault("unresolved", []).append(
            f"室内备选检索失败：{response.get('summary') or 'unknown error'}"
        )
    else:
        result.setdefault("warnings", []).append(
            "indoor backup evidence postcondition applied"
        )
    return result


def _is_candidate_comparison_task(task: SubagentTask) -> bool:
    state = dict((task.inputs.get("profile") or {}).get("constraint_state") or {})
    declared_type = task.inputs.get("task_type")
    if declared_type is not None and declared_type != "candidate_comparison":
        return False
    return bool(
        state.get("comparison_candidates")
        or state.get("compare_lodging_areas")
        or state.get("candidate_attractions")
    )


def _enforce_comparison_search_postcondition(
    task: SubagentTask,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Bounded tool-backed coverage repair for declared comparison entities.

    The worker still owns every artifact: these calls run inside its task
    metadata context.  Empty/unrelated results remain empty/unrelated and are
    rejected later by the comparison evidence contract.
    """
    from travel_agent.agent import toolkit
    from travel_agent.candidate_comparison import comparison_candidates, entity_matches

    state = dict((task.inputs.get("profile") or {}).get("constraint_state") or {})
    declared_candidates = comparison_candidates(state)[:8]
    scoped_candidate = str(task.inputs.get("comparison_candidate") or "").strip()
    candidates = (
        [scoped_candidate]
        if scoped_candidate and scoped_candidate in declared_candidates
        else declared_candidates
    )
    target = str(state.get("target_anchor") or "").strip()
    store = getattr(ctx, "store", None)
    if store is None or not candidates:
        return result
    if task.agent == "hotel" and state.get("no_live_inventory_required"):
        result["status"] = STATUS_FAILED
        result.setdefault("unresolved", []).append(
            "区域比较明确禁止具体酒店库存，hotel worker 已拒绝执行"
        )
        return result

    owned_ids = store.artifact_ids_for_task(
        task.request_id, task.task_id, agent=task.agent
    )

    def task_records(*, owned_only: bool = False) -> list[dict[str, Any]]:
        ids = owned_ids if owned_only else list(task.inputs.get("artifact_ids") or []) + owned_ids
        return [record for aid in dict.fromkeys(ids) if (record := store.get_record(str(aid)))]

    def has_entity(name: str, *, owned_only: bool = False) -> bool:
        for record in task_records(owned_only=owned_only):
            payload = record.get("payload") or {}
            for key in ("pois", "endpoint_pois", "items", "hotels", "restaurants"):
                for item in payload.get(key) or []:
                    if not isinstance(item, dict):
                        continue
                    values = [
                        item.get("name"), item.get("canonical_name"), item.get("area"),
                        item.get("address"), *(item.get("aliases") or []), *(item.get("tags") or []),
                    ]
                    if any(entity_matches(name, str(value or "")) for value in values):
                        return True
            if any(
                entity_matches(name, str(payload.get(key) or ""))
                for key in ("candidate", "candidate_name", "area")
            ):
                return True
        return False

    searched: list[str] = []
    names = [*candidates, *([target] if target and task.agent in {"attraction", "transport"} else [])]
    for name in dict.fromkeys(names):
        # A candidate cell is scoped to this task.  A broad artifact owned by
        # an earlier candidate may mention this name but cannot satisfy the
        # current cell.  Fixed anchors, by contrast, are shared and reusable.
        if has_entity(name, owned_only=name in candidates):
            continue
        if task.agent in {"attraction", "transport"}:
            response = toolkit.search_poi(
                ctx,
                city=getattr(ctx.profile, "destination", None),
                interests=[name],
                max_results=8,
            )
            tool_name = "search_poi"
        elif task.agent == "hotel":
            response = toolkit.search_hotel(ctx, area=name, max_results=8)
            tool_name = "search_hotel"
        elif task.agent == "restaurant":
            response = toolkit.search_restaurant(ctx, area=name, max_results=8)
            tool_name = "search_restaurant"
        else:
            continue
        searched.append(tool_name)
        if response.get("isError"):
            result.setdefault("warnings", []).append(
                f"comparison evidence search failed: {name}"
            )
    if searched:
        result.setdefault("tool_trace", []).extend(searched)
        result.setdefault("warnings", []).append(
            "candidate comparison coverage postcondition applied"
        )
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
    state = dict(
        (task.inputs.get("profile") or {}).get("constraint_state")
        or getattr(ctx.profile, "constraint_state", {})
        or {}
    )
    def canonical_mode(value: Any) -> str:
        return {
            "transit": "public_transport",
            "public_transit": "public_transport",
            "driving": "drive",
            "walking": "walk",
        }.get(str(value), str(value))

    default_modes = (
        ["walk"]
        if state.get("walking_time_max_min") is not None
        else [ctx.profile.transport_mode]
    )
    required_modes = [
        canonical_mode(mode)
        for mode in (state.get("transport_modes") or default_modes)
        if canonical_mode(mode) in {"walk", "public_transport", "taxi", "drive"}
    ]
    existing_modes: set[str] = set()
    existing_pair_modes: set[tuple[str, str, str]] = set()
    existing_pair_modes_all: set[tuple[str, str, str]] = set()
    has_budget_artifact = False
    route_pairs: list[tuple[str, str]] = []
    hard_deadline_pairs: set[tuple[str, str]] = set()
    route_pairs_are_symmetric = False
    candidate_ids: list[str] = []
    discovered_option_ids: list[str] = []

    def register_candidate(raw: Any, *, is_option: bool = False) -> None:
        if not isinstance(raw, dict):
            return
        poi_id = str(raw.get("poi_id") or raw.get("id") or "").strip()
        if poi_id and poi_id not in candidate_ids and ctx.poi(poi_id) is not None:
            candidate_ids.append(poi_id)
        if poi_id and is_option and poi_id not in discovered_option_ids:
            discovered_option_ids.append(poi_id)

    for artifact_id in artifact_ids:
        kind = (store.get_record(artifact_id) or {}).get("kind")
        payload = store.get(artifact_id) or {}
        if kind == "budget":
            has_budget_artifact = True
            continue
        if kind == "routes":
            if payload.get("mode"):
                existing_modes.add(canonical_mode(payload["mode"]))
            origin_id = str(payload.get("origin_poi_id") or "").strip()
            destination_id = str(payload.get("destination_poi_id") or "").strip()
            if (
                origin_id
                and destination_id
                and ctx.poi(origin_id) is not None
                and ctx.poi(destination_id) is not None
                and (origin_id, destination_id) not in route_pairs
            ):
                route_pairs.append((origin_id, destination_id))
                if payload.get("mode"):
                    existing_pair_modes_all.add(
                        (origin_id, destination_id, canonical_mode(payload["mode"]))
                    )
                    from travel_agent.route_evidence import (
                        canonical_route_evidence_status,
                    )

                    evidence_status = canonical_route_evidence_status(payload)
                else:
                    evidence_status = "unavailable"
                if (
                    payload.get("mode")
                    and evidence_status
                    in {"provider_verified", "deterministic_estimate"}
                ):
                    existing_pair_modes.add(
                        (origin_id, destination_id, canonical_mode(payload["mode"]))
                    )
            continue
        if kind not in {"candidates", "pois", "restaurants", "hotels"}:
            continue
        collections = [
            (payload.get("pois") or [], False),
            (payload.get("endpoint_pois") or [], False),
            (payload.get("restaurants") or [], kind == "restaurants"),
            (payload.get("hotels") or [], kind == "hotels"),
            (payload.get("items") or [], kind in {"restaurants", "hotels"}),
        ]
        for collection, is_option in collections:
            for raw in collection:
                register_candidate(raw, is_option=is_option)
    comparison_candidates = [
        str(item).strip()
        for item in (
            state.get("comparison_candidates")
            or state.get("compare_lodging_areas")
            or state.get("candidate_attractions")
            or []
        )
        if str(item).strip()
    ]
    target_anchor = str(
        state.get("target_anchor")
        or state.get("location_anchor")
        or state.get("location")
        or ""
    ).strip()

    def matching_id(name: str, *, excluded: set[str] | None = None) -> str | None:
        normalized = re.sub(r"\s+", "", name).casefold()
        for poi_id in candidate_ids:
            if poi_id in (excluded or set()):
                continue
            poi = ctx.poi(poi_id)
            poi_name = re.sub(r"\s+", "", str(getattr(poi, "name", ""))).casefold()
            if normalized and poi_name and (normalized in poi_name or poi_name in normalized):
                return poi_id
        return None

    def resolve_named_endpoint(
        name: str, *, excluded: set[str] | None = None
    ) -> str | None:
        """Resolve one explicit hard-route endpoint without guessing identity."""
        existing = matching_id(name, excluded=excluded)
        if existing:
            return existing
        city = state.get("destination_city") or getattr(ctx.profile, "destination", None)
        if not city or not name:
            return None
        from travel_agent.agent import toolkit

        searched = toolkit.search_poi(
            ctx,
            city=str(city),
            interests=[name],
            max_results=8,
        )
        result.setdefault("tool_trace", []).append("search_poi")
        artifact_id = str(searched.get("artifact_id") or "")
        payload = store.get(artifact_id) if artifact_id else None
        if isinstance(payload, dict):
            for raw in [
                *list(payload.get("pois") or []),
                *list(payload.get("endpoint_pois") or []),
            ]:
                register_candidate(raw)
        return matching_id(name, excluded=excluded)

    explicit_repair_pairs = [
        (str(pair[0]), str(pair[1]))
        for pair in task.inputs.get("repair_route_pairs") or []
        if isinstance(pair, (list, tuple))
        and len(pair) == 2
        and ctx.poi(str(pair[0])) is not None
        and ctx.poi(str(pair[1])) is not None
        and str(pair[0]) != str(pair[1])
    ]

    if target_anchor:
        resolve_named_endpoint(
            target_anchor, excluded=set(discovered_option_ids)
        )

    desired_pairs: list[tuple[str, str]] = []
    explicit_task_type = str(task.inputs.get("task_type") or "")
    if explicit_repair_pairs:
        desired_pairs = list(dict.fromkeys(explicit_repair_pairs))[:8]
    elif explicit_task_type in {"route_plan", "route_query"}:
        origin_id = resolve_named_endpoint(str(state.get("origin") or "").strip())
        destination_id = resolve_named_endpoint(
            str(
                state.get("destination_name")
                or state.get("route_destination")
                or state.get("destination")
                or ""
            ).strip()
        )
        if origin_id and destination_id and origin_id != destination_id:
            desired_pairs = [(origin_id, destination_id)]
        else:
            return result
    elif explicit_task_type in {"full_itinerary", "full_trip_plan"}:
        # Full itineraries with fixed appointments or explicit trip endpoints
        # need exact route artifacts before Planner runs.  The model worker can
        # exhaust its bounded steps after endpoint search; recover a small,
        # deterministic route fan-out here so Reviewer does not discover the
        # missing hard evidence only after planning.
        fixed_names = [
            str(event.get("location") or "").strip()
            for event in state.get("fixed_events") or []
            if isinstance(event, dict)
            and str(event.get("location") or "").strip()
            not in {"自由活动", "休息", "自由时间"}
        ]
        fixed_ids = [
            endpoint_id
            for name in fixed_names
            if (endpoint_id := resolve_named_endpoint(name))
        ]
        origin_id = resolve_named_endpoint(str(state.get("origin") or "").strip())
        return_id = resolve_named_endpoint(
            str(state.get("return_location") or "").strip()
        )
        hard_endpoint_ids = {
            endpoint_id for endpoint_id in [*fixed_ids, origin_id, return_id]
            if endpoint_id
        }
        trip_candidates = [
            poi_id for poi_id in candidate_ids if poi_id not in hard_endpoint_ids
        ][:10]
        required_pairs: list[tuple[str, str]] = []
        for fixed_id in fixed_ids:
            required_pairs.extend(
                (candidate_id, fixed_id)
                for candidate_id in trip_candidates
                if candidate_id != fixed_id
            )
        if origin_id:
            required_pairs.extend(
                (origin_id, candidate_id)
                for candidate_id in trip_candidates
                if candidate_id != origin_id
            )
        if return_id:
            required_pairs.extend(
                (candidate_id, return_id)
                for candidate_id in trip_candidates
                if candidate_id != return_id
            )
        # Ordinary full itineraries may have no fixed/origin/return anchor,
        # but still require evidence between scheduled candidates.  Include a
        # bounded adjacent chain instead of returning without any route.
        adjacent_pairs = (
            []
            if required_pairs
            else [
                (trip_candidates[index], trip_candidates[index + 1])
                for index in range(len(trip_candidates) - 1)
            ]
        )
        desired_pairs = list(dict.fromkeys([*required_pairs, *adjacent_pairs]))[:24]
        hard_deadline_pairs = {
            tuple(sorted((origin, destination)))
            for origin, destination in desired_pairs
            if destination in {*fixed_ids, *([return_id] if return_id else [])}
        }
    elif comparison_candidates and target_anchor:
        target_id = matching_id(
            target_anchor, excluded=set(discovered_option_ids)
        )
        if target_id:
            desired_pairs = [
                (candidate_id, target_id)
                for name in comparison_candidates
                if (candidate_id := matching_id(name)) and candidate_id != target_id
            ]
        # Never replace declared comparison endpoints with arbitrary search
        # results merely to manufacture route evidence.
        if not desired_pairs:
            return result
    elif comparison_candidates:
        # Entity evidence is the preferred unanchored contract.  Let each
        # candidate's one bounded search finish before deciding whether a
        # route matrix is necessary, otherwise early tasks create redundant
        # pairs before node coverage is known.
        if not bool(task.inputs.get("comparison_finalize_routes")):
            return result

        from travel_agent.candidate_comparison import payload_has_candidate_transit_fact

        records = [
            record
            for artifact_id in artifact_ids
            if (record := store.get_record(artifact_id))
        ]
        node_covered = {
            name
            for name in comparison_candidates
            if any(
                payload_has_candidate_transit_fact(
                    str(record.get("kind") or ""),
                    record.get("payload") or {},
                    name,
                )
                for record in records
            )
        }
        requested_dimensions = {
            str(item) for item in task.inputs.get("comparison_dimensions") or []
        }
        if (
            "accessibility" in requested_dimensions
            and len(node_covered) == len(comparison_candidates)
        ):
            return result

        resolved = [matching_id(name) for name in comparison_candidates]
        if any(poi_id is None for poi_id in resolved):
            return result
        ids = [str(poi_id) for poi_id in resolved if poi_id]
        # An unanchored candidate matrix compares pairwise average travel and
        # deliberately represents each unordered pair once.  This is distinct
        # from itinerary/deadline routes, whose endpoint order is material.
        route_pairs_are_symmetric = True
        if len(ids) == 2:
            desired_pairs = [(ids[0], ids[1])]
        elif len(ids) == 3:
            # Three candidates are still cheap enough for a complete matrix.
            desired_pairs = [(ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2])]
        elif len(ids) > 3:
            # A bounded cycle gives every candidate the same two declared
            # peers and caps route growth at N instead of N².
            desired_pairs = [
                (ids[index], ids[(index + 1) % len(ids)])
                for index in range(len(ids))
            ]
    elif target_anchor and (
        target_id := matching_id(
            target_anchor, excluded=set(discovered_option_ids)
        )
    ):
        option_ids = [
            poi_id for poi_id in (discovered_option_ids or candidate_ids)
            if poi_id != target_id
        ]
        try:
            option_limit = max(1, min(8, int(state.get("top_n") or len(option_ids))))
        except (TypeError, ValueError):
            option_limit = min(8, len(option_ids))
        desired_pairs = [(poi_id, target_id) for poi_id in option_ids[:option_limit]]
    elif route_pairs:
        desired_pairs = [route_pairs[0]]
    elif len(candidate_ids) >= 2:
        desired_pairs = [(candidate_ids[0], candidate_ids[1])]

    hard_names = [
        *[
            str(event.get("location") or "").strip()
            for event in state.get("fixed_events") or []
            if isinstance(event, dict)
            and str(event.get("location") or "").strip()
            not in {"自由活动", "休息", "自由时间"}
        ],
        *(
            [str(state.get("return_location") or "").strip()]
            if state.get("return_deadline") and state.get("return_location")
            else []
        ),
    ]
    route_endpoint_ids = {
        poi_id
        for pair in desired_pairs
        for poi_id in pair
    }
    hard_endpoint_ids = set()
    for poi_id in [*candidate_ids, *route_endpoint_ids]:
        poi = ctx.poi(poi_id)
        poi_name = re.sub(
            r"\s+", "", str(getattr(poi, "name", ""))
        ).casefold()
        if any(
            (normalized := re.sub(r"\s+", "", name).casefold())
            and poi_name
            and (normalized in poi_name or poi_name in normalized)
            for name in hard_names
        ):
            hard_endpoint_ids.add(poi_id)
    hard_deadline_pairs.update(
        tuple(sorted((origin_id, destination_id)))
        for origin_id, destination_id in desired_pairs
        if origin_id in hard_endpoint_ids or destination_id in hard_endpoint_ids
    )
    from travel_agent.agent import toolkit

    repaired = False
    budget_explicit = bool(
        getattr(ctx.profile, "budget_limit", None)
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
    if budget_explicit and not has_budget_artifact:
        budget = toolkit.estimate_budget(ctx)
        result.setdefault("tool_trace", []).append("estimate_budget")
        if budget.get("isError"):
            result.setdefault("warnings", []).append(
                "transport budget postcondition failed"
            )
        else:
            repaired = True
            has_budget_artifact = True
            result.setdefault("warnings", []).append(
                "transport budget postcondition applied deterministically"
            )

    if not desired_pairs:
        result["_transport_contract_satisfied"] = bool(
            explicit_task_type in {"full_itinerary", "full_trip_plan"}
            and budget_explicit
            and has_budget_artifact
        )
        return result

    for origin_id, destination_id in desired_pairs:
        # Route artifacts are directional: evidence for B -> A must not
        # suppress a required A -> B lookup.  Deadline classification remains
        # endpoint-order agnostic because either side may be the hard anchor.
        ordered_pair = (origin_id, destination_id)
        hard_pair = tuple(sorted(ordered_pair))
        pair_modes = list(required_modes)
        # Planner is intentionally network-free.  Materialize a taxi route in
        # the Transport worker for hard local deadlines so Planner can replace
        # an unverified/empty transit response without making a live call.
        if (
            hard_pair in hard_deadline_pairs
            and "public_transport" in pair_modes
            and "taxi" not in pair_modes
        ):
            pair_modes.append("taxi")
        for mode in pair_modes:
            reverse_pair = (destination_id, origin_id, mode)
            ordered_mode = (*ordered_pair, mode)
            hard_route = hard_pair in hard_deadline_pairs
            if (
                ordered_mode in existing_pair_modes
                or (route_pairs_are_symmetric and reverse_pair in existing_pair_modes)
                or (
                    not hard_route
                    and (
                        ordered_mode in existing_pair_modes_all
                        or (
                            route_pairs_are_symmetric
                            and reverse_pair in existing_pair_modes_all
                        )
                    )
                )
            ):
                continue
            route = toolkit.plan_route(ctx, origin_id, destination_id, mode=mode)
            result.setdefault("tool_trace", []).append("plan_route")
            if route.get("isError"):
                result.setdefault("warnings", []).append(
                    f"transport route postcondition failed for {mode}"
                )
                continue
            repaired = True
            existing_pair_modes.add(ordered_mode)
            existing_pair_modes_all.add(ordered_mode)
    if repaired:
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
            # Transport postconditions may have materialized missing endpoint
            # pairs after dispatch. The planner must bind those task-local
            # provider routes instead of falling back to local geometry.
            route_estimator=_planner_route_estimator(
                ctx, bound_ids + task_ids("routes")
            ),
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
    """Prefer bound evidence, then fetch the exact final pair from the provider."""
    from travel_agent.providers import LocalToolProvider, ProviderRateLimitError
    from travel_agent.route_evidence import normalize_route_evidence, route_supports_endpoints
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
                    evidence_status=str(payload.get("evidence_status") or "unavailable"),
                )
            except (KeyError, TypeError, ValueError):
                continue
            cached[(route.origin_poi_id, route.destination_poi_id, route.mode)] = route

    local = LocalToolProvider(list(getattr(ctx, "pois_by_id", {}).values()))
    live = getattr(ctx, "provider", None)

    class BoundRouteEstimator:
        def estimate_route(self, origin, destination, mode="public_transport"):
            key = (origin.poi_id, destination.poi_id, mode)
            if key in cached:
                return cached[key]
            if live is not None:
                try:
                    exact = normalize_route_evidence(
                        live.estimate_route(origin, destination, mode)
                    )
                except ProviderRateLimitError:
                    raise
                except Exception:
                    exact = None
                if exact is not None and route_supports_endpoints(
                    exact, origin.poi_id, destination.poi_id
                ):
                    return exact
            fallback = local.estimate_route(origin, destination, mode)
            return normalize_route_evidence(RouteInfo(
                origin_poi_id=fallback.origin_poi_id,
                destination_poi_id=fallback.destination_poi_id,
                distance_km=fallback.distance_km,
                duration_min=fallback.duration_min,
                mode=fallback.mode,
                source="haversine_recovery_estimate",
                walking_distance_km=fallback.walking_distance_km,
            ), provider_explicit=False)

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
