"""Shared runtime contract for every travel toolkit entry point.

The contract deliberately sits below LangChain and MCP adapters so local tools,
remote tools, and deterministic postcondition repairs observe identical input,
error, artifact-ownership, and concurrency behaviour.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Mapping, TypeVar


class ToolErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Execution properties used by the shared pre-call pipeline.

    Policies fail closed: an unregistered tool is session-serial and is not
    assumed to produce a durable artifact.
    """

    parallel_safe: bool = False
    mutates_session: bool = True
    artifact_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolValidationFailure:
    code: ToolErrorCode
    summary: str
    retryable: bool = False
    details: Mapping[str, Any] | None = None

    def envelope(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "isError": True,
            "summary": self.summary,
            "error_code": self.code.value,
            "retryable": self.retryable,
        }
        if self.details:
            result["details"] = dict(self.details)
        return result


TOOL_EXECUTION_POLICIES: dict[str, ToolExecutionPolicy] = {
    "update_travel_profile": ToolExecutionPolicy(),
    "request_travel_info": ToolExecutionPolicy(parallel_safe=True, mutates_session=False),
    "request_preference_guide": ToolExecutionPolicy(parallel_safe=True, mutates_session=False),
    "search_poi": ToolExecutionPolicy(True, True, ("candidates",)),
    "check_weather": ToolExecutionPolicy(True, True, ("weather",)),
    "plan_route": ToolExecutionPolicy(True, True, ("routes",)),
    "search_restaurant": ToolExecutionPolicy(True, True, ("restaurants",)),
    "search_hotel": ToolExecutionPolicy(True, True, ("hotels",)),
    "estimate_budget": ToolExecutionPolicy(True, True, ("budget",)),
    "build_constraints": ToolExecutionPolicy(False, True, ("constraints",)),
    "recommend_candidates": ToolExecutionPolicy(False, True, ("ranked",)),
    "plan_and_critique": ToolExecutionPolicy(False, True, ("itinerary",)),
    "render_itinerary": ToolExecutionPolicy(),
    "render_map": ToolExecutionPolicy(),
}


def tool_execution_policy(tool_name: str) -> ToolExecutionPolicy:
    return TOOL_EXECUTION_POLICIES.get(tool_name, ToolExecutionPolicy())


def _invalid(summary: str, **details: Any) -> ToolValidationFailure:
    return ToolValidationFailure(
        ToolErrorCode.INVALID_INPUT,
        summary,
        retryable=False,
        details=details or None,
    )


def _positive_int(value: Any, field: str, *, maximum: int | None = None) -> ToolValidationFailure | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return _invalid(f"{field} 必须是正整数。", field=field, value=value)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return _invalid(f"{field} 必须是正整数。", field=field, value=value)
    if parsed <= 0 or (maximum is not None and parsed > maximum):
        limit = f"且不大于 {maximum}" if maximum is not None else ""
        return _invalid(f"{field} 必须是正整数{limit}。", field=field, value=value)
    return None


def _validate_city(ctx: Any, arguments: Mapping[str, Any]) -> ToolValidationFailure | None:
    city = arguments.get("city")
    if city is not None and not str(city).strip():
        return _invalid("city 不能为空。", field="city")
    destination = getattr(getattr(ctx, "profile", None), "destination", None)
    if city is None and not destination:
        return _invalid("缺少目的地城市。", field="city")
    return None


_BUDGET_LEVEL_ALIASES = {
    "低": "low",
    "低档": "low",
    "经济": "low",
    "经济型": "low",
    "中": "mid",
    "中等": "mid",
    "中档": "mid",
    "中端": "mid",
    "高": "high",
    "高档": "high",
    "高端": "high",
    "中高端": "high",
    "豪华": "high",
}

_POI_CATEGORY_ALIASES = {
    "景点": "scenic",
    "景区": "scenic",
    "风景名胜": "scenic",
    "scenic_area": "scenic",
    "attraction": "scenic",
    "attractions": "scenic",
    "博物馆": "museum",
    "餐厅": "food",
    "美食": "food",
    "咖啡店": "food",
    "酒店": "hotel",
    "住宿": "hotel",
    "购物": "shopping",
    "商圈": "shopping",
}


def normalize_tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize common model vocabulary without weakening tool validation."""
    normalized = dict(arguments)
    if tool_name in {
        "update_travel_profile",
        "estimate_budget",
        "search_hotel",
        "search_restaurant",
    }:
        level = normalized.get("budget_level")
        if isinstance(level, str):
            normalized["budget_level"] = _BUDGET_LEVEL_ALIASES.get(level.strip(), level)

    if tool_name == "search_poi":
        category = str(normalized.get("category") or "").strip()
        if category:
            canonical = _POI_CATEGORY_ALIASES.get(category, category)
            if canonical in {"scenic", "museum", "food", "hotel", "shopping"}:
                normalized["category"] = canonical
            else:
                interests = [
                    str(item).strip()
                    for item in (normalized.get("interests") or [])
                    if str(item).strip()
                ]
                if category not in interests:
                    interests.append(category)
                normalized["interests"] = interests
                normalized["category"] = None
    return normalized


def _validate_planner_artifacts(
    ctx: Any,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> ToolValidationFailure | None:
    from travel_agent.agent.session import current_task_meta

    meta = current_task_meta()
    if meta.get("agent") != "planner" or tool_name not in {
        "recommend_candidates",
        "plan_and_critique",
    }:
        return None
    requested_ids = [str(item) for item in (arguments.get("artifact_ids") or []) if item]
    bound_ids = [str(item) for item in (meta.get("artifact_ids") or []) if item]
    if tool_name == "plan_and_critique":
        store = getattr(ctx, "store", None)
        if store is not None:
            bound_ids.extend(
                store.artifact_ids_for_task(
                    str(meta.get("request_id") or ""),
                    str(meta.get("task_id") or ""),
                    agent="planner",
                )
            )
            bound_ids = list(dict.fromkeys(bound_ids))
    unbound = [item for item in requested_ids if item not in bound_ids]
    if unbound:
        return ToolValidationFailure(
            ToolErrorCode.PERMISSION_DENIED,
            "Planner 只能读取任务明确绑定的 artifact_ids。",
            details={"unbound_artifact_ids": unbound},
        )
    artifact_ids = requested_ids or bound_ids
    if not artifact_ids:
        return ToolValidationFailure(
            ToolErrorCode.PERMISSION_DENIED,
            "Planner 只能读取任务明确绑定的 artifact_ids。",
            details={"field": "artifact_ids"},
        )
    store = getattr(ctx, "store", None)
    missing = [item for item in artifact_ids if store is None or store.get_record(item) is None]
    if missing:
        return ToolValidationFailure(
            ToolErrorCode.PERMISSION_DENIED,
            "Planner 绑定了当前会话中不存在的 artifact。",
            details={"missing_artifact_ids": missing},
        )
    return None


def before_tool_call(
    tool_name: str,
    ctx: Any,
    arguments: Mapping[str, Any],
) -> ToolValidationFailure | None:
    """Validate semantic preconditions before provider or store mutation."""

    if tool_name in {
        "search_poi",
        "check_weather",
        "search_restaurant",
        "search_hotel",
        "estimate_budget",
    }:
        failure = _validate_city(ctx, arguments)
        if failure is not None:
            return failure

    if tool_name in {"search_poi", "search_restaurant", "search_hotel"}:
        failure = _positive_int(arguments.get("max_results"), "max_results", maximum=50)
        if failure is not None:
            return failure

    if tool_name == "search_hotel" and arguments.get("min_rating") is not None:
        try:
            rating = float(arguments["min_rating"])
        except (TypeError, ValueError):
            return _invalid("min_rating 必须是 0 到 5 的数字。", field="min_rating")
        if not 0 <= rating <= 5:
            return _invalid("min_rating 必须是 0 到 5 的数字。", field="min_rating")

    if tool_name == "estimate_budget":
        for field in ("days", "companions"):
            failure = _positive_int(arguments.get(field), field)
            if failure is not None:
                return failure

    if tool_name in {"estimate_budget", "search_hotel", "search_restaurant"}:
        level = arguments.get("budget_level")
        if level is not None and level not in {"low", "mid", "high"}:
            return _invalid("budget_level 只能是 low、mid 或 high。", field="budget_level")

    if tool_name == "plan_route":
        origin = str(arguments.get("origin_poi_id") or "").strip()
        destination = str(arguments.get("destination_poi_id") or "").strip()
        if not origin or not destination:
            return _invalid("路线起点和终点 POI ID 均不能为空。")
        if origin == destination:
            return _invalid(
                "路线起点和终点不能相同。",
                origin_poi_id=origin,
                destination_poi_id=destination,
            )
        mode = arguments.get("mode")
        valid_modes = {
            "walk",
            "public_transport",
            "taxi",
            "drive",
            "walking",
            "transit",
            "public transit",
            "public-transit",
            "driving",
        }
        if mode is not None and str(mode).strip().lower() not in valid_modes:
            return _invalid("不支持的交通方式。", field="mode", value=mode)
        missing = [poi_id for poi_id in (origin, destination) if ctx.poi(poi_id) is None]
        if missing:
            return ToolValidationFailure(
                ToolErrorCode.NOT_FOUND,
                "找不到对应 POI，请先检索并使用当前会话返回的 poi_id。",
                retryable=True,
                details={"missing_poi_ids": missing},
            )

    if tool_name == "recommend_candidates":
        failure = _positive_int(arguments.get("top_k"), "top_k", maximum=50)
        if failure is not None:
            return failure
    if tool_name == "plan_and_critique":
        failure = _positive_int(arguments.get("max_iters"), "max_iters", maximum=10)
        if failure is not None:
            return failure

    return _validate_planner_artifacts(ctx, tool_name, arguments)


def _infer_error_code(summary: str) -> ToolErrorCode:
    lowered = summary.lower()
    if any(marker in lowered for marker in ("rate limit", "限流", "429", "too many requests")):
        return ToolErrorCode.RATE_LIMITED
    if any(
        marker in lowered
        for marker in ("找不到", "没有检索到", "没有可规划", "没有符合", "not found")
    ):
        return ToolErrorCode.NOT_FOUND
    if any(marker in lowered for marker in ("缺少", "不能为空", "不支持", "必须", "请确认")):
        return ToolErrorCode.INVALID_INPUT
    if any(marker in lowered for marker in ("暂时失败", "服务", "provider", "upstream")):
        return ToolErrorCode.UPSTREAM_UNAVAILABLE
    return ToolErrorCode.INTERNAL_ERROR


def normalize_tool_result(value: Any, tool_name: str) -> dict[str, Any]:
    """Enforce the stable envelope without changing successful payloads."""

    if not isinstance(value, dict) or "isError" not in value or "summary" not in value:
        return ToolValidationFailure(
            ToolErrorCode.CONTRACT_VIOLATION,
            f"{tool_name} 返回值不符合工具信封契约。",
        ).envelope()
    result = dict(value)
    if not result.get("isError"):
        return result
    code_value = result.get("error_code")
    try:
        code = ToolErrorCode(str(code_value)) if code_value else _infer_error_code(str(result["summary"]))
    except ValueError:
        code = ToolErrorCode.CONTRACT_VIOLATION
    result["error_code"] = code.value
    result.setdefault(
        "retryable",
        code in {ToolErrorCode.NOT_FOUND, ToolErrorCode.UPSTREAM_UNAVAILABLE, ToolErrorCode.RATE_LIMITED},
    )
    return result


def _exception_envelope(tool_name: str, exc: Exception) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if any(marker in lowered for marker in ("rate limit", "429", "too many requests", "限流")):
        code = ToolErrorCode.RATE_LIMITED
        retryable = True
    elif tool_name in {"search_poi", "check_weather", "search_restaurant", "search_hotel", "plan_route"}:
        code = ToolErrorCode.UPSTREAM_UNAVAILABLE
        retryable = True
    else:
        code = ToolErrorCode.INTERNAL_ERROR
        retryable = False
    return ToolValidationFailure(
        code,
        f"{tool_name} 调用失败：{text}",
        retryable=retryable,
    ).envelope()


def _enforce_artifact_postcondition(
    tool_name: str,
    ctx: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    policy = tool_execution_policy(tool_name)
    if result.get("isError") or not policy.artifact_kinds:
        return result
    artifact_id = str(result.get("artifact_id") or "")
    store = getattr(ctx, "store", None)
    record = store.get_record(artifact_id) if store is not None and artifact_id else None
    if not record or record.get("kind") not in policy.artifact_kinds:
        return ToolValidationFailure(
            ToolErrorCode.CONTRACT_VIOLATION,
            f"{tool_name} 成功返回但未生成目标 Artifact。",
            details={"artifact_id": artifact_id, "expected_kinds": list(policy.artifact_kinds)},
        ).envelope()

    from travel_agent.agent.session import current_task_meta

    meta = current_task_meta()
    for field in ("request_id", "task_id", "agent"):
        expected = meta.get(field)
        if expected and record.get(field) != expected:
            return ToolValidationFailure(
                ToolErrorCode.CONTRACT_VIOLATION,
                f"{tool_name} 生成的 Artifact 不属于当前任务。",
                details={"artifact_id": artifact_id, "field": field},
            ).envelope()
    return result


F = TypeVar("F", bound=Callable[..., dict[str, Any]])


def contracted_tool(tool_name: str) -> Callable[[F], F]:
    """Decorate a toolkit function with the shared deterministic pipeline."""

    def decorate(fn: F) -> F:
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(ctx: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            from travel_agent.agent.session import RequestCancelledError, session_tool_lock
            from travel_agent.providers import ProviderRateLimitError

            control = getattr(ctx, "request_control", None)
            if control is not None:
                control.check_active()
            try:
                bound = signature.bind(ctx, *args, **kwargs)
                bound.apply_defaults()
            except TypeError as exc:
                return ToolValidationFailure(
                    ToolErrorCode.INVALID_INPUT,
                    f"{tool_name} 参数不合法：{exc}",
                ).envelope()
            arguments = {key: value for key, value in bound.arguments.items() if key != "ctx"}
            if set(arguments) == {"fields"} and isinstance(arguments["fields"], dict):
                arguments = dict(arguments["fields"])
            arguments = normalize_tool_arguments(tool_name, arguments)
            for key, value in arguments.items():
                if key in bound.arguments:
                    bound.arguments[key] = value
            failure = before_tool_call(tool_name, ctx, arguments)
            if failure is not None:
                return failure.envelope()

            def invoke() -> dict[str, Any]:
                try:
                    if control is not None:
                        control.check_active()
                    value = fn(*bound.args, **bound.kwargs)
                except RequestCancelledError:
                    raise
                except ProviderRateLimitError:
                    # A live provider quota is an evaluation-environment failure,
                    # not a recoverable tool result. Cancel sibling work that
                    # shares this request and let the runner stop the full suite.
                    if control is not None:
                        control.cancel()
                    raise
                except Exception as exc:  # noqa: BLE001 - normalize tool/provider failures
                    return _exception_envelope(tool_name, exc)
                return _enforce_artifact_postcondition(
                    tool_name,
                    ctx,
                    normalize_tool_result(value, tool_name),
                )

            if tool_execution_policy(tool_name).parallel_safe:
                return invoke()
            with session_tool_lock(str(ctx.session_id)):
                return invoke()

        return wrapped  # type: ignore[return-value]

    return decorate
