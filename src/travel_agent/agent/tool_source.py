"""工具来源解析：进程内 lc_tools 优先，可选从 MCP Server 动态拉取（M9.5）。"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import time
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import create_model
from pydantic.fields import PydanticUndefined

from travel_agent.agent.lc_tools import build_tools
from travel_agent.agent.session import SessionContext, current_task_meta
from travel_agent.settings import Settings

logger = logging.getLogger(__name__)

LOGICAL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "update_travel_profile",
        "request_travel_info",
        "request_preference_guide",
        "search_poi",
        "check_weather",
        "plan_route",
        "build_constraints",
        "search_restaurant",
        "search_hotel",
        "estimate_budget",
        "recommend_candidates",
        "plan_and_critique",
        "render_itinerary",
        "render_map",
    }
)


class ToolContractError(RuntimeError):
    pass


def _mcp_url(settings: Settings) -> str:
    return f"http://{settings.mcp.host}:{settings.mcp.port}/mcp/"


def _loopback_httpx_client(headers=None, timeout=None, auth=None):
    """Local MCP must not be routed through process-wide HTTP proxies."""
    import httpx

    kwargs: dict[str, Any] = {
        "follow_redirects": True,
        "trust_env": False,
    }
    if headers is not None:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def _wrap_mcp_tool_with_session(tool: BaseTool, ctx: SessionContext | str) -> BaseTool:
    """Inject transport-only context without holding a lock across MCP I/O."""
    session_id = ctx.session_id if isinstance(ctx, SessionContext) else str(ctx)
    public_schema = _public_args_schema(tool)

    def _arguments(kwargs: dict[str, Any]) -> dict[str, Any]:
        meta = current_task_meta()
        if isinstance(ctx, SessionContext):
            with ctx._state_lock:
                meta["_profile"] = dataclasses.asdict(ctx.profile)
        return {
            **kwargs,
            "session_id": session_id,
            "task_context": meta,
        }

    async def _arun(**kwargs: Any) -> Any:
        return await _invoke_async(tool, ctx, _arguments(kwargs))

    def _run(**kwargs: Any) -> Any:
        return _invoke_sync(tool, ctx, _arguments(kwargs))

    return StructuredTool.from_function(
        coroutine=_arun,
        func=_run,
        name=tool.name,
        description=tool.description or "",
        args_schema=public_schema,
    )


def _before_remote(ctx: SessionContext | str) -> tuple[float, Any, str, float | None]:
    control = getattr(ctx, "request_control", None)
    if control is not None:
        control.check_active()
    from travel_agent.orchestration.meter import current_turn_meter

    meter = current_turn_meter()
    role = str(current_task_meta().get("agent") or "main")
    meter_started = meter.begin_tool(role) if meter is not None else None
    return time.perf_counter(), meter, role, meter_started


def _after_remote(
    ctx: SessionContext | str,
    started: tuple[float, Any, str, float | None],
    error: bool,
) -> None:
    control = getattr(ctx, "request_control", None)
    if control is not None:
        control.check_active()
    _wall_started, meter, role, meter_started = started
    if meter is not None and meter_started is not None:
        meter.finish_tool(role, meter_started, error=error)


def _invoke_sync(tool: BaseTool, ctx: SessionContext | str, arguments: dict[str, Any]) -> str:
    started = _before_remote(ctx)
    error = True
    try:
        result = _normalize_envelope(tool.invoke(arguments), tool.name, ctx)
        error = json.loads(result).get("isError") is True
        return result
    finally:
        _after_remote(ctx, started, error)


async def _invoke_async(tool: BaseTool, ctx: SessionContext | str, arguments: dict[str, Any]) -> str:
    started = _before_remote(ctx)
    error = True
    try:
        result = _normalize_envelope(await tool.ainvoke(arguments), tool.name, ctx)
        error = json.loads(result).get("isError") is True
        return result
    finally:
        _after_remote(ctx, started, error)


def _public_args_schema(tool: BaseTool):
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", {}) if schema is not None else {}
    public: dict[str, tuple[Any, Any]] = {}
    for name, info in fields.items():
        if name in {"session_id", "task_context"}:
            continue
        default = ... if info.default is PydanticUndefined else info.default
        public[name] = (info.annotation or Any, default)
    return create_model(f"{tool.name.title().replace('_', '')}PublicInput", **public)


def _normalize_envelope(
    value: Any,
    tool_name: str,
    ctx: SessionContext | str | None = None,
) -> str:
    # MCP adapters may expose FastMCP's content blocks rather than the decoded
    # structured payload.  Accept one textual block, but never invent semantics.
    structured = getattr(value, "structured_content", None)
    if isinstance(structured, dict):
        value = structured
    elif hasattr(value, "content"):
        value = getattr(value, "content")
    if isinstance(value, list) and len(value) == 1:
        block = value[0]
        value = getattr(block, "text", None) or (
            block.get("text") if isinstance(block, dict) else value
        )
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ToolContractError(f"{tool_name}: non-JSON MCP result") from exc
    else:
        raise ToolContractError(f"{tool_name}: unsupported MCP result {type(value).__name__}")
    if not isinstance(payload, dict) or "isError" not in payload or "summary" not in payload:
        raise ToolContractError(f"{tool_name}: result violates toolkit envelope")
    record = payload.pop("_artifact_record", None)
    profile = payload.pop("_profile_snapshot", None)
    if isinstance(ctx, SessionContext) and ctx.request_control is not None:
        ctx.request_control.check_active()
    if isinstance(ctx, SessionContext) and isinstance(record, dict) and record.get("artifact_id"):
        ctx.store.merge_records({str(record["artifact_id"]): record})
    if isinstance(ctx, SessionContext) and isinstance(profile, dict):
        with ctx._state_lock:
            for name in getattr(ctx.profile, "__dataclass_fields__", {}):
                if name in profile:
                    setattr(ctx.profile, name, copy.deepcopy(profile[name]))
    return json.dumps(payload, ensure_ascii=False, default=str)


def validate_tool_contract(tools: list[BaseTool], *, source: str) -> None:
    names = [tool.name for tool in tools]
    duplicates = {name for name in names if names.count(name) > 1}
    missing = LOGICAL_TOOL_NAMES - set(names)
    extra = set(names) - LOGICAL_TOOL_NAMES
    if duplicates or missing or extra:
        raise ToolContractError(
            f"{source} tool contract mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}, duplicates={sorted(duplicates)}"
        )


def tool_contract_signature(tools: list[BaseTool]) -> dict[str, dict[str, Any]]:
    """Return the public logical contract used for Local/MCP parity checks."""
    return {
        tool.name: getattr(tool, "args_schema").model_json_schema()
        for tool in tools
    }


def _logical_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Architecture variants share only the stable production tool contract."""
    return [tool for tool in tools if tool.name in LOGICAL_TOOL_NAMES]


async def resolve_tools_async(
    ctx: SessionContext,
    settings: Settings,
    *,
    user_id: str = "default",
) -> tuple[list[BaseTool], str]:
    """返回 (tools, source)，source 为 ``mcp`` 或 ``local``。"""
    mcp_settings = getattr(settings, "mcp", None)
    if bool(getattr(mcp_settings, "use_mcp_tools", False)):
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            connection: dict[str, Any] = {
                        "transport": "streamable_http",
                        "url": _mcp_url(settings),
                        "timeout": timedelta(seconds=mcp_settings.timeout_seconds),
                        "headers": {"X-Travel-Session-Id": ctx.session_id},
            }
            if str(mcp_settings.host).strip().lower() in {"127.0.0.1", "localhost", "::1"}:
                connection["httpx_client_factory"] = _loopback_httpx_client
            client = MultiServerMCPClient({mcp_settings.server_name: connection})
            tools = await client.get_tools()
            wrapped = [_wrap_mcp_tool_with_session(tool, ctx) for tool in tools]
            validate_tool_contract(wrapped, source="mcp")
            logger.info("[ToolSource] fetched %d tools from MCP at %s", len(wrapped), _mcp_url(settings))
            return wrapped, "mcp"
        except Exception as exc:  # noqa: BLE001
            raise ToolContractError(f"MCP tool source unavailable or incompatible: {exc}") from exc

    local = _logical_tools(build_tools(ctx, settings, user_id=user_id))
    validate_tool_contract(local, source="local")
    return local, "local"


def resolve_tools(
    ctx: SessionContext,
    settings: Settings,
    *,
    user_id: str = "default",
) -> tuple[list[BaseTool], str]:
    """Synchronous resolver for worker threads; never called inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_tools_async(ctx, settings, user_id=user_id))
    raise RuntimeError("resolve_tools() cannot run inside an active event loop; use resolve_tools_async()")
