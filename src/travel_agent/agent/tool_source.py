"""工具来源解析：进程内 lc_tools 优先，可选从 MCP Server 动态拉取（M9.5）。"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from travel_agent.agent.lc_tools import build_tools
from travel_agent.agent.session import (
    SessionContext,
    session_tool_lock,
    should_serialize_tool,
)
from travel_agent.settings import Settings

logger = logging.getLogger(__name__)


def _mcp_url(settings: Settings) -> str:
    return f"http://{settings.mcp.host}:{settings.mcp.port}/mcp/"


def _wrap_mcp_tool_with_session(tool: BaseTool, session_id: str) -> BaseTool:
    """为 MCP 工具自动注入 session_id，并按 session 串行执行。

    ToolNode 会并行发起多个远程调用，而 server 端同一 session 共享可变
    ctx；客户端持锁覆盖整个远程调用周期，即可避免 server 端竞态。

    dispatch 类工具不加锁：其内部派生的 Subagent 会再次调用（持锁的）业务
    工具，若此处也持同一把不可重入锁会死锁；此时 server 端竞态由业务工具
    自身的 per-session 锁保护。
    """
    if not should_serialize_tool(tool.name):
        return tool
    lock = session_tool_lock(session_id)

    async def _arun(**kwargs: Any) -> Any:
        kwargs.setdefault("session_id", session_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lock.acquire)
        try:
            if hasattr(tool, "ainvoke"):
                return await tool.ainvoke(kwargs)
            return tool.invoke(kwargs)
        finally:
            lock.release()

    def _run(**kwargs: Any) -> Any:
        kwargs.setdefault("session_id", session_id)
        with lock:
            return tool.invoke(kwargs)

    return StructuredTool.from_function(
        coroutine=_arun,
        func=_run,
        name=tool.name,
        description=tool.description or "",
    )


async def resolve_tools_async(
    ctx: SessionContext,
    settings: Settings,
    *,
    user_id: str = "default",
) -> tuple[list[BaseTool], str]:
    """返回 (tools, source)，source 为 ``mcp`` 或 ``local``。"""
    if settings.mcp.use_mcp_tools:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    settings.mcp.server_name: {
                        "transport": "streamable_http",
                        "url": _mcp_url(settings),
                        "timeout": timedelta(seconds=settings.mcp.timeout_seconds),
                        "headers": {"X-Travel-Session-Id": ctx.session_id},
                    }
                }
            )
            tools = await client.get_tools()
            wrapped = [_wrap_mcp_tool_with_session(tool, ctx.session_id) for tool in tools]
            logger.info("[ToolSource] fetched %d tools from MCP at %s", len(wrapped), _mcp_url(settings))
            return wrapped, "mcp"
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ToolSource] MCP fetch failed, fallback to local tools: %s", exc)

    return build_tools(ctx, settings, user_id=user_id), "local"
