"""FastMCP server：把同一套 toolkit 工具以 MCP 形式对外暴露（对应 PROJECT_PLAN M3）。

设计要点：

- 工具实现复用 ``travel_agent.agent.toolkit``（与 in-process agent 单一事实源，
  避免「黑盒大工具」与重复实现）；
- 每个工具携带 ``session_id``，server 端用 ``_SESSIONS`` 注册表做会话隔离 +
  artifact 落盘持久化，符合「session 注入 + artifact 持久化」约定；
- 统一返回 toolkit 的 ``{isError, summary, ...}`` 信封。

启动：``python -m travel_agent.mcp_server``（默认 streamable-http，端口 8765）。
客户端拉取工具示例见 ``scripts/mcp_client_demo.py``。
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP

from travel_agent.agent import toolkit
from travel_agent.agent.session import SessionContext, build_session

mcp: FastMCP = FastMCP("travel-agent-tools")

_SESSIONS: dict[str, SessionContext] = {}


def _ctx(session_id: str) -> SessionContext:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = build_session(session_id=session_id, persist=True)
    return _SESSIONS[session_id]


@mcp.tool
def update_travel_profile(
    session_id: str = "default",
    destination: str | None = None,
    days: int | None = None,
    interests: list[str] | None = None,
    budget_level: str | None = None,
    pace: str | None = None,
    must_visit: list[str] | None = None,
    avoid: list[str] | None = None,
) -> dict[str, Any]:
    """记录/更新用户出行画像。"""
    return toolkit.update_travel_profile(
        _ctx(session_id),
        destination=destination,
        days=days,
        interests=interests,
        budget_level=budget_level,
        pace=pace,
        must_visit=must_visit,
        avoid=avoid,
    )


@mcp.tool
def request_travel_info(session_id: str = "default", missing_fields: list[str] | None = None) -> dict[str, Any]:
    """信息缺失时生成追问。"""
    return toolkit.request_travel_info(_ctx(session_id), missing_fields)


@mcp.tool
def search_poi(
    session_id: str = "default",
    city: str | None = None,
    interests: list[str] | None = None,
    category: str | None = None,
    max_results: int = 30,
) -> dict[str, Any]:
    """检索候选 POI（高德优先，本地兜底）。"""
    return toolkit.search_poi(_ctx(session_id), city, interests, category, max_results)


@mcp.tool
def check_weather(session_id: str = "default", city: str | None = None) -> dict[str, Any]:
    """查询目的地天气。"""
    return toolkit.check_weather(_ctx(session_id), city)


@mcp.tool
def plan_route(
    session_id: str,
    origin_poi_id: str,
    destination_poi_id: str,
    mode: str | None = None,
) -> dict[str, Any]:
    """计算两个 POI 之间的路线。"""
    return toolkit.plan_route(_ctx(session_id), origin_poi_id, destination_poi_id, mode)


@mcp.tool
def recommend_candidates(session_id: str = "default", top_k: int = 12) -> dict[str, Any]:
    """对候选 POI 多目标打分 + 多样性重排。"""
    return toolkit.recommend_candidates(_ctx(session_id), top_k)


@mcp.tool
def plan_and_critique(session_id: str = "default", max_iters: int = 3) -> dict[str, Any]:
    """运行可控规划子图（plan→critic→revise）。"""
    return toolkit.plan_and_critique(_ctx(session_id), max_iters)


@mcp.tool
def render_itinerary(session_id: str = "default") -> dict[str, Any]:
    """渲染行程为 A2UI 卡片。"""
    return toolkit.render_itinerary(_ctx(session_id))


@mcp.tool
def render_map(session_id: str = "default") -> dict[str, Any]:
    """生成高德地图渲染数据。"""
    return toolkit.render_map(_ctx(session_id))


def main() -> None:
    host = os.getenv("TRAVEL_AGENT_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("TRAVEL_AGENT_MCP_PORT", "8765"))
    mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
