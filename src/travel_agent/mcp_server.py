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
import copy
import dataclasses
import threading
from typing import Any

from fastmcp import FastMCP

from travel_agent.agent import toolkit
from travel_agent.agent.session import (
    SessionContext,
    build_session,
    reset_task_meta,
    set_current_task_meta,
)

mcp: FastMCP = FastMCP("travel-agent-tools")

_SESSIONS: dict[str, SessionContext] = {}
_SESSIONS_LOCK = threading.Lock()


def _ctx(session_id: str) -> SessionContext:
    with _SESSIONS_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = build_session(session_id=session_id, persist=True)
        return _SESSIONS[session_id]


def _call(session_id: str, task_context: dict[str, Any] | None, fn, *args, **kwargs):
    """Apply the same request/task ownership metadata used by local tools."""
    meta = dict(task_context or {})
    ctx = _ctx(session_id)
    profile = meta.pop("_profile", None)
    if isinstance(profile, dict):
        with ctx._state_lock:
            for name in getattr(ctx.profile, "__dataclass_fields__", {}):
                if name in profile:
                    setattr(ctx.profile, name, copy.deepcopy(profile[name]))
    token = set_current_task_meta(meta)
    try:
        result = fn(ctx, *args, **kwargs)
        if isinstance(result, dict) and result.get("artifact_id"):
            record = ctx.store.get_record(str(result["artifact_id"]))
            if record is not None:
                result = {**result, "_artifact_record": record}
        if isinstance(result, dict):
            result = {**result, "_profile_snapshot": dataclasses.asdict(ctx.profile)}
        return result
    finally:
        reset_task_meta(token)


def _planner_ids(task_context: dict[str, Any] | None) -> list[str] | None:
    meta = dict(task_context or {})
    if meta.get("agent") != "planner":
        return None
    return [str(item) for item in (meta.get("artifact_ids") or []) if item]


@mcp.tool
def update_travel_profile(
    session_id: str = "default",
    destination: str | None = None,
    days: int | None = None,
    start_date: str | None = None,
    interests: list[str] | None = None,
    food_preference: list[str] | None = None,
    budget_level: str | None = None,
    budget_limit: float | None = None,
    pace: str | None = None,
    hotel_area: str | None = None,
    must_visit: list[str] | None = None,
    avoid: list[str] | None = None,
    remove_interests: list[str] | None = None,
    remove_food_preference: list[str] | None = None,
    remove_avoid: list[str] | None = None,
    remove_budget_level: str | None = None,
    remove_pace: str | None = None,
    remove_transport_mode: str | None = None,
    companions: str | None = None,
    party_size: int | None = None,
    transport_mode: str | None = None,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """记录/更新用户出行画像。"""
    return _call(
        session_id,
        task_context,
        toolkit.update_travel_profile,
        destination=destination,
        days=days,
        start_date=start_date,
        interests=interests,
        food_preference=food_preference,
        budget_level=budget_level,
        budget_limit=budget_limit,
        pace=pace,
        hotel_area=hotel_area,
        must_visit=must_visit,
        avoid=avoid,
        remove_interests=remove_interests,
        remove_food_preference=remove_food_preference,
        remove_avoid=remove_avoid,
        remove_budget_level=remove_budget_level,
        remove_pace=remove_pace,
        remove_transport_mode=remove_transport_mode,
        companions=companions,
        party_size=party_size,
        transport_mode=transport_mode,
    )


@mcp.tool
def request_travel_info(session_id: str = "default", missing_fields: list[str] | None = None, task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """信息缺失时生成追问。"""
    return _call(session_id, task_context, toolkit.request_travel_info, missing_fields)


@mcp.tool
def request_preference_guide(
    session_id: str = "default",
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """偏好不足时生成引导；MCP 核心契约不依赖可选 L3 存储。"""
    return _call(session_id, task_context, toolkit.request_preference_guide, None)


@mcp.tool
def search_poi(
    session_id: str = "default",
    city: str | None = None,
    interests: list[str] | None = None,
    category: str | None = None,
    max_results: int = 30,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """检索候选 POI（高德优先，本地兜底）。"""
    return _call(session_id, task_context, toolkit.search_poi, city, interests, category, max_results)


@mcp.tool
def check_weather(session_id: str = "default", city: str | None = None, task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """查询目的地天气。"""
    return _call(session_id, task_context, toolkit.check_weather, city)


@mcp.tool
def plan_route(
    session_id: str,
    origin_poi_id: str,
    destination_poi_id: str,
    mode: str | None = None,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """计算两个 POI 之间的路线。"""
    return _call(session_id, task_context, toolkit.plan_route, origin_poi_id, destination_poi_id, mode)


@mcp.tool
def build_constraints(session_id: str = "default", task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """把当前出行画像转换为可验证 ConstraintSet。"""
    return _call(session_id, task_context, toolkit.build_constraints)


@mcp.tool
def search_restaurant(
    session_id: str = "default",
    city: str | None = None,
    cuisine: str | None = None,
    area: str | None = None,
    budget_level: str | None = None,
    max_results: int = 10,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """检索餐厅候选。"""
    return _call(session_id, task_context, toolkit.search_restaurant, city, cuisine, area, budget_level, max_results)


@mcp.tool
def search_hotel(
    session_id: str = "default",
    city: str | None = None,
    area: str | None = None,
    budget_level: str | None = None,
    min_rating: float | None = None,
    max_results: int = 10,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """检索酒店候选。"""
    return _call(session_id, task_context, toolkit.search_hotel, city, area, budget_level, min_rating, max_results)


@mcp.tool
def estimate_budget(
    session_id: str = "default",
    city: str | None = None,
    days: int | None = None,
    companions: int | None = None,
    budget_level: str | None = None,
    hotel_required: bool = True,
    task_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """估算旅行预算。"""
    return _call(session_id, task_context, toolkit.estimate_budget, city, days, companions, budget_level, hotel_required)


@mcp.tool
def recommend_candidates(session_id: str = "default", top_k: int = 12, task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """对候选 POI 多目标打分 + 多样性重排。"""
    return _call(session_id, task_context, toolkit.recommend_candidates, top_k, _planner_ids(task_context))


@mcp.tool
def plan_and_critique(session_id: str = "default", max_iters: int = 3, task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """运行可控规划子图（plan→critic→revise）。"""
    artifact_ids = _planner_ids(task_context)
    if artifact_ids is not None:
        ctx = _ctx(session_id)
        meta = dict(task_context or {})
        artifact_ids.extend(
            ctx.store.artifact_ids_for_task(
                str(meta.get("request_id") or ""),
                str(meta.get("task_id") or ""),
                agent="planner",
            )
        )
        artifact_ids = list(dict.fromkeys(artifact_ids))
    return _call(session_id, task_context, toolkit.plan_and_critique, max_iters, artifact_ids)


@mcp.tool
def render_itinerary(session_id: str = "default", task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """渲染行程为 A2UI 卡片。"""
    return _call(session_id, task_context, toolkit.render_itinerary)


@mcp.tool
def render_map(session_id: str = "default", task_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """生成高德地图渲染数据。"""
    return _call(session_id, task_context, toolkit.render_map)


def main() -> None:
    host = os.getenv("TRAVEL_AGENT_MCP_HOST", "127.0.0.1")
    port = int(os.getenv("TRAVEL_AGENT_MCP_PORT", "8765"))
    mcp.run(transport="http", host=host, port=port)


if __name__ == "__main__":
    main()
