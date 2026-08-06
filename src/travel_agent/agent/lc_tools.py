"""把会话级工具函数包装成 LangChain 工具，供 ``create_react_agent`` 自主调用。

工具刻意拆细（search_poi / check_weather / plan_route / recommend_candidates /
plan_and_critique / render_*），让 LLM 在它们之间真正编排，而不是调用一个黑盒
大工具。``plan_and_critique`` 内部才是可控规划子图。
"""

from __future__ import annotations

import json
import inspect
import time
from functools import wraps
from typing import Any

from langchain_core.tools import StructuredTool

from travel_agent.agent import toolkit
from travel_agent.agent.session import SessionContext, session_tool_lock, should_serialize_tool
from travel_agent.settings import Settings
from travel_agent.skills.loader import build_skill_tools, load_skills


TOOL_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "search_poi": ("city",),
    "check_weather": ("city",),
    "plan_route": ("origin_poi_id", "destination_poi_id"),
    "search_restaurant": ("city",),
    "search_hotel": ("city",),
    "estimate_budget": ("city", "days"),
}

TOOL_CONTEXT_DEFAULTS: dict[str, dict[str, str]] = {
    "search_poi": {"city": "destination"},
    "check_weather": {"city": "destination"},
    "search_restaurant": {"city": "destination", "budget_level": "budget_level"},
    "search_hotel": {"city": "destination", "budget_level": "budget_level"},
    "estimate_budget": {"city": "destination", "days": "days", "budget_level": "budget_level"},
}


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)


def build_tools(
    ctx: SessionContext,
    settings: Settings | None = None,
    user_id: str = "default",
) -> list[StructuredTool]:
    def update_travel_profile(
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
    ) -> str:
        """记录/更新用户的出行画像。已知信息就填，未知留空。
        budget_level 取值 low/mid/high；pace 取值 relaxed/standard/intensive。"""
        return _dump(
            toolkit.update_travel_profile(
                ctx,
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
        )

    def request_travel_info(missing_fields: list[str] | None = None) -> str:
        """当缺少必要信息（如目的地/天数）时，生成向用户追问的问题并停止规划。"""
        return _dump(toolkit.request_travel_info(ctx, missing_fields))

    def request_preference_guide() -> str:
        """目的地与天数已齐、但用户未表达偏好时，引导兴趣/节奏/预算等；用户可说「随便」跳过。"""
        from travel_agent.settings import get_settings
        from travel_agent.storage.user_memory import get_user_memory_service

        settings_obj = settings or get_settings()
        l3 = get_user_memory_service(settings_obj.memory).load_stable_profile(user_id)
        return _dump(toolkit.request_preference_guide(ctx, l3))

    def search_poi(
        city: str | None = None,
        interests: list[str] | None = None,
        category: str | None = None,
        max_results: int = 30,
    ) -> str:
        """检索一次目的地候选 POI 并缓存。成功后不要再次调用，下一步调用 recommend_candidates。"""
        return _dump(toolkit.search_poi(ctx, city, interests, category, max_results))

    def check_weather(city: str | None = None) -> str:
        """查询目的地天气，用于雨天户外行程冲突判断。"""
        return _dump(toolkit.check_weather(ctx, city))

    def plan_route(origin_poi_id: str, destination_poi_id: str, mode: str | None = None) -> str:
        """计算两个 POI 之间的距离与通勤时长。poi_id 必须来自 search_poi 的返回。"""
        return _dump(toolkit.plan_route(ctx, origin_poi_id, destination_poi_id, mode))

    def build_constraints() -> str:
        """把当前出行画像转换为可验证 ConstraintSet，用于规划和评测。"""
        return _dump(toolkit.build_constraints(ctx))

    def search_restaurant(
        city: str | None = None,
        cuisine: str | None = None,
        area: str | None = None,
        budget_level: str | None = None,
        max_results: int = 10,
    ) -> str:
        """检索餐厅候选，支持城市、菜系、区域和预算档位。"""
        return _dump(
            toolkit.search_restaurant(ctx, city, cuisine, area, budget_level, max_results)
        )

    def search_hotel(
        city: str | None = None,
        area: str | None = None,
        budget_level: str | None = None,
        min_rating: float | None = None,
        max_results: int = 10,
    ) -> str:
        """检索酒店候选，支持城市、区域、预算和最低评分。"""
        return _dump(
            toolkit.search_hotel(ctx, city, area, budget_level, min_rating, max_results)
        )

    def estimate_budget(
        city: str | None = None,
        days: int | None = None,
        companions: int | None = None,
        budget_level: str | None = None,
        hotel_required: bool = True,
    ) -> str:
        """估算旅行预算，包括住宿、餐饮、门票、市内交通和总价区间。"""
        return _dump(
            toolkit.estimate_budget(
                ctx,
                city,
                days,
                companions,
                budget_level,
                hotel_required,
            )
        )

    def recommend_candidates(top_k: int = 12) -> str:
        """对已检索候选做一次重排。成功后不要重复调用，下一步调用 plan_and_critique。"""
        return _dump(toolkit.recommend_candidates(ctx, top_k))

    def plan_and_critique(max_iters: int = 3) -> str:
        """运行可控规划子图（plan→critic→revise 闭环），产出约束满足的结构化行程。
        调用前需先 recommend_candidates，且画像里已有 destination 和 days；成功后本轮结束，系统自动渲染。"""
        return _dump(toolkit.plan_and_critique(ctx, max_iters))

    def render_itinerary() -> str:
        """把规划好的行程渲染为前端 A2UI 卡片。"""
        return _dump(toolkit.render_itinerary(ctx))

    def render_map() -> str:
        """生成高德地图渲染数据（点位与按天路线）。"""
        return _dump(toolkit.render_map(ctx))

    functions = [
        update_travel_profile,
        request_travel_info,
        request_preference_guide,
        search_poi,
        check_weather,
        plan_route,
        build_constraints,
        search_restaurant,
        search_hotel,
        estimate_budget,
        recommend_candidates,
        plan_and_critique,
        render_itinerary,
        render_map,
    ]
    def traced(fn):
        if not ctx.evaluation_trace_enabled:
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            record: dict[str, Any] = {
                "kind": "tool",
                "name": fn.__name__,
                "arguments": _redact_arguments(kwargs),
            }
            try:
                result = fn(*args, **kwargs)
                payload = _parse_tool_payload(result)
                record["status"] = "error" if payload.get("isError") else "ok"
                record["data_source"] = _find_data_source(payload) or type(ctx.provider).__name__
                record["fallback_reason"] = payload.get("fallback_reason")
                return result
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
                ctx.evaluation_trace.append(record)

        return wrapper

    def serialized(fn):
        """同一 session 内串行执行工具：ToolNode 会并行跑同一 AIMessage
        里的多个 tool_call，而工具共享可变 ctx，fail-closed 按会话加锁。

        dispatch 类工具（内部会派生 Subagent）不加锁，否则不可重入的
        session 锁会同线程嵌套自锁 → 死锁（见 session.UNLOCKED_TOOL_NAMES）。"""
        if not should_serialize_tool(fn.__name__):
            return fn
        lock = session_tool_lock(ctx.session_id)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with lock:
                return fn(*args, **kwargs)

        return wrapper

    tools = [
        StructuredTool.from_function(
            traced(serialized(fn)),
            return_direct=fn.__name__ == "plan_and_critique",
        )
        for fn in functions
    ]
    if settings and settings.skills.enabled:
        tools.extend(build_skill_tools(load_skills(settings.skills.skills_dir)))
    return tools


def _parse_tool_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _find_data_source(payload: dict[str, Any]) -> str | None:
    for key in ("source", "provider", "data_source"):
        if payload.get(key):
            return str(payload[key])
    for value in payload.values():
        if isinstance(value, dict):
            if source := _find_data_source(value):
                return source
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and (source := _find_data_source(item)):
                    return source
    return None


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    sensitive = {"api_key", "key", "token", "authorization", "password"}
    return {
        key: "[REDACTED]" if key.lower() in sensitive else value
        for key, value in arguments.items()
        if value is not None
    }
