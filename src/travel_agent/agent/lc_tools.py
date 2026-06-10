"""把会话级工具函数包装成 LangChain 工具，供 ``create_react_agent`` 自主调用。

工具刻意拆细（search_poi / check_weather / plan_route / recommend_candidates /
plan_and_critique / render_*），让 LLM 在它们之间真正编排，而不是调用一个黑盒
大工具。``plan_and_critique`` 内部才是可控规划子图。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from travel_agent.agent import toolkit
from travel_agent.agent.session import SessionContext


def _dump(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)


def build_tools(ctx: SessionContext) -> list[StructuredTool]:
    def update_travel_profile(
        destination: str | None = None,
        days: int | None = None,
        interests: list[str] | None = None,
        budget_level: str | None = None,
        pace: str | None = None,
        must_visit: list[str] | None = None,
        avoid: list[str] | None = None,
        companions: str | None = None,
        transport_mode: str | None = None,
    ) -> str:
        """记录/更新用户的出行画像。已知信息就填，未知留空。
        budget_level 取值 low/mid/high；pace 取值 relaxed/standard/intensive。"""
        return _dump(
            toolkit.update_travel_profile(
                ctx,
                destination=destination,
                days=days,
                interests=interests,
                budget_level=budget_level,
                pace=pace,
                must_visit=must_visit,
                avoid=avoid,
                companions=companions,
                transport_mode=transport_mode,
            )
        )

    def request_travel_info(missing_fields: list[str] | None = None) -> str:
        """当缺少必要信息（如目的地/天数）时，生成向用户追问的问题并停止规划。"""
        return _dump(toolkit.request_travel_info(ctx, missing_fields))

    def search_poi(
        city: str | None = None,
        interests: list[str] | None = None,
        category: str | None = None,
        max_results: int = 30,
    ) -> str:
        """检索目的地候选景点/餐饮 POI（高德优先，本地兜底），结果会被缓存供后续工具使用。"""
        return _dump(toolkit.search_poi(ctx, city, interests, category, max_results))

    def check_weather(city: str | None = None) -> str:
        """查询目的地天气，用于雨天户外行程冲突判断。"""
        return _dump(toolkit.check_weather(ctx, city))

    def plan_route(origin_poi_id: str, destination_poi_id: str, mode: str | None = None) -> str:
        """计算两个 POI 之间的距离与通勤时长。poi_id 必须来自 search_poi 的返回。"""
        return _dump(toolkit.plan_route(ctx, origin_poi_id, destination_poi_id, mode))

    def recommend_candidates(top_k: int = 12) -> str:
        """对已检索的候选 POI 做多目标打分与多样性重排，输出排序候选（规划输入）。"""
        return _dump(toolkit.recommend_candidates(ctx, top_k))

    def plan_and_critique(max_iters: int = 3) -> str:
        """运行可控规划子图（plan→critic→revise 闭环），产出约束满足的结构化行程。
        调用前需先 recommend_candidates，且画像里已有 destination 和 days。"""
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
        search_poi,
        check_weather,
        plan_route,
        recommend_candidates,
        plan_and_critique,
        render_itinerary,
        render_map,
    ]
    return [StructuredTool.from_function(fn) for fn in functions]
