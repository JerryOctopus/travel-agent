"""会话级工具函数（被 LangChain 工具 / 离线兜底 / MCP server 三处复用）。

每个函数：

- 第一个参数是 ``SessionContext``，承载 provider / artifact store / 累积 profile；
- 入参/出参均为 JSON 友好类型；
- 返回统一信封 ``{"isError": bool, "summary": str, ...}``，工具间的大对象
  （POI 列表 / 行程）通过 ``store`` 传递并以 ``artifact_id`` 引用。

这样 LLM 只需在 ``search_poi -> recommend_candidates -> plan_and_critique ->
render_*`` 这些工具之间做编排决策，而不必搬运大对象。
"""

from __future__ import annotations

from typing import Any

from travel_agent.agent.render import build_map_payload, build_itinerary_cards
from travel_agent.agent.serde import (
    itinerary_to_dict,
    poi_brief,
    poi_from_dict,
    poi_to_dict,
)
from travel_agent.agent.session import SessionContext
from travel_agent.planning_subgraph import plan_and_critique as run_plan_and_critique
from travel_agent.recommendation import score_pois
from travel_agent.schemas import POI, ScoredPOI, TravelProfile
from travel_agent.workflow import merge_profile

VALID_BUDGET = {"low", "mid", "high"}
VALID_PACE = {"relaxed", "standard", "intensive"}
VALID_TRANSPORT = {"walk", "public_transport", "taxi", "drive"}


def _ok(summary: str, **data: Any) -> dict[str, Any]:
    return {"isError": False, "summary": summary, **data}


def _err(summary: str, **data: Any) -> dict[str, Any]:
    return {"isError": True, "summary": summary, **data}


# --------------------------------------------------------------------------- #
# 画像与追问
# --------------------------------------------------------------------------- #
def update_travel_profile(ctx: SessionContext, **fields: Any) -> dict[str, Any]:
    """把已知出行信息合并进会话画像（destination/days/interests/budget 等）。"""
    cleaned: dict[str, Any] = {}
    for key in (
        "destination",
        "start_date",
        "companions",
        "hotel_area",
    ):
        if fields.get(key):
            cleaned[key] = str(fields[key]).strip()
    if fields.get("days") is not None:
        try:
            days = int(fields["days"])
            if days > 0:
                cleaned["days"] = days
        except (TypeError, ValueError):
            pass
    if fields.get("budget_level") in VALID_BUDGET:
        cleaned["budget_level"] = fields["budget_level"]
    if fields.get("pace") in VALID_PACE:
        cleaned["pace"] = fields["pace"]
    if fields.get("transport_mode") in VALID_TRANSPORT:
        cleaned["transport_mode"] = fields["transport_mode"]
    for list_key in ("interests", "food_preference", "must_visit", "avoid"):
        value = fields.get(list_key)
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if value:
            cleaned[list_key] = list(value)

    update = TravelProfile(
        destination=cleaned.get("destination"),
        days=cleaned.get("days"),
        start_date=cleaned.get("start_date"),
        budget_level=cleaned.get("budget_level"),
        interests=cleaned.get("interests", []),
        companions=cleaned.get("companions"),
        pace=cleaned.get("pace", "standard"),
        hotel_area=cleaned.get("hotel_area"),
        food_preference=cleaned.get("food_preference", []),
        must_visit=cleaned.get("must_visit", []),
        avoid=cleaned.get("avoid", []),
        transport_mode=cleaned.get("transport_mode", "public_transport"),
    )
    ctx.profile = merge_profile(ctx.profile, update)
    missing = ctx.profile.missing_required_fields()
    return _ok(
        f"已更新出行画像，目的地={ctx.profile.destination}，天数={ctx.profile.days}。",
        profile=_profile_brief(ctx.profile),
        missing_required=missing,
    )


def request_travel_info(ctx: SessionContext, missing_fields: list[str] | None = None) -> dict[str, Any]:
    """信息不全时，生成面向用户的追问。"""
    missing = missing_fields or ctx.profile.missing_required_fields()
    questions = {
        "destination": "你想去哪个城市旅行？",
        "days": "计划玩几天？",
        "budget_level": "预算大概是什么档次（经济/中等/高）？",
        "interests": "更想体验哪些主题（自然/美食/历史/文化/购物等）？",
    }
    asks = [questions.get(field, f"请补充 {field}") for field in missing] or [
        "还有什么偏好需要我考虑吗？"
    ]
    question = "为了给你规划合适的行程，" + "；".join(asks)
    return _ok(question, question=question, missing_fields=missing)


# --------------------------------------------------------------------------- #
# 真实工具：POI / 天气 / 路线
# --------------------------------------------------------------------------- #
def search_poi(
    ctx: SessionContext,
    city: str | None = None,
    interests: list[str] | None = None,
    category: str | None = None,
    max_results: int = 30,
) -> dict[str, Any]:
    """检索候选 POI（高德 REST 优先，本地 seed 兜底），存入会话供后续工具复用。"""
    target_city = city or ctx.profile.destination
    if not target_city:
        return _err("缺少目的地城市，无法检索 POI，请先确认 destination。")
    tags = interests if interests is not None else ctx.profile.interests

    preference = ctx.provider.search_pois(
        city=target_city, query_tags=tags or None, category=category, max_results=max_results
    )
    citywide = ctx.provider.search_pois(city=target_city, max_results=max_results)
    candidates = _merge_pois(preference, citywide)
    if not candidates:
        return _err(
            f"没有检索到「{target_city}」的 POI（本地 seed 仅覆盖少数城市，建议配置高德 key）。",
            city=target_city,
        )

    ctx.remember_pois(candidates)
    source = candidates[0].source
    artifact_id = ctx.store.put(
        "candidates",
        {"city": target_city, "pois": [poi_to_dict(p) for p in candidates]},
    )
    return _ok(
        f"在「{target_city}」检索到 {len(candidates)} 个候选 POI（来源={source}）。",
        artifact_id=artifact_id,
        city=target_city,
        count=len(candidates),
        pois=[poi_brief(p) for p in candidates[:12]],
    )


def check_weather(ctx: SessionContext, city: str | None = None) -> dict[str, Any]:
    """查询目的地天气（用于雨天户外冲突判断）。"""
    target_city = city or ctx.profile.destination
    if not target_city:
        return _err("缺少目的地城市，无法查询天气。")
    weather = ctx.provider.get_weather(target_city)
    ctx.store.put("weather", {"city": target_city, "condition": weather.condition,
                              "temperature_c": weather.temperature_c, "source": weather.source})
    return _ok(
        f"{target_city} 天气：{weather.condition}，约 {weather.temperature_c}°C（{weather.source}）。",
        city=target_city,
        condition=weather.condition,
        temperature_c=weather.temperature_c,
        source=weather.source,
    )


def plan_route(
    ctx: SessionContext,
    origin_poi_id: str,
    destination_poi_id: str,
    mode: str | None = None,
) -> dict[str, Any]:
    """计算两个 POI 之间的路线（距离/时长），POI 需来自此前的 search_poi 结果。"""
    origin = ctx.poi(origin_poi_id)
    destination = ctx.poi(destination_poi_id)
    if origin is None or destination is None:
        return _err("找不到对应 POI，请确认 poi_id 来自 search_poi 返回的结果。")
    transport = mode if mode in VALID_TRANSPORT else ctx.profile.transport_mode
    route = ctx.provider.estimate_route(origin, destination, transport)
    return _ok(
        f"{origin.name} → {destination.name}：约 {route.duration_min} 分钟 / "
        f"{route.distance_km} 公里（{route.mode}, {route.source}）。",
        distance_km=route.distance_km,
        duration_min=route.duration_min,
        mode=route.mode,
        source=route.source,
    )


# --------------------------------------------------------------------------- #
# 推荐打分 + 可控规划子图
# --------------------------------------------------------------------------- #
def recommend_candidates(ctx: SessionContext, top_k: int = 12) -> dict[str, Any]:
    """对候选 POI 做多目标打分 + 多样性重排，产出排序后的规划输入。"""
    candidates_artifact = ctx.store.latest("candidates")
    if not candidates_artifact:
        return _err("还没有候选 POI，请先调用 search_poi。")
    pois = [poi_from_dict(p) for p in candidates_artifact["pois"]]
    ranked = score_pois(pois, ctx.profile)
    top = ranked[: max(1, top_k)]
    artifact_id = ctx.store.put(
        "ranked",
        {"pois": [_scored_to_dict(s) for s in ranked]},
    )
    return _ok(
        f"完成多目标打分与多样性重排，输出 {len(ranked)} 个排序候选。",
        artifact_id=artifact_id,
        count=len(ranked),
        top=[
            {"name": s.poi.name, "score": s.score, "reasons": s.reasons}
            for s in top
        ],
    )


def plan_and_critique(ctx: SessionContext, max_iters: int = 3) -> dict[str, Any]:
    """运行可控规划子图：plan → critic → revise 闭环，输出约束满足的行程。"""
    ranked_artifact = ctx.store.latest("ranked")
    if not ranked_artifact:
        return _err("还没有排序候选，请先调用 recommend_candidates。")
    if not ctx.profile.destination or not ctx.profile.days:
        return _err("缺少 destination 或 days，无法规划，请先补全画像。")

    ranked = [_scored_from_dict(d) for d in ranked_artifact["pois"]]
    result = run_plan_and_critique(
        ranked_pois=ranked,
        profile=ctx.profile,
        route_estimator=ctx.provider,
        max_iters=max_iters,
    )
    itinerary_dict = itinerary_to_dict(result.itinerary)
    artifact_id = ctx.store.put(
        "itinerary",
        {
            "itinerary": itinerary_dict,
            "original_itinerary": itinerary_to_dict(result.original_itinerary),
            "critic": {
                "passed": result.critic_result.passed,
                "issues": [
                    {"code": i.code, "message": i.message, "severity": i.severity}
                    for i in result.critic_result.issues
                ],
            },
            "revision_notes": result.revision_notes,
            "original_issue_count": result.original_issue_count,
            "final_issue_count": result.final_issue_count,
            "iterations": result.iterations,
        },
    )
    return _ok(
        f"规划完成：闭环 {result.iterations} 轮，违规项 {result.original_issue_count} → "
        f"{result.final_issue_count}，{'已通过 critic' if result.critic_result.passed else '仍有可解释告警'}。",
        artifact_id=artifact_id,
        passed=result.critic_result.passed,
        original_issue_count=result.original_issue_count,
        final_issue_count=result.final_issue_count,
        revision_notes=result.revision_notes,
        issues=[
            {"code": i.code, "message": i.message, "severity": i.severity}
            for i in result.critic_result.issues
        ],
    )


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def render_itinerary(ctx: SessionContext) -> dict[str, Any]:
    """把规划好的行程渲染为前端可用的 A2UI 卡片。"""
    payload = ctx.store.latest("itinerary")
    if not payload:
        return _err("还没有可渲染的行程，请先调用 plan_and_critique。")
    weather = ctx.store.latest("weather")
    cards = build_itinerary_cards(payload, weather)
    return _ok("已生成行程卡片。", cards=cards)


def render_map(ctx: SessionContext) -> dict[str, Any]:
    """生成前端高德地图渲染数据（点位 + 按天分组的路线折线）。"""
    payload = ctx.store.latest("itinerary")
    if not payload:
        return _err("还没有可渲染的行程，请先调用 plan_and_critique。")
    map_payload = build_map_payload(payload["itinerary"])
    return _ok(
        f"已生成地图数据：{len(map_payload['markers'])} 个点位。",
        **map_payload,
    )


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _merge_pois(left: list[POI], right: list[POI]) -> list[POI]:
    merged = list(left)
    seen = {p.poi_id for p in merged}
    for poi in right:
        if poi.poi_id not in seen:
            merged.append(poi)
            seen.add(poi.poi_id)
    return merged


def _profile_brief(profile: TravelProfile) -> dict[str, Any]:
    return {
        "destination": profile.destination,
        "days": profile.days,
        "interests": profile.interests,
        "budget_level": profile.budget_level,
        "pace": profile.pace,
        "must_visit": profile.must_visit,
        "avoid": profile.avoid,
    }


def _scored_to_dict(scored: ScoredPOI) -> dict[str, Any]:
    return {"poi": poi_to_dict(scored.poi), "score": scored.score, "reasons": scored.reasons}


def _scored_from_dict(data: dict[str, Any]) -> ScoredPOI:
    return ScoredPOI(
        poi=poi_from_dict(data["poi"]),
        score=float(data.get("score", 0.0)),
        reasons=list(data.get("reasons", [])),
    )
