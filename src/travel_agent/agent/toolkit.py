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

from travel_agent.agent.render import (
    build_itinerary_cards,
    build_map_payload,
)
from travel_agent.agent.serde import (
    itinerary_to_dict,
    poi_brief,
    poi_from_dict,
    poi_to_dict,
)
from travel_agent.agent.session import SessionContext, current_task_meta
from travel_agent.constraints import ConstraintSet
from travel_agent.planning_subgraph import plan_and_critique as run_plan_and_critique
from travel_agent.recommendation import score_pois
from travel_agent.schemas import POI, ScoredPOI, TravelProfile
from travel_agent.workflow import merge_profile
from travel_agent.workflow_rules import normalize_interests

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
def clean_profile_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """清洗/校验画像 SET 字段：类型不合法、枚举越界的值直接丢弃。"""
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
    if fields.get("budget_limit") is not None:
        try:
            value = float(fields["budget_limit"])
            if value > 0:
                cleaned["budget_limit"] = value
        except (TypeError, ValueError):
            pass
    if fields.get("party_size") is not None:
        try:
            value = int(fields["party_size"])
            if value > 0:
                cleaned["party_size"] = value
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
    return cleaned


def _record_set_observations(ctx: SessionContext, cleaned: dict[str, Any]) -> None:
    """SET 生效的偏好记为正向观察，供 L3 长期画像累积。"""
    for category in ("interests", "food_preference", "avoid"):
        for value in cleaned.get(category, []):
            ctx.pending_preference_observations.append(
                {"category": category, "value": str(value), "polarity": "positive"}
            )
    for category in ("budget_level", "pace", "transport_mode"):
        if category in cleaned:
            ctx.pending_preference_observations.append(
                {"category": category, "value": str(cleaned[category]), "polarity": "positive"}
            )


def _clear_profile_slot(profile: TravelProfile, name: str) -> None:
    """CLEAR 语义：列表清空，pace/transport_mode 复位默认，其余标量置 None。"""
    if name in ("interests", "food_preference", "must_visit", "avoid"):
        setattr(profile, name, [])
    elif name == "pace":
        profile.pace = "standard"
    elif name == "transport_mode":
        profile.transport_mode = "public_transport"
    elif hasattr(profile, name):
        setattr(profile, name, None)


def update_travel_profile(ctx: SessionContext, **fields: Any) -> dict[str, Any]:
    with ctx._state_lock:
        return _update_travel_profile_locked(ctx, **fields)


def _update_travel_profile_locked(ctx: SessionContext, **fields: Any) -> dict[str, Any]:
    """把已知出行信息合并进会话画像（destination/days/interests/budget 等）。"""
    cleaned = clean_profile_fields(fields)

    record_preferences = fields.get("_record_preferences", True)
    if record_preferences:
        _record_set_observations(ctx, cleaned)

    list_removal_fields = {
        "remove_interests": "interests",
        "remove_food_preference": "food_preference",
        "remove_avoid": "avoid",
    }
    list_removals: list[tuple[str, str]] = []
    for field_name, category in list_removal_fields.items():
        values = fields.get(field_name) or []
        if isinstance(values, str):
            values = [value.strip() for value in values.split(",") if value.strip()]
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            if record_preferences:
                ctx.pending_preference_observations.append(
                    {"category": category, "value": text, "polarity": "negative"}
                )
            list_removals.append((category, text))

    single_removals: list[tuple[str, str]] = []
    for field_name, category, valid_values in (
        ("remove_budget_level", "budget_level", VALID_BUDGET),
        ("remove_pace", "pace", VALID_PACE),
        ("remove_transport_mode", "transport_mode", VALID_TRANSPORT),
    ):
        value = fields.get(field_name)
        if value in valid_values:
            text = str(value)
            if record_preferences:
                ctx.pending_preference_observations.append(
                    {"category": category, "value": text, "polarity": "negative"}
                )
            single_removals.append((category, text))

    update = TravelProfile(
        destination=cleaned.get("destination"),
        days=cleaned.get("days"),
        start_date=cleaned.get("start_date"),
        budget_level=cleaned.get("budget_level"),
        budget_limit=cleaned.get("budget_limit"),
        interests=cleaned.get("interests", []),
        companions=cleaned.get("companions"),
        party_size=cleaned.get("party_size"),
        pace=cleaned.get("pace", "standard"),
        hotel_area=cleaned.get("hotel_area"),
        food_preference=cleaned.get("food_preference", []),
        must_visit=cleaned.get("must_visit", []),
        avoid=cleaned.get("avoid", []),
        transport_mode=cleaned.get("transport_mode", "public_transport"),
    )
    ctx.profile = merge_profile(ctx.profile, update)
    for category, value in list_removals:
        current = getattr(ctx.profile, category)
        setattr(ctx.profile, category, [item for item in current if item != value])
    for category, value in single_removals:
        if getattr(ctx.profile, category) != value:
            continue
        if category == "budget_level":
            ctx.profile.budget_level = None
        elif category == "pace":
            ctx.profile.pace = "standard"
        else:
            ctx.profile.transport_mode = "public_transport"
    if ctx.profile.interests:
        ctx.profile.interests = normalize_interests(ctx.profile.interests)
    missing = ctx.profile.missing_required_fields()
    return _ok(
        f"已更新出行画像，目的地={ctx.profile.destination}，天数={ctx.profile.days}。",
        profile=_profile_brief(ctx.profile),
        missing_required=missing,
    )


def apply_profile_patches(
    ctx: SessionContext,
    patches: dict[str, dict[str, Any]] | None = None,
    record_preferences: bool = True,
) -> dict[str, Any]:
    with ctx._state_lock:
        return _apply_profile_patches_locked(ctx, patches, record_preferences)


def _apply_profile_patches_locked(
    ctx: SessionContext,
    patches: dict[str, dict[str, Any]] | None,
    record_preferences: bool,
) -> dict[str, Any]:
    """按 SET/CLEAR patch 语义更新画像；未出现在 patches 中的槽位保持 UNCHANGED。

    patches 为 JSON 友好结构：``{slot: {"op": "set"|"clear", "value": ...}}``。
    """
    patches = patches or {}
    # 先 CLEAR 后 SET：「苏州不去了，换成杭州」这类替换句两个 patch 并存时新值生效
    cleared: list[str] = []
    for name, item in patches.items():
        if not isinstance(item, dict) or item.get("op") != "clear":
            continue
        _clear_profile_slot(ctx.profile, name)
        cleared.append(name)
        if record_preferences:
            ctx.pending_preference_observations.append(
                {
                    "category": name,
                    "value": str(item.get("value") or ""),
                    "polarity": "negative",
                }
            )
    set_fields = clean_profile_fields(
        {
            name: item.get("value")
            for name, item in patches.items()
            if isinstance(item, dict) and item.get("op") == "set"
        }
    )
    if set_fields:
        update = TravelProfile(
            destination=set_fields.get("destination"),
            days=set_fields.get("days"),
            start_date=set_fields.get("start_date"),
            budget_level=set_fields.get("budget_level"),
            budget_limit=set_fields.get("budget_limit"),
            interests=set_fields.get("interests", []),
            companions=set_fields.get("companions"),
            party_size=set_fields.get("party_size"),
            pace=set_fields.get("pace", "standard"),
            hotel_area=set_fields.get("hotel_area"),
            food_preference=set_fields.get("food_preference", []),
            must_visit=set_fields.get("must_visit", []),
            avoid=set_fields.get("avoid", []),
            transport_mode=set_fields.get("transport_mode", "public_transport"),
        )
        ctx.profile = merge_profile(ctx.profile, update)
        if record_preferences:
            _record_set_observations(ctx, set_fields)
    if ctx.profile.interests:
        ctx.profile.interests = normalize_interests(ctx.profile.interests)
    return _ok(
        f"已按 patch 更新画像（SET {len(set_fields)} 项，CLEAR {len(cleared)} 项），"
        f"目的地={ctx.profile.destination}，天数={ctx.profile.days}。",
        profile=_profile_brief(ctx.profile),
        set_fields=sorted(set_fields),
        cleared_fields=sorted(cleared),
    )


def request_travel_info(ctx: SessionContext, missing_fields: list[str] | None = None) -> dict[str, Any]:
    """信息不全时，生成面向用户的追问。"""
    missing = missing_fields or ctx.profile.missing_required_fields()
    questions = {
        "destination": "你想去哪个城市或地区？",
        "days": "计划玩几天？",
        "budget_level": "预算大概是什么档次（经济/中等/高）？",
        "interests": "更想体验哪些主题（自然/美食/历史/文化/购物等）？",
    }
    asks = [questions.get(field, f"请补充 {field}") for field in missing] or [
        "还有什么偏好需要我考虑吗？"
    ]
    question = "为了给你规划合适的行程，" + "；".join(asks)
    return _ok(question, question=question, missing_fields=missing)


def request_preference_guide(ctx: SessionContext, l3: TravelProfile | None = None) -> dict[str, Any]:
    """必要信息齐全后，引导用户补充兴趣、节奏等偏好（用户可说「随便」跳过）。"""
    from travel_agent.agent.preferences import build_preference_guide_text

    question = build_preference_guide_text(ctx.profile, l3)
    return _ok(question, question=question, awaiting_preferences=True)


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

    failures: list[str] = []
    try:
        preference = ctx.provider.search_pois(
            city=target_city, query_tags=tags or None, category=category, max_results=max_results
        )
    except Exception as exc:  # noqa: BLE001
        preference = []
        failures.append(f"偏好检索失败：{type(exc).__name__}")
    try:
        citywide = ctx.provider.search_pois(city=target_city, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        citywide = []
        failures.append(f"全量检索失败：{type(exc).__name__}")
    # 不会重复 POI
    # 偏好结果优先排在前面，citywide 结果只做补全
    candidates = _merge_pois(preference, citywide)
    fault_events = _consume_provider_faults(ctx, "search_pois")
    if fault_events:
        failures.append("检索服务返回异常，已重试并补全候选")
    if not candidates:
        return _err(
            f"没有检索到「{target_city}」的 POI（本地 seed 仅覆盖少数城市，建议配置高德 key）。",
            city=target_city,
        )

    ctx.remember_pois(candidates) # 把本轮检索到的 POI 列表“放进会话记忆”
    if failures:
        ctx.store.put(
            "recovery",
            {
                "operation": "search_poi",
                "reason": "；".join(failures),
                "suggestion": "建议出发前在地图或景区官方渠道复核开放时间与预约要求。",
            },
        )
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
        fallback_reason="；".join(failures) if failures else None,
    )


def check_weather(ctx: SessionContext, city: str | None = None) -> dict[str, Any]:
    """查询目的地天气（用于雨天户外冲突判断）。"""
    target_city = city or ctx.profile.destination
    if not target_city:
        return _err("缺少目的地城市，无法查询天气。")
    try:
        weather = ctx.provider.get_weather(target_city)
    except Exception as exc:  # noqa: BLE001
        ctx.store.put(
            "recovery",
            {
                "operation": "check_weather",
                "reason": f"天气查询失败：{type(exc).__name__}",
                "suggestion": "建议出发前通过官方天气服务复核，并准备室内备选点。",
            },
        )
        return _err(f"天气查询暂时失败：{exc}", fallback_reason=str(exc))
    weather_faults = _consume_provider_faults(ctx, "get_weather")
    if weather_faults:
        ctx.store.put(
            "recovery",
            {
                "operation": "check_weather",
                "reason": "天气服务返回异常或不完整结果",
                "suggestion": "建议出发前通过官方天气服务复核，并准备室内备选点。",
            },
        )
    artifact_id = ctx.store.put(
        "weather",
        {
            "city": target_city,
            "condition": weather.condition,
            "temperature_c": weather.temperature_c,
            "source": weather.source,
        },
    )
    return _ok(
        f"{target_city} 天气：{weather.condition}，约 {weather.temperature_c}°C（{weather.source}）。",
        artifact_id=artifact_id,
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
    route_payload = {
        "origin_poi_id": origin.poi_id,
        "destination_poi_id": destination.poi_id,
        "origin_name": origin.name,
        "destination_name": destination.name,
        "distance_km": route.distance_km,
        "duration_min": route.duration_min,
        "mode": route.mode,
        "source": route.source,
        "estimated_cost": _estimate_transport_cost(route.distance_km, route.mode),
    }
    artifact_id = ctx.store.put("routes", route_payload)
    return _ok(
        f"{origin.name} → {destination.name}：约 {route.duration_min} 分钟 / "
        f"{route.distance_km} 公里（{route.mode}, {route.source}）。",
        artifact_id=artifact_id,
        **route_payload,
    )


def build_constraints(ctx: SessionContext) -> dict[str, Any]:
    """从当前画像生成轻量 ConstraintSet，供评测/规划/trace 使用。"""
    constraints = ConstraintSet.from_profile(ctx.profile)
    artifact_id = ctx.store.put("constraints", constraints.to_dict())
    return _ok("已生成可验证旅行约束。", artifact_id=artifact_id, constraints=constraints.to_dict())


def search_restaurant(
    ctx: SessionContext,
    city: str | None = None,
    cuisine: str | None = None,
    area: str | None = None,
    budget_level: str | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """餐厅候选工具。第一版复用 food POI，高德/本地 provider 均可用。"""
    target_city = city or ctx.profile.destination
    if not target_city:
        return _err("缺少目的地城市，无法检索餐厅。")
    tags = [cuisine] if cuisine else list(ctx.profile.food_preference or [])
    restaurants = ctx.provider.search_pois(
        city=target_city,
        query_tags=[tag for tag in tags if tag],
        category="food",
        max_results=max_results * 2,
    )
    target_budget = budget_level or ctx.profile.budget_level
    filtered = []
    for poi in restaurants:
        if target_budget and poi.price_level != target_budget:
            continue
        if area and area not in poi.name and area not in " ".join(poi.tags):
            continue
        filtered.append(poi)
    if not filtered:
        filtered = restaurants[:max_results]
    filtered = filtered[:max_results]
    ctx.remember_pois(filtered)
    artifact_id = ctx.store.put(
        "restaurants",
        {
            "city": target_city,
            "restaurants": [poi_to_dict(poi) for poi in filtered],
            "cuisine": cuisine,
            "area": area,
            "budget_level": target_budget,
        },
    )
    return _ok(
        f"在「{target_city}」找到 {len(filtered)} 个餐厅候选。",
        artifact_id=artifact_id,
        count=len(filtered),
        restaurants=[poi_brief(poi) for poi in filtered],
    )


def search_hotel(
    ctx: SessionContext,
    city: str | None = None,
    area: str | None = None,
    budget_level: str | None = None,
    min_rating: float | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """酒店候选工具。优先使用 provider 的酒店 POI，缺数据时退回确定性 mock。"""
    target_city = city or ctx.profile.destination
    if not target_city:
        return _err("缺少目的地城市，无法检索酒店。")
    target_area = area or ctx.profile.hotel_area or "核心商圈"
    target_budget = budget_level or ctx.profile.budget_level or "mid"
    poi_hotels = ctx.provider.search_pois(
        city=target_city,
        query_tags=None,
        category="hotel",
        max_results=max_results * 2,
    )
    ctx.remember_pois(poi_hotels)
    hotels = _hotels_from_pois(poi_hotels, target_area, target_budget)
    if not hotels:
        hotels = _mock_hotels(target_city, target_area, target_budget)
    if min_rating is not None:
        hotels = [hotel for hotel in hotels if hotel["rating"] >= float(min_rating)]
    hotels = hotels[:max_results]
    artifact_id = ctx.store.put(
        "hotels",
        {
            "city": target_city,
            "area": target_area,
            "budget_level": target_budget,
            "hotels": hotels,
        },
    )
    return _ok(
        f"在「{target_city}」找到 {len(hotels)} 个酒店候选。",
        artifact_id=artifact_id,
        count=len(hotels),
        hotels=hotels,
    )


def estimate_budget(
    ctx: SessionContext,
    city: str | None = None,
    days: int | None = None,
    companions: int | None = None,
    budget_level: str | None = None,
    hotel_required: bool = True,
) -> dict[str, Any]:
    """确定性预算估算：住宿、餐饮、门票、市内交通与总价区间。"""
    target_city = city or ctx.profile.destination or "目的地"
    trip_days = max(1, int(days or ctx.profile.days or 1))
    people = max(
        1,
        int(companions or ctx.profile.party_size or _companions_to_count(ctx.profile.companions) or 1),
    )
    level = budget_level or ctx.profile.budget_level or "mid"
    per_person_meal = {"low": 45, "mid": 90, "high": 180}.get(level, 90)
    hotel_per_room = {"low": 280, "mid": 550, "high": 1200}.get(level, 550)
    ticket_per_day = {"low": 50, "mid": 120, "high": 220}.get(level, 120)
    transport_per_day = {"low": 25, "mid": 45, "high": 120}.get(level, 45)
    rooms = max(1, (people + 1) // 2)
    nights = max(0, trip_days - 1)
    hotel_total = hotel_per_room * rooms * nights if hotel_required else 0
    meal_total = per_person_meal * people * trip_days * 2
    ticket_total = ticket_per_day * people * trip_days
    transport_total = transport_per_day * people * trip_days
    total = hotel_total + meal_total + ticket_total + transport_total
    estimate = {
        "city": target_city,
        "days": trip_days,
        "companions": people,
        "budget_level": level,
        "hotel_required": hotel_required,
        "hotel": hotel_total,
        "meals": meal_total,
        "tickets": ticket_total,
        "inner_city_transport": transport_total,
        "total_low": round(total * 0.85, 2),
        "total_high": round(total * 1.15, 2),
    }
    artifact_id = ctx.store.put("budget", estimate)
    return _ok(
        f"{target_city}{trip_days}天预算估算：约 {estimate['total_low']} - {estimate['total_high']} 元。",
        artifact_id=artifact_id,
        budget=estimate,
    )


# --------------------------------------------------------------------------- #
# 推荐打分 + 可控规划子图
# --------------------------------------------------------------------------- #
def recommend_candidates(
    ctx: SessionContext,
    top_k: int = 12,
    artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """只按明确 artifact_id 对候选 POI/餐厅做重排。"""
    records, error = _planner_input_records(
        ctx,
        artifact_ids,
        allowed_kinds={"candidates", "restaurants"},
        legacy_kinds=("candidates", "restaurants"),
    )
    if error:
        return _err(error)
    pois_by_id: dict[str, POI] = {}
    source_ids: list[str] = []
    for record in records:
        payload = record["payload"]
        raw_pois = payload.get("pois") or payload.get("restaurants") or []
        for raw_poi in raw_pois:
            try:
                poi = poi_from_dict(raw_poi)
            except Exception:
                continue
            pois_by_id[poi.poi_id] = poi
        source_ids.append(record["artifact_id"])
    if not pois_by_id:
        return _err("明确指定的 artifact 中没有可规划 POI，请先调用 search_poi。")
    pois = list(pois_by_id.values())
    ranked = score_pois(pois, ctx.profile)
    top = ranked[: max(1, top_k)]
    artifact_id = ctx.store.put(
        "ranked",
        {
            "pois": [_scored_to_dict(s) for s in ranked],
            "source_artifact_ids": source_ids,
        },
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


def plan_and_critique(
    ctx: SessionContext,
    max_iters: int = 3,
    artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """按明确 artifact_id 聚合全部领域结果并生成 TravelPlan。"""
    records, error = _planner_input_records(
        ctx,
        artifact_ids,
        allowed_kinds={
            "candidates",
            "ranked",
            "weather",
            "hotels",
            "restaurants",
            "routes",
            "budget",
            "constraints",
            "itinerary",
        },
        legacy_kinds=(
            "ranked",
            "candidates",
            "weather",
            "hotels",
            "restaurants",
            "routes",
            "budget",
            "constraints",
            "itinerary",
        ),
        include_current_task_outputs=True,
    )
    if error:
        return _err(error)
    ranked_artifact = next(
        (record["payload"] for record in reversed(records) if record["kind"] == "ranked"),
        None,
    )
    if not ranked_artifact:
        ranked_artifact = _ranked_from_previous_itinerary(records, ctx)
    if not ranked_artifact:
        return _err("明确指定的 artifact 中没有排序候选，请先调用 recommend_candidates。")
    if not ctx.profile.destination or not ctx.profile.days:
        return _err("缺少 destination 或 days，无法规划，请先补全画像。")

    ranked = [_scored_from_dict(d) for d in ranked_artifact["pois"]]
    result = run_plan_and_critique(
        ranked_pois=ranked,
        profile=ctx.profile,
        route_estimator=ctx.provider,
        max_iters=max_iters,
    )
    revision_directives = dict(current_task_meta().get("revision_directives") or {})
    result = _apply_revision_directives(
        result,
        ranked,
        ctx,
        revision_directives,
    )
    itinerary_dict = itinerary_to_dict(result.itinerary)
    original_itinerary_dict = itinerary_to_dict(result.original_itinerary)
    route_faults = _consume_provider_faults(ctx, "estimate_route")
    if route_faults or any(
        (stop.get("route_from_previous") or {}).get("source") == "haversine_recovery_estimate"
        for itinerary in (itinerary_dict, original_itinerary_dict)
        for day in itinerary.get("days", [])
        for stop in day.get("stops", [])
    ):
        ctx.store.put(
            "recovery",
            {
                "operation": "estimate_route",
                "reason": "实时路线服务暂时不可用，已改用直线距离估算",
                "suggestion": "建议出发前使用地图导航复核实际路况和通勤时间。",
            },
        )
    domain_inputs = _collect_domain_inputs(records)
    artifact_id = ctx.store.put(
        "itinerary",
        {
            "itinerary": itinerary_dict,
            "original_itinerary": original_itinerary_dict,
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
            "source_artifact_ids": [record["artifact_id"] for record in records],
            "domain_inputs": domain_inputs,
            "revision_directives": revision_directives,
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


def _apply_revision_directives(
    result: Any,
    ranked: list[ScoredPOI],
    ctx: SessionContext,
    directives: dict[str, Any],
) -> Any:
    """Apply deterministic non-profile revisions before the plan artifact is written."""
    indoor_days = {
        int(day) for day in (directives.get("indoor_days") or []) if str(day).isdigit()
    }
    if not indoor_days:
        return result

    from dataclasses import replace

    from travel_agent.critic import critique_itinerary

    used_ids = {stop.poi.poi_id for day in result.itinerary.days for stop in day.stops}
    indoor_pool = [item.poi for item in ranked if item.poi.indoor and item.poi.poi_id not in used_ids]
    day_stops = [list(day.stops) for day in result.itinerary.days]
    donors = [
        (day_index, stop_index)
        for day_index, day in enumerate(result.itinerary.days)
        if day.day_index not in indoor_days
        for stop_index, stop in enumerate(day.stops)
        if stop.poi.indoor
    ]
    changed = False
    days = []
    for day_index, day in enumerate(result.itinerary.days):
        if day.day_index not in indoor_days:
            continue
        stops = []
        for stop in day_stops[day_index]:
            if stop.poi.indoor:
                stops.append(stop)
                continue
            replacement = indoor_pool.pop(0) if indoor_pool else None
            if replacement is None and donors:
                donor_day, donor_index = donors.pop(0)
                donor_stop = day_stops[donor_day][donor_index]
                replacement = donor_stop.poi
                day_stops[donor_day][donor_index] = replace(
                    donor_stop,
                    poi=stop.poi,
                    duration_min=stop.poi.estimated_duration_min,
                    note="与指定日期的室内活动互换",
                    route_from_previous=None,
                )
            if replacement is None:
                stops.append(stop)
                continue
            stops.append(
                replace(
                    stop,
                    poi=replacement,
                    duration_min=replacement.estimated_duration_min,
                    note="按本轮修改要求替换为室内活动",
                    route_from_previous=None,
                )
            )
            changed = True
        day_stops[day_index] = stops
    if not changed:
        return result
    for day_index, day in enumerate(result.itinerary.days):
        days.append(
            replace(
                day,
                stops=day_stops[day_index],
                theme="室内活动" if day.day_index in indoor_days else day.theme,
            )
        )
    itinerary = replace(result.itinerary, days=days)
    critic_result = critique_itinerary(itinerary, ctx.profile)
    return replace(
        result,
        itinerary=itinerary,
        critic_result=critic_result,
        revision_notes=list(result.revision_notes) + ["按要求将指定日期调整为室内活动"],
    )


def _planner_input_records(
    ctx: SessionContext,
    artifact_ids: list[str] | None,
    *,
    allowed_kinds: set[str],
    legacy_kinds: tuple[str, ...],
    include_current_task_outputs: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """解析工具输入；planner 永远只读任务绑定 ID，非 planner 保留旧链兼容。"""
    meta = current_task_meta()
    planner_task = meta.get("agent") == "planner"
    requested = [str(aid) for aid in (artifact_ids or []) if aid]

    if planner_task:
        bound = [str(aid) for aid in (meta.get("artifact_ids") or []) if aid]
        if include_current_task_outputs:
            bound.extend(
                ctx.store.artifact_ids_for_task(
                    str(meta.get("request_id") or ""),
                    str(meta.get("task_id") or ""),
                    agent="planner",
                )
            )
        allowed_ids = list(dict.fromkeys(bound))
        if requested:
            unbound = [artifact_id for artifact_id in requested if artifact_id not in allowed_ids]
            if unbound:
                return [], f"planner 收到未绑定的 artifact_id: {unbound}"
        else:
            requested = allowed_ids
    elif not requested:
        # 兼容 V0/离线单 Agent；该分支不属于 planner Subagent。
        requested = [
            artifact_id
            for kind in legacy_kinds
            if (artifact_id := ctx.store.latest_id(kind)) is not None
        ]

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for artifact_id in dict.fromkeys(requested):
        record = ctx.store.get_record(artifact_id)
        if record is None:
            missing.append(artifact_id)
            continue
        if record.get("kind") not in allowed_kinds:
            continue
        payload = ctx.store.get(artifact_id)
        if not isinstance(payload, dict):
            missing.append(artifact_id)
            continue
        records.append(
            {
                "artifact_id": artifact_id,
                "kind": str(record.get("kind") or ""),
                "payload": payload,
            }
        )
    if missing:
        return [], f"artifact_id 不存在或 payload 非法: {missing}"
    return records, None


def _ranked_from_previous_itinerary(
    records: list[dict[str, Any]],
    ctx: SessionContext,
) -> dict[str, Any] | None:
    """行程修改时从明确绑定的旧 itinerary 恢复候选，禁止读 latest。"""
    previous = next(
        (record for record in reversed(records) if record["kind"] == "itinerary"),
        None,
    )
    if previous is None:
        return None
    itinerary = previous["payload"].get("itinerary") or {}
    pois_by_id: dict[str, POI] = {}
    for day in itinerary.get("days") or []:
        for stop in day.get("stops") or []:
            raw_poi = stop.get("poi") if isinstance(stop, dict) else None
            if not isinstance(raw_poi, dict):
                continue
            try:
                poi = poi_from_dict(raw_poi)
            except Exception:
                continue
            pois_by_id[poi.poi_id] = poi
    if not pois_by_id:
        return None
    ranked = score_pois(list(pois_by_id.values()), ctx.profile)
    return {
        "pois": [_scored_to_dict(item) for item in ranked],
        "source_artifact_ids": [previous["artifact_id"]],
    }


def _collect_domain_inputs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """把全部明确绑定的领域结果写入 Planner 产出的 plan artifact。"""
    inputs: dict[str, Any] = {
        "attractions": [],
        "hotels": [],
        "restaurants": [],
        "transport": [],
        "weather": [],
        "budgets": [],
        "constraints": [],
        "previous_itineraries": [],
    }
    key_by_kind = {
        "candidates": "attractions",
        "hotels": "hotels",
        "restaurants": "restaurants",
        "routes": "transport",
        "weather": "weather",
        "budget": "budgets",
        "constraints": "constraints",
        "itinerary": "previous_itineraries",
    }
    for record in records:
        target = key_by_kind.get(record["kind"])
        if target is None:
            continue
        inputs[target].append(
            {
                "artifact_id": record["artifact_id"],
                "payload": record["payload"],
            }
        )
    return inputs


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
    cards.extend(
        _build_supplement_cards(
            restaurants=ctx.store.latest("restaurants"),
            hotels=ctx.store.latest("hotels"),
            budget=ctx.store.latest("budget"),
        )
    )
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


def _validate_plan_gate(
    ctx: SessionContext,
    plan_artifact_id: str | None,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
) -> tuple[str, dict | None]:
    """Renderer Gate 校验：返回 (reason, record)，reason 非空即拒绝渲染。

    校验项：1) Artifact 存在；2) 类型为 itinerary；3) 由 planner 产出；
    4) 计划状态不是 failed（critic 结果可读）。渲染与否的后续状态由
    调用方结合 Reviewer 结果决定（critical 未解决 → 只能标 incomplete）。
    """
    if not plan_artifact_id:
        return "缺少 plan_artifact_id，禁止凭空渲染行程", None
    record = ctx.store.get_record(plan_artifact_id)
    if record is None:
        return f"plan artifact 不存在: {plan_artifact_id}", None
    if record.get("kind") != "itinerary":
        return f"plan artifact 类型不是 itinerary: {record.get('kind')}", None
    if record.get("agent") not in allowed_agents:
        return (
            "plan artifact 产出者无权渲染: "
            f"agent={record.get('agent')}, allowed={sorted(allowed_agents)}"
        ), None
    payload = record.get("payload") or {}
    if not isinstance(payload.get("itinerary"), dict):
        return "plan payload 缺少 itinerary，状态异常（failed）", None
    return "", record


def gated_render_itinerary(
    ctx: SessionContext,
    plan_artifact_id: str | None,
    *,
    mark_incomplete: bool = False,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
) -> dict[str, Any]:
    """经 Renderer Gate 的行程渲染：必须绑定 planner 产出的 plan_artifact_id。

    ``mark_incomplete=True``（critic 未通过或存在未解决 critical 问题）时
    只能渲染 incomplete 状态，不得标记为成功行程。
    """
    reason, record = _validate_plan_gate(ctx, plan_artifact_id, allowed_agents)
    if reason:
        return _err(reason, gate_status="rejected", plan_artifact_id=plan_artifact_id)
    payload = record["payload"]
    critic_passed = bool(payload.get("critic", {}).get("passed") is True)
    incomplete = mark_incomplete or not critic_passed
    domain_inputs = payload.get("domain_inputs") or {}
    weather = _last_domain_payload(domain_inputs, "weather")
    cards = build_itinerary_cards(payload, weather)
    cards.extend(
        _build_supplement_cards(
            restaurants=_last_domain_payload(domain_inputs, "restaurants"),
            hotels=_last_domain_payload(domain_inputs, "hotels"),
            budget=_last_domain_payload(domain_inputs, "budgets"),
        )
    )
    gate_status = "rendered_incomplete" if incomplete else "rendered"
    summary = (
        "行程存在未解决问题，仅渲染 incomplete 状态。"
        if incomplete
        else "已生成行程卡片。"
    )
    return _ok(
        summary,
        cards=cards,
        gate_status=gate_status,
        plan_artifact_id=plan_artifact_id,
        critic_passed=critic_passed,
    )


def _last_domain_payload(domain_inputs: dict[str, Any], key: str) -> dict[str, Any] | None:
    entries = domain_inputs.get(key) or []
    if not isinstance(entries, list) or not entries:
        return None
    item = entries[-1]
    payload = item.get("payload") if isinstance(item, dict) else None
    return payload if isinstance(payload, dict) else None


def _build_supplement_cards(
    *,
    restaurants: dict[str, Any] | None = None,
    hotels: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Planner 领域输入的轻量卡片，不依赖未暂存 renderer 扩展。"""
    cards: list[dict[str, Any]] = []
    if restaurants:
        cards.append(
            {
                "type": "restaurants",
                "city": restaurants.get("city"),
                "items": list(restaurants.get("restaurants") or [])[:6],
            }
        )
    if hotels:
        cards.append(
            {
                "type": "hotels",
                "city": hotels.get("city"),
                "items": list(hotels.get("hotels") or [])[:5],
            }
        )
    if budget:
        cards.append(
            {
                "type": "budget",
                "city": budget.get("city"),
                "days": budget.get("days"),
                "total_low": budget.get("total_low"),
                "total_high": budget.get("total_high"),
            }
        )
    return cards


def gated_render_map(
    ctx: SessionContext,
    plan_artifact_id: str | None,
    *,
    mark_incomplete: bool = False,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
) -> dict[str, Any]:
    """经 Renderer Gate 的地图渲染：依赖完整计划时绑定同一 plan_artifact_id。"""
    reason, record = _validate_plan_gate(ctx, plan_artifact_id, allowed_agents)
    if reason:
        return _err(reason, gate_status="rejected", plan_artifact_id=plan_artifact_id)
    payload = record["payload"]
    critic_passed = bool(payload.get("critic", {}).get("passed") is True)
    incomplete = mark_incomplete or not critic_passed
    map_payload = build_map_payload(payload["itinerary"])
    return _ok(
        f"已生成地图数据：{len(map_payload['markers'])} 个点位。",
        gate_status="rendered_incomplete" if incomplete else "rendered",
        plan_artifact_id=plan_artifact_id,
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


def _estimate_transport_cost(distance_km: float, mode: str) -> float:
    if mode == "walk":
        return 0.0
    if mode in {"taxi", "drive"}:
        return round(14 + max(0.0, distance_km - 3) * 2.5, 2)
    return 4.0 if distance_km <= 6 else 7.0


def _mock_hotels(city: str, area: str, budget_level: str) -> list[dict[str, Any]]:
    price = {"low": 280, "mid": 550, "high": 1200}.get(budget_level, 550)
    suffix = {"low": "精选酒店", "mid": "舒适酒店", "high": "精品酒店"}.get(
        budget_level,
        "舒适酒店",
    )
    return [
        {
            "hotel_id": f"mock_hotel_{city}_{index}",
            "name": f"{city}{area}{suffix}{index}",
            "city": city,
            "area": area,
            "budget_level": budget_level,
            "price_per_night": price + index * 40,
            "rating": round(4.6 - index * 0.1, 1),
            "source": "mock",
        }
        for index in range(1, 4)
    ]


def _hotels_from_pois(pois: list[POI], area: str, budget_level: str) -> list[dict[str, Any]]:
    hotels = []
    base_price = {"low": 280, "mid": 550, "high": 1200}.get(budget_level, 550)
    for index, poi in enumerate(pois, start=1):
        if poi.category != "hotel" and "hotel" not in poi.tags and "accommodation" not in poi.tags:
            continue
        if poi.price_level and poi.price_level != budget_level:
            continue
        hotels.append(
            {
                "hotel_id": poi.poi_id,
                "name": poi.name,
                "city": poi.city,
                "area": area,
                "budget_level": budget_level,
                "price_per_night": base_price + min(index, 5) * 35,
                "rating": poi.rating,
                "lat": poi.lat,
                "lng": poi.lng,
                "source": poi.source,
            }
        )
    return hotels


def _companions_to_count(companions: str | None) -> int | None:
    if not companions:
        return None
    text = companions.strip()
    for marker, count in {
        "情侣": 2,
        "夫妻": 2,
        "老人": 2,
        "亲子": 3,
        "家庭": 3,
        "朋友": 2,
    }.items():
        if marker in text:
            return count
    return None


def _profile_brief(profile: TravelProfile) -> dict[str, Any]:
    return {
        "destination": profile.destination,
        "days": profile.days,
        "interests": profile.interests,
        "budget_level": profile.budget_level,
        "budget_limit": profile.budget_limit,
        "companions": profile.companions,
        "party_size": profile.party_size,
        "start_date": profile.start_date,
        "hotel_area": profile.hotel_area,
        "food_preference": profile.food_preference,
        "transport_mode": profile.transport_mode,
        "pace": profile.pace,
        "must_visit": profile.must_visit,
        "avoid": profile.avoid,
    }


def _consume_provider_faults(ctx: SessionContext, operation: str) -> list[dict[str, str]]:
    consume = getattr(ctx.provider, "consume_fault_events", None)
    if not callable(consume):
        return []
    try:
        events = consume(operation)
    except Exception:  # noqa: BLE001
        return []
    return list(events) if isinstance(events, list) else []


def _scored_to_dict(scored: ScoredPOI) -> dict[str, Any]:
    return {"poi": poi_to_dict(scored.poi), "score": scored.score, "reasons": scored.reasons}


def _scored_from_dict(data: dict[str, Any]) -> ScoredPOI:
    return ScoredPOI(
        poi=poi_from_dict(data["poi"]),
        score=float(data.get("score", 0.0)),
        reasons=list(data.get("reasons", [])),
    )
