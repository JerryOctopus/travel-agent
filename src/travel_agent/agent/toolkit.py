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

import copy
import re
from datetime import date
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
TRANSPORT_ALIASES = {
    "walking": "walk",
    "transit": "public_transport",
    "public transit": "public_transport",
    "public-transit": "public_transport",
    "driving": "drive",
}


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
    if current_task_meta().get("agent") == "attraction":
        from travel_agent.critic import poi_matches_interest

        raw_interests = (ctx.profile.constraint_state or {}).get("interests") or []
        if isinstance(raw_interests, str):
            raw_interests = [raw_interests]
        candidates = [
            poi
            for poi in candidates
            if (
                poi.category not in {"food", "hotel", "transport"}
                or (
                    poi.category == "food"
                    and any(poi_matches_interest(poi, str(item)) for item in raw_interests)
                )
            )
            and not _is_unavailable_or_infrastructure_poi(poi, ctx.profile)
        ]
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
    normalized_mode = TRANSPORT_ALIASES.get(str(mode or "").strip().lower(), mode)
    transport = normalized_mode if normalized_mode in VALID_TRANSPORT else ctx.profile.transport_mode
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
        "walking_distance_km": route.walking_distance_km,
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
    payload = constraints.to_dict()
    # Explicit constraints are authoritative where the compact profile cannot
    # represent the requirement (or intentionally preserves the user's literal
    # wording, e.g. dietary/accessibility requirements).
    for key, value in (ctx.profile.constraint_state or {}).items():
        # A stale/empty structured mirror must not erase a non-empty compact
        # hard constraint promoted from a fixed event or explicit user patch.
        if key in payload and payload[key] not in (None, "", [], {}) and value in (
            None,
            "",
            [],
            {},
        ):
            continue
        payload[key] = value
    if payload.get("self_driving_allowed") is False and payload.get("transport_mode") == "drive":
        payload["transport_mode"] = "public_transport"
    artifact_id = ctx.store.put("constraints", payload)
    return _ok("已生成可验证旅行约束。", artifact_id=artifact_id, constraints=payload)


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
    dietary = [
        str(item).strip().lower()
        for item in ((ctx.profile.constraint_state or {}).get("dietary") or [])
        if str(item).strip()
    ]
    halal_required = any(
        marker in item
        for item in dietary
        for marker in ("清真", "halal")
    )
    if halal_required:
        restaurants = [
            poi
            for poi in restaurants
            if any(
                marker in " ".join([poi.name, poi.address or "", *poi.tags]).lower()
                for marker in ("清真", "halal")
            )
        ]
    required_halal_count = max(1, int(ctx.profile.days or 1))
    if halal_required and len(restaurants) < required_halal_count:
        broad = ctx.provider.search_pois(
            city=target_city,
            query_tags=None,
            category="food",
            max_results=max_results * 3,
        )
        broad_halal = [
            poi
            for poi in broad
            if any(
                marker in " ".join([poi.name, poi.address or "", *poi.tags]).lower()
                for marker in ("清真", "halal")
            )
        ]
        seen_ids = {poi.poi_id for poi in restaurants}
        restaurants.extend(poi for poi in broad_halal if poi.poi_id not in seen_ids)
    if halal_required and len(restaurants) < required_halal_count:
        explicit_halal = ctx.provider.search_pois(
            city=target_city,
            query_tags=["清真餐厅"],
            category="food",
            max_results=max_results * 3,
        )
        explicit_halal = [
            poi
            for poi in explicit_halal
            if any(
                marker in " ".join([poi.name, poi.address or "", *poi.tags]).lower()
                for marker in ("清真", "halal")
            )
        ]
        seen_ids = {poi.poi_id for poi in restaurants}
        restaurants.extend(poi for poi in explicit_halal if poi.poi_id not in seen_ids)
    target_budget = budget_level or ctx.profile.budget_level
    per_person_limit = (ctx.profile.constraint_state or {}).get("budget_per_person_cny")
    filtered = []
    for poi in restaurants:
        if target_budget and poi.price_level != target_budget:
            continue
        if per_person_limit is not None and poi.average_cost is not None:
            if poi.average_cost > float(per_person_limit):
                continue
        if area and area not in poi.name and area not in (poi.address or "") and area not in " ".join(poi.tags):
            continue
        filtered.append(poi)
    # ``area`` is a search preference, not automatically a user-level hard
    # constraint.  If it yields fewer restaurants than trip days, retain the
    # matching results first and supplement with other compliant city results
    # so every day can have a grounded meal.
    required_meals = max(1, int(ctx.profile.days or 1))
    if len(filtered) < required_meals:
        seen_ids = {poi.poi_id for poi in filtered}
        for poi in restaurants:
            if poi.poi_id in seen_ids:
                continue
            if target_budget and poi.price_level != target_budget:
                continue
            if per_person_limit is not None and poi.average_cost is not None:
                if poi.average_cost > float(per_person_limit):
                    continue
            filtered.append(poi)
            seen_ids.add(poi.poi_id)
            if len(filtered) >= required_meals:
                break
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
    user_area = ctx.profile.hotel_area
    requested_area = area or user_area
    target_area = requested_area or "核心商圈"
    target_budget = budget_level or ctx.profile.budget_level or "mid"
    poi_hotels = ctx.provider.search_pois(
        city=target_city,
        query_tags=[target_area] if requested_area else None,
        category="hotel",
        max_results=max_results * 2,
    )
    area_evidenced = not requested_area
    if requested_area:
        area_term = re.sub(r"(?:附近|核心|周边|商圈)$", "", str(requested_area)).strip()
        matching_hotels = [
            poi
            for poi in poi_hotels
            if area_term and area_term in " ".join([poi.name, poi.address or ""])
        ]
        if user_area:
            if not matching_hotels:
                broad_hotels = ctx.provider.search_pois(
                    city=target_city,
                    query_tags=None,
                    category="hotel",
                    max_results=max_results * 2,
                )
                matching_hotels = [
                    poi
                    for poi in broad_hotels
                    if area_term and area_term in " ".join([poi.name, poi.address or ""])
                ]
            poi_hotels = matching_hotels
            area_evidenced = bool(matching_hotels)
        elif matching_hotels:
            poi_hotels = matching_hotels
            area_evidenced = True
        else:
            # A Worker-selected area is an optimization hint.  Keep grounded
            # city results when the provider has no exact-area match.
            poi_hotels = ctx.provider.search_pois(
                city=target_city,
                query_tags=None,
                category="hotel",
                max_results=max_results * 2,
            )
            area_evidenced = False
    ctx.remember_pois(poi_hotels)
    effective_area = target_area if area_evidenced else None
    hotels = _hotels_from_pois(poi_hotels, effective_area, target_budget)
    if not hotels and not requested_area:
        hotels = _mock_hotels(target_city, target_area, target_budget)
    if min_rating is not None:
        hotels = [hotel for hotel in hotels if hotel["rating"] >= float(min_rating)]
    hotels = hotels[:max_results]
    artifact_id = ctx.store.put(
        "hotels",
        {
            "city": target_city,
            "area": effective_area,
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
        int(
            companions
            or ctx.profile.party_size
            or (ctx.profile.constraint_state or {}).get("traveler_count")
            or _companions_to_count(ctx.profile.companions)
            or 1
        ),
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
    collected: list[POI] = []
    source_ids: list[str] = []
    for record in records:
        payload = record["payload"]
        raw_pois = payload.get("pois") or payload.get("restaurants") or []
        for raw_poi in raw_pois:
            try:
                poi = poi_from_dict(raw_poi)
            except Exception:
                continue
            # Accommodation and transport infrastructure are evidence/context,
            # not day-activity candidates.  Letting them into the shared ranked
            # pool produced zero-duration hotel stops and bus stops masquerading
            # as museums.
            if poi.category in {"hotel", "transport"} or _is_unavailable_or_infrastructure_poi(
                poi, ctx.profile
            ):
                continue
            if not _poi_city_allowed_for_plan(poi, ctx.profile):
                continue
            collected.append(poi)
        source_ids.append(record["artifact_id"])
    pois = _dedupe_plannable_entities(collected, ctx.profile)
    if not pois:
        return _err("明确指定的 artifact 中没有可规划 POI，请先调用 search_poi。")
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
    route_estimator: Any | None = None,
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

    ranked = _merge_ranked_planner_inputs(ranked_artifact, records, ctx.profile)
    # The three-iteration deterministic critic loop is an architecture
    # invariant, not a model-tunable argument.  Do not let a tool call weaken
    # it (or expand it beyond the calibrated bound).
    effective_max_iters = 3
    result = run_plan_and_critique(
        ranked_pois=ranked,
        profile=ctx.profile,
        route_estimator=ctx.provider if route_estimator is None else route_estimator,
        max_iters=effective_max_iters,
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
    return_plan = _build_return_plan(ctx.profile, itinerary_dict, domain_inputs)
    lodging_plan = _build_lodging_plan(ctx.profile, domain_inputs)
    budget_plan = _build_budget_plan(ctx.profile, domain_inputs, lodging_plan)
    fixed_event_plan = _build_fixed_event_plan(ctx.profile, domain_inputs)
    mobility_plan = _build_mobility_plan(ctx.profile, itinerary_dict)
    candidate_verification = _build_candidate_verification(ctx.profile, ranked)
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
            "return_plan": return_plan,
            "lodging_plan": lodging_plan,
            "budget_plan": budget_plan,
            "fixed_event_plan": fixed_event_plan,
            "mobility_plan": mobility_plan,
            "candidate_verification": candidate_verification,
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


def _build_candidate_verification(
    profile: TravelProfile,
    ranked_pois: list[ScoredPOI],
) -> dict[str, Any] | None:
    """Record the outcome of explicit named-candidate suitability checks."""
    state = profile.constraint_state or {}
    terms = state.get("candidate_attractions") or []
    if isinstance(terms, str):
        terms = [terms]
    if not terms:
        return None
    weekday = str(state.get("weekday") or "").strip()
    results: list[dict[str, Any]] = []
    for raw_term in terms:
        term = str(raw_term).strip()
        matches = [
            item.poi for item in ranked_pois
            if _candidate_name_matches(item.poi, term)
        ]
        suitable = next(
            (poi for poi in matches if _candidate_weekday_suitable(poi, weekday)),
            None,
        )
        if suitable is not None:
            results.append({
                "requested_name": term,
                "status": "suitable",
                "matched_name": suitable.name,
                "weekday": weekday or None,
                "opening_hours": suitable.opening_hours,
                "source": suitable.source,
            })
        else:
            closed = matches[0] if matches else None
            results.append({
                "requested_name": term,
                "status": "unsuitable_or_unverified",
                "matched_name": closed.name if closed else None,
                "weekday": weekday or None,
                "opening_hours": closed.opening_hours if closed else None,
                "source": closed.source if closed else None,
                "reason": "当天闭馆或未取得可核验的地点本体开放证据",
            })
    return {"status": "verified_named_candidates", "results": results}


def _candidate_name_matches(poi: Any, term: str) -> bool:
    name = str(getattr(poi, "name", "") or "").strip()
    if not name or not term or getattr(poi, "category", None) in {"food", "hotel", "transport"}:
        return False
    if any(marker in name for marker in ("服务中心", "游客中心", "停车场", "售票处")):
        return False
    return term in name or name in term


def _candidate_weekday_suitable(poi: Any, weekday: str) -> bool:
    hours = str(getattr(poi, "opening_hours", "") or "").replace(" ", "")
    if not weekday or not hours:
        return bool(hours)
    return not (
        f"{weekday}全天不开放" in hours
        or f"{weekday}闭馆" in hours
        or (weekday == "周一" and "周一闭馆" in hours)
    )


def _build_fixed_event_plan(
    profile: TravelProfile,
    domain_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    """Expose fixed appointments and grounded transfer buffers explicitly."""
    events = [
        dict(event)
        for event in ((profile.constraint_state or {}).get("fixed_events") or [])
        if isinstance(event, dict)
    ]
    if not events:
        return None
    routes = [
        entry.get("payload", entry)
        for entry in (domain_inputs.get("transport") or [])
        if isinstance(entry, dict)
    ]
    items: list[dict[str, Any]] = []
    for event in events:
        location = str(event.get("location") or "").strip()
        route = next(
            (
                item for item in routes
                if isinstance(item, dict)
                and (
                    location in str(item.get("destination_name") or "")
                    or location in str(item.get("origin_name") or "")
                )
            ),
            None,
        )
        duration = int(route.get("duration_min") or 0) if route else None
        items.append({
            **event,
            "transfer_duration_min": duration,
            "recommended_departure": _offset_clock(event.get("start"), -duration if duration else None),
            "recommended_return_arrival": _offset_clock(event.get("end"), duration),
            "route_source": route.get("source") if route else None,
        })
    return {"status": "fixed_appointment_buffers", "events": items}


def _offset_clock(value: object, delta_minutes: int | None) -> str | None:
    if not value or delta_minutes is None:
        return None
    try:
        hour, minute = str(value).split(":", 1)
        total = max(0, min(23 * 60 + 59, int(hour) * 60 + int(minute) + delta_minutes))
    except (TypeError, ValueError):
        return None
    return f"{total // 60:02d}:{total % 60:02d}"


def _build_lodging_plan(
    profile: TravelProfile,
    domain_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    """Choose one grounded lodging candidate as a recommendation, never a booking."""
    state = profile.constraint_state or {}
    explicit = bool(
        profile.hotel_area
        or any(
            state.get(key) not in (None, "", [], {})
            for key in (
                "lodging_area",
                "compare_lodging_areas",
                "hotel_budget_per_night_cny",
                "prepaid_lodging_cny",
                "lodging_flexibility",
            )
        )
    )
    nights = max(0, int(profile.days or 1) - 1)
    if not explicit and nights == 0:
        return None
    candidates: list[dict[str, Any]] = []
    source_artifact_ids: list[str] = []
    for entry in domain_inputs.get("hotels") or []:
        if not isinstance(entry, dict):
            continue
        payload = entry.get("payload") or {}
        source_artifact_ids.append(str(entry.get("artifact_id") or ""))
        candidates.extend(
            hotel for hotel in (payload.get("hotels") or []) if isinstance(hotel, dict)
        )
    grounded = [hotel for hotel in candidates if hotel.get("source") != "mock"]
    area_requirement = str(profile.hotel_area or state.get("lodging_area") or "").strip()
    if area_requirement:
        area_term = re.sub(r"(?:附近|核心|周边|商圈)$", "", area_requirement).strip()
        grounded = [
            hotel for hotel in grounded
            if area_term and area_term in " ".join([
                str(hotel.get("area") or ""),
                str(hotel.get("name") or ""),
                str(hotel.get("address") or ""),
            ])
        ]
    if not grounded:
        return {
            "required": explicit or nights > 0,
            "explicit_requirement": explicit,
            "status": "evidence_unavailable",
            "nights": nights,
            "area_requirement": area_requirement or None,
            "source_artifact_ids": [item for item in source_artifact_ids if item],
        }
    selected = sorted(
        grounded,
        key=lambda hotel: (
            float(hotel.get("price_per_night") or float("inf")),
            -float(hotel.get("rating") or 0),
            str(hotel.get("name") or ""),
        ),
    )[0]
    nightly = float(selected.get("price_per_night") or 0)
    people = int(
        profile.party_size
        or state.get("traveler_count")
        or _companions_to_count(profile.companions)
        or 1
    )
    rooms = max(1, (people + 1) // 2)
    return {
        "required": explicit or nights > 0,
        "explicit_requirement": explicit,
        "status": "recommended_not_booked",
        "selection_basis": "lowest_price_then_rating",
        "hotel": dict(selected),
        "nights": nights,
        "rooms": rooms,
        "nightly_price_cny": nightly,
        "lodging_subtotal_cny": round(nightly * nights * rooms, 2),
        "source_artifact_ids": [item for item in source_artifact_ids if item],
    }


def _build_budget_plan(
    profile: TravelProfile,
    domain_inputs: dict[str, Any],
    lodging_plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Turn the budget artifact into a user-limit comparison with traceable inputs."""
    entries = [entry for entry in (domain_inputs.get("budgets") or []) if isinstance(entry, dict)]
    if not entries:
        return None
    entry = entries[-1]
    estimate = dict(entry.get("payload") or {})
    base_hotel = float(estimate.get("hotel") or 0)
    selected_hotel = (
        float(lodging_plan.get("lodging_subtotal_cny") or 0)
        if isinstance(lodging_plan, dict) and lodging_plan.get("status") == "recommended_not_booked"
        else base_hotel
    )
    base_low = float(estimate.get("total_low") or 0)
    base_high = float(estimate.get("total_high") or 0)
    delta = selected_hotel - base_hotel
    total_low = round(max(0.0, base_low + delta), 2)
    total_high = round(max(total_low, base_high + delta), 2)
    state = profile.constraint_state or {}
    people = int(
        estimate.get("companions")
        or profile.party_size
        or state.get("traveler_count")
        or 1
    )
    if state.get("budget_max_cny") is not None:
        limit = float(state["budget_max_cny"])
        limit_basis = "constraint_total_budget"
    elif profile.budget_limit is not None:
        limit = float(profile.budget_limit)
        limit_basis = "total_budget"
    elif state.get("budget_per_person_cny") is not None:
        limit = float(state["budget_per_person_cny"]) * people
        limit_basis = "per_person_times_people"
    else:
        limit = None
        limit_basis = None
    breakdown = {
        "lodging": selected_hotel,
        "meals": float(estimate.get("meals") or 0),
        "tickets": float(estimate.get("tickets") or 0),
        "inner_city_transport": float(estimate.get("inner_city_transport") or 0),
    }
    expected_total = round(sum(breakdown.values()), 2)
    status = "estimate"
    note = "估算不含未提供价格的城际票、实时房价波动及个人购物。"
    if limit is not None and expected_total > limit and total_low <= limit:
        # The provider already supplies an estimate interval.  When the normal
        # scenario misses a hard cap but its grounded low scenario fits, expose
        # that scenario explicitly instead of either hiding the overrun or
        # inventing cheaper individual prices.
        variable_total = sum(value for key, value in breakdown.items() if key != "lodging")
        variable_target = max(0.0, total_low - selected_hotel)
        scale = min(1.0, variable_target / variable_total) if variable_total else 0.0
        for key in ("meals", "tickets", "inner_city_transport"):
            breakdown[key] = round(breakdown[key] * scale, 2)
        expected_total = round(sum(breakdown.values()), 2)
        status = "budget_optimized_low_scenario"
        note = (
            "硬预算下采用工具估算区间的低位方案；餐饮、门票与市内交通需按预算卡"
            "分项上限执行，实时价格上涨时应删减可选项目。"
        )
    return {
        "status": status,
        "currency": "CNY",
        "people": people,
        "days": int(estimate.get("days") or profile.days or 1),
        "breakdown_cny": breakdown,
        "total_low_cny": total_low,
        "total_expected_cny": expected_total,
        "total_high_cny": total_high,
        "user_limit_cny": limit,
        "user_limit_basis": limit_basis,
        "within_user_limit": expected_total <= limit if limit is not None else None,
        "risk_high_exceeds_limit": total_high > limit if limit is not None else None,
        "source_artifact_id": entry.get("artifact_id"),
        "note": note,
    }


def _build_mobility_plan(
    profile: TravelProfile,
    itinerary: dict[str, Any],
) -> dict[str, Any] | None:
    """Represent a walking cap without pretending unknown transit walks are zero."""
    state = profile.constraint_state or {}
    cap = state.get("max_walking_km_per_day")
    if cap is None:
        return None
    days: list[dict[str, Any]] = []
    for day in itinerary.get("days") or []:
        known = 0.0
        unknown_legs: list[str] = []
        for stop in day.get("stops") or []:
            route = stop.get("route_from_previous") or {}
            if not route:
                continue
            distance = route.get("walking_distance_km")
            if distance is None:
                unknown_legs.append(
                    f"{route.get('origin_name') or '上一站'}→{route.get('destination_name') or stop.get('name') or '下一站'}"
                )
            else:
                known += float(distance or 0)
        days.append({
            "day_index": day.get("day_index"),
            "known_walking_km": round(known, 2),
            "unknown_walking_legs": unknown_legs,
            "taxi_fallback_required": bool(unknown_legs) or known > float(cap),
        })
    return {
        "required": True,
        "status": "bounded_with_taxi_fallback",
        "max_walking_km_per_day": float(cap),
        "days": days,
        "policy": (
            "已知步行距离计入每日上限；公交接驳步行距离未知或累计可能超限的区段，"
            "必须改用点到点出租车/网约车，并在出发前用地图复核。"
        ),
    }


def _build_return_plan(
    profile: TravelProfile,
    itinerary: dict[str, Any],
    domain_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a truthful cross-city return commitment without inventing inventory."""
    state = profile.constraint_state or {}
    deadline = str(state.get("return_deadline") or "").strip()
    destination = str(state.get("return_location") or "").strip()
    trip_city = str(profile.destination or "").strip()
    if not deadline or not destination or (trip_city and trip_city in destination):
        return None
    try:
        hour, minute = deadline.split(":", 1)
        deadline_minutes = int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return None
    cutoff_minutes = max(0, deadline_minutes - 180)
    transport = domain_inputs.get("transport") or []
    routes = [
        entry.get("payload", entry)
        for entry in transport
        if isinstance(entry, dict)
    ]
    # Prefer a terminal transfer whose origin is actually scheduled.
    scheduled_ids = {
        str(stop.get("poi", {}).get("poi_id") or "")
        for day in itinerary.get("days") or []
        for stop in day.get("stops") or []
        if isinstance(stop, dict)
    }
    terminal_route = next(
        (
            route
            for route in routes
            if isinstance(route, dict)
            and str(route.get("origin_poi_id") or "") in scheduled_ids
            and any(
                marker in str(route.get("destination_name") or "")
                for marker in ("站", "机场", "码头")
            )
        ),
        None,
    )
    return_route = next(
        (
            route
            for route in routes
            if isinstance(route, dict)
            and destination in str(route.get("destination_name") or "")
            and destination not in str(route.get("origin_name") or "")
        ),
        None,
    )
    return {
        "required": True,
        "from_city": trip_city,
        "to_location": destination,
        "arrival_deadline": deadline,
        "activity_cutoff": f"{cutoff_minutes // 60:02d}:{cutoff_minutes % 60:02d}",
        "mode": "public_transport",
        "terminal_transfer": terminal_route,
        "intercity_segment": {
            "status": "route_estimate_only" if return_route else "requires_live_verification",
            "route_evidence": return_route,
            "instruction": (
                f"选择可在 {deadline} 前抵达{destination}的城际班次，"
                "并在购票平台核验实时班次、余票和检票时间。"
            ),
        },
        "buffer_min": 180,
    }


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
    reviewer_issue_types = {
        str(item).strip().lower()
        for item in (directives.get("reviewer_issue_types") or [])
        if str(item).strip()
    }
    if not indoor_days and not reviewer_issue_types:
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
                # The user explicitly asked for an indoor day. An honest
                # shorter indoor schedule is better than silently retaining an
                # outdoor stop when the bound evidence has no replacement.
                changed = True
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
    if not changed and not reviewer_issue_types:
        return result
    for day_index, day in enumerate(result.itinerary.days):
        days.append(
            replace(
                day,
                stops=day_stops[day_index],
                theme="室内活动" if day.day_index in indoor_days else day.theme,
            )
        )
    itinerary = replace(result.itinerary, days=days) if days else result.itinerary
    critic_result = critique_itinerary(itinerary, ctx.profile)
    return replace(
        result,
        itinerary=itinerary,
        critic_result=critic_result,
        revision_notes=list(result.revision_notes)
        + (["按要求将指定日期调整为室内活动"] if changed else []),
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
    """把明确绑定的领域结果写入计划，且不递归嵌入旧计划祖先树。"""
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
        payload = record["payload"]
        if record["kind"] == "itinerary":
            # A previous itinerary already embeds its own domain_inputs.  Copying
            # that payload verbatim into the next plan makes revision histories
            # grow exponentially (multi-turn cases reached several GB).  The
            # current planner has already recovered POIs from the full bound
            # artifact in _ranked_from_previous_itinerary; provenance here only
            # needs a bounded snapshot of the previous delivered plan.
            payload = _bounded_previous_itinerary_payload(payload)
        inputs[target].append(
            {
                "artifact_id": record["artifact_id"],
                "payload": payload,
            }
        )
    return inputs


def _bounded_previous_itinerary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    itinerary = payload.get("itinerary")
    compact: dict[str, Any] = {
        "itinerary": copy.deepcopy(itinerary) if isinstance(itinerary, dict) else {},
    }
    critic = payload.get("critic")
    if isinstance(critic, dict):
        compact["critic"] = {
            "passed": critic.get("passed"),
            "issues": list(critic.get("issues") or [])[:20],
        }
    for key in (
        "revision_notes",
        "candidate_verification",
        "return_plan",
        "lodging_plan",
        "budget_plan",
        "fixed_event_plan",
        "mobility_plan",
    ):
        value = payload.get(key)
        if value not in (None, [], {}):
            compact[key] = copy.deepcopy(value)
    return compact


def _merge_ranked_planner_inputs(
    ranked_artifact: dict[str, Any],
    records: list[dict[str, Any]],
    profile: TravelProfile,
) -> list[ScoredPOI]:
    """Merge every explicitly bound activity source before deterministic planning.

    A repair wave may create a newer ``ranked`` artifact from attraction-only
    candidates while the restaurant artifact remains separately bound to the
    Planner.  Consuming only that newest ranked payload silently dropped meal
    evidence even though the domain handoff was complete.
    """
    restaurant_ids: set[str] = set()
    for record in records:
        if record["kind"] != "restaurants":
            continue
        payload = record["payload"]
        for raw in payload.get("restaurants") or payload.get("items") or []:
            if isinstance(raw, dict) and raw.get("poi_id"):
                restaurant_ids.add(str(raw["poi_id"]))

    pois: list[POI] = []
    for raw in ranked_artifact.get("pois") or []:
        try:
            poi = _scored_from_dict(raw).poi
        except Exception:
            continue
        if poi.category == "food" and poi.poi_id not in restaurant_ids:
            continue
        if any(term in poi.name or poi.name in term for term in profile.avoid):
            continue
        pois.append(poi)
    for record in records:
        if record["kind"] not in {"candidates", "restaurants"}:
            continue
        payload = record["payload"]
        for raw in payload.get("pois") or payload.get("restaurants") or []:
            try:
                poi = poi_from_dict(raw)
            except Exception:
                continue
            if poi.category == "food" and poi.poi_id not in restaurant_ids:
                continue
            if poi.category in {"hotel", "transport"} or _is_unavailable_or_infrastructure_poi(
                poi, profile
            ):
                continue
            if any(term in poi.name or poi.name in term for term in profile.avoid):
                continue
            if _poi_city_allowed_for_plan(poi, profile):
                pois.append(poi)
    return score_pois(_dedupe_plannable_entities(pois, profile), profile)


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
    budget_plan = payload.get("budget_plan") or {}
    budget_exceeded = (
        isinstance(budget_plan, dict)
        and budget_plan.get("user_limit_cny") is not None
        and budget_plan.get("within_user_limit") is False
    )
    lodging_plan = payload.get("lodging_plan") or {}
    lodging_unverified = bool(
        isinstance(lodging_plan, dict)
        and lodging_plan.get("explicit_requirement")
        and lodging_plan.get("status") == "evidence_unavailable"
    )
    incomplete = mark_incomplete or not critic_passed or budget_exceeded or lodging_unverified
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
        budget_exceeded=budget_exceeded,
        lodging_unverified=lodging_unverified,
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
    budget_plan = payload.get("budget_plan") or {}
    budget_exceeded = (
        isinstance(budget_plan, dict)
        and budget_plan.get("user_limit_cny") is not None
        and budget_plan.get("within_user_limit") is False
    )
    lodging_plan = payload.get("lodging_plan") or {}
    lodging_unverified = bool(
        isinstance(lodging_plan, dict)
        and lodging_plan.get("explicit_requirement")
        and lodging_plan.get("status") == "evidence_unavailable"
    )
    incomplete = mark_incomplete or not critic_passed or budget_exceeded or lodging_unverified
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


_UNAVAILABLE_NAME_MARKERS = (
    "暂停开放",
    "暂不开放",
    "停止开放",
    "永久关闭",
    "已关闭",
    "暂停营业",
    "停业",
)
_INFRASTRUCTURE_NAME_MARKERS = (
    "上客点",
    "下客点",
    "乘车点",
    "乘车服务点",
    "直通车",
    "检票处",
    "售票处",
    "停车场",
    "出入口",
    "游客中心",
    "旅游广场",
    "纪念品",
    "枢纽",
    "换乘中心",
)
_NON_VISITABLE_NAME_SUFFIXES = (
    "有限公司",
    "有限责任公司",
    "分公司",
    "办事处",
    "服务门店",
)


def _is_unavailable_or_infrastructure_poi(
    poi: POI,
    profile: TravelProfile | None = None,
) -> bool:
    name = str(poi.name or "")
    return (
        any(marker in name for marker in _UNAVAILABLE_NAME_MARKERS)
        or any(marker in name for marker in _INFRASTRUCTURE_NAME_MARKERS)
        or any(name.endswith(suffix) for suffix in _NON_VISITABLE_NAME_SUFFIXES)
        or _is_closed_on_requested_weekday(poi, profile)
        or _is_outside_explicit_opening_season(poi, profile)
    )


def _is_closed_on_requested_weekday(
    poi: POI,
    profile: TravelProfile | None,
) -> bool:
    if profile is None:
        return False
    weekday = str((profile.constraint_state or {}).get("weekday") or "").strip()
    if weekday not in {"周一", "周二", "周三", "周四", "周五", "周六", "周日"}:
        return False
    opening = re.sub(r"\s+", "", str(poi.opening_hours or ""))
    if not opening:
        return False
    return bool(
        re.search(
            rf"(?:每)?{re.escape(weekday)}.{{0,12}}(?:全天)?"
            r"(?:不开放|闭馆|休息|暂停营业|停止开放)",
            opening,
        )
    )


def _is_outside_explicit_opening_season(
    poi: POI,
    profile: TravelProfile | None,
) -> bool:
    """Reject a POI only when its evidence gives explicit seasonal ranges.

    Unknown or generic opening hours remain eligible.  Multiple ranges are
    treated as alternatives, including ranges that cross the year boundary.
    """
    if profile is None or not profile.start_date:
        return False
    try:
        requested = date.fromisoformat(str(profile.start_date))
    except ValueError:
        return False
    opening = re.sub(r"\s+", "", str(poi.opening_hours or ""))
    ranges = re.findall(
        r"(\d{1,2})月(?:(\d{1,2})日)?(?:至|到|-)(?:次年)?"
        r"(\d{1,2})月(?:(\d{1,2})日)?",
        opening,
    )
    if not ranges:
        return False
    month_day = (requested.month, requested.day)
    for start_month, start_day, end_month, end_day in ranges:
        start = (int(start_month), int(start_day or 1))
        end = (int(end_month), int(end_day or 31))
        if start <= end:
            if start <= month_day <= end:
                return False
        elif month_day >= start or month_day <= end:
            return False
    return True


def _dedupe_plannable_entities(pois: list[POI], profile: TravelProfile) -> list[POI]:
    """Collapse entrances, halls and shops belonging to the same named venue."""
    import re

    required = list(profile.must_visit or [])

    def priority(poi: POI) -> tuple[int, int, int, float]:
        exact_required = any(poi.name == term for term in required)
        canonical_venue = any(
            marker in poi.name for marker in ("博物馆", "景区", "风景名胜区")
        )
        return (
            0 if exact_required else 1,
            0 if canonical_venue else 1,
            len(poi.name),
            -poi.rating,
        )

    def normalized(poi: POI) -> str:
        name = str(poi.name or "").lower()
        if "兵马俑" in name:
            return "兵马俑"
        if "城墙" in name:
            return "城墙"
        city = str(poi.city or "").lower().removesuffix("市")
        if city and name.startswith(city) and len(name) > len(city) + 2:
            name = name[len(city) :]
        name = re.sub(r"[（(][^）)]*[）)]", "", name)
        name = re.sub(r"(?:官方)?(?:文创店|陈列馆|青铜馆|展览馆|检票处|售票处)$", "", name)
        return re.sub(r"[\s·.\-—_/]", "", name)

    kept: list[POI] = []
    keys: list[tuple[str, str]] = []
    for poi in sorted({poi.poi_id: poi for poi in pois}.values(), key=priority):
        key = normalized(poi)
        if poi.category != "food" and any(
            category != "food"
            and (
                key == existing
                or (
                    min(len(key), len(existing)) >= 4
                    and (key in existing or existing in key)
                )
            )
            for existing, category in keys
        ):
            continue
        kept.append(poi)
        keys.append((key, poi.category))
    return kept


def _poi_city_allowed_for_plan(poi: POI, profile: TravelProfile) -> bool:
    """Keep cross-city POIs only when the user explicitly requires the venue/city."""
    from travel_agent.critic import poi_matches_must_visit

    def city_key(value: Any) -> str:
        text = str(value or "").strip()
        return text[:-1] if text.endswith("市") and len(text) > 1 else text

    destination = city_key(profile.destination)
    city = city_key(poi.city)
    if not destination or not city or destination == city:
        return True
    allowed_cities = {
        city_key(item)
        for item in ((profile.constraint_state or {}).get("destinations") or [])
        if item
    }
    if city in allowed_cities:
        return True
    return any(
        poi_matches_must_visit(poi, term) for term in (profile.must_visit or [])
    )


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


def _hotels_from_pois(pois: list[POI], area: str | None, budget_level: str) -> list[dict[str, Any]]:
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
                "address": poi.address,
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
    # Replies and review traces are historical snapshots.  Returning the live
    # list/dict objects lets a later turn retroactively rewrite an earlier
    # turn's profile in the evaluation record.
    return copy.deepcopy({
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
        "constraint_state": dict(profile.constraint_state),
    })


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
