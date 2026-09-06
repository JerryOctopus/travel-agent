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
import math
import re
from datetime import date, datetime, time, timedelta
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
from travel_agent.agent.tool_contract import contracted_tool
from travel_agent.constraints import ConstraintSet
from travel_agent.planning_subgraph import plan_and_critique as run_plan_and_critique
from travel_agent.planning import TRANSFER_BUFFER_MIN, rebind_itinerary_routes
from travel_agent.providers import ProviderRateLimitError
from travel_agent.poi_evidence import (
    canonical_entity_match_evidence,
    canonical_identity_key,
    is_verified_plannable_poi,
    normalize_candidate_requirement,
    normalize_entity_name,
    normalize_poi_entity,
    partition_verified_candidates,
    poi_avoid_match,
    poi_covers_requirement,
)
from travel_agent.recommendation import score_pois
from travel_agent.route_evidence import (
    LODGING_ACTIVITY_AREA_CONFLICT_KM,
    canonical_route_evidence_status,
    evidence_status_for_source,
    normalize_route_evidence,
    route_supports_endpoints,
    route_evidence_reason_codes,
)
from travel_agent.schemas import CriticIssue, CriticResult, POI, RouteInfo, ScoredPOI, TravelProfile
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


def _filter_directional_area_pois(
    provider: Any,
    city: str,
    area: str,
    hotels: list[POI],
) -> list[POI] | None:
    """Validate ``landmark + direction`` areas against provider coordinates.

    Text-search relevance alone is insufficient for phrases such as "湖东侧":
    provider results often contain an unrelated "东侧" in the street address.
    ``None`` means that no landmark anchor could be verified and the caller may
    retain its normal evidence policy; an empty list means the direction was
    verified and none of the hotel candidates satisfies it.
    """
    match = re.fullmatch(r"(.+?)(东侧|西侧|南侧|北侧)(?:附近|周边)?", area.strip())
    if match is None:
        return None
    landmark_name, direction = match.groups()
    try:
        landmarks = provider.search_pois(
            city=city,
            query_tags=[landmark_name],
            # An untyped exact-name search preserves the provider's primary
            # landmark result. Adding the generic scenic type causes AMap to
            # broaden "湖" into unrelated city-wide attractions.
            category=None,
            max_results=10,
        )
    except Exception:
        return None
    matching = [
        poi
        for poi in landmarks
        if isinstance(poi, POI)
        and poi.verification_status == "verified"
        and poi_covers_requirement(poi, landmark_name)
    ]
    if not matching:
        return None
    required = normalize_entity_name(landmark_name)
    anchor = min(
        matching,
        key=lambda poi: (
            0 if required in {
                normalize_entity_name(poi.name),
                normalize_entity_name(poi.canonical_name),
                *(normalize_entity_name(alias) for alias in poi.aliases),
            } else 1,
            len(normalize_entity_name(poi.name)),
            -float(poi.rating or 0),
        ),
    )
    tolerance = 0.001
    predicates = {
        "东侧": lambda poi: poi.lng >= anchor.lng + tolerance,
        "西侧": lambda poi: poi.lng <= anchor.lng - tolerance,
        "南侧": lambda poi: poi.lat <= anchor.lat - tolerance,
        "北侧": lambda poi: poi.lat >= anchor.lat + tolerance,
    }
    return [poi for poi in hotels if predicates[direction](poi)]


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


@contracted_tool("update_travel_profile")
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


@contracted_tool("request_travel_info")
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


@contracted_tool("request_preference_guide")
def request_preference_guide(ctx: SessionContext, l3: TravelProfile | None = None) -> dict[str, Any]:
    """必要信息齐全后，引导用户补充兴趣、节奏等偏好（用户可说「随便」跳过）。"""
    from travel_agent.agent.preferences import build_preference_guide_text

    question = build_preference_guide_text(ctx.profile, l3)
    return _ok(question, question=question, awaiting_preferences=True)


# --------------------------------------------------------------------------- #
# 真实工具：POI / 天气 / 路线
# --------------------------------------------------------------------------- #
def _execute_intent_retrieval_bridge(
    ctx: SessionContext,
    *,
    city: str,
    existing_tags: list[str],
    category: str | None,
    max_results: int,
) -> tuple[list[POI], dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    """Execute one request-scoped, bounded taxonomy retrieval plan."""
    settings = getattr(ctx, "runtime_settings", None)
    hybrid = getattr(settings, "hybrid_planning", None)
    if not bool(getattr(hybrid, "enable_llm_intent_normalizer", False)):
        return [], None, {}
    if str(ctx.active_task_type or "") not in {"full_itinerary", "full_trip_plan"}:
        return [], None, {}
    state = ctx.profile.constraint_state or {}
    normalized_context = state.get("normalized_interest_context")
    if not isinstance(normalized_context, dict) or not normalized_context.get("normalized_interests"):
        return [], None, {}

    from travel_agent.artifact_policy import constraint_version
    from travel_agent.hybrid_planning.soft_preference_actuation import (
        build_retrieval_plan,
        retrieval_provenance_for_pois,
        retrieval_queries,
    )

    meta = current_task_meta()
    request_id = str(meta.get("request_id") or _hybrid_request_scope(ctx))
    turn_id = str(meta.get("turn_id") or _hybrid_source_turn(ctx.profile) or "")
    version = constraint_version(ctx.profile)
    plan = build_retrieval_plan(
        normalized_context,
        destination=city,
        request_id=request_id,
        turn_id=turn_id,
        constraint_revision=version["revision"],
        constraint_hash=version["constraint_hash"],
    )
    queries = retrieval_queries(plan)
    if not queries:
        return [], None, {}
    fingerprint = str(plan["fingerprint"])
    reservation, cached = ctx.hybrid_call_cache.reserve(
        request_id, "soft_preference_retrieval", fingerprint
    )
    if reservation == "in_progress":
        ready, cached = ctx.hybrid_call_cache.wait(cached, timeout_seconds=60.0)
        reservation = "cache_hit" if ready else "dedup_wait_timeout"
    if reservation == "cache_hit" and isinstance(cached, dict):
        result = copy.deepcopy(cached)
        plan.update({
            "execution_status": "cache_hit",
            "retrieval_misses": list(result.get("retrieval_misses") or []),
        })
        return list(result.get("pois") or []), plan, dict(result.get("retrieval_provenance") or {})
    if reservation == "dedup_wait_timeout":
        plan.update({
            "execution_status": "dedup_wait_timeout",
            "retrieval_misses": [
                {**query, "reason": "dedup_wait_timeout"} for query in queries
            ],
        })
        return [], plan, {}

    existing = {
        re.sub(r"[^\w\u3400-\u9fff]+", "", str(tag).casefold())
        for tag in existing_tags if str(tag).strip()
    }
    found: list[POI] = []
    provenance: dict[str, list[dict[str, Any]]] = {}
    misses: list[dict[str, Any]] = []
    for query in queries:
        normalized_query = re.sub(
            r"[^\w\u3400-\u9fff]+", "", str(query.get("query") or "").casefold()
        )
        if normalized_query in existing:
            continue
        try:
            items = ctx.provider.search_pois(
                city=city,
                query_tags=[str(query["query"])],
                category=category,
                max_results=min(max(1, max_results), 12),
            )
        except ProviderRateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded as an explicit miss
            misses.append({**query, "reason": f"provider_error:{type(exc).__name__}"})
            continue
        if not items:
            misses.append({**query, "reason": "no_provider_results"})
            continue
        normalized_items = [normalize_poi_entity(item, ctx.profile) for item in items]
        found.extend(normalized_items)
        for candidate_id, entries in retrieval_provenance_for_pois(query, normalized_items).items():
            provenance.setdefault(candidate_id, []).extend(entries)
    result = {
        "pois": _merge_pois(found, []),
        "retrieval_provenance": provenance,
        "retrieval_misses": misses,
    }
    ctx.hybrid_call_cache.publish(
        request_id, "soft_preference_retrieval", fingerprint, copy.deepcopy(result)
    )
    plan.update({"execution_status": "executed", "retrieval_misses": misses})
    return result["pois"], plan, provenance


@contracted_tool("search_poi")
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

    from travel_agent.tool_recovery import recover_search

    recovery = recover_search(
        ctx.provider,
        city=target_city,
        query_tags=list(tags or []),
        category=category,
        max_results=max_results,
        hard_constraints=ctx.profile.constraint_state,
    )
    preference = recovery.items
    supplemental, retrieval_plan, retrieval_provenance = _execute_intent_retrieval_bridge(
        ctx,
        city=target_city,
        existing_tags=list(tags or []),
        category=category,
        max_results=max_results,
    )
    preference = _merge_pois(preference, supplemental)
    citywide: list[POI] = []
    # A successful preference query can be narrow. One distinct city-supply
    # query preserves candidate diversity without repeating the same intent.
    # Twenty-five candidates already fill one complete live-provider page and
    # are sufficient for downstream ranking, even when callers request 30.
    supply_target = min(max(1, int(max_results)), 25)
    if tags and len(preference) < supply_target:
        try:
            citywide = ctx.provider.search_pois(
                city=target_city,
                query_tags=None,
                category=category,
                max_results=max_results,
            )
        except ProviderRateLimitError:
            raise
        except Exception as exc:  # noqa: BLE001
            failures = [f"city_supply:{type(exc).__name__}"]
        else:
            failures = []
    else:
        failures = []
    failures.extend([
        f"{item.stage}:{item.outcome}:{item.provider}"
        for item in recovery.attempts
        if item.outcome != "success"
    ])
    # 不会重复 POI
    # 偏好结果优先排在前面，citywide 结果只做补全
    from travel_agent.critic import poi_matches_interest

    candidates = [
        normalize_poi_entity(poi, ctx.profile)
        for poi in _merge_pois(preference, citywide)
    ]
    rejected: list[POI] = []
    if category not in {"food", "hotel"}:
        requested_food = [
            poi for poi in candidates
            if poi.category == "food" and any(
                poi_matches_interest(poi, str(item))
                for item in ((ctx.profile.constraint_state or {}).get("interests") or [])
            )
        ] if current_task_meta().get("agent") == "attraction" else []
        candidates, rejected = partition_verified_candidates(
            [poi for poi in candidates if poi.category != "food"], ctx.profile
        )
        candidates.extend(requested_food)
    if current_task_meta().get("agent") == "attraction":
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
    # Multi-day plans need enough independently verified activity entities to
    # populate every day.  A preference query plus a generic city page can be
    # taxonomically narrow, so broaden by provider category only when the
    # verified supply is still sparse.  This remains generic and the added
    # entities pass the same evidence and infrastructure filters.
    days = int(ctx.profile.days or 1)
    activity_supply_target = min(max_results, max(4, days * 2))
    distinct_activities = [
        poi for poi in _dedupe_plannable_entities(candidates, ctx.profile)
        if poi.category not in {"food", "hotel", "transport"}
    ]
    activity_categories = {poi.category for poi in distinct_activities}
    if (
        category is None
        and days >= 3
        and (
            len(distinct_activities) < activity_supply_target
            or len(activity_categories) < 2
        )
    ):
        diversified: list[POI] = []
        for supply_category in ("scenic", "museum"):
            try:
                supplied = ctx.provider.search_pois(
                    city=target_city,
                    query_tags=None,
                    category=supply_category,
                    max_results=max_results,
                )
            except ProviderRateLimitError:
                raise
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    f"category_supply_{supply_category}:{type(exc).__name__}"
                )
                continue
            normalized = [normalize_poi_entity(poi, ctx.profile) for poi in supplied]
            verified, supply_rejected = partition_verified_candidates(
                normalized, ctx.profile
            )
            rejected.extend(supply_rejected)
            diversified.extend(
                poi for poi in verified
                if poi.category not in {"food", "hotel", "transport"}
                and not _is_unavailable_or_infrastructure_poi(poi, ctx.profile)
            )
        candidates = _merge_pois(candidates, diversified)[:max_results]
    fault_events = _consume_provider_faults(ctx, "search_pois")
    if fault_events:
        failures.append("检索服务返回异常，已重试并补全候选")
    endpoint_pois = [
        poi for poi in rejected
        if poi.entity_type in {"transport", "hotel", "parking", "visitor_center", "retail"}
    ]
    if not candidates and not endpoint_pois:
        return _err(
            f"没有检索到「{target_city}」的可靠 POI；恢复链已完成，证据仍不足。",
            city=target_city,
            failure_kind=recovery.failure_kind,
            recovery_attempts=[item.__dict__ for item in recovery.attempts],
            unmet_constraints=recovery.unmet_constraints,
        )

    ctx.remember_pois([*candidates, *endpoint_pois])
    if failures:
        ctx.store.put(
            "recovery",
            {
                "operation": "search_poi",
                "reason": "；".join(failures),
                "suggestion": "建议出发前在地图或景区官方渠道复核开放时间与预约要求。",
            },
        )
    source = (candidates or endpoint_pois)[0].source
    candidate_payload = {
            "city": target_city,
            "query_tags": list(tags or []),
            "category": category,
            "pois": [poi_to_dict(p) for p in candidates],
            "rejected_pois": [poi_to_dict(p) for p in rejected],
            "endpoint_pois": [poi_to_dict(p) for p in endpoint_pois],
            "recovery_attempts": [item.__dict__ for item in recovery.attempts],
            "hard_constraints_preserved": recovery.unmet_constraints,
        }
    if retrieval_plan is not None:
        candidate_payload["retrieval_plan"] = retrieval_plan
        candidate_payload["retrieval_provenance"] = {
            candidate_id: entries
            for candidate_id, entries in retrieval_provenance.items()
            if any(poi.poi_id == candidate_id for poi in candidates)
        }
    artifact_id = ctx.store.put("candidates", candidate_payload)
    return _ok(
        f"在「{target_city}」检索到 {len(candidates)} 个候选 POI（来源={source}）。",
        artifact_id=artifact_id,
        city=target_city,
        count=len(candidates),
        pois=[poi_brief(p) for p in candidates[:12]],
        endpoint_pois=[poi_brief(p) for p in endpoint_pois[:12]],
        fallback_reason="；".join(failures) if failures else None,
        recovery_state="recovered" if recovery.recovered else "not_needed",
    )


@contracted_tool("check_weather")
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


@contracted_tool("plan_route")
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
    route = normalize_route_evidence(ctx.provider.estimate_route(origin, destination, transport))
    if not route_supports_endpoints(route, origin.poi_id, destination.poi_id):
        return _err(
            "路线工具返回的端点与请求不一致，证据已拒绝。",
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            evidence_status="unavailable",
        )
    route_payload = {
        "origin_poi_id": origin.poi_id,
        "destination_poi_id": destination.poi_id,
        "origin_name": origin.name,
        "destination_name": destination.name,
        "distance_km": route.distance_km,
        "duration_min": route.duration_min,
        "mode": route.mode,
        "source": route.source,
        "evidence_status": route.evidence_status,
        "estimated_cost": _estimate_transport_cost(route.distance_km, route.mode),
        "walking_distance_km": route.walking_distance_km,
    }
    conflict_codes = route_evidence_reason_codes(route_payload)
    route_payload["evidence_status"] = canonical_route_evidence_status(route_payload)
    if conflict_codes:
        return _err(
            "路线证据状态冲突，已按不可用 fail-closed。",
            error_code=conflict_codes[0],
            reason_codes=conflict_codes,
            evidence_status="unavailable",
        )
    artifact_id = ctx.store.put("routes", route_payload)
    return _ok(
        f"{origin.name} → {destination.name}：约 {route.duration_min} 分钟 / "
        f"{route.distance_km} 公里（{route.mode}, {route.source}）。",
        artifact_id=artifact_id,
        **route_payload,
    )


@contracted_tool("build_constraints")
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


@contracted_tool("search_restaurant")
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
    provider_tags = [tag for tag in tags if tag]
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
    nearby_restaurants: list[POI] = []
    if area:
        nearby_search = getattr(ctx.provider, "search_pois_nearby", None)
        if callable(nearby_search):
            anchors = ctx.provider.search_pois(
                city=target_city,
                query_tags=[str(area)],
                category=None,
                max_results=5,
            )
            normalized_area = normalize_entity_name(area)
            anchor = next(
                (
                    poi
                    for poi in anchors
                    if normalized_area
                    and normalized_area in normalize_entity_name(poi.name)
                ),
                anchors[0] if anchors else None,
            )
            if anchor is not None:
                ctx.remember_pois([anchor])
                nearby_terms = (
                    ["清真餐厅"]
                    if halal_required
                    else [str(cuisine or "餐厅")]
                )
                nearby_restaurants = nearby_search(
                    city=target_city,
                    anchor=anchor,
                    query_tags=nearby_terms,
                    category="food",
                    radius_m=8000,
                    max_results=max_results * 2,
                )
    if area:
        provider_tags.extend([str(area), "餐厅"])
    text_restaurants = ctx.provider.search_pois(
        city=target_city,
        query_tags=provider_tags,
        category="food",
        max_results=max_results * 2,
    )
    restaurants = []
    seen_restaurant_ids: set[str] = set()
    for poi in [*nearby_restaurants, *text_restaurants]:
        if poi.poi_id in seen_restaurant_ids:
            continue
        seen_restaurant_ids.add(poi.poi_id)
        restaurants.append(poi)
    restaurants = [
        poi
        for poi in (normalize_poi_entity(item, ctx.profile) for item in restaurants)
        if poi.category == "food" and poi.entity_type == "restaurant"
    ]
    nearby_ids = {poi.poi_id for poi in nearby_restaurants}
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
        if (
            area
            and poi.poi_id not in nearby_ids
            and area not in poi.name
            and area not in (poi.address or "")
            and area not in " ".join(poi.tags)
        ):
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


@contracted_tool("search_hotel")
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
    state = ctx.profile.constraint_state or {}
    if (
        state.get("lodging_flexibility") == "can_downgrade"
        and state.get("hotel_budget_per_night_cny") is None
        and (
            state.get("budget_max_cny") is not None
            or state.get("budget_total_cny") is not None
            or ctx.profile.budget_limit is not None
        )
    ):
        target_budget = "low"
    initial_query_tags = (
        [target_area]
        if requested_area
        else ["市中心", "核心商圈"]
    )
    poi_hotels = ctx.provider.search_pois(
        city=target_city,
        query_tags=initial_query_tags,
        category="hotel",
        max_results=max_results * 2,
    )
    area_query_evidenced = bool(poi_hotels)
    if not poi_hotels and not requested_area:
        poi_hotels = ctx.provider.search_pois(
            city=target_city,
            query_tags=None,
            category="hotel",
            max_results=max_results * 2,
        )
        area_query_evidenced = False
    if not poi_hotels:
        # Some live providers apply their type code before tokenization and
        # return an empty set for "area + hotel" even though an untyped text
        # query returns taxonomy-backed hotel entities.  Broaden once, then
        # retain only provider-classified hotels.
        broad_terms = [f"{target_area} 酒店"] if requested_area else ["酒店"]
        poi_hotels = [
            poi
            for poi in ctx.provider.search_pois(
                city=target_city,
                query_tags=broad_terms,
                category=None,
                max_results=max_results * 2,
            )
            if poi.category == "hotel" and poi.verification_status == "verified"
        ]
        area_query_evidenced = bool(requested_area and poi_hotels)
    area_evidenced = bool(not requested_area and area_query_evidenced)
    if requested_area:
        area_term = re.sub(r"(?:附近|核心|周边|商圈)$", "", str(requested_area)).strip()
        matching_hotels = [
            poi
            for poi in poi_hotels
            if area_term and area_term in " ".join([poi.name, poi.address or ""])
        ]
        if user_area:
            directional = _filter_directional_area_pois(
                ctx.provider, target_city, str(requested_area), poi_hotels
            )
            if directional is not None:
                matching_hotels = directional
                area_query_evidenced = bool(directional)
                if not matching_hotels:
                    # The explicit text query returned hotels on the wrong
                    # side of a verified landmark. Broaden the hotel supply and
                    # apply the coordinate predicate again; never treat query
                    # relevance alone as proof of a directional area.
                    broad_hotels = ctx.provider.search_pois(
                        city=target_city,
                        query_tags=None,
                        category="hotel",
                        max_results=max_results * 2,
                    )
                    rechecked = _filter_directional_area_pois(
                        ctx.provider,
                        target_city,
                        str(requested_area),
                        broad_hotels,
                    )
                    matching_hotels = rechecked or []
                    area_query_evidenced = bool(matching_hotels)
            if not matching_hotels:
                if directional is not None:
                    matching_hotels = []
                elif area_query_evidenced:
                    # A provider result returned by the explicit area query is
                    # itself bounded area relevance evidence; hotel names and
                    # street addresses need not repeat phrases such as “东侧”.
                    matching_hotels = list(poi_hotels)
                else:
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
    nightly_limit = state.get("hotel_budget_per_night_cny")
    nightly_limit_value: float | None = None
    if nightly_limit is not None:
        try:
            nightly_limit_value = float(nightly_limit)
        except (TypeError, ValueError):
            nightly_limit_value = None
        if nightly_limit_value is not None:
            hotels = [
                hotel
                for hotel in hotels
                if float(hotel.get("price_per_night") or float("inf")) <= nightly_limit_value
            ]
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
        hotel_budget_per_night_cny=nightly_limit_value,
    )


@contracted_tool("estimate_budget")
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
def _hybrid_soft_preference_context(profile: TravelProfile) -> dict[str, Any]:
    state = profile.constraint_state or {}
    context: dict[str, Any] = {}
    if profile.interests:
        context["interests"] = list(profile.interests)
    if profile.pace != "standard":
        context["pace"] = profile.pace
    if profile.companions:
        context["companions"] = profile.companions
    if profile.budget_level:
        context["budget_level"] = profile.budget_level
    for key in (
        "preference", "mobility", "accessibility_priority", "soft_avoid_interests",
    ):
        if state.get(key) not in (None, "", [], {}):
            context[key] = state[key]
    return context


def _hybrid_request_scope(ctx: SessionContext) -> str:
    try:
        from travel_agent.orchestration.multi_agent.trace import current_trace

        trace = current_trace()
        if trace is not None:
            return str(trace.request_id)
    except Exception:
        pass
    return str(ctx.hybrid_request_scope or "standalone")


def _hybrid_source_turn(profile: TravelProfile) -> str | None:
    context = (profile.constraint_state or {}).get("normalized_interest_context") or {}
    for item in context.get("normalized_interests") or []:
        if isinstance(item, dict) and item.get("source_turn"):
            return str(item["source_turn"])
    return None


def _apply_candidate_order(
    ranked: list[ScoredPOI], candidate_order: list[str]
) -> list[ScoredPOI]:
    """Consume only a permutation/subset of already validated candidate IDs."""
    known = {item.poi.poi_id: item for item in ranked}
    if any(candidate_id not in known for candidate_id in candidate_order):
        return ranked
    ordered = [known[candidate_id] for candidate_id in candidate_order]
    used = set(candidate_order)
    ordered.extend(item for item in ranked if item.poi.poi_id not in used)
    return ordered


@contracted_tool("recommend_candidates")
def recommend_candidates(
    ctx: SessionContext,
    top_k: int = 12,
    artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """只按明确 artifact_id 对候选 POI/餐厅做重排。"""
    records, error = _planner_input_records(
        ctx,
        artifact_ids,
        allowed_kinds={"candidates", "restaurants", "routes"},
        legacy_kinds=("candidates", "restaurants", "routes"),
    )
    if error:
        return _err(error)
    collected: list[POI] = []
    source_ids: list[str] = []
    for record in records:
        if record["kind"] == "routes":
            # Route records feed only the evidence matrix; they do not own POIs
            # and must not change the ranked artifact's candidate source list.
            continue
        payload = record["payload"]
        raw_pois = payload.get("pois") or payload.get("restaurants") or []
        for raw_poi in raw_pois:
            try:
                poi = normalize_poi_entity(poi_from_dict(raw_poi), ctx.profile)
            except Exception:
                continue
            # Accommodation and transport infrastructure are evidence/context,
            # not day-activity candidates.  Letting them into the shared ranked
            # pool produced zero-duration hotel stops and bus stops masquerading
            # as museums.
            if (
                poi.category != "food"
                and not is_verified_plannable_poi(poi)
            ) or _is_unavailable_or_infrastructure_poi(poi, ctx.profile):
                continue
            if not _poi_city_allowed_for_plan(poi, ctx.profile):
                continue
            collected.append(poi)
        source_ids.append(record["artifact_id"])
    pois = _dedupe_plannable_entities(collected, ctx.profile)
    if not pois:
        return _err("明确指定的 artifact 中没有可规划 POI，请先调用 search_poi。")
    excluded_by_avoid: list[dict[str, str]] = []
    eligible: list[POI] = []
    for poi in pois:
        avoid_match = poi_avoid_match(poi, ctx.profile)
        if avoid_match is not None:
            excluded_by_avoid.append(avoid_match.to_audit_dict(poi))
            continue
        eligible.append(poi)
    pois = eligible
    if not pois:
        return _err(
            "明确指定的 artifact 中没有符合当前避开约束的可规划 POI。",
            excluded_by_avoid=excluded_by_avoid,
        )
    ranked = score_pois(pois, ctx.profile)
    planning_policy = None
    actuation_contract = None
    settings = getattr(ctx, "runtime_settings", None)
    hybrid = getattr(settings, "hybrid_planning", None)
    resolver_enabled = bool(
        getattr(hybrid, "enable_llm_preference_resolver", False)
    )
    client = getattr(ctx, "hybrid_llm_client", None)
    if resolver_enabled and client is None and settings is not None and settings.llm.enabled:
        from travel_agent.hybrid_planning.structured_client import (
            ProductionStructuredLLMClient,
        )

        client = ProductionStructuredLLMClient(settings, phase="preference_resolver")
    if settings is not None:
        from travel_agent.artifact_policy import constraint_version
        from travel_agent.hybrid_planning.preference_resolver import PreferenceResolver
        from travel_agent.hybrid_planning.soft_preference_actuation import (
            apply_preference_score,
            build_actuation_contract,
            build_candidate_evidence_matrix,
            hard_legal_ranked_from_matrix,
            preference_candidates_from_matrix,
        )

        meta = current_task_meta()
        active_version = constraint_version(ctx.profile)
        request_id = str(meta.get("request_id") or _hybrid_request_scope(ctx))
        turn_id = str(meta.get("turn_id") or _hybrid_source_turn(ctx.profile) or "")
        evidence_matrix = build_candidate_evidence_matrix(
            ranked,
            records,
            ctx.profile,
            request_id=request_id,
            turn_id=turn_id,
            constraint_revision=active_version["revision"],
            constraint_hash=active_version["constraint_hash"],
        )
        if resolver_enabled:
            ranked = hard_legal_ranked_from_matrix(ranked, evidence_matrix)
            if not ranked:
                return _err(
                    "候选均未通过候选级硬约束校验，无法进入偏好解析或规划。",
                    hard_filter=evidence_matrix.get("hard_filter") or {},
                )
        resolver_candidates = preference_candidates_from_matrix(evidence_matrix)

        policy = PreferenceResolver(
            client,
            enabled=resolver_enabled,
            cache=ctx.hybrid_call_cache,
            request_scope=_hybrid_request_scope(ctx),
        ).resolve(
            task_type=ctx.active_task_type or "clarification",
            candidates=resolver_candidates,
            soft_preferences=_hybrid_soft_preference_context(ctx.profile),
            hard_constraints={
                "budget_limit": ctx.profile.budget_limit,
                "must_visit": list(ctx.profile.must_visit),
                **dict(ctx.profile.constraint_state or {}),
            },
            source_turn=_hybrid_source_turn(ctx.profile),
            delivery_intent=ctx.active_delivery_intent,
        )
        if resolver_enabled and policy.source == "llm":
            planning_policy = policy.to_dict()
            base_ranking = [item.poi.poi_id for item in ranked]
            ranked, score_trace = apply_preference_score(ranked, evidence_matrix, policy)
            retrieval_plans = [
                record["payload"]["retrieval_plan"]
                for record in records
                if isinstance(record.get("payload"), dict)
                and isinstance(record["payload"].get("retrieval_plan"), dict)
            ]
            actuation_contract = build_actuation_contract(
                matrix=evidence_matrix,
                retrieval_plans=retrieval_plans,
                policy=policy,
                score_trace=score_trace,
                base_ranking=base_ranking,
                final_ranking=[item.poi.poi_id for item in ranked],
            )
    top = ranked[: max(1, top_k)]
    ranked_payload = {
        "pois": [_scored_to_dict(s) for s in ranked],
        "source_artifact_ids": source_ids,
        "excluded_by_avoid": excluded_by_avoid,
    }
    if planning_policy is not None:
        ranked_payload["planning_policy"] = planning_policy
    if actuation_contract is not None:
        ranked_payload["soft_preference_actuation"] = actuation_contract
    artifact_id = ctx.store.put(
        "ranked",
        ranked_payload,
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


@contracted_tool("plan_and_critique")
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
    ranked_record = next(
        (record for record in reversed(records) if record["kind"] == "ranked"),
        None,
    )
    ranked_artifact = ranked_record["payload"] if ranked_record else None
    if not ranked_artifact:
        ranked_artifact = _ranked_from_previous_itinerary(records, ctx)
    if not ranked_artifact:
        return _err("明确指定的 artifact 中没有排序候选，请先调用 recommend_candidates。")
    if not ctx.profile.destination or not ctx.profile.days:
        return _err("缺少 destination 或 days，无法规划，请先补全画像。")

    domain_inputs = _collect_domain_inputs(records)
    ranked = _merge_ranked_planner_inputs(ranked_artifact, records, ctx.profile)
    ranked = _prioritize_ranked_near_lodging(
        ranked,
        _build_lodging_plan(ctx.profile, domain_inputs),
        ctx.profile,
    )
    source_actuation = ranked_artifact.get("soft_preference_actuation")
    planner_actuation = source_actuation
    duration_estimates: dict[str, dict[str, Any]] = {}
    settings = getattr(ctx, "runtime_settings", None)
    hybrid = getattr(settings, "hybrid_planning", None)
    duration_enabled = bool(
        getattr(hybrid, "enable_structured_duration_estimator", False)
    )
    if duration_enabled:
        from travel_agent.hybrid_planning.duration_estimator import (
            DeterministicDurationEstimator,
        )

        ranked, duration_estimates = DeterministicDurationEstimator().apply_to_ranked(
            ranked, ctx.profile
        )
    if isinstance(planner_actuation, dict):
        from travel_agent.hybrid_planning.soft_preference_actuation import (
            with_planner_input_fingerprint,
        )

        planner_actuation = with_planner_input_fingerprint(planner_actuation, ranked)
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
    terminal_audit = revision_directives.get("terminal_preservation_audit")
    if isinstance(terminal_audit, dict):
        scheduled = [day for day in result.itinerary.days if day.stops]
        terminal_audit["post_directive_terminal_poi_id"] = (
            scheduled[-1].stops[-1].poi.poi_id if scheduled else None
        )
        terminal_audit["ranked_contains_terminal"] = any(
            item.poi.poi_id == terminal_audit.get("requested_poi_id")
            for item in ranked
        )
    from dataclasses import replace
    from travel_agent.critic import critique_itinerary

    verified_ids = {item.poi.poi_id for item in ranked}
    filtered_days = [
        replace(
            day,
            stops=[stop for stop in day.stops if stop.poi.poi_id in verified_ids],
        )
        for day in result.itinerary.days
    ]
    if isinstance(terminal_audit, dict):
        filtered_scheduled = [day for day in filtered_days if day.stops]
        terminal_audit["post_filter_terminal_poi_id"] = (
            filtered_scheduled[-1].stops[-1].poi.poi_id
            if filtered_scheduled else None
        )
    final_itinerary = rebind_itinerary_routes(
        replace(result.itinerary, days=filtered_days),
        ctx.profile,
        ctx.provider if route_estimator is None else route_estimator,
    )
    if isinstance(terminal_audit, dict):
        rebound_scheduled = [day for day in final_itinerary.days if day.stops]
        terminal_audit["post_rebind_terminal_poi_id"] = (
            rebound_scheduled[-1].stops[-1].poi.poi_id
            if rebound_scheduled else None
        )
    result = replace(
        result,
        itinerary=final_itinerary,
        critic_result=critique_itinerary(final_itinerary, ctx.profile),
    )
    from travel_agent.reviser import _reconcile_revision_notes

    result = replace(
        result,
        revision_notes=_reconcile_revision_notes(
            list(result.revision_notes), final_itinerary, ctx.profile
        ),
    )
    itinerary_dict = itinerary_to_dict(result.itinerary)
    original_itinerary_dict = itinerary_to_dict(result.original_itinerary)
    def audit_assembled_terminal(stage: str) -> None:
        if not isinstance(terminal_audit, dict):
            return
        audit_days = [
            day for day in itinerary_dict.get("days") or [] if day.get("stops")
        ]
        terminal_audit[stage] = (
            audit_days[-1]["stops"][-1].get("poi", {}).get("poi_id")
            if audit_days else None
        )
    if isinstance(terminal_audit, dict):
        serialized_days = [
            day for day in itinerary_dict.get("days") or [] if day.get("stops")
        ]
        terminal_audit["post_serialize_terminal_poi_id"] = (
            serialized_days[-1]["stops"][-1].get("poi", {}).get("poi_id")
            if serialized_days else None
        )
    selected_candidate_ids = [
        str(stop.get("poi", {}).get("poi_id") or "")
        for day in itinerary_dict.get("days") or []
        for stop in day.get("stops") or []
        if isinstance(stop, dict) and isinstance(stop.get("poi"), dict)
    ]
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
    _close_post_plan_origin_route(
        ctx,
        itinerary_dict,
        domain_inputs,
        ctx.provider if route_estimator is None else route_estimator,
    )
    audit_assembled_terminal("post_origin_close_terminal_poi_id")
    _close_post_plan_return_route(
        ctx,
        itinerary_dict,
        domain_inputs,
        ctx.provider if route_estimator is None else route_estimator,
    )
    audit_assembled_terminal("post_return_close_terminal_poi_id")
    _close_post_plan_fixed_event_routes(
        ctx,
        itinerary_dict,
        domain_inputs,
        ctx.provider if route_estimator is None else route_estimator,
    )
    audit_assembled_terminal("post_fixed_close_terminal_poi_id")
    required_route_anchors, anchor_issues = _build_required_route_anchors(
        ctx.profile, itinerary_dict, domain_inputs
    )
    return_plan = _build_return_plan(ctx.profile, itinerary_dict, domain_inputs)
    audit_assembled_terminal("post_required_anchors_terminal_poi_id")
    lodging_plan = _build_lodging_plan(ctx.profile, domain_inputs)
    lodging_route_anchors = _build_lodging_route_anchors(
        ctx.profile,
        itinerary_dict,
        lodging_plan,
        ctx.provider if route_estimator is None else route_estimator,
    )
    audit_assembled_terminal("post_lodging_anchors_terminal_poi_id")
    required_route_anchors = _bind_lodging_fixed_event_routes(
        required_route_anchors, lodging_route_anchors
    )
    fixed_anchor_legs = [
        leg
        for leg in (required_route_anchors.get("legs") or [])
        if isinstance(leg, dict) and leg.get("kind") == "fixed_event_transfer"
    ]
    if fixed_anchor_legs and all(
        leg.get("evidence_status") == "provider_verified"
        for leg in fixed_anchor_legs
    ):
        anchor_issues = [
            issue for issue in anchor_issues
            if issue.code != "fixed_event_route_evidence_insufficient"
        ]
    if anchor_issues:
        critic_issues = [*result.critic_result.issues, *anchor_issues]
        result = replace(
            result,
            critic_result=CriticResult(
                passed=not any(issue.severity == "error" for issue in critic_issues),
                issues=critic_issues,
            ),
        )
    budget_plan = _build_budget_plan(ctx.profile, domain_inputs, lodging_plan)
    meal_strategy = _build_meal_strategy(
        ctx.profile, itinerary_dict, domain_inputs, return_plan
    )
    fixed_event_plan = _build_fixed_event_plan(
        ctx.profile, domain_inputs, required_route_anchors, lodging_route_anchors
    )
    mobility_plan = _build_mobility_plan(ctx.profile, itinerary_dict)
    candidate_verification = _build_candidate_verification(ctx.profile, ranked, domain_inputs)
    audit_assembled_terminal("post_supplemental_plans_terminal_poi_id")
    if isinstance(terminal_audit, dict):
        assembled_days = [
            day for day in itinerary_dict.get("days") or [] if day.get("stops")
        ]
        terminal_audit["pre_artifact_terminal_poi_id"] = (
            assembled_days[-1]["stops"][-1].get("poi", {}).get("poi_id")
            if assembled_days else None
        )
    parent_plan_artifact_id = str(
        revision_directives.get("parent_plan_artifact_id")
        or current_task_meta().get("parent_plan_artifact_id")
        or ""
    ) or None
    requested_changes = [
        str(item).strip()
        for item in str(revision_directives.get("reviewer_instructions") or "").splitlines()
        if str(item).strip()
    ]
    parent_payload = ctx.store.get(parent_plan_artifact_id) if parent_plan_artifact_id else None
    from travel_agent.artifact_policy import constraint_version

    active_version = constraint_version(ctx.profile)
    revision_lineage = list(
        (parent_payload or {}).get("revision_lineage") or []
        if isinstance(parent_payload, dict) else []
    )
    if parent_plan_artifact_id and parent_plan_artifact_id not in revision_lineage:
        revision_lineage.append(parent_plan_artifact_id)
    itinerary_changed = bool(
        isinstance(parent_payload, dict)
        and parent_payload.get("itinerary") != itinerary_dict
    )
    applied_changes = [str(item) for item in result.revision_notes if str(item).strip()]
    # Reviewer-directed repairs are verified by the engine against the parent
    # and repaired artifacts.  Planner revision prose is audit material, not a
    # trustworthy acceptance oracle: a real structural repair often has no
    # lexical overlap with the Reviewer's natural-language instruction.
    repair_verification_pending = list(requested_changes)
    unresolved_changes: list[str] = []
    # The stale parent is revision lineage, not active evidence in the newly
    # validated artifact.  Candidate recovery may read it transiently above,
    # but the final payload must not recursively embed or cite it as current.
    domain_inputs["previous_itineraries"] = []
    artifact_payload = {
            "parent_plan_artifact_id": parent_plan_artifact_id,
            "revision_lineage": revision_lineage,
            "state_version": {
                "constraint_revision": active_version["revision"],
                "constraint_hash": active_version["constraint_hash"],
                "constraint_snapshot": active_version["constraint_snapshot"],
            },
            "repair_targets": list(revision_directives.get("repair_targets") or []),
            "applied_changes": applied_changes,
            "unresolved_changes": unresolved_changes,
            "repair_verification_pending": repair_verification_pending,
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
            "source_artifact_ids": [
                record["artifact_id"]
                for record in records
                if record.get("kind") != "itinerary"
            ],
            "domain_inputs": domain_inputs,
            "return_plan": return_plan,
            "required_route_anchors": required_route_anchors,
            "lodging_plan": lodging_plan,
            "lodging_route_anchors": lodging_route_anchors,
            "budget_plan": budget_plan,
            "meal_strategy": meal_strategy,
            "fixed_event_plan": fixed_event_plan,
            "mobility_plan": mobility_plan,
            "candidate_verification": candidate_verification,
            "revision_directives": revision_directives,
        }
    if isinstance(planner_actuation, dict):
        from travel_agent.hybrid_planning.soft_preference_actuation import (
            finalize_selection_provenance,
        )

        artifact_payload["soft_preference_actuation"] = finalize_selection_provenance(
            planner_actuation,
            selected_candidate_ids,
            source_ranked_artifact_id=(
                str(ranked_record.get("artifact_id")) if ranked_record else None
            ),
            itinerary_payload=itinerary_dict,
            state_version=artifact_payload["state_version"],
        )
    from travel_agent.plan_invariants import validate_plan_artifact

    artifact_payload["validation_result"] = validate_plan_artifact(artifact_payload, ctx.profile)
    deliverable = bool(
        artifact_payload["validation_result"].get("passed") is True
        and artifact_payload["critic"].get("passed") is True
        and not artifact_payload.get("unresolved_changes")
    )
    # Finalizer owns the only current-state transition.  A Planner attempt is
    # immutable candidate material until deterministic gates and (for V3) the
    # semantic Reviewer select it for atomic promotion.
    artifact_payload["artifact_status"] = "candidate" if deliverable else "validation_failure"
    artifact_id = ctx.store.put("itinerary", artifact_payload)
    if duration_enabled:
        _persist_duration_diagnostic(
            ctx,
            candidate_artifact_id=artifact_id,
            duration_estimates=duration_estimates,
            ranked=ranked,
            state_version=active_version,
            source_artifact_ids=artifact_payload["source_artifact_ids"],
        )
    if deliverable:
        ctx.profile.constraint_state["_plan_status"] = "candidate_ready"
    else:
        # A failed rebuild must not clear the pending lifecycle or replace its
        # last usable parent with a validation_failure artifact.
        ctx.profile.constraint_state["_plan_status"] = "rebuild_pending"
        if parent_plan_artifact_id:
            ctx.profile.constraint_state[
                "_revisable_parent_plan_artifact_id"
            ] = parent_plan_artifact_id
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


def _persist_duration_diagnostic(
    ctx: SessionContext,
    *,
    candidate_artifact_id: str,
    duration_estimates: dict[str, dict[str, Any]],
    ranked: list[ScoredPOI],
    state_version: dict[str, Any],
    source_artifact_ids: list[str],
) -> str:
    """Persist request-owned read-only estimates outside itinerary lifecycle."""
    from travel_agent.hybrid_planning.duration_estimator import DURATION_BASELINE_VERSION

    meta = current_task_meta()
    request_id = str(meta.get("request_id") or "")
    turn_id = str(meta.get("turn_id") or (f"turn:{request_id}" if request_id else ""))
    poi_by_id = {item.poi.poi_id: item.poi for item in ranked}
    estimates: list[dict[str, Any]] = []
    for candidate_id, estimate in duration_estimates.items():
        poi = poi_by_id.get(candidate_id)
        source = str(estimate.get("source") or "")
        estimates.append(
            {
                "activity_id": candidate_id,
                "candidate_id": candidate_id,
                "category": estimate.get("category"),
                "estimated_minutes": estimate.get("estimated_minutes"),
                "range_minutes": list(estimate.get("range_minutes") or []),
                "source": source,
                "adjustments": list(estimate.get("adjustments") or []),
                "confidence": estimate.get("confidence"),
                "tool_evidence_reference": {
                    "source_artifact_ids": list(source_artifact_ids),
                    "poi_id": candidate_id,
                    "poi_source": getattr(poi, "source", None),
                },
                "is_fallback": source in {"category_baseline", "fallback_estimate"},
            }
        )
    payload = {
        "artifact_status": "diagnostic",
        "read_only": True,
        "request_id": request_id,
        "turn_id": turn_id,
        "candidate_artifact_id": candidate_artifact_id,
        "active_constraint_revision": state_version.get("revision"),
        "active_constraint_hash": state_version.get("constraint_hash"),
        "duration_estimator_version": DURATION_BASELINE_VERSION,
        "estimates": estimates,
    }
    return ctx.store.put(
        "duration_diagnostic",
        payload,
        request_id=request_id or None,
        task_id=str(meta.get("task_id") or "duration-diagnostic"),
        agent="duration_estimator",
    )


def _build_candidate_verification(
    profile: TravelProfile,
    ranked_pois: list[ScoredPOI],
    domain_inputs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record the outcome of explicit named-candidate suitability checks."""
    state = profile.constraint_state or {}
    terms = state.get("candidate_attractions") or []
    if isinstance(terms, str):
        terms = [terms]
    if not terms:
        return None
    weekday = str(state.get("weekday") or "").strip()
    rejected: list[POI] = []
    for entry in (domain_inputs or {}).get("attractions") or []:
        payload = entry.get("payload") if isinstance(entry, dict) else None
        for raw in (payload or {}).get("rejected_pois") or []:
            try:
                rejected.append(poi_from_dict(raw))
            except Exception:
                continue
    results: list[dict[str, Any]] = []
    for raw_term in terms:
        term = str(raw_term).strip()
        requirement = normalize_candidate_requirement(term)
        matches = [
            item.poi for item in ranked_pois
            if poi_covers_requirement(item.poi, requirement)
        ]
        suitable = next(
            (poi for poi in matches if _candidate_weekday_suitable(poi, weekday)),
            None,
        )
        if suitable is not None:
            match_evidence = canonical_entity_match_evidence(suitable, term)
            results.append({
                "requested_name": term,
                "status": "suitable",
                "matched_name": suitable.name,
                "original_name": suitable.name,
                "canonical_name": suitable.canonical_name or suitable.name,
                "canonical_match_evidence": match_evidence,
                "weekday": weekday or None,
                "opening_hours": suitable.opening_hours,
                "source": suitable.source,
            })
        else:
            closed = matches[0] if matches else next(
                (
                    poi for poi in rejected
                    if poi_covers_requirement(poi, requirement)
                    or requirement in {poi.name, poi.canonical_name, *poi.aliases}
                ),
                None,
            )
            results.append({
                "requested_name": term,
                "status": closed.verification_status if closed else "evidence_insufficient",
                "matched_name": closed.name if closed else None,
                "weekday": weekday or None,
                "opening_hours": closed.opening_hours if closed else None,
                "source": closed.source if closed else None,
                "reason": (
                    closed.verification_reason
                    if closed and closed.verification_reason
                    else "当天闭馆或未取得可核验的地点本体开放证据"
                ),
            })
    return {"status": "verified_named_candidates", "results": results}


def _candidate_name_matches(poi: Any, term: str) -> bool:
    return isinstance(poi, POI) and poi_covers_requirement(poi, term)


def _candidate_weekday_suitable(poi: Any, weekday: str) -> bool:
    hours = str(getattr(poi, "opening_hours", "") or "").replace(" ", "")
    if not weekday or not hours:
        return True
    return not (
        f"{weekday}全天不开放" in hours
        or f"{weekday}闭馆" in hours
        or (weekday == "周一" and "周一闭馆" in hours)
    )


def _build_meal_strategy(
    profile: TravelProfile,
    itinerary: dict[str, Any],
    domain_inputs: dict[str, Any],
    return_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose dining semantics as a first-class Reviewer input."""
    state = profile.constraint_state or {}
    raw_dietary = state.get("dietary") or []
    dietary_constraints = (
        [str(raw_dietary)]
        if isinstance(raw_dietary, str)
        else [str(item) for item in raw_dietary]
    )
    restaurant_evidence = [
        {
            "artifact_id": item.get("artifact_id"),
            "status": (item.get("payload") or {}).get("status"),
            "restaurants": (item.get("payload") or {}).get("restaurants"),
        }
        for item in (domain_inputs.get("restaurants") or [])
        if isinstance(item, dict)
    ]
    verified_dietary_restaurants: list[dict[str, Any]] = []
    if dietary_constraints:
        from travel_agent.critic import poi_complies_with_dietary

        for evidence in restaurant_evidence:
            for raw in evidence.get("restaurants") or []:
                if not isinstance(raw, dict):
                    continue
                try:
                    poi = poi_from_dict(raw)
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    poi.verification_status == "verified"
                    and poi.source
                    and poi_complies_with_dietary(poi, profile)
                ):
                    verified_dietary_restaurants.append(raw)

    def grounded_reservation(
        *, day_index: int, period: str, start: str, end: str
    ) -> dict[str, Any]:
        if verified_dietary_restaurants:
            restaurant = verified_dietary_restaurants[
                (max(1, day_index) - 1) % len(verified_dietary_restaurants)
            ]
            return {
                "day_index": day_index,
                "period": period,
                "name": restaurant.get("name"),
                "poi_id": restaurant.get("poi_id"),
                "start_time": start,
                "end_time": end,
                "source": restaurant.get("source"),
                "verification_status": restaurant.get("verification_status"),
                "is_reservation_only": True,
            }
        label = "晚餐" if period == "dinner" else "午餐"
        return {
            "day_index": day_index,
            "period": period,
            "name": f"{label}时段（当日活动区域就近自行安排）",
            "start_time": start,
            "end_time": end,
            "source": "deterministic_schedule_reservation",
            "is_reservation_only": True,
        }

    scheduled: list[dict[str, Any]] = []
    unscheduled_meal_days: list[dict[str, Any]] = []
    for day in itinerary.get("days") or []:
        meal_periods: set[str] = set()
        for stop in day.get("stops") or []:
            poi = stop.get("poi") if isinstance(stop, dict) else None
            if not isinstance(poi, dict):
                continue
            if str(poi.get("category") or "").lower() in {
                "food", "restaurant", "cafe", "meal",
            }:
                try:
                    meal_start = time.fromisoformat(str(stop.get("start_time")))
                    period = "dinner" if meal_start.hour >= 16 else "lunch"
                except (TypeError, ValueError):
                    period = "lunch"
                meal_periods.add(period)
                scheduled.append({
                    "day_index": day.get("day_index"),
                    "period": period,
                    "name": poi.get("name"),
                    "poi_id": poi.get("poi_id"),
                    "start_time": stop.get("start_time"),
                    "source": poi.get("source"),
                })
        latest_end_min = None
        cutoff_values: list[object] = []
        if (
            isinstance(return_plan, dict)
            and int(day.get("day_index") or 0) == len(itinerary.get("days") or [])
        ):
            cutoff_values.append(return_plan.get("activity_cutoff"))
        if state.get("activity_end_deadline"):
            cutoff_values.append(state.get("activity_end_deadline"))
        for value in cutoff_values:
            try:
                from travel_agent.constraint_events import clock_minutes

                parsed = clock_minutes(value)
            except (TypeError, ValueError):
                continue
            latest_end_min = parsed if latest_end_min is None else min(latest_end_min, parsed)

        if not meal_periods:
            meal_window = _available_meal_reservation(
                day, latest_end_min=latest_end_min
            )
            if meal_window is None:
                unscheduled_meal_days.append({
                    "day_index": day.get("day_index"),
                    "reason": "no_non_overlapping_meal_window",
                })
            else:
                meal_start, meal_end = meal_window
                period = "dinner" if meal_start >= "17:00" else "lunch"
                meal_periods.add(period)
                scheduled.append(grounded_reservation(
                    day_index=int(day.get("day_index") or 0),
                    period=period,
                    start=meal_start,
                    end=meal_end,
                ))
        # A dietary hard constraint applies to every meal, not merely to an
        # already scheduled dinner. Reserve a separately evidenced lunch even
        # when another food stop exists that day.
        if dietary_constraints and "lunch" not in meal_periods:
            lunch = _available_meal_reservation(day, latest_end_min=latest_end_min)
            if lunch is not None and lunch[0] < "15:00":
                meal_start, meal_end = lunch
                meal_periods.add("lunch")
                scheduled.append(grounded_reservation(
                    day_index=int(day.get("day_index") or 0),
                    period="lunch",
                    start=meal_start,
                    end=meal_end,
                ))
            else:
                unscheduled_meal_days.append({
                    "day_index": day.get("day_index"),
                    "reason": "no_non_overlapping_dietary_lunch_window",
                })
        occupied_end = max(
            (
                _clock_to_minute(stop.get("start_time"))
                + int(stop.get("duration_min") or 0)
                for stop in (day.get("stops") or [])
                if _clock_to_minute(stop.get("start_time")) is not None
            ),
            default=0,
        )
        is_multi_day_plan = len(itinerary.get("days") or []) > 1
        dinner_needed = bool(
            (latest_end_min is not None and latest_end_min >= 19 * 60)
            or occupied_end >= 19 * 60
            or (is_multi_day_plan and latest_end_min is None)
        )
        if dinner_needed and "dinner" not in meal_periods:
            dinner = _available_dinner_reservation(day, latest_end_min=latest_end_min)
            if dinner is not None:
                meal_start, meal_end = dinner
                scheduled.append(grounded_reservation(
                    day_index=int(day.get("day_index") or 0),
                    period="dinner",
                    start=meal_start,
                    end=meal_end,
                ))
    scheduled.sort(key=lambda item: (
        int(item.get("day_index") or 0), str(item.get("start_time") or "")
    ))
    return {
        "dietary_constraints": dietary_constraints,
        "dietary_policy": (
            {
                "mode": "confirm_or_replace",
                "requirements": dietary_constraints,
                "instruction": (
                    "所有就餐均按饮食限制筛选；下单前向餐厅逐项确认，"
                    "无法确认则更换餐厅或改为自行安排，不把未核实餐厅视为合规。"
                ),
            }
            if dietary_constraints
            else None
        ),
        "scheduled_meals": scheduled,
        "unscheduled_meal_days": unscheduled_meal_days,
        "restaurant_evidence": restaurant_evidence,
        "concrete_restaurant_required": bool(
            state.get("specific_restaurant_required")
            or state.get("concrete_restaurant_required")
        ),
    }


def _available_meal_reservation(
    day: dict[str, Any], *, latest_end_min: int | None = None
) -> tuple[str, str] | None:
    """Choose a meal window that also preserves inbound transfer time."""
    occupied = _meal_occupied_intervals(day)
    latest_lunch_start = 14 * 60
    if latest_end_min is not None:
        latest_lunch_start = min(15 * 60, latest_end_min - 60)
    for candidate in range(11 * 60 + 30, latest_lunch_start + 1, 15):
        if _meal_window_fits(candidate, occupied):
            return _offset_clock("00:00", candidate) or "12:00", _offset_clock("00:00", candidate + 60) or "13:00"
    # A stop ending at 11:00 and a following inbound transfer beginning just
    # before 12:45 leaves a valid 11:15 lunch, even though the canonical search
    # starts at 11:30.  Use this bounded early slot before declaring lunch
    # unschedulable or falling through to dinner.
    early_lunch = 11 * 60 + 15
    if (
        early_lunch <= latest_lunch_start
        and _meal_window_fits(early_lunch, occupied)
    ):
        return "11:15", "12:15"
    for candidate in range(17 * 60 + 30, 20 * 60 + 1, 15):
        if latest_end_min is not None and candidate + 60 > latest_end_min:
            continue
        if _meal_window_fits(candidate, occupied):
            return _offset_clock("00:00", candidate) or "18:00", _offset_clock("00:00", candidate + 60) or "19:00"
    return None


def _available_dinner_reservation(
    day: dict[str, Any], *, latest_end_min: int | None = None
) -> tuple[str, str] | None:
    """Choose a non-overlapping dinner window within the usable day."""
    occupied = _meal_occupied_intervals(day)
    latest_start = 20 * 60
    if latest_end_min is not None:
        latest_start = min(latest_start, latest_end_min - 60)
    for candidate in range(16 * 60 + 30, latest_start + 1, 15):
        if _meal_window_fits(candidate, occupied):
            return (
                _offset_clock("00:00", candidate) or "18:00",
                _offset_clock("00:00", candidate + 60) or "19:00",
            )
    # A long inbound transfer to an explicitly late activity can consume the
    # normal dinner slots.  In that case offer an earlier meal, while keeping
    # 16:30 as the default whenever it is feasible.
    for candidate in range(15 * 60 + 30, 16 * 60 + 30, 15):
        if latest_end_min is not None and candidate + 60 > latest_end_min:
            continue
        if _meal_window_fits(candidate, occupied):
            return (
                _offset_clock("00:00", candidate) or "15:30",
                _offset_clock("00:00", candidate + 60) or "16:30",
            )
    return None


def _meal_occupied_intervals(day: dict[str, Any]) -> list[tuple[int, int]]:
    """Return stop intervals extended by the route needed to reach each stop."""
    occupied: list[tuple[int, int]] = []
    for stop in day.get("stops") or []:
        if not isinstance(stop, dict):
            continue
        start_min = _clock_to_minute(stop.get("start_time"))
        if start_min is None:
            continue
        route = stop.get("route_from_previous") or {}
        try:
            inbound_min = max(0, int(route.get("duration_min") or 0))
        except (TypeError, ValueError):
            inbound_min = 0
        occupied.append((
            start_min - inbound_min,
            start_min + int(stop.get("duration_min") or 0),
        ))
    return occupied


def _meal_window_fits(
    candidate_start_min: int,
    occupied: list[tuple[int, int]],
    *,
    duration_min: int = 60,
    transfer_buffer_min: int = 15,
) -> bool:
    """Keep a small transfer buffer around a reserved meal window."""
    candidate_end_min = candidate_start_min + duration_min
    return not any(
        candidate_start_min < end + transfer_buffer_min
        and candidate_end_min > start - transfer_buffer_min
        for start, end in occupied
    )


def _clock_to_minute(value: object) -> int | None:
    try:
        parsed = time.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.hour * 60 + parsed.minute


def _build_fixed_event_plan(
    profile: TravelProfile,
    domain_inputs: dict[str, Any],
    required_route_anchors: dict[str, Any] | None = None,
    lodging_route_anchors: dict[str, Any] | None = None,
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
        anchor = next((
            item
            for item in ((required_route_anchors or {}).get("legs") or [])
            if isinstance(item, dict)
            and item.get("kind") == "fixed_event_transfer"
            and normalize_entity_name(item.get("required_name"))
            == normalize_entity_name(location)
        ), None)
        anchor_routes = list((anchor or {}).get("routes") or [])
        route = next((
            item for item in anchor_routes
            if isinstance(item, dict)
            and canonical_route_evidence_status(item) == "provider_verified"
        ), None)
        day_index = None
        try:
            if event.get("day") is not None:
                day_index = int(event["day"])
            elif event.get("date") and profile.start_date:
                day_index = (
                    date.fromisoformat(str(event["date"]))
                    - date.fromisoformat(str(profile.start_date))
                ).days + 1
        except (TypeError, ValueError):
            day_index = None
        event_poi_id = str((anchor or {}).get("event_poi_id") or "")
        # If the appointment is the day's first itinerary stop, the concrete
        # departure context is the selected hotel. A short subvenue-to-parent
        # route inside the appointment complex must not masquerade as the
        # traveller's inbound transfer.
        hotel_route = next((
            leg
            for daily in ((lodging_route_anchors or {}).get("daily_routes") or [])
            if isinstance(daily, dict) and daily.get("day_index") == day_index
            for leg in (daily.get("legs") or [])
            if isinstance(leg, dict)
            and leg.get("position") == "hotel_to_first_stop"
            and str(leg.get("destination_poi_id") or "") == event_poi_id
            and canonical_route_evidence_status(leg) != "unavailable"
        ), None)
        route = hotel_route or route
        duration = int(route.get("duration_min") or 0) if route else None
        buffer_min = TRANSFER_BUFFER_MIN
        evidence_status = (
            canonical_route_evidence_status(route) if route else "unavailable"
        )
        items.append({
            **event,
            "transfer_duration_min": duration,
            "recommended_departure": _offset_clock(event.get("start"), -(duration + buffer_min) if duration else None),
            "recommended_return_arrival": _offset_clock(event.get("end"), duration),
            "required_buffer_min": buffer_min,
            "route_source": route.get("source") if route else None,
            "route_evidence_status": evidence_status,
            "route_evidence_reference": route,
        })
    return {"status": "fixed_appointment_buffers", "events": items}


def _bind_lodging_fixed_event_routes(
    required_route_anchors: dict[str, Any] | None,
    lodging_route_anchors: dict[str, Any] | None,
) -> dict[str, Any]:
    """Replace irrelevant venue-internal evidence with the hotel's inbound leg.

    This applies only when the fixed event is the day's first itinerary stop,
    which is proven by the hotel-to-first-stop destination ID.
    """
    if not isinstance(required_route_anchors, dict):
        return {}
    anchors = copy.deepcopy(required_route_anchors)
    daily_routes = [
        day
        for day in ((lodging_route_anchors or {}).get("daily_routes") or [])
        if isinstance(day, dict)
    ]
    for leg in anchors.get("legs") or []:
        if not isinstance(leg, dict) or leg.get("kind") != "fixed_event_transfer":
            continue
        event_poi_id = str(leg.get("event_poi_id") or "")
        hotel_leg = next((
            route
            for day in daily_routes
            for route in (day.get("legs") or [])
            if isinstance(route, dict)
            and route.get("position") == "hotel_to_first_stop"
            and str(route.get("destination_poi_id") or "") == event_poi_id
        ), None)
        if hotel_leg is None:
            continue
        status = canonical_route_evidence_status(hotel_leg)
        leg["origin_poi_id"] = hotel_leg.get("origin_poi_id")
        leg["routes"] = [dict(hotel_leg)]
        leg["evidence_status"] = (
            "provider_verified" if status == "provider_verified" else "unavailable"
        )
    for leg in anchors.get("legs") or []:
        if isinstance(leg, dict) and leg.get("kind"):
            anchors[str(leg["kind"])] = leg
    return anchors


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
    prepaid_lodging = state.get("prepaid_lodging_cny")
    explicit = bool(
        profile.hotel_area
        or any(
            state.get(key) not in (None, "", [], {})
            for key in (
                "lodging_area",
                "compare_lodging_areas",
                "hotel_budget_per_night_cny",
                "lodging_flexibility",
            )
        )
    )
    nights = max(0, int(profile.days or 1) - 1)
    # A stated prepaid lodging amount is user-provided ownership/cost evidence,
    # not a request for the system to invent or replace the hotel.  Only a
    # separate area, comparison, nightly-budget, or flexibility request opens a
    # new lodging recommendation that must be backed by provider evidence.
    if prepaid_lodging not in (None, "", [], {}) and not explicit:
        return {
            "required": True,
            "explicit_requirement": False,
            "status": "user_owned_prepaid",
            "nights": nights,
            "prepaid_lodging_cny": float(prepaid_lodging),
            "evidence_status": "user_provided",
            "source_artifact_ids": [],
        }
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


def _endpoint_name_matches(poi: POI, required_name: str) -> bool:
    """Match a provider POI as an endpoint without treating a nearby POI as equal."""
    required = normalize_entity_name(required_name)
    names = {
        normalize_entity_name(value)
        for value in (poi.name, poi.canonical_name, *poi.aliases)
        if normalize_entity_name(value)
    }
    if required in names:
        return True
    if poi.entity_type == "transport":
        transport_suffixes = {
            "站", "火车站", "高铁站", "地铁站", "公交站", "客运站",
            "机场", "航站楼", "码头", "轮渡码头",
        }
        if any(
            name.startswith(required) and name[len(required):] in transport_suffixes
            for name in names
        ):
            return True
    required_key = canonical_identity_key(
        required_name, city=poi.city, entity_type=poi.entity_type
    )
    return bool(
        required_key
        and any(
            required_key
            == canonical_identity_key(
                value, city=poi.city, entity_type=poi.entity_type
            )
            for value in (poi.name, poi.canonical_name, *poi.aliases)
        )
    )


def _is_grounded_route_endpoint(poi: POI) -> bool:
    """Allow provider-classified transport POIs as route endpoints, not activities."""
    return poi.verification_status == "verified" or (
        poi.entity_type == "transport"
        and evidence_status_for_source(poi.source) == "provider_verified"
    )


def _deadline_departure(
    deadline: object,
    *,
    duration_min: int,
    buffer_min: int,
) -> str | None:
    text = str(deadline or "").strip()
    if not text:
        return None
    delta = timedelta(minutes=max(0, duration_min) + max(0, buffer_min))
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if "T" in text or " " in text:
            return (parsed - delta).isoformat(timespec="minutes")
    except ValueError:
        pass
    try:
        parsed_time = time.fromisoformat(text)
    except ValueError:
        return None
    total = parsed_time.hour * 60 + parsed_time.minute - int(delta.total_seconds() // 60)
    total = max(0, total)
    return f"{total // 60:02d}:{total % 60:02d}"


def _same_city(left: object, right: object) -> bool:
    def key(value: object) -> str:
        text = str(value or "").strip()
        return text[:-1] if text.endswith("市") and len(text) > 1 else text

    left_key, right_key = key(left), key(right)
    return bool(left_key and right_key and left_key == right_key)


def _estimate_hard_route(
    route_estimator: Any,
    origin: POI,
    endpoint: POI,
    preferred_mode: str,
    *,
    allow_taxi_fallback: bool,
) -> RouteInfo | None:
    """Get endpoint-bound evidence, retrying taxi for a hard local deadline.

    AMap can return a geometry fallback for transit even when its driving API
    has a fully verified route.  For a fixed appointment or same-city return,
    that taxi route is valid feasibility evidence.  Cross-city returns never
    take this fallback because a car route is not a substitute for timetable
    inventory.
    """
    primary: RouteInfo | None = None
    try:
        primary = normalize_route_evidence(
            route_estimator.estimate_route(origin, endpoint, preferred_mode)
        )
    except Exception:  # noqa: BLE001 - callers fail closed
        primary = None
    if (
        primary is not None
        and route_supports_endpoints(primary, origin.poi_id, endpoint.poi_id)
        and canonical_route_evidence_status(primary) == "provider_verified"
    ):
        return primary
    if allow_taxi_fallback and preferred_mode not in {"taxi", "drive"}:
        try:
            taxi = normalize_route_evidence(
                route_estimator.estimate_route(origin, endpoint, "taxi")
            )
        except Exception:  # noqa: BLE001 - retain primary evidence below
            taxi = None
        if (
            taxi is not None
            and route_supports_endpoints(taxi, origin.poi_id, endpoint.poi_id)
            and canonical_route_evidence_status(taxi) == "provider_verified"
        ):
            return taxi
    if primary is not None and route_supports_endpoints(
        primary, origin.poi_id, endpoint.poi_id
    ):
        return primary
    return None


def _close_post_plan_origin_route(
    ctx: SessionContext,
    itinerary: dict[str, Any],
    domain_inputs: dict[str, Any],
    route_estimator: Any,
) -> dict[str, Any] | None:
    """Bind an explicit trip origin to the Planner's actual first stop."""
    state = ctx.profile.constraint_state or {}
    origin_name = str(state.get("origin") or "").strip()
    days = [day for day in (itinerary.get("days") or []) if day.get("stops")]
    if not origin_name or not days:
        return None
    try:
        first_poi = poi_from_dict(days[0]["stops"][0]["poi"])
    except (KeyError, TypeError, ValueError):
        return None
    if not first_poi.poi_id:
        return None
    ctx.remember_pois([first_poi])

    routes = [
        entry.get("payload", entry)
        for entry in (domain_inputs.get("transport") or [])
        if isinstance(entry, dict)
    ]
    exact_existing = next((
        dict(route)
        for route in reversed(routes)
        if isinstance(route, dict)
        and str(route.get("destination_poi_id") or "") == first_poi.poi_id
        and normalize_entity_name(route.get("origin_name"))
        == normalize_entity_name(origin_name)
        and str(route.get("origin_poi_id") or "")
    ), None)
    endpoint = (
        ctx.poi(str(exact_existing.get("origin_poi_id") or ""))
        if exact_existing else None
    )
    if endpoint is None:
        endpoint = next((
            poi for poi in ctx.pois_by_id.values()
            if poi.poi_id != first_poi.poi_id
            and _is_grounded_route_endpoint(poi)
            and _endpoint_name_matches(poi, origin_name)
        ), None)
    if endpoint is None:
        try:
            candidates = ctx.provider.search_pois(
                city=str(ctx.profile.destination or ""),
                query_tags=[origin_name],
                category=None,
                max_results=5,
            )
        except ProviderRateLimitError:
            raise
        except Exception:  # noqa: BLE001 - unavailable remains explicit below
            candidates = []
        normalized = [normalize_poi_entity(item, ctx.profile) for item in candidates]
        endpoint = next((
            poi for poi in normalized
            if _is_grounded_route_endpoint(poi)
            and _endpoint_name_matches(poi, origin_name)
        ), None)
        if endpoint is not None:
            ctx.remember_pois([endpoint])
    if endpoint is None:
        return None

    if (
        exact_existing
        and canonical_route_evidence_status(exact_existing) == "provider_verified"
    ):
        route_payload = exact_existing
        route_payload.update({
            "origin_poi_id": endpoint.poi_id,
            "destination_poi_id": first_poi.poi_id,
        })
    else:
        route = _estimate_hard_route(
            route_estimator,
            endpoint,
            first_poi,
            ctx.profile.transport_mode,
            allow_taxi_fallback=_same_city(endpoint.city, first_poi.city),
        )
        if route is None or canonical_route_evidence_status(route) != "provider_verified":
            live_route = _estimate_hard_route(
                ctx.provider,
                endpoint,
                first_poi,
                ctx.profile.transport_mode,
                allow_taxi_fallback=_same_city(endpoint.city, first_poi.city),
            )
            if (
                live_route is not None
                and canonical_route_evidence_status(live_route)
                == "provider_verified"
            ):
                route = live_route
        if route is None or not route_supports_endpoints(
            route, endpoint.poi_id, first_poi.poi_id
        ):
            route = RouteInfo(
                origin_poi_id=endpoint.poi_id,
                destination_poi_id=first_poi.poi_id,
                distance_km=0,
                duration_min=0,
                mode=ctx.profile.transport_mode,
                source="unavailable",
                evidence_status="unavailable",
            )
        route_payload = {
            "origin_poi_id": endpoint.poi_id,
            "destination_poi_id": first_poi.poi_id,
            "mode": route.mode,
            "duration_min": route.duration_min,
            "distance_km": route.distance_km,
            "source": route.source,
            "evidence_status": canonical_route_evidence_status(route),
            "walking_distance_km": route.walking_distance_km,
        }
    route_payload.update({
        "origin_name": origin_name,
        "origin_provider_name": endpoint.name,
        "destination_name": first_poi.name,
        "trip_origin": True,
    })
    artifact_id = ctx.store.put("routes", route_payload)
    domain_inputs.setdefault("transport", []).append({
        "artifact_id": artifact_id,
        "payload": route_payload,
        "post_plan_origin_closure": True,
    })
    return route_payload


def _close_post_plan_return_route(
    ctx: SessionContext,
    itinerary: dict[str, Any],
    domain_inputs: dict[str, Any],
    route_estimator: Any,
) -> dict[str, Any] | None:
    """Bind the actual final stop to the return endpoint after planning.

    Endpoint discovery is bounded to one provider lookup.  Any lookup or route
    failure remains explicit unavailable evidence and is rejected by the
    deterministic return-deadline gate.
    """
    state = ctx.profile.constraint_state or {}
    return_name = str(state.get("return_location") or "").strip()
    origin_name = str(state.get("origin") or "").strip()
    endpoint_query = (
        origin_name
        if origin_name
        and normalize_entity_name(return_name) != normalize_entity_name(origin_name)
        and normalize_entity_name(return_name) in normalize_entity_name(origin_name)
        else return_name
    )
    deadline = state.get("return_deadline")
    days = [day for day in (itinerary.get("days") or []) if day.get("stops")]
    if not return_name or not deadline or not days:
        return None
    try:
        final_poi = poi_from_dict(days[-1]["stops"][-1]["poi"])
    except (KeyError, TypeError, ValueError):
        return None
    if not final_poi.poi_id:
        return None
    ctx.remember_pois([final_poi])
    routes = [
        entry.get("payload", entry)
        for entry in (domain_inputs.get("transport") or [])
        if isinstance(entry, dict)
    ]
    exact_existing = next((
        dict(route)
        for route in routes
        if isinstance(route, dict)
        and (
            str(route.get("origin_poi_id") or "") == final_poi.poi_id
            or normalize_entity_name(route.get("origin_name"))
            == normalize_entity_name(final_poi.name)
        )
        and normalize_entity_name(return_name) in {
            normalize_entity_name(route.get("destination_name")),
            normalize_entity_name(route.get("return_location_name")),
        }
        and str(route.get("destination_poi_id") or "")
    ), None)
    endpoint: POI | None = None
    if exact_existing:
        endpoint = ctx.poi(str(exact_existing.get("destination_poi_id") or ""))
    if endpoint is None:
        endpoint = next((
            poi for poi in ctx.pois_by_id.values()
            if poi.poi_id != final_poi.poi_id
            and _is_grounded_route_endpoint(poi)
            and _endpoint_name_matches(poi, endpoint_query)
        ), None)
    if endpoint is None:
        evidence_pois: list[POI] = []
        for entries in domain_inputs.values():
            for entry in (entries if isinstance(entries, list) else []):
                payload = entry.get("payload", entry) if isinstance(entry, dict) else {}
                if not isinstance(payload, dict):
                    continue
                for key in ("endpoint_pois", "pois", "items"):
                    for raw in payload.get(key) or []:
                        candidate = raw.get("poi") if isinstance(raw, dict) else None
                        candidate = candidate if isinstance(candidate, dict) else raw
                        if not isinstance(candidate, dict):
                            continue
                        try:
                            evidence_pois.append(
                                normalize_poi_entity(
                                    poi_from_dict(candidate), ctx.profile
                                )
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
        endpoint = next((
            poi for poi in evidence_pois
            if _is_grounded_route_endpoint(poi)
            and _endpoint_name_matches(poi, endpoint_query)
        ), None)
        if endpoint is not None:
            ctx.remember_pois([endpoint])
    if endpoint is None:
        try:
            candidates = ctx.provider.search_pois(
                city=str(ctx.profile.destination or ""),
                query_tags=[endpoint_query],
                category=None,
                max_results=5,
            )
        except ProviderRateLimitError:
            raise
        except Exception:  # noqa: BLE001 - fail closed below
            candidates = []
        normalized = [normalize_poi_entity(item, ctx.profile) for item in candidates]
        endpoint = next((
            poi for poi in normalized
            if _is_grounded_route_endpoint(poi)
            and _endpoint_name_matches(poi, endpoint_query)
        ), None)
        if endpoint is not None:
            ctx.remember_pois([endpoint])
    if endpoint is None:
        return None

    if (
        exact_existing
        and str(exact_existing.get("destination_poi_id") or "") == endpoint.poi_id
        and canonical_route_evidence_status(exact_existing) == "provider_verified"
    ):
        route_payload = exact_existing
        # A rebuild may assign a fresh local ID to the same provider entity.
        # Exact normalized endpoint names let the verified route be rebound to
        # the current revision without weakening endpoint identity.
        route_payload.update({
            "origin_poi_id": final_poi.poi_id,
            "destination_poi_id": endpoint.poi_id,
            "origin_name": final_poi.name,
            "destination_name": endpoint.name,
        })
        route_payload["evidence_status"] = str(
            route_payload.get("evidence_status")
            or evidence_status_for_source(route_payload.get("source"))
        )
    else:
        route = _estimate_hard_route(
            route_estimator,
            final_poi,
            endpoint,
            ctx.profile.transport_mode,
            allow_taxi_fallback=_same_city(
                endpoint.city, ctx.profile.destination
            ),
        )
        if route is None or canonical_route_evidence_status(route) != "provider_verified":
            # The Planner normally receives a no-network estimator backed by
            # Transport artifacts. A rebuild can change the actual final stop
            # after those artifacts were produced. Close that new hard
            # endpoint once through the configured provider rather than
            # publishing a geometry estimate or reusing a stale route.
            live_route = _estimate_hard_route(
                ctx.provider,
                final_poi,
                endpoint,
                ctx.profile.transport_mode,
                allow_taxi_fallback=_same_city(
                    endpoint.city, ctx.profile.destination
                ),
            )
            if (
                live_route is not None
                and canonical_route_evidence_status(live_route)
                == "provider_verified"
            ):
                route = live_route
        if route is None:
            route = RouteInfo(
                origin_poi_id=final_poi.poi_id,
                destination_poi_id=endpoint.poi_id,
                distance_km=0,
                duration_min=0,
                mode=ctx.profile.transport_mode,
                source="unavailable",
                evidence_status="unavailable",
            )
        if not route_supports_endpoints(route, final_poi.poi_id, endpoint.poi_id):
            route = RouteInfo(
                origin_poi_id=final_poi.poi_id,
                destination_poi_id=endpoint.poi_id,
                distance_km=0,
                duration_min=0,
                mode=ctx.profile.transport_mode,
                source="unavailable",
                evidence_status="unavailable",
            )
        route_payload = {
            "origin_poi_id": final_poi.poi_id,
            "destination_poi_id": endpoint.poi_id,
            "origin_name": final_poi.name,
            "destination_name": endpoint.name,
            "mode": route.mode,
            "duration_min": route.duration_min,
            "distance_km": route.distance_km,
            "source": route.source,
            "evidence_status": route.evidence_status,
            "walking_distance_km": route.walking_distance_km,
        }
    buffer_min = max(30, TRANSFER_BUFFER_MIN)
    route_payload.update({
        "return_location_name": return_name,
        "return_deadline": str(deadline),
        "required_buffer_min": buffer_min,
        "recommended_latest_departure": _deadline_departure(
            deadline,
            duration_min=int(route_payload.get("duration_min") or 0),
            buffer_min=buffer_min,
        ),
        "hard_feasibility_proven": str(route_payload.get("evidence_status") or "")
        in {"provider_verified", "deterministic_estimate"},
    })
    latest_departure = str(route_payload.get("recommended_latest_departure") or "")
    try:
        latest_clock = (
            datetime.fromisoformat(latest_departure.replace("Z", "+00:00")).time()
            if "T" in latest_departure or " " in latest_departure
            else time.fromisoformat(latest_departure)
        )
        final_start = time.fromisoformat(str(days[-1]["stops"][-1].get("start_time")))
        final_end_minutes = (
            final_start.hour * 60
            + final_start.minute
            + int(days[-1]["stops"][-1].get("duration_min") or 0)
        )
        latest_minutes = latest_clock.hour * 60 + latest_clock.minute
    except (KeyError, TypeError, ValueError):
        final_end_minutes = latest_minutes = 0
    fixed_locations = {
        normalize_entity_name(event.get("location"))
        for event in state.get("fixed_events") or []
        if isinstance(event, dict) and event.get("location")
    }
    final_is_required = any(
        canonical_entity_match_evidence(final_poi, term) is not None
        for term in ctx.profile.must_visit
    ) or normalize_entity_name(final_poi.name) in fixed_locations
    if (
        latest_minutes
        and final_end_minutes > latest_minutes
        and not final_is_required
        and len(days[-1].get("stops") or []) > 1
    ):
        # The return deadline is a hard arrival constraint.  Remove a late
        # optional tail and recompute evidence from the new actual final stop;
        # never delete a must-visit or fixed appointment to manufacture fit.
        days[-1]["stops"].pop()
        return _close_post_plan_return_route(
            ctx, itinerary, domain_inputs, route_estimator
        )
    artifact_id = ctx.store.put("routes", route_payload)
    domain_inputs.setdefault("transport", []).append({
        "artifact_id": artifact_id,
        "payload": route_payload,
        "post_plan_endpoint_closure": True,
    })
    return route_payload


def _close_post_plan_fixed_event_routes(
    ctx: SessionContext,
    itinerary: dict[str, Any],
    domain_inputs: dict[str, Any],
    route_estimator: Any,
) -> list[dict[str, Any]]:
    """Ground transfers from the final scheduled predecessor to fixed events.

    Named events reuse their exact scheduled venue.  Area-only user-owned
    events keep their reserved block and bind to a verified geographic anchor
    without inventing a restaurant or activity stop.
    """
    state = ctx.profile.constraint_state or {}
    unspecified = {
        normalize_entity_name(item)
        for item in state.get("user_owned_unspecified_fixed_event_locations") or []
        if normalize_entity_name(item)
    }
    events = [
        event
        for event in state.get("fixed_events") or []
        if isinstance(event, dict)
        and str(event.get("location") or "").strip()
        and str(event.get("location") or "").strip()
        not in {"自由活动", "休息", "自由时间"}
    ]
    if not events:
        return []
    existing_routes = [
        entry.get("payload", entry)
        for entry in (domain_inputs.get("transport") or [])
        if isinstance(entry, dict)
    ]
    closed: list[dict[str, Any]] = []
    for event in events:
        location = str(event.get("location") or "").strip()
        try:
            day_index = int(event.get("day") or 0)
            event_start = time.fromisoformat(str(event.get("start")))
            start_minutes = event_start.hour * 60 + event_start.minute
        except (TypeError, ValueError):
            continue
        day = next((
            item for item in itinerary.get("days") or []
            if int(item.get("day_index") or 0) == day_index
        ), None)
        scheduled_endpoint_raw = next((
            stop.get("poi")
            for stop in (day or {}).get("stops") or []
            if normalize_entity_name(
                stop.get("poi", {}).get("canonical_name")
                or stop.get("poi", {}).get("name")
            ) == normalize_entity_name(location)
        ), None)
        prior_stops: list[tuple[int, dict[str, Any]]] = []
        for stop in (day or {}).get("stops") or []:
            try:
                start = time.fromisoformat(str(stop.get("start_time")))
                end_minutes = (
                    start.hour * 60 + start.minute + int(stop.get("duration_min") or 0)
                )
            except (TypeError, ValueError):
                continue
            if end_minutes <= start_minutes:
                prior_stops.append((end_minutes, stop))
        if not prior_stops:
            continue
        prior_raw = max(prior_stops, key=lambda item: item[0])[1].get("poi")
        try:
            origin = poi_from_dict(prior_raw)
        except (KeyError, TypeError, ValueError):
            continue
        try:
            endpoint = poi_from_dict(scheduled_endpoint_raw) if scheduled_endpoint_raw else None
        except (KeyError, TypeError, ValueError):
            endpoint = None
        if endpoint is None:
            # A targeted Transport repair is stronger than re-resolving an
            # area name to whichever matching POI happened to be inserted
            # first (often a metro station).  Preserve the exact destination
            # of a verified route from the final scheduled predecessor.
            target = normalize_entity_name(location)
            repaired_endpoints: list[tuple[int, POI]] = []
            for route in existing_routes:
                if (
                    not isinstance(route, dict)
                    or str(route.get("origin_poi_id") or "") != origin.poi_id
                    or canonical_route_evidence_status(route) != "provider_verified"
                ):
                    continue
                destination_name = normalize_entity_name(route.get("destination_name"))
                destination_id = str(route.get("destination_poi_id") or "").strip()
                destination = ctx.poi(destination_id) if destination_id else None
                if destination is None or not _is_grounded_route_endpoint(destination):
                    continue
                if destination_name == target:
                    repaired_endpoints.append((0, destination))
                elif destination_name.startswith(target):
                    repaired_endpoints.append((1, destination))
            if repaired_endpoints:
                endpoint = min(repaired_endpoints, key=lambda item: item[0])[1]
        if endpoint is None:
            endpoint = next((
                poi for poi in ctx.pois_by_id.values()
                if poi.poi_id != origin.poi_id
                and _is_grounded_route_endpoint(poi)
                and _endpoint_name_matches(poi, location)
            ), None)
        if endpoint is None:
            target = normalize_entity_name(location)
            session_candidates = sorted(
                (
                    poi for poi in ctx.pois_by_id.values()
                    if poi.poi_id != origin.poi_id
                    and _is_grounded_route_endpoint(poi)
                    and poi.category not in {"food", "hotel", "lodging"}
                    and poi.entity_type not in {"restaurant", "hotel", "lodging"}
                    and (
                        normalize_entity_name(poi.name).startswith(target)
                        or normalize_entity_name(poi.canonical_name).startswith(target)
                    )
                ),
                key=lambda poi: (
                    poi.entity_type not in {"district", "attraction", "transport"},
                    -float(poi.popularity or 0.0),
                    poi.poi_id,
                ),
            )
            endpoint = session_candidates[0] if session_candidates else None
        if endpoint is None:
            # Reuse provider-verified evidence already bound into this planning
            # transaction.  It may not have been promoted into ``pois_by_id``
            # when the user owns an area-only appointment, but it is still a
            # stronger endpoint than issuing another broad search.
            bound_candidates: list[POI] = []
            for kind, entries in domain_inputs.items():
                if kind == "transport" or not isinstance(entries, list):
                    continue
                for entry in entries:
                    payload = entry.get("payload", entry) if isinstance(entry, dict) else {}
                    if not isinstance(payload, dict):
                        continue
                    for field in ("pois", "restaurants", "hotels", "endpoint_pois"):
                        for raw in payload.get(field) or []:
                            try:
                                poi = normalize_poi_entity(poi_from_dict(raw), ctx.profile)
                            except (KeyError, TypeError, ValueError):
                                continue
                            if poi.poi_id != origin.poi_id and _is_grounded_route_endpoint(poi):
                                bound_candidates.append(poi)
            endpoint = next((
                poi for poi in bound_candidates
                if _endpoint_name_matches(poi, location)
            ), None)
            if endpoint is None:
                target = normalize_entity_name(location)
                endpoint = next((
                    poi for poi in bound_candidates
                    if normalize_entity_name(poi.name).startswith(target)
                    or normalize_entity_name(poi.canonical_name).startswith(target)
                ), None)
            if endpoint is not None:
                ctx.remember_pois([endpoint])
        if endpoint is None:
            try:
                candidates = ctx.provider.search_pois(
                    city=str(ctx.profile.destination or ""),
                    query_tags=[location],
                    category=None,
                    max_results=30,
                )
            except ProviderRateLimitError:
                raise
            except Exception:  # noqa: BLE001 - fail closed below
                candidates = []
            normalized = [normalize_poi_entity(item, ctx.profile) for item in candidates]
            endpoint = next((
                poi for poi in normalized
                if _is_grounded_route_endpoint(poi)
                and _endpoint_name_matches(poi, location)
            ), None)
            if endpoint is None:
                target = normalize_entity_name(location)
                endpoint = next((
                    poi for poi in normalized
                    if _is_grounded_route_endpoint(poi)
                    and normalize_entity_name(poi.name).startswith(target)
                ), None)
            if endpoint is not None:
                ctx.remember_pois([endpoint])
        if endpoint is None:
            continue
        existing = next((
            dict(route)
            for route in existing_routes
            if isinstance(route, dict)
            and str(route.get("origin_poi_id") or "") == origin.poi_id
            and str(route.get("destination_poi_id") or "") == endpoint.poi_id
            and canonical_route_evidence_status(route) == "provider_verified"
        ), None)
        if existing is not None:
            labelled = {
                **existing,
                "fixed_event_name": location,
                "required_buffer_min": int(
                    existing.get("required_buffer_min") or TRANSFER_BUFFER_MIN
                ),
            }
            labelled.setdefault(
                "recommended_latest_departure",
                _offset_clock(
                    event.get("start"),
                    -(
                        int(labelled.get("duration_min") or 0)
                        + int(labelled["required_buffer_min"])
                    ),
                ),
            )
            if labelled != existing:
                artifact_id = ctx.store.put("routes", labelled)
                domain_inputs.setdefault("transport", []).append({
                    "artifact_id": artifact_id,
                    "payload": labelled,
                    "post_plan_fixed_event_closure": True,
                })
                existing_routes.append(labelled)
            closed.append(labelled)
            continue
        route = _estimate_hard_route(
            route_estimator,
            origin,
            endpoint,
            ctx.profile.transport_mode,
            allow_taxi_fallback=True,
        )
        if route is None:
            continue
        if (
            route.evidence_status != "provider_verified"
            or not route_supports_endpoints(route, origin.poi_id, endpoint.poi_id)
        ):
            continue
        buffer_min = TRANSFER_BUFFER_MIN
        route_payload = {
            "origin_poi_id": origin.poi_id,
            "destination_poi_id": endpoint.poi_id,
            "origin_name": origin.name,
            "destination_name": endpoint.name,
            "fixed_event_name": location,
            "mode": route.mode,
            "duration_min": route.duration_min,
            "distance_km": route.distance_km,
            "source": route.source,
            "evidence_status": route.evidence_status,
            "walking_distance_km": route.walking_distance_km,
            "required_buffer_min": buffer_min,
            "recommended_latest_departure": _offset_clock(
                event.get("start"), -(route.duration_min + buffer_min)
            ),
        }
        artifact_id = ctx.store.put("routes", route_payload)
        domain_inputs.setdefault("transport", []).append({
            "artifact_id": artifact_id,
            "payload": route_payload,
            "post_plan_fixed_event_closure": True,
        })
        existing_routes.append(route_payload)
        closed.append(route_payload)
    return closed


def _build_required_route_anchors(
    profile: TravelProfile,
    itinerary: dict[str, Any],
    domain_inputs: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[CriticIssue]]:
    """Bind optional trip endpoint requirements to exact scheduled POI ids."""
    state = profile.constraint_state or {}
    origin_name = str(state.get("origin") or "").strip()
    return_name = str(state.get("return_location") or "").strip()
    events = [event for event in (state.get("fixed_events") or []) if isinstance(event, dict)]
    if not origin_name and not return_name and not events:
        return None, []

    routes = [
        entry.get("payload", entry)
        for entry in (domain_inputs.get("transport") or [])
        if isinstance(entry, dict)
    ]

    def route_for(origin_id: str | None, destination_id: str | None) -> dict[str, Any] | None:
        if not origin_id or not destination_id:
            return None
        matches = [
            (index, dict(route)) for index, route in enumerate(routes)
            if isinstance(route, dict)
            and str(route.get("origin_poi_id") or "") == origin_id
            and str(route.get("destination_poi_id") or "") == destination_id
        ]
        if not matches:
            return None
        evidence_rank = {
            "unavailable": 0,
            "haversine_estimate": 1,
            "deterministic_estimate": 2,
            "provider_verified": 3,
        }
        # A repair appends a stronger route for the same endpoints.  Select
        # the strongest evidence and, on ties, the newest Artifact instead of
        # letting an older fallback permanently shadow the repair.
        return max(
            matches,
            key=lambda item: (
                evidence_rank.get(canonical_route_evidence_status(item[1]), 0),
                item[0],
            ),
        )[1]

    def endpoint_id(name: str) -> str | None:
        target = normalize_entity_name(name)
        for route in routes:
            if not isinstance(route, dict):
                continue
            if normalize_entity_name(route.get("return_location_name")) == target:
                return str(route.get("destination_poi_id") or "") or None
            if normalize_entity_name(route.get("origin_name")) == target:
                return str(route.get("origin_poi_id") or "") or None
            if normalize_entity_name(route.get("destination_name")) == target:
                return str(route.get("destination_poi_id") or "") or None
        return None

    days = [day for day in (itinerary.get("days") or []) if day.get("stops")]
    legs: list[dict[str, Any]] = []
    issues: list[CriticIssue] = []
    if origin_name and days:
        origin_id = endpoint_id(origin_name)
        first_id = str(days[0]["stops"][0].get("poi", {}).get("poi_id") or "")
        route = route_for(origin_id, first_id)
        legs.append({
            "kind": "trip_origin_to_first_stop",
            "required_name": origin_name,
            "origin_poi_id": origin_id,
            "destination_poi_id": first_id,
            "evidence_status": canonical_route_evidence_status(route),
            "route": route,
        })
    if return_name and days:
        last_id = str(days[-1]["stops"][-1].get("poi", {}).get("poi_id") or "")
        return_id = endpoint_id(return_name)
        route = route_for(last_id, return_id)
        status = canonical_route_evidence_status(route)
        legs.append({
            "kind": "last_stop_to_return_location",
            "required_name": return_name,
            "origin_poi_id": last_id,
            "destination_poi_id": return_id,
            "evidence_status": status,
            "mode": route.get("mode") if route else None,
            "duration_min": route.get("duration_min") if route else None,
            "distance_km": route.get("distance_km") if route else None,
            "source": route.get("source") if route else None,
            "return_deadline": state.get("return_deadline"),
            "recommended_latest_departure": (
                route.get("recommended_latest_departure") if route else None
            ),
            "required_buffer_min": (
                route.get("required_buffer_min") if route else None
            ),
            "route": route,
        })
        if state.get("return_deadline") and status not in {
            "provider_verified", "deterministic_estimate"
        }:
            issues.append(CriticIssue(
                code="return_route_evidence_insufficient",
                message=f"最后一站到返程地点 {return_name} 缺少可证明截止时间的路线证据。",
                severity="error",
            ))
    for event in events:
        location = str(event.get("location") or "").strip()
        if not location or location in {"自由活动", "休息", "自由时间"}:
            continue
        is_unspecified = normalize_entity_name(location) in {
            normalize_entity_name(item)
            for item in state.get("user_owned_unspecified_fixed_event_locations") or []
            if normalize_entity_name(item)
        }
        event_poi_id = next((
            str(stop.get("poi", {}).get("poi_id") or "")
            for day in days for stop in (day.get("stops") or [])
            if normalize_entity_name(stop.get("poi", {}).get("canonical_name") or stop.get("poi", {}).get("name"))
            == normalize_entity_name(location)
        ), None)
        if event_poi_id is None and is_unspecified:
            event_poi_id = next((
                str(route.get("destination_poi_id") or "") or None
                for route in routes
                if isinstance(route, dict)
                and normalize_entity_name(route.get("fixed_event_name"))
                == normalize_entity_name(location)
            ), None)
        if event_poi_id is None and is_unspecified:
            target = normalize_entity_name(location)
            endpoint_candidates: list[tuple[int, str]] = []
            for route in routes:
                if not isinstance(route, dict):
                    continue
                for side in ("destination", "origin"):
                    name = normalize_entity_name(route.get(f"{side}_name"))
                    poi_id = str(route.get(f"{side}_poi_id") or "").strip()
                    if not name or not poi_id:
                        continue
                    if name == target:
                        endpoint_candidates.append((0, poi_id))
                    elif name.startswith(target):
                        endpoint_candidates.append((1, poi_id))
            if endpoint_candidates:
                event_poi_id = min(endpoint_candidates, key=lambda item: item[0])[1]
        event_origin_id = None
        try:
            event_day = int(event.get("day") or 0)
            parsed_start = time.fromisoformat(str(event.get("start")))
            event_start_minutes = parsed_start.hour * 60 + parsed_start.minute
        except (TypeError, ValueError):
            event_day = 0
            event_start_minutes = -1
        preceding: list[tuple[int, str]] = []
        for day in days:
            if int(day.get("day_index") or 0) != event_day:
                continue
            for stop in day.get("stops") or []:
                try:
                    stop_start = time.fromisoformat(str(stop.get("start_time")))
                    stop_end = (
                        stop_start.hour * 60
                        + stop_start.minute
                        + int(stop.get("duration_min") or 0)
                    )
                except (TypeError, ValueError):
                    continue
                stop_id = str(stop.get("poi", {}).get("poi_id") or "").strip()
                if stop_id and stop_end <= event_start_minutes:
                    preceding.append((stop_end, stop_id))
        if preceding:
            event_origin_id = max(preceding, key=lambda item: item[0])[1]
        event_routes = [
            route for route in routes
            if isinstance(route, dict)
            and event_poi_id
            and event_poi_id in {
                str(route.get("origin_poi_id") or ""),
                str(route.get("destination_poi_id") or ""),
            }
            and (
                not is_unspecified
                or normalize_entity_name(route.get("fixed_event_name"))
                == normalize_entity_name(location)
            )
        ]
        verified_routes = [
            route for route in event_routes
            if canonical_route_evidence_status(route) == "provider_verified"
        ]
        verified = bool(verified_routes)
        legs.append({
            "kind": "fixed_event_transfer",
            "required_name": location,
            "origin_poi_id": event_origin_id,
            "event_poi_id": event_poi_id,
            "evidence_status": "provider_verified" if verified else "unavailable",
            "routes": verified_routes,
        })
        if event_poi_id and not verified:
            issues.append(CriticIssue(
                code="fixed_event_route_evidence_insufficient",
                message=f"固定预约 {location} 的前往或离开路线未取得高可信证据。",
                severity="error",
            ))
    anchors: dict[str, Any] = {
        "status": "bound_required_endpoints",
        "legs": legs,
    }
    for leg in legs:
        kind = str(leg.get("kind") or "")
        if kind:
            anchors[kind] = leg
    return anchors, issues


def _build_lodging_route_anchors(
    profile: TravelProfile,
    itinerary: dict[str, Any],
    lodging_plan: dict[str, Any] | None,
    route_estimator: Any,
) -> dict[str, Any] | None:
    """Bind a concrete selected hotel to each day's first and last actual POI."""
    if not isinstance(lodging_plan, dict):
        return None
    if lodging_plan.get("status") != "recommended_not_booked":
        return {
            "status": "area_anchor_only" if profile.hotel_area else "not_required",
            "daily_routes": [],
        }
    hotel = lodging_plan.get("hotel") or {}
    if not all(hotel.get(key) is not None for key in ("hotel_id", "lat", "lng")):
        return {"status": "evidence_unavailable", "daily_routes": []}
    hotel_poi = POI(
        poi_id=str(hotel["hotel_id"]),
        name=str(hotel.get("name") or "住宿地"),
        city=str(hotel.get("city") or profile.destination or ""),
        category="hotel",
        lat=float(hotel["lat"]),
        lng=float(hotel["lng"]),
        rating=float(hotel.get("rating") or 0),
        popularity=0.0,
        tags=["hotel"],
        estimated_duration_min=0,
        price_level=str(hotel.get("budget_level") or profile.budget_level or "mid"),
        address=hotel.get("address"),
        source=str(hotel.get("source") or "unknown"),
        canonical_name=str(hotel.get("name") or "住宿地"),
        entity_type="hotel",
        source_poi_id=str(hotel["hotel_id"]),
    )
    daily_routes: list[dict[str, Any]] = []
    for day in itinerary.get("days") or []:
        stops = day.get("stops") or []
        if not stops:
            continue
        try:
            first = poi_from_dict(stops[0]["poi"])
            last = poi_from_dict(stops[-1]["poi"])
        except Exception:
            continue
        legs: list[dict[str, Any]] = []
        for origin, destination, position in (
            (hotel_poi, first, "hotel_to_first_stop"),
            (last, hotel_poi, "last_stop_to_hotel"),
        ):
            try:
                route = normalize_route_evidence(
                    route_estimator.estimate_route(origin, destination, profile.transport_mode)
                )
            except Exception:
                route = RouteInfo(
                    origin_poi_id=origin.poi_id,
                    destination_poi_id=destination.poi_id,
                    distance_km=0,
                    duration_min=0,
                    mode=profile.transport_mode,
                    source="unavailable",
                    evidence_status="unavailable",
                )
            if not route_supports_endpoints(route, origin.poi_id, destination.poi_id):
                route = RouteInfo(
                    origin_poi_id=origin.poi_id,
                    destination_poi_id=destination.poi_id,
                    distance_km=0,
                    duration_min=0,
                    mode=profile.transport_mode,
                    source="unavailable",
                    evidence_status="unavailable",
                )
            legs.append({"position": position, **route.__dict__})
        first_start = str(stops[0].get("start_time") or "")
        outbound_duration = int(legs[0].get("duration_min") or 0)
        recommended_departure = _offset_clock(
            first_start, -(outbound_duration + 15)
        ) if first_start and legs[0].get("evidence_status") != "unavailable" else None
        daily_routes.append({
            "day_index": day.get("day_index"),
            "recommended_hotel_departure": recommended_departure,
            "legs": legs,
        })
    outbound_distances = [
        float(day["legs"][0].get("distance_km") or 0)
        for day in daily_routes if day.get("legs")
    ]
    main_area_conflict = bool(
        outbound_distances
        and sum(outbound_distances) / len(outbound_distances) > LODGING_ACTIVITY_AREA_CONFLICT_KM
    )
    return {
        "status": "concrete_hotel_anchored",
        "hotel_poi_id": hotel_poi.poi_id,
        "main_activity_area_conflict": main_area_conflict,
        "average_outbound_distance_km": (
            round(sum(outbound_distances) / len(outbound_distances), 2)
            if outbound_distances else None
        ),
        "daily_routes": daily_routes,
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
    base_hotel = _known_cost(estimate, "hotel")
    selected_hotel = (
        float(lodging_plan.get("lodging_subtotal_cny") or 0)
        if isinstance(lodging_plan, dict) and lodging_plan.get("status") == "recommended_not_booked"
        else base_hotel
    )
    selected_hotel = float(selected_hotel or 0.0)
    base_hotel_for_delta = float(base_hotel or 0.0)
    state = profile.constraint_state or {}
    people = int(
        state.get("traveler_count")
        or profile.party_size
        or 1
    )
    intercity = _intercity_transport_allowance(domain_inputs, people)
    days = int(state.get("duration_days") or profile.days or 1)
    mobility_sensitive = bool(
        state.get("elderly")
        or state.get("accessibility_priority")
        or state.get("max_walking_km_per_day") is not None
        or any(
            marker in str(item).casefold()
            for item in (
                [state.get("avoid")]
                if isinstance(state.get("avoid"), str)
                else (state.get("avoid") or [])
            )
            for marker in ("爬坡", "楼梯", "台阶", "wheelchair", "stairs", "steps")
        )
    )
    # The mobility plan requires point-to-point taxi fallback whenever public
    # transport walking detail is unknown or exceeds the cap. Reserve a modest,
    # explicit daily allowance so that safety advice and the budget cannot
    # contradict one another.
    mobility_fallback_transport = float(80 * days) if mobility_sensitive else 0.0
    base_low = float(estimate.get("total_low") or 0)
    base_high = float(estimate.get("total_high") or 0)
    delta = selected_hotel - base_hotel_for_delta
    total_low = round(max(
        0.0, base_low + delta + intercity["low"] + mobility_fallback_transport
    ), 2)
    total_high = round(max(
        total_low, base_high + delta + intercity["high"] + mobility_fallback_transport
    ), 2)
    if state.get("budget_remaining_cny") is not None:
        limit = float(state["budget_remaining_cny"])
        limit_basis = "remaining_budget_excludes_prepaid"
    elif state.get("budget_max_cny") is not None:
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
    prepaid = float(state.get("prepaid_lodging_cny") or 0)
    remaining_basis = limit_basis == "remaining_budget_excludes_prepaid"
    lodging_cost = 0.0 if remaining_basis and prepaid else selected_hotel
    fixed_event_cost = _known_cost(estimate, "fixed_event_cost")
    fixed_events = [
        event for event in (state.get("fixed_events") or []) if isinstance(event, dict)
    ]
    user_owned_locations = {
        normalize_entity_name(location)
        for location in (state.get("user_owned_unspecified_fixed_event_locations") or [])
        if normalize_entity_name(location)
    }
    external_commitment_costs_excluded: list[str] = []
    if fixed_event_cost is None:
        explicit_costs = [event.get("cost_cny") for event in fixed_events]
        all_costs_explicit = bool(fixed_events) and all(cost is not None for cost in explicit_costs)
        no_incremental_cost = all(
            normalize_entity_name(event.get("location")) in user_owned_locations
            or str(event.get("location") or "").strip() in {"自由活动", "休息", "自由时间"}
            for event in fixed_events
        )
        if all_costs_explicit:
            fixed_event_cost = sum(float(cost or 0) for cost in explicit_costs)
        elif not fixed_events or no_incremental_cost:
            fixed_event_cost = 0.0
            external_commitment_costs_excluded = [
                str(event.get("location") or "用户自有安排") for event in fixed_events
            ]
    contingency = _known_cost(estimate, "contingency")
    if contingency is None and estimate.get("total_high") is not None:
        contingency = 0.0
    inner_city_transport = _known_cost(estimate, "inner_city_transport")
    breakdown = {
        "lodging": lodging_cost,
        "transport": (
            round(
                inner_city_transport
                + intercity["expected"]
                + mobility_fallback_transport,
                2,
            )
            if inner_city_transport is not None
            else None
        ),
        "tickets": _known_cost(estimate, "tickets"),
        "meals": _known_cost(estimate, "meals"),
        "fixed_event_cost": fixed_event_cost,
        "contingency": contingency,
    }
    unknown_items = [key for key, value in breakdown.items() if value is None]
    known_breakdown = {key: value for key, value in breakdown.items() if value is not None}
    expected_total = round(sum(known_breakdown.values()), 2) if not unknown_items else None
    status = "estimate"
    note = (
        "城际交通按已验证路线距离计入预算余量，属于非实时票价估算；"
        "实时房价、实际票价及个人购物仍需复核。"
        if intercity["expected"]
        else "估算不含未提供价格的城际票、实时房价波动及个人购物。"
    )
    if mobility_fallback_transport:
        note += (
            f" 已为行动不便/步行上限约束预留每日80元、共"
            f"{mobility_fallback_transport:.0f}元点到点出租车兜底。"
        )
    if limit is not None and expected_total is not None and expected_total > limit and total_low <= limit:
        # The provider already supplies an estimate interval.  When the normal
        # scenario misses a hard cap but its grounded low scenario fits, expose
        # that scenario explicitly instead of either hiding the overrun or
        # inventing cheaper individual prices.
        variable_total = sum(value for key, value in known_breakdown.items() if key != "lodging")
        variable_target = max(0.0, total_low - selected_hotel)
        scale = min(1.0, variable_target / variable_total) if variable_total else 0.0
        for key in ("meals", "tickets", "transport"):
            if breakdown.get(key) is not None:
                scalable = float(breakdown[key])
                protected = mobility_fallback_transport if key == "transport" else 0.0
                breakdown[key] = round(
                    protected + max(0.0, scalable - protected) * scale,
                    2,
                )
        expected_total = round(sum(value for value in breakdown.values() if value is not None), 2)
        status = "budget_optimized_low_scenario"
        note = (
            "硬预算下采用工具估算区间的低位方案；餐饮、门票与市内交通需按预算卡"
            "分项上限执行，实时价格上涨时应删减可选项目。"
        )
    from travel_agent.artifact_policy import constraint_version

    active_version = constraint_version(profile)
    return {
        "status": status,
        "currency": "CNY",
        "people": people,
        "days": days,
        "constraint_revision": active_version["revision"],
        "constraint_hash": active_version["constraint_hash"],
        "prepaid_cost": prepaid,
        "lodging": lodging_cost,
        "transport": breakdown["transport"],
        "mobility_fallback_transport_cny": mobility_fallback_transport,
        "intercity_transport_estimate_cny": intercity["expected"],
        "intercity_transport_low_cny": intercity["low"],
        "intercity_transport_high_cny": intercity["high"],
        "intercity_estimate_method": (
            "verified_distance_allowance_not_live_fare"
            if intercity["expected"] else None
        ),
        "intercity_route_artifact_ids": intercity["artifact_ids"],
        "tickets": breakdown["tickets"],
        "meals": breakdown["meals"],
        "fixed_event_cost": breakdown["fixed_event_cost"],
        "contingency": breakdown["contingency"],
        "breakdown_cny": breakdown,
        "total_low_cny": total_low,
        "total_expected_cny": expected_total,
        "total_high_cny": total_high,
        "expected_total": expected_total,
        "high_total": total_high if not unknown_items else None,
        "user_limit_cny": limit,
        "user_limit_basis": limit_basis,
        "within_user_limit": expected_total <= limit if limit is not None and expected_total is not None else None,
        "risk_high_exceeds_limit": total_high > limit if limit is not None else None,
        "source_artifact_id": entry.get("artifact_id"),
        "unknown_items": unknown_items,
        "external_commitment_costs_excluded": external_commitment_costs_excluded,
        "note": note,
    }


def _known_cost(estimate: dict[str, Any], key: str) -> float | None:
    value = estimate.get(key)
    return float(value) if value is not None else None


def _intercity_transport_allowance(
    domain_inputs: dict[str, Any], people: int
) -> dict[str, Any]:
    """Estimate a transparent fare band from verified long-distance legs.

    AMap supplies route distance and duration but not rail inventory or fare.
    The budget therefore uses a labelled distance allowance, never presents it
    as a quoted ticket price, and retains the source artifact ids for review.
    """
    legs: list[tuple[float, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in domain_inputs.get("transport") or []:
        if not isinstance(entry, dict) or not (
            entry.get("post_plan_origin_closure")
            or entry.get("post_plan_endpoint_closure")
        ):
            continue
        payload = entry.get("payload", entry)
        if not isinstance(payload, dict):
            continue
        if canonical_route_evidence_status(payload) != "provider_verified":
            continue
        if str(payload.get("mode") or "") != "public_transport":
            continue
        try:
            distance_km = float(payload.get("distance_km") or 0)
        except (TypeError, ValueError):
            continue
        # Below this range the provider's normal inner-city allowance remains
        # the better category and avoids double counting local transfers.
        if distance_km < 50:
            continue
        key = (
            str(payload.get("origin_poi_id") or ""),
            str(payload.get("destination_poi_id") or ""),
            str(payload.get("mode") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        legs.append((distance_km, str(entry.get("artifact_id") or "")))
    multiplier = max(1, int(people or 1))
    return {
        "low": round(sum(max(20.0, km * 0.35) for km, _ in legs) * multiplier, 2),
        "expected": round(sum(max(30.0, km * 0.50) for km, _ in legs) * multiplier, 2),
        "high": round(sum(max(50.0, km * 0.75) for km, _ in legs) * multiplier, 2),
        "artifact_ids": [artifact_id for _, artifact_id in legs if artifact_id],
    }


def _change_claim_matches(requested: str, applied: str) -> bool:
    """Require meaningful lexical overlap before calling a repair applied."""
    requested_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_:.-]+", requested.lower()))
    applied_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_:.-]+", applied.lower()))
    return bool(requested_tokens and requested_tokens.intersection(applied_tokens))


def _build_mobility_plan(
    profile: TravelProfile,
    itinerary: dict[str, Any],
) -> dict[str, Any] | None:
    """Represent a walking cap without pretending unknown transit walks are zero."""
    state = profile.constraint_state or {}
    cap = state.get("max_walking_km_per_day")
    raw_avoid = state.get("avoid") or []
    avoid_values = [raw_avoid] if isinstance(raw_avoid, str) else list(raw_avoid)
    terrain_avoidance = [
        str(item)
        for item in avoid_values
        if any(
            marker in str(item).casefold()
            for marker in (
                "爬坡", "上坡", "坡道", "楼梯", "台阶",
                "hill", "slope", "stairs", "steps",
            )
        )
    ]
    if cap is None and not terrain_avoidance:
        return None
    numeric_cap = float(cap) if cap is not None else None
    days: list[dict[str, Any]] = []
    venue_internal_access: list[dict[str, Any]] = []
    for day in itinerary.get("days") or []:
        known = 0.0
        unknown_legs: list[str] = []
        for stop in day.get("stops") or []:
            route = stop.get("route_from_previous") or {}
            if not route:
                continue
            distance = route.get("walking_distance_km")
            is_verified_point_to_point = (
                route.get("mode") in {"taxi", "drive"}
                and canonical_route_evidence_status(route) == "provider_verified"
            )
            if distance is None and is_verified_point_to_point:
                distance = 0.0
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
            "taxi_fallback_required": bool(unknown_legs) or (
                numeric_cap is not None and known > numeric_cap
            ),
        })
        if terrain_avoidance:
            for stop in day.get("stops") or []:
                poi = stop.get("poi") or {}
                if not isinstance(poi, dict) or not poi.get("poi_id"):
                    continue
                venue_internal_access.append({
                    "day_index": day.get("day_index"),
                    "poi_id": poi.get("poi_id"),
                    "name": poi.get("name"),
                    "evidence_status": "unverified",
                    "required_action": "verify_step_free_access_before_departure",
                    "fallback": "replace_candidate",
                })
    route_policy = (
        "已知步行距离计入每日上限；公交接驳步行距离未知或累计可能超限的区段，"
        "必须改用点到点出租车/网约车，并在出发前用地图复核。"
    )
    internal_access_policy = (
        "景区内部无坡度、台阶或无障碍证据时不承诺可达，必须先核验无台阶入口/电梯；"
        "不能确认时替换候选，禁止走连续爬坡或长楼梯。"
        if terrain_avoidance
        else ""
    )
    return {
        "required": True,
        "status": "bounded_with_taxi_fallback",
        "max_walking_km_per_day": numeric_cap,
        "avoidance_requirements": terrain_avoidance,
        "days": days,
        "venue_internal_access": venue_internal_access,
        "policy": route_policy + internal_access_policy,
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
    if not deadline or not destination:
        return None
    try:
        from travel_agent.constraint_events import clock_minutes

        deadline_minutes = clock_minutes(deadline)
    except (TypeError, ValueError):
        return None
    same_city_terminal = bool(trip_city and trip_city in destination)
    buffer_min = 60 if same_city_terminal else 180
    cutoff_minutes = max(0, deadline_minutes - buffer_min)
    transport = domain_inputs.get("transport") or []
    routes = [
        entry.get("payload", entry)
        for entry in transport
        if isinstance(entry, dict)
    ]
    # A return transfer must start at the actual final stop.  Merely requiring
    # any scheduled origin can resurrect a stale route after shortlist pruning
    # or reviewer revision.
    scheduled_days = [
        day for day in itinerary.get("days") or []
        if isinstance(day, dict) and day.get("stops")
    ]
    final_stop_id = str(
        ((scheduled_days[-1].get("stops") or [])[-1].get("poi") or {}).get("poi_id")
        if scheduled_days else ""
    )
    terminal_route = next(
        (
            route
            for route in routes
            if isinstance(route, dict)
            and final_stop_id
            and str(route.get("origin_poi_id") or "") == final_stop_id
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
    same_city_return_route = next(
        (
            route
            for route in routes
            if isinstance(route, dict)
            and final_stop_id
            and str(route.get("origin_poi_id") or "") == final_stop_id
            and destination in str(route.get("destination_name") or "")
            and destination not in str(route.get("origin_name") or "")
        ),
        None,
    )
    terminal_verified = bool(
        terminal_route
        and str(terminal_route.get("evidence_status") or evidence_status_for_source(terminal_route.get("source"))) in {
            "provider_verified", "deterministic_estimate"
        }
    )
    return_verified = bool(
        return_route
        and str(return_route.get("evidence_status") or evidence_status_for_source(return_route.get("source"))) in {
            "provider_verified", "deterministic_estimate"
        }
    )
    same_city_return_verified = bool(
        same_city_return_route
        and str(
            same_city_return_route.get("evidence_status")
            or evidence_status_for_source(same_city_return_route.get("source"))
        ) in {"provider_verified", "deterministic_estimate"}
    )
    return {
        "required": True,
        "from_city": trip_city,
        "to_location": destination,
        "arrival_deadline": deadline,
        "activity_cutoff": f"{cutoff_minutes // 60:02d}:{cutoff_minutes % 60:02d}",
        "mode": "public_transport",
        "terminal_transfer": (
            same_city_return_route
            if same_city_terminal and same_city_return_verified
            else terminal_route if terminal_verified else None
        ),
        "intercity_segment": None if same_city_terminal else {
            "status": "verified_route" if return_verified else "requires_live_verification",
            "route_evidence": return_route,
            "instruction": (
                f"选择可在 {deadline} 前抵达{destination}的城际班次，"
                "并在购票平台核验实时班次、余票和检票时间。"
            ),
        },
        "buffer_min": buffer_min,
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
    preserve_terminal_poi_id = str(
        directives.get("preserve_terminal_poi_id") or ""
    ).strip()
    if not indoor_days and not reviewer_issue_types and not preserve_terminal_poi_id:
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
    indoor_changed = False
    terminal_preserved = False
    terminal_audit: dict[str, Any] | None = None
    if preserve_terminal_poi_id:
        terminal_audit = {
            "requested_poi_id": preserve_terminal_poi_id,
            "source": None,
            "candidate_legal": False,
            "displaced_must_visit": None,
            "applied": False,
        }
        directives["terminal_preservation_audit"] = terminal_audit
    if preserve_terminal_poi_id and day_stops and day_stops[-1]:
        terminal_slot: tuple[int, int] | None = next((
            (day_index, stop_index)
            for day_index, stops in enumerate(day_stops)
            for stop_index, stop in enumerate(stops)
            if stop.poi.poi_id == preserve_terminal_poi_id
        ), None)
        last_slot = (len(day_stops) - 1, len(day_stops[-1]) - 1)
        if terminal_slot is not None and terminal_slot != last_slot:
            source_day, source_stop = terminal_slot
            last_day, last_stop = last_slot
            terminal = day_stops[source_day][source_stop]
            displaced = day_stops[last_day][last_stop]
            terminal_legal = (
                is_verified_plannable_poi(terminal.poi)
                and poi_avoid_match(terminal.poi, ctx.profile) is None
            )
            if terminal_legal and all(
                item.poi.poi_id != terminal.poi.poi_id for item in ranked
            ):
                ranked.append(ScoredPOI(
                    poi=terminal.poi,
                    score=min((item.score for item in ranked), default=0.0),
                    reasons=["保留已核验返程路线的末站端点"],
                ))
            # Keep the schedule slots stable while restoring the exact POI
            # endpoint whose route evidence the Transport repair just bound.
            # Route legs are rebound after this directive is applied.
            if terminal_legal:
                day_stops[source_day][source_stop] = replace(
                    displaced,
                    start_time=terminal.start_time,
                    route_from_previous=None,
                )
                day_stops[last_day][last_stop] = replace(
                    terminal,
                    start_time=displaced.start_time,
                    route_from_previous=None,
                )
                changed = True
                terminal_preserved = True
            terminal_audit.update({
                "source": "rebuilt_itinerary",
                "candidate_legal": terminal_legal,
                "displaced_must_visit": False,
                "applied": terminal_legal,
            })
        elif terminal_slot is None:
            reviewed_terminal = next((
                item.poi for item in ranked
                if item.poi.poi_id == preserve_terminal_poi_id
                and is_verified_plannable_poi(item.poi)
            ), None)
            if reviewed_terminal is not None:
                terminal_audit["source"] = "ranked"
                terminal_audit["candidate_legal"] = True
            if reviewed_terminal is None:
                session_terminal = ctx.poi(preserve_terminal_poi_id)
                if (
                    session_terminal is not None
                    and is_verified_plannable_poi(session_terminal)
                    and poi_avoid_match(session_terminal, ctx.profile) is None
                ):
                    reviewed_terminal = session_terminal
                    terminal_audit["source"] = "session_cache"
                    terminal_audit["candidate_legal"] = True
            if reviewed_terminal is None:
                directive_raw = directives.get("preserve_terminal_poi")
                try:
                    directive_terminal = normalize_poi_entity(
                        poi_from_dict(directive_raw), ctx.profile
                    ) if isinstance(directive_raw, dict) else None
                except (KeyError, TypeError, ValueError):
                    directive_terminal = None
                if (
                    directive_terminal is not None
                    and directive_terminal.poi_id == preserve_terminal_poi_id
                    and is_verified_plannable_poi(directive_terminal)
                    and poi_avoid_match(directive_terminal, ctx.profile) is None
                ):
                    reviewed_terminal = directive_terminal
                    ctx.remember_pois([directive_terminal])
                    terminal_audit["source"] = "reviewed_parent_snapshot"
                    terminal_audit["candidate_legal"] = True
            if reviewed_terminal is None:
                parent_id = str(
                    directives.get("parent_plan_artifact_id") or ""
                ).strip()
                parent_payload = ctx.store.get(parent_id) if parent_id else None
                parent_stops = [
                    stop
                    for day in (
                        ((parent_payload or {}).get("itinerary") or {}).get("days")
                        or []
                    )
                    for stop in day.get("stops") or []
                    if isinstance(stop, dict)
                ]
                parent_raw = next((
                    stop.get("poi")
                    for stop in parent_stops
                    if str((stop.get("poi") or {}).get("poi_id") or "")
                    == preserve_terminal_poi_id
                ), None)
                try:
                    parent_terminal = normalize_poi_entity(
                        poi_from_dict(parent_raw), ctx.profile
                    ) if isinstance(parent_raw, dict) else None
                except (KeyError, TypeError, ValueError):
                    parent_terminal = None
                if (
                    parent_terminal is not None
                    and is_verified_plannable_poi(parent_terminal)
                    and poi_avoid_match(parent_terminal, ctx.profile) is None
                ):
                    reviewed_terminal = parent_terminal
                    ctx.remember_pois([parent_terminal])
                    terminal_audit["source"] = "parent_artifact"
                    terminal_audit["candidate_legal"] = True
            if (
                reviewed_terminal is not None
                and all(
                    item.poi.poi_id != reviewed_terminal.poi_id
                    for item in ranked
                )
            ):
                # Keep the restored endpoint in the verified selection
                # universe so the post-directive grounding filter does
                # not immediately remove it again.
                ranked.append(ScoredPOI(
                    poi=reviewed_terminal,
                    score=min((item.score for item in ranked), default=0.0),
                    reasons=["保留已核验返程路线的末站端点"],
                ))
            displaced = day_stops[-1][-1]
            displaced_is_required = any(
                poi_covers_requirement(displaced.poi, requirement)
                for requirement in (ctx.profile.must_visit or [])
            )
            terminal_audit["displaced_must_visit"] = displaced_is_required
            if reviewed_terminal is not None and not displaced_is_required:
                # The deterministic rebuild may trim one optional POI and
                # thereby discard the exact terminal endpoint repaired by
                # Transport. Restore that already-reviewed, provider-grounded
                # endpoint, but never evict a must-visit to do so.
                day_stops[-1][-1] = replace(
                    displaced,
                    poi=reviewed_terminal,
                    duration_min=reviewed_terminal.estimated_duration_min,
                    note="保留已核验返程路线的末站",
                    route_from_previous=None,
                )
                changed = True
                terminal_preserved = True
                terminal_audit["applied"] = True
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
                indoor_changed = True
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
            indoor_changed = True
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
    reviewer_revision_notes: list[str] = []
    if reviewer_issue_types:
        # Reviewer findings must be applied to the rebuilt artifact, not merely
        # acknowledged in revision prose.  Re-run the same deterministic,
        # general-purpose reviser against matching current issues before this
        # candidate can be promoted and rebound to final routes.
        from travel_agent.schemas import CriticResult
        from travel_agent.reviser import revise_itinerary

        matching_issues = [
            issue
            for issue in critic_result.issues
            if issue.code.casefold() in reviewer_issue_types
        ]
        if matching_issues:
            itinerary, critic_result, reviewer_revision_notes = revise_itinerary(
                itinerary,
                ranked,
                ctx.profile,
                CriticResult(passed=False, issues=matching_issues),
            )
    return replace(
        result,
        itinerary=itinerary,
        critic_result=critic_result,
        revision_notes=list(result.revision_notes)
        + (["按要求将指定日期调整为室内活动"] if indoor_changed else [])
        + (["保留已核验返程路线的末站端点"] if terminal_preserved else [])
        + reviewer_revision_notes,
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
    result = {
        "pois": [_scored_to_dict(item) for item in ranked],
        "source_artifact_ids": [previous["artifact_id"]],
    }
    contract = previous["payload"].get("soft_preference_actuation")
    if isinstance(contract, dict):
        from travel_agent.artifact_policy import constraint_version
        from travel_agent.hybrid_planning.soft_preference_actuation import (
            ranked_from_contract,
        )

        active_version = constraint_version(ctx.profile)
        if contract.get("constraint_hash") == active_version["constraint_hash"]:
            ranked = ranked_from_contract(ranked, contract)
            result["pois"] = [_scored_to_dict(item) for item in ranked]
            result["soft_preference_actuation"] = contract
    return result


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
        "required_route_anchors",
        "lodging_plan",
        "lodging_route_anchors",
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

    structured_interests = (profile.constraint_state or {}).get("interests") or []
    if isinstance(structured_interests, str):
        structured_interests = [structured_interests]

    def explicitly_requested_food_venue(poi: POI) -> bool:
        if poi.category != "food":
            return False
        from travel_agent.critic import poi_matches_interest

        generic_food_terms = {"food", "美食", "餐厅", "当地美食", "本地美食"}
        return any(
            str(term).strip() not in generic_food_terms
            and poi_matches_interest(poi, str(term))
            for term in structured_interests
            if str(term).strip()
        )

    pois: list[POI] = []
    preserved_ranked: list[ScoredPOI] = []
    actuation = ranked_artifact.get("soft_preference_actuation")
    candidate_hard_gate = None
    if isinstance(actuation, dict):
        from travel_agent.hybrid_planning.soft_preference_actuation import (
            candidate_hard_gate as _actuation_candidate_hard_gate,
        )

        candidate_hard_gate = _actuation_candidate_hard_gate
    for raw in ranked_artifact.get("pois") or []:
        try:
            scored = _scored_from_dict(raw)
            poi = normalize_poi_entity(scored.poi, profile)
        except Exception:
            continue
        if poi.category != "food" and not is_verified_plannable_poi(poi):
            continue
        if (
            poi.category == "food"
            and poi.poi_id not in restaurant_ids
            and not explicitly_requested_food_venue(poi)
        ):
            continue
        if poi_avoid_match(poi, profile) is not None:
            continue
        if candidate_hard_gate is not None and not candidate_hard_gate(
            poi, profile, []
        )["passed"]:
            continue
        pois.append(poi)
        if isinstance(actuation, dict):
            preserved_ranked.append(
                ScoredPOI(poi=poi, score=scored.score, reasons=list(scored.reasons))
            )
    for record in records:
        if record["kind"] not in {"candidates", "restaurants"}:
            continue
        payload = record["payload"]
        for raw in payload.get("pois") or payload.get("restaurants") or []:
            try:
                poi = normalize_poi_entity(poi_from_dict(raw), profile)
            except Exception:
                continue
            if (
                poi.category == "food"
                and poi.poi_id not in restaurant_ids
                and not explicitly_requested_food_venue(poi)
            ):
                continue
            if poi.category != "food" and not is_verified_plannable_poi(poi):
                continue
            if _is_unavailable_or_infrastructure_poi(poi, profile):
                continue
            if poi_avoid_match(poi, profile) is not None:
                continue
            if candidate_hard_gate is not None and not candidate_hard_gate(
                poi, profile, [str(record.get("artifact_id") or "")]
            )["passed"]:
                continue
            if _poi_city_allowed_for_plan(poi, profile):
                pois.append(poi)
    deduped = _dedupe_plannable_entities(pois, profile)
    if not isinstance(actuation, dict):
        return score_pois(deduped, profile)

    from travel_agent.hybrid_planning.soft_preference_actuation import ranked_from_contract

    preserved_ids = {item.poi.poi_id for item in preserved_ranked}
    extras = score_pois(
        [poi for poi in deduped if poi.poi_id not in preserved_ids], profile
    )
    return ranked_from_contract([*preserved_ranked, *extras], actuation)


def _prioritize_ranked_near_lodging(
    ranked: list[ScoredPOI],
    lodging_plan: dict[str, Any] | None,
    profile: TravelProfile,
) -> list[ScoredPOI]:
    """Keep remote city-edge candidates from displacing coherent day options."""
    hotel = (lodging_plan or {}).get("hotel") or {}
    try:
        hotel_lng = float(hotel["lng"])
        hotel_lat = float(hotel["lat"])
    except (KeyError, TypeError, ValueError):
        return ranked

    def distance(item: ScoredPOI) -> float:
        radius = 6371.0
        dlat = math.radians(item.poi.lat - hotel_lat)
        dlng = math.radians(item.poi.lng - hotel_lng)
        value = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(hotel_lat))
            * math.cos(math.radians(item.poi.lat))
            * math.sin(dlng / 2) ** 2
        )
        return radius * 2 * math.asin(math.sqrt(value))

    from travel_agent.critic import poi_matches_must_visit

    mobility_sensitive = bool(
        (profile.constraint_state or {}).get("elderly")
        or (profile.constraint_state or {}).get("travel_with_parents")
        or (profile.constraint_state or {}).get("max_walking_km_per_day") is not None
        or (profile.constraint_state or {}).get("walking_time_max_min") is not None
        or (profile.constraint_state or {}).get("max_single_walk_min") is not None
    )
    locality_radius_km = 12.0 if mobility_sensitive else 30.0
    nearby = [
        item
        for item in ranked
        if (item.poi.category == "food" and not mobility_sensitive)
        or distance(item) <= locality_radius_km
        or any(
            poi_matches_must_visit(item.poi, term)
            for term in profile.must_visit
        )
    ]
    minimum_supply = max(2, int(profile.days or 1) * 2)
    if mobility_sensitive and sum(
        item.poi.category != "food" for item in nearby
    ) < minimum_supply:
        expanded = [
            item
            for item in ranked
            if (item.poi.category == "food" and not mobility_sensitive)
            or distance(item) <= 18.0
            or any(
                poi_matches_must_visit(item.poi, term)
                for term in profile.must_visit
            )
        ]
        if sum(item.poi.category != "food" for item in expanded) >= minimum_supply:
            nearby = expanded
    if mobility_sensitive:
        # With an explicit mobility constraint, a truthful sparse local plan
        # is safer than re-appending city-edge supply merely to fill slots.
        # Must-visits are already retained above.
        return nearby
    if sum(item.poi.category != "food" for item in nearby) < minimum_supply:
        return ranked
    nearby_ids = {item.poi.poi_id for item in nearby}
    return [*nearby, *[item for item in ranked if item.poi.poi_id not in nearby_ids]]


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
@contracted_tool("render_itinerary")
def render_itinerary(ctx: SessionContext) -> dict[str, Any]:
    """把规划好的行程渲染为前端可用的 A2UI 卡片。"""
    artifact_id = ctx.store.latest_id("itinerary")
    record = ctx.store.get_record(artifact_id) if artifact_id else None
    status = str((record or {}).get("artifact_status") or "")
    payload = (record or {}).get("payload")
    if not payload:
        return _err("还没有可渲染的行程，请先调用 plan_and_critique。")
    if status in {"stale", "historical"}:
        return _err(f"该行程状态为 {status}，不能作为当前计划渲染。")
    weather = ctx.store.latest("weather")
    cards = build_itinerary_cards(payload, weather)
    cards.extend(
        _build_supplement_cards(
            restaurants=ctx.store.latest("restaurants"),
            hotels=ctx.store.latest("hotels"),
            budget=ctx.store.latest("budget"),
        )
    )
    return _ok(
        "已生成行程卡片。", cards=cards,
        artifact_status=status or "current",
        incomplete=status == "validation_failure",
    )


@contracted_tool("render_map")
def render_map(ctx: SessionContext) -> dict[str, Any]:
    """生成前端高德地图渲染数据（点位 + 按天分组的路线折线）。"""
    artifact_id = ctx.store.latest_id("itinerary")
    record = ctx.store.get_record(artifact_id) if artifact_id else None
    status = str((record or {}).get("artifact_status") or "")
    payload = (record or {}).get("payload")
    if not payload:
        return _err("还没有可渲染的行程，请先调用 plan_and_critique。")
    if status in {"stale", "historical"}:
        return _err(f"该行程状态为 {status}，不能作为当前计划渲染。")
    map_payload = build_map_payload(payload["itinerary"])
    return _ok(
        f"已生成地图数据：{len(map_payload['markers'])} 个点位。",
        **map_payload,
    )


def _validate_plan_gate(
    ctx: SessionContext,
    plan_artifact_id: str | None,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
    *,
    require_validated: bool = True,
    allow_candidate: bool = False,
    allow_earlier_attempt: bool = False,
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
    if require_validated and not isinstance(payload.get("state_version"), dict):
        return "plan artifact 缺少约束版本绑定，必须按当前状态重建", None
    from travel_agent.plan_invariants import validate_plan_artifact

    validation = validate_plan_artifact(payload, ctx.profile)
    if require_validated and validation.get("passed") is not True:
        return "plan artifact 未通过当前约束的最终确定性不变量校验", None
    request_id = record.get("request_id")
    if request_id and ctx.store.latest_id_for_request(str(request_id), "itinerary") != plan_artifact_id:
        latest_id = ctx.store.latest_id_for_request(str(request_id), "itinerary")
        latest_record = ctx.store.get_record(latest_id) if latest_id else None
        latest_status = str((latest_record or {}).get("artifact_status") or "")
        selected_status = str(record.get("artifact_status") or "")
        rejected_attempt = latest_status in {
            "review_failed", "rework_failed", "validation_failure",
        }
        if not (
            allow_earlier_attempt
            or (selected_status == "current" and rejected_attempt)
        ):
            return "Renderer Gate 只能读取本 request 最新的计划 Artifact", None
    artifact_status = str(record.get("artifact_status") or payload.get("artifact_status") or "")
    allowed_statuses = (
        {"current"}
        if require_validated
        else {"current", "validation_failure", "review_failed", "rework_failed"}
    )
    if allow_candidate:
        allowed_statuses.add("candidate")
    if artifact_status not in allowed_statuses:
        return f"plan artifact 不是当前有效计划: status={artifact_status or 'unknown'}", None
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
    reason, record = _validate_plan_gate(
        ctx, plan_artifact_id, allowed_agents, require_validated=not mark_incomplete
    )
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
    reason, record = _validate_plan_gate(
        ctx, plan_artifact_id, allowed_agents, require_validated=not mark_incomplete
    )
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
    "连接线",
    "检票处",
    "售票处",
    "停车场",
    "出入口",
    "游客中心",
    "旅游广场",
    "纪念品",
    "文创店",
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
            r"(?:不开放|关闭|闭馆|休息|暂停营业|停止开放)",
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

    state = profile.constraint_state or {}
    raw_candidates = state.get("candidate_attractions") or []
    candidate_terms = (
        [str(raw_candidates)]
        if isinstance(raw_candidates, str)
        else [str(item) for item in raw_candidates]
    )
    required = [*list(profile.must_visit or []), *candidate_terms]

    def priority(poi: POI) -> tuple[int, int, int, float]:
        exact_required = any(poi.name == term for term in required)
        matches_required = any(
            canonical_entity_match_evidence(poi, term) is not None
            for term in required
        )
        main_branch = any(marker in poi.name for marker in ("本馆", "主馆"))
        side_branch = any(
            marker in poi.name
            for marker in ("分馆", "西馆", "东馆", "南馆", "北馆", "新馆")
        )
        canonical_venue = any(
            marker in poi.name
            for marker in ("博物馆", "景区", "风景名胜区", "旅游区")
        )
        return (
            (
                0
                if exact_required
                else 1
                if matches_required and main_branch
                else 2
                if matches_required and not side_branch
                else 4
                if side_branch
                else 3
            ),
            0 if canonical_venue else 1,
            len(poi.name),
            -poi.rating,
        )

    def normalized(poi: POI) -> str:
        key = canonical_identity_key(
            poi.canonical_name or poi.name,
            city=poi.city,
            entity_type=poi.entity_type,
        )
        if len(key) < 2:
            # A venue whose name starts with its city can collapse to an empty
            # key after both the city prefix and class suffix are removed
            # (e.g. "City Museum West Wing"). Preserve the city-bearing key
            # so main/branch records can still be compared semantically.
            return canonical_identity_key(
                poi.canonical_name or poi.name,
                entity_type=poi.entity_type,
            )
        return key

    required_keys = {
        canonical_identity_key(term, city=profile.destination)
        for term in required
        if canonical_identity_key(term, city=profile.destination)
    }

    def same_entity(
        left: str,
        left_category: str,
        right: str,
        right_category: str,
    ) -> bool:
        # Equal spelling after stripping class suffixes is insufficient across
        # categories: e.g. "X Museum" is not the scenic area X.  Non-equal
        # cross-category keys can still describe a museum and its explicitly
        # named hall/subvenue and retain the established collapse behavior.
        if left == right and left_category != right_category:
            return False
        if left == right:
            return True
        shorter, longer = sorted((left, right), key=len)
        if (
            len(shorter) >= 2
            and shorter in longer
            and (len(shorter) >= 4 or shorter in required_keys)
        ):
            return True
        common_suffix = ""
        for size in range(1, min(len(left), len(right)) + 1):
            if left[-size:] != right[-size:]:
                break
            common_suffix = left[-size:]
        generic_suffixes = {
            "博物馆", "博物院", "纪念馆", "美术馆", "展览馆",
            "风景区", "景区", "公园", "广场", "中心",
        }
        return len(common_suffix) >= 3 and common_suffix not in generic_suffixes

    kept: list[POI] = []
    keys: list[tuple[str, str]] = []
    for poi in sorted({poi.poi_id: poi for poi in pois}.values(), key=priority):
        key = normalized(poi)
        provider_parent_duplicate = poi.category != "food" and any(
            existing.category != "food"
            and (
                (
                    poi.parent_poi_id
                    and existing.source_poi_id
                    and poi.parent_poi_id == existing.source_poi_id
                )
                or (
                    existing.parent_poi_id
                    and poi.source_poi_id
                    and existing.parent_poi_id == poi.source_poi_id
                )
                or (
                    poi.parent_poi_id
                    and existing.parent_poi_id
                    and poi.parent_poi_id == existing.parent_poi_id
                )
            )
            and (
                poi.category == existing.category
                or same_entity(key, poi.category, normalized(existing), existing.category)
            )
            for existing in kept
        )
        if provider_parent_duplicate or (
            poi.category != "food" and any(
                category != "food"
                and same_entity(key, poi.category, existing, category)
                for existing, category in keys
            )
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
    state = profile.constraint_state or {}
    unspecified = {
        normalize_entity_name(item)
        for item in state.get("user_owned_unspecified_fixed_event_locations") or []
        if normalize_entity_name(item)
    }
    fixed_locations = [
        str(event.get("location") or "").strip()
        for event in state.get("fixed_events") or []
        if isinstance(event, dict)
        and str(event.get("location") or "").strip()
        and normalize_entity_name(event.get("location")) not in unspecified
    ]
    return any(
        poi_matches_must_visit(poi, term)
        for term in [*(profile.must_visit or []), *fixed_locations]
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
        if poi.price_level in {"low", "mid", "high"} and poi.price_level != budget_level:
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
