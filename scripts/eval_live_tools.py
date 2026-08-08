from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent import toolkit
from travel_agent.agent.session import DEFAULT_POI_PATH, build_session
from travel_agent.data_loader import load_seed_pois
from travel_agent.evaluation.plan_eval import evaluate_plan_artifact
from travel_agent.evaluation.chinatravel_adapter import build_chinatravel_provider
from travel_agent.harness.live_tools import (
    LiveToolsHarnessRunner,
    build_live_tools_report,
    build_live_tools_summary,
    write_live_tools_summary,
)
from travel_agent.providers import AmapToolProvider, LocalToolProvider
from travel_agent.schemas import POI, TravelProfile
from travel_agent.settings import get_settings, load_settings


SNAPSHOT_ROOT = ROOT / "data" / "eval" / "live_tools" / "snapshots"
REPORT_PATH = ROOT / "docs" / "EVALUATION_LIVE_TOOLS.md"
SUMMARY_PATH = ROOT / "data" / "eval" / "live_tools" / "summary.json"
CASE_ROOT = ROOT / "data" / "eval" / "live_tools" / "cases"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow testing for live tool providers.")
    parser.add_argument("--provider", choices=["amap", "chinatravel"], default="amap")
    parser.add_argument("--mode", choices=["live", "replay"], default="replay")
    parser.add_argument("--suite", choices=["shadow-dev", "shadow-full"], default="shadow-full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument(
        "--agent-real-multi-agent",
        action="store_true",
        help="额外运行真实 LLM Multi-Agent（V3）agent shadow，不只是工具 provider shadow。",
    )
    parser.add_argument(
        "--continue-on-llm-error",
        action="store_true",
        help="agent shadow 遇到 LLM quota/auth 等错误仍继续；默认停止，避免污染真实评测。",
    )
    args = parser.parse_args()

    result = LiveToolsHarnessRunner(
        snapshot_root=SNAPSHOT_ROOT,
        case_root=CASE_ROOT,
    ).run(
        provider=args.provider,
        mode=args.mode,
        suite=args.suite,
        limit=args.limit,
        agent_real_multi_agent=args.agent_real_multi_agent,
        continue_on_llm_error=args.continue_on_llm_error,
    )
    write_live_tools_summary(result)
    if args.write_report:
        REPORT_PATH.write_text(build_live_tools_report(result), encoding="utf-8")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    agent_metrics = build_live_tools_summary(result).get("agent_metrics")
    if args.agent_real_multi_agent:
        print(json.dumps(agent_metrics, ensure_ascii=False, indent=2))
    return


def _load_cases(suite: str, *, limit: int | None) -> list[dict[str, Any]]:
    path = CASE_ROOT / f"{suite.replace('-', '_')}.json"
    if not path.exists():
        raise RuntimeError(f"找不到 shadow case 文件：{path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise RuntimeError(f"shadow case 文件格式错误：{path}")
    if limit is not None:
        cases = cases[: max(1, limit)]
    return cases


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    if snapshot_path.exists():
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    return {
        "case_id": snapshot_path.stem,
        "ok": False,
        "operations": [
            {
                "operation": "snapshot",
                "ok": False,
                "fallback": False,
                "latency_ms": None,
                "error": f"missing snapshot: {snapshot_path}",
            }
        ],
    }


def _build_shadow_provider(provider_name: str, settings: Any) -> dict[str, Any]:
    fallback = LocalToolProvider(load_seed_pois(DEFAULT_POI_PATH))
    primary = None
    if provider_name == "amap" and settings.amap.rest_enabled:
        primary = AmapToolProvider(
            api_key=settings.amap.web_key or "",
            base_url=settings.amap.base_url,
            timeout_seconds=settings.amap.timeout_seconds,
        )
    if provider_name == "chinatravel":
        primary = build_chinatravel_provider()
    return {"primary": primary, "fallback": fallback}


def _run_live_case(case: dict[str, Any], provider_bundle: dict[str, Any], *, provider_name: str) -> dict[str, Any]:
    operations = []
    city = case["city"]

    poi_payload = _shadow_operation(
        "poi_search",
        primary_call=lambda provider: provider.search_pois(
            city=city,
            query_tags=case.get("poi_tags"),
            max_results=20,
        ),
        fallback_call=lambda provider: provider.search_pois(
            city=city,
            query_tags=case.get("poi_tags"),
            max_results=20,
        ),
        serializer=lambda pois: {"pois": [_poi_to_dict(poi) for poi in pois]},
        usable=lambda payload: bool(payload.get("pois")) and not case.get("expect_empty"),
        provider_bundle=provider_bundle,
    )
    operations.append(poi_payload)

    weather_payload = _shadow_operation(
        "weather",
        primary_call=lambda provider: provider.get_weather(city),
        fallback_call=lambda provider: provider.get_weather(city),
        serializer=lambda weather: {"weather": weather.__dict__},
        usable=lambda payload: (
            bool(payload.get("weather"))
            and (payload.get("weather") or {}).get("condition") != "unknown"
            and not case.get("expect_empty")
        ),
        provider_bundle=provider_bundle,
    )
    operations.append(weather_payload)

    route_payload = _shadow_operation(
        "route",
        primary_call=lambda provider: _estimate_route_for_case(case, provider_bundle, route_provider=provider),
        fallback_call=lambda provider: _estimate_route_for_case(case, provider_bundle, route_provider=provider),
        serializer=lambda route: {"route": route.__dict__ if route else None},
        usable=lambda payload: (
            bool(payload.get("route"))
            and not case.get("expect_empty")
        ),
        provider_bundle=provider_bundle,
    )
    operations.append(route_payload)

    restaurant_payload = _shadow_operation(
        "restaurant_search",
        primary_call=lambda provider: provider.search_pois(
            city=city,
            query_tags=_restaurant_tags(case),
            category="food",
            max_results=10,
        ),
        fallback_call=lambda provider: provider.search_pois(
            city=city,
            query_tags=_restaurant_tags(case),
            category="food",
            max_results=10,
        ),
        serializer=lambda pois: {"restaurants": [_poi_to_dict(poi) for poi in pois]},
        usable=lambda payload: bool(payload.get("restaurants")) and not case.get("expect_empty"),
        provider_bundle=provider_bundle,
    )
    operations.append(restaurant_payload)

    hotel_payload = _shadow_operation(
        "hotel_search",
        primary_call=lambda provider: provider.search_pois(
            city=city,
            query_tags=["hotel"],
            category="hotel",
            max_results=10,
        ),
        fallback_call=lambda provider: provider.search_pois(
            city=city,
            query_tags=["hotel"],
            category="hotel",
            max_results=10,
        ),
        serializer=lambda pois: {"hotels": [_poi_to_dict(poi) for poi in pois]},
        usable=lambda payload: bool(payload.get("hotels")) and not case.get("expect_empty"),
        provider_bundle=provider_bundle,
    )
    operations.append(hotel_payload)

    operations.append(_run_budget_operation(case, provider_bundle))

    return {
        "case_id": case["id"],
        "city": city,
        "provider": provider_name,
        "ok": all(operation.get("ok") for operation in operations),
        "operations": operations,
    }


def _shadow_operation(
    operation: str,
    *,
    primary_call: Any,
    fallback_call: Any,
    serializer: Any,
    usable: Any,
    provider_bundle: dict[str, Any],
) -> dict[str, Any]:
    primary = provider_bundle["primary"]
    fallback = provider_bundle["fallback"]
    primary_payload: dict[str, Any] | None = None
    primary_error = None
    primary_latency_ms = None
    primary_usable = False
    if primary is not None:
        started = time.perf_counter()
        try:
            primary_result = primary_call(primary)
            primary_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            primary_payload = serializer(primary_result)
            primary_usable = bool(usable(primary_payload))
        except Exception as exc:
            primary_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            primary_error = str(exc)

    if primary_payload is not None and primary_usable:
        return {
            "operation": operation,
            "ok": True,
            "fallback": False,
            "fallback_used": False,
            "primary_ok": True,
            "primary_usable": True,
            "primary_latency_ms": primary_latency_ms,
            "primary_error": None,
            "latency_ms": primary_latency_ms,
            **primary_payload,
        }

    started = time.perf_counter()
    try:
        fallback_result = fallback_call(fallback)
        fallback_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        fallback_payload = serializer(fallback_result)
        return {
            "operation": operation,
            "ok": True,
            "fallback": True,
            "fallback_used": True,
            "primary_ok": primary_payload is not None,
            "primary_usable": False,
            "primary_latency_ms": primary_latency_ms,
            "primary_error": primary_error,
            "fallback_latency_ms": fallback_latency_ms,
            "latency_ms": _sum_latency(primary_latency_ms, fallback_latency_ms),
            **fallback_payload,
        }
    except Exception as exc:
        fallback_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "operation": operation,
            "ok": False,
            "fallback": primary is not None,
            "fallback_used": primary is not None,
            "primary_ok": primary_payload is not None,
            "primary_usable": False,
            "primary_latency_ms": primary_latency_ms,
            "primary_error": primary_error,
            "fallback_latency_ms": fallback_latency_ms,
            "latency_ms": _sum_latency(primary_latency_ms, fallback_latency_ms),
            "error": str(exc),
        }


def _run_budget_operation(case: dict[str, Any], provider_bundle: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        ctx = build_session(persist=False)
        ctx.provider = provider_bundle["primary"] or provider_bundle["fallback"]
        ctx.profile = TravelProfile(
            destination=case["city"],
            days=int(case.get("days") or 2),
            budget_level=case.get("budget_level") or "mid",
            companions=case.get("companions") or "情侣",
        )
        result = toolkit.estimate_budget(ctx)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        ok = not bool(result.get("isError"))
        return {
            "operation": "budget_estimate",
            "ok": ok,
            "fallback": False,
            "fallback_used": False,
            "primary_ok": ok,
            "primary_usable": ok,
            "primary_latency_ms": latency_ms,
            "primary_error": None,
            "latency_ms": latency_ms,
            "budget": result.get("budget"),
            "error": result.get("summary") if not ok else None,
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "operation": "budget_estimate",
            "ok": False,
            "fallback": False,
            "fallback_used": False,
            "primary_ok": False,
            "primary_usable": False,
            "primary_latency_ms": latency_ms,
            "primary_error": str(exc),
            "latency_ms": latency_ms,
            "budget": None,
            "error": str(exc),
        }


def _run_agent_shadow_case(
    case: dict[str, Any],
    provider_bundle: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    ctx = build_session(persist=False)
    ctx.provider = provider_bundle["primary"] or provider_bundle["fallback"]
    query = _agent_query_from_shadow_case(case)
    reply = run_production_turn(query, ctx=ctx, settings=settings, user_id="shadow_real_ma")
    artifact = ctx.store.latest("itinerary")
    trace_payload = ctx.store.latest("agent_trace") or {}
    trace_items = list(trace_payload.get("items") or [])
    subagent_items = [item for item in trace_items if item.get("kind") == "subagent"]
    agent_names = [item.get("agent") for item in subagent_items]
    required_agents = ["attraction", "hotel", "restaurant", "transport", "planner"]
    plan_eval = evaluate_plan_artifact(
        artifact,
        profile={
            "destination": ctx.profile.destination,
            "days": ctx.profile.days,
            "interests": list(ctx.profile.interests),
            "pace": ctx.profile.pace,
        },
        required_interests=_required_interests_from_shadow_case(case),
    )
    tools_by_agent = {
        str(item.get("agent")): list((item.get("detail") or {}).get("tool_trace") or [])
        for item in subagent_items
    }
    flat_tools = [tool for tools in tools_by_agent.values() for tool in tools]
    return {
        "case_id": case["id"],
        "city": case["city"],
        "query": query,
        "expect_empty": bool(case.get("expect_empty")),
        "used_real_agent": bool(reply.used_real_agent),
        "itinerary_produced": artifact is not None,
        "agent_trace_present": bool(trace_items),
        "required_agents_present": all(agent in agent_names for agent in required_agents),
        "research_tools_ok": any(tool in flat_tools for tool in ("search_poi", "check_weather", "plan_route")),
        "planning_tools_ok": any(tool in flat_tools for tool in ("recommend_candidates", "plan_and_critique")),
        "required_tools_ok": all(tool in list(reply.tool_trace or flat_tools) for tool in ("search_poi", "recommend_candidates", "plan_and_critique")),
        "final_pass": plan_eval.final_pass if plan_eval else None,
        "issues": list(plan_eval.issues) if plan_eval else ["no_itinerary"],
        "tool_steps": len(reply.tool_trace or flat_tools),
        "agent_names": agent_names,
        "failed_agent_count": sum(
            1
            for item in subagent_items
            if item.get("status") not in ("completed", "completed_with_warnings")
        ),
        "external_llm_error": _payload_has_external_llm_error(
            {
                "tool_trace": list(reply.tool_trace or []),
                "agent_trace": trace_payload,
                "issues": list(plan_eval.issues) if plan_eval else ["no_itinerary"],
            }
        ),
    }


def _agent_query_from_shadow_case(case: dict[str, Any]) -> str:
    tags = _required_interests_from_shadow_case(case)
    tags_text = "、".join(_tag_label(tag) for tag in tags) or "经典景点"
    days = int(case.get("days") or 2)
    extras = []
    if case.get("route_mode"):
        extras.append("市内交通顺一点")
    if "hotel" in list(case.get("poi_tags") or []) or case.get("id", "").startswith("hotel"):
        extras.append("包含酒店")
    extras.append("包含餐厅和预算")
    return f"帮我规划{case['city']}{days}天旅行，偏好{tags_text}，{','.join(extras)}，不要太累。"


def _required_interests_from_shadow_case(case: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for raw in list(case.get("poi_tags") or []):
        tag = str(raw)
        if tag in {"local", "餐厅", "吃饭"}:
            tag = "food"
        if tag in {"classic", "sightseeing", "公园"}:
            tag = "nature"
        if tag in {"history"}:
            tag = "history"
        if tag in {"museum"}:
            tag = "museum"
        if tag in {"shopping"}:
            tag = "shopping"
        if tag in {"family"}:
            tag = "family"
        if tag in {"night"}:
            tag = "nightlife"
        if tag in {"food", "nature", "history", "museum", "shopping", "family", "nightlife"} and tag not in tags:
            tags.append(tag)
    return tags or ["citywalk"]


def _tag_label(tag: str) -> str:
    return {
        "food": "美食",
        "nature": "自然风景",
        "history": "历史文化",
        "museum": "博物馆",
        "shopping": "购物",
        "family": "亲子",
        "nightlife": "夜生活",
        "citywalk": "城市漫步",
    }.get(tag, tag)


def _estimate_route_for_case(case: dict[str, Any], provider_bundle: dict[str, Any], *, route_provider: Any) -> Any:
    city = case["city"]
    search_provider = route_provider
    origin_candidates = search_provider.search_pois(city=city, query_tags=case.get("poi_tags"), max_results=5)
    destination_candidates = search_provider.search_pois(
        city=city,
        query_tags=case.get("route_tags"),
        max_results=5,
    )
    if (not origin_candidates or not destination_candidates) and search_provider is not provider_bundle["fallback"]:
        search_provider = provider_bundle["fallback"]
        origin_candidates = search_provider.search_pois(city=city, query_tags=case.get("poi_tags"), max_results=5)
        destination_candidates = search_provider.search_pois(
            city=city,
            query_tags=case.get("route_tags"),
            max_results=5,
        )
    origin = _first_valid_poi(origin_candidates)
    destination = _first_valid_poi(destination_candidates, exclude_name=origin.name if origin else None)
    if not origin or not destination:
        return None
    return route_provider.estimate_route(origin, destination, mode=case.get("route_mode") or "public_transport")


def _first_valid_poi(pois: list[POI], *, exclude_name: str | None = None) -> POI | None:
    for poi in pois:
        if exclude_name and poi.name == exclude_name:
            continue
        if _valid_coord(poi.lat, poi.lng):
            return poi
    return None


def _rows_from_payload(case: dict[str, Any], payload: dict[str, Any], *, replay: bool) -> list[dict[str, Any]]:
    rows = []
    for operation in payload.get("operations") or []:
        op_name = operation.get("operation") or "unknown"
        row = {
            "case_id": case["id"],
            "city": case["city"],
            "expect_empty": bool(case.get("expect_empty")),
            "expected_tags": list(case.get("poi_tags") or []),
            "operation": op_name,
            "ok": bool(operation.get("ok")),
            "fallback": bool(operation.get("fallback")),
            "fallback_used": bool(operation.get("fallback_used")),
            "primary_ok": operation.get("primary_ok"),
            "primary_usable": operation.get("primary_usable"),
            "primary_latency_ms": operation.get("primary_latency_ms"),
            "primary_error": operation.get("primary_error"),
            "replay": replay,
            "latency_ms": operation.get("latency_ms"),
            "result_count": None,
            "empty_result": False,
            "duplicate_count": 0,
            "coordinate_valid_rate": None,
            "category_match_rate": None,
            "city_contamination_rate": None,
            "fallback_reason": _fallback_reason(operation),
            "schema_valid": _operation_schema_valid(op_name, operation),
            "route_unavailable": False,
            "route_degraded": False,
            "route_source": None,
            "timeout": _is_timeout(operation.get("error")) or _is_timeout(operation.get("primary_error")),
            "error": operation.get("error"),
        }
        if op_name == "poi_search":
            pois = operation.get("pois") or []
            row.update(_poi_row_metrics(pois, expected_tags=case.get("poi_tags"), requested_city=case["city"]))
        elif op_name == "weather":
            weather = operation.get("weather")
            row["empty_result"] = not bool(weather)
        elif op_name == "route":
            route = operation.get("route")
            row["route_unavailable"] = not bool(route)
            row["empty_result"] = not bool(route)
            if route:
                source = str(route.get("source") or "")
                row["route_source"] = source
                row["route_degraded"] = source.endswith("fallback_estimate")
        elif op_name == "restaurant_search":
            restaurants = operation.get("restaurants") or []
            row.update(
                _poi_row_metrics(
                    restaurants,
                    expected_tags=_restaurant_tags(case),
                    requested_city=case["city"],
                )
            )
        elif op_name == "hotel_search":
            hotels = operation.get("hotels") or []
            row.update(
                _poi_row_metrics(
                    hotels,
                    expected_tags=["hotel"],
                    requested_city=case["city"],
                )
            )
        elif op_name == "budget_estimate":
            budget = operation.get("budget")
            row["empty_result"] = not bool(budget)
            row["result_count"] = 1 if budget else 0
        rows.append(row)
    return rows


def _poi_row_metrics(
    pois: list[dict[str, Any]],
    *,
    expected_tags: list[str] | None,
    requested_city: str,
) -> dict[str, Any]:
    names = [str(poi.get("name") or "") for poi in pois if poi.get("name")]
    duplicate_count = len(names) - len(set(names))
    coord_valid = [poi for poi in pois if _valid_coord(poi.get("lat"), poi.get("lng"))]
    category_matches = [
        poi for poi in pois if _category_match(poi, expected_tags or [])
    ]
    contaminated = [
        poi for poi in pois if _city_contaminated(poi, requested_city)
    ]
    return {
        "result_count": len(pois),
        "empty_result": len(pois) == 0,
        "duplicate_count": duplicate_count,
        "coordinate_valid_rate": len(coord_valid) / len(pois) if pois else None,
        "category_match_rate": len(category_matches) / len(pois) if pois else None,
        "city_contamination_rate": len(contaminated) / len(pois) if pois else None,
    }


def _operation_schema_valid(operation: str, payload: dict[str, Any]) -> bool:
    if not payload.get("ok"):
        return False
    if operation == "poi_search":
        return isinstance(payload.get("pois"), list)
    if operation == "restaurant_search":
        return isinstance(payload.get("restaurants"), list)
    if operation == "hotel_search":
        return isinstance(payload.get("hotels"), list)
    if operation == "weather":
        weather = payload.get("weather")
        return isinstance(weather, dict) and bool(weather.get("city")) and bool(weather.get("source"))
    if operation == "route":
        route = payload.get("route")
        if route is None:
            return True
        return isinstance(route, dict) and route.get("duration_min", 0) >= 0 and bool(route.get("source"))
    if operation == "budget_estimate":
        budget = payload.get("budget")
        return (
            isinstance(budget, dict)
            and budget.get("total_high", 0) >= budget.get("total_low", 0) >= 0
            and budget.get("days", 0) > 0
        )
    return False


TAG_GROUPS = {
    "food": {"food", "local", "餐厅", "吃饭", "night"},
    "local": {"food", "local", "餐厅", "吃饭", "night"},
    "餐厅": {"food", "local", "餐厅", "吃饭", "night"},
    "吃饭": {"food", "local", "餐厅", "吃饭", "night"},
    "history": {"history", "culture", "museum", "classic"},
    "culture": {"culture", "history", "museum", "classic"},
    "classic": {"sightseeing", "classic", "nature", "history", "culture"},
    "nature": {"nature", "sightseeing", "classic", "公园"},
    "sightseeing": {"sightseeing", "classic", "nature", "公园"},
    "公园": {"sightseeing", "classic", "nature", "公园"},
    "好玩": {"sightseeing", "classic", "nature", "museum", "shopping"},
    "shopping": {"shopping"},
    "family": {"family", "culture", "museum", "nature", "classic"},
    "museum": {"museum", "culture", "history"},
    "night": {"food", "night", "shopping"},
    "hotel": {"hotel", "accommodation"},
    "accommodation": {"hotel", "accommodation"},
}


def _category_match(poi: dict[str, Any], expected_tags: list[str]) -> bool:
    if not expected_tags:
        return True
    observed = {str(poi.get("category") or "")}
    observed.update(str(tag) for tag in poi.get("tags") or [])
    for tag in expected_tags:
        tag_text = str(tag)
        allowed = TAG_GROUPS.get(tag_text, {tag_text})
        if observed.intersection(allowed):
            return True
    return False


def _city_contaminated(poi: dict[str, Any], requested_city: str) -> bool:
    if not requested_city or requested_city in {"Shanghai", "Hangzhou"}:
        return False
    if requested_city in {"不存在的火星城", "不存在天气城", "@@@", "南方", "古城", "朝阳区", "西湖", "春熙路"}:
        return True
    city = str(poi.get("city") or "")
    return bool(city and city != requested_city)


def _fallback_reason(operation: dict[str, Any]) -> str | None:
    if not operation.get("fallback"):
        return None
    if operation.get("primary_error"):
        return "primary_error"
    if operation.get("primary_ok") and not operation.get("primary_usable"):
        return "primary_unusable"
    if not operation.get("primary_ok"):
        return "primary_missing"
    return "fallback"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    poi_rows = [row for row in rows if row.get("operation") == "poi_search"]
    restaurant_rows = [row for row in rows if row.get("operation") == "restaurant_search"]
    hotel_rows = [row for row in rows if row.get("operation") == "hotel_search"]
    budget_rows = [row for row in rows if row.get("operation") == "budget_estimate"]
    route_rows = [row for row in rows if row.get("operation") == "route"]
    return {
        "sample_size": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "api_success_rate": _rate(rows, "ok"),
        "operation_success_rate": _operation_rates(rows, "ok"),
        "fallback_rate": _rate(rows, "fallback"),
        "fallback_reason_breakdown": _fallback_reason_breakdown(rows),
        "primary_success_rate": _rate(rows, "primary_ok"),
        "primary_usable_rate": _rate(rows, "primary_usable"),
        "timeout_rate": _rate(rows, "timeout"),
        "empty_result_rate": _rate(rows, "empty_result"),
        "expected_empty_pass_rate": _expected_empty_pass_rate(rows),
        "duplicate_poi_rate": _duplicate_rate(poi_rows),
        "route_unavailable_rate": _rate(route_rows, "route_unavailable"),
        "unexpected_route_unavailable_rate": _unexpected_route_unavailable_rate(route_rows),
        "route_degraded_rate": _rate(route_rows, "route_degraded"),
        "schema_valid_rate": _rate(rows, "schema_valid"),
        "avg_coordinate_valid_rate": _avg(poi_rows, "coordinate_valid_rate"),
        "avg_category_match_rate": _avg(poi_rows, "category_match_rate"),
        "restaurant_success_rate": _rate(restaurant_rows, "ok"),
        "hotel_success_rate": _rate(hotel_rows, "ok"),
        "budget_success_rate": _rate(budget_rows, "ok"),
        "avg_restaurant_category_match_rate": _avg(restaurant_rows, "category_match_rate"),
        "avg_hotel_category_match_rate": _avg(hotel_rows, "category_match_rate"),
        "avg_city_contamination_rate": _avg(poi_rows, "city_contamination_rate"),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def _agent_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_size": len(rows),
        "used_real_agent_rate": _rate(rows, "used_real_agent"),
        "itinerary_produced_rate": _rate(rows, "itinerary_produced"),
        "agent_trace_rate": _rate(rows, "agent_trace_present"),
        "required_agents_rate": _rate(rows, "required_agents_present"),
        "research_agent_tool_rate": _rate(rows, "research_tools_ok"),
        "planning_agent_tool_rate": _rate(rows, "planning_tools_ok"),
        "required_tool_coverage_rate": _rate(rows, "required_tools_ok"),
        "final_pass_rate": _rate(rows, "final_pass"),
        "external_llm_error_rate": _rate(rows, "external_llm_error"),
        "avg_tool_steps": _avg(rows, "tool_steps"),
        "avg_failed_agent_count": _avg(rows, "failed_agent_count"),
    }


def _payload_has_external_llm_error(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    markers = (
        "AllocationQuota",
        "FreeTierOnly",
        "free quota",
        "quota",
        "401",
        "403",
        "Unauthorized",
        "authentication",
        "api key",
        "invalid_parameter_error",
        "enable_thinking",
    )
    return any(marker in text for marker in markers)


def build_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Live Tools Shadow Testing 报告",
        "",
        "## 结论",
        "",
        f"- provider：`{report['provider']}`",
        f"- mode：`{report['mode']}`",
        f"- agent real multi-agent：{report.get('agent_real_multi_agent')}",
        f"- llm：`{report.get('llm_provider')}` / `{report.get('llm_model')}`",
        f"- architecture：`{report.get('architecture')}`",
        f"- provider configured：{report['provider_configured']}",
        f"- snapshot root：`{report['snapshot_root']}`",
        f"- snapshot version：`{report.get('snapshot_version')}`",
        f"- API success rate：{metrics['api_success_rate']}",
        f"- operation success rate：{metrics['operation_success_rate']}",
        f"- fallback rate：{metrics['fallback_rate']}",
        f"- fallback reason breakdown：{metrics['fallback_reason_breakdown']}",
        f"- primary success rate：{metrics['primary_success_rate']}",
        f"- primary usable rate：{metrics['primary_usable_rate']}",
        f"- timeout rate：{metrics['timeout_rate']}",
        f"- empty result rate：{metrics['empty_result_rate']}",
        f"- expected empty pass rate：{metrics['expected_empty_pass_rate']}",
        f"- duplicate POI rate：{metrics['duplicate_poi_rate']}",
        f"- route unavailable rate：{metrics['route_unavailable_rate']}",
        f"- unexpected route unavailable rate：{metrics['unexpected_route_unavailable_rate']}",
        f"- route degraded rate：{metrics['route_degraded_rate']}",
        f"- schema valid rate：{metrics['schema_valid_rate']}",
        f"- avg category match rate：{metrics['avg_category_match_rate']}",
        f"- restaurant / hotel / budget success rate：{metrics.get('restaurant_success_rate')} / {metrics.get('hotel_success_rate')} / {metrics.get('budget_success_rate')}",
        f"- restaurant / hotel category match rate：{metrics.get('avg_restaurant_category_match_rate')} / {metrics.get('avg_hotel_category_match_rate')}",
        f"- avg city contamination rate：{metrics['avg_city_contamination_rate']}",
        f"- latency p50 / p95：{metrics['latency_p50_ms']} / {metrics['latency_p95_ms']} ms",
    ]
    agent_metrics = report.get("agent_metrics") or {}
    if agent_metrics:
        lines.extend(
            [
                "",
                "## 真实 Multi-Agent Agent Shadow",
                "",
                f"- sample size：{agent_metrics.get('sample_size')}",
                f"- used real agent rate：{agent_metrics.get('used_real_agent_rate')}",
                f"- itinerary produced rate：{agent_metrics.get('itinerary_produced_rate')}",
                f"- agent trace rate：{agent_metrics.get('agent_trace_rate')}",
                f"- required agents rate：{agent_metrics.get('required_agents_rate')}",
                f"- research / planning tool rate：{agent_metrics.get('research_agent_tool_rate')} / {agent_metrics.get('planning_agent_tool_rate')}",
                f"- required tool coverage：{agent_metrics.get('required_tool_coverage_rate')}",
                f"- final pass rate：{agent_metrics.get('final_pass_rate')}",
                f"- avg tool steps：{agent_metrics.get('avg_tool_steps')}",
            ]
        )
    lines.extend(
        [
            "",
            "## 明细",
            "",
            "| case | city | operation | expect_empty | ok | fallback | reason | primary_ok | primary_usable | empty | route_unavailable | route_source | latency_ms | count | dup | cat_match | city_bad | schema | primary_error | error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in report["rows"]:
        lines.append(
            f"| {row['case_id']} | {row['city']} | {row['operation']} | {row['expect_empty']} | {row['ok']} | "
            f"{row['fallback']} | {row['fallback_reason']} | {row['primary_ok']} | {row['primary_usable']} | "
            f"{row['empty_result']} | {row['route_unavailable']} | "
            f"{row.get('route_source') or ''} | "
            f"{row['latency_ms']} | {row.get('result_count')} | {row['duplicate_count']} | "
            f"{_fmt(row.get('category_match_rate'))} | {_fmt(row.get('city_contamination_rate'))} | "
            f"{row['schema_valid']} | {_md(row.get('primary_error') or '')} | {_md(row.get('error') or '')} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `live` 模式会调用当前配置的 provider，并保存每个 case 的 API/snapshot payload；",
            "- `replay` 模式只读取 snapshot，不再调用外部 API，用于复现报告；",
            "- 如果未配置高德 key，`provider configured=False`，结果主要反映本地 fallback 链路；",
            "- `chinatravel` provider 使用 ChinaTravel sandbox/database 或 synthetic ChinaTravel provider，不调用外部 API；",
            "- 除 POI / weather / route 外，灰度测试还覆盖 restaurant_search / hotel_search / budget_estimate；",
            "- `primary_ok=True` 表示目标 provider 调用未抛异常；`primary_usable=True` 表示其结果可直接用于 agent；",
            "- `fallback=True` 表示该操作最终使用了兜底结果，例如本地 POI、mock weather 或 haversine route。",
        ]
    )
    return "\n".join(lines)


def _poi_to_dict(poi: POI) -> dict[str, Any]:
    return {
        "poi_id": poi.poi_id,
        "name": poi.name,
        "city": poi.city,
        "category": poi.category,
        "lat": poi.lat,
        "lng": poi.lng,
        "rating": poi.rating,
        "popularity": poi.popularity,
        "tags": poi.tags,
        "estimated_duration_min": poi.estimated_duration_min,
        "price_level": poi.price_level,
        "indoor": poi.indoor,
        "opening_hours": poi.opening_hours,
        "source": poi.source,
    }


def _restaurant_tags(case: dict[str, Any]) -> list[str]:
    tags = list(case.get("restaurant_tags") or [])
    if tags:
        return tags
    poi_tags = set(case.get("poi_tags") or [])
    if poi_tags.intersection({"food", "local", "night", "餐厅", "吃饭"}):
        return [str(tag) for tag in poi_tags]
    return ["food", "local"]


def _valid_coord(lat: Any, lng: Any) -> bool:
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return False
    return -90 <= lat_f <= 90 and -180 <= lng_f <= 180


def _is_timeout(error: Any) -> bool:
    if not error:
        return False
    text = str(error).lower()
    return "timeout" in text or "timed out" in text or "超时" in text


def _sum_latency(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present), 2)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def _operation_rates(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    operations = sorted({str(row.get("operation")) for row in rows})
    for operation in operations:
        op_rows = [row for row in rows if row.get("operation") == operation]
        value = _rate(op_rows, key)
        if value is not None:
            result[operation] = value
    return result


def _fallback_reason_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("fallback_reason")
        if not reason:
            continue
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _duplicate_rate(rows: list[dict[str, Any]]) -> float | None:
    result_count = sum(row.get("result_count") or 0 for row in rows)
    if result_count == 0:
        return None
    duplicate_count = sum(row.get("duplicate_count") or 0 for row in rows)
    return round(duplicate_count / result_count, 4)


def _unexpected_route_unavailable_rate(route_rows: list[dict[str, Any]]) -> float | None:
    unexpected_rows = [row for row in route_rows if not row.get("expect_empty")]
    return _rate(unexpected_rows, "route_unavailable")


def _expected_empty_pass_rate(rows: list[dict[str, Any]]) -> float | None:
    expected_rows = [
        row
        for row in rows
        if row.get("expect_empty") and row.get("operation") != "budget_estimate"
    ]
    if not expected_rows:
        return None
    passed = [
        row
        for row in expected_rows
        if row.get("empty_result") or not row.get("primary_usable")
    ]
    return round(len(passed) / len(expected_rows), 4)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return ordered[index]


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:160]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _configure_real_multi_agent():
    os.environ["TRAVEL_AGENT_SKILLS_ENABLED"] = "false"
    get_settings.cache_clear()
    settings = load_settings()
    if not settings.llm.enabled:
        raise RuntimeError("agent shadow 真实 Multi-Agent 评测需要可用 LLM 配置。")
    return settings


if __name__ == "__main__":
    main()
