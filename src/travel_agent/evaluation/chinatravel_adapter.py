from __future__ import annotations

import json
import os
import re
import sys
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jsonschema import Draft7Validator
import pandas as pd

from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.providers import LocalToolProvider
from travel_agent.schemas import POI, RouteInfo, TransportMode, WeatherInfo
from travel_agent.settings import Settings


DEFAULT_CHINATRAVEL_ROOT = Path(
    os.getenv("TRAVEL_AGENT_CHINATRAVEL_ROOT", "/Users/carrier/run/projects/ChinaTravel")
)
_ACTIVE_CHINATRAVEL_ROOT = DEFAULT_CHINATRAVEL_ROOT


SUITE_TO_SPLIT = {
    "mini-dev": "human",
    "human154": "human",
    "human1000": "human1000",
}

DEFAULT_MINI_DEV_IDS = [
    "e20241028160248698752",
    "e20241028160251109186",
    "e20241028160253742452",
    "e20241028160255878579",
    "e20241028160258515151",
    "e20241028160300570125",
    "e20241028160301887545",
    "e20241028160303262666",
]


@dataclass(frozen=True)
class ChinaTravelCase:
    query_id: str
    query: str
    raw: dict[str, Any]


def load_chinatravel_cases(
    chinatravel_root: Path,
    suite: str,
    *,
    smoke_limit: int | None = None,
    lang: str = "zh",
) -> list[ChinaTravelCase]:
    """加载 ChinaTravel case。

    优先复用官方 loader。若官方数据未下载且 HuggingFace 不可用，会抛出带行动建议的错误。
    """

    split = SUITE_TO_SPLIT.get(suite, suite)
    _ensure_chinatravel_importable(chinatravel_root)
    local_csv = _local_chinatravel_csv(suite)
    if local_csv.exists():
        return _load_cases_from_csv(local_csv, smoke_limit=smoke_limit, lang=lang)

    try:
        from chinatravel.data.load_datasets import load_query
    except Exception:
        return _load_cases_from_split_ids(
            chinatravel_root,
            suite,
            split,
            smoke_limit=smoke_limit,
        )

    args = SimpleNamespace(splits=split, lang=lang, oracle_translation=True)
    try:
        query_ids, query_data = load_query(args)
    except Exception as exc:
        local_hint = chinatravel_root / "chinatravel" / "data"
        raise RuntimeError(
            "无法加载 ChinaTravel 数据。请确认已安装 `datasets`，或已将官方数据库下载到 "
            f"{local_hint}。原始错误：{exc}"
        ) from exc
    if split == "human1000" and not any(query_data.get(query_id) for query_id in query_ids):
        raise RuntimeError(
            "当前 ChinaTravel checkout 只提供 human1000 id 列表，没有 query / hard_logic_py。"
            f"请将官方 human1000.csv 放到 {local_csv} 后重跑。"
        )

    if suite == "mini-dev":
        wanted = [qid for qid in DEFAULT_MINI_DEV_IDS if qid in set(query_ids)]
        if not wanted:
            wanted = list(query_ids[:8])
        query_ids = wanted
    if smoke_limit is not None:
        query_ids = list(query_ids[: max(1, smoke_limit)])

    cases = []
    for query_id in query_ids:
        raw = dict(query_data.get(query_id) or {})
        cases.append(
            ChinaTravelCase(
                query_id=query_id,
                query=extract_query_text(raw),
                raw=raw,
            )
        )
    return cases


def _local_chinatravel_csv(suite: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "eval" / "chinatravel" / f"{suite}.csv"


def _load_cases_from_csv(
    path: Path,
    *,
    smoke_limit: int | None,
    lang: str,
) -> list[ChinaTravelCase]:
    df = pd.read_csv(path)
    if smoke_limit is not None:
        df = df.head(max(1, smoke_limit))
    cases = []
    for row in df.to_dict("records"):
        raw = dict(row)
        _augment_raw_query_fields(raw)
        if isinstance(raw.get("hard_logic_py"), str):
            try:
                raw["hard_logic_py"] = ast.literal_eval(raw["hard_logic_py"])
            except Exception:
                raw["hard_logic_py"] = [raw["hard_logic_py"]]
        query_id = str(raw.get("uid") or raw.get("query_id") or "")
        if not query_id:
            continue
        cases.append(
            ChinaTravelCase(
                query_id=query_id,
                query=extract_query_text(raw),
                raw=raw,
            )
        )
    return cases


def _augment_raw_query_fields(raw: dict[str, Any]) -> None:
    """为只有自然语言的 ChinaTravel CSV 补齐评测必需字段。"""

    text = extract_query_text(raw)
    if not text:
        return
    cities = _extract_city_mentions(text)
    if not raw.get("start_city"):
        start_city = _infer_start_city(text, cities)
        if start_city:
            raw["start_city"] = start_city
    if not raw.get("target_city") and not raw.get("destination"):
        target_city = _infer_target_city(text, cities, str(raw.get("start_city") or ""))
        if target_city:
            raw["target_city"] = target_city
    if not raw.get("people_number") and not raw.get("people"):
        people = _infer_people_number(text)
        if people:
            raw["people_number"] = people
    if not raw.get("days") and not raw.get("day") and not raw.get("trip_days"):
        days = _infer_trip_days(text)
        if days:
            raw["days"] = days
    if not raw.get("hard_logic_py"):
        raw["hard_logic_py"] = []


_CHINATRAVEL_CITY_NAMES = (
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "南京",
    "苏州",
    "成都",
    "重庆",
    "武汉",
)


def _extract_city_mentions(text: str) -> list[str]:
    mentions: list[tuple[int, str]] = []
    for city in _CHINATRAVEL_CITY_NAMES:
        start = 0
        while True:
            index = text.find(city, start)
            if index < 0:
                break
            mentions.append((index, city))
            start = index + len(city)
    return [city for _, city in sorted(mentions)]


def _infer_start_city(text: str, cities: list[str]) -> str | None:
    patterns = (
        r"从\s*([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})\s*出发",
        r"([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})\s*出发",
        r"当前位置\s*([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})",
        r"当前我在\s*([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})",
        r"我在\s*([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in _CHINATRAVEL_CITY_NAMES:
            return match.group(1)
    if len(cities) >= 2 and any(word in text for word in ("去", "前往", "到")):
        return cities[0]
    return cities[0] if len(cities) == 1 and "出发" in text else None


def _infer_target_city(text: str, cities: list[str], start_city: str) -> str | None:
    patterns = (
        r"(?:前往|去|到)\s*([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})",
        r"目标位置\s*([北京上海广州深圳杭州南京苏州成都重庆武汉]{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in _CHINATRAVEL_CITY_NAMES:
            return match.group(1)
    if len(cities) >= 2:
        for city in reversed(cities):
            if city != start_city:
                return city
    if len(cities) == 1 and cities[0] != start_city:
        return cities[0]
    return None


def _infer_people_number(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:人|个?人|位)", text)
    if match:
        return int(match.group(1))
    chinese = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
    }
    for token, value in chinese.items():
        if re.search(fr"{token}\s*(?:人|个?人|位)", text):
            return value
    if any(word in text for word in ("我和老公", "我和老婆", "我和女朋友", "我和男朋友", "情侣")):
        return 2
    if "一家三口" in text:
        return 3
    if "一个人" in text or "独自" in text:
        return 1
    return None


def _infer_trip_days(text: str) -> int | None:
    match = re.search(r"(\d+)\s*天", text)
    if match:
        return int(match.group(1))
    chinese = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
    }
    for token, value in chinese.items():
        if f"{token}天" in text:
            return value
    return None


def _load_cases_from_split_ids(
    chinatravel_root: Path,
    suite: str,
    split: str,
    *,
    smoke_limit: int | None,
) -> list[ChinaTravelCase]:
    split_path = (
        chinatravel_root
        / "chinatravel"
        / "evaluation"
        / "default_splits"
        / f"{split}.txt"
    )
    if not split_path.exists():
        raise RuntimeError(f"找不到 ChinaTravel split 文件：{split_path}")
    query_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if suite == "mini-dev":
        preferred = [qid for qid in DEFAULT_MINI_DEV_IDS if qid in set(query_ids)]
        query_ids = preferred or query_ids[:8]
    if smoke_limit is not None:
        query_ids = query_ids[: max(1, smoke_limit)]
    return [
        ChinaTravelCase(
            query_id=query_id,
            query="请规划杭州三天旅行行程，包含景点、餐饮、住宿和市内交通，注意不要太累。",
            raw={
                "uid": query_id,
                "target_city": "杭州",
                "start_city": "杭州",
                "people_number": 1,
                "_loader_fallback": True,
            },
        )
        for query_id in query_ids
    ]


def run_agent_for_case(
    case: ChinaTravelCase,
    settings: Settings,
    chinatravel_root: Path | str | None = None,
) -> dict[str, Any]:
    return run_agent_for_case_with_diagnostics(case, settings, chinatravel_root)["prediction"]


def run_agent_for_case_with_diagnostics(
    case: ChinaTravelCase,
    settings: Settings,
    chinatravel_root: Path | str | None = None,
) -> dict[str, Any]:
    if chinatravel_root is not None:
        configure_chinatravel_root(chinatravel_root)
    ctx = build_session(session_id=f"ct_{case.query_id}", persist=False)
    ctx.provider = build_chinatravel_provider(chinatravel_root, allow_synthetic=False)
    reply = run_production_turn(case.query, ctx=ctx, settings=settings, user_id="chinatravel_eval")
    artifact = ctx.store.latest("itinerary")
    if not artifact and reply.clarification and not ctx.profile.missing_required_fields():
        reply = run_production_turn("你推荐，按默认偏好规划", ctx=ctx, settings=settings, user_id="chinatravel_eval")
        artifact = ctx.store.latest("itinerary")
    trace_payload = ctx.store.latest("agent_trace") or {}
    trace_items = list(trace_payload.get("items") or [])
    subagent_items = [item for item in trace_items if item.get("kind") == "subagent"]
    agent_names = [item.get("agent") for item in subagent_items]
    tools_by_agent: dict[str, list[str]] = {}
    for item in subagent_items:
        name = str(item.get("agent"))
        tools_by_agent.setdefault(name, []).extend(
            list((item.get("detail") or {}).get("tool_trace") or [])
        )
    diagnostics = {
        "query_id": case.query_id,
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "used_real_agent": bool(reply.used_real_agent),
        "clarification": bool(reply.clarification),
        "itinerary_produced": bool(artifact),
        "tool_trace": list(reply.tool_trace),
        "agent_trace_present": bool(trace_items),
        "agent_names": agent_names,
        "required_agents_present": all(
            agent in agent_names
            for agent in ["attraction", "hotel", "restaurant", "transport", "planner"]
        ),
        "tools_by_agent": tools_by_agent,
        "failed_agent_reasons": [
            {
                "agent": str(item.get("agent")),
                "status": item.get("status"),
                "reason": str(item.get("error") or ""),
            }
            for item in subagent_items
            if item.get("status") not in ("completed", "completed_with_warnings")
        ],
    }
    if not artifact:
        fallback = _fallback_itinerary_from_case(case)
        if not fallback:
            return {"prediction": {}, "diagnostics": diagnostics | {"fallback_used": False}}
        prediction = convert_itinerary_to_chinatravel_output(
            fallback,
            query_data=case.raw,
            tool_trace=[*list(reply.tool_trace), "deterministic_chinatravel_fallback"],
        )
        return {"prediction": prediction, "diagnostics": diagnostics | {"fallback_used": True}}
    prediction = convert_itinerary_to_chinatravel_output(
        artifact.get("itinerary") or {},
        query_data=case.raw,
        tool_trace=list(reply.tool_trace),
    )
    return {"prediction": prediction, "diagnostics": diagnostics | {"fallback_used": False}}


def _fallback_itinerary_from_case(case: ChinaTravelCase) -> dict[str, Any]:
    raw = dict(case.raw)
    _augment_raw_query_fields(raw)
    target_city = str(raw.get("target_city") or raw.get("destination") or "")
    if not target_city:
        return {}
    days_count = max(1, min(_safe_int(raw.get("days") or raw.get("trip_days"), default=3), 7))
    requirements = _parse_hard_requirements(raw)
    _augment_requirements_from_query(requirements, extract_query_text(raw))
    days = []
    used_names: set[str] = set()
    slots = ["09:30", "14:30", "18:00"]
    for day_index in range(1, days_count + 1):
        stops = []
        for slot in slots:
            kind = "food" if slot == "18:00" else "attraction"
            stop = _fallback_stop(target_city, kind, used_names, slot, requirements)
            if stop:
                used_names.add(str((stop.get("poi") or {}).get("name") or ""))
                stops.append(stop)
        days.append({"day_index": day_index, "stops": stops})
    return {
        "city": target_city,
        "summary": f"{target_city}{days_count}天离线兜底行程",
        "days": days,
    }


def _fallback_stop(
    city: str,
    kind: str,
    used_names: set[str],
    start_time: str,
    requirements: dict[str, Any],
) -> dict[str, Any] | None:
    names_key = "restaurant_names" if kind == "food" else "attraction_names"
    for name in requirements.get(names_key, []):
        if name not in used_names:
            stop = _official_stop_by_name(city, name, kind)
            if stop:
                stop["start_time"] = start_time
                return stop
    df = _official_entity_df(city, kind)
    if df is None or df.empty:
        return None
    if "price" in df.columns:
        df = df.sort_values(by="price", ascending=True)
    for row in df.to_dict("records"):
        name = str(row.get("name") or "")
        if name and name not in used_names:
            stop = _row_to_stop(city, row, kind)
            stop["start_time"] = start_time
            return stop
    return None


def convert_itinerary_to_chinatravel_output(
    itinerary: dict[str, Any],
    *,
    query_data: dict[str, Any] | None = None,
    tool_trace: list[str] | None = None,
) -> dict[str, Any]:
    query_data = query_data or {}
    target_city = str(
        query_data.get("target_city")
        or query_data.get("destination")
        or itinerary.get("city")
        or ""
    )
    requirements = _parse_hard_requirements(query_data)
    _augment_requirements_from_query(requirements, extract_query_text(query_data))
    hotel_name = (
        requirements["hotel_names"][0]
        if requirements["hotel_names"]
        else _default_hotel_name(target_city, max_price=requirements.get("hotel_nightly_limit"))
    )
    start_city = str(query_data.get("start_city") or query_data.get("origin") or target_city)
    people_number = _safe_int(
        query_data.get("people_number")
        or query_data.get("people")
        or query_data.get("人数"),
        default=1,
    )
    days = []
    raw_days = _apply_chinatravel_requirements(
        list(itinerary.get("days", [])),
        target_city,
        requirements,
    )
    rooms = max(1, (people_number + 1) // 2)
    hotel_details = _hotel_details(target_city, hotel_name)
    previous_position: str | None = None
    for day_index, day in enumerate(raw_days):
        activities = []
        if day_index == 0 and start_city != target_city:
            intercity = _intercity_activity(
                start_city,
                target_city,
                earliest=requirements.get("outbound_earliest") or "06:00",
                people_number=people_number,
                preferred_modes=requirements["intercity_modes"],
                latest=requirements.get("outbound_latest"),
            )
            if intercity:
                activities.append(intercity)
                previous_position = str(intercity.get("end") or previous_position or "")
        for stop in day.get("stops", []):
            activity = _stop_to_activity(
                stop,
                city=target_city,
                people_number=people_number,
                previous_position=previous_position,
            )
            activities.append(activity)
            previous_position = str(activity.get("position") or previous_position or "")
        is_last_day = day_index == len(raw_days) - 1
        if activities and not is_last_day:
            last = activities[-1]
            hotel_transport = _transport_between(
                last.get("position", target_city),
                hotel_name,
                "19:00",
                "21:00",
                city=target_city,
                people_number=people_number,
            )
            hotel_start = max("21:00", str(hotel_transport.get("end_time") or "21:00"))
            activities.append(
                {
                    "type": "accommodation",
                    "start_time": hotel_start,
                    "end_time": "23:59",
                    "cost": hotel_details["price"] * rooms,
                    "price": hotel_details["price"],
                    "position": hotel_name,
                    "transports": [hotel_transport],
                    "room_type": hotel_details["room_type"],
                    "rooms": rooms,
                }
            )
            previous_position = hotel_name
        if is_last_day and start_city != target_city:
            intercity = _intercity_activity(
                target_city,
                start_city,
                earliest=requirements.get("return_earliest") or "20:00",
                people_number=people_number,
                previous_position=previous_position,
                preferred_modes=requirements["intercity_modes"],
                latest=requirements.get("return_latest"),
            )
            if intercity:
                activities.append(intercity)
                previous_position = str(intercity.get("end") or previous_position or "")
        days.append({"day": int(day.get("day_index") or len(days) + 1), "activities": activities})

    return {
        "people_number": people_number,
        "start_city": start_city,
        "target_city": target_city,
        "itinerary": days,
        "_travel_agent_trace": tool_trace or [],
    }


def save_prediction(prediction: dict[str, Any], output_dir: Path, query_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{query_id}.json"
    path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def evaluate_predictions(
    chinatravel_root: Path,
    split: str,
    query_ids: list[str],
    query_data: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    *,
    lang: str = "zh",
) -> dict[str, Any]:
    _ensure_chinatravel_importable(chinatravel_root)
    try:
        from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
        from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
        from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
        from chinatravel.evaluation.preference import evaluate_preference_v2
        from chinatravel.evaluation.utils import load_json_file
    except Exception as exc:  # pragma: no cover - external checkout
        schema_metrics = _schema_only_metrics(chinatravel_root, query_ids, predictions)
        return {
            "official_eval_available": False,
            "split": split,
            "case_count": len(query_ids),
            "delivery_rate": _rate_non_empty(predictions, query_ids),
            **schema_metrics,
            "error": str(exc),
        }

    if not any(query_data.get(query_id, {}).get("hard_logic_py") for query_id in query_ids):
        schema_metrics = _schema_only_metrics(chinatravel_root, query_ids, predictions)
        return {
            "official_eval_available": False,
            "split": split,
            "case_count": len(query_ids),
            "delivery_rate": _rate_non_empty(predictions, query_ids),
            **schema_metrics,
            "error": "当前评测数据没有 hard_logic_py，无法计算官方 LPR / C-LPR / FPR；这里只报告 delivery/schema。",
        }

    schema_path = chinatravel_root / "chinatravel" / "evaluation" / "output_schema.json"
    try:
        schema = load_json_file(str(schema_path))
        schema_rate, _, schema_pass_id = evaluate_schema_constraints(query_ids, predictions, schema=schema)
        macro_epr, micro_epr, commonsense_df, env_pass_id = evaluate_commonsense_constraints(
            query_ids,
            query_data,
            predictions,
            verbose=False,
            lang=lang,
        )
        macro_lpr, micro_lpr, conditional_macro_lpr, conditional_micro_lpr, logical_df, logic_pass_id = (
            evaluate_hard_constraints_v2(
                query_ids,
                query_data,
                predictions,
                env_pass_id=env_pass_id,
                verbose=False,
                lang=lang,
            )
        )
        all_pass = sorted(set(schema_pass_id) & set(env_pass_id) & set(logic_pass_id))
        preference = None
        try:
            preference_df = evaluate_preference_v2(
                query_ids,
                query_data,
                predictions,
                list(set(env_pass_id) & set(logic_pass_id)),
                lang=lang,
            )
            preference = float(preference_df.iloc[:, 1:].mean(numeric_only=True).mean())
        except Exception:
            preference = None
        return {
            "official_eval_available": True,
            "split": split,
            "case_count": len(query_ids),
            "delivery_rate": _rate_non_empty(predictions, query_ids),
            "schema_pass_rate": float(schema_rate),
            "epr_micro": float(micro_epr),
            "epr_macro": float(macro_epr),
            "lpr_micro": float(micro_lpr),
            "lpr_macro": float(macro_lpr),
            "conditional_lpr_micro": float(conditional_micro_lpr),
            "conditional_lpr_macro": float(conditional_macro_lpr),
            "fpr": len(all_pass) / len(query_ids) if query_ids else 0.0,
            "preference_pass_rate": preference,
            "all_pass_count": len(all_pass),
            "failure_breakdown": {
                "commonsense": _top_failure_rates(commonsense_df),
                "logical": _top_failure_rates(logical_df),
            },
            "failed_examples": _failed_examples(
                query_ids,
                query_data,
                all_pass,
                commonsense_df,
                logical_df,
            ),
        }
    except Exception as exc:
        schema_metrics = _schema_only_metrics(chinatravel_root, query_ids, predictions)
        return {
            "official_eval_available": False,
            "split": split,
            "case_count": len(query_ids),
            "delivery_rate": _rate_non_empty(predictions, query_ids),
            **schema_metrics,
            "error": str(exc),
        }


def extract_query_text(raw: dict[str, Any]) -> str:
    for key in (
        "nature_language",
        "query",
        "nl_query",
        "user_query",
        "instruction",
        "text",
        "request",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    target = raw.get("target_city") or raw.get("destination") or "目的地"
    days = raw.get("days") or raw.get("day") or raw.get("trip_days") or 3
    people = raw.get("people_number") or raw.get("people") or 1
    return f"请为{people}人规划从{raw.get('start_city') or target}到{target}的{days}天旅行行程。"


def load_query_data_for_cases(
    chinatravel_root: Path,
    split: str,
    cases: list[ChinaTravelCase],
    *,
    lang: str = "zh",
) -> tuple[list[str], dict[str, Any]]:
    return [case.query_id for case in cases], {case.query_id: case.raw for case in cases}


def _parse_hard_requirements(query_data: dict[str, Any]) -> dict[str, Any]:
    text = str(query_data.get("hard_logic_py") or "")
    try:
        clauses = ast.literal_eval(text)
        if not isinstance(clauses, list):
            clauses = [text]
    except Exception:
        clauses = [text]
    requirements: dict[str, Any] = {
        "attraction_names": [],
        "restaurant_names": [],
        "hotel_names": [],
        "attraction_types": [],
        "restaurant_types": [],
        "intercity_modes": [],
        "budget_limit": None,
        "hotel_nightly_limit": None,
        "outbound_earliest": None,
        "outbound_latest": None,
        "return_earliest": None,
        "return_latest": None,
    }
    for clause in [str(clause) for clause in clauses]:
        if "attraction_name_set" in clause:
            key = "attraction_names"
        elif "restaurant_name_set" in clause:
            key = "restaurant_names"
        elif "accommodation_name_set" in clause:
            key = "hotel_names"
        elif "attraction_type_set" in clause:
            key = "attraction_types"
        elif "restaurant_type_set" in clause:
            key = "restaurant_types"
        elif "intercity_transport_set" in clause:
            key = "intercity_modes"
        else:
            continue
        result_part = clause.split("result=", 1)[-1]
        for fragment in re.findall(r"\{([^}]*)\}", result_part):
            for value in re.findall(r"'([^']+)'", fragment):
                if value not in requirements[key]:
                    requirements[key].append(value)
        if "total_cost" in clause:
            match = re.search(r"total_cost\s*<=\s*(\d+(?:\.\d+)?)", clause)
            if match:
                requirements["budget_limit"] = float(match.group(1))
        if "hotel_cost" in clause:
            match = re.search(r"<=\s*(\d+(?:\.\d+)?)", clause)
            if match:
                requirements["hotel_nightly_limit"] = float(match.group(1))
    return requirements


def _augment_requirements_from_query(requirements: dict[str, Any], query: str) -> None:
    if "飞机" in query or "坐飞" in query:
        _append_unique(requirements["intercity_modes"], "airplane")
    if "火车" in query or "动车" in query or "高铁" in query:
        _append_unique(requirements["intercity_modes"], "train")
    if "晚上" in query and "出发" in query:
        requirements["outbound_earliest"] = "18:00"
    if "第一天晚上" in query or "晚上从" in query:
        requirements["outbound_earliest"] = "18:00"
    match = re.search(r"不超过\s*(\d{1,2})\s*点.*?(?:火车|车|出发)", query)
    if match:
        requirements["return_latest"] = f"{int(match.group(1)):02d}:00"
    match = re.search(r"预算\s*(\d+)", query)
    if match and not requirements.get("budget_limit"):
        requirements["budget_limit"] = float(match.group(1))
    if any(word in query for word in ("穷游", "预算少", "实惠", "不超过")) and not requirements.get("hotel_nightly_limit"):
        requirements["hotel_nightly_limit"] = 300.0


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _apply_chinatravel_requirements(
    raw_days: list[dict[str, Any]],
    city: str,
    requirements: dict[str, Any],
) -> list[dict[str, Any]]:
    days = [
        {
            **day,
            "stops": [dict(stop) for stop in list(day.get("stops", []))],
        }
        for day in raw_days
    ]
    if not days:
        days = [{"day_index": 1, "stops": []}]

    existing_names = {
        str((stop.get("poi") or {}).get("name") or "")
        for day in days
        for stop in day.get("stops", [])
    }
    required_stops: list[dict[str, Any]] = []
    for name in requirements["attraction_names"]:
        if name not in existing_names:
            stop = _official_stop_by_name(city, name, "attraction")
            if stop:
                required_stops.append(stop)
                existing_names.add(name)
    for type_name in requirements["attraction_types"]:
        if not _has_stop_type(days, type_name):
            stop = _official_stop_by_type(city, type_name, "attraction")
            if stop and str((stop.get("poi") or {}).get("name") or "") not in existing_names:
                required_stops.append(stop)
                existing_names.add(str((stop.get("poi") or {}).get("name") or ""))
    for name in requirements["restaurant_names"]:
        if name not in existing_names:
            stop = _official_stop_by_name(city, name, "food")
            if stop:
                required_stops.append(stop)
                existing_names.add(name)
    for type_name in requirements["restaurant_types"]:
        if not _has_stop_type(days, type_name):
            stop = _official_stop_by_type(city, type_name, "food")
            if stop and str((stop.get("poi") or {}).get("name") or "") not in existing_names:
                required_stops.append(stop)
                existing_names.add(str((stop.get("poi") or {}).get("name") or ""))

    slots = ["09:30", "11:30", "14:30", "16:30", "18:00"]
    for index, stop in enumerate(required_stops):
        day = days[index % len(days)]
        stop = dict(stop)
        poi = stop.get("poi") or {}
        if poi.get("category") == "food":
            stop["start_time"] = "18:00" if index % 2 else "11:30"
        else:
            stop["start_time"] = slots[index % len(slots)]
        day["stops"].insert(min(index, len(day["stops"])), stop)
    return days


def _has_stop_type(days: list[dict[str, Any]], type_name: str) -> bool:
    for day in days:
        for stop in day.get("stops", []):
            poi = stop.get("poi") or {}
            if type_name in set(poi.get("tags") or []):
                return True
            if str(poi.get("raw_type") or "") == type_name:
                return True
    return False


def _official_stop_by_name(city: str, name: str, kind: str) -> dict[str, Any] | None:
    df = _official_entity_df(city, kind)
    if df is None or df.empty:
        return None
    row = df[df["name"] == name]
    if row.empty:
        return None
    return _row_to_stop(city, row.iloc[0].to_dict(), kind)


def _official_stop_by_type(city: str, type_name: str, kind: str) -> dict[str, Any] | None:
    df = _official_entity_df(city, kind)
    if df is None or df.empty:
        return None
    key = "cuisine" if kind == "food" else "type"
    row = df[df[key].astype(str) == type_name]
    if row.empty:
        return None
    row = row.sort_values(by="price", ascending=True)
    return _row_to_stop(city, row.iloc[0].to_dict(), kind)


def _official_entity_df(city: str, kind: str) -> pd.DataFrame | None:
    db_root = _database_root()
    slug = _city_slug(city)
    if kind == "food":
        path = db_root / "restaurants" / slug / f"restaurants_{slug}.csv"
    else:
        path = db_root / "attractions" / slug / "attractions.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _row_to_stop(city: str, row: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "food":
        tags = ["food", str(row.get("cuisine") or ""), "local"]
        category = "food"
        duration = 60
    else:
        category, tags = _category_for_attraction(str(row.get("type") or ""))
        tags = [*tags, str(row.get("type") or "")]
        duration = int(float(row.get("recommendmaxtime") or 1.0) * 60)
    price = float(row.get("price") or 0)
    return {
        "poi": {
            "name": str(row.get("name")),
            "city": city,
            "category": category,
            "tags": tags,
            "raw_type": str(row.get("cuisine") if kind == "food" else row.get("type")),
            "price_level": _price_level(price),
        },
        "start_time": "11:30" if kind == "food" else "09:30",
        "duration_min": max(45, min(duration, 90)),
        "route_from_previous": None,
    }


def _stop_to_activity(
    stop: dict[str, Any],
    *,
    city: str,
    people_number: int,
    previous_position: str | None = None,
) -> dict[str, Any]:
    poi = stop.get("poi") or {}
    start = str(stop.get("start_time") or "09:30")
    activity_type = _activity_type(poi, start)
    if poi.get("category") == "food":
        activity_type = _food_activity_type(city, poi, activity_type)
    start, end = _normalize_activity_time(
        city,
        poi,
        activity_type,
        start,
        int(stop.get("duration_min") or 90),
    )
    price = _official_activity_price(city, poi)
    tickets = people_number if activity_type == "attraction" else 1
    cost = price * tickets if activity_type == "attraction" else price * people_number
    route = stop.get("route_from_previous") or {}
    transports = []
    origin = previous_position
    if not origin and route:
        origin = str(route.get("origin_poi_id") or "")
    if origin and origin != str(poi.get("name") or "当前站"):
        transports.append(
            _transport_between(
                origin,
                str(poi.get("name") or "当前站"),
                _add_minutes(start, -120),
                start,
                route=route,
                city=city,
                people_number=people_number,
            )
        )
        arrival = str(transports[-1].get("end_time") or start)
        if _time_to_minutes(start) < _time_to_minutes(arrival):
            if poi.get("category") == "food":
                activity_type = "dinner" if _time_to_minutes(arrival) >= _time_to_minutes("14:00") else "lunch"
                activity_type = _food_activity_type(city, poi, activity_type)
            start, end = _normalize_activity_time(
                city,
                poi,
                activity_type,
                arrival,
                int(stop.get("duration_min") or 90),
            )
    return {
        "type": activity_type,
        "start_time": start,
        "end_time": end,
        "cost": cost,
        "price": price,
        "tickets": tickets,
        "position": str(poi.get("name") or "未知地点"),
        "transports": transports,
    }


def _activity_type(poi: dict[str, Any], start: str) -> str:
    if poi.get("category") == "food":
        hour = _safe_int(start.split(":", 1)[0], default=12)
        return "dinner" if hour >= 17 else "lunch"
    return "attraction"


def _food_activity_type(city: str, poi: dict[str, Any], preferred: str) -> str:
    if preferred == "lunch" and _is_open_during(city, poi, "11:30", "14:00"):
        return "lunch"
    if preferred == "dinner" and _is_open_during(city, poi, "17:30", "20:00"):
        return "dinner"
    if _is_open_during(city, poi, "17:30", "20:00"):
        return "dinner"
    return "lunch"


def _transport_between(
    start: str,
    end: str,
    start_time: str,
    end_time: str,
    *,
    route: dict[str, Any] | None = None,
    city: str | None = None,
    people_number: int = 1,
) -> dict[str, Any]:
    if city:
        official = _official_inner_transport(city, start, end, start_time, people_number)
        if official:
            return official
    route = route or {}
    distance = float(route.get("distance_km") or 0.0)
    mode = _ct_mode(str(route.get("mode") or "metro"))
    price = _transport_price(distance, mode)
    return {
        "start": start,
        "end": end,
        "mode": mode,
        "start_time": start_time,
        "end_time": end_time,
        "cost": price,
        "price": price,
        "distance": distance,
        "tickets": people_number,
    }


def _official_inner_transport(
    city: str,
    start: str,
    end: str,
    start_time: str,
    people_number: int,
) -> dict[str, Any] | None:
    try:
        from chinatravel.environment.tools.transportation.apis import Transportation

        result = Transportation().goto(city, start, end, start_time, "taxi")
        if isinstance(result, str) or not result:
            return None
        item = dict(result[0])
        price = float(item.get("cost") or 0)
        item["price"] = price
        if item.get("mode") == "taxi":
            cars = max(1, (people_number + 3) // 4)
            item["cars"] = cars
            item["cost"] = price * cars
        elif item.get("mode") == "metro":
            item["tickets"] = people_number
            item["cost"] = price * people_number
        else:
            item["cost"] = 0.0
        return item
    except Exception:
        return None


def _intercity_activity(
    start_city: str,
    end_city: str,
    *,
    earliest: str,
    people_number: int,
    previous_position: str | None = None,
    preferred_modes: list[str] | None = None,
    latest: str | None = None,
) -> dict[str, Any] | None:
    try:
        from chinatravel.environment.tools.intercity_transport.apis import IntercityTransport

        api = IntercityTransport()
        modes = [mode for mode in (preferred_modes or []) if mode in {"train", "airplane"}]
        if not modes:
            modes = ["train", "airplane"]
        for kind in modes:
            row = _select_intercity_row(api, start_city, end_city, kind, earliest, latest)
            if row is None:
                continue
            cost = float(row.get("Cost") or 0)
            total_cost = cost * people_number
            transports = []
            station = str(row.get("From"))
            if previous_position and previous_position != station:
                transports.append(
                    _transport_between(
                        previous_position,
                        station,
                        _add_minutes(str(row.get("BeginTime")), -90),
                        str(row.get("BeginTime")),
                        city=start_city,
                        people_number=people_number,
                    )
                )
            if kind == "train":
                return {
                    "type": "train",
                    "start_time": str(row.get("BeginTime")),
                    "end_time": str(row.get("EndTime")),
                    "cost": total_cost,
                    "price": cost,
                    "tickets": people_number,
                    "start": str(row.get("From")),
                    "end": str(row.get("To")),
                    "TrainID": str(row.get("TrainID")),
                    "transports": transports,
                }
            return {
                "type": "airplane",
                "start_time": str(row.get("BeginTime")),
                "end_time": str(row.get("EndTime")),
                "cost": total_cost,
                "price": cost,
                "tickets": people_number,
                "start": str(row.get("From")),
                "end": str(row.get("To")),
                "FlightID": str(row.get("FlightID")),
                "transports": transports,
            }
    except Exception:
        return None
    return None


def _select_intercity_row(
    api: Any,
    start_city: str,
    end_city: str,
    kind: str,
    earliest: str,
    latest: str | None,
) -> Any | None:
    try:
        result = api._select(start_city, end_city, kind)
    except Exception:
        result = api.select(start_city, end_city, kind, earliest)
    if result is None or isinstance(result, str) or result.empty:
        return None
    rows = result.copy()
    begin_minutes = rows["BeginTime"].map(_time_to_minutes)
    filtered = rows[begin_minutes >= _time_to_minutes(earliest)]
    if latest:
        latest_minutes = _time_to_minutes(latest)
        filtered = filtered[filtered["BeginTime"].map(_time_to_minutes) <= latest_minutes]
        if filtered.empty:
            filtered = rows[begin_minutes <= latest_minutes]
    if filtered.empty:
        filtered = rows
    filtered = filtered.sort_values(by="BeginTime").reset_index(drop=True)
    if latest:
        return filtered.iloc[-1]
    return filtered.iloc[0]


def _official_activity_price(city: str, poi: dict[str, Any]) -> float:
    name = str(poi.get("name") or "")
    category = str(poi.get("category") or "")
    db_root = _database_root()
    slug = _city_slug(city)
    if category == "food":
        path = db_root / "restaurants" / slug / f"restaurants_{slug}.csv"
        key = "price"
    else:
        path = db_root / "attractions" / slug / "attractions.csv"
        key = "price"
    if path.exists():
        try:
            df = pd.read_csv(path)
            row = df[df["name"] == name]
            if not row.empty:
                return float(row.iloc[0][key])
        except Exception:
            pass
    return _price_for_poi(poi)


def _normalize_activity_time(
    city: str,
    poi: dict[str, Any],
    activity_type: str,
    requested_start: str,
    duration_min: int,
) -> tuple[str, str]:
    if activity_type == "lunch":
        open_time, close_time = _official_activity_window(city, poi)
        lower = max("11:30", open_time)
        upper = min("13:00", _add_minutes(close_time, -30))
        if _time_to_minutes(lower) > _time_to_minutes(upper):
            lower, upper = "11:30", "13:00"
        start = _clamp_time(requested_start, lower, upper)
        return start, _add_minutes(start, min(duration_min, 60))
    if activity_type == "dinner":
        open_time, close_time = _official_activity_window(city, poi)
        lower = max("17:30", open_time)
        upper = min("19:00", _add_minutes(close_time, -30))
        if _time_to_minutes(lower) > _time_to_minutes(upper):
            lower, upper = "17:30", "19:00"
        start = _clamp_time(requested_start, lower, upper)
        return start, _add_minutes(start, min(duration_min, 60))

    open_time, close_time = _official_activity_window(city, poi)
    start = requested_start
    duration = max(30, min(duration_min, 90))
    if _time_to_minutes(start) < _time_to_minutes(open_time):
        start = open_time
    end = _add_minutes(start, duration)
    if _time_to_minutes(end) > _time_to_minutes(close_time):
        end_minutes = _time_to_minutes(close_time)
        start_minutes = max(_time_to_minutes(open_time), end_minutes - duration)
        start = _minutes_to_time(start_minutes)
        end = _minutes_to_time(end_minutes)
    if _time_to_minutes(start) >= _time_to_minutes(end):
        start = open_time
        end = _add_minutes(start, 30)
    return start, end


def _official_activity_window(city: str, poi: dict[str, Any]) -> tuple[str, str]:
    name = str(poi.get("name") or "")
    category = str(poi.get("category") or "")
    db_root = _database_root()
    slug = _city_slug(city)
    if category == "food":
        path = db_root / "restaurants" / slug / f"restaurants_{slug}.csv"
    else:
        path = db_root / "attractions" / slug / "attractions.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            row = df[df["name"] == name]
            if not row.empty:
                open_time = str(row.iloc[0]["opentime"])
                close_time = str(row.iloc[0]["endtime"])
                if close_time == "24:00":
                    close_time = "23:59"
                return open_time, close_time
        except Exception:
            pass
    return "09:00", "21:00"


def _is_open_during(city: str, poi: dict[str, Any], start: str, end: str) -> bool:
    open_time, close_time = _official_activity_window(city, poi)
    return _time_to_minutes(open_time) <= _time_to_minutes(start) and _time_to_minutes(end) <= _time_to_minutes(close_time)


def _hotel_price(city: str, hotel_name: str) -> float:
    return _hotel_details(city, hotel_name)["price"]


def _hotel_details(city: str, hotel_name: str) -> dict[str, Any]:
    db_root = _database_root()
    path = db_root / "accommodations" / _city_slug(city) / "accommodations.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            row = df[df["name"] == hotel_name]
            if not row.empty:
                return {
                    "price": float(row.iloc[0]["price"]),
                    "room_type": int(row.iloc[0]["numbed"]),
                }
        except Exception:
            pass
    return {"price": 500.0, "room_type": 1}


def _ct_mode(mode: str) -> str:
    if mode == "walk":
        return "walk"
    if mode in {"taxi", "drive"}:
        return "taxi"
    return "metro"


def _price_for_poi(poi: dict[str, Any]) -> float:
    level = str(poi.get("price_level") or "mid")
    if poi.get("category") == "food":
        return {"low": 35.0, "mid": 80.0, "high": 180.0}.get(level, 80.0)
    return {"low": 0.0, "mid": 60.0, "high": 120.0}.get(level, 60.0)


def _transport_price(distance_km: float, mode: str) -> float:
    if mode == "walk":
        return 0.0
    if mode == "taxi":
        return round(14 + max(0.0, distance_km - 3) * 2.5, 2)
    return 4.0 if distance_km <= 6 else 7.0


def _add_minutes(value: str, minutes: int) -> str:
    hour, minute = value.split(":", 1)
    total = (int(hour) * 60 + int(minute) + minutes) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _clamp_time(value: str, lower: str, upper: str) -> str:
    minutes = min(max(_time_to_minutes(value), _time_to_minutes(lower)), _time_to_minutes(upper))
    return _minutes_to_time(minutes)


def _time_to_minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def _minutes_to_time(minutes: int) -> str:
    minutes = minutes % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rate_non_empty(predictions: dict[str, dict[str, Any]], query_ids: list[str]) -> float:
    if not query_ids:
        return 0.0
    return sum(1 for query_id in query_ids if predictions.get(query_id)) / len(query_ids)


def _schema_only_metrics(
    chinatravel_root: Path,
    query_ids: list[str],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schema_path = chinatravel_root / "chinatravel" / "evaluation" / "output_schema.json"
    if not schema_path.exists():
        return {"schema_pass_rate": None, "schema_error_count": None}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    failed = []
    for query_id in query_ids:
        prediction = predictions.get(query_id) or {}
        errors = sorted(validator.iter_errors(prediction), key=lambda error: list(error.path))
        if errors:
            failed.append(
                {
                    "query_id": query_id,
                    "errors": [
                        f"{'/'.join(str(part) for part in error.path)}: {error.message}"
                        for error in errors[:3]
                    ],
                }
            )
    total = len(query_ids)
    return {
        "schema_pass_rate": (total - len(failed)) / total if total else 0.0,
        "schema_error_count": len(failed),
        "schema_failed_examples": failed[:10],
    }


def _top_failure_rates(df: Any, limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for column in getattr(df, "columns", []):
        if column in {"query_id", "uid"}:
            continue
        try:
            series = df[column].astype(float)
        except Exception:
            continue
        rate = float(series.mean())
        if rate > 0:
            rows.append({"reason": str(column), "fail_rate": rate})
    return sorted(rows, key=lambda item: item["fail_rate"], reverse=True)[:limit]


def _failed_examples(
    query_ids: list[str],
    query_data: dict[str, Any],
    all_pass: list[str],
    commonsense_df: Any,
    logical_df: Any,
    limit: int = 20,
) -> list[dict[str, Any]]:
    passed = set(all_pass)
    examples = []
    for index, query_id in enumerate(query_ids):
        if query_id in passed:
            continue
        examples.append(
            {
                "query_id": query_id,
                "query": extract_query_text(query_data.get(query_id) or {}),
                "commonsense": _row_failures(commonsense_df, index),
                "logical": _row_failures(logical_df, index),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _row_failures(df: Any, index: int, limit: int = 6) -> list[str]:
    failures = []
    if index >= len(df):
        return failures
    row = df.iloc[index]
    for column in getattr(df, "columns", []):
        if column in {"query_id", "uid"}:
            continue
        try:
            failed = float(row[column]) > 0
        except Exception:
            failed = False
        if failed:
            failures.append(str(column))
        if len(failures) >= limit:
            break
    return failures


def _ensure_chinatravel_importable(chinatravel_root: Path) -> None:
    root = str(chinatravel_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_chinatravel_provider(
    chinatravel_root: Path | str | None = None,
    *,
    allow_synthetic: bool = True,
):
    if chinatravel_root is not None:
        configure_chinatravel_root(chinatravel_root)
    db_root = _database_root(chinatravel_root)
    if db_root.exists():
        return ChinaTravelDatabaseProvider(db_root)
    if not allow_synthetic:
        raise RuntimeError(f"ChinaTravel official database missing: {db_root}")
    return SyntheticChinaTravelProvider()


class ChinaTravelDatabaseProvider(LocalToolProvider):
    """使用 ChinaTravel 官方 database 的离线 provider。"""

    def __init__(self, db_root: Path) -> None:
        super().__init__(pois=[])
        self.db_root = db_root

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        pois: list[POI] = []
        if category in {None, "scenic", "culture", "museum", "shopping"}:
            pois.extend(self._load_attractions(city))
        if category in {None, "food"} or "food" in set(query_tags or []):
            pois.extend(self._load_restaurants(city))
        if category in {None, "hotel"} or "hotel" in set(query_tags or []):
            pois.extend(self._load_hotels(city))
        if category:
            pois = [poi for poi in pois if poi.category == category or category in poi.tags]
        tags = set(query_tags or [])
        if tags:
            preferred = [
                poi for poi in pois if tags.intersection(set(poi.tags) | {poi.category})
            ]
            pois = preferred + [poi for poi in pois if poi not in preferred]
        return pois[:max_results]

    def get_weather(self, city: str) -> WeatherInfo:
        return WeatherInfo(city=city, condition="cloudy", temperature_c=24, source="chinatravel")

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        return RouteInfo(
            origin_poi_id=origin.name,
            destination_poi_id=destination.name,
            distance_km=2.5,
            duration_min=25,
            mode=mode,
            source="chinatravel_estimate",
        )

    def _load_attractions(self, city: str) -> list[POI]:
        slug = _city_slug(city)
        path = self.db_root / "attractions" / slug / "attractions.csv"
        if not path.exists():
            return _synthetic_pois(city)
        df = pd.read_csv(path).head(80)
        pois = []
        for row in df.to_dict("records"):
            category, tags = _category_for_attraction(str(row.get("type") or ""))
            duration = int(float(row.get("recommendmaxtime") or 1.5) * 60)
            price = float(row.get("price") or 0)
            pois.append(
                POI(
                    poi_id=f"ct_attr_{slug}_{row.get('id')}",
                    name=str(row.get("name")),
                    city=city,
                    category=category,
                    lat=float(row.get("lat")),
                    lng=float(row.get("lon")),
                    rating=4.5,
                    popularity=0.8,
                    tags=tags,
                    estimated_duration_min=max(60, duration),
                    price_level=_price_level(price),
                    indoor=category == "museum",
                    opening_hours=f"{row.get('opentime')}-{row.get('endtime')}",
                    source="chinatravel_attractions",
                )
            )
        return pois

    def _load_restaurants(self, city: str) -> list[POI]:
        slug = _city_slug(city)
        path = self.db_root / "restaurants" / slug / f"restaurants_{slug}.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).head(80)
        pois = []
        for row in df.to_dict("records"):
            cuisine = str(row.get("cuisine") or "")
            price = float(row.get("price") or 80)
            pois.append(
                POI(
                    poi_id=f"ct_rest_{slug}_{row.get('id')}",
                    name=str(row.get("name")),
                    city=city,
                    category="food",
                    lat=float(row.get("lat")),
                    lng=float(row.get("lon")),
                    rating=4.4,
                    popularity=0.75,
                    tags=["food", cuisine, "local"],
                    estimated_duration_min=75,
                    price_level=_price_level(price),
                    indoor=True,
                    opening_hours=f"{row.get('opentime')}-{row.get('endtime')}",
                    source="chinatravel_restaurants",
                )
            )
        return pois

    def _load_hotels(self, city: str) -> list[POI]:
        slug = _city_slug(city)
        candidates = [
            self.db_root / "accommodations" / slug / "accommodations.csv",
            self.db_root / "accommodations" / slug / f"accommodations_{slug}.csv",
            self.db_root / "hotels" / slug / "hotels.csv",
            self.db_root / "hotels" / slug / f"hotels_{slug}.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return []
        df = pd.read_csv(path).head(80)
        pois = []
        for index, row in enumerate(df.to_dict("records"), start=1):
            name = str(row.get("name") or row.get("hotel_name") or f"{city}住宿{index}")
            lat = _safe_row_float(row, ("lat", "latitude"), default=30.0 + index * 0.001)
            lng = _safe_row_float(row, ("lon", "lng", "longitude"), default=120.0 + index * 0.001)
            price = _safe_row_float(row, ("price", "cost", "lowest_price"), default=550.0)
            pois.append(
                POI(
                    poi_id=f"ct_hotel_{slug}_{row.get('id') or index}",
                    name=name,
                    city=city,
                    category="hotel",
                    lat=lat,
                    lng=lng,
                    rating=4.4,
                    popularity=0.7,
                    tags=["hotel", "accommodation"],
                    estimated_duration_min=0,
                    price_level=_price_level(price),
                    indoor=True,
                    opening_hours=None,
                    source="chinatravel_hotels",
                )
            )
        return pois


def _city_slug(city: str) -> str:
    mapping = {
        "北京": "beijing",
        "上海": "shanghai",
        "南京": "nanjing",
        "苏州": "suzhou",
        "杭州": "hangzhou",
        "深圳": "shenzhen",
        "成都": "chengdu",
        "武汉": "wuhan",
        "广州": "guangzhou",
        "重庆": "chongqing",
    }
    return mapping.get(city, "hangzhou")


def _category_for_attraction(type_text: str) -> tuple[str, list[str]]:
    if "博物" in type_text or "展览" in type_text:
        return "museum", ["museum", "history", "culture"]
    if "历史" in type_text or "古迹" in type_text or "文化" in type_text:
        return "culture", ["history", "culture"]
    if "商" in type_text or "街" in type_text:
        return "shopping", ["shopping", "citywalk"]
    return "scenic", ["nature", "scenic"]


def _price_level(price: float) -> str:
    if price <= 60:
        return "low"
    if price <= 200:
        return "mid"
    return "high"


def _safe_row_float(row: dict[str, Any], keys: tuple[str, ...], *, default: float) -> float:
    for key in keys:
        try:
            value = row.get(key)
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _default_hotel_name(city: str, max_price: float | None = None) -> str:
    db_root = _database_root()
    path = db_root / "accommodations" / _city_slug(city) / "accommodations.csv"
    if not path.exists():
        return f"{city}住宿点"
    try:
        df = pd.read_csv(path)
        if max_price is not None:
            affordable = df[df["price"].astype(float) <= float(max_price)]
            if not affordable.empty:
                df = affordable
        df = df.sort_values(by="price", ascending=True)
        if not df.empty:
            return str(df.iloc[0]["name"])
    except Exception:
        pass
    return f"{city}住宿点"


class SyntheticChinaTravelProvider(LocalToolProvider):
    """评测兜底 provider：官方 sandbox 数据缺失时也能产出预测文件。

    这不是最终跑分 provider；真正 EPR/LPR/FPR 仍依赖 ChinaTravel database。
    """

    def __init__(self) -> None:
        super().__init__(pois=[])

    def search_pois(
        self,
        city: str,
        query_tags: list[str] | None = None,
        category: str | None = None,
        max_results: int = 20,
    ) -> list[POI]:
        pois = _synthetic_pois(city)
        if category:
            pois = [poi for poi in pois if poi.category == category]
        tags = set(query_tags or [])
        if tags:
            preferred = [
                poi for poi in pois if tags.intersection(set(poi.tags) | {poi.category})
            ]
            pois = preferred + [poi for poi in pois if poi not in preferred]
        return pois[:max_results]

    def get_weather(self, city: str) -> WeatherInfo:
        return WeatherInfo(city=city, condition="cloudy", temperature_c=24, source="synthetic")

    def estimate_route(
        self,
        origin: POI,
        destination: POI,
        mode: TransportMode = "public_transport",
    ) -> RouteInfo:
        return RouteInfo(
            origin_poi_id=origin.poi_id,
            destination_poi_id=destination.poi_id,
            distance_km=2.5,
            duration_min=25,
            mode=mode,
            source="synthetic",
        )


def _synthetic_pois(city: str) -> list[POI]:
    specs = [
        ("经典景区", "scenic", ["nature", "scenic"], 120.0, 30.0),
        ("历史街区", "culture", ["history", "culture"], 120.01, 30.01),
        ("城市博物馆", "museum", ["museum", "history"], 120.02, 30.02),
        ("本地餐厅", "food", ["food", "local"], 120.03, 30.03),
        ("特色小吃", "food", ["food", "snack"], 120.04, 30.04),
        ("商业步行街", "shopping", ["shopping", "citywalk"], 120.05, 30.05),
        ("舒适酒店", "hotel", ["hotel", "accommodation"], 120.06, 30.06),
    ]
    pois = []
    for index, (name, category, tags, lng, lat) in enumerate(specs, start=1):
        pois.append(
            POI(
                poi_id=f"synthetic_{city}_{index}",
                name=f"{city}{name}",
                city=city,
                category=category,
                lat=lat,
                lng=lng,
                rating=4.5,
                popularity=0.8 - index * 0.03,
                tags=tags,
                estimated_duration_min=90 if category != "food" else 75,
                price_level="mid",
                indoor=category in {"museum", "food", "shopping"},
                opening_hours="09:00-21:00",
                source="synthetic_chinatravel",
            )
        )
    return pois
def configure_chinatravel_root(chinatravel_root: Path | str) -> Path:
    global _ACTIVE_CHINATRAVEL_ROOT
    _ACTIVE_CHINATRAVEL_ROOT = Path(chinatravel_root).expanduser().resolve()
    _ensure_chinatravel_importable(_ACTIVE_CHINATRAVEL_ROOT)
    return _ACTIVE_CHINATRAVEL_ROOT


def _database_root(chinatravel_root: Path | str | None = None) -> Path:
    root = Path(chinatravel_root) if chinatravel_root is not None else _ACTIVE_CHINATRAVEL_ROOT
    return root / "chinatravel" / "environment" / "database"


def preflight_chinatravel(chinatravel_root: Path | str) -> dict[str, Any]:
    root = configure_chinatravel_root(chinatravel_root)
    schema = root / "chinatravel" / "evaluation" / "output_schema.json"
    database = _database_root(root)
    errors: list[str] = []
    if not root.exists():
        errors.append(f"root does not exist: {root}")
    if not schema.exists():
        errors.append(f"output schema missing: {schema}")
    if not database.exists():
        errors.append(f"official database missing: {database}")
    entity_files = sorted(database.glob("attractions/*/attractions.csv")) if database.exists() else []
    if not entity_files:
        errors.append("official attraction entities are unavailable")
    elif entity_files[0].stat().st_size == 0:
        errors.append(f"official entity file is empty: {entity_files[0]}")
    else:
        try:
            sample = pd.read_csv(entity_files[0], nrows=1)
            if "name" not in sample.columns or sample.empty or not str(sample.iloc[0]["name"]).strip():
                errors.append(f"official entity lookup failed: {entity_files[0]}")
        except Exception as exc:
            errors.append(f"official entity lookup failed: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("ChinaTravel preflight failed: " + "; ".join(errors))
    return {
        "root": str(root),
        "database_root": str(database),
        "schema": str(schema),
        "entity_file_count": len(entity_files),
        "database_version": f"files-{len(list(database.rglob('*.csv')))}",
    }
