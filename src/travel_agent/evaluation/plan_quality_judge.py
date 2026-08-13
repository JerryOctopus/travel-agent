from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from travel_agent.settings import JudgeSettings


RUBRIC_VERSION = "travel-plan-quality-v1"
PROMPT_VERSION = "travel-plan-judge-v2"
SCHEMA_VERSION = "travel-plan-judge-output-v1"
REASONABLE_THRESHOLD = 70
DIMENSIONS = {
    "schedule": 20,
    "route": 20,
    "constraints": 20,
    "personalization": 15,
    "completeness": 10,
    "diversity": 10,
    "clarity": 5,
}


class JudgeOutputError(ValueError):
    pass


class PlanQualityJudge:
    def __init__(
        self,
        settings: JudgeSettings,
        cache_dir: Path | str,
        *,
        invoke: Callable[[list[dict[str, str]]], Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.invoke = invoke or self._build_invoke()
        self.sleeper = sleeper
        self._last_request_at: float | None = None

    def evaluate(
        self,
        case_output: dict[str, Any],
        *,
        tested_model: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        itinerary = case_output.get("final_itinerary")
        if not itinerary:
            return {
                "status": "not_applicable",
                "rubric_version": RUBRIC_VERSION,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "model": self.settings.model,
                "experimental": True,
                "reason": "no itinerary was generated",
            }
        payload = _judge_payload(case_output)
        cache_key = _cache_key(payload, self.settings)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if resume and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return {**cached, "cache_hit": True}

        errors: list[str] = []
        started = time.perf_counter()
        for attempt in (1, 2):
            try:
                self._rate_limit()
                response = self.invoke(_messages(payload, repair=attempt == 2))
                content, usage = _response_content_and_usage(response)
                parsed = parse_judge_output(content)
                hard_pass = bool(
                    ((case_output.get("evaluation") or {}).get("rule_quality") or {}).get(
                        "hard_feasibility_pass"
                    )
                )
                has_critical = any(
                    issue.get("severity") == "critical"
                    for issue in parsed["critical_issues"]
                )
                same_model = bool(tested_model and tested_model == self.settings.model)
                result = {
                    "status": "ok",
                    "rubric_version": RUBRIC_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "provider": self.settings.provider,
                    "model": self.settings.model,
                    "experimental": True,
                    "independence_warning": same_model,
                    "scores": parsed["scores"],
                    "total_score": sum(parsed["scores"].values()),
                    "critical_issues": parsed["critical_issues"],
                    "insufficient_evidence": parsed["insufficient_evidence"],
                    "reason": parsed["reason"],
                    "reasonable": hard_pass
                    and sum(parsed["scores"].values()) >= REASONABLE_THRESHOLD
                    and not has_critical,
                    "hard_feasibility_pass": hard_pass,
                    "threshold": REASONABLE_THRESHOLD,
                    "attempt_count": attempt,
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    "usage": usage,
                    "raw_structured_response": parsed,
                    "errors": errors,
                }
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                return result
            except Exception as exc:  # noqa: BLE001
                errors.append(_safe_error(exc))
                if attempt == 1:
                    retry_after = _retry_after_seconds(exc)
                    if retry_after is not None:
                        self.sleeper(retry_after)
        return {
            "status": "missing",
            "rubric_version": RUBRIC_VERSION,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "provider": self.settings.provider,
            "model": self.settings.model,
            "experimental": True,
            "cache_key": cache_key,
            "cache_hit": False,
            "attempt_count": 2,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "errors": errors,
        }

    def _rate_limit(self) -> None:
        rps = self.settings.requests_per_second
        if rps <= 0:
            self._last_request_at = time.monotonic()
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            wait = (1 / rps) - (now - self._last_request_at)
            if wait > 0:
                self.sleeper(wait)
        self._last_request_at = time.monotonic()

    def _build_invoke(self) -> Callable[[list[dict[str, str]]], Any]:
        if not self.settings.enabled:
            raise RuntimeError(
                "Judge API is not configured; set TRAVEL_AGENT_JUDGE_API_KEY "
                "or the selected provider's API key (for example FREELLMAPI_API_KEY)"
            )
        from langchain_openai import ChatOpenAI

        extra_body: dict[str, Any] = {}
        client_options: dict[str, Any] = {}
        if self.settings.provider.lower() in {
            "freellmapi",
            "free_llm_api",
            "free-llm-api",
            "freellmapi-docker",
            "freellmapi_server",
            "freellmapi-server",
        }:
            # httpx otherwise inherits the macOS system proxy and may send localhost
            # traffic through it, producing a misleading 502 from the proxy.
            import httpx

            client_options["http_client"] = httpx.Client(trust_env=False)
        if self.settings.model.startswith("glm-"):
            extra_body["thinking"] = {
                "type": "enabled" if self.settings.thinking_enabled else "disabled"
            }
        model = ChatOpenAI(
            model=self.settings.model,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            temperature=self.settings.temperature,
            timeout=self.settings.timeout_seconds,
            max_retries=0,
            extra_body=extra_body or None,
            **client_options,
        )

        def invoke(messages: list[dict[str, str]]) -> Any:
            return model.invoke(messages)

        return invoke


def parse_judge_output(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"judge output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeOutputError("judge output must be an object")
    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise JudgeOutputError(f"scores must contain exactly {sorted(DIMENSIONS)}")
    normalized_scores: dict[str, int] = {}
    for name, maximum in DIMENSIONS.items():
        value = scores.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JudgeOutputError(f"score {name} must be numeric")
        if isinstance(value, float) and not value.is_integer():
            raise JudgeOutputError(f"score {name} must be an integer")
        if value < 0 or value > maximum:
            raise JudgeOutputError(f"score {name} must be between 0 and {maximum}")
        normalized_scores[name] = int(value)
    issues = payload.get("critical_issues") or []
    if not isinstance(issues, list):
        raise JudgeOutputError("critical_issues must be a list")
    normalized_issues = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise JudgeOutputError("each critical issue must be an object")
        severity = str(issue.get("severity") or "").lower()
        if severity not in {"info", "warning", "critical"}:
            raise JudgeOutputError("critical issue severity is invalid")
        normalized_issues.append(
            {
                "code": str(issue.get("code") or "unspecified"),
                "severity": severity,
                "evidence": str(issue.get("evidence") or ""),
            }
        )
    insufficient = payload.get("insufficient_evidence") or []
    if not isinstance(insufficient, list):
        raise JudgeOutputError("insufficient_evidence must be a list")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeOutputError("reason is required")
    return {
        "scores": normalized_scores,
        "critical_issues": normalized_issues,
        "insufficient_evidence": [str(item) for item in insufficient],
        "reason": reason.strip(),
    }


def aggregate_judge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "ok"]
    applicable = [item for item in results if item.get("status") != "not_applicable"]
    scores = [float(item["total_score"]) for item in completed]
    return {
        "applicable_count": len(applicable),
        "completed_count": len(completed),
        "missing_count": sum(item.get("status") == "missing" for item in results),
        "completion_rate": len(completed) / len(applicable) if applicable else None,
        "judge_average_score": sum(scores) / len(scores) if scores else None,
        "judge_reasonable_rate": (
            sum(bool(item.get("reasonable")) for item in completed) / len(completed)
            if completed
            else None
        ),
        "critical_issue_rate": (
            sum(
                any(issue.get("severity") == "critical" for issue in item.get("critical_issues") or [])
                for item in completed
            )
            / len(completed)
            if completed
            else None
        ),
        "experimental": True,
        "rubric_version": RUBRIC_VERSION,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def _judge_payload(case_output: dict[str, Any]) -> dict[str, Any]:
    artifacts = case_output.get("final_artifacts") or {}
    case = case_output.get("case") or {}
    itinerary_artifact = case_output.get("final_itinerary")
    raw_plan = (
        itinerary_artifact.get("itinerary")
        if isinstance(itinerary_artifact, dict) and isinstance(itinerary_artifact.get("itinerary"), dict)
        else itinerary_artifact
    )
    turns = case_output.get("turns") or []
    last_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
    final_answer = last_turn.get("reply_text") or ""
    final_plan = _compact_itinerary(raw_plan)
    if isinstance(final_plan, dict) and isinstance(itinerary_artifact, dict):
        for key in (
            "return_plan", "lodging_plan", "budget_plan", "fixed_event_plan",
            "candidate_verification", "mobility_plan",
        ):
            structured_plan = itinerary_artifact.get(key)
            if isinstance(structured_plan, dict):
                final_plan[key] = structured_plan
    facts = _compact_tool_facts(artifacts, itinerary_artifact, final_answer)
    evaluation = case_output.get("evaluation") or {}
    rule_metrics = evaluation.get("rule_metrics") or {}
    execution = case_output.get("execution") or {}
    return {
        "user_requests": case.get("turns") or [],
        "hard_constraints": case.get("hard_constraints") or {},
        "soft_preferences": case.get("soft_preferences") or {},
        "final_profile": case_output.get("final_profile") or {},
        # Deliberately exclude the Agent's internal critic and original draft.
        "itinerary": final_plan,
        "final_answer": final_answer,
        "tool_facts": facts,
        "execution_summary": {
            "status": last_turn.get("status"),
            "passed": execution.get("passed"),
            "errors": execution.get("errors") or [],
            "tool_trace": last_turn.get("tool_trace") or [],
            "duration_ms": last_turn.get("duration_ms"),
        },
        "automated_evaluator": {
            key: rule_metrics.get(key)
            for key in (
                "strict_task_success",
                "gating_passed",
                "gating_triggered",
                "grounding_ok",
                "authorization_ok",
                "architecture_policy_ok",
                "expected_outcome_match",
                "constraint_tree_score",
                "constraint_tree_missing",
            )
            if key in rule_metrics
        },
    }


def _compact_itinerary(plan: Any) -> Any:
    """Keep schedule, grounding and route semantics without map/UI payload bloat."""
    if not isinstance(plan, dict):
        return plan
    compact_days: list[dict[str, Any]] = []
    for raw_day in plan.get("days") or []:
        if not isinstance(raw_day, dict):
            continue
        stops: list[dict[str, Any]] = []
        for raw_stop in raw_day.get("stops") or []:
            if not isinstance(raw_stop, dict):
                continue
            poi = raw_stop.get("poi") or {}
            route = raw_stop.get("route_from_previous")
            stops.append(
                {
                    "poi": _pick_fields(
                        poi,
                        (
                            "poi_id",
                            "name",
                            "city",
                            "category",
                            "opening_hours",
                            "average_cost",
                            "source",
                        ),
                    ),
                    **_pick_fields(raw_stop, ("start_time", "duration_min", "note")),
                    "route_from_previous": _pick_fields(
                        route,
                        (
                            "origin_poi_id",
                            "destination_poi_id",
                            "distance_km",
                            "duration_min",
                            "mode",
                            "walking_distance_km",
                            "source",
                        ),
                    )
                    if isinstance(route, dict)
                    else None,
                }
            )
        compact_days.append(
            {
                **_pick_fields(raw_day, ("day_index", "date", "theme")),
                "stops": stops,
            }
        )
    return {
        **_pick_fields(plan, ("city", "summary")),
        "days": compact_days,
    }


def _compact_tool_facts(
    artifacts: dict[str, Any],
    itinerary_artifact: Any,
    final_answer: str,
) -> dict[str, Any]:
    raw_plan = (
        itinerary_artifact.get("itinerary")
        if isinstance(itinerary_artifact, dict)
        and isinstance(itinerary_artifact.get("itinerary"), dict)
        else itinerary_artifact
    )
    referenced_ids: set[str] = set()
    referenced_names: set[str] = set()
    if isinstance(raw_plan, dict):
        for day in raw_plan.get("days") or []:
            for stop in day.get("stops") or [] if isinstance(day, dict) else []:
                poi = stop.get("poi") or {} if isinstance(stop, dict) else {}
                if poi.get("poi_id"):
                    referenced_ids.add(str(poi["poi_id"]))
                if poi.get("name"):
                    referenced_names.add(str(poi["name"]))

    facts: dict[str, Any] = {}
    for key in ("candidates", "ranked", "restaurants", "hotels"):
        value = artifacts.get(key)
        if isinstance(value, dict):
            facts[key] = _compact_evidence_collection(
                value,
                referenced_ids=referenced_ids,
                referenced_names=referenced_names,
                final_answer=final_answer,
            )
    for key in ("weather", "budget", "routes"):
        value = artifacts.get(key)
        if value is not None:
            facts[key] = value
    domain_inputs = (
        itinerary_artifact.get("domain_inputs")
        if isinstance(itinerary_artifact, dict)
        else None
    ) or {}
    for key in ("restaurants", "hotels"):
        entries = domain_inputs.get(key) or []
        merged_items: list[dict[str, Any]] = []
        city = None
        for entry in entries:
            payload = entry.get("payload", entry) if isinstance(entry, dict) else {}
            if not isinstance(payload, dict):
                continue
            city = city or payload.get("city")
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    merged_items.append(item)
        if merged_items:
            facts[key] = _compact_evidence_collection(
                {"city": city, key: merged_items},
                referenced_ids=referenced_ids,
                referenced_names=referenced_names,
                final_answer=final_answer,
            )
    transport = domain_inputs.get("transport") or []
    if transport:
        facts["routes"] = [
            entry.get("payload", entry) if isinstance(entry, dict) else entry
            for entry in transport[-8:]
        ]
    return facts


def _compact_evidence_collection(
    value: dict[str, Any],
    *,
    referenced_ids: set[str],
    referenced_names: set[str],
    final_answer: str,
) -> dict[str, Any]:
    list_key = next(
        (key for key in ("pois", "items", "candidates", "restaurants", "hotels") if isinstance(value.get(key), list)),
        None,
    )
    if list_key is None:
        return _pick_fields(value, ("city", "query", "source", "summary"))
    items = [item for item in value[list_key] if isinstance(item, dict)]
    selected = [
        item
        for item in items
        if str(item.get("poi_id") or item.get("id") or "") in referenced_ids
        or str(item.get("name") or "") in referenced_names
        or (
            bool(str(item.get("name") or ""))
            and str(item.get("name")) in final_answer
        )
    ]
    if not selected:
        selected = items[:5]
    return {
        **_pick_fields(value, ("city", "query", "source", "summary")),
        list_key: [
            _pick_fields(
                item,
                (
                    "poi_id",
                    "id",
                    "name",
                    "city",
                    "category",
                    "opening_hours",
                    "average_cost",
                    "price_level",
                    "source",
                ),
            )
            for item in selected[:8]
        ],
    }


def _pick_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in fields if key in value and value[key] is not None}


def _messages(payload: dict[str, Any], *, repair: bool) -> list[dict[str, str]]:
    repair_note = "上一次响应格式不合法。本次只能输出合法JSON，不要使用Markdown。" if repair else ""
    system = f"""你是独立旅行行程质量评审。只根据给定需求、行程和工具事实评分，不补充外部事实。
Rubric版本：{RUBRIC_VERSION}。分值上限：{json.dumps(DIMENSIONS, ensure_ascii=False)}。
输出JSON必须包含 scores、critical_issues、insufficient_evidence、reason。
critical_issues元素包含 code、severity(info/warning/critical)、evidence。
住宿候选与选定住宿通过 tool_facts.hotels 独立呈现，不要求把酒店伪装成 itinerary.days[].stops；
不得仅因酒店未出现在每日景点 stops 中判 critical。只有明确住宿硬约束且 tool_facts.hotels 也无合规证据时，才可判住宿约束失败。
不要输出total_score，系统会根据分项计算。证据不足必须记录，不得猜测。{repair_note}"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _cache_key(payload: dict[str, Any], settings: JudgeSettings) -> str:
    encoded = json.dumps(
        {
            "payload": payload,
            "provider": settings.provider,
            "model": settings.model,
            "rubric": RUBRIC_VERSION,
            "prompt": PROMPT_VERSION,
            "schema": SCHEMA_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_content_and_usage(response: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(response, str):
        return response, {}
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
    return str(content), dict(usage)


def _safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(
        r"(?i)(api[_-]?key|authorization|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    return text[:500]


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honor Gemini's 429 retry hint instead of immediately retrying."""
    text = str(exc)
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return None
    match = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE)
    if match is None:
        return 60.0
    return max(1.0, float(match.group(1)))
