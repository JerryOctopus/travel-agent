"""Artifact-first evaluator routing shared by harness, Judges and reports.

The production-v1 schema historically labels nearly every successful answer as
``full_plan``.  That label remains immutable, but it is not a reliable statement
of the requested artifact.  This module derives a compatible artifact contract
from explicit new-schema fields when present and otherwise from the user task.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping


ARTIFACT_TYPES = frozenset(
    {
        "full_itinerary",
        "partial_itinerary",
        "route_plan",
        "candidate_comparison",
        "itinerary_patch",
        "local_adjustment_advice",
        "clarification",
        "constraint_negotiation",
        "safe_decline",
    }
)
ITINERARY_TYPES = frozenset({"full_itinerary", "partial_itinerary"})
NON_ITINERARY_TYPES = ARTIFACT_TYPES - ITINERARY_TYPES
JUDGE_STATUSES = frozenset({"ok", "not_applicable", "not_run", "missing", "error"})
ARTIFACT_CONTRACT_VERSION = "dev34-artifact-contract-v3"


def artifact_contract_fingerprint(cases: Any = None) -> str:
    """Hash the evaluator contract and, optionally, the case→artifact map."""
    payload: dict[str, Any] = {
        "version": ARTIFACT_CONTRACT_VERSION,
        "artifact_types": sorted(ARTIFACT_TYPES),
        "itinerary_types": sorted(ITINERARY_TYPES),
        "required_fields": {
            key: list(value) for key, value in sorted(REQUIRED_ARTIFACT_FIELDS.items())
        },
    }
    if cases is not None:
        payload["expected_artifacts"] = {
            str(_case_value(case, "case_id") or ""): expected_artifact_type(case)
            for case in cases
        }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_artifact_type(case: Any) -> str:
    """Infer the requested deliverable without changing legacy gold labels."""
    explicit = _case_value(case, "expected_artifact_type")
    if explicit in ARTIFACT_TYPES:
        return str(explicit)
    metadata = _case_value(case, "metadata") or {}
    if isinstance(metadata, Mapping) and metadata.get("expected_artifact_type") in ARTIFACT_TYPES:
        return str(metadata["expected_artifact_type"])

    turns = _case_value(case, "turns") or []
    text = str(turns[-1] if turns else "").strip()
    gold = str(_case_value(case, "gold_outcome") or "").lower()
    if gold in {"clarify", "clarification"}:
        return "clarification"
    if gold in {"negotiate_constraints", "constraint_negotiation"}:
        return "constraint_negotiation"
    if gold in {"safe_decline_action", "safe_decline"}:
        return "safe_decline"
    if gold in {"partial_plan_with_limitations", "partial_itinerary"}:
        return "partial_itinerary"

    # Runtime and evaluator must share one task-scope router.  Keep this import
    # local so the artifact helpers remain safe at module import boundaries.
    from travel_agent.agent.turn_analysis import TaskType, classify_task_type_rule_based

    routed = classify_task_type_rule_based(text)
    if routed != TaskType.CLARIFICATION and routed.value in ARTIFACT_TYPES:
        return routed.value
    return "full_itinerary"


def effective_required_tools(
    case: Any,
    *,
    expected: str | None = None,
    profile: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return the minimum tool contract for the requested artifact.

    Frozen production-v1 gold attached full-planning tools to every successful
    case.  Specialized Dev34 deliveries intentionally exclude Planner, so that
    legacy list is retained only for itinerary artifacts.  This correction
    changes evaluator interpretation, not product behavior.
    """
    expected = expected or expected_artifact_type(case)
    legacy_source = (
        _case_value(case, "required_tools")
        or _case_value(case, "expected_tools")
        or []
    )
    legacy = [str(item) for item in legacy_source]
    if expected in ITINERARY_TYPES:
        return list(dict.fromkeys(legacy))
    if expected in {"clarification", "constraint_negotiation", "safe_decline"}:
        return []

    turns = _case_value(case, "turns") or []
    text = str(turns[-1] if turns else "")
    state = dict((profile or {}).get("constraint_state") or {})
    hard = _case_value(case, "hard_constraints") or {}
    if isinstance(hard, Mapping):
        for key, value in hard.items():
            state.setdefault(str(key), value)

    tools: list[str] = []
    if expected == "route_plan":
        tools.extend(("search_poi", "plan_route"))
    elif expected in {"local_adjustment_advice", "itinerary_patch"}:
        if any(marker in text for marker in ("天气", "下雨", "有雨", "高温", "大风", "降雨")):
            tools.append("check_weather")
        if state.get("need_indoor_backup") or any(marker in text for marker in ("室内备选", "替换")):
            tools.append("search_poi")
    elif expected == "candidate_comparison":
        restaurant = bool(
            state.get("specific_restaurant_recommendation")
            or any(marker in text for marker in ("餐厅", "饭店", "晚餐", "午餐", "用餐"))
        )
        lodging = bool(
            state.get("compare_lodging_areas")
            or any(marker in text for marker in ("酒店", "住宿", "住哪里"))
        ) and not state.get("no_live_inventory_required")
        tools.append(
            "search_restaurant" if restaurant else "search_hotel" if lodging else "search_poi"
        )
        if any(
            state.get(key) not in (None, "", [], {})
            for key in ("origin", "target_anchor", "walking_time_max_min")
        ) or any(marker in text for marker in ("路线", "通勤", "交通时间", "步行")):
            if "search_poi" not in tools:
                tools.append("search_poi")
            tools.append("plan_route")
    return list(dict.fromkeys(tools))


def required_agents_for_delivery(
    expected: str,
    *,
    task_brief: str,
    profile: Mapping[str, Any] | None = None,
) -> list[str]:
    """Derive required worker roles from the same artifact-scoped router."""
    from travel_agent.agent.turn_analysis import TaskType
    from travel_agent.orchestration.multi_agent.dispatch_rules import agents_required_for_turn

    task_type = {
        "full_itinerary": TaskType.FULL_ITINERARY,
        "partial_itinerary": TaskType.FULL_ITINERARY,
        "route_plan": TaskType.ROUTE_PLAN,
        "candidate_comparison": TaskType.CANDIDATE_COMPARISON,
        "itinerary_patch": TaskType.ITINERARY_PATCH,
        "local_adjustment_advice": TaskType.LOCAL_ADJUSTMENT_ADVICE,
    }.get(expected)
    if task_type is None:
        return []
    return list(agents_required_for_turn(
        task_type,
        task_brief=task_brief,
        inputs={"profile": dict(profile or {})},
    ))


def actual_artifact_type(case_output: Mapping[str, Any], expected: str | None = None) -> str | None:
    """Identify the delivered artifact independently from Judge applicability."""
    evaluation = case_output.get("evaluation") or {}
    explicit = evaluation.get("actual_artifact_type") or case_output.get("actual_artifact_type")
    if explicit in ARTIFACT_TYPES:
        return str(explicit)

    turns = case_output.get("turns") or []
    last = turns[-1] if turns and isinstance(turns[-1], Mapping) else {}
    reply = str(last.get("reply_text") or "")
    status = str(last.get("status") or "")
    if case_output.get("errors") or last.get("error"):
        return None
    if _has_itinerary(case_output):
        limited = status not in {"", "completed", "completed_with_warnings"} or any(
            marker in reply for marker in ("尚不可交付", "仅供审阅", "部分满足", "存在限制")
        )
        return "partial_itinerary" if limited else "full_itinerary"
    if bool(last.get("clarification")) or status == "clarification_required":
        if any(marker in reply for marker in ("无法同时满足", "需要放宽", "取舍", "二选一")):
            return "constraint_negotiation"
        return "clarification"
    if any(marker in reply for marker in ("不能代你", "无法替您", "不能替您", "无权")) and any(
        marker in reply for marker in ("预订", "支付", "取消", "购票", "联系商家")
    ):
        return "safe_decline"
    if not reply.strip():
        return None

    expected = expected or expected_artifact_type(case_output.get("case") or {})
    artifacts = case_output.get("final_artifacts") or {}
    for artifact_type in (
        "candidate_comparison",
        "itinerary_patch",
        "local_adjustment_advice",
        "route_plan",
    ):
        payload = artifacts.get(artifact_type)
        if isinstance(payload, Mapping) and artifact_content_valid(artifact_type, payload):
            return artifact_type
    trace = " ".join(
        str(item)
        for item in [
            *(last.get("tool_trace") or []),
            *artifacts.keys(),
            reply,
        ]
    ).lower()
    return None


REQUIRED_ARTIFACT_FIELDS: dict[str, tuple[str, ...]] = {
    "candidate_comparison": (
        "subject", "candidates", "comparison_dimensions", "core_dimensions",
        "evidence", "coverage", "recommendation", "limitations",
    ),
    "itinerary_patch": (
        "subject", "before", "after", "affected_periods", "constraint_checks",
        "evidence", "limitations",
    ),
    "local_adjustment_advice": (
        "subject", "recommendation", "alternatives", "evidence", "limitations",
    ),
    "route_plan": ("origin", "destination", "evidence", "limitations"),
}


def artifact_content_valid(artifact_type: str, payload: Mapping[str, Any] | None) -> bool:
    """Fail closed on missing or structurally empty specialized artifacts."""
    if not isinstance(payload, Mapping):
        return False
    required = REQUIRED_ARTIFACT_FIELDS.get(artifact_type)
    if not required:
        return True
    if any(field not in payload for field in required):
        return False
    nonempty = {
        "subject", "candidates", "comparison_dimensions", "recommendation",
        "before", "after", "affected_periods", "constraint_checks", "evidence",
    }
    structurally_nonempty = all(
        payload.get(field) not in (None, "", [], {})
        for field in required
        if field in nonempty
    )
    if not structurally_nonempty:
        return False
    if artifact_type == "candidate_comparison":
        return _candidate_comparison_content_valid(payload)
    return True


def _candidate_comparison_content_valid(payload: Mapping[str, Any]) -> bool:
    from travel_agent.candidate_comparison import (
        ComparisonEvidenceSet,
        accessibility_contract,
        accessibility_mode,
        normalize_comparison_evidence_rows,
    )

    names = [
        str(item.get("name") or "").strip()
        for item in payload.get("candidates") or []
        if isinstance(item, Mapping) and str(item.get("name") or "").strip()
    ]
    dimensions = [str(item).strip() for item in payload.get("comparison_dimensions") or [] if str(item).strip()]
    evidence = payload.get("evidence") or []
    if not names or not dimensions or not isinstance(evidence, list):
        return False
    required_fields = {
        "candidate", "dimension", "source_artifact_id", "source_kind",
        "source_agent", "source_task_id", "facts",
    }
    valid_rows = [
        item
        for item in evidence
        if isinstance(item, Mapping)
        and required_fields.issubset(item)
        and item.get("candidate") in names
        and item.get("dimension") in dimensions
        and item.get("source_artifact_id")
        and item.get("source_agent") not in {"", "engine", "planner"}
        and item.get("facts") not in (None, "", [], {})
    ]
    subject = payload.get("subject") or {}
    if not isinstance(subject, Mapping):
        return False
    target_anchor = str(subject.get("target_anchor") or "").strip()
    normalized_rows, strategies = normalize_comparison_evidence_rows(
        valid_rows,
        names,
        dimensions,
        target_anchor=target_anchor,
    )
    coverage = {(item["candidate"], item["dimension"]) for item in normalized_rows}
    core_dimensions = [
        str(item).strip()
        for item in payload.get("core_dimensions") or dimensions[:1]
        if str(item).strip()
    ]
    if any(
        (name, dimension) not in coverage
        for name in names
        for dimension in core_dimensions
    ):
        return False
    expected_coverage = {
        "required": len(names) * len(dimensions),
        "covered": len(coverage),
        "complete": len(coverage) == len(names) * len(dimensions),
    }
    if payload.get("coverage") != expected_coverage:
        return False
    if subject.get("accessibility_mode") != accessibility_mode(target_anchor):
        return False
    if "accessibility" in dimensions:
        expected_accessibility = accessibility_contract(
            names,
            ComparisonEvidenceSet(
                evidence=normalized_rows,
                dimension_strategies=strategies,
            ),
            target_anchor=target_anchor,
        )
        if payload.get("accessibility_contract") != expected_accessibility:
            return False
    recommendation = payload.get("recommendation")
    if not isinstance(recommendation, Mapping):
        return False
    source_ids = {str(item["source_artifact_id"]) for item in normalized_rows}
    recommendation_sources = {
        str(item) for item in recommendation.get("source_artifact_ids") or [] if item
    }
    if not recommendation_sources or not recommendation_sources.issubset(source_ids):
        return False
    winner = recommendation.get("candidate")
    if winner is not None and winner not in names:
        return False
    used_dimensions = [
        str(item) for item in recommendation.get("dimensions_used") or [] if item
    ]
    if not used_dimensions or any(dimension not in dimensions for dimension in used_dimensions):
        return False
    if any(dimension not in used_dimensions for dimension in core_dimensions):
        return False
    comparable_metrics = recommendation.get("comparable_metrics")
    if not isinstance(comparable_metrics, list):
        return False
    if winner is None and not recommendation.get("decision"):
        return False
    if any(
        (name, dimension) not in coverage
        for name in names
        for dimension in used_dimensions
    ):
        return False
    missing_dimensions = [
        dimension
        for dimension in dimensions
        if any((name, dimension) not in coverage for name in names)
    ]
    limitation_text = " ".join(str(item) for item in payload.get("limitations") or [])
    if any(dimension not in limitation_text for dimension in missing_dimensions):
        return False
    if "accessibility_needs" in missing_dimensions:
        if "适老或无障碍优势" not in limitation_text:
            return False
        if "accessibility_needs" in used_dimensions:
            return False
    for basis in recommendation.get("basis") or []:
        if not isinstance(basis, Mapping):
            return False
        if basis.get("source_artifact_id") not in source_ids:
            return False
        basis_sources = basis.get("source_artifact_ids") or [basis.get("source_artifact_id")]
        if any(str(item) not in source_ids for item in basis_sources if item):
            return False
        if basis.get("candidate") not in names or basis.get("dimension") not in dimensions:
            return False
        if basis.get("metric") not in comparable_metrics:
            return False
    return True


def artifact_type_matches(expected: str, actual: str | None) -> bool:
    if expected == "full_itinerary":
        return actual == "full_itinerary"
    return actual == expected


def itinerary_judge_route(expected: str, actual: str | None, *, system_error: bool = False) -> dict[str, Any]:
    """Select full/partial itinerary Judge or explain why it will not run."""
    if expected not in ITINERARY_TYPES:
        return {"status": "not_applicable", "rubric": None, "reason": "task_does_not_require_itinerary"}
    if system_error or actual is None:
        return {"status": "not_run", "rubric": None, "reason": "system_error" if system_error else "missing_expected_artifact"}
    if actual == "partial_itinerary":
        return {"status": "pending", "rubric": "partial_itinerary", "diagnostic_only": expected == "full_itinerary"}
    if actual == "full_itinerary":
        return {"status": "pending", "rubric": "full_itinerary", "diagnostic_only": False}
    return {"status": "not_run", "rubric": None, "reason": "missing_expected_artifact"}


def task_completion_evaluation(
    expected: str,
    actual: str | None,
    *,
    constraints_passed: bool,
    grounding_passed: bool,
    status_success: bool,
    reply_present: bool,
    artifact_complete: bool = True,
) -> dict[str, Any]:
    """Task-level evaluator used for all non-itinerary deliverables."""
    if expected in ITINERARY_TYPES:
        return {"status": "not_applicable", "evaluator": None, "passed": None}
    passed = all(
        (
            artifact_type_matches(expected, actual),
            constraints_passed,
            grounding_passed,
            status_success,
            reply_present,
            artifact_complete,
        )
    )
    return {
        "status": "ok",
        "evaluator": expected,
        "passed": passed,
        "checks": {
            "artifact_type_match": artifact_type_matches(expected, actual),
            "constraints": constraints_passed,
            "grounding": grounding_passed,
            "completeness": reply_present,
            "artifact_content": artifact_complete,
            "status": status_success,
        },
    }


def aggregate_artifact_metrics(case_outputs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate artifact/Judge metrics without mixing rubric score scales."""
    rows = []
    judge_statuses: Counter[str] = Counter()
    judge_reasons: Counter[str] = Counter()
    rubric_scores: dict[str, list[float]] = {}
    task_results = []
    for output in case_outputs:
        evaluation = output.get("evaluation") or {}
        metrics = evaluation.get("rule_metrics") or {}
        expected = str(metrics.get("expected_artifact_type") or expected_artifact_type(output.get("case") or {}))
        actual = metrics.get("actual_artifact_type") or actual_artifact_type(output, expected)
        judge = evaluation.get("independent_judge") or {}
        status = str(judge.get("status") or "not_run")
        judge_statuses[status] += 1
        if status != "ok" and judge.get("reason"):
            judge_reasons[str(judge["reason"])] += 1
        rubric = str(judge.get("rubric") or judge.get("artifact_type") or "full_itinerary")
        if status == "ok" and isinstance(judge.get("total_score"), (int, float)):
            rubric_scores.setdefault(rubric, []).append(float(judge["total_score"]))
        task = evaluation.get("task_completion_judge") or metrics.get("task_completion_judge") or {}
        if task.get("status") == "ok":
            task_results.append(bool(task.get("passed")))
        rows.append((expected, actual, bool(metrics.get("strict_task_success")), bool(metrics.get("artifact_type_match"))))

    full = [row for row in rows if row[0] == "full_itinerary"]
    non = [row for row in rows if row[0] in NON_ITINERARY_TYPES]
    # not_run is still applicable in principle (the expected artifact/Judge
    # invocation was missing); only not_applicable is excluded.
    applicable = sum(judge_statuses.values()) - judge_statuses["not_applicable"]
    completed = judge_statuses["ok"]
    return {
        "overall_strict_success_rate": _rate([row[2] for row in rows]),
        "artifact_type_match_rate": _rate([row[3] for row in rows]),
        "full_itinerary_success_rate": _rate([row[2] for row in full]),
        "non_itinerary_success_rate": _rate([row[2] for row in non]),
        "expected_itinerary_missing_rate": _rate([row[1] not in ITINERARY_TYPES for row in full]),
        "itinerary_judge_applicable_count": applicable,
        "itinerary_judge_completed_count": completed,
        "itinerary_judge_completion_rate": completed / applicable if applicable else None,
        "itinerary_judge_average_by_rubric": {
            rubric: sum(scores) / len(scores) for rubric, scores in rubric_scores.items()
        },
        "task_completion_judge_applicable_count": len(task_results),
        "task_completion_judge_pass_rate": _rate(task_results),
        "judge_status_distribution": dict(sorted(judge_statuses.items())),
        "judge_reason_distribution": dict(sorted(judge_reasons.items())),
    }


def _has_itinerary(case_output: Mapping[str, Any]) -> bool:
    value = case_output.get("final_itinerary")
    if not value:
        value = (case_output.get("final_artifacts") or {}).get("itinerary")
    plan = value.get("itinerary") if isinstance(value, Mapping) else value
    return bool(isinstance(plan, Mapping) and plan.get("days"))


def _case_value(case: Any, name: str) -> Any:
    if isinstance(case, Mapping):
        if name in case:
            return case.get(name)
        gold = case.get("gold") or {}
        if name == "gold_outcome":
            return gold.get("expected_outcome")
        if name == "turns":
            return case.get("turns") or [
                item.get("content") for item in (case.get("conversation") or []) if item.get("role") == "user"
            ]
        return None
    return getattr(case, name, None)


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None
