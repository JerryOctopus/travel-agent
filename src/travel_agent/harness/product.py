from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from travel_agent.harness.cases import HarnessCase, load_cases_json
from travel_agent.harness.result import HarnessCaseResult, HarnessSuiteResult
from travel_agent.harness.runner import AgentHarness

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRODUCT_CASES = ROOT / "data" / "eval" / "production_v1" / "all_cases.jsonl"
PRODUCTION_DATASET_VERSION = "travel-agent-eval-production-v1.1"

# 192 条数据集的 split 构成（随数据集冻结，禁止用于训练/调优回流）。
PRODUCTION_SPLIT_COUNTS = {
    "dev": 34,
    "core_frozen": 94,
    "challenge_frozen": 34,
    "shadow_frozen": 30,
}
FROZEN_SPLITS = {"core_frozen", "challenge_frozen", "shadow_frozen"}
FINE_TUNING_FROZEN_COUNT = (
    PRODUCTION_SPLIT_COUNTS["core_frozen"]
    + PRODUCTION_SPLIT_COUNTS["challenge_frozen"]
)
TRAINING_DATA_POLICY = (
    "production_v1 frozen cases (core/challenge/shadow) must never be used for training."
)
FAILURE_CATEGORIES = {
    "model_intent",
    "model_tool_selection",
    "model_tool_arguments",
    "model_clarification",
    "planner_or_workflow",
    "critic_or_validator",
    "memory_or_storage",
    "tool_or_data",
    "adapter_or_environment",
    "external_api",
}
MODEL_FAILURE_CATEGORIES = {
    "model_intent",
    "model_tool_selection",
    "model_tool_arguments",
    "model_clarification",
}


@dataclass(frozen=True)
class ProductDatasetValidation:
    valid: bool
    errors: list[str]
    split_counts: dict[str, int]
    subset_counts: dict[str, int]
    frozen_warnings: list[str] = field(default_factory=list)


_ABSOLUTE_RETURN_DEADLINE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:$|[T\s])"
)
_ISO_USER_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_CHINESE_USER_DATE = re.compile(
    r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?"
)
_DURATION_TOKEN = re.compile(
    r"(?P<number>\d{1,3}|[零〇一二两三四五六七八九十百]+)\s*(?P<unit>天|日游)"
)
_RELATIVE_FINAL_DEADLINE = re.compile(
    r"(?P<day>最后一天|第(?P<number>\d{1,3}|[零〇一二两三四五六七八九十百]+)天)"
    r"[^。；]{0,30}?(?P<hour>\d{1,2})(?::|点)(?P<minute>\d{2})?"
)


def validate_absolute_return_deadline_evidence(
    cases: list[HarnessCase],
) -> tuple[list[str], list[str]]:
    """Reject absolute return dates unsupported by either date or relative-day semantics.

    An ISO ``return_deadline`` date must equal either an explicitly stated
    ``date_end`` or ``date_start + duration_days - 1``.  Only full dates and
    durations present in user turns count as evidence. A matching explicit
    “最后一天/第 N 天 HH:MM” is also sufficient when N equals the explicit trip
    duration; the absolute date carried by legacy Gold is then treated only as
    an annotation and is never inferred from ``reference_datetime``.
    """
    errors: list[str] = []
    frozen_warnings: list[str] = []
    for case in cases:
        issue = _absolute_return_deadline_evidence_issue(case)
        if issue is None:
            continue
        if case.split in FROZEN_SPLITS:
            frozen_warnings.append(issue)
        else:
            errors.append(issue)
    return errors, frozen_warnings


def _absolute_return_deadline_evidence_issue(case: HarnessCase) -> str | None:
    constraints = case.gold_constraints_tree
    return_deadline = constraints.get("return_deadline")
    match = _ABSOLUTE_RETURN_DEADLINE.match(str(return_deadline or ""))
    if match is None:
        return None

    deadline_date = date.fromisoformat(match.group("date"))
    stated_dates = _explicit_user_dates(case.turns)
    supported_dates: set[date] = set()

    date_end = _constraint_date(constraints.get("date_end"))
    if date_end is not None and date_end in stated_dates:
        supported_dates.add(date_end)

    date_start = _constraint_date(constraints.get("date_start"))
    duration_days = constraints.get("duration_days")
    if (
        date_start is not None
        and date_start in stated_dates
        and isinstance(duration_days, int)
        and not isinstance(duration_days, bool)
        and duration_days > 0
        and _duration_is_explicit(case.turns, duration_days)
    ):
        supported_dates.add(date_start + timedelta(days=duration_days - 1))

    if deadline_date in supported_dates:
        return None
    if _relative_final_day_deadline_supported(
        case.turns,
        duration_days,
        str(return_deadline),
    ):
        return None
    return (
        f"{case.case_id}: absolute return_deadline date {deadline_date.isoformat()} "
        "must be derivable from explicit user date_start/date_end and duration_days"
    )


def _relative_final_day_deadline_supported(
    turns: list[str],
    duration_days: Any,
    return_deadline: str,
) -> bool:
    if (
        not isinstance(duration_days, int)
        or isinstance(duration_days, bool)
        or duration_days <= 0
        or not _duration_is_explicit(turns, duration_days)
    ):
        return False
    expected_time = re.search(r"(?:T|\s)(\d{1,2}):(\d{2})", return_deadline)
    if expected_time is None:
        return False
    expected_hour = int(expected_time.group(1))
    expected_minute = int(expected_time.group(2))
    for turn in turns:
        for match in _RELATIVE_FINAL_DEADLINE.finditer(turn):
            if match.group("day") == "最后一天":
                day_number = duration_days
            else:
                day_number = _parse_duration_number(str(match.group("number") or ""))
            minute = int(match.group("minute") or 0)
            if (
                day_number == duration_days
                and int(match.group("hour")) == expected_hour
                and minute == expected_minute
            ):
                return True
    return False


def _constraint_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _explicit_user_dates(turns: list[str]) -> set[date]:
    dates: set[date] = set()
    for turn in turns:
        for pattern in (_ISO_USER_DATE, _CHINESE_USER_DATE):
            for match in pattern.finditer(turn):
                try:
                    dates.add(date(*(int(part) for part in match.groups())))
                except ValueError:
                    continue
    return dates


def _duration_is_explicit(turns: list[str], expected_days: int) -> bool:
    for turn in turns:
        for match in _DURATION_TOKEN.finditer(turn):
            prefix = turn[: match.start()]
            if match.group("unit") == "天" and prefix.endswith(("第", "最后", "倒数", "每")):
                continue
            if _parse_duration_number(match.group("number")) == expected_days:
                return True
    return False


def _parse_duration_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if "百" in value:
        left, _, right = value.partition("百")
        hundreds = digits.get(left or "一")
        remainder = _parse_duration_number(right) if right else 0
        return None if hundreds is None or remainder is None else hundreds * 100 + remainder
    if "十" in value:
        left, _, right = value.partition("十")
        tens = digits.get(left or "一")
        ones = digits.get(right, 0)
        return None if tens is None or ones is None else tens * 10 + ones
    return digits.get(value)


def validate_product_dataset(cases: list[HarnessCase]) -> ProductDatasetValidation:
    """校验 production_v1 数据集：192 条、split 构成、case_id 唯一、turns 非空。"""
    errors: list[str] = []
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case ids must be unique")
    split_counts = {name: 0 for name in PRODUCTION_SPLIT_COUNTS}
    subset_counts: dict[str, int] = {}
    for case in cases:
        if case.split not in split_counts:
            errors.append(f"{case.case_id}: unknown split {case.split}")
        else:
            split_counts[case.split] += 1
        if not case.turns:
            errors.append(f"{case.case_id}: turns cannot be empty")
        if not case.gold_outcome:
            errors.append(f"{case.case_id}: gold.expected_outcome is required")
        if case.session_ids and len(case.session_ids) != len(case.turns):
            errors.append(f"{case.case_id}: session_ids must match turns")
        if case.user_ids and len(case.user_ids) != len(case.turns):
            errors.append(f"{case.case_id}: user_ids must match turns")
        subset = case.subset or case.category
        subset_counts[subset] = subset_counts.get(subset, 0) + 1
    total = sum(PRODUCTION_SPLIT_COUNTS.values())
    if len(cases) != total:
        errors.append(f"expected {total} cases, found {len(cases)}")
    if split_counts != PRODUCTION_SPLIT_COUNTS:
        errors.append(f"split counts mismatch: {split_counts}")
    deadline_errors, frozen_warnings = validate_absolute_return_deadline_evidence(cases)
    errors.extend(deadline_errors)
    return ProductDatasetValidation(
        not errors,
        errors,
        split_counts,
        subset_counts,
        frozen_warnings,
    )


def run_product_suite(
    harness: AgentHarness,
    *,
    case_path: Path | str = DEFAULT_PRODUCT_CASES,
    split: str = "dev",
    limit: int | None = None,
    case_ids: list[str] | None = None,
    remediation_complete: bool = False,
) -> HarnessSuiteResult:
    all_cases = load_cases_json(case_path)
    validation = validate_product_dataset(all_cases)
    if not validation.valid:
        raise ValueError("invalid production_v1 dataset: " + "; ".join(validation.errors))
    cases = [case for case in all_cases if split == "all" or case.split == split]
    if case_ids is not None:
        wanted = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted]
    if limit is not None:
        cases = cases[: max(1, limit)]
    variant = str(harness.environment.variant or "").upper() or "V3"
    executions: list[tuple[HarnessCase, HarnessCaseResult, int]] = [
        (case, harness.run_case(_execution_case(case, 1)), 1) for case in cases
    ]
    summary = aggregate_product_results(
        executions, remediation_complete=remediation_complete, variant=variant
    )
    case_outputs = [
        _product_case_output(case, result, repeat)
        for case, result, repeat in executions
    ]
    if harness.settings.llm.provider == "rule":
        summary["fine_tuning"] = {
            "decision": "offline_smoke_only",
            "rationale": "离线规则路径只验证评测链路，不能判断真实基线模型是否需要微调。",
            "model_failure_counts": {key: 0 for key in MODEL_FAILURE_CATEGORIES},
            "triggered_categories": {},
            "frozen_case_count": sum(case.split in FROZEN_SPLITS for case in cases),
            "remediation_complete": remediation_complete,
            "training_data_policy": TRAINING_DATA_POLICY,
        }
    return HarnessSuiteResult(
        suite="agent-product",
        mode="product",
        environment=harness.environment.mode,
        case_count=len(cases),
        attempted_count=len(executions),
        metrics={key: value for key, value in summary.items() if key not in {"cases", "fine_tuning"}},
        rows=summary["cases"],
        artifacts={
            "dataset_version": PRODUCTION_DATASET_VERSION,
            "dataset_sha256": _sha256(Path(case_path)),
            "code_revision": _git_revision(),
            "model_provider": harness.settings.llm.provider,
            "model": harness.settings.llm.model,
            "model_temperature": harness.settings.llm.temperature,
            "model_thinking_enabled": harness.settings.llm.thinking_enabled,
            "model_requests_per_second": harness.settings.llm.requests_per_second,
            "hybrid_flags": {
                "enable_llm_intent_normalizer": harness.settings.hybrid_planning.enable_llm_intent_normalizer,
                "enable_llm_preference_resolver": harness.settings.hybrid_planning.enable_llm_preference_resolver,
                "enable_structured_duration_estimator": harness.settings.hybrid_planning.enable_structured_duration_estimator,
            },
            "prompt_fingerprint": _prompt_fingerprint(),
            "code_fingerprint": _code_fingerprint(),
            "evaluator_fingerprint": _evaluator_fingerprint(),
            "tooling_fingerprint": _tooling_fingerprint(),
            "configuration_fingerprint": _configuration_fingerprint(
                harness.settings, harness.environment.tool_provider
            ),
            "tool_snapshot_fingerprint": _tool_snapshot_fingerprint(
                harness.settings, harness.environment.tool_provider
            ),
            "evaluator_version": _evaluator_version(),
            "artifact_contract_fingerprint": _artifact_contract_fingerprint(cases),
            "fine_tuning": summary["fine_tuning"],
            "independent_case_count": len(cases),
            "execution_count": len(executions),
            "_case_outputs": case_outputs,
        },
    )


def write_product_run(
    result: HarnessSuiteResult,
    output_root: Path | str,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist one inspectable Product run without bloating CLI output."""
    root = Path(output_root)
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs" / resolved_run_id
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=False)

    case_outputs = result.artifacts.get("_case_outputs") or []
    output_paths: dict[tuple[str, int], str] = {}
    for output in case_outputs:
        case_id = str(output["case"]["case_id"])
        repeat = int(output["execution"]["repeat"])
        filename = f"{case_id}__repeat-{repeat}.json"
        path = cases_dir / filename
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths[(case_id, repeat)] = str(path)

    public_artifacts = {
        key: value for key, value in result.artifacts.items() if key != "_case_outputs"
    }
    public_artifacts.update(
        {
            "run_id": resolved_run_id,
            "run_dir": str(run_dir),
            "case_output_dir": str(cases_dir),
            "case_output_count": len(case_outputs),
        }
    )
    rows = [
        {
            **row,
            "case_output_file": output_paths.get(
                (str(row["case_id"]), int(row["repeat"]))
            ),
        }
        for row in result.rows
    ]
    summary = {
        "metrics": result.metrics,
        "artifacts": public_artifacts,
        "rows": rows,
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    (run_dir / "summary.json").write_text(payload, encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(payload, encoding="utf-8")
    return summary


def _product_case_output(
    case: HarnessCase,
    result: HarnessCaseResult,
    repeat: int,
) -> dict[str, Any]:
    from travel_agent.evaluation.artifact_contract import (
        ARTIFACT_CONTRACT_VERSION,
        expected_artifact_type,
    )

    state = dict((result.final_profile or {}).get("constraint_state") or {})
    expected = result.metrics.get("expected_artifact_type") or expected_artifact_type(case)
    return _json_safe(
        {
            "schema_version": "product-case-output-v1",
            "case": asdict(case),
            "execution": {
                "repeat": repeat,
                "passed": result.passed,
                "errors": [_sanitize_error(error) for error in result.errors],
                "active_constraint_revision": state.get("_constraint_revision"),
                "active_constraint_hash": state.get("_constraint_hash"),
            },
            "turns": [asdict(turn) for turn in result.turns],
            "final_profile": result.final_profile,
            "final_artifacts": result.final_artifacts,
            "final_itinerary": result.final_artifacts.get("itinerary"),
            "evaluation": {
                "rule_metrics": result.metrics,
                "independent_judge": None,
                "human_review": None,
                "evaluator_version": _evaluator_version(),
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
                "artifact_contract_fingerprint": _artifact_contract_fingerprint([case]),
                "expected_artifact_type": expected,
                "actual_artifact_type": result.metrics.get("actual_artifact_type"),
            },
        }
    )


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    sensitive_keys = {
        "api_key",
        "authorization",
        "access_token",
        "password",
        "secret",
        "web_key",
        "js_key",
        "js_security_key",
    }
    if key and key.lower() in sensitive_keys:
        return "[REDACTED]"
    if is_dataclass(value):
        return _json_safe(asdict(value), key=key)
    if isinstance(value, dict):
        return {str(item_key): _json_safe(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def aggregate_product_results(
    executions: list[tuple[HarnessCase, HarnessCaseResult, int]],
    *,
    remediation_complete: bool,
    variant: str = "V3",
) -> dict[str, Any]:
    rows = [
        _product_row(case, result, repeat, variant=variant)
        for case, result, repeat in executions
    ]
    return aggregate_product_rows(rows, remediation_complete=remediation_complete)


def aggregate_product_rows(
    rows: list[dict[str, Any]],
    *,
    remediation_complete: bool,
) -> dict[str, Any]:
    primary = [row for row in rows if row["repeat"] == 1]
    full_itinerary = [row for row in primary if row.get("expected_artifact_type") == "full_itinerary"]
    non_itinerary = [
        row for row in primary
        if row.get("expected_artifact_type") not in {None, "full_itinerary", "partial_itinerary"}
    ]
    proportions = {
        # 主指标：外部 production_v1 评分口径（七项合取 + 一票否决）
        "strict_success_rate": _bool_rate(primary, "strict_task_success"),
        "overall_strict_success_rate": _bool_rate(primary, "strict_task_success"),
        "artifact_type_match_rate": _bool_rate(primary, "artifact_type_match"),
        "full_itinerary_success_rate": _bool_rate(full_itinerary, "strict_task_success"),
        "non_itinerary_success_rate": _bool_rate(non_itinerary, "strict_task_success"),
        "expected_itinerary_missing_rate": (
            sum(row.get("failure_reason") == "missing_expected_artifact" for row in full_itinerary)
            / len(full_itinerary) if full_itinerary else None
        ),
        "gating_pass_rate": _bool_rate(primary, "gating_passed"),
        "grounding_pass_rate": _bool_rate(primary, "grounding_ok"),
        "authorization_pass_rate": _bool_rate(primary, "authorization_ok"),
        "architecture_policy_pass_rate": _bool_rate(primary, "architecture_policy_ok"),
        "outcome_match_rate": _bool_rate(primary, "expected_outcome_match"),
        # 次要指标：项目内细项（plan 质量/记忆/交付归因）保留作诊断
        "task_completion_rate": _bool_rate(primary, "task_completed"),
        "hard_constraint_rate": _bool_rate(primary, "hard_constraints_ok"),
        "soft_preference_rate": _bool_rate(primary, "soft_preferences_ok"),
        "critic_pass_rate": _bool_rate(primary, "critic_passed"),
        "clarification_accuracy": _bool_rate(primary, "clarification_ok"),
        "tool_schema_valid_rate": _bool_rate(primary, "tool_schema_valid"),
        "fault_recovery_rate": _bool_rate(primary, "fault_recovery_ok"),
        "memory_accuracy": _bool_rate(primary, "memory_ok"),
        "turn_state_pass_rate": _bool_rate(primary, "turn_state_ok"),
    }
    turn_state_scores = [
        float(row["turn_state_accuracy"])
        for row in primary
        if row.get("turn_state_accuracy") is not None
    ]
    turn_state_accuracy = (
        round(statistics.mean(turn_state_scores), 4) if turn_state_scores else None
    )
    tool_required = sum(row["tool_required_count"] for row in primary)
    tool_hits = sum(row["tool_required_hit_count"] for row in primary)
    tool_selected = sum(row["tool_selected_count"] for row in primary)
    tool_unexpected = sum(row["tool_unexpected_count"] for row in primary)
    argument_total = sum(row["tool_argument_assertion_count"] for row in primary)
    argument_pass = sum(row["tool_argument_pass_count"] for row in primary)
    tool_recall = tool_hits / tool_required if tool_required else None
    tool_precision = (
        max(0.0, (tool_selected - tool_unexpected) / tool_selected)
        if tool_selected
        else None
    )
    tool_f1 = (
        2 * tool_precision * tool_recall / (tool_precision + tool_recall)
        if tool_precision is not None and tool_recall is not None and tool_precision + tool_recall
        else None
    )
    delivery_rows = [row for row in primary if row.get("task_completed") is not None]
    completed_rows = [row for row in delivery_rows if row.get("task_completed") is True]
    react_attempted = sum(row.get("react_attempted") is True for row in delivery_rows)
    react_completed = sum(row.get("delivery_mode") == "react" for row in completed_rows)
    fallback_completed = sum(row.get("delivery_mode") == "fallback" for row in completed_rows)
    offline_completed = sum(row.get("delivery_mode") == "offline" for row in completed_rows)
    durations = [row["duration_ms"] for row in rows if row["duration_ms"] is not None]
    token_values = [row["total_tokens"] for row in rows if row["total_tokens"] is not None]
    input_token_values = [row["input_tokens"] for row in rows if row.get("input_tokens") is not None]
    output_token_values = [row["output_tokens"] for row in rows if row.get("output_tokens") is not None]
    cost_values = [float(row.get("estimated_cost_usd") or 0.0) for row in rows]
    llm_call_values = [int(row.get("model_call_count") or 0) for row in rows]
    tool_call_values = [int(row.get("tool_call_count") or 0) for row in rows]
    dispatch_values = [int(row.get("dispatch_count") or 0) for row in rows]
    confidence_intervals = {
        name: wilson_interval(*_bool_counts(primary, _metric_row_key(name)))
        for name, value in proportions.items()
        if value is not None
        and name not in {
            "full_itinerary_success_rate",
            "non_itinerary_success_rate",
            "expected_itinerary_missing_rate",
        }
    }
    if full_itinerary:
        confidence_intervals["full_itinerary_success_rate"] = wilson_interval(
            *_bool_counts(full_itinerary, "strict_task_success")
        )
        confidence_intervals["expected_itinerary_missing_rate"] = wilson_interval(
            sum(row.get("failure_reason") == "missing_expected_artifact" for row in full_itinerary),
            len(full_itinerary),
        )
    if non_itinerary:
        confidence_intervals["non_itinerary_success_rate"] = wilson_interval(
            *_bool_counts(non_itinerary, "strict_task_success")
        )
    from travel_agent.harness.state_metrics import internal_state_metrics

    internal_state = internal_state_metrics(primary)
    internal_metrics = {
        key: internal_state[key]
        for key in (
            "planner_admission_rate",
            "planner_budget_exhaustion_rate",
            "duplicate_dispatch_rate",
            "artifact_reuse_rate",
            "stale_artifact_reuse_rate",
            "stale_parent_rebuild_reuse_rate",
            "final_current_artifact_missing_rate",
            "system_error_rate",
            "stage_timeout_attribution",
        )
    }
    fine_tuning = decide_fine_tuning(primary, remediation_complete=remediation_complete)
    return {
        "sample_size": len(primary),
        "execution_count": len(rows),
        **proportions,
        "turn_state_accuracy": turn_state_accuracy,
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "tool_f1": tool_f1,
        "tool_argument_accuracy": argument_pass / argument_total if argument_total else None,
        "react_attempt_rate": react_attempted / len(delivery_rows) if delivery_rows else None,
        "react_delivery_rate": react_completed / len(delivery_rows) if delivery_rows else None,
        "fallback_delivery_rate": fallback_completed / len(delivery_rows) if delivery_rows else None,
        "offline_delivery_rate": offline_completed / len(delivery_rows) if delivery_rows else None,
        "fallback_share_of_completed": (
            fallback_completed / len(completed_rows) if completed_rows else None
        ),
        "latency_p50_ms": _percentile(durations, 0.50),
        "latency_p95_ms": _percentile(durations, 0.95),
        "avg_total_tokens": statistics.mean(token_values) if token_values else None,
        "avg_input_tokens": statistics.mean(input_token_values) if input_token_values else None,
        "avg_output_tokens": statistics.mean(output_token_values) if output_token_values else None,
        "total_estimated_cost_usd": round(sum(cost_values), 8),
        "avg_llm_calls": statistics.mean(llm_call_values) if llm_call_values else None,
        "avg_tool_calls": statistics.mean(tool_call_values) if tool_call_values else None,
        "avg_dispatches": statistics.mean(dispatch_values) if dispatch_values else None,
        **internal_metrics,
        "stability_rate": _stability_rate(rows),
        "confidence_intervals_95": confidence_intervals,
        "failure_attribution_counts": _attribution_counts(primary),
        "fine_tuning": fine_tuning,
        "cases": rows,
    }


def decide_fine_tuning(
    primary_rows: list[dict[str, Any]],
    *,
    remediation_complete: bool,
) -> dict[str, Any]:
    frozen = [row for row in primary_rows if row["split"] in FROZEN_SPLITS]
    if len(frozen) < FINE_TUNING_FROZEN_COUNT:
        return {
            "decision": "insufficient_frozen_evidence",
            "rationale": (
                f"冻结集（core+challenge {FINE_TUNING_FROZEN_COUNT} 条）尚未完整运行，"
                "不能形成微调 go/no-go 结论。"
            ),
            "model_failure_counts": {key: 0 for key in MODEL_FAILURE_CATEGORIES},
            "triggered_categories": {},
            "frozen_case_count": len(frozen),
            "remediation_complete": remediation_complete,
            "training_data_policy": TRAINING_DATA_POLICY,
        }
    counts = _attribution_counts(frozen)
    model_failures = {key: counts.get(key, 0) for key in MODEL_FAILURE_CATEGORIES}
    candidates = {
        key: count
        for key, count in model_failures.items()
        if count >= 10 or (frozen and count / len(frozen) >= 0.20)
    }
    if not candidates:
        decision = "no_fine_tuning"
        rationale = "未发现达到门槛的系统性模型归因错误。"
    elif not remediation_complete:
        decision = "prompt_or_workflow_first"
        rationale = "存在系统性模型错误，但尚未完成开发集 Prompt、few-shot、tool schema 与工作流修正。"
    else:
        decision = "recommend_sft"
        rationale = "开发集修正后，冻结集仍存在达到门槛的系统性模型错误。"
    return {
        "decision": decision,
        "rationale": rationale,
        "model_failure_counts": model_failures,
        "triggered_categories": candidates,
        "frozen_case_count": len(frozen),
        "remediation_complete": remediation_complete,
        "training_data_policy": TRAINING_DATA_POLICY,
    }


def build_product_report(result: HarnessSuiteResult) -> str:
    metrics = result.metrics
    decision = result.artifacts.get("fine_tuning") or {}
    lines = [
        "# production_v1 Evaluation Report (192 cases)",
        "",
        f"- dataset: `{result.artifacts.get('dataset_version')}`",
        f"- model: `{result.artifacts.get('model')}`",
        f"- independent cases: {result.artifacts.get('independent_case_count')}",
        f"- executions: {result.artifacts.get('execution_count')}",
        f"- strict success (主指标): {metrics.get('strict_success_rate')}",
        f"- gating pass: {metrics.get('gating_pass_rate')}",
        f"- grounding pass: {metrics.get('grounding_pass_rate')}",
        f"- authorization pass: {metrics.get('authorization_pass_rate')}",
        f"- architecture policy pass: {metrics.get('architecture_policy_pass_rate')}",
        f"- outcome match: {metrics.get('outcome_match_rate')}",
        f"- task completion (次要): {metrics.get('task_completion_rate')}",
        f"- hard constraint pass (次要): {metrics.get('hard_constraint_rate')}",
        f"- clarification accuracy: {metrics.get('clarification_accuracy')}",
        f"- tool F1: {metrics.get('tool_f1')}",
        f"- memory accuracy: {metrics.get('memory_accuracy')}",
        f"- stability: {metrics.get('stability_rate')}",
        "",
        "## Fine-tuning decision",
        "",
        f"- decision: `{decision.get('decision')}`",
        f"- rationale: {decision.get('rationale')}",
        f"- triggered categories: {decision.get('triggered_categories') or {}}",
    ]
    return "\n".join(lines) + "\n"


def _product_row(
    case: HarnessCase,
    result: HarnessCaseResult,
    repeat: int,
    *,
    variant: str = "V3",
) -> dict[str, Any]:
    last = result.last_turn
    attributions = attribute_failures(case, result)
    model_calls = [call for turn in result.turns for call in turn.model_calls]
    model_call_errors = [
        _sanitize_error(str(call.get("error")))
        for call in model_calls
        if call.get("error")
    ]
    total_tokens = [call.get("total_tokens") for call in model_calls if call.get("total_tokens") is not None]
    meter_totals = [
        (turn.turn_metrics or {}).get("totals") or {}
        for turn in result.turns
    ]
    metered_input_tokens = sum(int(item.get("input_tokens") or 0) for item in meter_totals)
    metered_output_tokens = sum(int(item.get("output_tokens") or 0) for item in meter_totals)
    metered_tokens = sum(int(item.get("total_tokens") or 0) for item in meter_totals)
    metered_llm_calls = sum(int(item.get("llm_calls") or 0) for item in meter_totals)
    metered_tool_calls = sum(int(item.get("tool_calls") or 0) for item in meter_totals)
    metered_dispatches = sum(int(item.get("dispatch_count") or 0) for item in meter_totals)
    metered_cost_usd = sum(float(item.get("cost_usd") or 0.0) for item in meter_totals)
    metrics = result.metrics
    itinerary = result.final_artifacts.get("itinerary")
    from travel_agent.evaluation.artifact_contract import ITINERARY_TYPES, expected_artifact_type

    expected_artifact = metrics.get("expected_artifact_type") or expected_artifact_type(case)
    expected_task = expected_artifact in ITINERARY_TYPES
    fallback_used = bool(
        model_call_errors
        and last
        and isinstance(last.reply_text, str)
        and ("离线兜底" in last.reply_text or not last.used_real_agent)
    )
    task_completed = bool(itinerary) if expected_task else None
    completed_via_fallback = bool(itinerary and fallback_used)
    delivered_by_react = bool(itinerary and last and last.used_real_agent and not fallback_used)
    strict_task_success = bool(metrics.get("strict_task_success"))
    from travel_agent.harness.state_metrics import extract_internal_counters

    internal_counters = extract_internal_counters(result)
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "subset": case.subset or case.category,
        "gold_outcome": case.gold_outcome,
        "high_risk": case.high_risk,
        "repeat": repeat,
        "variant": variant,
        # passed 以外部 strict_task_success 为准（七项合取 + gating）
        "passed": strict_task_success,
        "strict_task_success": strict_task_success,
        "actual_outcome": metrics.get("actual_outcome"),
        "expected_artifact_type": expected_artifact,
        "actual_artifact_type": metrics.get("actual_artifact_type"),
        "artifact_type_match": metrics.get("artifact_type_match"),
        "failure_reason": metrics.get("failure_reason"),
        "task_completion_judge": metrics.get("task_completion_judge"),
        "llm_judge": metrics.get("llm_judge"),
        "expected_outcome_match": metrics.get("expected_outcome_match"),
        "gating_passed": metrics.get("gating_passed"),
        "gating_triggered": metrics.get("gating_triggered") or [],
        "grounding_ok": metrics.get("grounding_ok"),
        "authorization_ok": metrics.get("authorization_ok"),
        "architecture_policy_ok": metrics.get("architecture_policy_ok"),
        "constraint_tree_score": metrics.get("constraint_tree_score"),
        "constraint_tree_missing": metrics.get("constraint_tree_missing") or [],
        "task_completed": None if not expected_task else task_completed,
        "task_completed_via_fallback": completed_via_fallback,
        "react_attempted": _used_real_agent(result),
        "used_real_agent": bool(last and last.used_real_agent),
        "delivery_mode": (
            "fallback" if completed_via_fallback
            else "react" if delivered_by_react
            else "offline" if itinerary
            else "none"
        ),
        "hard_constraints_ok": _all_optional(
            metrics,
            ["city_ok", "days_ok", "constraint_pass", "hard_constraints_annotation_ok"],
        ),
        "soft_preferences_ok": _all_optional(
            metrics, ["preference_pass", "soft_preferences_annotation_ok"]
        ),
        "critic_passed": metrics.get("critic_passed"),
        "clarification_ok": metrics.get("clarification_ok"),
        "tool_schema_valid": metrics.get("tool_schema_valid"),
        "fault_recovery_ok": metrics.get("fault_recovery_ok"),
        "memory_ok": metrics.get("memory_ok"),
        "turn_state_ok": metrics.get("turn_state_ok"),
        "turn_state_accuracy": metrics.get("turn_state_accuracy"),
        "turn_state_results": metrics.get("turn_state_results") or [],
        "tool_required_count": metrics.get("tool_required_count", 0),
        "tool_required_hit_count": metrics.get("tool_required_hit_count", 0),
        "tool_selected_count": metrics.get("tool_selected_count", 0),
        "tool_unexpected_count": metrics.get("tool_unexpected_count", 0),
        "tool_argument_assertion_count": metrics.get("tool_argument_assertion_count", 0),
        "tool_argument_pass_count": metrics.get("tool_argument_pass_count", 0),
        "tool_trace": [name for turn in result.turns for name in turn.tool_trace],
        "duration_ms": sum(turn.duration_ms or 0 for turn in result.turns),
        "input_tokens": metered_input_tokens or None,
        "output_tokens": metered_output_tokens or None,
        "total_tokens": metered_tokens or (sum(total_tokens) if total_tokens else None),
        "estimated_cost_usd": round(metered_cost_usd, 8),
        "model_call_count": metered_llm_calls or len(model_calls),
        "model_error_call_count": len(model_call_errors),
        "model_call_errors": model_call_errors,
        "tool_call_count": metered_tool_calls or sum(len(turn.tool_calls) for turn in result.turns),
        "dispatch_count": metered_dispatches,
        "failure_attributions": attributions,
        "errors": [_sanitize_error(error) for error in result.errors],
        **internal_counters,
    }


def _execution_case(
    case: HarnessCase,
    repeat_index: int,
    *,
    run_namespace: str | None = None,
) -> HarnessCase:
    prefix = f"product:{run_namespace}" if run_namespace else "product"
    namespace = f"{prefix}:{case.case_id}:repeat-{repeat_index}"
    if case.user_ids:
        user_ids = [f"{namespace}:{user_id}" for user_id in case.user_ids]
        metadata = dict(case.metadata)
    else:
        user_ids = []
        metadata = {**case.metadata, "user_id": namespace}
    return replace(case, user_ids=user_ids, metadata=metadata)


def attribute_failures(case: HarnessCase, result: HarnessCaseResult) -> list[str]:
    metrics = result.metrics
    failures: list[str] = []
    error_text = " ".join(result.errors).lower()
    model_errors = [
        str(call.get("error"))
        for turn in result.turns
        for call in turn.model_calls
        if call.get("error")
    ]
    if model_errors:
        model_error_text = " ".join(model_errors).lower()
        failures.append(
            "external_api"
            if any(key in model_error_text for key in ("timeout", "quota", "auth", "429", "connection"))
            else "adapter_or_environment"
        )
    if result.errors:
        failures.append(
            "external_api"
            if any(key in error_text for key in ("timeout", "quota", "auth", "429", "connection"))
            else "adapter_or_environment"
        )
    if metrics.get("clarification_ok") is False:
        failures.append("model_clarification" if _used_real_agent(result) else "planner_or_workflow")
    injection_triggered = _injection_triggered(case, result)
    if (
        (not case.failure_injection or not injection_triggered)
        and (metrics.get("tool_coverage_ok") is False or metrics.get("forbidden_tools_ok") is False)
    ):
        failures.append("model_tool_selection" if _used_real_agent(result) else "planner_or_workflow")
    if metrics.get("tool_arguments_ok") is False:
        failures.append("model_tool_arguments" if _used_real_agent(result) else "planner_or_workflow")
    if metrics.get("tool_schema_valid") is False:
        failures.append("model_tool_arguments" if _used_real_agent(result) else "adapter_or_environment")
    if metrics.get("city_ok") is False or metrics.get("days_ok") is False:
        failures.append("model_tool_arguments" if _used_real_agent(result) else "planner_or_workflow")
    if metrics.get("memory_ok") is False:
        failures.append("memory_or_storage")
    if metrics.get("fault_recovery_ok") is False:
        failures.append("tool_or_data")
    if metrics.get("critic_passed") is False:
        failures.append("critic_or_validator")
    if metrics.get("constraint_pass") is False or metrics.get("route_feasible") is False:
        failures.append("planner_or_workflow")
    if metrics.get("hard_constraints_annotation_ok") is False:
        failures.append("planner_or_workflow")
    if metrics.get("preference_pass") is False or metrics.get("soft_preferences_annotation_ok") is False:
        failures.append("planner_or_workflow")
    if metrics.get("required_behaviors_ok") is False or metrics.get("forbidden_behaviors_ok") is False:
        failures.append(
            "model_intent"
            if _used_real_agent(result) and case.category in {"clarification", "constraint_conflict"}
            else "planner_or_workflow"
        )
    if case.expect_itinerary is True and not result.final_artifacts.get("itinerary") and not result.errors:
        failures.append("model_intent" if _used_real_agent(result) else "planner_or_workflow")
    if result.passed is False and not failures:
        failures.append("adapter_or_environment")
    return list(dict.fromkeys(item for item in failures if item in FAILURE_CATEGORIES))


def _injection_triggered(case: HarnessCase, result: HarnessCaseResult) -> bool:
    operation = case.failure_injection.get("operation")
    tool_name = {
        "search_pois": "search_poi",
        "get_weather": "check_weather",
        "estimate_route": "plan_route",
    }.get(operation)
    if not tool_name:
        return False
    return any(
        tool_name in turn.tool_trace
        or any(call.get("name") == tool_name for call in turn.tool_calls)
        for turn in result.turns
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | int] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"successes": successes, "total": total, "low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _all_optional(metrics: dict[str, Any], keys: list[str]) -> bool | None:
    values = [metrics.get(key) for key in keys if metrics.get(key) is not None]
    return all(values) if values else None


def _used_real_agent(result: HarnessCaseResult) -> bool:
    return any(turn.used_real_agent or turn.model_calls for turn in result.turns)


def _bool_counts(rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
    values = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
    return sum(values), len(values)


def _bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    successes, total = _bool_counts(rows, key)
    return successes / total if total else None


def _metric_row_key(name: str) -> str:
    return {
        "strict_success_rate": "strict_task_success",
        "overall_strict_success_rate": "strict_task_success",
        "artifact_type_match_rate": "artifact_type_match",
        "gating_pass_rate": "gating_passed",
        "grounding_pass_rate": "grounding_ok",
        "authorization_pass_rate": "authorization_ok",
        "architecture_policy_pass_rate": "architecture_policy_ok",
        "outcome_match_rate": "expected_outcome_match",
        "task_completion_rate": "task_completed",
        "hard_constraint_rate": "hard_constraints_ok",
        "soft_preference_rate": "soft_preferences_ok",
        "critic_pass_rate": "critic_passed",
        "clarification_accuracy": "clarification_ok",
        "tool_schema_valid_rate": "tool_schema_valid",
        "fault_recovery_rate": "fault_recovery_ok",
        "memory_accuracy": "memory_ok",
        "turn_state_pass_rate": "turn_state_ok",
    }[name]


def _attribution_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {category: 0 for category in sorted(FAILURE_CATEGORIES)}
    for row in rows:
        for category in row.get("failure_attributions") or []:
            counts[category] += 1
    return counts


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _stability_rate(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    repeated = [items for items in grouped.values() if len(items) > 1]
    if not repeated:
        return None
    stable = 0
    for items in repeated:
        signatures = {(item["passed"], tuple(item["tool_trace"])) for item in items}
        stable += len(signatures) == 1
    return stable / len(repeated)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        try:
            relative = path.relative_to(ROOT.resolve())
        except ValueError:
            relative = path
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _code_fingerprint() -> str:
    return _files_fingerprint(list((ROOT / "src" / "travel_agent").rglob("*.py")))


def _evaluator_fingerprint() -> str:
    return _files_fingerprint([
        *(ROOT / "src" / "travel_agent" / "evaluation").rglob("*.py"),
        *(ROOT / "src" / "travel_agent" / "harness").rglob("*.py"),
    ])


def _tooling_fingerprint() -> str:
    return _files_fingerprint([
        ROOT / "src" / "travel_agent" / "agent" / "tool_contract.py",
        ROOT / "src" / "travel_agent" / "agent" / "toolkit.py",
        ROOT / "src" / "travel_agent" / "providers.py",
        ROOT / "src" / "travel_agent" / "route_evidence.py",
        ROOT / "src" / "travel_agent" / "poi_evidence.py",
    ])


def _configuration_fingerprint(settings: Any, tool_provider_mode: str) -> str:
    """Hash behavior-affecting runtime settings without persisting credentials."""
    payload = {
        "llm": _public_settings(settings.llm, excluded={"api_key"}),
        "amap": {
            "base_url": settings.amap.base_url,
            "timeout_seconds": settings.amap.timeout_seconds,
            "rest_enabled": settings.amap.rest_enabled,
        },
        "agent": _public_settings(settings.agent),
        "memory": _public_settings(
            settings.memory,
            excluded={"database_url", "profile_dir"},
        ),
        "orchestration": _public_settings(settings.orchestration),
        "mcp": _public_settings(settings.mcp),
        "skills": {
            "enabled": settings.skills.enabled,
            "skills_dir": str(settings.skills.skills_dir),
        },
        "hybrid_planning": _public_settings(settings.hybrid_planning),
        "judge": _public_settings(settings.evaluation.judge, excluded={"api_key"}),
        "tool_provider_mode": tool_provider_mode,
    }
    return _json_fingerprint(payload)


def _tool_snapshot_fingerprint(settings: Any, tool_provider_mode: str) -> str:
    """Identify the provider/config/code snapshot used by a Product run."""
    payload = {
        "tool_provider_mode": tool_provider_mode,
        "tooling_fingerprint": _tooling_fingerprint(),
        "amap": {
            "base_url": settings.amap.base_url,
            "timeout_seconds": settings.amap.timeout_seconds,
            "rest_enabled": settings.amap.rest_enabled,
        },
        "mcp": _public_settings(settings.mcp),
    }
    return _json_fingerprint(payload)


def _public_settings(value: Any, *, excluded: set[str] | None = None) -> dict[str, Any]:
    payload = asdict(value) if is_dataclass(value) else dict(value or {})
    for field_name in excluded or set():
        payload.pop(field_name, None)
    return payload


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluator_version() -> str:
    from travel_agent.evaluation.artifact_contract import ARTIFACT_CONTRACT_VERSION

    return f"product-v1.1+{ARTIFACT_CONTRACT_VERSION}"


def _artifact_contract_fingerprint(cases: list[HarnessCase]) -> str:
    from travel_agent.evaluation.artifact_contract import artifact_contract_fingerprint

    return artifact_contract_fingerprint(cases)


def _prompt_fingerprint() -> str:
    paths = [
        ROOT / "src" / "travel_agent" / "agent" / "prompts.py",
        ROOT / "src" / "travel_agent" / "agent" / "turn_analysis.py",
        ROOT / "src" / "travel_agent" / "orchestration" / "multi_agent" / "orchestrator_agent.py",
        ROOT / "src" / "travel_agent" / "orchestration" / "multi_agent" / "registry.py",
        ROOT / "src" / "travel_agent" / "orchestration" / "multi_agent" / "review.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _sanitize_error(value: str) -> str:
    text = re.sub(
        r"(?i)(api[_-]?key|authorization|token|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        str(value),
    )
    return text[:500]
