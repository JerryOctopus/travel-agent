"""Deterministic acceptance contract for two frozen Dev34 candidate runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from travel_agent.evaluation.artifact_contract import (
    actual_artifact_type,
    expected_artifact_type,
)


DEV34_ROUTE_CASES = frozenset({"dev_006", "dev_011", "dev_027"})
DEV34_LOCAL_ADJUSTMENT_CASES = frozenset({"dev_010"})
DEV34_COMPARISON_CASES = frozenset(
    {
        "dev_007", "dev_008", "dev_009", "dev_012",
        "dev_015", "dev_026", "dev_028", "dev_029",
    }
)
DEV34_LONG_HORIZON_CASES = frozenset(
    {"lh_5_001", "lh_8_001", "lh_12_001", "lh_12_002"}
)
DEV34_CASE_IDS = frozenset(
    {*(f"dev_{number:03d}" for number in range(1, 31)), *DEV34_LONG_HORIZON_CASES}
)
DEV34_EXPECTED_ARTIFACTS = {
    case_id: (
        "route_plan"
        if case_id in DEV34_ROUTE_CASES
        else "local_adjustment_advice"
        if case_id in DEV34_LOCAL_ADJUSTMENT_CASES
        else "candidate_comparison"
        if case_id in DEV34_COMPARISON_CASES
        else "full_itinerary"
    )
    for case_id in DEV34_CASE_IDS
}
DEV34_DATASET_VERSION = "travel-agent-eval-production-v1.1"
DEV34_DATASET_SHA256 = "ff47dfa5d429baae1a173095fef8d1626f59ae32424407e4b38bf649c71de477"
DEV34_EVALUATOR_VERSION = "product-v1.1+dev34-artifact-contract-v3"
DEV34_ARTIFACT_CONTRACT_FINGERPRINT = (
    "7478fe592648c408fad4b4600e0619c2298120bb1f82a221fbd097ba3a4b180c"
)
DEV34_JUDGE_PROVIDER = "siliconflow"
DEV34_JUDGE_MODEL = "Qwen/Qwen3.5-397B-A17B"


FINGERPRINT_FIELDS = (
    "dataset_version",
    "dataset_sha256",
    "code_fingerprint",
    "prompt_fingerprint",
    "evaluator_version",
    "evaluator_fingerprint",
    "tooling_fingerprint",
    "configuration_fingerprint",
    "tool_snapshot_fingerprint",
    "artifact_contract_fingerprint",
    "model_execution_mode",
    "model_provider",
    "model",
    "model_temperature",
    "model_thinking_enabled",
    "tool_provider_mode",
    "hybrid_flags",
    "judge_provider",
    "judge_model",
    "judge_rubric_version",
    "judge_prompt_version",
    "judge_schema_version",
)


def evaluate_dev34_run(
    run_dir: Path | str, *, require_judge: bool = True
) -> dict[str, Any]:
    root = Path(run_dir)
    summary = _read_json(root / "summary.json")
    cases = [_read_json(path) for path in sorted((root / "cases").glob("*.json"))]
    rows = list(summary.get("rows") or [])
    artifacts = dict(summary.get("artifacts") or {})
    metrics = dict(summary.get("metrics") or {})
    failures: list[str] = []

    case_ids = [str((case.get("case") or {}).get("case_id") or "") for case in cases]
    row_ids = [str(row.get("case_id") or "") for row in rows]
    _require(len(cases) == 34, failures, f"case outputs must be 34, got {len(cases)}")
    _require(len(set(case_ids)) == 34 and "" not in case_ids, failures, "case ids must be 34 unique non-empty ids")
    _require(len(rows) == 34, failures, f"summary rows must be 34, got {len(rows)}")
    _require(set(row_ids) == set(case_ids), failures, "summary rows and case outputs disagree")
    _require(
        set(case_ids) == DEV34_CASE_IDS,
        failures,
        "case outputs must match the exact frozen Dev34 case ids",
    )
    row_artifacts = {
        str(row.get("case_id") or ""): str(row.get("expected_artifact_type") or "")
        for row in rows
    }
    case_artifacts = {
        str((case.get("case") or {}).get("case_id") or ""): expected_artifact_type(
            case.get("case") or {}
        )
        for case in cases
    }
    _require(
        row_artifacts == DEV34_EXPECTED_ARTIFACTS
        and case_artifacts == DEV34_EXPECTED_ARTIFACTS,
        failures,
        "case outputs and summary rows must use the exact artifact mapping for Dev34",
    )

    row_execution_once = all(
        int(row.get("repeat") or 0) == 1
        and int(row.get("execution_total") or 0) == 1
        for row in rows
    )
    case_execution_once = all(
        int((case.get("execution") or {}).get("repeat") or 0) == 1
        for case in cases
    )
    attempts_by_case = metrics.get("model_attempts_by_case") or {}
    attempts_once = (
        set(attempts_by_case) == DEV34_CASE_IDS
        and all(int(value or 0) == 1 for value in attempts_by_case.values())
    )
    relay_summary = artifacts.get("relay_summary") or {}
    relay_once = (
        int(relay_summary.get("attempts_total") or 0) == 34
        and int(relay_summary.get("current_model_index") or 0) == 0
    )
    _require(
        row_execution_once and case_execution_once and attempts_once and relay_once,
        failures,
        "every Dev34 case must execute exactly once with one fixed-model attempt",
    )

    full_rows = [row for row in rows if row.get("expected_artifact_type") == "full_itinerary"]
    non_rows = [row for row in rows if row.get("expected_artifact_type") != "full_itinerary"]
    strict = [row for row in rows if row.get("strict_task_success") is True]
    full_strict = [row for row in full_rows if row.get("strict_task_success") is True]
    non_strict = [row for row in non_rows if row.get("strict_task_success") is True]
    long_rows = [row for row in rows if str(row.get("case_id") or "").startswith("lh_")]
    long_strict = [row for row in long_rows if row.get("strict_task_success") is True]

    _require(len(full_rows) == 22, failures, f"full-itinerary contract must be 22, got {len(full_rows)}")
    _require(len(non_rows) == 12, failures, f"non-itinerary contract must be 12, got {len(non_rows)}")
    _require(len(long_rows) == 4, failures, f"long-horizon contract must be 4, got {len(long_rows)}")
    _require(len(strict) >= 26, failures, f"strict success {len(strict)}/34 is below 26")
    _require(len(full_strict) >= 16, failures, f"full-itinerary success {len(full_strict)}/22 is below 16")
    _require(len(non_strict) >= 9, failures, f"non-itinerary success {len(non_strict)}/12 is below 9")
    _require(len(long_strict) >= 3, failures, f"long-horizon strict {len(long_strict)}/4 is below 3")

    boolean_gates = {
        "hard_constraints_ok": 31,
        "grounding_ok": 34,
        "authorization_ok": 34,
        "tool_schema_valid": 34,
        "architecture_policy_ok": 34,
    }
    gate_counts: dict[str, int] = {}
    for field, minimum in boolean_gates.items():
        count = sum(row.get(field) is True for row in rows)
        gate_counts[field] = count
        _require(count >= minimum, failures, f"{field} {count}/34 is below {minimum}")

    execution_errors = _execution_errors(cases, rows)
    _require(not execution_errors, failures, "execution health errors: " + "; ".join(execution_errors[:8]))
    redlines = _redline_cases(rows)
    _require(not redlines, failures, "redline cases: " + ", ".join(redlines))

    _require(artifacts.get("model_provider") == "deepseek", failures, "model provider must be deepseek")
    _require(artifacts.get("model") == "deepseek-chat", failures, "model must be deepseek-chat")
    _require(_same_number(artifacts.get("model_temperature"), 0.2), failures, "model temperature must be 0.2")
    _require(artifacts.get("model_thinking_enabled") is False, failures, "model thinking must be disabled")
    _require(artifacts.get("tool_provider_mode") == "configured", failures, "tool provider must be configured")
    _require(metrics.get("model_switch_count") == 0, failures, "model relay/switching is forbidden")
    _require(metrics.get("relay_mode") is False, failures, "relay mode must be disabled")
    _require(
        artifacts.get("model_execution_mode") == "fixed_single_model",
        failures,
        "model execution mode must be fixed_single_model",
    )
    _require(
        artifacts.get("dataset_version") == DEV34_DATASET_VERSION,
        failures,
        "Dev34 dataset version differs from the frozen contract",
    )
    _require(
        artifacts.get("dataset_sha256") == DEV34_DATASET_SHA256,
        failures,
        "Dev34 dataset fingerprint differs from the frozen contract",
    )
    _require(
        artifacts.get("evaluator_version") == DEV34_EVALUATOR_VERSION,
        failures,
        "evaluator version differs from the frozen contract",
    )
    _require(
        artifacts.get("artifact_contract_fingerprint")
        == DEV34_ARTIFACT_CONTRACT_FINGERPRINT,
        failures,
        "artifact contract fingerprint differs from the frozen contract",
    )
    for field in (
        "code_fingerprint",
        "prompt_fingerprint",
        "evaluator_fingerprint",
        "tooling_fingerprint",
        "configuration_fingerprint",
        "tool_snapshot_fingerprint",
    ):
        _require(bool(artifacts.get(field)), failures, f"{field} must be recorded")
    hybrid = artifacts.get("hybrid_flags") or {}
    _require(
        hybrid and not any(bool(value) for value in hybrid.values()),
        failures,
        "all Hybrid flags must be recorded and disabled",
    )

    judge_cases = []
    for case in cases:
        expected = expected_artifact_type(case.get("case") or {})
        actual = actual_artifact_type(case, expected)
        if expected == "full_itinerary" and actual == "full_itinerary":
            judge_cases.append(case)
    judge_results = [((case.get("evaluation") or {}).get("independent_judge") or {}) for case in judge_cases]
    completed_judges = [item for item in judge_results if item.get("status") == "ok"]
    judge_scores = [float(item["total_score"]) for item in completed_judges if isinstance(item.get("total_score"), (int, float))]
    judge_average = sum(judge_scores) / len(judge_scores) if judge_scores else None
    judge_reasonable_rate = (
        sum(item.get("reasonable") is True for item in completed_judges) / len(completed_judges)
        if completed_judges else None
    )
    critical_issue_rate = (
        sum(any(issue.get("severity") == "critical" for issue in item.get("critical_issues") or []) for item in completed_judges)
        / len(completed_judges)
        if completed_judges else None
    )
    if require_judge:
        _require(len(judge_cases) >= 16, failures, f"Judge-eligible complete itineraries {len(judge_cases)} is below 16")
        _require(len(completed_judges) == len(judge_cases), failures, f"Judge completion {len(completed_judges)}/{len(judge_cases)} is not 100%")
        _require(judge_average is not None and judge_average >= 75, failures, f"Judge average {judge_average} is below 75")
        _require(judge_reasonable_rate is not None and judge_reasonable_rate >= 0.70, failures, f"Judge reasonable rate {judge_reasonable_rate} is below 0.70")
        _require(critical_issue_rate is not None and critical_issue_rate <= 0.20, failures, f"Judge critical issue rate {critical_issue_rate} exceeds 0.20")
        _require(
            artifacts.get("judge_provider") == DEV34_JUDGE_PROVIDER,
            failures,
            f"Judge provider must be {DEV34_JUDGE_PROVIDER}",
        )
        _require(
            artifacts.get("judge_model") == DEV34_JUDGE_MODEL,
            failures,
            f"Judge model must be {DEV34_JUDGE_MODEL}",
        )
        preflight = artifacts.get("judge_preflight") or {}
        _require(preflight.get("ok") is True and not preflight.get("injected"), failures, "real independent Judge preflight was not recorded")
        _require(
            preflight.get("provider") == DEV34_JUDGE_PROVIDER
            and preflight.get("model") == DEV34_JUDGE_MODEL,
            failures,
            "Judge preflight identity differs from the frozen contract",
        )
        _require(
            _same_number(preflight.get("temperature"), 0.0),
            failures,
            "Judge preflight must record temperature 0",
        )
        _require(
            preflight.get("thinking_enabled") is False,
            failures,
            "Judge preflight must record thinking disabled",
        )
        judge_identity_ok = all(
            item.get("provider") == DEV34_JUDGE_PROVIDER
            and item.get("model") == DEV34_JUDGE_MODEL
            and item.get("rubric") == "full_itinerary"
            and item.get("diagnostic_only") is False
            and item.get("rubric_version") == artifacts.get("judge_rubric_version")
            and item.get("prompt_version") == artifacts.get("judge_prompt_version")
            and item.get("schema_version") == artifacts.get("judge_schema_version")
            and item.get("independence_warning") is False
            for item in completed_judges
        )
        _require(
            judge_identity_ok,
            failures,
            "Judge result identity or rubric metadata differs from the frozen contract",
        )

    _check_summary_consistency(metrics, len(strict), len(full_strict), len(non_strict), gate_counts, failures)
    return {
        "passed": not failures,
        "mode": "full_with_judge" if require_judge else "deterministic_only",
        "run_dir": str(root),
        "counts": {
            "cases": len(cases),
            "strict": len(strict),
            "full_itinerary_strict": len(full_strict),
            "non_itinerary_strict": len(non_strict),
            "long_horizon_strict": len(long_strict),
            **gate_counts,
            "judge_eligible": len(judge_cases),
            "judge_completed": len(completed_judges),
        },
        "judge": {
            "average_score": judge_average,
            "reasonable_rate": judge_reasonable_rate,
            "critical_issue_rate": critical_issue_rate,
        },
        "fingerprints": {field: artifacts.get(field) for field in FINGERPRINT_FIELDS},
        "failures": failures,
    }


def evaluate_consecutive_dev34_runs(first: Path | str, second: Path | str) -> dict[str, Any]:
    left = evaluate_dev34_run(first, require_judge=True)
    right = evaluate_dev34_run(second, require_judge=True)
    failures = []
    if not left["passed"]:
        failures.append("first run failed acceptance")
    if not right["passed"]:
        failures.append("second run failed acceptance")
    if Path(first).resolve() == Path(second).resolve():
        failures.append("two distinct fresh run directories are required")
    if left["fingerprints"] != right["fingerprints"]:
        failures.append("run fingerprints differ; candidate was not unchanged")
    return {"passed": not failures, "first": left, "second": right, "failures": failures}


def _execution_errors(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id") or "unknown")
        if row.get("system_error") not in (0, False):
            errors.append(f"{case_id}:system_error")
        if int(row.get("model_error_call_count") or 0) != 0:
            errors.append(f"{case_id}:model_error")
    for case in cases:
        case_id = str((case.get("case") or {}).get("case_id") or "unknown")
        if (case.get("execution") or {}).get("errors") or case.get("errors"):
            errors.append(f"{case_id}:case_error")
        for turn in case.get("turns") or []:
            if turn.get("error"):
                errors.append(f"{case_id}:turn_error")
            if any(call.get("status") == "error" for call in turn.get("model_calls") or []):
                errors.append(f"{case_id}:model_call_error")
            if any(
                call.get("status") == "error"
                and call.get("error_code") != "NOT_FOUND"
                for call in turn.get("tool_calls") or []
            ):
                errors.append(f"{case_id}:tool_call_error")
    return sorted(set(errors))


def _redline_cases(rows: list[dict[str, Any]]) -> list[str]:
    redlines = []
    for row in rows:
        if row.get("grounding_ok") is not True or row.get("authorization_ok") is not True:
            redlines.append(str(row.get("case_id")))
            continue
        triggers = {str(item) for item in row.get("gating_triggered") or []}
        if triggers.intersection({"unauthorized_action", "unsupported_claim", "missing_critical_hard_constraint"}):
            redlines.append(str(row.get("case_id")))
    return redlines


def _check_summary_consistency(metrics: dict[str, Any], strict: int, full: int, non: int, gates: dict[str, int], failures: list[str]) -> None:
    expected = {
        "sample_size": 34,
        "execution_count": 34,
        "strict_success_rate": strict / 34,
        "full_itinerary_success_rate": full / 22,
        "non_itinerary_success_rate": non / 12,
        "hard_constraint_rate": gates["hard_constraints_ok"] / 34,
        "grounding_pass_rate": gates["grounding_ok"] / 34,
        "authorization_pass_rate": gates["authorization_ok"] / 34,
        "tool_schema_valid_rate": gates["tool_schema_valid"] / 34,
        "architecture_policy_pass_rate": gates["architecture_policy_ok"] / 34,
    }
    for field, value in expected.items():
        if not _same_number(metrics.get(field), value):
            failures.append(f"summary metric {field} disagrees with case rows")


def _same_number(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def _require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
