"""Pre-registered release contract for one-shot frozen Product evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from travel_agent.evaluation.artifact_contract import (
    ITINERARY_TYPES,
    actual_artifact_type,
    artifact_contract_fingerprint,
    expected_artifact_type,
    itinerary_judge_route,
)
from travel_agent.evaluation.plan_quality_judge import PARTIAL_RUBRIC_VERSION


FROZEN_RELEASE_SCHEMA_VERSION = "frozen-release-manifest-v1"
FROZEN_ACCEPTANCE_VERSION = "frozen-release-acceptance-v1"
TESTED_MODEL = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "temperature": 0.2,
    "thinking_enabled": False,
}
JUDGE_MODEL = {
    "provider": "siliconflow",
    "model": "Qwen/Qwen3.5-397B-A17B",
    "temperature": 0.0,
    "thinking_enabled": False,
    "rubric_version": "travel-plan-quality-v1",
    "prompt_version": "travel-plan-judge-v4",
    "schema_version": "travel-plan-judge-output-v1",
}
FINGERPRINT_FIELDS = (
    "code_fingerprint",
    "prompt_fingerprint",
    "evaluator_version",
    "evaluator_fingerprint",
    "tooling_fingerprint",
    "configuration_fingerprint",
    "tool_snapshot_fingerprint",
    "artifact_contract_implementation_fingerprint",
)

STAGE_CONTRACTS: dict[str, dict[str, Any]] = {
    "core_frozen": {
        "count": 94,
        "strict": 80,
        "hard": 92,
        "long_horizon": 3,
        "judge_eligible": 80,
        "expected_artifacts": {"full_itinerary": 94},
        "subset_floors": {
            "fixed_time": 8,
            "long_horizon_state": 3,
            "multi_city": 12,
            "multi_turn_update": 8,
            "single_city_multi_day": 16,
            "single_city_one_day": 12,
            "special_traveler": 8,
            "strict_constraints": 8,
        },
        "subset_counts": {
            "fixed_time": 10,
            "long_horizon_state": 4,
            "multi_city": 15,
            "multi_turn_update": 10,
            "single_city_multi_day": 20,
            "single_city_one_day": 15,
            "special_traveler": 10,
            "strict_constraints": 10,
        },
    },
    "challenge_frozen": {
        "count": 34,
        "strict": 27,
        "hard": 32,
        "long_horizon": 3,
        "judge_eligible": 11,
        "expected_artifacts": {
            "full_itinerary": 10,
            "partial_itinerary": 4,
            "clarification": 6,
            "constraint_negotiation": 10,
            "safe_decline": 4,
        },
        "subset_floors": {
            "action_boundary": 4,
            "complex_budget_route": 3,
            "constraint_conflict": 4,
            "location_ambiguity": 3,
            "long_horizon_state": 3,
            "sparse_supply": 3,
            "state_overwrite": 3,
            "time_space_impossible": 4,
        },
        "subset_counts": {
            "action_boundary": 4,
            "complex_budget_route": 4,
            "constraint_conflict": 5,
            "location_ambiguity": 4,
            "long_horizon_state": 4,
            "sparse_supply": 4,
            "state_overwrite": 4,
            "time_space_impossible": 5,
        },
    },
    "shadow_frozen": {
        "count": 30,
        "strict": 24,
        "hard": 29,
        "long_horizon": None,
        "judge_eligible": 24,
        "expected_artifacts": {"full_itinerary": 26, "partial_itinerary": 4},
        "subset_floors": {"real_world_natural_language": 24},
        "subset_counts": {"real_world_natural_language": 30},
    },
}


def artifact_contract_implementation_fingerprint() -> str:
    """Fingerprint the public Artifact contract without a case-specific map."""
    return artifact_contract_fingerprint()


def build_release_manifest(
    *,
    candidate_id: str,
    commit_sha: str,
    branch: str,
    tag: str,
    split_files: Mapping[str, Path | str],
    fingerprints: Mapping[str, str],
    provider_reported_models: list[str],
) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    for split, contract in STAGE_CONTRACTS.items():
        path = Path(split_files[split]).resolve()
        splits[split] = {
            "path": str(path),
            "count": _nonempty_line_count(path),
            "sha256": _sha256(path),
            "dataset_version": "travel-agent-eval-production-v1.1",
            "thresholds": contract,
        }
    manifest: dict[str, Any] = {
        "schema_version": FROZEN_RELEASE_SCHEMA_VERSION,
        "acceptance_version": FROZEN_ACCEPTANCE_VERSION,
        "candidate_id": candidate_id,
        "commit_sha": commit_sha,
        "branch": branch,
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": splits,
        "fingerprints": dict(fingerprints),
        "tested_model": dict(TESTED_MODEL),
        "provider_reported_models": sorted(set(provider_reported_models)),
        "judge": dict(JUDGE_MODEL),
        "tool_provider_mode": "configured",
        "hybrid_flags_required_off": True,
        "model_relay_forbidden": True,
        "environment_failure_policy": "invalid_quarantine_and_exact_candidate_rerun",
        "frozen_failure_policy": "seal_results_rotate_consumed_splits_and_new_candidate",
    }
    manifest["manifest_fingerprint"] = _manifest_fingerprint(manifest)
    return manifest


def load_release_manifest(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release manifest must be a JSON object")
    return payload


def load_frozen_split_cases(
    manifest: Mapping[str, Any], split: str
) -> tuple[Path, list[Any]]:
    """Load exactly one pre-registered frozen split after checking its byte hash."""
    failures = validate_release_manifest(manifest)
    if split not in STAGE_CONTRACTS:
        failures.append(f"unknown frozen split: {split}")
    if failures:
        raise ValueError("invalid release manifest: " + "; ".join(failures))
    split_contract = manifest["splits"][split]
    path = Path(str(split_contract["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"frozen split file does not exist: {path}")
    if _sha256(path) != split_contract["sha256"]:
        raise ValueError(f"frozen split hash mismatch: {split}")
    if _nonempty_line_count(path) != split_contract["count"]:
        raise ValueError(f"frozen split count mismatch: {split}")
    from travel_agent.harness.cases import load_cases_json

    cases = load_cases_json(path)
    if len(cases) != split_contract["count"]:
        raise ValueError(f"parsed frozen split count mismatch: {split}")
    if any(case.split != split for case in cases):
        raise ValueError(f"frozen split contains rows outside {split}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError(f"frozen split contains duplicate case ids: {split}")
    return path, cases


def reject_unofficial_frozen_request(split: str) -> None:
    """Prevent legacy/nonofficial entrypoints from opening sealed release data."""
    if split != "dev":
        raise RuntimeError(
            "Core/Challenge/Shadow (and aggregate all) require --official-frozen "
            "with a validated release manifest"
        )


def runtime_release_fingerprints(settings: Any, tool_provider_mode: str) -> dict[str, str]:
    """Return the behavior-affecting fingerprints frozen by a release manifest."""
    from travel_agent.harness.product import (
        _code_fingerprint,
        _configuration_fingerprint,
        _evaluator_fingerprint,
        _evaluator_version,
        _prompt_fingerprint,
        _tool_snapshot_fingerprint,
        _tooling_fingerprint,
    )

    return {
        "code_fingerprint": _code_fingerprint(),
        "prompt_fingerprint": _prompt_fingerprint(),
        "evaluator_version": _evaluator_version(),
        "evaluator_fingerprint": _evaluator_fingerprint(),
        "tooling_fingerprint": _tooling_fingerprint(),
        "configuration_fingerprint": _configuration_fingerprint(settings, tool_provider_mode),
        "tool_snapshot_fingerprint": _tool_snapshot_fingerprint(settings, tool_provider_mode),
        "artifact_contract_implementation_fingerprint": artifact_contract_implementation_fingerprint(),
    }


def validate_runtime_against_manifest(
    manifest: Mapping[str, Any], settings: Any, tool_provider_mode: str
) -> list[str]:
    failures: list[str] = []
    actual = runtime_release_fingerprints(settings, tool_provider_mode)
    expected = manifest.get("fingerprints") or {}
    for field in FINGERPRINT_FIELDS:
        _require(actual.get(field) == expected.get(field), failures, f"runtime {field} differs from manifest")
    return failures


def validate_shadow_prerequisites(
    manifest: Mapping[str, Any],
    core_acceptance_path: Path | str | None,
    challenge_acceptance_path: Path | str | None,
) -> list[str]:
    failures: list[str] = []
    for split, path in (
        ("core_frozen", core_acceptance_path),
        ("challenge_frozen", challenge_acceptance_path),
    ):
        if path is None:
            failures.append(f"Shadow requires {split} acceptance proof")
            continue
        try:
            proof = _read_json(Path(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {split} acceptance proof: {exc}")
            continue
        _require(proof.get("passed") is True, failures, f"{split} acceptance proof is not passing")
        _require(proof.get("status") == "accepted", failures, f"{split} acceptance proof status is not accepted")
        _require(proof.get("schema_version") == FROZEN_ACCEPTANCE_VERSION, failures, f"{split} acceptance proof schema mismatch")
        _require(proof.get("split") == split, failures, f"{split} acceptance proof has wrong split")
        _require(proof.get("candidate_id") == manifest.get("candidate_id"), failures, f"{split} acceptance proof candidate mismatch")
        _require(proof.get("manifest_fingerprint") == manifest.get("manifest_fingerprint"), failures, f"{split} acceptance proof manifest mismatch")
        _require(proof.get("acceptance_fingerprint") == _acceptance_fingerprint(proof), failures, f"{split} acceptance proof fingerprint mismatch")
        run_dir = Path(str(proof.get("run_dir") or ""))
        summary_path = run_dir / "summary.json"
        _require(summary_path.is_file(), failures, f"{split} acceptance proof run summary is missing")
        if summary_path.is_file():
            _require(proof.get("run_summary_sha256") == _sha256(summary_path), failures, f"{split} acceptance proof is stale")
    return failures


def validate_release_manifest(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    _require(manifest.get("schema_version") == FROZEN_RELEASE_SCHEMA_VERSION, failures, "manifest schema_version mismatch")
    for field in ("candidate_id", "commit_sha", "branch", "tag", "created_at", "manifest_fingerprint"):
        _require(bool(manifest.get(field)), failures, f"manifest {field} is required")
    _require(len(str(manifest.get("commit_sha") or "")) == 40, failures, "manifest commit_sha must be a full 40-character revision")
    _require(manifest.get("tested_model") == TESTED_MODEL, failures, "manifest tested_model mismatch")
    provider_models = manifest.get("provider_reported_models")
    _require(
        isinstance(provider_models, list)
        and bool(provider_models)
        and all(isinstance(item, str) and item.strip() for item in provider_models),
        failures,
        "manifest provider_reported_models must be a non-empty string list",
    )
    _require(manifest.get("judge") == JUDGE_MODEL, failures, "manifest judge mismatch")
    _require(manifest.get("tool_provider_mode") == "configured", failures, "manifest tool provider must be configured")
    fingerprints = manifest.get("fingerprints") or {}
    for field in FINGERPRINT_FIELDS:
        _require(bool(fingerprints.get(field)), failures, f"manifest fingerprint {field} is required")
    splits = manifest.get("splits") or {}
    for split, contract in STAGE_CONTRACTS.items():
        item = splits.get(split) or {}
        _require(item.get("count") == contract["count"], failures, f"{split} count must be {contract['count']}")
        _require(bool(item.get("sha256")), failures, f"{split} sha256 is required")
        _require(bool(item.get("path")), failures, f"{split} path is required")
        _require(item.get("thresholds") == contract, failures, f"{split} thresholds differ from the pre-registered contract")
    recorded = manifest.get("manifest_fingerprint")
    _require(recorded == _manifest_fingerprint(manifest), failures, "manifest fingerprint does not match its contents")
    return failures


def validate_frozen_run_request(
    args: Any,
    settings: Any,
    model_specs: list[tuple[str, str]],
    manifest: Mapping[str, Any],
) -> list[str]:
    failures = validate_release_manifest(manifest)
    split = str(getattr(args, "product_split", ""))
    _require(split in STAGE_CONTRACTS, failures, "--official-frozen requires one exact frozen split")
    _require(getattr(args, "limit", None) is None, failures, "--limit is forbidden for an official frozen run")
    _require(not getattr(args, "case_id", None), failures, "--case-id is forbidden for an official frozen run")
    _require(not bool(getattr(args, "resume", False)), failures, "--resume is forbidden for an official frozen run")
    _require(int(getattr(args, "max_model_tokens", 0) or 0) == 0, failures, "local token cap is forbidden for an official frozen run")
    _require(getattr(args, "base_url", None) in (None, ""), failures, "--base-url override is forbidden for an official frozen run")
    _require(bool(getattr(args, "preflight", True)), failures, "--no-preflight is forbidden for an official frozen run")
    _require(getattr(args, "tool_provider", None) == "configured", failures, "official frozen tool provider must be configured")
    _require(model_specs == [(TESTED_MODEL["provider"], TESTED_MODEL["model"])], failures, "official frozen model must be fixed to deepseek/deepseek-v4-flash with no relay")
    llm = settings.llm
    _require(str(llm.provider).lower() == TESTED_MODEL["provider"], failures, "configured provider must be deepseek")
    _require(llm.model == TESTED_MODEL["model"], failures, "configured model must be deepseek-v4-flash")
    _require(_same_number(llm.temperature, TESTED_MODEL["temperature"]), failures, "configured model temperature must be 0.2")
    _require(llm.thinking_enabled is False, failures, "configured model thinking must be disabled")
    hybrid = settings.hybrid_planning
    hybrid_values = (
        hybrid.enable_llm_intent_normalizer,
        hybrid.enable_llm_preference_resolver,
        hybrid.enable_structured_duration_estimator,
    )
    _require(not any(bool(value) for value in hybrid_values), failures, "all Hybrid flags must be disabled")
    return failures


def evaluate_release_readiness_dev34(
    run_dir: Path | str, *, require_judge: bool = True
) -> dict[str, Any]:
    from travel_agent.evaluation.dev34_acceptance import evaluate_dev34_run

    root = Path(run_dir)
    legacy = evaluate_dev34_run(root, require_judge=require_judge)
    summary = _read_json(root / "summary.json")
    rows = list(summary.get("rows") or [])
    failures: list[str] = list(legacy.get("failures") or [])
    _require(len(rows) == 34, failures, f"Dev34 rows must be 34, got {len(rows)}")
    strict = _true_count(rows, "strict_task_success")
    full = [row for row in rows if row.get("expected_artifact_type") == "full_itinerary"]
    non = [row for row in rows if row.get("expected_artifact_type") != "full_itinerary"]
    _require(strict >= 30, failures, f"strict success {strict}/34 is below 30")
    _require(_true_count(full, "strict_task_success") >= 19, failures, f"full-itinerary strict {_true_count(full, 'strict_task_success')}/22 is below 19")
    _require(_true_count(non, "strict_task_success") >= 11, failures, f"non-itinerary strict {_true_count(non, 'strict_task_success')}/12 is below 11")
    _require(_true_count(rows, "hard_constraints_ok") >= 33, failures, "hard constraints are below 33/34")
    for field in ("grounding_ok", "authorization_ok", "tool_schema_valid", "architecture_policy_ok"):
        _require(_true_count(rows, field) == 34, failures, f"{field} must be 34/34")
    long_rows = [row for row in rows if str(row.get("case_id") or "").startswith("lh_")]
    _require(len(long_rows) == 4 and _true_count(long_rows, "strict_task_success") == 4, failures, "long-horizon strict must be 4/4")
    _require(not _row_execution_errors(rows), failures, "Dev34 contains execution health errors")
    _require(not _redline_rows(rows), failures, "Dev34 contains redline failures")
    judge = _judge_metrics(root, rows, min_eligible=19) if require_judge else None
    if judge:
        failures.extend(judge["failures"])
        _require(judge["average_score"] is not None and judge["average_score"] >= 80, failures, "Dev34 Judge average must be at least 80")
    return {
        "passed": not failures,
        "status": legacy.get("status", "rejected"),
        "run_dir": str(root),
        "counts": {
            "strict": strict,
            "full_itinerary_strict": _true_count(full, "strict_task_success"),
            "non_itinerary_strict": _true_count(non, "strict_task_success"),
            "long_horizon_strict": _true_count(long_rows, "strict_task_success"),
        },
        "judge": judge,
        "fingerprints": legacy.get("fingerprints") or {},
        "failures": list(dict.fromkeys(failures)),
    }


def evaluate_consecutive_release_dev34(
    first_run_dir: Path | str,
    second_run_dir: Path | str,
) -> dict[str, Any]:
    first = evaluate_release_readiness_dev34(first_run_dir, require_judge=True)
    second = evaluate_release_readiness_dev34(second_run_dir, require_judge=True)
    failures: list[str] = []
    _require(first["passed"], failures, "first Dev34 release-readiness run failed")
    _require(second["passed"], failures, "second Dev34 release-readiness run failed")
    _require(Path(first_run_dir).resolve() != Path(second_run_dir).resolve(), failures, "two distinct Dev34 run directories are required")
    _require(first.get("fingerprints") == second.get("fingerprints"), failures, "Dev34 run fingerprints differ")
    return {"passed": not failures, "first": first, "second": second, "failures": failures}


def evaluate_frozen_stage(
    run_dir: Path | str,
    manifest: Mapping[str, Any],
    split: str,
    *,
    require_judge: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir)
    failures = validate_release_manifest(manifest)
    if split not in STAGE_CONTRACTS:
        return {"passed": False, "split": split, "run_dir": str(root), "failures": [*failures, f"unknown frozen split: {split}"]}
    contract = STAGE_CONTRACTS[split]
    summary = _read_json(root / "summary.json")
    rows = list(summary.get("rows") or [])
    artifacts = dict(summary.get("artifacts") or {})
    metrics = dict(summary.get("metrics") or {})
    cases = [_read_json(path) for path in sorted((root / "cases").glob("*.json"))]
    row_ids = [str(row.get("case_id") or "") for row in rows]
    case_ids = [str((case.get("case") or {}).get("case_id") or "") for case in cases]
    expected_count = int(contract["count"])
    _require(len(rows) == expected_count, failures, f"{split} summary rows must be {expected_count}, got {len(rows)}")
    _require(len(cases) == expected_count, failures, f"{split} case outputs must be {expected_count}, got {len(cases)}")
    _require(len(set(row_ids)) == expected_count and "" not in row_ids, failures, f"{split} row ids must be unique")
    _require(set(row_ids) == set(case_ids), failures, f"{split} summary and case ids disagree")
    _require(all(row.get("split") == split for row in rows), failures, f"every row must belong to {split}")
    _check_case_row_consistency(rows, cases, failures)
    attempts = metrics.get("model_attempts_by_case") or {}
    once = (
        all(int(row.get("repeat") or 0) == 1 and int(row.get("execution_total") or 0) == 1 for row in rows)
        and all(int((case.get("execution") or {}).get("repeat") or 0) == 1 for case in cases)
        and set(attempts) == set(row_ids)
        and all(int(value or 0) == 1 for value in attempts.values())
        and int((artifacts.get("relay_summary") or {}).get("attempts_total") or 0) == expected_count
    )
    _require(once, failures, f"every {split} case must execute exactly once")

    strict = _true_count(rows, "strict_task_success")
    hard = _true_count(rows, "hard_constraints_ok")
    _require(strict >= contract["strict"], failures, f"{split} strict success {strict}/{expected_count} is below {contract['strict']}")
    _require(hard >= contract["hard"], failures, f"{split} hard constraints {hard}/{expected_count} is below {contract['hard']}")
    for field in ("grounding_ok", "authorization_ok", "tool_schema_valid", "architecture_policy_ok"):
        _require(_true_count(rows, field) == expected_count, failures, f"{split} {field} must be {expected_count}/{expected_count}")
    for subset, floor in contract["subset_floors"].items():
        subset_rows = [row for row in rows if row.get("subset") == subset]
        _require(_true_count(subset_rows, "strict_task_success") >= floor, failures, f"{split} subset {subset} strict is below {floor}")
    subset_counts = {
        subset: sum(row.get("subset") == subset for row in rows)
        for subset in contract["subset_counts"]
    }
    _require(subset_counts == contract["subset_counts"], failures, f"{split} subset distribution mismatch: {subset_counts}")
    long_floor = contract.get("long_horizon")
    if long_floor is not None:
        long_rows = [row for row in rows if row.get("subset") == "long_horizon_state"]
        _require(_true_count(long_rows, "strict_task_success") >= long_floor, failures, f"{split} long-horizon strict is below {long_floor}/4")

    artifact_counts: dict[str, int] = {}
    for case in cases:
        expected = expected_artifact_type(case.get("case") or {})
        artifact_counts[expected] = artifact_counts.get(expected, 0) + 1
    _require(artifact_counts == contract["expected_artifacts"], failures, f"{split} expected artifact distribution mismatch: {artifact_counts}")
    if split == "shadow_frozen":
        full_rows = [row for row in rows if row.get("expected_artifact_type") == "full_itinerary"]
        partial_rows = [row for row in rows if row.get("expected_artifact_type") == "partial_itinerary"]
        _require(_true_count(full_rows, "strict_task_success") >= 21, failures, f"shadow_frozen full-itinerary strict {_true_count(full_rows, 'strict_task_success')}/26 is below 21")
        _require(_true_count(partial_rows, "strict_task_success") >= 3, failures, f"shadow_frozen partial-itinerary strict {_true_count(partial_rows, 'strict_task_success')}/4 is below 3")
    row_execution_errors = _row_execution_errors(rows)
    case_execution_errors = _case_execution_errors(cases)
    _require(not row_execution_errors, failures, f"{split} contains execution health errors")
    _require(not case_execution_errors, failures, f"{split} contains case/tool environment errors")
    _require(not _redline_rows(rows), failures, f"{split} contains redline failures")
    _check_identity(artifacts, manifest, split, failures, require_judge=require_judge)

    _check_summary_consistency(metrics, rows, failures)
    judge = _judge_metrics(
        root,
        rows,
        min_eligible=contract["judge_eligible"],
        summary_metrics=metrics,
        summary_artifacts=artifacts,
    ) if require_judge else None
    if judge:
        failures.extend(judge["failures"])
    result = {
        "schema_version": FROZEN_ACCEPTANCE_VERSION,
        "passed": not failures,
        "status": "invalid" if row_execution_errors or case_execution_errors else ("accepted" if not failures else "rejected"),
        "split": split,
        "run_dir": str(root),
        "candidate_id": manifest.get("candidate_id"),
        "manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "counts": {"cases": len(cases), "strict": strict, "hard": hard, "expected_artifacts": artifact_counts},
        "judge": judge,
        "fingerprints": {field: artifacts.get(field) for field in FINGERPRINT_FIELDS},
        "failures": failures,
    }
    result["run_summary_sha256"] = _sha256(root / "summary.json")
    result["acceptance_fingerprint"] = _acceptance_fingerprint(result)
    return result


def evaluate_frozen_release(
    manifest: Mapping[str, Any],
    core_run_dir: Path | str,
    challenge_run_dir: Path | str,
    shadow_run_dir: Path | str,
) -> dict[str, Any]:
    stages = {
        "core_frozen": evaluate_frozen_stage(core_run_dir, manifest, "core_frozen", require_judge=True),
        "challenge_frozen": evaluate_frozen_stage(challenge_run_dir, manifest, "challenge_frozen", require_judge=True),
        "shadow_frozen": evaluate_frozen_stage(shadow_run_dir, manifest, "shadow_frozen", require_judge=True),
    }
    failures = [f"{name}: stage acceptance failed" for name, result in stages.items() if not result["passed"]]
    fingerprints = [result.get("fingerprints") for result in stages.values()]
    _require(all(item == fingerprints[0] for item in fingerprints[1:]), failures, "frozen stage fingerprints differ")
    return {
        "schema_version": FROZEN_ACCEPTANCE_VERSION,
        "passed": not failures,
        "candidate_id": manifest.get("candidate_id"),
        "manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "stages": stages,
        "failures": failures,
    }


def _check_identity(
    artifacts: Mapping[str, Any],
    manifest: Mapping[str, Any],
    split: str,
    failures: list[str],
    *,
    require_judge: bool,
) -> None:
    _require(artifacts.get("release_manifest_schema_version") == FROZEN_RELEASE_SCHEMA_VERSION, failures, "run release manifest schema mismatch")
    _require(artifacts.get("release_candidate_id") == manifest.get("candidate_id"), failures, "run release candidate mismatch")
    _require(artifacts.get("release_manifest_fingerprint") == manifest.get("manifest_fingerprint"), failures, "run manifest fingerprint mismatch")
    _require(artifacts.get("frozen_split") == split, failures, "run frozen split metadata mismatch")
    _require(artifacts.get("frozen_split_sha256") == (manifest.get("splits") or {}).get(split, {}).get("sha256"), failures, "run frozen split hash mismatch")
    _require(artifacts.get("code_revision") == manifest.get("commit_sha"), failures, "run commit differs from candidate")
    for field in FINGERPRINT_FIELDS:
        _require(artifacts.get(field) == (manifest.get("fingerprints") or {}).get(field), failures, f"run {field} differs from manifest")
    _require(artifacts.get("model_provider") == TESTED_MODEL["provider"] and artifacts.get("model") == TESTED_MODEL["model"], failures, "run tested model identity mismatch")
    _require(
        sorted(set(artifacts.get("provider_reported_models") or []))
        == manifest.get("provider_reported_models"),
        failures,
        "run provider-reported model identity differs from manifest",
    )
    preflights = artifacts.get("model_preflight") or []
    _require(
        isinstance(preflights, list)
        and len(preflights) == 1
        and preflights[0].get("ok") is True
        and preflights[0].get("provider") == TESTED_MODEL["provider"]
        and preflights[0].get("model") == TESTED_MODEL["model"]
        and preflights[0].get("provider_reported_model")
        in manifest.get("provider_reported_models", [])
        and _same_number(preflights[0].get("temperature"), TESTED_MODEL["temperature"])
        and preflights[0].get("thinking_enabled") is TESTED_MODEL["thinking_enabled"],
        failures,
        "run DeepSeek preflight identity/config differs from manifest",
    )
    _require(_same_number(artifacts.get("model_temperature"), 0.2) and artifacts.get("model_thinking_enabled") is False, failures, "run tested model settings mismatch")
    _require(artifacts.get("tool_provider_mode") == "configured", failures, "run tool provider must be configured")
    tool_preflight = artifacts.get("tool_preflight") or {}
    tool_checks = tool_preflight.get("checks") or {}
    _require(
        tool_preflight.get("ok") is True
        and tool_preflight.get("skipped") is not True
        and (tool_checks.get("weather") or {}).get("ok") is True
        and (tool_checks.get("place_search") or {}).get("ok") is True,
        failures,
        "run configured AMap preflight is missing, skipped, or failed",
    )
    _require(not any(bool(value) for value in (artifacts.get("hybrid_flags") or {}).values()), failures, "run Hybrid flags must all be disabled")
    _require(artifacts.get("model_execution_mode") == "fixed_single_model", failures, "run must use one fixed model")
    if require_judge:
        _require(artifacts.get("judge_provider") == JUDGE_MODEL["provider"] and artifacts.get("judge_model") == JUDGE_MODEL["model"], failures, "run Judge identity mismatch")
        preflight = artifacts.get("judge_preflight") or {}
        _require(preflight.get("ok") is True and preflight.get("injected") is not True, failures, "real Judge preflight is required")


def _judge_metrics(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    min_eligible: int,
    summary_metrics: Mapping[str, Any] | None = None,
    summary_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = [_read_json(path) for path in sorted((root / "cases").glob("*.json"))]
    eligible: list[dict[str, Any]] = []
    for case in cases:
        expected = expected_artifact_type(case.get("case") or {})
        evaluation = case.get("evaluation") or {}
        actual = (evaluation.get("rule_metrics") or {}).get("actual_artifact_type") or evaluation.get("actual_artifact_type") or actual_artifact_type(case, expected)
        if expected in ITINERARY_TYPES and actual in ITINERARY_TYPES:
            result = dict(evaluation.get("independent_judge") or {})
            route = itinerary_judge_route(expected, actual)
            result["_expected_rubric"] = route.get("rubric")
            result["_expected_diagnostic_only"] = bool(
                route.get("diagnostic_only")
            )
            result["_expected_rubric_version"] = (
                PARTIAL_RUBRIC_VERSION
                if route.get("rubric") == "partial_itinerary"
                else JUDGE_MODEL["rubric_version"]
            )
            eligible.append(result)
    completed = [item for item in eligible if item.get("status") == "ok"]
    scores = [float(item["total_score"]) for item in completed if isinstance(item.get("total_score"), (int, float))]
    average = sum(scores) / len(scores) if scores else None
    reasonable = sum(item.get("reasonable") is True for item in completed) / len(completed) if completed else None
    critical = sum(any(issue.get("severity") == "critical" for issue in item.get("critical_issues") or []) for item in completed) / len(completed) if completed else None
    failures: list[str] = []
    _require(len(eligible) >= min_eligible, failures, f"Judge eligible {len(eligible)} is below {min_eligible}")
    _require(len(completed) == len(eligible), failures, f"Judge completion {len(completed)}/{len(eligible)} is not 100%")
    _require(average is not None and average >= 80, failures, f"Judge average {average} is below 80")
    _require(reasonable is not None and reasonable >= 0.75, failures, f"Judge reasonable rate {reasonable} is below 0.75")
    _require(critical is not None and critical <= 0.15, failures, f"Judge critical issue rate {critical} exceeds 0.15")
    identity_ok = all(
        item.get("provider") == JUDGE_MODEL["provider"]
        and item.get("model") == JUDGE_MODEL["model"]
        and item.get("rubric") == item.get("_expected_rubric")
        and item.get("diagnostic_only")
        is item.get("_expected_diagnostic_only")
        and item.get("rubric_version") == item.get("_expected_rubric_version")
        and item.get("prompt_version") == JUDGE_MODEL["prompt_version"]
        and item.get("schema_version") == JUDGE_MODEL["schema_version"]
        and item.get("independence_warning") is False
        for item in completed
    )
    _require(identity_ok, failures, "Judge result identity mismatch")
    if summary_artifacts is not None:
        _require(summary_artifacts.get("judge_rubric_version") == JUDGE_MODEL["rubric_version"], failures, "Judge rubric version mismatch")
        _require(summary_artifacts.get("judge_prompt_version") == JUDGE_MODEL["prompt_version"], failures, "Judge prompt version mismatch")
        _require(summary_artifacts.get("judge_schema_version") == JUDGE_MODEL["schema_version"], failures, "Judge schema version mismatch")
    if summary_metrics is not None:
        reported = summary_metrics.get("plan_quality_judge") or {}
        _require(int(reported.get("applicable_count") or 0) == len(eligible), failures, "Judge eligible count disagrees with summary")
        _require(int(reported.get("completed_count") or 0) == len(completed), failures, "Judge completed count disagrees with summary")
        _require(_same_optional_number(reported.get("completion_rate"), 1.0 if eligible else None), failures, "Judge completion rate disagrees with summary")
        _require(_same_optional_number(reported.get("judge_average_score"), average), failures, "Judge average disagrees with summary")
        _require(_same_optional_number(reported.get("judge_reasonable_rate"), reasonable), failures, "Judge reasonable rate disagrees with summary")
        _require(_same_optional_number(reported.get("critical_issue_rate"), critical), failures, "Judge critical issue rate disagrees with summary")
        _require(reported.get("provider") == JUDGE_MODEL["provider"] and reported.get("model") == JUDGE_MODEL["model"], failures, "Judge summary identity mismatch")
    return {"eligible": len(eligible), "completed": len(completed), "average_score": average, "reasonable_rate": reasonable, "critical_issue_rate": critical, "failures": failures}


def _check_case_row_consistency(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    failures: list[str],
) -> None:
    case_by_id = {
        str((case.get("case") or {}).get("case_id") or ""): case
        for case in cases
    }
    rule_fields = {
        "strict_task_success": "strict_task_success",
        "hard_constraints_ok": "constraint_pass",
        "grounding_ok": "grounding_ok",
        "authorization_ok": "authorization_ok",
        "tool_schema_valid": "tool_schema_valid",
        "architecture_policy_ok": "architecture_policy_ok",
        "gating_passed": "gating_passed",
    }
    for row in rows:
        case_id = str(row.get("case_id") or "")
        case = case_by_id.get(case_id)
        if case is None:
            continue
        meta = case.get("case") or {}
        evaluation = case.get("evaluation") or {}
        rule = evaluation.get("rule_metrics") or {}
        expected = expected_artifact_type(meta)
        actual = (
            rule.get("actual_artifact_type")
            or evaluation.get("actual_artifact_type")
            or actual_artifact_type(case, expected)
        )
        disagreements = []
        if row.get("split") != meta.get("split"):
            disagreements.append("split")
        if row.get("subset") != meta.get("subset"):
            disagreements.append("subset")
        if row.get("expected_artifact_type") != expected:
            disagreements.append("expected_artifact_type")
        if row.get("actual_artifact_type") != actual:
            disagreements.append("actual_artifact_type")
        for row_field, rule_field in rule_fields.items():
            if rule_field not in rule or row.get(row_field) is not rule.get(rule_field):
                disagreements.append(row_field)
        execution = case.get("execution") or {}
        identity = (
            execution.get("requested_model") == TESTED_MODEL["model"]
            and execution.get("runtime_model") == TESTED_MODEL["model"]
            and execution.get("runtime_model_provider") == TESTED_MODEL["provider"]
            and int(execution.get("runtime_model_index") or 0) == 0
            and int(execution.get("runtime_model_switches") or 0) == 0
        )
        if not identity:
            disagreements.append("model_identity")
        if disagreements:
            failures.append(
                f"summary/case disagreement for {case_id}: "
                + ", ".join(disagreements)
            )


def _check_summary_consistency(
    metrics: Mapping[str, Any],
    rows: list[dict[str, Any]],
    failures: list[str],
) -> None:
    total = len(rows)
    _require(int(metrics.get("sample_size") or 0) == total, failures, "summary sample_size disagrees with rows")
    _require(int(metrics.get("execution_count") or 0) == total, failures, "summary execution_count disagrees with rows")
    rate_fields = {
        "overall_strict_success_rate": "strict_task_success",
        "strict_success_rate": "strict_task_success",
        "hard_constraint_rate": "hard_constraints_ok",
        "grounding_pass_rate": "grounding_ok",
        "authorization_pass_rate": "authorization_ok",
        "tool_schema_valid_rate": "tool_schema_valid",
        "architecture_policy_pass_rate": "architecture_policy_ok",
    }
    for metric_field, row_field in rate_fields.items():
        expected = _true_count(rows, row_field) / total if total else None
        _require(
            _same_optional_number(metrics.get(metric_field), expected),
            failures,
            f"summary {metric_field} disagrees with rows",
        )


def _row_execution_errors(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("case_id") or "unknown") for row in rows if row.get("system_error") not in (0, False, None) or int(row.get("model_error_call_count") or 0) > 0]


def _case_execution_errors(cases: list[dict[str, Any]]) -> list[str]:
    failures = []
    benign = {"NOT_FOUND", "EMPTY_RESULT", "NO_RESULTS"}
    for output in cases:
        case_id = str((output.get("case") or {}).get("case_id") or "unknown")
        if (output.get("execution") or {}).get("errors"):
            failures.append(case_id)
            continue
        for turn in output.get("turns") or []:
            for call in turn.get("tool_calls") or []:
                if call.get("status") == "error" and str(call.get("error_code") or "").upper() not in benign:
                    failures.append(case_id)
                    break
    return sorted(set(failures))


def _redline_rows(rows: list[dict[str, Any]]) -> list[str]:
    redline_markers = {"unauthorized_transaction", "fabricated_critical_fact", "missing_critical_hard_constraint", "infeasible_plan_labeled_feasible"}
    failed = []
    for row in rows:
        triggered = {str(item) for item in row.get("gating_triggered") or []}
        if row.get("gating_passed") is False or triggered & redline_markers:
            failed.append(str(row.get("case_id") or "unknown"))
    return failed


def _true_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(row.get(field) is True for row in rows)


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _acceptance_fingerprint(acceptance: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in acceptance.items()
        if key != "acceptance_fingerprint"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty_line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _same_number(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _same_optional_number(value: Any, expected: float | None) -> bool:
    if value is None or expected is None:
        return value is None and expected is None
    return _same_number(value, expected)


def _require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)
