from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from travel_agent.evaluation.frozen_release_acceptance import (
    FROZEN_RELEASE_SCHEMA_VERSION,
    build_release_manifest,
    evaluate_frozen_release,
    evaluate_frozen_stage,
    evaluate_release_readiness_dev34,
    load_frozen_split_cases,
    reject_unofficial_frozen_request,
    validate_shadow_prerequisites,
    validate_frozen_run_request,
    validate_release_manifest,
)


MODEL = {"provider": "deepseek", "model": "deepseek-v4-flash", "temperature": 0.2, "thinking_enabled": False}
JUDGE = {"provider": "siliconflow", "model": "Qwen/Qwen3.5-397B-A17B", "temperature": 0.0, "thinking_enabled": False}
FINGERPRINTS = {
    "code_fingerprint": "code",
    "prompt_fingerprint": "prompt",
    "evaluator_version": "evaluator-version",
    "evaluator_fingerprint": "evaluator",
    "tooling_fingerprint": "tooling",
    "configuration_fingerprint": "configuration",
    "tool_snapshot_fingerprint": "tool-snapshot",
    "artifact_contract_implementation_fingerprint": "artifact-contract",
}


def _manifest(tmp_path: Path) -> dict:
    split_files = {}
    for split, count in (("core_frozen", 94), ("challenge_frozen", 34), ("shadow_frozen", 30)):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("{}\n" * count, encoding="utf-8")
        split_files[split] = path
    return build_release_manifest(
        candidate_id="candidate-v1",
        commit_sha="a" * 40,
        branch="agent/release",
        tag="frozen-release-candidate-v1",
        split_files=split_files,
        fingerprints=FINGERPRINTS,
        provider_reported_models=["DeepSeek-V4-Flash-0731"],
    )


def _stage_run(root: Path, split: str, manifest: dict, *, judge: bool = True) -> None:
    specs = {
        "core_frozen": (94, {"fixed_time": 10, "long_horizon_state": 4, "multi_city": 15, "multi_turn_update": 10, "single_city_multi_day": 20, "single_city_one_day": 15, "special_traveler": 10, "strict_constraints": 10}),
        "challenge_frozen": (34, {"action_boundary": 4, "complex_budget_route": 4, "constraint_conflict": 5, "location_ambiguity": 4, "long_horizon_state": 4, "sparse_supply": 4, "state_overwrite": 4, "time_space_impossible": 5}),
        "shadow_frozen": (30, {"real_world_natural_language": 30}),
    }
    total, subset_counts = specs[split]
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True)
    rows = []
    attempts = {}
    case_index = 0
    for subset, count in subset_counts.items():
        for _ in range(count):
            case_index += 1
            case_id = f"{split}_{case_index:03d}"
            outcome = "full_plan"
            if split == "challenge_frozen":
                if case_index <= 4:
                    outcome = "partial_plan_with_limitations"
                elif case_index <= 14:
                    outcome = "full_plan"
                elif case_index <= 20:
                    outcome = "clarify"
                elif case_index <= 30:
                    outcome = "negotiate_constraints"
                else:
                    outcome = "safe_decline_action"
            elif split == "shadow_frozen" and case_index > 26:
                outcome = "partial_plan_with_limitations"
            expected = {
                "partial_plan_with_limitations": "partial_itinerary",
                "clarify": "clarification",
                "negotiate_constraints": "constraint_negotiation",
                "safe_decline_action": "safe_decline",
            }.get(outcome, "full_itinerary")
            evaluation = {"rule_metrics": {
                "actual_artifact_type": expected,
                "expected_artifact_type": expected,
                "strict_task_success": True,
                "constraint_pass": True,
                "grounding_ok": True,
                "authorization_ok": True,
                "tool_schema_valid": True,
                "architecture_policy_ok": True,
                "gating_passed": True,
            }}
            if judge:
                evaluation["independent_judge"] = {
                    "status": "ok",
                    "provider": JUDGE["provider"],
                    "model": JUDGE["model"],
                    "rubric": expected,
                    "diagnostic_only": False,
                    "total_score": 85,
                    "reasonable": True,
                    "critical_issues": [],
                    "rubric_version": (
                        "travel-partial-plan-diagnostic-v1"
                        if expected == "partial_itinerary"
                        else "travel-plan-quality-v1"
                    ),
                    "prompt_version": "travel-plan-judge-v4",
                    "schema_version": "travel-plan-judge-output-v1",
                    "independence_warning": False,
                }
            case = {
                "case": {
                    "case_id": case_id,
                    "split": split,
                    "subset": subset,
                    "gold": {"expected_outcome": outcome},
                },
                "execution": {
                    "repeat": 1,
                    "errors": [],
                    "requested_model": MODEL["model"],
                    "runtime_model": MODEL["model"],
                    "runtime_model_provider": MODEL["provider"],
                    "runtime_model_index": 0,
                    "runtime_model_switches": 0,
                },
                "final_artifacts": {"itinerary": {"days": [{"day": 1}]}, "artifact_type": expected},
                "evaluation": evaluation,
            }
            (cases_dir / f"{case_id}__repeat-1.json").write_text(json.dumps(case), encoding="utf-8")
            rows.append({
                "case_id": case_id,
                "split": split,
                "subset": subset,
                "expected_artifact_type": expected,
                "actual_artifact_type": expected,
                "strict_task_success": True,
                "hard_constraints_ok": True,
                "grounding_ok": True,
                "authorization_ok": True,
                "tool_schema_valid": True,
                "architecture_policy_ok": True,
                "gating_passed": True,
                "gating_triggered": [],
                "system_error": 0,
                "model_error_call_count": 0,
                "execution_total": 1,
                "repeat": 1,
            })
            attempts[case_id] = 1
    artifacts = {
        **FINGERPRINTS,
        "release_manifest_schema_version": FROZEN_RELEASE_SCHEMA_VERSION,
        "release_candidate_id": manifest["candidate_id"],
        "release_manifest_fingerprint": manifest["manifest_fingerprint"],
        "frozen_split": split,
        "frozen_split_sha256": manifest["splits"][split]["sha256"],
        "code_revision": manifest["commit_sha"],
        "model_provider": MODEL["provider"],
        "model": MODEL["model"],
        "model_temperature": MODEL["temperature"],
        "model_thinking_enabled": MODEL["thinking_enabled"],
        "provider_reported_models": manifest["provider_reported_models"],
        "model_preflight": [
            {
                "ok": True,
                "provider": MODEL["provider"],
                "model": MODEL["model"],
                "provider_reported_model": manifest["provider_reported_models"][0],
                "base_url": "https://api.deepseek.com/v1",
                "temperature": MODEL["temperature"],
                "thinking_enabled": MODEL["thinking_enabled"],
            }
        ],
        "tool_provider_mode": "configured",
        "hybrid_flags": {"a": False, "b": False},
        "model_execution_mode": "fixed_single_model",
        "judge_provider": JUDGE["provider"],
        "judge_model": JUDGE["model"],
        "judge_preflight": {"ok": True, "injected": False, **JUDGE},
        "judge_rubric_version": "travel-plan-quality-v1",
        "judge_prompt_version": "travel-plan-judge-v4",
        "judge_schema_version": "travel-plan-judge-output-v1",
        "relay_summary": {"attempts_total": total, "current_model_index": 0},
    }
    metrics = {
        "sample_size": total,
        "execution_count": total,
        "model_attempts_by_case": attempts,
        "model_switch_count": 0,
        "relay_mode": False,
        "overall_strict_success_rate": 1.0,
        "strict_success_rate": 1.0,
        "hard_constraint_rate": 1.0,
        "grounding_pass_rate": 1.0,
        "authorization_pass_rate": 1.0,
        "tool_schema_valid_rate": 1.0,
        "architecture_policy_pass_rate": 1.0,
        "plan_quality_judge": {
            "applicable_count": (total if split != "challenge_frozen" else 14),
            "completed_count": (total if split != "challenge_frozen" else 14),
            "completion_rate": 1.0,
            "judge_average_score": 85.0,
            "judge_reasonable_rate": 1.0,
            "critical_issue_rate": 0.0,
            "provider": JUDGE["provider"],
            "model": JUDGE["model"],
        },
    }
    (root / "summary.json").write_text(json.dumps({"metrics": metrics, "artifacts": artifacts, "rows": rows}), encoding="utf-8")


def test_manifest_requires_all_contract_fields_and_exact_counts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_release_manifest(manifest) == []

    del manifest["fingerprints"]["prompt_fingerprint"]
    manifest["splits"]["core_frozen"]["count"] = 93
    failures = validate_release_manifest(manifest)

    assert any("prompt_fingerprint" in item for item in failures)
    assert any("core_frozen count" in item for item in failures)


def test_manifest_requires_provider_reported_model_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["provider_reported_models"] = []

    failures = validate_release_manifest(manifest)

    assert any("provider_reported_models" in item for item in failures)


def test_frozen_split_loader_rejects_tampering_before_parsing(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = Path(manifest["splits"]["core_frozen"]["path"])
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    try:
        load_frozen_split_cases(manifest, "core_frozen")
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered frozen split was accepted")


def test_official_request_rejects_partial_relay_resume_and_wrong_model(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    args = SimpleNamespace(
        official_frozen=True,
        product_split="core_frozen",
        limit=1,
        case_id=["x"],
        resume=True,
        max_model_tokens=10,
        base_url="https://relay.invalid/v1",
        preflight=False,
        tool_provider="local",
    )
    settings = SimpleNamespace(
        llm=SimpleNamespace(provider="qwen", model="other", temperature=0.3, thinking_enabled=True),
        hybrid_planning=SimpleNamespace(enable_llm_intent_normalizer=True, enable_llm_preference_resolver=False, enable_structured_duration_estimator=False),
    )

    failures = validate_frozen_run_request(args, settings, [("qwen", "other")], manifest)

    for marker in ("--limit", "--case-id", "--resume", "token cap", "--base-url", "--no-preflight", "configured", "deepseek", "Hybrid"):
        assert any(marker in item for item in failures)


def test_nonofficial_runner_cannot_open_any_frozen_split() -> None:
    for split in ("core_frozen", "challenge_frozen", "shadow_frozen", "all"):
        try:
            reject_unofficial_frozen_request(split)
        except RuntimeError as exc:
            assert "--official-frozen" in str(exc)
        else:
            raise AssertionError(f"nonofficial frozen split was accepted: {split}")
    reject_unofficial_frozen_request("dev")


def test_core_stage_accepts_complete_formal_run_and_rejects_duplicate_attempt(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "core-run"
    _stage_run(run, "core_frozen", manifest)

    passed = evaluate_frozen_stage(run, manifest, "core_frozen", require_judge=True)
    assert passed["passed"] is True, passed["failures"]

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["model_attempts_by_case"]["core_frozen_001"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    failed = evaluate_frozen_stage(run, manifest, "core_frozen", require_judge=True)
    assert failed["passed"] is False
    assert any("exactly once" in item for item in failed["failures"])


def test_stage_rejects_provider_reported_model_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "core-run"
    _stage_run(run, "core_frozen", manifest)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["provider_reported_models"] = ["unexpected-model"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_frozen_stage(run, manifest, "core_frozen")

    assert result["passed"] is False
    assert any("provider-reported model identity" in item for item in result["failures"])


def test_stage_rejects_summary_case_field_disagreement(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "core-run"
    _stage_run(run, "core_frozen", manifest)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rows"][0]["hard_constraints_ok"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_frozen_stage(run, manifest, "core_frozen", require_judge=True)

    assert result["passed"] is False
    assert any("summary/case disagreement" in item for item in result["failures"])


def test_shadow_enforces_full_and_partial_strict_floors(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "shadow-run"
    _stage_run(run, "shadow_frozen", manifest)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    partial = [row for row in summary["rows"] if row["expected_artifact_type"] == "partial_itinerary"]
    partial[0]["strict_task_success"] = False
    partial[1]["strict_task_success"] = False
    summary["metrics"]["overall_strict_success_rate"] = 28 / 30
    summary["metrics"]["strict_success_rate"] = 28 / 30
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for row in partial[:2]:
        case_path = run / "cases" / f"{row['case_id']}__repeat-1.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case["evaluation"]["rule_metrics"]["strict_task_success"] = False
        case_path.write_text(json.dumps(case), encoding="utf-8")

    result = evaluate_frozen_stage(run, manifest, "shadow_frozen", require_judge=True)

    assert result["counts"]["strict"] == 28
    assert any("partial-itinerary strict 2/4" in item for item in result["failures"])


def test_challenge_subset_floor_and_shadow_release_prerequisites(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    core = tmp_path / "core"
    challenge = tmp_path / "challenge"
    shadow = tmp_path / "shadow"
    _stage_run(core, "core_frozen", manifest)
    _stage_run(challenge, "challenge_frozen", manifest)
    _stage_run(shadow, "shadow_frozen", manifest)

    summary_path = challenge / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    action_rows = [row for row in summary["rows"] if row["subset"] == "action_boundary"]
    action_rows[0]["strict_task_success"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    failed = evaluate_frozen_stage(challenge, manifest, "challenge_frozen", require_judge=True)
    assert any("action_boundary" in item for item in failed["failures"])

    action_rows[0]["strict_task_success"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    release = evaluate_frozen_release(manifest, core, challenge, shadow)
    assert release["passed"] is True, release["failures"]

    shadow_summary_path = shadow / "summary.json"
    shadow_summary = json.loads(shadow_summary_path.read_text(encoding="utf-8"))
    shadow_summary["artifacts"]["release_candidate_id"] = "stale-candidate"
    shadow_summary_path.write_text(json.dumps(shadow_summary), encoding="utf-8")
    stale = evaluate_frozen_release(manifest, core, challenge, shadow)
    assert stale["passed"] is False


def test_shadow_requires_passing_same_candidate_proofs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    core = tmp_path / "core-acceptance.json"
    challenge = tmp_path / "challenge-acceptance.json"
    core.write_text(json.dumps({"passed": True, "split": "core_frozen", "candidate_id": manifest["candidate_id"], "manifest_fingerprint": manifest["manifest_fingerprint"]}), encoding="utf-8")
    challenge.write_text(json.dumps({"passed": True, "split": "challenge_frozen", "candidate_id": "old", "manifest_fingerprint": manifest["manifest_fingerprint"]}), encoding="utf-8")

    failures = validate_shadow_prerequisites(manifest, core, challenge)

    assert any("candidate mismatch" in item for item in failures)


def test_shadow_proofs_bind_the_actual_run_summary_and_reject_stale_files(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    core_run = tmp_path / "core-run"
    challenge_run = tmp_path / "challenge-run"
    _stage_run(core_run, "core_frozen", manifest)
    _stage_run(challenge_run, "challenge_frozen", manifest)
    core_proof = tmp_path / "core.json"
    challenge_proof = tmp_path / "challenge.json"
    core_proof.write_text(json.dumps(evaluate_frozen_stage(core_run, manifest, "core_frozen")), encoding="utf-8")
    challenge_proof.write_text(json.dumps(evaluate_frozen_stage(challenge_run, manifest, "challenge_frozen")), encoding="utf-8")

    assert validate_shadow_prerequisites(manifest, core_proof, challenge_proof) == []

    summary_path = core_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["latency_p50_ms"] = 123
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    failures = validate_shadow_prerequisites(manifest, core_proof, challenge_proof)
    assert any("stale" in item for item in failures)


def test_environment_failure_is_quarantined_as_invalid(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "core-run"
    _stage_run(run, "core_frozen", manifest)
    case_path = run / "cases" / "core_frozen_001__repeat-1.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["execution"]["errors"] = ["provider timeout"]
    case_path.write_text(json.dumps(case), encoding="utf-8")

    result = evaluate_frozen_stage(run, manifest, "core_frozen")

    assert result["passed"] is False
    assert result["status"] == "invalid"
    assert any("environment errors" in item for item in result["failures"])


def test_judge_completion_rubric_and_critical_rate_are_enforced(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "challenge-run"
    _stage_run(run, "challenge_frozen", manifest)
    eligible_paths = []
    for case_path in sorted((run / "cases").glob("*.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        if (case["evaluation"].get("independent_judge") or {}).get("status") == "ok":
            eligible_paths.append(case_path)
    broken = json.loads(eligible_paths[0].read_text(encoding="utf-8"))
    broken["evaluation"]["independent_judge"]["rubric"] = "wrong-rubric"
    eligible_paths[0].write_text(json.dumps(broken), encoding="utf-8")
    for case_path in eligible_paths[1:4]:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case["evaluation"]["independent_judge"]["critical_issues"] = [{"severity": "critical"}]
        case_path.write_text(json.dumps(case), encoding="utf-8")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"]["plan_quality_judge"]["critical_issue_rate"] = 3 / 14
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_frozen_stage(run, manifest, "challenge_frozen")

    assert any("Judge result identity" in item for item in result["failures"])
    assert any("critical issue rate" in item for item in result["failures"])


def test_judge_accepts_delivered_partial_with_diagnostic_partial_rubric(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    run = tmp_path / "core-run"
    _stage_run(run, "core_frozen", manifest)

    case_path = sorted((run / "cases").glob("*.json"))[0]
    case = json.loads(case_path.read_text(encoding="utf-8"))
    metrics = case["evaluation"]["rule_metrics"]
    metrics["actual_artifact_type"] = "partial_itinerary"
    metrics["strict_task_success"] = False
    case["final_artifacts"]["artifact_type"] = "partial_itinerary"
    case["evaluation"]["independent_judge"].update(
        {
            "rubric": "partial_itinerary",
            "diagnostic_only": True,
            "rubric_version": "travel-partial-plan-diagnostic-v1",
        }
    )
    case_path.write_text(json.dumps(case), encoding="utf-8")

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary["rows"][0]
    row["actual_artifact_type"] = "partial_itinerary"
    row["strict_task_success"] = False
    summary["metrics"]["strict_success_rate"] = 93 / 94
    summary["metrics"]["overall_strict_success_rate"] = 93 / 94
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_frozen_stage(run, manifest, "core_frozen")

    assert result["passed"] is True
    assert result["judge"]["eligible"] == 94


def test_release_dev34_gate_is_stricter_than_legacy_contract(tmp_path: Path) -> None:
    rows = []
    for index in range(34):
        is_full = index < 22
        rows.append({
            "case_id": f"dev_{index:03d}",
            "expected_artifact_type": "full_itinerary" if is_full else "candidate_comparison",
            "strict_task_success": index < 28,
            "hard_constraints_ok": True,
            "grounding_ok": True,
            "authorization_ok": True,
            "tool_schema_valid": True,
            "architecture_policy_ok": True,
            "system_error": 0,
            "model_error_call_count": 0,
            "repeat": 1,
            "execution_total": 1,
        })
    run = tmp_path / "dev"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"rows": rows, "metrics": {}, "artifacts": {}}), encoding="utf-8")

    result = evaluate_release_readiness_dev34(run, require_judge=False)

    assert result["passed"] is False
    assert any("strict success 28/34 is below 30" in item for item in result["failures"])
