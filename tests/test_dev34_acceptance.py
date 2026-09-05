from __future__ import annotations

import json
from pathlib import Path

from travel_agent.evaluation.dev34_acceptance import (
    evaluate_consecutive_dev34_runs,
    evaluate_dev34_run,
)


ROUTE_CASES = {"dev_006", "dev_011", "dev_027"}
LOCAL_CASES = {"dev_010"}
COMPARISON_CASES = {
    "dev_007", "dev_008", "dev_009", "dev_012",
    "dev_015", "dev_026", "dev_028", "dev_029",
}
LONG_CASES = {"lh_5_001", "lh_8_001", "lh_12_001", "lh_12_002"}
ALL_CASES = [*(f"dev_{number:03d}" for number in range(1, 31)), *sorted(LONG_CASES)]


def _expected_type(case_id: str) -> str:
    if case_id in ROUTE_CASES:
        return "route_plan"
    if case_id in LOCAL_CASES:
        return "local_adjustment_advice"
    if case_id in COMPARISON_CASES:
        return "candidate_comparison"
    return "full_itinerary"


def _write_passing_deterministic_run(root: Path, *, with_judge: bool = False) -> None:
    cases_dir = root / "cases"
    cases_dir.mkdir(parents=True)
    rows = []
    attempts = {}
    for case_id in ALL_CASES:
        expected = _expected_type(case_id)
        evaluation = {"actual_artifact_type": expected}
        if with_judge and expected == "full_itinerary":
            evaluation["independent_judge"] = {
                "status": "ok",
                "provider": "siliconflow",
                "model": "Qwen/Qwen3.5-397B-A17B",
                "rubric": "full_itinerary",
                "diagnostic_only": False,
                "rubric_version": "travel-plan-quality-v1",
                "prompt_version": "travel-plan-judge-v2",
                "schema_version": "travel-plan-judge-output-v1",
                "independence_warning": False,
                "total_score": 80,
                "reasonable": True,
                "critical_issues": [],
            }
        case = {
            "case": {"case_id": case_id, "expected_artifact_type": expected},
            "execution": {"errors": [], "repeat": 1},
            "turns": [],
            "evaluation": evaluation,
        }
        (cases_dir / f"{case_id}__repeat-1.json").write_text(
            json.dumps(case), encoding="utf-8"
        )
        rows.append(
            {
                "case_id": case_id,
                "expected_artifact_type": expected,
                "strict_task_success": True,
                "hard_constraints_ok": True,
                "grounding_ok": True,
                "authorization_ok": True,
                "tool_schema_valid": True,
                "architecture_policy_ok": True,
                "system_error": 0,
                "model_error_call_count": 0,
                "repeat": 1,
                "execution_total": 1,
            }
        )
        attempts[case_id] = 1
    metrics = {
        "sample_size": 34,
        "execution_count": 34,
        "strict_success_rate": 1.0,
        "full_itinerary_success_rate": 1.0,
        "non_itinerary_success_rate": 1.0,
        "hard_constraint_rate": 1.0,
        "grounding_pass_rate": 1.0,
        "authorization_pass_rate": 1.0,
        "tool_schema_valid_rate": 1.0,
        "architecture_policy_pass_rate": 1.0,
        "model_switch_count": 0,
        "relay_mode": False,
        "model_attempts_by_case": attempts,
    }
    artifacts = {
        "dataset_version": "travel-agent-eval-production-v1.1",
        "dataset_sha256": "ff47dfa5d429baae1a173095fef8d1626f59ae32424407e4b38bf649c71de477",
        "evaluator_version": "product-v1.1+dev34-artifact-contract-v3",
        "code_fingerprint": "code",
        "prompt_fingerprint": "prompt",
        "evaluator_fingerprint": "evaluator",
        "tooling_fingerprint": "tooling",
        "configuration_fingerprint": "configuration",
        "tool_snapshot_fingerprint": "tool-snapshot",
        "artifact_contract_fingerprint": "7478fe592648c408fad4b4600e0619c2298120bb1f82a221fbd097ba3a4b180c",
        "model_execution_mode": "fixed_single_model",
        "model_provider": "deepseek",
        "model": "deepseek-chat",
        "model_temperature": 0.2,
        "model_thinking_enabled": False,
        "tool_provider_mode": "configured",
        "hybrid_flags": {"enabled": False},
        "relay_summary": {"attempts_total": 34, "current_model_index": 0},
    }
    if with_judge:
        artifacts.update(
            {
                "judge_provider": "siliconflow",
                "judge_model": "Qwen/Qwen3.5-397B-A17B",
                "judge_rubric_version": "travel-plan-quality-v1",
                "judge_prompt_version": "travel-plan-judge-v2",
                "judge_schema_version": "travel-plan-judge-output-v1",
                "judge_preflight": {
                    "ok": True,
                    "injected": False,
                    "provider": "siliconflow",
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "temperature": 0.0,
                    "thinking_enabled": False,
                },
            }
        )
    (root / "summary.json").write_text(
        json.dumps({"metrics": metrics, "artifacts": artifacts, "rows": rows}),
        encoding="utf-8",
    )


def test_dev34_acceptance_rejects_partial_or_nonformal_run(tmp_path) -> None:
    run_dir = tmp_path / "partial"
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True)
    case = {
        "case": {"case_id": "dev_001", "expected_artifact_type": "full_itinerary"},
        "execution": {"errors": []},
        "turns": [],
        "final_artifacts": {},
        "evaluation": {},
    }
    (cases_dir / "dev_001__repeat-1.json").write_text(
        json.dumps(case), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "metrics": {"sample_size": 1, "execution_count": 1},
                "artifacts": {
                    "model_provider": "deepseek",
                    "model": "deepseek-chat",
                    "model_temperature": 0.2,
                    "model_thinking_enabled": False,
                    "tool_provider_mode": "local",
                },
                "rows": [
                    {
                        "case_id": "dev_001",
                        "expected_artifact_type": "full_itinerary",
                        "strict_task_success": False,
                        "system_error": 0,
                        "model_error_call_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_dev34_run(run_dir, require_judge=False)

    assert result["passed"] is False
    assert "case outputs must be 34, got 1" in result["failures"]
    assert "tool provider must be configured" in result["failures"]
    assert not any("Judge provider" in item for item in result["failures"])


def test_dev34_consecutive_contract_requires_two_distinct_runs(tmp_path) -> None:
    run_dir = tmp_path / "same"
    (run_dir / "cases").mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"metrics": {}, "artifacts": {}, "rows": []}), encoding="utf-8"
    )

    result = evaluate_consecutive_dev34_runs(run_dir, run_dir)

    assert result["passed"] is False
    assert "two distinct fresh run directories are required" in result["failures"]


def test_dev34_acceptance_requires_exact_case_to_artifact_mapping(tmp_path) -> None:
    run_dir = tmp_path / "mapping"
    _write_passing_deterministic_run(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row for row in summary["rows"]}
    by_id["dev_001"]["expected_artifact_type"] = "route_plan"
    by_id["dev_006"]["expected_artifact_type"] = "full_itinerary"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_dev34_run(run_dir, require_judge=False)

    assert result["passed"] is False
    assert any("exact artifact mapping" in item for item in result["failures"])


def test_dev34_acceptance_requires_each_case_exactly_once(tmp_path) -> None:
    run_dir = tmp_path / "duplicates"
    _write_passing_deterministic_run(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rows"][0]["execution_total"] = 2
    summary["metrics"]["model_attempts_by_case"]["dev_001"] = 2
    summary["artifacts"]["relay_summary"]["attempts_total"] = 35
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_dev34_run(run_dir, require_judge=False)

    assert result["passed"] is False
    assert any("exactly once" in item for item in result["failures"])


def test_dev34_acceptance_distinguishes_empty_lookup_from_tool_failure(tmp_path) -> None:
    run_dir = tmp_path / "tool-health"
    _write_passing_deterministic_run(run_dir)
    case_path = run_dir / "cases" / "dev_001__repeat-1.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["turns"] = [{
        "tool_calls": [{
            "name": "search_poi",
            "status": "error",
            "error_code": "NOT_FOUND",
        }],
    }]
    case_path.write_text(json.dumps(case), encoding="utf-8")

    empty_lookup = evaluate_dev34_run(run_dir, require_judge=False)

    assert empty_lookup["passed"] is True, empty_lookup["failures"]

    case["turns"][0]["tool_calls"][0]["error_code"] = "UPSTREAM_UNAVAILABLE"
    case_path.write_text(json.dumps(case), encoding="utf-8")

    upstream_failure = evaluate_dev34_run(run_dir, require_judge=False)

    assert upstream_failure["passed"] is False
    assert "execution health errors: dev_001:tool_call_error" in upstream_failure["failures"]


def test_dev34_acceptance_requires_frozen_contract_and_single_model_metadata(tmp_path) -> None:
    run_dir = tmp_path / "metadata"
    _write_passing_deterministic_run(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["dataset_sha256"] = "changed"
    summary["artifacts"]["evaluator_version"] = "changed"
    summary["artifacts"]["artifact_contract_fingerprint"] = "changed"
    summary["artifacts"]["model_execution_mode"] = "relay"
    summary["artifacts"]["configuration_fingerprint"] = ""
    summary["artifacts"]["tool_snapshot_fingerprint"] = ""
    summary["metrics"]["relay_mode"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_dev34_run(run_dir, require_judge=False)

    assert result["passed"] is False
    assert "Dev34 dataset fingerprint differs from the frozen contract" in result["failures"]
    assert "evaluator version differs from the frozen contract" in result["failures"]
    assert "artifact contract fingerprint differs from the frozen contract" in result["failures"]
    assert "model execution mode must be fixed_single_model" in result["failures"]
    assert "relay mode must be disabled" in result["failures"]
    assert "configuration_fingerprint must be recorded" in result["failures"]
    assert "tool_snapshot_fingerprint must be recorded" in result["failures"]


def test_dev34_acceptance_requires_independent_fixed_judge_per_complete_itinerary(tmp_path) -> None:
    run_dir = tmp_path / "judge"
    _write_passing_deterministic_run(run_dir, with_judge=True)

    passing = evaluate_dev34_run(run_dir, require_judge=True)

    assert passing["passed"] is True, passing["failures"]

    case_path = run_dir / "cases" / "dev_001__repeat-1.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["evaluation"]["independent_judge"]["provider"] = "google"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["judge_preflight"]["temperature"] = 0.2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = evaluate_dev34_run(run_dir, require_judge=True)

    assert result["passed"] is False
    assert any("Judge result identity" in item for item in result["failures"])
    assert "Judge preflight must record temperature 0" in result["failures"]
