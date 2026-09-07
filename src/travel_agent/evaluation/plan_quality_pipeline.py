from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent.evaluation.plan_quality import (
    aggregate_rule_quality,
    evaluate_plan_quality,
)
from travel_agent.evaluation.plan_quality_human import (
    calibrate_judge,
    load_human_reviews,
    select_human_review_sample,
    write_review_template,
)
from travel_agent.evaluation.plan_quality_judge import (
    PROMPT_VERSION,
    RUBRIC_VERSION,
    SCHEMA_VERSION,
    PlanQualityJudge,
    aggregate_judge_results,
    preflight_judge,
)
from travel_agent.settings import JudgeSettings
from travel_agent.evaluation.artifact_contract import aggregate_artifact_metrics


QUALITY_REPORT_NAME = "quality_report.md"


def apply_quality_rules(run_dir: Path | str) -> dict[str, Any]:
    root = Path(run_dir)
    cases = load_run_cases(root)
    rule_results: list[dict[str, Any]] = []
    for path, case_output in cases:
        result = evaluate_plan_quality(
            case_output.get("case") or {},
            case_output.get("final_profile") or {},
            case_output.get("final_artifacts") or {},
        )
        evaluation = case_output.setdefault("evaluation", {})
        evaluation.setdefault("rule_metrics", {})
        evaluation["rule_quality"] = result.to_dict() if result else {
            "status": "not_applicable",
            "reason": "no itinerary was generated",
        }
        if result:
            rule_results.append(result.to_dict())
        _write_json_atomic(path, case_output)
    summary = _load_summary(root)
    rule_summary = aggregate_rule_quality(rule_results)
    summary.setdefault("metrics", {}).update(rule_summary)
    summary["metrics"]["plan_quality_rules"] = rule_summary
    representative_lock = select_human_review_sample(
        [
            item
            for item in cases
            if int((item[1].get("execution") or {}).get("repeat") or 1) == 1
        ],
        size=20,
    )
    summary.setdefault("artifacts", {}).update(
        {
            "plan_quality_rule_version": "plan-quality-rules-v1",
            "human_review_representative_lock": {
                "selection_method": "category_stratified_sha256_v1",
                "case_ids": [
                    row["case_id"]
                    for row in representative_lock
                    if row["sample_group"] == "representative"
                ],
            },
        }
    )
    _save_summary(root, summary)
    _write_report(root, summary)
    return rule_summary


def apply_independent_judge(
    run_dir: Path | str,
    settings: JudgeSettings,
    *,
    resume: bool = False,
    judge: PlanQualityJudge | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    cases = load_run_cases(root)
    if any(not (case.get("evaluation") or {}).get("rule_quality") for _, case in cases):
        apply_quality_rules(root)
        cases = load_run_cases(root)
    summary = _load_summary(root)
    tested_model = str((summary.get("artifacts") or {}).get("model") or "")
    if judge is None:
        preflight = preflight_judge(settings)
        if not preflight.get("ok"):
            raise RuntimeError(
                "independent Judge preflight failed: "
                f"provider={preflight.get('provider')} model={preflight.get('model')} "
                f"detail={preflight.get('detail')}"
            )
    evaluator = judge or PlanQualityJudge(settings, root / "judge_cache")
    results = []
    for path, case_output in cases:
        evaluation = case_output.setdefault("evaluation", {})
        existing = evaluation.get("independent_judge") or {}
        if resume and existing.get("status") == "ok":
            result = existing
        else:
            result = evaluator.evaluate(
                case_output,
                tested_model=tested_model,
                resume=resume,
            )
            evaluation["independent_judge"] = result
            _write_json_atomic(path, case_output)
        results.append(result)
    judge_summary = aggregate_judge_results(results)
    artifact_summary = aggregate_artifact_metrics([case for _, case in cases])
    independence_warning = bool(tested_model and tested_model == settings.model)
    judge_summary.update(
        {
            "provider": settings.provider,
            "model": settings.model,
            "independence_warning": independence_warning,
        }
    )
    summary.setdefault("metrics", {}).update(
        {
            "judge_average_score": judge_summary["judge_average_score"],
            "judge_reasonable_rate": judge_summary["judge_reasonable_rate"],
            "judge_completion_rate": judge_summary["completion_rate"],
            "plan_quality_judge": judge_summary,
            "artifact_evaluation": artifact_summary,
        }
    )
    summary.setdefault("artifacts", {}).update(
        {
            "judge_provider": settings.provider,
            "judge_model": settings.model,
            "judge_rubric_version": RUBRIC_VERSION,
            "judge_prompt_version": PROMPT_VERSION,
            "judge_schema_version": SCHEMA_VERSION,
            "judge_cache_dir": str(root / "judge_cache"),
            "judge_resume": resume,
            "judge_preflight": (
                preflight if judge is None else {"ok": True, "injected": True}
            ),
        }
    )
    _save_summary(root, summary)
    _write_report(root, summary)
    return judge_summary


def export_human_review_sample(
    run_dir: Path | str,
    *,
    size: int = 30,
    output: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    primary_cases = [
        item
        for item in load_run_cases(root)
        if int((item[1].get("execution") or {}).get("repeat") or 1) == 1
    ]
    rows = select_human_review_sample(primary_cases, size=size)
    output_path = Path(output) if output else root / "human_review.csv"
    write_review_template(rows, output_path)
    manifest = {
        "requested_unique_cases": size,
        "unique_case_count": len({row["case_id"] for row in rows}),
        "csv_row_count": len(rows),
        "representative_count": sum(row["sample_group"] == "representative" for row in rows),
        "diagnostic_count": sum(row["sample_group"] == "diagnostic" for row in rows),
        "blind_rescore_count": sum(row["sample_group"] == "blind_rescore" for row in rows),
        "output": str(output_path),
    }
    summary = _load_summary(root)
    summary.setdefault("artifacts", {})["human_review_sample"] = manifest
    _save_summary(root, summary)
    _write_report(root, summary)
    return manifest


def apply_human_calibration(
    run_dir: Path | str, reviews_path: Path | str
) -> dict[str, Any]:
    root = Path(run_dir)
    cases = load_run_cases(root)
    primary_by_id = {
        _case_id(case): case
        for _, case in cases
        if int((case.get("execution") or {}).get("repeat") or 1) == 1
    }
    reviews = load_human_reviews(reviews_path)
    calibration = calibrate_judge(reviews, primary_by_id)
    reviews_by_id: dict[str, list[dict[str, Any]]] = {}
    for review in reviews:
        reviews_by_id.setdefault(review["case_id"], []).append(review)
    for path, case_output in cases:
        evaluation = case_output.setdefault("evaluation", {})
        evaluation["human_review"] = reviews_by_id.get(_case_id(case_output), [])
        judge = evaluation.get("independent_judge") or {}
        rule = evaluation.get("rule_quality") or {}
        if judge.get("status") == "ok":
            judge["experimental"] = not calibration["trusted"]
        evaluation["quality_pass"] = (
            bool(rule.get("hard_feasibility_pass"))
            and bool(judge.get("reasonable"))
            if calibration["trusted"] and judge.get("status") == "ok"
            else None
        )
        _write_json_atomic(path, case_output)

    summary = _load_summary(root)
    summary.setdefault("metrics", {})["human_calibration"] = calibration
    summary["metrics"]["judge_status"] = calibration["status"]
    if isinstance(summary["metrics"].get("plan_quality_judge"), dict):
        summary["metrics"]["plan_quality_judge"]["experimental"] = not calibration["trusted"]
    if calibration["trusted"]:
        quality_by_key = {
            (_case_id(case), int((case.get("execution") or {}).get("repeat") or 1)): (
                case.get("evaluation") or {}
            ).get("quality_pass")
            for _, case in load_run_cases(root)
        }
        expected_primary = [
            case
            for _, case in load_run_cases(root)
            if int((case.get("execution") or {}).get("repeat") or 1) == 1
            and (case.get("case") or {}).get("expect_itinerary") is True
        ]
        passed = sum(
            bool((case.get("evaluation") or {}).get("quality_pass"))
            for case in expected_primary
        )
        adjusted = passed / len(expected_primary) if expected_primary else None
        summary["metrics"]["quality_adjusted_task_success_rate"] = adjusted
        for row in summary.get("rows") or []:
            row["quality_pass"] = quality_by_key.get(
                (str(row.get("case_id")), int(row.get("repeat") or 1))
            )
    else:
        summary["metrics"]["quality_adjusted_task_success_rate"] = None
    summary.setdefault("artifacts", {})["human_reviews_file"] = str(reviews_path)
    _save_summary(root, summary)
    _write_report(root, summary)
    return calibration


def load_run_cases(run_dir: Path | str) -> list[tuple[Path, dict[str, Any]]]:
    root = Path(run_dir)
    cases_dir = root / "cases"
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Product run cases directory not found: {cases_dir}")
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"case output must be a JSON object: {path}")
        payload.setdefault("evaluation", {})
        cases.append((path, payload))
    return cases


def build_quality_report(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics") or {}
    rules = metrics.get("plan_quality_rules") or {}
    judge = metrics.get("plan_quality_judge") or {}
    calibration = metrics.get("human_calibration") or {}
    artifacts = summary.get("artifacts") or {}
    lines = [
        "# Product Plan Quality Report",
        "",
        "## Run",
        "",
        f"- tested model: `{artifacts.get('model')}`",
        f"- run id: `{artifacts.get('run_id')}`",
        f"- evaluated itineraries: {rules.get('evaluated_itinerary_count')}",
        "",
        "## Deterministic Rules",
        "",
        f"- grounded POI rate: {rules.get('grounded_poi_rate')}",
        f"- schedule conflict-free rate: {rules.get('schedule_conflict_free_rate')}",
        f"- transfer feasible rate: {rules.get('transfer_feasible_rate')}",
        f"- route coverage rate: {rules.get('route_coverage_rate')}",
        f"- opening-hours coverage: {rules.get('opening_hours_coverage')}",
        f"- rule quality pass rate: {rules.get('rule_quality_pass_rate')}",
        f"- unknown checks: {rules.get('unknown_check_counts') or {}}",
        "",
        "## Independent Judge",
        "",
        f"- judge: `{judge.get('provider')}` / `{judge.get('model')}`",
        f"- completion rate: {judge.get('completion_rate')}",
        f"- full-itinerary average score: {judge.get('judge_average_score')}",
        f"- average by rubric (not pooled): {judge.get('judge_average_by_rubric') or {}}",
        f"- not applicable: {judge.get('not_applicable_count')}",
        f"- not run: {judge.get('not_run_count')}",
        f"- missing/error: {judge.get('missing_count')}/{judge.get('error_count')}",
        f"- reasonable rate: {judge.get('judge_reasonable_rate')}",
        f"- experimental: {judge.get('experimental', True)}",
        f"- independence warning: {judge.get('independence_warning')}",
        "",
        "## Human Calibration",
        "",
        f"- status: `{calibration.get('status', 'not_run')}`",
        f"- agreement: {calibration.get('accuracy')}",
        f"- Cohen kappa: {calibration.get('cohen_kappa')}",
        f"- score MAE: {calibration.get('score_mae')}",
        f"- quality-adjusted task success: {metrics.get('quality_adjusted_task_success_rate')}",
    ]
    return "\n".join(lines) + "\n"


def _load_summary(root: Path) -> dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Product run summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Product run summary must be a JSON object")
    return payload


def _save_summary(root: Path, summary: dict[str, Any]) -> None:
    _write_json_atomic(root / "summary.json", summary)
    latest = root.parent.parent / "summary.json" if root.parent.name == "runs" else None
    if latest and latest.parent.exists():
        _write_json_atomic(latest, summary)


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    (root / QUALITY_REPORT_NAME).write_text(build_quality_report(summary), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _case_id(case_output: dict[str, Any]) -> str:
    case = case_output.get("case") or {}
    return str(case.get("case_id") or case.get("id") or "")
