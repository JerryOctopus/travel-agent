from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from travel_agent.evaluation.plan_quality_judge import DIMENSIONS


REVIEW_FIELDS = [
    "case_id",
    "case_file",
    "sample_group",
    "reviewer_id",
    "review_round",
    *[f"{name}_score" for name in DIMENSIONS],
    "critical_issue",
    "human_reasonable",
    "notes",
]


def select_human_review_sample(
    cases: list[tuple[Path, dict[str, Any]]], *, size: int = 30
) -> list[dict[str, Any]]:
    eligible = [(path, case) for path, case in cases if case.get("final_itinerary")]
    if not eligible:
        return []
    representative_size = min(20, size, len(eligible))
    representative = _stratified_sample(eligible, representative_size)
    selected_paths = {str(path) for path, _ in representative}
    remaining = [(path, case) for path, case in eligible if str(path) not in selected_paths]
    diagnostic_size = min(max(0, size - representative_size), len(remaining))
    diagnostic = sorted(remaining, key=_diagnostic_sort_key)[:diagnostic_size]

    rows = [
        _review_row(path, case, "representative", 1)
        for path, case in representative
    ]
    rows.extend(_review_row(path, case, "diagnostic", 1) for path, case in diagnostic)
    blind = sorted(
        representative,
        key=lambda item: _stable_hash(f"blind:{_case_id(item[1])}"),
    )[: min(10, len(representative))]
    rows.extend(_review_row(path, case, "blind_rescore", 2) for path, case in blind)
    return rows


def write_review_template(rows: list[dict[str, Any]], output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_human_reviews(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        case_id = str(row.get("case_id") or "").strip()
        reviewer = str(row.get("reviewer_id") or "").strip()
        try:
            round_index = int(row.get("review_round") or 0)
        except ValueError:
            round_index = 0
        if not case_id or not reviewer or round_index not in {1, 2}:
            errors.append(f"row {index}: case_id, reviewer_id and review_round(1/2) are required")
            continue
        key = (case_id, reviewer, round_index)
        if key in seen:
            errors.append(f"row {index}: duplicate review {key}")
            continue
        seen.add(key)
        scores: dict[str, int] = {}
        for dimension, maximum in DIMENSIONS.items():
            field = f"{dimension}_score"
            try:
                score = int(row.get(field) or "")
            except ValueError:
                score = -1
            if not 0 <= score <= maximum:
                errors.append(f"row {index}: {field} must be 0..{maximum}")
            scores[dimension] = score
        critical = _parse_bool(row.get("critical_issue"))
        reasonable = _parse_bool(row.get("human_reasonable"))
        if critical is None or reasonable is None:
            errors.append(f"row {index}: critical_issue and human_reasonable must be true/false")
        normalized.append(
            {
                "case_id": case_id,
                "case_file": str(row.get("case_file") or ""),
                "sample_group": str(row.get("sample_group") or ""),
                "reviewer_id": reviewer,
                "review_round": round_index,
                "scores": scores,
                "total_score": sum(scores.values()),
                "critical_issue": critical,
                "human_reasonable": reasonable,
                "notes": str(row.get("notes") or "").strip(),
            }
        )
    if errors:
        raise ValueError("invalid human review CSV:\n" + "\n".join(errors))
    return normalized


def calibrate_judge(
    reviews: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    unknown_cases = sorted({review["case_id"] for review in reviews} - set(cases_by_id))
    if unknown_cases:
        raise ValueError(f"reviews reference unknown cases: {unknown_cases}")
    primary = [
        review
        for review in reviews
        if review["sample_group"] == "representative" and review["review_round"] == 1
    ]
    if not primary:
        raise ValueError("at least one representative round-1 review is required")
    pairs = []
    for review in primary:
        judge = ((cases_by_id[review["case_id"]].get("evaluation") or {}).get("independent_judge") or {})
        if judge.get("status") == "ok":
            pairs.append((review, judge))
    if not pairs:
        raise ValueError("no completed Judge results match representative reviews")

    tp = sum(review["human_reasonable"] and judge["reasonable"] for review, judge in pairs)
    tn = sum(not review["human_reasonable"] and not judge["reasonable"] for review, judge in pairs)
    fp = sum(not review["human_reasonable"] and judge["reasonable"] for review, judge in pairs)
    fn = sum(review["human_reasonable"] and not judge["reasonable"] for review, judge in pairs)
    total = len(pairs)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    human_scores = [float(review["total_score"]) for review, _ in pairs]
    judge_scores = [float(judge["total_score"]) for _, judge in pairs]
    mae = sum(abs(left - right) for left, right in zip(human_scores, judge_scores)) / total
    correlation = _pearson(human_scores, judge_scores)
    kappa = _cohen_kappa(tp, tn, fp, fn)
    human_critical = sum(bool(review["critical_issue"]) for review, _ in pairs)
    critical_hits = sum(
        bool(review["critical_issue"])
        and any(issue.get("severity") == "critical" for issue in judge.get("critical_issues") or [])
        for review, judge in pairs
    )
    critical_recall = critical_hits / human_critical if human_critical else None

    intra = _intra_rater_metrics(reviews)
    threshold_sweep = [
        _threshold_metrics(pairs, threshold) for threshold in range(60, 90, 5)
    ]
    checks = {
        "agreement_at_least_0_80": accuracy >= 0.80,
        "kappa_at_least_0_60": kappa is not None and kappa >= 0.60,
        "mae_at_most_10": mae <= 10,
        "critical_recall_at_least_0_90": human_critical < 5
        or (critical_recall is not None and critical_recall >= 0.90),
        "intra_agreement_at_least_0_90": intra["pair_count"] >= 10
        and intra["reasonable_agreement"] >= 0.90,
        "intra_score_delta_at_most_8": intra["pair_count"] >= 10
        and intra["mean_absolute_score_delta"] <= 8,
    }
    trusted = all(checks.values())
    return {
        "status": "trusted" if trusted else "experimental",
        "trusted": trusted,
        "representative_review_count": len(primary),
        "matched_judge_count": total,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": kappa,
        "score_mae": mae,
        "score_pearson_correlation": correlation,
        "human_critical_count": human_critical,
        "critical_issue_recall": critical_recall,
        "intra_rater": intra,
        "threshold_sweep": threshold_sweep,
        "trust_checks": checks,
        "threshold_changed": False,
        "active_threshold": 70,
    }


def _stratified_sample(
    cases: list[tuple[Path, dict[str, Any]]], size: int
) -> list[tuple[Path, dict[str, Any]]]:
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for item in cases:
        case = item[1].get("case") or {}
        groups[str(case.get("category") or "unknown")].append(item)
    for category in groups:
        groups[category].sort(key=lambda item: _stable_hash(_case_id(item[1])))
    selected: list[tuple[Path, dict[str, Any]]] = []
    categories = sorted(groups)
    while len(selected) < size and any(groups.values()):
        for category in categories:
            if groups[category] and len(selected) < size:
                selected.append(groups[category].pop(0))
    return selected


def _diagnostic_sort_key(item: tuple[Path, dict[str, Any]]) -> tuple[int, float, str]:
    case = item[1]
    evaluation = case.get("evaluation") or {}
    rule = evaluation.get("rule_quality") or {}
    judge = evaluation.get("independent_judge") or {}
    judge_reasonable = judge.get("reasonable")
    rule_pass = rule.get("hard_feasibility_pass")
    disagreement = judge_reasonable is not None and rule_pass is not None and judge_reasonable != rule_pass
    critical = any(issue.get("severity") == "critical" for issue in judge.get("critical_issues") or [])
    score = float(judge.get("total_score") or 70)
    priority = 0 if disagreement else 1 if critical else 2
    return priority, abs(score - 70), _stable_hash(_case_id(case))


def _review_row(
    path: Path, case_output: dict[str, Any], sample_group: str, review_round: int
) -> dict[str, Any]:
    return {
        "case_id": _case_id(case_output),
        "case_file": str(path),
        "sample_group": sample_group,
        "reviewer_id": "",
        "review_round": review_round,
        **{f"{name}_score": "" for name in DIMENSIONS},
        "critical_issue": "",
        "human_reasonable": "",
        "notes": "",
    }


def _intra_rater_metrics(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for review in reviews:
        grouped[(review["case_id"], review["reviewer_id"])][review["review_round"]] = review
    pairs = [rounds for rounds in grouped.values() if 1 in rounds and 2 in rounds]
    if not pairs:
        return {
            "pair_count": 0,
            "reasonable_agreement": None,
            "mean_absolute_score_delta": None,
        }
    agreement = sum(pair[1]["human_reasonable"] == pair[2]["human_reasonable"] for pair in pairs)
    delta = sum(abs(pair[1]["total_score"] - pair[2]["total_score"]) for pair in pairs)
    return {
        "pair_count": len(pairs),
        "reasonable_agreement": agreement / len(pairs),
        "mean_absolute_score_delta": delta / len(pairs),
    }


def _threshold_metrics(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], threshold: int
) -> dict[str, Any]:
    predictions = [
        bool(judge.get("hard_feasibility_pass"))
        and float(judge["total_score"]) >= threshold
        and not any(issue.get("severity") == "critical" for issue in judge.get("critical_issues") or [])
        for _, judge in pairs
    ]
    labels = [bool(review["human_reasonable"]) for review, _ in pairs]
    accuracy = sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)
    return {"threshold": threshold, "accuracy": accuracy}


def _cohen_kappa(tp: int, tn: int, fp: int, fn: int) -> float | None:
    total = tp + tn + fp + fn
    if not total:
        return None
    observed = (tp + tn) / total
    human_positive = (tp + fn) / total
    judge_positive = (tp + fp) / total
    expected = human_positive * judge_positive + (1 - human_positive) * (1 - judge_positive)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else None


def _parse_bool(value: Any) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y", "是"}:
        return True
    if normalized in {"false", "0", "no", "n", "否"}:
        return False
    return None


def _case_id(case_output: dict[str, Any]) -> str:
    case = case_output.get("case") or {}
    return str(case.get("case_id") or case.get("id") or "")


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
