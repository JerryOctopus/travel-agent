from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.workflow import run_mvp_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Travel Agent eval cases.")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "eval" / "cases.json"),
        help="Path to eval cases JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON summary.",
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    summary = run_eval(cases)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human_summary(summary)


def run_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_evaluate_case(case) for case in cases]
    case_count = len(rows)
    metrics = {
        "case_count": case_count,
        "city_accuracy": _rate(rows, "city_ok"),
        "days_accuracy": _rate(rows, "days_ok"),
        "itinerary_generated_rate": _rate(rows, "itinerary_generated"),
        "clarification_accuracy": _rate(rows, "clarification_ok"),
        "critic_pass_rate": _rate(rows, "critic_passed"),
        "interest_coverage_rate": _rate(rows, "interests_ok"),
    }
    return {"metrics": metrics, "cases": rows}


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    result = run_mvp_workflow(case["query"])
    expected_clarification = bool(case.get("expected_clarification", False))
    itinerary_generated = result.itinerary is not None
    critic_passed = bool(result.critic_result and result.critic_result.passed)

    row = {
        "id": case["id"],
        "query": case["query"],
        "city_ok": _optional_equal(result.profile.destination, case.get("expected_city")),
        "days_ok": _optional_equal(result.profile.days, case.get("expected_days")),
        "itinerary_generated": itinerary_generated,
        "clarification_ok": (
            result.clarification_question is not None
            if expected_clarification
            else result.clarification_question is None
        ),
        "critic_passed": critic_passed if itinerary_generated else None,
        "interests_ok": _interests_covered(result, case.get("required_interests", [])),
        "clarification_question": result.clarification_question,
        "critic_issues": [
            issue.code
            for issue in result.critic_result.issues
        ] if result.critic_result else [],
    }
    return row


def _optional_equal(actual: object, expected: object | None) -> bool | None:
    if expected is None:
        return None
    return actual == expected


def _interests_covered(result, required_interests: list[str]) -> bool | None:
    if not required_interests:
        return None
    if not result.itinerary:
        return False
    tags = {
        tag
        for day in result.itinerary.days
        for stop in day.stops
        for tag in stop.poi.tags
    }
    categories = {
        stop.poi.category
        for day in result.itinerary.days
        for stop in day.stops
    }
    for interest in required_interests:
        if interest in tags or interest in categories:
            continue
        return False
    return True


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def print_human_summary(summary: dict[str, Any]) -> None:
    print("Eval Summary")
    print("============")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value}")
    print("")
    print("Cases")
    print("-----")
    for row in summary["cases"]:
        status = "PASS" if _case_passed(row) else "FAIL"
        print(f"{status} {row['id']}")
        if row["clarification_question"]:
            print(f"  clarification: {row['clarification_question']}")
        if row["critic_issues"]:
            print(f"  critic_issues: {', '.join(row['critic_issues'])}")


def _case_passed(row: dict[str, Any]) -> bool:
    if row["clarification_question"]:
        return bool(row["clarification_ok"])
    check_keys = [
        "city_ok",
        "days_ok",
        "itinerary_generated",
        "clarification_ok",
        "critic_passed",
        "interests_ok",
    ]
    values = [row[key] for key in check_keys if row[key] is not None]
    return all(values)


if __name__ == "__main__":
    main()
