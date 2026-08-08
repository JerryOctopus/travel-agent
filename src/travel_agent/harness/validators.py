from __future__ import annotations

from typing import Any

from travel_agent.evaluation.plan_eval import evaluate_plan_artifact
from travel_agent.harness.cases import HarnessCase
from travel_agent.harness.production_evaluators import evaluate_production_case
from travel_agent.harness.result import HarnessCaseResult


def validate_case_result(
    case: HarnessCase,
    result: HarnessCaseResult,
    *,
    variant: str = "V3",
) -> dict[str, Any]:
    """Return basic case-level validation metrics for harness runs."""
    last = result.last_turn
    final_artifacts = result.final_artifacts
    itinerary = final_artifacts.get("itinerary")
    plan_eval = evaluate_plan_artifact(
        itinerary,
        profile=result.final_profile,
        required_interests=case.required_interests,
        supporting_artifacts=final_artifacts,
    )
    tool_trace = last.tool_trace if last else []
    all_tool_trace = [name for turn in result.turns for name in turn.tool_trace]
    all_tool_calls = [call for turn in result.turns for call in turn.tool_calls]
    expected_tools = list(dict.fromkeys(case.required_tools or case.expected_tools))
    required_hits = sum(tool in all_tool_trace for tool in expected_tools)
    forbidden_hits = sum(tool in all_tool_trace for tool in case.forbidden_tools)
    selected = set(all_tool_trace)
    allowed = set(case.allowed_tools) | set(expected_tools)
    unexpected = selected - allowed if allowed else set()
    argument_results = [
        _validate_tool_argument(assertion, all_tool_calls)
        for assertion in case.tool_argument_assertions
    ]
    memory_results = [
        _validate_memory_assertion(assertion, result)
        for assertion in case.expected_memory
    ]
    reply_text = last.reply_text if last else ""
    required_behaviors_ok = (
        None
        if not case.required_behaviors
        else all(text in reply_text for text in case.required_behaviors)
    )
    hard_annotation_results = [
        _matches_annotation(result.final_profile.get(field), expected)
        for field, expected in case.hard_constraints.items()
    ]
    soft_annotation_results = [
        _matches_annotation(result.final_profile.get(field), expected)
        for field, expected in case.soft_preferences.items()
        if field in result.final_profile
    ]
    metrics = {
        "city_ok": _opt_eq(result.final_profile.get("destination"), case.expected_city),
        "days_ok": _opt_eq(result.final_profile.get("days"), case.expected_days),
        "clarification_ok": (
            None
            if case.expect_clarification is None or last is None
            else last.clarification == case.expect_clarification
        ),
        "itinerary_produced": (
            None
            if case.expect_itinerary is None
            else (itinerary is not None) == case.expect_itinerary
        ),
        "tool_coverage_ok": (
            None
            if not expected_tools
            else required_hits == len(expected_tools)
        ),
        "forbidden_tools_ok": None if not case.forbidden_tools else forbidden_hits == 0,
        "tool_arguments_ok": None if not argument_results else all(argument_results),
        "tool_schema_valid": (
            None
            if not all_tool_calls
            else all(isinstance(call.get("arguments", {}), dict) for call in all_tool_calls)
        ),
        "required_behaviors_ok": required_behaviors_ok,
        "forbidden_behaviors_ok": (
            None
            if not case.forbidden_behaviors
            else all(text not in reply_text for text in case.forbidden_behaviors)
        ),
        "memory_ok": None if not memory_results else all(memory_results),
        "hard_constraints_annotation_ok": (
            None if not hard_annotation_results else all(hard_annotation_results)
        ),
        "soft_preferences_annotation_ok": (
            None if not soft_annotation_results else all(soft_annotation_results)
        ),
        "fault_recovery_ok": (
            None
            if not case.failure_injection
            else (
                not result.errors
                and bool(last and (last.reply_text or last.clarification))
                and required_behaviors_ok is not False
            )
        ),
        "tool_required_count": len(expected_tools),
        "tool_required_hit_count": required_hits,
        "tool_selected_count": len(selected),
        "tool_unexpected_count": len(unexpected) + forbidden_hits,
        "tool_argument_assertion_count": len(argument_results),
        "tool_argument_pass_count": sum(argument_results),
        "critic_passed": (
            itinerary.get("critic", {}).get("passed") if itinerary is not None else None
        ),
        "environment_pass": plan_eval.environment_pass if plan_eval else None,
        "constraint_pass": plan_eval.constraint_pass if plan_eval else None,
        "preference_pass": plan_eval.preference_pass if plan_eval else None,
        "final_pass": plan_eval.final_pass if plan_eval else None,
        "meal_time_valid": plan_eval.meal_time_valid if plan_eval else None,
        "route_feasible": plan_eval.route_feasible if plan_eval else None,
        "daily_load_valid": plan_eval.daily_load_valid if plan_eval else None,
        "poi_city_valid": plan_eval.poi_city_valid if plan_eval else None,
        "interest_coverage_valid": (
            plan_eval.interest_coverage_valid if plan_eval else None
        ),
        "plan_eval_issues": list(plan_eval.issues) if plan_eval else [],
    }
    production = evaluate_production_case(case, result, variant=variant)
    metrics.update(
        {
            "strict_task_success": production["strict_task_success"],
            "actual_outcome": production["actual_outcome"],
            "expected_outcome_match": production["expected_outcome_match"],
            "constraint_tree_score": production["hard_constraint_satisfaction"]["score"],
            "constraint_tree_missing": production["hard_constraint_satisfaction"][
                "missing_or_mismatched"
            ],
            "grounding_ok": production["grounding"]["passed"],
            "grounding_unsupported": production["grounding"]["unsupported_claims"],
            "feasibility_ok": production["itinerary_feasibility"]["passed"],
            "feasibility_issues": production["itinerary_feasibility"]["issues"],
            "authorization_ok": production["authorization"]["passed"],
            "architecture_policy_ok": production["architecture_policy"]["passed"],
            "architecture_policy_issues": production["architecture_policy"]["issues"],
            "gating_passed": production["gating"]["passed"],
            "gating_triggered": production["gating"]["triggered"],
            "run_status": production["status"],
        }
    )
    return metrics


def _opt_eq(actual: Any, expected: Any) -> bool | None:
    if expected is None:
        return None
    return actual == expected


def _validate_tool_argument(
    assertion: dict[str, Any],
    calls: list[dict[str, Any]],
) -> bool:
    tool = assertion.get("tool")
    argument = assertion.get("argument") or assertion.get("arg")
    candidates = [call for call in calls if call.get("name") == tool]
    if not candidates or not argument:
        return False
    expected = assertion.get("value")
    operation = assertion.get("op", "eq")
    for call in candidates:
        arguments = call.get("arguments") or {}
        actual = arguments.get(argument)
        if operation == "present" and argument in arguments:
            return True
        if operation == "eq" and actual == expected:
            return True
        if operation == "contains" and expected in (actual or []):
            return True
        if operation == "in" and actual in (expected or []):
            return True
    return False


def _validate_memory_assertion(
    assertion: dict[str, Any],
    result: HarnessCaseResult,
) -> bool:
    if not result.turns:
        return False
    snapshot = result.turns[-1].memory_snapshot
    scope = assertion.get("scope", "stable_profile")
    target: Any = result.final_profile if scope == "final_profile" else snapshot.get(scope)
    field = assertion.get("field")
    if field and isinstance(target, dict):
        target = target.get(field)
    operation = assertion.get("op", "eq")
    expected = assertion.get("value")
    if operation == "eq":
        return target == expected
    if operation == "not_eq":
        return target != expected
    if operation == "contains":
        return expected in (target or [])
    if operation == "not_contains":
        return expected not in (target or [])
    if operation == "empty":
        return not target
    if operation == "count_lte":
        return isinstance(target, list) and len(target) <= int(expected)
    return False


def _matches_annotation(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return all(item in (actual or []) for item in expected)
    return actual == expected
