"""Internal metrics for state, artifact, dispatch, recovery, and time budgets."""

from __future__ import annotations

from collections import Counter
from typing import Any


METRIC_NAMES = (
    "turn_state_pass_rate", "turn_state_accuracy", "duration_state_accuracy",
    "removed_constraint_reactivation_rate", "fixed_event_retention_rate",
    "stale_artifact_reuse_rate", "stale_parent_rebuild_reuse_rate",
    "final_current_artifact_missing_rate", "planner_admission_rate",
    "reviewer_rework_preservation_rate", "validation_revision_hash_match_rate",
    "planner_budget_exhaustion_rate", "duplicate_dispatch_rate",
    "artifact_reuse_rate", "empty_result_recovery_rate", "system_error_rate",
)


def extract_internal_counters(result: Any) -> dict[str, int]:
    """Derive auditable counters from persisted turn traces and meter snapshots."""
    counters = {
        "planner_required": 0,
        "planner_admitted": 0,
        "planner_budget_exhausted": 0,
        "dispatch_total": 0,
        "duplicate_dispatches": 0,
        "artifact_reuse_total": 0,
        "artifact_reused": 0,
        "stale_artifact_reused": 0,
        "rebuild_parent_reuse_total": 0,
        "stale_parent_rebuild_reused": 0,
        "final_current_artifact_required": 0,
        "final_current_artifact_missing": 0,
        "reviewer_rework_total": 0,
        "reviewer_rework_preserved": 0,
        "validation_revision_hash_required": 0,
        "validation_revision_hash_match": 0,
        "execution_total": 1,
        "system_error": int(
            bool(getattr(result, "errors", None))
            or any(bool(getattr(turn, "error", None)) for turn in getattr(result, "turns", ()))
        ),
    }
    for turn in getattr(result, "turns", ()):
        turn_request_id = str(getattr(turn, "request_id", None) or "")
        entries = [
            item
            for item in (getattr(turn, "agent_trace", None) or [])
            if isinstance(item, dict)
            and (
                not turn_request_id
                or not item.get("request_id")
                or str(item.get("request_id")) == turn_request_id
            )
        ]
        contract = next((item for item in entries if item.get("kind") == "turn_contract"), None)
        detail = (contract or {}).get("detail") or {}
        planner_required = (
            detail.get("delivery_intent") == "rebuild_now"
            if detail.get("delivery_intent")
            else bool(detail.get("planner_required"))
        )
        counters["planner_required"] += int(planner_required)
        admitted = any(
            item.get("kind") == "admission"
            and item.get("agent") == "planner"
            and item.get("status") == "admitted"
            for item in entries
        )
        counters["planner_admitted"] += int(planner_required and admitted)
        stop_reasons = {
            str((item.get("detail") or {}).get("stop_reason") or "")
            for item in entries
            if item.get("kind") == "routing_stop"
        }
        planner_exhausted = (
            getattr(turn, "failure_reason", None) == "budget_exhausted"
            or "planner_admission_denied" in stop_reasons
            or any(
                item.get("kind") == "subagent"
                and item.get("agent") == "planner"
                and item.get("status") == "budget_exhausted"
                for item in entries
            )
        )
        counters["planner_budget_exhausted"] += int(planner_required and planner_exhausted)
        counters["final_current_artifact_required"] += int(planner_required)
        turn_artifacts = getattr(turn, "artifacts", None) or {}
        counters["final_current_artifact_missing"] += int(
            planner_required and not turn_artifacts.get("itinerary")
        )
        itinerary = turn_artifacts.get("itinerary") or {}
        if isinstance(itinerary, dict) and itinerary.get("artifact_status") == "current":
            version = itinerary.get("state_version") or {}
            validation = itinerary.get("validation_result") or {}
            counters["validation_revision_hash_required"] += 1
            counters["validation_revision_hash_match"] += int(
                validation.get("passed") is True
                and version.get("constraint_revision")
                == validation.get("validated_constraint_revision")
                and version.get("constraint_hash")
                == validation.get("validated_constraint_hash")
            )
        rework_used = any(
            item.get("kind") == "orchestration"
            and int((item.get("detail") or {}).get("rework_used") or 0) > 0
            for item in entries
        )
        counters["reviewer_rework_total"] += int(rework_used)
        counters["reviewer_rework_preserved"] += int(any(
            item.get("kind") == "candidate_preservation"
            and (item.get("detail") or {}).get("same_revision_preserved") is True
            for item in entries
        ))

        totals = (getattr(turn, "turn_metrics", None) or {}).get("totals") or {}
        counters["dispatch_total"] += int(totals.get("dispatch_count") or 0)
        objectives = [
            str(task.get("objective_key"))
            for item in entries
            if item.get("kind") == "wave_execution"
            for task in ((item.get("detail") or {}).get("tasks") or [])
            if isinstance(task, dict) and task.get("objective_key")
        ]
        counters["duplicate_dispatches"] += len(objectives) - len(set(objectives))

        reuse_audit = [
            item for item in (detail.get("artifact_reuse_audit") or [])
            if isinstance(item, dict) and item.get("reason") != "superseded_cache_entry"
        ]
        counters["artifact_reuse_total"] += len(reuse_audit)
        counters["artifact_reused"] += sum(
            item.get("reason") == "artifact_reuse" for item in reuse_audit
        )
        counters["stale_artifact_reused"] += sum(
            item.get("reason") == "stale_artifact_reuse" for item in reuse_audit
        )
        stale_parent_reuses = sum(
            item.get("reason") == "stale_parent_rebuild" for item in reuse_audit
        )
        counters["rebuild_parent_reuse_total"] += stale_parent_reuses
        counters["stale_parent_rebuild_reused"] += stale_parent_reuses
    return counters


def internal_state_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate precomputed binary counters without changing Judge scoring."""
    def ratio(numerator: str, denominator: str) -> float:
        den = sum(int(row.get(denominator) or 0) for row in rows)
        return round(sum(int(row.get(numerator) or 0) for row in rows) / den, 4) if den else 0.0

    metrics = {
        "turn_state_pass_rate": ratio("turn_state_pass", "turn_state_total"),
        "turn_state_accuracy": ratio("state_fields_correct", "state_fields_total"),
        "duration_state_accuracy": ratio("duration_correct", "duration_total"),
        "removed_constraint_reactivation_rate": ratio("removed_reactivated", "removed_total"),
        "fixed_event_retention_rate": ratio("fixed_event_retained", "fixed_event_total"),
        "stale_artifact_reuse_rate": ratio("stale_artifact_reused", "artifact_reuse_total"),
        "stale_parent_rebuild_reuse_rate": ratio(
            "stale_parent_rebuild_reused", "rebuild_parent_reuse_total"
        ),
        "final_current_artifact_missing_rate": ratio(
            "final_current_artifact_missing", "final_current_artifact_required"
        ),
        "reviewer_rework_preservation_rate": ratio(
            "reviewer_rework_preserved", "reviewer_rework_total"
        ),
        "validation_revision_hash_match_rate": ratio(
            "validation_revision_hash_match", "validation_revision_hash_required"
        ),
        "artifact_reuse_rate": ratio("artifact_reused", "artifact_reuse_total"),
        "planner_admission_rate": ratio("planner_admitted", "planner_required"),
        "planner_budget_exhaustion_rate": ratio("planner_budget_exhausted", "planner_required"),
        "duplicate_dispatch_rate": ratio("duplicate_dispatches", "dispatch_total"),
        "empty_result_recovery_rate": ratio("empty_result_recovered", "empty_result_total"),
        "system_error_rate": ratio("system_error", "execution_total"),
    }
    timeouts = Counter()
    for row in rows:
        for item in row.get("stage_timeouts") or []:
            if isinstance(item, dict):
                timeouts[f"{item.get('stage', 'unknown')}:{item.get('component', 'unknown')}:{item.get('reason', 'timeout')}"] += 1
    metrics["stage_timeout_attribution"] = dict(timeouts)
    return metrics
