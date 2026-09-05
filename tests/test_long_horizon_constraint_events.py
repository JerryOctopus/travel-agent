from __future__ import annotations

from travel_agent.constraint_events import merge_constraint_update, validate_constraint_state
from travel_agent.harness.state_metrics import internal_state_metrics


def _run(updates):
    state = {}
    for turn, (update, explicitness) in enumerate(updates, 1):
        merge_constraint_update(state, update, source_turn=turn, explicitness=explicitness)
    return state


def test_five_turn_additive_constraints_survive() -> None:
    state = _run([
        ({"duration_days": 5, "destination": "福州"}, "explicit"),
        ({"budget_max_cny": 7200}, "explicit"),
        ({"elderly": True, "mobility": "low_walking"}, "explicit"),
        ({"lodging_area": "三坊七巷周边"}, "explicit"),
        ({"max_single_walk_min": 15}, "explicit"),
    ])
    assert state["budget_max_cny"] == 7200
    assert state["elderly"] is True
    assert state["lodging_area"] == "三坊七巷周边"
    assert state["max_single_walk_min"] == 15


def test_eight_turn_budget_replace_retains_dietary_and_must_visit() -> None:
    state = _run([
        ({"duration_days": 4, "destination": "昆明"}, "explicit"),
        ({"must_visit": ["湿地公园"]}, "explicit"),
        ({"dietary": ["不吃辣"]}, "explicit"),
        ({"budget_max_cny": 8000}, "explicit"), ({}, "explicit"),
        ({"lodging_area": "翠湖周边"}, "explicit"), ({}, "explicit"),
        ({"budget_max_cny": 6400}, "explicit"),
    ])
    assert state["budget_max_cny"] == 6400
    assert state["dietary"] == ["不吃辣"]
    assert state["must_visit"] == ["湿地公园"]


def test_twelve_turn_delete_add_chitchat_and_soft_proposal() -> None:
    state = _run([
        ({"destination": "厦门", "duration_days": 6}, "explicit"),
        ({"must_visit": ["海岛公园"]}, "explicit"), ({}, "explicit"),
        ({"removed": ["海岛公园"]}, "explicit"), ({}, "explicit"),
        ({"must_visit": ["植物园"]}, "explicit"), ({}, "explicit"),
        ({"fixed_events": [{"day": 2, "start": "15:00", "end": "17:00", "location": "音乐厅"}]}, "explicit"),
        ({}, "explicit"), ({"must_visit": ["海岛公园"]}, "soft"),
        ({"budget_max_cny": 7000}, "explicit"), ({}, "explicit"),
    ])
    assert state["duration_days"] == 6
    assert state["must_visit"] == ["植物园"]
    assert state["removed"] == ["海岛公园"]
    assert state["fixed_events"][0]["day"] == 2
    assert validate_constraint_state(state) == []


def test_synthetic_acceptance_metrics_are_reportable() -> None:
    metrics = internal_state_metrics([{
        "turn_state_pass": 25, "turn_state_total": 25,
        "state_fields_correct": 42, "state_fields_total": 42,
        "duration_correct": 3, "duration_total": 3,
        "removed_reactivated": 0, "removed_total": 2,
        "fixed_event_retained": 2, "fixed_event_total": 2,
        "stale_artifact_reused": 0, "artifact_reused": 2, "artifact_reuse_total": 3,
        "planner_admitted": 2, "planner_required": 2,
        "planner_budget_exhausted": 0,
        "duplicate_dispatches": 0, "dispatch_total": 5,
        "empty_result_recovered": 2, "empty_result_total": 3,
        "system_error": 0, "execution_total": 1,
        "stage_timeouts": [],
    }])
    assert metrics == {
        "turn_state_pass_rate": 1.0,
        "turn_state_accuracy": 1.0,
        "duration_state_accuracy": 1.0,
        "removed_constraint_reactivation_rate": 0.0,
        "fixed_event_retention_rate": 1.0,
        "stale_artifact_reuse_rate": 0.0,
        "stale_parent_rebuild_reuse_rate": 0.0,
            "final_current_artifact_missing_rate": 0.0,
            "reviewer_rework_preservation_rate": 0.0,
            "validation_revision_hash_match_rate": 0.0,
        "artifact_reuse_rate": 0.6667,
        "planner_admission_rate": 1.0,
        "planner_budget_exhaustion_rate": 0.0,
        "duplicate_dispatch_rate": 0.0,
        "empty_result_recovery_rate": 0.6667,
        "system_error_rate": 0.0,
        "stage_timeout_attribution": {},
    }
