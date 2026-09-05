from types import SimpleNamespace

from travel_agent.harness.state_metrics import extract_internal_counters, internal_state_metrics


def test_internal_metrics_cover_goal_acceptance_names_and_timeout_attribution() -> None:
    metrics = internal_state_metrics([{
        "turn_state_pass": 4, "turn_state_total": 5,
        "state_fields_correct": 18, "state_fields_total": 20,
        "duration_correct": 3, "duration_total": 3,
        "removed_reactivated": 0, "removed_total": 2,
        "fixed_event_retained": 2, "fixed_event_total": 2,
        "stale_artifact_reused": 0, "artifact_reuse_total": 3,
        "planner_admitted": 2, "planner_required": 2,
        "planner_budget_exhausted": 0,
        "duplicate_dispatches": 0, "dispatch_total": 5,
        "empty_result_recovered": 1, "empty_result_total": 2,
        "stage_timeouts": [{"stage": "worker", "component": "transport", "reason": "provider_timeout"}],
    }])
    assert metrics["turn_state_pass_rate"] == 0.8
    assert metrics["turn_state_accuracy"] == 0.9
    assert metrics["planner_admission_rate"] == 1.0
    assert metrics["stage_timeout_attribution"] == {"worker:transport:provider_timeout": 1}


def test_trace_counters_feed_planner_dispatch_reuse_and_system_metrics() -> None:
    admitted = SimpleNamespace(
        agent_trace=[
            {
                "kind": "turn_contract",
                "detail": {
                    "planner_required": True,
                    "artifact_reuse_audit": [
                        {"artifact_id": "a", "reason": "artifact_reuse"},
                        {"artifact_id": "b", "reason": "constraint_changed"},
                    ],
                },
            },
            {"kind": "admission", "agent": "planner", "status": "admitted"},
            {
                "kind": "wave_execution",
                "detail": {
                    "tasks": [
                        {"objective_key": "transport:route"},
                        {"objective_key": "transport:route"},
                    ]
                },
            },
        ],
        turn_metrics={"totals": {"dispatch_count": 2}},
        failure_reason=None,
        error=None,
    )
    denied = SimpleNamespace(
        agent_trace=[
            {"kind": "turn_contract", "detail": {"planner_required": True}},
            {
                "kind": "routing_stop",
                "detail": {"stop_reason": "planner_admission_denied"},
            },
        ],
        turn_metrics={"totals": {"dispatch_count": 0}},
        failure_reason=None,
        error=None,
    )
    result = SimpleNamespace(turns=[admitted, denied], errors=[])

    assert extract_internal_counters(result) == {
        "planner_required": 2,
        "planner_admitted": 1,
        "planner_budget_exhausted": 1,
        "dispatch_total": 2,
        "duplicate_dispatches": 1,
        "artifact_reuse_total": 2,
        "artifact_reused": 1,
        "stale_artifact_reused": 0,
        "rebuild_parent_reuse_total": 0,
        "stale_parent_rebuild_reused": 0,
        "final_current_artifact_required": 2,
        "final_current_artifact_missing": 2,
        "reviewer_rework_total": 0,
        "reviewer_rework_preserved": 0,
        "validation_revision_hash_required": 0,
        "validation_revision_hash_match": 0,
        "execution_total": 1,
        "system_error": 0,
    }


def test_lifecycle_metrics_distinguish_stale_parent_from_missing_current() -> None:
    rebuilt = SimpleNamespace(
        agent_trace=[{
            "kind": "turn_contract",
            "detail": {
                "planner_required": True,
                "artifact_reuse_audit": [{
                    "artifact_id": "itinerary_old",
                    "reason": "stale_parent_rebuild",
                }],
            },
        }],
        artifacts={"itinerary": {
            "artifact_status": "current",
            "state_version": {"constraint_revision": 3, "constraint_hash": "hash-3"},
            "validation_result": {
                "passed": True,
                "validated_constraint_revision": 3,
                "validated_constraint_hash": "hash-3",
            },
        }},
        turn_metrics={"totals": {}},
        failure_reason=None,
        error=None,
    )
    missing = SimpleNamespace(
        agent_trace=[{
            "kind": "turn_contract",
            "detail": {"planner_required": True, "artifact_reuse_audit": []},
        }],
        artifacts={},
        turn_metrics={"totals": {}},
        failure_reason=None,
        error=None,
    )

    counters = extract_internal_counters(
        SimpleNamespace(turns=[rebuilt, missing], errors=[])
    )
    assert counters["stale_parent_rebuild_reused"] == 1
    assert counters["stale_artifact_reused"] == 0
    assert counters["final_current_artifact_missing"] == 1
    assert counters["validation_revision_hash_match"] == 1
    metrics = internal_state_metrics([counters])
    assert metrics["stale_parent_rebuild_reuse_rate"] == 1.0
    assert metrics["final_current_artifact_missing_rate"] == 0.5
    assert metrics["validation_revision_hash_match_rate"] == 1.0


def test_metrics_ignore_trace_entries_owned_by_a_previous_request() -> None:
    turn = SimpleNamespace(
        request_id="request-current",
        agent_trace=[
            {
                "request_id": "request-previous",
                "kind": "turn_contract",
                "detail": {"delivery_intent": "rebuild_now"},
            },
            {
                "request_id": "request-previous",
                "kind": "admission",
                "agent": "planner",
                "status": "admitted",
            },
            {
                "request_id": "request-current",
                "kind": "turn_contract",
                "detail": {"delivery_intent": "lightweight_advice"},
            },
        ],
        artifacts={},
        turn_metrics={"totals": {}},
        failure_reason=None,
        error=None,
    )

    counters = extract_internal_counters(SimpleNamespace(turns=[turn], errors=[]))

    assert counters["planner_required"] == 0
    assert counters["planner_admitted"] == 0
