from __future__ import annotations

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.plan_invariants import validate_plan_artifact
from travel_agent.schemas import POI, ScoredPOI, TravelProfile


def _poi(poi_id: str, name: str) -> dict:
    return {
        "poi_id": poi_id, "name": name, "source": "provider",
        "verification_status": "verified",
    }


def _payload(day_count: int) -> dict:
    return {
        "itinerary": {
            "days": [
                {"day_index": index, "stops": [{"poi": _poi(f"p{index}", f"地点{index}")}]}
                for index in range(1, day_count + 1)
            ]
        },
        "critic": {"passed": True, "issues": []},
    }


def test_date_range_requires_every_calendar_day_even_when_sparse() -> None:
    profile = TravelProfile(
        destination="任意城市", days=3, start_date="2026-09-14",
        constraint_state={"date_start": "2026-09-14", "date_end": "2026-09-16"},
    )
    result = validate_plan_artifact(_payload(3), profile)
    assert result["passed"] is True


def test_date_range_generalizes_across_lengths_and_rejects_silent_day_drop() -> None:
    for start, end, count in (
        ("2026-01-01", "2026-01-04", 4),
        ("2028-02-28", "2028-03-01", 3),
    ):
        profile = TravelProfile(
            destination="任意城市", days=count, start_date=start,
            constraint_state={"date_start": start, "date_end": end},
        )
        assert validate_plan_artifact(_payload(count), profile)["passed"] is True
        codes = {item["code"] for item in validate_plan_artifact(_payload(count - 1), profile)["issues"]}
        assert "trip_day_count_mismatch" in codes


def test_duplicate_day_index_is_a_delivery_error() -> None:
    payload = _payload(3)
    payload["itinerary"]["days"][2]["day_index"] = 2
    result = validate_plan_artifact(payload, TravelProfile(destination="任意城市", days=3))
    assert result["passed"] is False
    assert "day_index_not_contiguous" in {item["code"] for item in result["issues"]}


def test_final_artifact_rejects_stop_outside_date_specific_opening_window() -> None:
    profile = TravelProfile(
        destination="任意城市",
        days=1,
        start_date="2026-11-07",
    )
    payload = _payload(1)
    payload["itinerary"]["days"][0]["stops"][0] = {
        "poi": {
            **_poi("dated-hours", "日期营业场馆"),
            "city": "任意城市",
            "category": "museum",
            "lat": 30.0,
            "lng": 120.0,
            "opening_hours": "09-07至12-31 周一至周日 09:30-18:30",
        },
        "start_time": "19:00",
        "duration_min": 90,
    }

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "outside_applicable_opening_hours" in {
        item["code"] for item in result["issues"]
    }


def test_final_artifact_rejects_arrival_after_last_admission() -> None:
    profile = TravelProfile(destination="任意城市", days=1, start_date="2026-10-15")
    payload = _payload(1)
    payload["itinerary"]["days"][0]["stops"][0] = {
        "poi": {
            **_poi("entry-cutoff", "停止入场场馆"),
            "city": "任意城市", "category": "museum", "lat": 30.0, "lng": 120.0,
            "opening_hours": "周一至周日 08:00-20:00开放 17:30停止入园",
        },
        "start_time": "18:00", "duration_min": 90,
    }

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "after_last_admission" in {item["code"] for item in result["issues"]}


def test_final_artifact_rejects_insufficient_transfer_time() -> None:
    profile = TravelProfile(destination="任意城市", days=1)
    payload = _payload(1)
    payload["itinerary"]["days"][0]["stops"] = [
        {
            "poi": _poi("origin", "上一站"),
            "start_time": "18:00", "duration_min": 60,
        },
        {
            "poi": _poi("destination", "下一站"),
            "start_time": "19:00", "duration_min": 60,
            "route_from_previous": {
                "origin_poi_id": "origin", "destination_poi_id": "destination",
                "distance_km": 5, "duration_min": 30,
                "mode": "public_transport", "source": "amap",
                "evidence_status": "provider_verified",
            },
        },
    ]

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "insufficient_transfer_time" in {
        item["code"] for item in result["issues"]
    }


def test_remaining_budget_excludes_prepaid_lodging_and_unknown_is_not_zero() -> None:
    profile = TravelProfile(
        destination="任意城市", days=3,
        constraint_state={"budget_remaining_cny": 1800, "prepaid_lodging_cny": 1200},
    )
    inputs = {"budgets": [{"artifact_id": "b1", "payload": {
        "days": 3, "companions": 1, "hotel": 1200, "meals": 500,
        "tickets": 300, "inner_city_transport": 200,
        "total_low": 1870, "total_high": 2530,
    }}]}
    budget = toolkit._build_budget_plan(profile, inputs, None)
    assert budget["prepaid_cost"] == 1200
    assert budget["lodging"] == 0
    assert budget["user_limit_basis"] == "remaining_budget_excludes_prepaid"
    assert budget["expected_total"] == 1000
    assert budget["unknown_items"] == []

    inputs["budgets"][0]["payload"].pop("tickets")
    unknown = toolkit._build_budget_plan(profile, inputs, None)
    assert unknown["tickets"] is None
    assert "tickets" in unknown["unknown_items"]
    assert unknown["expected_total"] is None


def test_renderer_rejects_older_artifact_from_same_request() -> None:
    ctx = build_session(session_id="latest-gate", persist=False)
    from travel_agent.artifact_policy import constraint_version

    version = constraint_version(ctx.profile)
    payload = {
        **_payload(1),
        "state_version": {
            "constraint_revision": version["revision"],
            "constraint_hash": version["constraint_hash"],
            "constraint_snapshot": version["constraint_snapshot"],
        },
        "validation_result": {"passed": True, "issues": []},
    }
    old_id = ctx.store.put("itinerary", payload, request_id="req-latest", task_id="p1", agent="planner")
    new_id = ctx.store.put("itinerary", payload, request_id="req-latest", task_id="p2", agent="planner")
    old_error, _ = toolkit._validate_plan_gate(ctx, old_id)
    new_error, _ = toolkit._validate_plan_gate(ctx, new_id)
    assert "最新" in old_error
    assert new_error == ""


def test_renderer_rejects_legacy_plan_without_constraint_version() -> None:
    ctx = build_session(session_id="legacy-version-gate", persist=False)
    artifact_id = ctx.store.put(
        "itinerary",
        {**_payload(1), "validation_result": {"passed": True, "issues": []}},
        agent="planner",
    )
    error, _ = toolkit._validate_plan_gate(ctx, artifact_id)
    assert "缺少约束版本" in error


def test_unapplied_repair_is_unresolved_and_blocks_validation() -> None:
    payload = _payload(1)
    payload.update({
        "parent_plan_artifact_id": "itinerary_parent",
        "repair_targets": ["planner"],
        "applied_changes": [],
        "unresolved_changes": ["补充返程路线"],
    })
    result = validate_plan_artifact(payload, TravelProfile(destination="任意城市", days=1))
    assert result["passed"] is False
    assert "repair_unresolved" in {item["code"] for item in result["issues"]}


def test_last_day_wall_clock_deadline_is_comparable_without_calendar_date() -> None:
    profile = TravelProfile(
        destination="任意城市",
        days=3,
        constraint_state={
            "return_deadline": "17:00",
            "return_deadline_local_time": "17:00",
            "return_deadline_day": "last_day",
        },
    )

    result = validate_plan_artifact(_payload(3), profile)

    assert "return_deadline_not_comparable" not in {
        item["code"] for item in result["issues"]
    }


def test_same_city_terminal_deadline_builds_return_plan() -> None:
    profile = TravelProfile(
        destination="任意城市",
        days=1,
        constraint_state={"return_location": "任意城市东站", "return_deadline": "17:00"},
    )

    plan = toolkit._build_return_plan(profile, _payload(1)["itinerary"], {})

    assert plan is not None
    assert plan["to_location"] == "任意城市东站"
    assert plan["arrival_deadline"] == "17:00"
    assert plan["activity_cutoff"] == "16:00"
    assert plan["intercity_segment"] is None


def test_return_plan_terminal_transfer_must_match_actual_final_stop() -> None:
    profile = TravelProfile(
        destination="任意城市",
        days=1,
        constraint_state={"return_location": "任意城市东站", "return_deadline": "19:00"},
    )
    payload = _payload(1)
    route = {
        "origin_poi_id": "p1", "destination_poi_id": "station",
        "mode": "public_transport", "duration_min": 30, "distance_km": 8,
        "source": "amap", "evidence_status": "provider_verified",
        "recommended_latest_departure": "18:00", "required_buffer_min": 30,
    }
    payload["required_route_anchors"] = {
        "last_stop_to_return_location": {**route, "route": route},
        "legs": [{"kind": "last_stop_to_return_location", **route, "route": route}],
    }
    payload["return_plan"] = {
        "required": True,
        "terminal_transfer": {**route, "origin_poi_id": "stale-stop"},
    }

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "return_plan_origin_mismatch" in {
        item["code"] for item in result["issues"]
    }


def test_user_owned_fixed_event_without_price_does_not_poison_budget_total() -> None:
    profile = TravelProfile(
        destination="任意城市",
        days=2,
        budget_limit=2000,
        constraint_state={
            "fixed_events": [{"day": 1, "start": "18:00", "end": "20:00", "location": "自有安排"}],
            "user_owned_unspecified_fixed_event_locations": ["自有安排"],
        },
    )
    inputs = {"budgets": [{"artifact_id": "b1", "payload": {
        "hotel": 500, "meals": 300, "tickets": 100,
        "inner_city_transport": 100, "total_low": 900, "total_high": 1200,
    }}]}

    budget = toolkit._build_budget_plan(profile, inputs, None)

    assert budget["fixed_event_cost"] == 0
    assert budget["expected_total"] == 1000
    assert budget["within_user_limit"] is True
    assert budget["external_commitment_costs_excluded"] == ["自有安排"]


def test_rejected_named_candidate_cannot_remain_scheduled() -> None:
    payload = _payload(1)
    payload["itinerary"]["days"][0]["stops"][0]["poi"]["name"] = "候选场馆"
    payload["candidate_verification"] = {
        "status": "verified_named_candidates",
        "results": [{
            "requested_name": "候选场馆",
            "matched_name": "候选场馆",
            "status": "closed",
        }],
    }

    result = validate_plan_artifact(payload, TravelProfile(destination="任意城市", days=1))

    assert result["passed"] is False
    assert "rejected_candidate_scheduled" in {item["code"] for item in result["issues"]}


def test_timed_fixed_event_requires_transfer_evidence() -> None:
    event = {"day": 1, "start": "18:00", "end": "20:00", "location": "自有晚餐区域"}
    profile = TravelProfile(
        destination="任意城市", days=1,
        constraint_state={
            "fixed_events": [event],
            "user_owned_unspecified_fixed_event_locations": ["自有晚餐区域"],
        },
    )
    payload = _payload(1)
    payload["fixed_event_plan"] = {"events": [{
        **event,
        "route_evidence_status": "unavailable",
        "required_buffer_min": 15,
    }]}
    payload["required_route_anchors"] = {"legs": [{
        "kind": "fixed_event_transfer",
        "required_name": "自有晚餐区域",
        "evidence_status": "unavailable",
        "routes": [],
    }]}

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "fixed_event_route_missing" in {item["code"] for item in result["issues"]}


def test_explicit_trip_origin_requires_route_to_actual_first_stop() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"origin": "上海虹桥"},
    )
    payload = _payload(1)
    payload["required_route_anchors"] = {"legs": [{
        "kind": "trip_origin_to_first_stop",
        "required_name": "上海虹桥",
        "origin_poi_id": None,
        "destination_poi_id": "p1",
        "evidence_status": "unavailable",
        "route": None,
    }]}

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "trip_origin_route_missing" in {
        item["code"] for item in result["issues"]
    }


def test_explicit_trip_origin_accepts_endpoint_bound_route_evidence() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=1,
        constraint_state={"origin": "杭州东站"},
    )
    payload = _payload(1)
    route = {
        "origin_poi_id": "station",
        "destination_poi_id": "p1",
        "distance_km": 8.5,
        "duration_min": 28,
        "mode": "public_transport",
        "source": "amap",
        "evidence_status": "provider_verified",
    }
    payload["required_route_anchors"] = {"legs": [{
        "kind": "trip_origin_to_first_stop",
        "required_name": "杭州东站",
        "origin_poi_id": "station",
        "destination_poi_id": "p1",
        "evidence_status": "provider_verified",
        "route": route,
    }]}

    assert validate_plan_artifact(payload, profile)["passed"] is True


def test_exact_activity_end_target_fails_closed_when_plan_ends_too_early() -> None:
    profile = TravelProfile(
        destination="任意城市",
        days=1,
        constraint_state={
            "activity_end_deadline": "20:30",
            "activity_end_target": "20:30",
        },
    )
    payload = _payload(1)
    payload["itinerary"]["days"][0]["stops"][0].update({
        "start_time": "14:00",
        "duration_min": 90,
    })

    result = validate_plan_artifact(payload, profile)

    assert result["passed"] is False
    assert "activity_end_target_missed" in {
        item["code"] for item in result["issues"]
    }


def test_named_candidate_verification_uses_venue_identity_without_trip_suffix() -> None:
    poi = POI(
        "museum", "甲省博物馆", "任意城市", "museum", 30.0, 120.0,
        4.8, 0.9, ["museum"], 120, "free", source="provider",
        canonical_name="甲省博物馆", entity_type="museum", source_poi_id="museum",
        verification_status="verified", opening_hours="09:00-17:00",
    )
    profile = TravelProfile(
        destination="任意城市", days=1,
        constraint_state={"candidate_attractions": ["甲省博物馆一日游"]},
    )

    verification = toolkit._build_candidate_verification(
        profile, [ScoredPOI(poi, 1.0, [])], {}
    )

    assert verification["results"][0]["status"] == "suitable"
    assert verification["results"][0]["matched_name"] == "甲省博物馆"
