from __future__ import annotations

from travel_agent.agent.session import build_session
from travel_agent.artifact_policy import constraint_fingerprint, reusable_artifact_ids


def _put_bound(ctx, kind: str, state: dict, payload: dict | None = None) -> str:
    bound_payload = payload or {"items": [1]}
    artifact_id = ctx.store.put(kind, bound_payload)
    ctx.store._items[artifact_id]["constraint_fingerprint"] = constraint_fingerprint(
        kind, state, bound_payload
    )
    return artifact_id


def test_budget_change_reuses_poi_but_invalidates_budget() -> None:
    ctx = build_session(persist=False)
    old = {"destination": "青岛", "budget_max_cny": 8000, "must_visit": ["海滨栈道"]}
    poi = _put_bound(ctx, "candidates", old)
    budget = _put_bound(ctx, "budget", old)

    ids, audit = reusable_artifact_ids(ctx.store, {**old, "budget_max_cny": 6500})

    assert poi in ids and budget not in ids
    assert {item["reason"] for item in audit if item["artifact_id"] == budget} == {"constraint_changed"}


def test_lodging_change_invalidates_hotel_and_route_variants() -> None:
    ctx = build_session(persist=False)
    old = {"destination": "泉州", "lodging_area": "古城", "transport_modes": ["public_transport"]}
    hotel = _put_bound(ctx, "hotels", old)
    route = _put_bound(ctx, "routes", old)
    candidate = _put_bound(ctx, "candidates", old)

    ids, _ = reusable_artifact_ids(ctx.store, {**old, "lodging_area": "东海"})

    assert candidate in ids
    assert hotel not in ids and route not in ids


def test_transport_change_invalidates_route_counterexample() -> None:
    ctx = build_session(persist=False)
    old = {"destination": "大连", "transport_mode": "drive"}
    route = _put_bound(ctx, "routes", old)
    ids, _ = reusable_artifact_ids(ctx.store, {**old, "transport_mode": "public_transport"})
    assert route not in ids


def test_location_anchor_change_invalidates_route_and_candidate_evidence() -> None:
    ctx = build_session(persist=False)
    old = {
        "destination_city": "成都",
        "location_anchor": "太古里",
        "comparison_candidates": ["甲区", "乙区"],
    }
    route = _put_bound(ctx, "routes", old)
    candidate = _put_bound(ctx, "candidates", old)

    ids, _ = reusable_artifact_ids(ctx.store, {**old, "location_anchor": "火车南站"})

    assert route not in ids
    assert candidate not in ids


def test_date_change_invalidates_temporal_poi_facts_but_not_identity_only_evidence() -> None:
    old = {"destination": "测试城", "date_start": "2026-09-01"}

    temporal_ctx = build_session(persist=False)
    temporal = _put_bound(temporal_ctx, "candidates", old, {"pois": [{
        "poi_id": "poi-1", "source_poi_id": "canonical-1", "name": "自然馆",
        "canonical_name": "测试城市自然博物馆", "source": "provider",
        "verification_status": "verified", "opening_hours": "09:00-17:00",
    }]})
    temporal_ids, _ = reusable_artifact_ids(
        temporal_ctx.store, {**old, "date_start": "2026-09-02"}
    )
    assert temporal not in temporal_ids

    identity_ctx = build_session(persist=False)
    identity = _put_bound(identity_ctx, "candidates", old, {"pois": [{
        "poi_id": "poi-1", "source_poi_id": "canonical-1", "name": "自然馆",
        "canonical_name": "测试城市自然博物馆", "source": "provider",
        "verification_status": "verified",
    }]})
    identity_ids, _ = reusable_artifact_ids(
        identity_ctx.store, {**old, "date_start": "2026-09-02"}
    )
    assert identity in identity_ids


def test_verified_poi_from_stale_lineage_is_reusable_with_reason_and_lineage() -> None:
    ctx = build_session(persist=False)
    state = {"destination": "测试城", "must_visit": ["自然博物馆"]}
    poi = _put_bound(ctx, "candidates", state, {"pois": [{
        "poi_id": "poi-1", "source_poi_id": "canonical-1", "name": "自然馆",
        "canonical_name": "测试城市自然博物馆", "source": "provider",
        "verification_status": "verified",
    }], "source_artifact_ids": ["source-search-1"]})
    ctx.store._items[poi]["artifact_status"] = "stale"

    ids, audit = reusable_artifact_ids(ctx.store, {**state, "budget_max_cny": 5000})

    assert poi in ids
    reused = next(item for item in audit if item["artifact_id"] == poi and item["reason"] == "stale_artifact_reuse")
    assert reused["reuse_reason"] == "kind_specific_fingerprint_match"
    assert reused["source_lineage"] == [poi, "source-search-1"]
