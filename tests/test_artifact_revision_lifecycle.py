from __future__ import annotations

from copy import deepcopy

import pytest

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType, analyze_travel_turn
from travel_agent.agent.turn_lifecycle import prepare_turn
from travel_agent.artifact_policy import (
    active_constraint_snapshot,
    constraint_version,
    material_changed_fields,
    update_constraint_version,
)
from travel_agent.constraint_events import merge_constraint_update
from travel_agent.plan_invariants import validate_plan_artifact
from travel_agent.schemas import TravelProfile


def _payload(profile: TravelProfile, names: list[str], *, parent: str | None = None) -> dict:
    version = constraint_version(profile)
    days = []
    for index in range(1, int(profile.days or 1) + 1):
        day_names = names if index == 1 else [f"普通地点{index}"]
        stops = []
        for offset, name in enumerate(day_names, 1):
            stop = {
                "poi": {
                    "poi_id": f"p{index}-{offset}", "name": name,
                    "canonical_name": name, "source": "provider",
                    "verification_status": "verified",
                }
            }
            if offset > 1:
                stop["route_from_previous"] = {
                    "origin_poi_id": f"p{index}-{offset - 1}",
                    "destination_poi_id": f"p{index}-{offset}",
                    "evidence_status": "deterministic_estimate",
                }
            stops.append(stop)
        days.append({
            "day_index": index,
            "stops": stops,
        })
    payload = {
        "artifact_status": "current",
        "parent_plan_artifact_id": parent,
        "revision_lineage": [parent] if parent else [],
        "state_version": {
            "constraint_revision": version["revision"],
            "constraint_hash": version["constraint_hash"],
            "constraint_snapshot": version["constraint_snapshot"],
        },
        "itinerary": {"city": profile.destination, "days": days},
        "critic": {"passed": True, "issues": []},
    }
    payload["validation_result"] = validate_plan_artifact(payload, profile)
    return payload


def _put_current(ctx, names: list[str], *, parent: str | None = None) -> str:
    payload = _payload(ctx.profile, names, parent=parent)
    assert payload["validation_result"]["passed"] is True
    return ctx.store.put("itinerary", payload, agent="planner")


def _material_update(ctx, update: dict) -> tuple[str, dict]:
    before = active_constraint_snapshot(ctx.profile)
    previous = ctx.store.latest_current_id("itinerary")
    merge_constraint_update(ctx.profile.constraint_state, update, source_turn=2)
    if "must_visit" in update:
        ctx.profile.must_visit = list(ctx.profile.constraint_state.get("must_visit") or [])
    if "removed" in update:
        removed = list(ctx.profile.constraint_state.get("removed") or [])
        ctx.profile.must_visit = [
            item for item in ctx.profile.must_visit
            if not any(term in item or item in term for term in removed)
        ]
    version = update_constraint_version(ctx.profile, before)
    assert version["material_changed"] is True
    ctx.store.invalidate_itineraries(
        constraint_revision=version["revision"],
        constraint_hash=version["constraint_hash"],
        changed_fields=version["changed_fields"],
    )
    return str(previous), version


def test_added_must_visit_rebuilds_final_artifact_and_keeps_lineage() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["旧景点"])

    parent, version = _material_update(ctx, {"must_visit": ["新增地点"]})
    assert parent == old_id
    assert ctx.store.get_record(old_id)["artifact_status"] == "stale"
    assert ctx.store.get(old_id)["validation_result"]["passed"] is False

    new_id = _put_current(ctx, ["旧景点", "新增地点"], parent=old_id)
    new_payload = ctx.store.get(new_id)
    assert new_payload["state_version"]["constraint_revision"] == version["revision"]
    assert new_payload["state_version"]["constraint_hash"] == version["constraint_hash"]
    assert (
        new_payload["validation_result"]["validated_constraint_revision"]
        == version["revision"]
    )
    assert (
        new_payload["validation_result"]["validated_constraint_hash"]
        == version["constraint_hash"]
    )
    assert old_id in new_payload["revision_lineage"]
    assert new_payload["validation_result"]["passed"] is True


def test_validation_failure_never_becomes_current_or_final_artifact() -> None:
    from travel_agent.harness.runner import _artifact_snapshot

    ctx = build_session(persist=False)
    failed_id = ctx.store.put(
        "itinerary",
        {
            "artifact_status": "validation_failure",
            "itinerary": {"days": []},
            "validation_result": {"passed": False},
        },
        agent="planner",
    )

    assert ctx.store.latest_id("itinerary") == failed_id
    assert ctx.store.latest_current_id("itinerary") is None
    assert ctx.store.latest_revisable_id("itinerary") is None
    assert "itinerary" not in _artifact_snapshot(ctx)


def test_removed_place_is_absent_from_new_artifact_and_old_one_is_stale() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1, must_visit=["待删除地点"])
    ctx.profile.constraint_state = {"must_visit": ["待删除地点"]}
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["待删除地点"])

    _material_update(ctx, {"removed": ["待删除地点"]})
    new_id = _put_current(ctx, ["替代地点"], parent=old_id)
    rendered = str(ctx.store.get(new_id)["itinerary"])
    assert "待删除地点" not in rendered
    assert ctx.store.get_record(old_id)["artifact_status"] == "stale"


def test_new_fixed_event_revalidates_old_plan_and_detects_missing_event() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["普通地点"])
    old_passed = deepcopy(ctx.store.get(old_id)["validation_result"])

    _material_update(ctx, {"fixed_events": [{
        "day": 1, "start": "10:00", "end": "12:00", "location": "固定活动"
    }]})
    current_result = validate_plan_artifact(ctx.store.get(old_id), ctx.profile)
    assert old_passed["passed"] is True
    assert current_result["passed"] is False
    assert {item["code"] for item in current_result["issues"]} >= {
        "artifact_constraint_version_mismatch", "fixed_event_artifact_missing"
    }


@pytest.mark.parametrize("update", [
    {"budget_max_cny": 5200},
    {"budget_per_person_cny": 900},
    {"return_deadline": "2026-10-03T17:00+08:00"},
    {"return_deadline": "2026-10-03T18:30+08:00"},
])
def test_budget_or_return_deadline_variants_invalidate_old_artifact(update: dict) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=3, start_date="2026-10-01")
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["普通地点"])
    _material_update(ctx, update)
    assert ctx.store.get_record(old_id)["artifact_status"] == "stale"
    assert ctx.store.latest_current_id("itinerary") is None
    assert ctx.store.latest_revisable_id("itinerary") == old_id


def test_material_change_keeps_stable_rebuild_parent_across_followups(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=2)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["普通地点"])

    first = prepare_turn(
        "整体预算改成5000元，先记住，之后再出最终版。",
        ctx,
        offline_settings,
        [],
    )
    assert first.early_reply is not None
    assert first.early_reply.planner_status == "deferred_state_update"
    assert first.revisable_parent_artifact_id == old_id
    assert ctx.store.latest_current_id("itinerary") is None
    assert ctx.store.latest_revisable_id("itinerary") == old_id
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"

    second = prepare_turn(
        "再补充：全程饮食清淡，先记住，等我说最终版。",
        ctx,
        offline_settings,
        [("user", "整体预算改成5000元，先记住，之后再出最终版。")],
    )
    assert second.early_reply is not None
    assert "没有可修改的既有行程" not in second.early_reply.text
    assert second.revisable_parent_artifact_id == old_id
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"

    final = prepare_turn(
        "按全部条件生成最终版完整行程。",
        ctx,
        offline_settings,
        [],
    )
    assert final.early_reply is None
    assert final.analysis.task_type == TaskType.FULL_ITINERARY
    assert final.existing_plan_artifact_id is None
    assert final.revisable_parent_artifact_id == old_id
    assert final.turn_inputs["parent_plan_artifact_id"] == old_id
    assert any(
        item.get("reason") == "stale_parent_rebuild"
        for item in final.turn_inputs["artifact_reuse_audit"]
    )


def test_undated_last_day_deadline_is_preserved_without_inventing_date(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=3)

    prepared = prepare_turn(
        "请生成三天完整行程，最后一天17:00前返回车站。",
        ctx,
        offline_settings,
        [],
    )

    state = ctx.profile.constraint_state
    assert state["return_deadline"] == "17:00"
    assert state["return_deadline_local_time"] == "17:00"
    assert state["return_deadline_day"] == "last_day"
    assert prepared.early_reply is None
    assert prepared.turn_inputs["profile"]["constraint_state"][
        "return_deadline_day"
    ] == "last_day"


@pytest.mark.parametrize(("before", "after"), [
    ({"date_start": "2026-10-01"}, {"date_start": "2026-10-02"}),
    ({"duration_days": 3}, {"duration_days": 4}),
    ({"destination": "甲地"}, {"destination": "乙地"}),
    ({"must_visit": ["甲馆"]}, {"must_visit": ["甲馆", "乙馆"]}),
    ({"removed": []}, {"removed": ["甲馆"]}),
    ({"fixed_events": [{"start": "10:00"}]}, {"fixed_events": [{"start": "11:00"}]}),
    ({"budget_max_cny": 8000}, {"budget_max_cny": 7000}),
    ({"lodging_area": "北区"}, {"lodging_area": "南区"}),
    ({"return_deadline": "2026-10-03T17:00+08:00"}, {"return_deadline": "2026-10-03T16:00+08:00"}),
    ({"transport_mode": "drive"}, {"transport_mode": "public_transport"}),
    ({"dietary": ["素食"]}, {"dietary": ["无麸质"]}),
    ({"mobility": "standard"}, {"mobility": "low_walking"}),
])
def test_each_material_rule_has_semantic_variants(before: dict, after: dict) -> None:
    assert material_changed_fields(before, after)


def test_non_material_counterexample_does_not_invalidate_plan() -> None:
    assert material_changed_fields(
        {"referenced_day_index": 1}, {"referenced_day_index": 2}
    ) == []


@pytest.mark.parametrize("message", [
    "最后一天17:00前返回住处。",
    "最后一天最晚17:00要回到住处。",
])
def test_return_deadline_wording_variants_save_full_timestamp(
    message: str, offline_settings
) -> None:
    ctx = build_session(persist=False)
    ctx.reference_datetime = "2026-09-20T09:00:00+08:00"
    ctx.profile = TravelProfile(destination="任意城市", days=3, start_date="2026-10-01")
    ctx.profile.constraint_state = {
        "date_start": "2026-10-01", "date_end": "2026-10-03", "duration_days": 3
    }
    update_constraint_version(ctx.profile, {})
    _put_current(ctx, ["普通地点"])

    prepare_turn(message, ctx, offline_settings, [])
    assert ctx.profile.constraint_state["return_deadline"] == "2026-10-03T17:00+08:00"


def test_return_clock_counterexample_is_not_saved_as_deadline(offline_settings) -> None:
    ctx = build_session(persist=False)
    analysis = analyze_travel_turn("最后一天17:00去看展。", ctx, offline_settings)
    assert "return_deadline" not in analysis.constraint_state


def test_gate_uses_current_constraints_not_historical_pass_bit() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["普通地点"])
    assert toolkit._validate_plan_gate(ctx, old_id)[0] == ""

    _material_update(ctx, {"must_visit": ["新增地点"]})
    error, _ = toolkit._validate_plan_gate(ctx, old_id)
    assert "当前" in error or "最新" in error


def test_gate_detects_direct_profile_mutation_even_before_stale_marker() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["普通地点"])

    ctx.profile.must_visit = ["临时新增地点"]
    error, _ = toolkit._validate_plan_gate(ctx, old_id)
    assert "当前约束" in error


def test_candidate_does_not_replace_current_until_atomic_promotion() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["旧地点"])
    candidate_payload = _payload(ctx.profile, ["新地点"], parent=old_id)
    candidate_payload["artifact_status"] = "candidate"

    candidate_id = ctx.store.put("itinerary", candidate_payload, agent="planner")

    assert ctx.store.latest_current_id("itinerary") == old_id
    assert ctx.store.get_record(candidate_id)["artifact_status"] == "candidate"
    assert ctx.store.promote_itinerary(candidate_id, promotion_reason="unit_finalizer")
    assert ctx.store.latest_current_id("itinerary") == candidate_id
    assert ctx.store.get_record(old_id)["artifact_status"] == "historical"
    current = ctx.store.get(candidate_id)
    assert current["state_version"]["constraint_revision"] == current[
        "validation_result"
    ]["validated_constraint_revision"]
    assert current["state_version"]["constraint_hash"] == current[
        "validation_result"
    ]["validated_constraint_hash"]


def test_invalid_candidate_cannot_atomically_replace_current() -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="任意城市", days=1)
    update_constraint_version(ctx.profile, {})
    old_id = _put_current(ctx, ["旧地点"])
    invalid = _payload(ctx.profile, ["新地点"], parent=old_id)
    invalid["artifact_status"] = "candidate"
    invalid["validation_result"]["validated_constraint_hash"] = "wrong-hash"
    candidate_id = ctx.store.put("itinerary", invalid, agent="planner")

    assert ctx.store.promote_itinerary(candidate_id) is False
    assert ctx.store.latest_current_id("itinerary") == old_id
    assert ctx.store.get_record(old_id)["artifact_status"] == "current"
