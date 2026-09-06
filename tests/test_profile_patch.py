"""Profile Patch 语义测试：SET / CLEAR / UNCHANGED 三态与清洗校验。"""

from __future__ import annotations

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import TaskType, build_rule_patches
from travel_agent.agent.turn_lifecycle import _sync_profile_constraints
from travel_agent.profile_patch import (
    PatchOp,
    SlotPatch,
    extract_slot_clears,
    extracted_to_patches,
    merge_patches,
    patches_to_payload,
)
from travel_agent.workflow_rules import extract_profile_rule_based


def test_extracted_to_patches_skips_absent_fields() -> None:
    patches = extracted_to_patches(extract_profile_rule_based("去杭州三天"))
    assert patches["destination"] == SlotPatch(PatchOp.SET, "杭州")
    assert patches["days"] == SlotPatch(PatchOp.SET, 3)
    # 未提及的槽位不产生 patch（UNCHANGED），默认值也不写入
    assert "companions" not in patches
    assert "budget_level" not in patches
    assert "pace" not in patches
    assert "transport_mode" not in patches


def test_missing_field_does_not_override_previous_turn() -> None:
    """第一轮「去杭州三天」，第二轮「带一位老人」：destination 不能被 None 覆盖。"""
    ctx = build_session(persist=False)
    first = build_rule_patches("去杭州三天", TaskType.FULL_TRIP_PLAN)
    toolkit.apply_profile_patches(ctx, patches=patches_to_payload(first))
    assert ctx.profile.destination == "杭州"
    assert ctx.profile.days == 3

    second = build_rule_patches("带一位老人", TaskType.FULL_TRIP_PLAN)
    assert "destination" not in second
    toolkit.apply_profile_patches(ctx, patches=patches_to_payload(second))
    assert ctx.profile.destination == "杭州"
    assert ctx.profile.days == 3
    assert ctx.profile.companions == "老人"


def test_clear_removes_confirmed_value() -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "苏州"
    ctx.profile.days = 3

    toolkit.apply_profile_patches(ctx, patches={"destination": {"op": "clear", "value": "苏州"}})
    assert ctx.profile.destination is None
    assert ctx.profile.days == 3  # 未提及的天数保持 UNCHANGED


def test_clear_list_and_defaults_reset() -> None:
    ctx = build_session(persist=False)
    ctx.profile.must_visit = ["西湖"]
    ctx.profile.pace = "relaxed"
    ctx.profile.transport_mode = "drive"

    toolkit.apply_profile_patches(
        ctx,
        patches={
            "must_visit": {"op": "clear"},
            "pace": {"op": "clear"},
            "transport_mode": {"op": "clear"},
        },
    )
    assert ctx.profile.must_visit == []
    assert ctx.profile.pace == "standard"
    assert ctx.profile.transport_mode == "public_transport"


def test_invalid_values_are_dropped() -> None:
    ctx = build_session(persist=False)
    toolkit.apply_profile_patches(
        ctx,
        patches={
            "days": {"op": "set", "value": -2},
            "budget_level": {"op": "set", "value": "luxury"},
            "destination": {"op": "set", "value": "  北京  "},
        },
    )
    assert ctx.profile.days is None
    assert ctx.profile.budget_level is None
    assert ctx.profile.destination == "北京"


def test_extract_slot_clears_high_precision() -> None:
    clears = extract_slot_clears("苏州不去了，换成杭州")
    assert clears["destination"].op == PatchOp.CLEAR
    assert clears["destination"].value == "苏州"

    assert extract_slot_clears("目的地先不定")["destination"].op == PatchOp.CLEAR
    assert extract_slot_clears("天数先不定")["days"].op == PatchOp.CLEAR
    # 无明确取消语义时不产生 CLEAR
    assert extract_slot_clears("帮我规划北京三天") == {}


def test_merge_patches_drops_conflicting_set_of_cancelled_city() -> None:
    sets = {"destination": SlotPatch(PatchOp.SET, "苏州")}
    clears = {"destination": SlotPatch(PatchOp.CLEAR, "苏州")}
    merged = merge_patches(sets, clears)
    assert "destination" not in merged

    # 取消苏州但明确设置杭州：SET 新值本身就是替换，保留 SET 忽略 CLEAR
    sets = {"destination": SlotPatch(PatchOp.SET, "杭州")}
    merged = merge_patches(sets, clears)
    assert merged["destination"] == SlotPatch(PatchOp.SET, "杭州")


def test_cancel_city_and_set_new_city_end_to_end() -> None:
    ctx = build_session(persist=False)
    ctx.profile.destination = "苏州"

    patches = build_rule_patches("苏州不去了，换成杭州", TaskType.FULL_TRIP_PLAN)
    payload = patches_to_payload(patches)
    # CLEAR 与 SET 并存时，应用顺序保证新值生效
    toolkit.apply_profile_patches(ctx, patches=payload)
    assert ctx.profile.destination == "杭州"


def test_preference_observations_recorded() -> None:
    ctx = build_session(persist=False)
    toolkit.apply_profile_patches(
        ctx,
        patches={
            "interests": {"op": "set", "value": ["food"]},
            "must_visit": {"op": "clear"},
        },
    )
    categories = {(o["category"], o["polarity"]) for o in ctx.pending_preference_observations}
    assert ("interests", "positive") in categories
    assert ("must_visit", "negative") in categories


def test_constraint_state_syncs_and_sanitizes_compact_profile() -> None:
    ctx = build_session(persist=False)
    ctx.profile.must_visit = ["虎丘", "海边", "三天行程"]
    ctx.profile.interests = ["海边"]
    ctx.profile.constraint_state = {
        "must_visit": ["鼓浪屿", "海边"],
        "removed": ["虎丘"],
        "fixed_events": [
            {"day": 2, "start": "14:30", "end": "16:30", "location": "陕西历史博物馆"}
        ],
        "lodging_area": "钟楼附近",
    }

    _sync_profile_constraints(ctx.profile)

    # A fixed-event location is scheduled literally but is not promoted into
    # the independent must-visit set.
    assert ctx.profile.must_visit == ["鼓浪屿"]
    assert ctx.profile.interests == ["nature"]
    assert ctx.profile.hotel_area == "钟楼附近"
    assert ctx.profile.constraint_state["must_visit"] == ["鼓浪屿"]


def test_constraint_state_syncs_latest_explicit_pace_to_compact_profile() -> None:
    ctx = build_session(persist=False)
    ctx.profile.pace = "standard"
    ctx.profile.constraint_state = {"pace": "relaxed"}

    _sync_profile_constraints(ctx.profile)

    assert ctx.profile.pace == "relaxed"


def test_constraint_state_sync_lets_avoid_override_stale_must_visit() -> None:
    ctx = build_session(persist=False)
    ctx.profile.must_visit = ["外滩", "迪士尼"]
    ctx.profile.avoid = ["迪士尼"]
    ctx.profile.constraint_state = {
        "must_visit": ["外滩", "迪士尼"],
        "avoid": ["迪士尼"],
    }

    _sync_profile_constraints(ctx.profile)

    assert ctx.profile.must_visit == ["外滩"]
    assert ctx.profile.constraint_state["must_visit"] == ["外滩"]


def test_stable_avoid_does_not_override_current_trip_must_visit() -> None:
    ctx = build_session(persist=False)
    ctx.profile.must_visit = ["迪士尼"]
    ctx.profile.avoid = ["迪士尼"]
    ctx.profile.constraint_state = {"must_visit": ["迪士尼"]}

    _sync_profile_constraints(ctx.profile)

    assert ctx.profile.must_visit == ["迪士尼"]
    assert ctx.profile.constraint_state["must_visit"] == ["迪士尼"]


def test_profile_brief_is_a_historical_snapshot() -> None:
    ctx = build_session(persist=False)
    ctx.profile.must_visit = ["拙政园"]
    ctx.profile.constraint_state = {"removed": ["虎丘"]}

    snapshot = toolkit._profile_brief(ctx.profile)
    ctx.profile.must_visit.append("山塘街")
    ctx.profile.constraint_state["removed"].append("留园")

    assert snapshot["must_visit"] == ["拙政园"]
    assert snapshot["constraint_state"]["removed"] == ["虎丘"]
