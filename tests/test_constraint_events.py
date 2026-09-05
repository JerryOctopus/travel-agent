from __future__ import annotations

from travel_agent.constraint_events import (
    META_KEY,
    merge_constraint_update,
    replay_events,
    validate_constraint_state,
)


def test_scalar_replace_preserves_unmodified_fields_and_provenance() -> None:
    state: dict = {}
    merge_constraint_update(state, {"duration_days": 5, "lodging_area": "湖滨"}, source_turn=1)
    merge_constraint_update(state, {"budget_max_cny": 6800}, source_turn=2)
    merge_constraint_update(state, {"budget_max_cny": 5200}, source_turn=7)

    assert state["duration_days"] == 5
    assert state["lodging_area"] == "湖滨"
    assert state["budget_max_cny"] == 5200
    budget = [item for item in state[META_KEY] if item["field"] == "budget_max_cny"]
    assert [item["active"] for item in budget] == [False, True]
    assert budget[-1]["supersedes_event_id"] == budget[0]["event_id"]


def test_remove_tombstone_blocks_soft_reactivation_variant() -> None:
    state: dict = {}
    merge_constraint_update(state, {"must_visit": ["海洋公园"]}, source_turn=1)
    merge_constraint_update(state, {"removed": ["海洋公园"]}, source_turn=5)
    merge_constraint_update(
        state, {"must_visit": ["海洋公园"]}, source_turn=10, explicitness="soft"
    )

    assert "海洋公园" not in state.get("must_visit", [])
    assert state["removed"] == ["海洋公园"]


def test_explicit_readd_can_override_tombstone_counterexample() -> None:
    state: dict = {}
    merge_constraint_update(state, {"removed": ["古城墙"]}, source_turn=3)
    merge_constraint_update(state, {"must_visit": ["古城墙"]}, source_turn=8)

    assert state["must_visit"] == ["古城墙"]
    # The historical removal remains in the event log but is no longer active.
    assert "removed" not in state
    assert any(item["field"] == "removed" and not item["active"] for item in state[META_KEY])


def test_confirm_is_idempotent_for_scalar_and_fixed_event() -> None:
    event = {"day": 2, "start": "18:00", "end": "20:00", "location": "江畔餐厅"}
    state: dict = {}
    merge_constraint_update(state, {"lodging_area": "老城南", "fixed_events": [event]}, source_turn=1)
    merge_constraint_update(state, {}, source_turn=4, confirmations={"lodging_area": "老城南"})
    merge_constraint_update(state, {}, source_turn=5, confirmations={"lodging_area": "老城南"})

    assert state["lodging_area"] == "老城南"
    assert state["fixed_events"] == [event]
    assert len([item for item in state[META_KEY] if item["operation"] == "confirm"]) == 2


def test_replay_rebuilds_active_state_after_twelve_turn_style_log() -> None:
    state: dict = {}
    updates = [
        {"duration_days": 6}, {"budget_max_cny": 9000}, {"dietary": ["不吃辣"]},
        {"must_visit": ["山城步道"]}, {}, {"removed": ["山城步道"]},
        {"must_visit": ["江岸公园"]}, {}, {"fixed_events": [{"day": 2, "start": "14:00", "end": "16:00", "location": "艺术中心"}]},
        {}, {"budget_max_cny": 7600}, {},
    ]
    for turn, update in enumerate(updates, 1):
        merge_constraint_update(state, update, source_turn=turn)

    assert replay_events(state[META_KEY]) == {key: value for key, value in state.items() if key != META_KEY}
    assert state["duration_days"] == 6
    assert state["dietary"] == ["不吃辣"]
    assert state["must_visit"] == ["江岸公园"]
    assert state["fixed_events"][0]["day"] == 2


def test_gate_reports_specific_conflicts_instead_of_selecting_old_value() -> None:
    state: dict = {}
    merge_constraint_update(
        state,
        {
            "duration_days": 3,
            "date_start": "2026-10-01",
            "date_end": "2026-10-05",
            "fixed_events": [{"day": 4, "location": "剧院"}],
        },
        source_turn=2,
    )

    codes = {item["code"] for item in validate_constraint_state(state)}
    assert codes == {"duration_date_range_mismatch", "fixed_event_out_of_range"}
