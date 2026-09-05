from __future__ import annotations

from travel_agent.harness.cases import HarnessCase, load_cases_json
from travel_agent.harness.long_horizon import (
    DEFAULT_LONG_HORIZON_CASES,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TURN_BANDS,
    validate_long_horizon_dataset,
)
from travel_agent.harness.result import HarnessCaseResult, HarnessTurnResult
from travel_agent.harness.validators import validate_case_result


def _turn(profile: dict) -> HarnessTurnResult:
    return HarnessTurnResult(
        user_message="update",
        reply_text="done",
        tool_trace=[],
        used_real_agent=True,
        clarification=False,
        profile=profile,
        artifacts={},
    )


def test_long_horizon_dataset_has_balanced_5_8_12_turn_bands() -> None:
    cases = load_cases_json(DEFAULT_LONG_HORIZON_CASES)

    validation = validate_long_horizon_dataset(cases)

    assert validation.valid is True, validation.errors
    assert validation.turn_counts == EXPECTED_TURN_BANDS
    assert validation.split_counts == EXPECTED_SPLIT_COUNTS
    assert validation.frozen_warnings == []
    long_cases = [case for case in cases if case.subset == "long_horizon_state"]
    assert len(long_cases) == 12
    assert all(len(case.turn_expectations) == len(case.turns) for case in long_cases)
    target = next(case for case in long_cases if case.case_id == "lh_12_001")
    assert target.turns[0] == "杭州五天，两个人，总预算5000元。"
    assert "date_start" not in target.turn_expectations[0]["constraints"]
    assert target.gold_constraints_tree["return_deadline"] == "2026-10-05T17:00:00+08:00"


def test_turn_expectations_score_each_checkpoint_not_only_final_state() -> None:
    case = HarnessCase(
        case_id="checkpoint",
        turns=["杭州三天", "预算3000", "预算改成2500"],
        turn_expectations=[
            {"turn": 1, "constraints": {"destinations": ["杭州"], "duration_days": 3}},
            {"turn": 2, "constraints": {"budget_max_cny": 3000}},
            {"turn": 3, "constraints": {"budget_max_cny": 2500}},
        ],
    )
    result = HarnessCaseResult(
        case_id=case.case_id,
        turns=[
            _turn({"destination": "杭州", "days": 3}),
            _turn({"destination": "杭州", "days": 3, "budget_limit": 9999}),
            _turn({"destination": "杭州", "days": 3, "budget_limit": 2500}),
        ],
        final_profile={"destination": "杭州", "days": 3, "budget_limit": 2500},
        final_artifacts={},
    )

    metrics = validate_case_result(case, result)

    assert metrics["turn_state_ok"] is False
    assert metrics["turn_state_accuracy"] == 0.6667
    assert [item["passed"] for item in metrics["turn_state_results"]] == [True, False, True]
