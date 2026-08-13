from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from travel_agent.harness.cases import HarnessCase, load_cases_json
from travel_agent.harness.product import (
    DEFAULT_PRODUCT_CASES,
    FROZEN_SPLITS,
    aggregate_product_results,
    _execution_case,
)
from travel_agent.harness.result import HarnessSuiteResult
from travel_agent.harness.runner import AgentHarness


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LONG_HORIZON_CASES = DEFAULT_PRODUCT_CASES
LONG_HORIZON_DATASET_VERSION = "travel-agent-eval-production-v1.1"
EXPECTED_TURN_BANDS = {5: 4, 8: 4, 12: 4}
EXPECTED_SPLIT_COUNTS = {"dev": 4, "core_frozen": 4, "challenge_frozen": 4}
TRAINING_DATA_POLICY = "production_v1 frozen cases must never be used for training or tuning."


@dataclass(frozen=True)
class LongHorizonDatasetValidation:
    valid: bool
    errors: list[str]
    turn_counts: dict[int, int]
    split_counts: dict[str, int]


def validate_long_horizon_dataset(
    cases: list[HarnessCase],
) -> LongHorizonDatasetValidation:
    cases = [case for case in cases if case.subset == "long_horizon_state"]
    errors: list[str] = []
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case ids must be unique")
    turn_counts = dict(Counter(len(case.turns) for case in cases))
    if turn_counts != EXPECTED_TURN_BANDS:
        errors.append(f"turn-band counts mismatch: {turn_counts}")
    split_counts = dict(Counter(case.split for case in cases))
    if split_counts != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split counts mismatch: {split_counts}")
    for case in cases:
        if case.subset != "long_horizon_state":
            errors.append(f"{case.case_id}: subset must be long_horizon_state")
        if not case.gold_outcome or not case.gold_constraints_tree:
            errors.append(f"{case.case_id}: gold outcome and constraints are required")
        expected_turns = [int(item.get("turn") or 0) for item in case.turn_expectations]
        if expected_turns != list(range(1, len(case.turns) + 1)):
            errors.append(f"{case.case_id}: every turn needs one ordered expectation")
        if case.metadata.get("turn_mode") != "multi_turn":
            errors.append(f"{case.case_id}: turn_mode must be multi_turn")
    return LongHorizonDatasetValidation(not errors, errors, turn_counts, split_counts)


def run_long_horizon_suite(
    harness: AgentHarness,
    *,
    case_path: Path | str = DEFAULT_LONG_HORIZON_CASES,
    split: str = "dev",
    limit: int | None = None,
) -> HarnessSuiteResult:
    all_cases = load_cases_json(case_path)
    validation = validate_long_horizon_dataset(all_cases)
    if not validation.valid:
        raise ValueError("invalid long_horizon_v1 dataset: " + "; ".join(validation.errors))
    if split not in {"dev", "frozen", "all"}:
        raise ValueError(f"unknown long_horizon_v1 split: {split}")
    cases = [case for case in all_cases if case.subset == "long_horizon_state"]
    if split == "dev":
        cases = [case for case in cases if case.split == "dev"]
    elif split == "frozen":
        cases = [case for case in cases if case.split in FROZEN_SPLITS]
    if limit is not None:
        cases = cases[: max(1, limit)]
    variant = str(harness.environment.variant or "").upper() or "V3"
    executions = [
        (case, harness.run_case(_execution_case(case, 1)), 1)
        for case in cases
    ]
    summary = aggregate_product_results(
        executions,
        remediation_complete=True,
        variant=variant,
    )
    return HarnessSuiteResult(
        suite="agent-long-horizon",
        mode="long_horizon",
        environment=harness.environment.mode,
        case_count=len(cases),
        attempted_count=len(executions),
        metrics={key: value for key, value in summary.items() if key not in {"cases", "fine_tuning"}},
        rows=summary["cases"],
        artifacts={
            "dataset_version": LONG_HORIZON_DATASET_VERSION,
            "independent_case_count": len(cases),
            "execution_count": len(executions),
            "turn_band_counts": validation.turn_counts,
            "split_counts": validation.split_counts,
            "selected_split": split,
            "training_data_policy": TRAINING_DATA_POLICY,
        },
    )
