from __future__ import annotations

from scripts.eval_hybrid_actuation_causal import run


def test_minimal_causal_ab_is_fixed_isolated_and_passes_all_gates() -> None:
    result = run()

    assert result["cell_count"] == 6
    assert result["conclusion"] == "READY_FOR_TARGETED_DEV"
    assert all(result["gates"].values())
    assert result["scope"] == {
        "real_llm": False,
        "judge": False,
        "dev34": False,
        "frozen": False,
        "isolated_variable": "enable_llm_preference_resolver",
    }
    assert all(item["same_input"] for item in result["comparisons"])
    assert sum(
        item["target_ranked_first_delta"] > 0
        for item in result["comparisons"]
    ) == 3
    assert sum(
        item["target_selected_first_delta"] > 0
        for item in result["comparisons"]
    ) == 2
    budget = next(
        item for item in result["comparisons"]
        if item["scenario_id"] == "budget_hard_gate"
    )
    assert budget["hard_invalid_selected_delta"] == -1

    treatments = [
        cell for cell in result["cells"] if cell["variant"] == "actuation_on"
    ]
    assert all(cell["validation_passed"] is True for cell in treatments)
    assert all(cell["critic_passed"] is True for cell in treatments)
    assert all(cell["artifact_status"] == "candidate" for cell in treatments)
    assert all(cell["fixture_policy_calls"] == 1 for cell in treatments)
    assert all(cell["real_llm_calls"] == 0 for cell in treatments)
    assert all(cell["judge_calls"] == 0 for cell in treatments)
