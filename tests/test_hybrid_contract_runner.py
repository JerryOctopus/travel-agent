from __future__ import annotations

from scripts.run_hybrid_contract import run_contract_suite


def test_contract_runner_is_offline_by_default_and_covers_twelve_scenarios() -> None:
    result = run_contract_suite(live=False)

    assert result["mode"] == "offline_no_network"
    assert result["summary"]["scenario_count"] == 12
    assert result["summary"]["business_calls"] == 0
    assert result["summary"]["total_model_calls"] == 0
    assert result["summary"]["limits_respected"] is True
    assert {item["module"] for item in result["scenarios"]} == {"intent", "preference"}
    assert all("input" in item and "raw_model_outputs" in item for item in result["scenarios"])


def test_supplemental_intent_probes_remain_offline_without_live_flag() -> None:
    result = run_contract_suite(live=False, supplemental_intent_probes=True)

    assert result["mode"] == "offline_no_network"
    assert result["summary"]["scenario_count"] == 5
    assert result["summary"]["business_calls"] == 0
    assert all(item["module"] == "intent" for item in result["scenarios"])
