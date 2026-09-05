from __future__ import annotations

from travel_agent.evaluation.artifact_contract import (
    actual_artifact_type,
    aggregate_artifact_metrics,
    artifact_type_matches,
    effective_required_tools,
    expected_artifact_type,
    itinerary_judge_route,
)
from travel_agent.agent.turn_analysis import classify_task_type_rule_based
from travel_agent.harness.cases import HarnessCase


def _output(message: str, *, reply: str, artifacts: dict | None = None, status: str = "completed") -> dict:
    return {
        "case": {"turns": [message], "gold_outcome": "full_plan"},
        "turns": [{"reply_text": reply, "status": status, "tool_trace": []}],
        "final_artifacts": artifacts or {},
    }


def test_legacy_gold_does_not_override_non_itinerary_task_scope() -> None:
    examples = (
        ("只看从机场到酒店的路线怎么走", "route_plan"),
        ("比较三个住宿区域，选一个", "candidate_comparison"),
        ("只找三个餐厅候选", "candidate_comparison"),
        ("住哪里方便，比较多个区域", "candidate_comparison"),
        ("不要重写整个行程，只判断第二天户外活动是否因下雨替换", "local_adjustment_advice"),
        ("仅规划从火车站到景区的公共交通路线", "route_plan"),
        ("把候选景点排序后推荐一个", "candidate_comparison"),
        ("从甲站去乙地，比较公共交通和打车", "route_plan"),
        ("从车站出发，比较去甲馆、乙园和丙滩三个地点的公共交通时间，选一个", "candidate_comparison"),
        ("在湖区附近找3家适合聚餐的餐厅", "candidate_comparison"),
        ("从市中心附近选4个不同类型的下午活动", "candidate_comparison"),
    )
    for message, expected in examples:
        case = HarnessCase(case_id="generic", turns=[message], gold_outcome="full_plan")
        assert expected_artifact_type(case) == expected
        assert classify_task_type_rule_based(message).value == expected


def test_explicit_artifact_type_round_trips_through_harness_schema() -> None:
    case = HarnessCase.from_dict(
        {
            "case_id": "new-schema",
            "turns": ["给我一个有限方案"],
            "expected_artifact_type": "partial_itinerary",
        }
    )
    assert case.expected_artifact_type == "partial_itinerary"
    assert expected_artifact_type(case) == "partial_itinerary"


def test_actual_artifact_is_identified_before_judge_selection() -> None:
    output = _output(
        "从机场到酒店怎么走",
        reply="路线：机场 → 酒店，公共交通 45 分钟；打车 30 分钟。",
        artifacts={
            "route_plan": {
                "origin": "机场",
                "destination": "酒店",
                "evidence": [{"artifact_id": "route-1"}],
                "limitations": [],
            }
        },
    )
    expected = expected_artifact_type(output["case"])
    actual = actual_artifact_type(output, expected)
    assert expected == actual == "route_plan"
    assert artifact_type_matches(expected, actual)
    assert itinerary_judge_route(expected, actual)["status"] == "not_applicable"


def test_expected_full_itinerary_missing_is_not_run_not_na() -> None:
    route = itinerary_judge_route("full_itinerary", None)
    assert route == {
        "status": "not_run",
        "rubric": None,
        "reason": "missing_expected_artifact",
    }


def test_partial_plan_uses_diagnostic_rubric_and_does_not_match_full() -> None:
    assert artifact_type_matches("full_itinerary", "partial_itinerary") is False
    assert itinerary_judge_route("full_itinerary", "partial_itinerary") == {
        "status": "pending",
        "rubric": "partial_itinerary",
        "diagnostic_only": True,
    }


def test_specialized_response_artifacts_match_their_own_evaluators() -> None:
    for expected in ("clarification", "constraint_negotiation", "safe_decline"):
        assert artifact_type_matches(expected, expected)
        assert itinerary_judge_route(expected, expected)["status"] == "not_applicable"


def test_specialized_tool_contract_does_not_inherit_full_planner_tools() -> None:
    legacy = ["poi_search", "route_planning", "plan_and_critique"]
    local = HarnessCase(
        case_id="local",
        turns=["只判断第二天户外活动是否因下雨替换，并给室内备选"],
        required_tools=legacy,
        hard_constraints={"need_indoor_backup": True},
        gold_outcome="full_plan",
    )
    route = HarnessCase(
        case_id="route",
        turns=["只看从机场到酒店的公共交通路线"],
        required_tools=legacy,
        gold_outcome="full_plan",
    )

    assert effective_required_tools(local) == ["check_weather", "search_poi"]
    assert effective_required_tools(route) == ["search_poi", "plan_route"]
    assert "plan_and_critique" not in effective_required_tools(local)


def test_aggregate_keeps_judge_averages_separate_by_rubric() -> None:
    outputs = [
        {
            "case": {"turns": ["完整行程"]},
            "evaluation": {
                "rule_metrics": {
                    "expected_artifact_type": "full_itinerary",
                    "actual_artifact_type": "full_itinerary",
                    "artifact_type_match": True,
                    "strict_task_success": True,
                },
                "independent_judge": {"status": "ok", "rubric": "full_itinerary", "total_score": 80},
            },
        },
        {
            "case": {"turns": ["有限方案"]},
            "evaluation": {
                "rule_metrics": {
                    "expected_artifact_type": "partial_itinerary",
                    "actual_artifact_type": "partial_itinerary",
                    "artifact_type_match": True,
                    "strict_task_success": True,
                },
                "independent_judge": {"status": "ok", "rubric": "partial_itinerary", "total_score": 40},
            },
        },
    ]
    metrics = aggregate_artifact_metrics(outputs)
    assert metrics["itinerary_judge_average_by_rubric"] == {
        "full_itinerary": 80.0,
        "partial_itinerary": 40.0,
    }
    assert "itinerary_judge_average" not in metrics


def test_not_run_judge_is_applicable_but_not_completed() -> None:
    output = {
        "case": {"turns": ["规划完整行程"]},
        "evaluation": {
            "rule_metrics": {
                "expected_artifact_type": "full_itinerary",
                "actual_artifact_type": None,
                "artifact_type_match": False,
                "strict_task_success": False,
            },
            "independent_judge": {"status": "not_run", "reason": "missing_expected_artifact"},
        },
    }
    metrics = aggregate_artifact_metrics([output])
    assert metrics["itinerary_judge_applicable_count"] == 1
    assert metrics["itinerary_judge_completed_count"] == 0
    assert metrics["itinerary_judge_completion_rate"] == 0.0
