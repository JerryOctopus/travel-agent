from scripts.run_eval import run_eval
from scripts.eval_agent import (
    _configure_offline_eval,
    run_nl_eval,
)


def test_run_eval_reports_basic_metrics() -> None:
    summary = run_eval(
        [
            {
                "id": "beijing_case",
                "query": "帮我规划北京两天，喜欢历史和美食，不要太累",
                "expected_city": "北京",
                "expected_days": 2,
                "required_interests": ["history", "food"],
            },
            {
                "id": "missing_case",
                "query": "我想轻松一点，喜欢自然和美食",
                "expected_clarification": True,
            },
        ]
    )

    assert summary["metrics"]["case_count"] == 2
    assert summary["metrics"]["city_accuracy"] == 1.0
    assert summary["metrics"]["days_accuracy"] == 1.0
    assert summary["metrics"]["clarification_accuracy"] == 1.0


def test_eval_agent_offline_nl_metrics() -> None:
    cases = [
        {
            "id": "beijing_case",
            "query": "帮我规划北京两天，喜欢历史和美食，不要太累",
            "expected_city": "北京",
            "expected_days": 2,
        }
    ]

    summary = run_nl_eval(cases, _configure_offline_eval())

    assert summary["task_completion_rate"] == 1.0
    assert summary["city_accuracy"] == 1.0
