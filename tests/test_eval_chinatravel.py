from scripts.eval_chinatravel import _multi_agent_metrics, build_report


def test_multi_agent_metrics_reports_strict_valid_count() -> None:
    rows = [
        {
            "used_real_agent": True,
            "agent_trace_present": True,
            "required_agents_present": True,
            "itinerary_produced": True,
            "fallback_used": False,
            "external_llm_error": False,
        },
        {
            "used_real_agent": True,
            "agent_trace_present": True,
            "required_agents_present": False,
            "itinerary_produced": False,
            "fallback_used": True,
            "external_llm_error": False,
        },
    ]

    metrics = _multi_agent_metrics(rows)

    assert metrics["row_count"] == 2
    assert metrics["strict_valid_count"] == 1
    assert metrics["strict_valid_rate"] == 0.5


def test_chinatravel_report_separates_current_and_cumulative_agent_metrics() -> None:
    report = build_report(
        {
            "suite": "human154",
            "split": "human",
            "mode": "real_multi_agent",
            "llm_provider": "qwen",
            "llm_model": "qwen-test",
            "architecture": "multi_agent_full_v3",
            "case_count": 154,
            "attempted_case_count": 2,
            "evaluated_case_count": 1,
            "prediction_file_count": 10,
            "stopped_early": True,
            "stop_reason": "external_llm_error",
            "predictions_dir": "/tmp/predictions",
            "diagnostics_dir": "/tmp/diagnostics",
            "metrics": {
                "official_eval_available": True,
                "delivery_rate": 1.0,
                "schema_pass_rate": 100.0,
            },
            "multi_agent_metrics": {"row_count": 2, "strict_valid_count": 1},
            "cumulative_multi_agent_metrics": {
                "row_count": 154,
                "strict_valid_count": 65,
                "strict_valid_rate": 0.4221,
            },
            "rows": [],
        }
    )

    assert "真实 Multi-Agent 诊断（本次命令）" in report
    assert "真实 Multi-Agent 诊断（累计 diagnostics）" in report
    assert "prediction_file_count：10" in report
    assert "strict_valid_count：65" in report
