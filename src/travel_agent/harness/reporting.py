from __future__ import annotations

from typing import Any

from travel_agent.harness.cases import HarnessCase
from travel_agent.harness.result import HarnessCaseResult


def case_result_to_nl_row(
    case: HarnessCase,
    result: HarnessCaseResult,
) -> dict[str, Any]:
    """Convert a harness result into the legacy NL eval row shape."""
    last = result.last_turn
    metrics = result.metrics
    itinerary = result.final_artifacts.get("itinerary")
    produced = itinerary is not None
    expected_clarification = bool(case.expect_clarification)
    expect_no_itinerary = case.expect_itinerary is False
    tool_trace = list(last.tool_trace if last else [])
    tool_ok = (
        True
        if not case.expected_tools
        else all(tool in tool_trace for tool in case.expected_tools)
    )
    no_itin_ok = (not produced) if expect_no_itinerary else None
    gate_ok = None
    if expect_no_itinerary or case.expected_tools:
        gate_ok = (no_itin_ok is not False) and tool_ok
    return {
        "id": case.case_id,
        "city_ok": metrics.get("city_ok"),
        "days_ok": metrics.get("days_ok"),
        "clarification_ok": (
            last.clarification == expected_clarification if last else False
        ),
        "itinerary_produced": (
            None if expected_clarification or expect_no_itinerary else produced
        ),
        "gate_ok": gate_ok,
        "critic_passed": metrics.get("critic_passed"),
        "delivered": produced if not expect_no_itinerary else None,
        "environment_pass": metrics.get("environment_pass"),
        "constraint_pass": metrics.get("constraint_pass"),
        "preference_pass": metrics.get("preference_pass"),
        "china_travel_style_final_pass": metrics.get("final_pass"),
        "meal_time_valid": metrics.get("meal_time_valid"),
        "route_feasible": metrics.get("route_feasible"),
        "daily_load_valid": metrics.get("daily_load_valid"),
        "poi_city_valid": metrics.get("poi_city_valid"),
        "interest_coverage_valid": metrics.get("interest_coverage_valid"),
        "plan_eval_issues": list(metrics.get("plan_eval_issues") or []),
        "tool_steps": len(tool_trace),
        "tool_coverage_ok": tool_ok if case.expected_tools else None,
        "error": "; ".join(result.errors) or None,
    }


def aggregate_case_results(
    results: list[HarnessCaseResult],
    cases: list[HarnessCase] | None = None,
) -> dict[str, Any]:
    """Aggregate harness case results using the legacy NL eval summary keys."""
    if cases is None:
        raise ValueError("cases are required to aggregate NL harness results.")
    rows = [
        case_result_to_nl_row(case, result)
        for case, result in zip(cases, results, strict=True)
    ]
    completable = [r for r in rows if r["itinerary_produced"] is not None]
    gate_rows = [r for r in rows if r["gate_ok"] is not None]
    delivered_rows = [r for r in rows if r["itinerary_produced"] is True]
    env_pass_rows = [r for r in delivered_rows if r.get("environment_pass") is True]
    return {
        "sample_size": len(rows),
        "city_accuracy": _rate(rows, "city_ok"),
        "days_accuracy": _rate(rows, "days_ok"),
        "clarification_accuracy": _rate(rows, "clarification_ok"),
        "intent_gate_accuracy": _rate(gate_rows, "gate_ok") if gate_rows else None,
        "task_completion_rate": _rate(completable, "itinerary_produced"),
        "delivery_rate": _rate(completable, "itinerary_produced"),
        "critic_pass_rate": _rate(rows, "critic_passed"),
        "environment_pass_rate": _rate(rows, "environment_pass"),
        "epr": _rate(delivered_rows, "environment_pass"),
        "constraint_pass_rate": _rate(rows, "constraint_pass"),
        "lpr": _rate(delivered_rows, "constraint_pass"),
        "preference_pass_rate": _rate(rows, "preference_pass"),
        "china_travel_style_final_pass_rate": _rate(
            rows,
            "china_travel_style_final_pass",
        ),
        "fpr": _rate(delivered_rows, "china_travel_style_final_pass"),
        "conditional_logical_pass_rate": _rate(env_pass_rows, "constraint_pass"),
        "meal_time_validity": _rate(rows, "meal_time_valid"),
        "route_feasibility_rate": _rate(rows, "route_feasible"),
        "daily_load_validity": _rate(rows, "daily_load_valid"),
        "poi_city_validity": _rate(rows, "poi_city_valid"),
        "tool_coverage_rate": _rate(rows, "tool_coverage_ok"),
        "avg_tool_steps": _avg_count(rows, "tool_steps", ndigits=2),
        "cases": rows,
    }


def aggregate_real_multi_agent_results(
    results: list[HarnessCaseResult],
    cases: list[HarnessCase],
    *,
    allowed_by_agent: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Aggregate real multi-agent (agent_trace) diagnostics from harness results."""
    rows = [
        _real_multi_agent_row(case, result, allowed_by_agent or {})
        for case, result in zip(cases, results, strict=True)
    ]
    return {
        "sample_size": len(rows),
        "task_completion_rate": _rate(rows, "itinerary_produced"),
        "critic_pass_rate": _rate(rows, "critic_passed"),
        "tool_coverage_rate": _rate(rows, "required_tools_ok"),
        "avg_tool_steps": _avg_count(rows, "tool_steps", ndigits=2),
        "agent_completed_rate": _rate(rows, "required_agents_present"),
        "used_real_agent_rate": _rate(rows, "used_real_agent"),
        "agent_trace_rate": _rate(rows, "agent_trace_present"),
        "research_agent_tool_rate": _rate(rows, "research_tools_ok"),
        "planning_agent_tool_rate": _rate(rows, "planning_tools_ok"),
        "agent_tool_whitelist_rate": _rate(rows, "agent_tool_whitelist_ok"),
        "external_llm_error_rate": _rate(rows, "external_llm_error"),
        "final_pass_rate": _rate(rows, "final_pass"),
        "full_multi_agent_pass_rate": _rate(rows, "full_multi_agent_pass"),
        "cases": rows,
    }


def build_harness_report(summary: dict[str, Any]) -> str:
    """Build a compact Markdown report for harness-only NL eval output."""
    return "\n".join(
        [
            "# Harness NL Eval Report",
            "",
            f"- sample_size: {summary.get('sample_size')}",
            f"- city_accuracy: {summary.get('city_accuracy')}",
            f"- days_accuracy: {summary.get('days_accuracy')}",
            f"- clarification_accuracy: {summary.get('clarification_accuracy')}",
            f"- task_completion_rate: {summary.get('task_completion_rate')}",
            f"- critic_pass_rate: {summary.get('critic_pass_rate')}",
            f"- final_pass_rate: {summary.get('fpr')}",
            f"- avg_tool_steps: {summary.get('avg_tool_steps')}",
        ]
    )


def build_release_report(summary: dict[str, Any]) -> str:
    gates = summary.get("gates") or {}
    blockers = list(summary.get("blockers") or [])
    status = summary.get("status") or "unknown"
    lines = [
        "# Agent Release Evaluation Report",
        "",
        f"- status: `{status}`",
        f"- suites: {', '.join(summary.get('suites') or [])}",
        "",
        "## Gates",
        "",
        "| gate | passed | detail |",
        "| --- | --- | --- |",
    ]
    for name, gate in gates.items():
        lines.append(
            f"| {name} | {gate.get('passed')} | {gate.get('detail') or ''} |"
        )
    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def evaluate_release_gates(results: dict[str, Any]) -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []

    agent_nl = results.get("agent-nl")
    if agent_nl:
        gates["agent-nl"] = _gate(
            all(
                agent_nl.get(key) == 1.0
                for key in (
                    "task_completion_rate",
                    "clarification_accuracy",
                    "critic_pass_rate",
                    "fpr",
                )
            ),
            "task/clarification/critic/fpr must be 1.0",
        )

    agent_real = results.get("agent-real")
    if agent_real:
        gates["agent-real"] = _gate(
            all(
                agent_real.get(key) == expected
                for key, expected in (
                    ("external_llm_error_rate", 0.0),
                    ("used_real_agent_rate", 1.0),
                    ("agent_trace_rate", 1.0),
                    ("full_multi_agent_pass_rate", 1.0),
                )
            ),
            "real LLM full multi-agent pass required",
        )

    agent_product = results.get("agent-product")
    if agent_product:
        product_metrics = agent_product.get("metrics") or {}
        product_artifacts = agent_product.get("artifacts") or {}
        gates["agent-product"] = _gate(
            bool(product_artifacts.get("model"))
            and bool(product_artifacts.get("model_provider"))
            and product_metrics.get("sample_size", 0) > 0,
            "Product suite must record one fixed model/provider and contain samples",
        )

    long_horizon = results.get("agent-long-horizon")
    if long_horizon:
        long_metrics = long_horizon.get("metrics") or {}
        long_artifacts = long_horizon.get("artifacts") or {}
        gates["agent-long-horizon"] = _gate(
            long_artifacts.get("independent_case_count", 0) > 0
            and long_metrics.get("turn_state_pass_rate") == 1.0,
            "Long-horizon suite requires every per-turn state checkpoint to pass",
        )

    chinatravel = {
        key: value for key, value in results.items()
        if key.startswith("chinatravel-")
    }
    for name, payload in chinatravel.items():
        metrics = payload.get("metrics") if isinstance(payload, dict) else payload
        gates[name] = _gate(
            metrics.get("delivery_rate") == 1.0
            and metrics.get("schema_pass_rate") == 100.0
            and (metrics.get("fpr") is None or metrics.get("fpr") > 0),
            "delivery=1.0, schema=100.0, fpr>0 when available",
        )

    live_tools = {
        key: value for key, value in results.items()
        if key.startswith("live-tools-")
    }
    for name, metrics in live_tools.items():
        gates[name] = _gate(
            metrics.get("api_success_rate") == 1.0
            and metrics.get("schema_valid_rate") == 1.0
            and metrics.get("timeout_rate") == 0.0
            and metrics.get("unexpected_route_unavailable_rate") == 0.0
            and metrics.get("duplicate_poi_rate") == 0.0,
            "api/schema success, no timeout/unexpected route/duplicate POI",
        )

    for name, gate in gates.items():
        if not gate["passed"]:
            blockers.append(f"{name}: {gate['detail']}")
    status = "pass" if gates and not blockers else "blocked"
    return {
        "status": status,
        "suites": list(results.keys()),
        "gates": gates,
        "blockers": blockers,
        "results": results,
    }


def _gate(passed: bool, detail: str) -> dict[str, Any]:
    return {"passed": bool(passed), "detail": detail}


def _real_multi_agent_row(
    case: HarnessCase,
    result: HarnessCaseResult,
    allowed_by_agent: dict[str, set[str]],
) -> dict[str, Any]:
    last = result.last_turn
    artifacts = result.final_artifacts
    itinerary = artifacts.get("itinerary")
    trace_payload = artifacts.get("agent_trace") or {}
    trace_items = list(trace_payload.get("items") or [])
    subagent_items = [item for item in trace_items if item.get("kind") == "subagent"]
    agent_names = [item.get("agent") for item in subagent_items]
    tools_by_agent: dict[str, list[str]] = {}
    for item in subagent_items:
        name = str(item.get("agent"))
        tools = list((item.get("detail") or {}).get("tool_trace") or [])
        tools_by_agent.setdefault(name, []).extend(tools)
    failed_agent_reasons = [
        {
            "agent": str(item.get("agent")),
            "status": item.get("status"),
            "reason": str(item.get("error") or ""),
        }
        for item in subagent_items
        if item.get("status") not in ("completed", "completed_with_warnings")
    ]
    failure_text = "\n".join(
        item["reason"] for item in failed_agent_reasons + [{"reason": "; ".join(result.errors)}]
    )
    flat_agent_tools = [
        tool
        for tools in tools_by_agent.values()
        for tool in tools
    ]
    all_tools = list((last.tool_trace if last else []) or flat_agent_tools)
    from travel_agent.agent.turn_analysis import classify_task_type_rule_based
    from travel_agent.orchestration.multi_agent.dispatch_rules import resolve_task_batches

    task_type = classify_task_type_rule_based(case.turns[-1]) if case.turns else None
    batches = resolve_task_batches(task_type) or ()
    required_agents = list(dict.fromkeys(agent for batch in batches for agent in batch))
    required_tools = ["search_poi", "recommend_candidates", "plan_and_critique"]
    disallowed_tools_by_agent = {
        agent: [
            tool
            for tool in tools
            if tool not in allowed_by_agent.get(agent, set())
        ]
        for agent, tools in tools_by_agent.items()
    }
    row = {
        "id": case.case_id,
        "used_real_agent": bool(last.used_real_agent if last else False),
        "itinerary_produced": itinerary is not None,
        "agent_trace_present": bool(trace_items),
        "required_agents_present": all(agent in agent_names for agent in required_agents),
        "required_agents": required_agents,
        "research_tools_ok": any(
            tool in flat_agent_tools
            for tool in ("search_poi", "check_weather", "plan_route")
        ),
        "planning_tools_ok": any(
            tool in flat_agent_tools
            for tool in ("recommend_candidates", "plan_and_critique")
        ),
        "required_tools_ok": all(tool in all_tools for tool in required_tools),
        "critic_passed": result.metrics.get("critic_passed"),
        "environment_pass": result.metrics.get("environment_pass"),
        "constraint_pass": result.metrics.get("constraint_pass"),
        "preference_pass": result.metrics.get("preference_pass"),
        "final_pass": result.metrics.get("final_pass"),
        "tool_steps": len(all_tools),
        "agent_names": agent_names,
        "tools_by_agent": tools_by_agent,
        "failed_agent_reasons": failed_agent_reasons,
        "external_llm_error": _is_llm_quota_or_auth_error(failure_text),
        "disallowed_tools_by_agent": disallowed_tools_by_agent,
        "tool_trace": all_tools,
        "issues": list(result.metrics.get("plan_eval_issues") or []),
        "errors": list(result.errors),
    }
    row["agent_tool_whitelist_ok"] = not any(disallowed_tools_by_agent.values())
    row["full_multi_agent_pass"] = all(
        bool(row[key])
        for key in (
            "used_real_agent",
            "itinerary_produced",
            "agent_trace_present",
            "required_agents_present",
            "agent_tool_whitelist_ok",
            "research_tools_ok",
            "planning_tools_ok",
            "required_tools_ok",
            "final_pass",
        )
    )
    return row


def _is_llm_quota_or_auth_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "allocationquota",
            "free quota",
            "free tier",
            "quota",
            "403",
            "unauthorized",
            "invalid api key",
        )
    )


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def _avg_count(rows: list[dict[str, Any]], key: str, ndigits: int = 4) -> float | None:
    if not rows:
        return None
    return round(sum(r.get(key, 0) for r in rows) / len(rows), ndigits)
