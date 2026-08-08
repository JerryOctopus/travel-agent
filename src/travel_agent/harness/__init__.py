"""Reusable harness primitives for running and evaluating the travel agent."""

from travel_agent.harness.cases import HarnessCase, load_cases_json
from travel_agent.harness.environments import HarnessEnvironment
from travel_agent.harness.result import HarnessCaseResult, HarnessTurnResult
from travel_agent.harness.result import (
    HarnessBenchmarkResult,
    HarnessOperationResult,
    HarnessSuiteResult,
)
from travel_agent.harness.runner import AgentHarness
from travel_agent.harness.preflight import preflight_llm
from travel_agent.harness.validators import validate_case_result
from travel_agent.harness.reporting import (
    aggregate_case_results,
    aggregate_real_multi_agent_results,
    build_harness_report,
)
from travel_agent.harness.product import (
    aggregate_product_results,
    build_product_report,
    decide_fine_tuning,
    run_product_suite,
    validate_product_dataset,
    wilson_interval,
    write_product_run,
)

__all__ = [
    "AgentHarness",
    "HarnessCase",
    "HarnessCaseResult",
    "HarnessBenchmarkResult",
    "HarnessEnvironment",
    "HarnessOperationResult",
    "HarnessSuiteResult",
    "HarnessTurnResult",
    "aggregate_case_results",
    "aggregate_real_multi_agent_results",
    "build_harness_report",
    "load_cases_json",
    "preflight_llm",
    "validate_case_result",
    "aggregate_product_results",
    "build_product_report",
    "decide_fine_tuning",
    "run_product_suite",
    "validate_product_dataset",
    "wilson_interval",
    "write_product_run",
]
