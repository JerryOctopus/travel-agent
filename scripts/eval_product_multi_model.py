"""Run production_v1 (180 cases) using multiple LLMs with automatic relay on quota exhaustion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.harness import AgentHarness, HarnessEnvironment
from travel_agent.harness.cases import load_cases_json
from travel_agent.harness.preflight import preflight_llm
from travel_agent.harness.product import (
    DEFAULT_PRODUCT_CASES,
    PRODUCTION_DATASET_VERSION,
    _execution_case,
    _product_case_output,
    _product_row,
    _git_revision,
    _prompt_fingerprint,
    _sha256,
    aggregate_product_rows,
    validate_product_dataset,
    write_product_run,
)
from travel_agent.harness.result import HarnessSuiteResult
from travel_agent.settings import load_settings

REQUIRED_AGENTS = ["attraction", "hotel", "restaurant", "transport", "planner"]


DEFAULT_RELAY_MODELS = [
    "qwen3.7-plus-2026-05-26",
    "qwen3.8-max",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    "deepseek-v3.2-exp",
]

DEFAULT_BASE_URL_BY_PROVIDER = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
}

QUOTA_KEYWORDS = {
    "quota",
    "allocationquota",
    "free tier",
    "freetier",
    "rate limit",
    "too many requests",
    "out of quota",
    "quota exceeded",
    "insufficient balance",
    "billing limit",
}

UNAVAILABLE_KEYWORDS = {
    "model not found",
    "unsupported model",
    "does not exist",
    "not found",
    "is not enabled",
}


def parse_models(entries: list[str], default_provider: str) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in entries:
        value = raw.strip()
        if not value:
            continue
        if ":" in value:
            provider, model = value.split(":", 1)
            provider = provider.strip().lower()
            model = model.strip()
        elif "/" in value:
            provider, model = value.split("/", 1)
            provider = provider.strip().lower()
            model = model.strip()
        else:
            provider = default_provider
            model = value
        key = f"{provider}:{model}"
        if key not in seen:
            seen.add(key)
            models.append((provider, model))
    return models


def pick_api_key(
    provider: str,
    override: str | None,
    configured_provider: str,
    fallback: str | None,
) -> str | None:
    if override:
        return override
    provider_key = os.getenv(f"TRAVEL_AGENT_{provider.upper()}_API_KEY")
    if provider_key:
        return provider_key
    # Never send one provider's credential to another provider's endpoint.
    return fallback if provider == configured_provider else None


def pick_base_url(provider: str, override: str | None, fallback: str | None) -> str:
    if override:
        return override
    return DEFAULT_BASE_URL_BY_PROVIDER.get(provider, fallback or "")


def is_quota_error(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in QUOTA_KEYWORDS)


def is_unavailable_error(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in UNAVAILABLE_KEYWORDS)


def is_auth_error(text: str, status: object) -> bool:
    lower = text.lower()
    if status in (401, 403):
        return True
    if "401" in lower or "403" in lower:
        return True
    return any(marker in lower for marker in ("invalid api key", "authentication", "unauthorized"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production_v1 (180 cases) with multi-model relay."
    )
    parser.add_argument("--product-cases", default=str(DEFAULT_PRODUCT_CASES))
    parser.add_argument(
        "--product-split",
        choices=["dev", "core_frozen", "challenge_frozen", "shadow_frozen", "all"],
        default="all",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model in order. Supports provider:model or model.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Default provider for model-only entries.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL override for all relay models.",
    )
    parser.add_argument("--api-key", default=None, help="API key override for all relay models.")
    parser.add_argument(
        "--max-model-tokens",
        type=int,
        default=900000,
        help="Token cap per model; switch when reaching this local threshold.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "eval" / "product"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--remediation-complete", action="store_true")
    parser.add_argument(
        "--preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preflight all models before running.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and resolve the relay pool, then exit without running cases.",
    )
    return parser


def build_state_template(
    run_id: str,
    model_specs: list[tuple[str, str]],
    base_url_override: str | None,
    fallback_base_url: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": 0,
        "split": "all",
        "limit": None,
        "models": [
            {
                "provider": provider,
                "model": model,
                "base_url": pick_base_url(provider, base_url_override, fallback_base_url),
            }
            for provider, model in model_specs
        ],
        "current_model_index": 0,
        "completed_case_ids": [],
        "attempts_by_case": {},
        "attempts_total": 0,
        "model_usage": {
            f"{provider}:{model}": {"tokens": 0, "cases": 0}
            for provider, model in model_specs
        },
        "model_unavailable": [],
        "model_switch_log": [],
        "rows": [],
    }


def multi_agent_fields(result) -> dict[str, Any]:
    """Strict proof that all five real LLM Subagents completed for this execution."""
    payload = result.final_artifacts.get("agent_trace") or {}
    items = list(payload.get("items") or [])
    successful_agents = {
        str(item.get("agent"))
        for item in items
        if item.get("kind") == "subagent"
        and item.get("status") in ("completed", "completed_with_warnings")
    }
    missing_agents = [agent for agent in REQUIRED_AGENTS if agent not in successful_agents]
    last = result.last_turn
    real_delivery = bool(last and last.used_real_agent)
    itinerary_produced = result.final_artifacts.get("itinerary") is not None
    return {
        "multi_agent_mode": True,
        "multi_agent_required_agents": list(REQUIRED_AGENTS),
        "multi_agent_successful_agents": sorted(successful_agents),
        "multi_agent_missing_agents": missing_agents,
        "multi_agent_all_agents_completed": not missing_agents,
        "multi_agent_strict_success": bool(
            real_delivery and itinerary_produced and not missing_agents
        ),
    }


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def build_failed_row(case, reason: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "split": case.split,
        "category": case.category,
        "subset": case.subset or case.category,
        "gold_outcome": case.gold_outcome,
        "high_risk": case.high_risk,
        "repeat": 1,
        "passed": False,
        "strict_task_success": False,
        "actual_outcome": "system_error",
        "expected_outcome_match": False,
        "gating_passed": True,
        "gating_triggered": [],
        "grounding_ok": None,
        "authorization_ok": None,
        "architecture_policy_ok": None,
        "constraint_tree_score": None,
        "task_completed": None,
        "hard_constraints_ok": None,
        "soft_preferences_ok": None,
        "critic_passed": None,
        "clarification_ok": None,
        "tool_schema_valid": None,
        "fault_recovery_ok": None,
        "memory_ok": None,
        "tool_required_count": 0,
        "tool_required_hit_count": 0,
        "tool_selected_count": 0,
        "tool_unexpected_count": 0,
        "tool_argument_assertion_count": 0,
        "tool_argument_pass_count": 0,
        "tool_trace": [],
        "duration_ms": None,
        "total_tokens": None,
        "model_call_count": 0,
        "model_error_call_count": 0,
        "model_call_errors": [reason],
        "tool_call_count": 0,
        "failure_attributions": ["external_api"],
        "errors": [reason],
        "multi_agent_mode": True,
        "multi_agent_required_agents": list(REQUIRED_AGENTS),
        "multi_agent_successful_agents": [],
        "multi_agent_missing_agents": list(REQUIRED_AGENTS),
        "multi_agent_all_agents_completed": False,
        "multi_agent_strict_success": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    default_provider = (args.provider or settings.llm.provider).lower()
    configured_provider = settings.llm.provider.lower()

    # This entrypoint is exclusively for a real Multi-Agent Full (V3) run;
    # production orchestration is fixed and not configurable here.

    model_specs = parse_models(args.model or DEFAULT_RELAY_MODELS, default_provider)
    if not model_specs:
        raise ValueError("No model specified.")

    cases = load_cases_json(args.product_cases)
    validation = validate_product_dataset(cases)
    if not validation.valid:
        raise RuntimeError("invalid production_v1 dataset: " + "; ".join(validation.errors))
    selected_cases = [case for case in cases if args.product_split == "all" or case.split == args.product_split]
    if args.limit is not None:
        selected_cases = selected_cases[: max(1, args.limit)]

    output_root = Path(args.output_root)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.run_id:
        run_dir = output_root / "runs" / run_id
        if not args.resume and run_dir.exists():
            raise RuntimeError(f"run_id already exists: {run_id}")
    else:
        base_run_id = run_id
        run_dir = output_root / "runs" / run_id
        suffix = 1
        while run_dir.exists():
            run_id = f"{base_run_id}-{suffix}"
            run_dir = output_root / "runs" / run_id
            suffix += 1
    state_path = run_dir / "relay_state.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_url = str(settings.llm.base_url)
    state = load_state(state_path)
    if not state:
        state = build_state_template(run_id, model_specs, args.base_url, base_url)
        state["case_count"] = len(selected_cases)
        state["split"] = args.product_split
        state["limit"] = args.limit
    else:
        if args.resume:
            state["run_id"] = run_id
            state["case_count"] = len(selected_cases)
            state["split"] = args.product_split
            state["limit"] = args.limit
        else:
            raise RuntimeError("run already exists. Use --resume or choose different --run-id.")

    if args.preflight and not args.resume:
        print("Preflighting model list...")
        usable: list[tuple[str, str, str]] = []
        for provider, model in model_specs:
            api_key = pick_api_key(
                provider,
                args.api_key,
                configured_provider,
                settings.llm.api_key,
            )
            if not api_key:
                detail = {
                    "ok": False,
                    "provider": provider,
                    "model": model,
                    "detail": f"missing TRAVEL_AGENT_{provider.upper()}_API_KEY",
                }
                print(f"[preflight] {provider}:{model} -> fail missing provider API key")
                state["model_unavailable"].append(
                    {"provider": provider, "model": model, "detail": detail}
                )
                continue
            model_settings = settings
            model_settings = replace(
                model_settings,
                llm=replace(
                    model_settings.llm,
                    provider=provider,
                    model=model,
                    base_url=pick_base_url(provider, args.base_url, base_url),
                    api_key=api_key,
                ),
            )
            result = preflight_llm(model_settings)
            print(
                f"[preflight] {provider}:{model} -> "
                f"{'ok' if result.get('ok') else 'fail'} "
                f"{result.get('status', '')} {result.get('detail', '').strip()}"
            )
            detail = str(result.get("detail", ""))
            status = result.get("status")
            if result.get("ok"):
                usable.append((provider, model, pick_base_url(provider, args.base_url, base_url)))
            elif is_quota_error(detail):
                state["model_unavailable"].append(
                    {
                        "provider": provider,
                        "model": model,
                        "reason": "quota_exhausted",
                        "detail": result,
                    }
                )
            elif is_unavailable_error(detail) or is_auth_error(detail, status):
                state["model_unavailable"].append(
                    {"provider": provider, "model": model, "detail": result}
                )
            else:
                usable.append((provider, model, pick_base_url(provider, args.base_url, base_url)))
        if not usable:
            raise RuntimeError("No usable models after preflight.")
        model_specs = [(provider, model) for provider, model, _ in usable]
        state["models"] = [
            {"provider": provider, "model": model, "base_url": url}
            for provider, model, url in usable
        ]
        if not state["models"]:
            state["models"] = [
                    {
                        "provider": provider,
                        "model": model,
                        "base_url": pick_base_url(provider, args.base_url, base_url),
                    }
                    for provider, model in model_specs
                ]
        state["model_usage"] = {
            f"{provider}:{model}": {"tokens": 0, "cases": 0}
            for provider, model in model_specs
        }
        if "current_model_index" not in state:
            state["current_model_index"] = 0
        if "attempts_total" not in state:
            state["attempts_total"] = 0
    else:
        if not state["models"]:
            state["models"] = [
                {
                    "provider": provider,
                    "model": model,
                    "base_url": pick_base_url(provider, args.base_url, base_url),
                }
                for provider, model in model_specs
            ]
        if not state["model_usage"]:
            state["model_usage"] = {
                f"{provider}:{model}": {"tokens": 0, "cases": 0}
                for provider, model in model_specs
            }

    if args.preflight_only:
        save_state(state_path, state)
        payload = {
            "run_id": run_id,
            "real_multi_agent": True,
            "usable_models": state["models"],
            "unavailable_models": state["model_unavailable"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    model_specs = [(item["provider"], item["model"]) for item in state["models"]]
    if not model_specs:
        raise RuntimeError("No usable model to execute with.")

    environment = HarnessEnvironment(
        mode="real_agent",
        persist=False,
        user_id="product_eval",
    )

    harnesses: dict[tuple[str, str], AgentHarness] = {}
    for provider, model in model_specs:
        harnesses[(provider, model)] = AgentHarness(
            settings=replace(
                settings,
                llm=replace(
                    settings.llm,
                    provider=provider,
                    model=model,
                    base_url=next(
                        item["base_url"]
                        for item in state["models"]
                        if item["provider"] == provider and item["model"] == model
                    ),
                    api_key=pick_api_key(
                        provider,
                        args.api_key,
                        configured_provider,
                        settings.llm.api_key,
                    ),
                ),
            ),
            environment=environment,
        )

    completed_cases = set(state.get("completed_case_ids", []))
    rows = [row for row in state.get("rows", []) if isinstance(row, dict)]
    case_outputs = []
    attempts_total = int(state.get("attempts_total", 0))
    model_usage = {
        key: {"tokens": int(value.get("tokens", 0)), "cases": int(value.get("cases", 0))}
        for key, value in state.get("model_usage", {}).items()
    }
    current_model_index = int(state.get("current_model_index", 0))

    def append_case_output(case_output: dict[str, Any]) -> None:
        case_outputs.append(case_output)
        state["rows"] = rows
        save_state(state_path, state)

    def try_models_for_case(case) -> None:
        nonlocal current_model_index, attempts_total
        local_switches = 0
        case_key = case.case_id
        last_reason: str | None = None
        for _ in range(len(model_specs) - current_model_index):
            provider, model = model_specs[current_model_index]
            model_key = f"{provider}:{model}"
            usage = model_usage.get(model_key, {"tokens": 0, "cases": 0})
            if args.max_model_tokens and usage["tokens"] >= args.max_model_tokens:
                print(f"[switch] {model_key} token cap reached, switching.")
                state["model_switch_log"].append(
                    {
                        "case_id": case.case_id,
                        "from": model_key,
                        "reason": "local_token_cap",
                    }
                )
                current_model_index += 1
                state["current_model_index"] = current_model_index
                save_state(state_path, state)
                local_switches += 1
                last_reason = "local token cap reached"
                continue

            harness = harnesses[(provider, model)]
            started = time.time()
            result = harness.run_case(_execution_case(case, 1))
            row = _product_row(case, result, 1)
            row.update(multi_agent_fields(result))
            attempts_total += 1
            state["attempts_total"] = attempts_total

            row["runtime_model_provider"] = provider
            row["runtime_model"] = model
            row["runtime_model_base_url"] = str(harness.settings.llm.base_url)
            row["runtime_model_index"] = current_model_index
            row["runtime_model_switches"] = local_switches
            row["runtime_model_latency_ms"] = round((time.time() - started) * 1000, 2)
            usage["cases"] += 1
            usage["tokens"] += row.get("total_tokens") or 0
            model_usage[model_key] = usage
            state["model_usage"] = model_usage

            state["attempts_by_case"][case_key] = state["attempts_by_case"].get(case_key, 0) + 1

            errors = [item for item in row.get("model_call_errors", [])]
            if errors:
                all_error = " ".join(str(item) for item in errors).lower()
                if is_quota_error(all_error):
                    print(f"[quota] {case.case_id} hits quota on {model_key}, switch next.")
                    last_reason = f"quota exhausted on {model_key}"
                    state["model_switch_log"].append(
                        {
                            "case_id": case.case_id,
                            "from": model_key,
                            "reason": "quota_exhausted",
                        }
                    )
                    current_model_index += 1
                    state["current_model_index"] = current_model_index
                    save_state(state_path, state)
                    local_switches += 1
                    continue
                if is_unavailable_error(all_error):
                    print(f"[unavailable] {case.case_id} model {model_key} unavailable, switch.")
                    last_reason = f"model unavailable: {model_key}"
                    state["model_switch_log"].append(
                        {
                            "case_id": case.case_id,
                            "from": model_key,
                            "reason": "model_unavailable",
                        }
                    )
                    current_model_index += 1
                    state["current_model_index"] = current_model_index
                    save_state(state_path, state)
                    local_switches += 1
                    continue
                if is_auth_error(all_error, None):
                    print(f"[auth] {case.case_id} model {model_key} unauthorized, switch.")
                    last_reason = f"model unauthorized: {model_key}"
                    state["model_switch_log"].append(
                        {
                            "case_id": case.case_id,
                            "from": model_key,
                            "reason": "authentication",
                        }
                    )
                    current_model_index += 1
                    state["current_model_index"] = current_model_index
                    save_state(state_path, state)
                    local_switches += 1
                    continue

            output = _product_case_output(case, result, 1)
            output["execution"]["runtime_model"] = model
            output["execution"]["runtime_model_provider"] = provider
            output["execution"]["runtime_model_base_url"] = str(harness.settings.llm.base_url)
            output["execution"]["runtime_model_switches"] = local_switches
            output["execution"]["runtime_model_index"] = current_model_index
            append_case_output(output)
            row["runtime_model"] = model
            rows.append(row)
            completed_cases.add(case_key)
            state["completed_case_ids"] = sorted(completed_cases)
            state["rows"] = rows
            state["current_model_index"] = current_model_index
            save_state(state_path, state)
            return

        # if all remaining models fail on this case, write an explicit failure row
        failed_row = build_failed_row(case, last_reason or "all relay models unavailable")
        failed_row["runtime_model_provider"] = "quota-relay"
        failed_row["runtime_model"] = "n/a"
        failed_row["runtime_model_switches"] = local_switches
        rows.append(failed_row)
        output = _product_case_output(case, result, 1) if "result" in locals() else {
            "schema_version": "product-case-output-v1",
            "case": case.__dict__,
            "execution": {
                "repeat": 1,
                "passed": False,
                "errors": [failed_row["errors"]],
            },
            "turns": [],
            "final_profile": {},
            "final_artifacts": {},
            "final_itinerary": None,
            "evaluation": {
                "rule_metrics": None,
                "independent_judge": None,
                "human_review": None,
            },
        }
        output["execution"]["runtime_model"] = failed_row["runtime_model"]
        output["execution"]["runtime_model_provider"] = failed_row["runtime_model_provider"]
        output["execution"]["runtime_model_switches"] = local_switches
        append_case_output(output)
        completed_cases.add(case_key)
        state["completed_case_ids"] = sorted(completed_cases)
        state["rows"] = rows
        save_state(state_path, state)

    for index, case in enumerate(selected_cases, start=1):
        if case.case_id in completed_cases:
            print(f"[skip] {case.case_id} already done")
            continue
        print(f"[case {index}/{len(selected_cases)}] {case.case_id}")
        if current_model_index >= len(model_specs):
            failed_row = build_failed_row(
                case, f"all models used: {len(model_specs)}"
            )
            failed_row["runtime_model_provider"] = "quota-relay"
            failed_row["runtime_model"] = "n/a"
            rows.append(failed_row)
            output = {
                "schema_version": "product-case-output-v1",
                "case": case.__dict__,
                "execution": {
                    "repeat": 1,
                    "passed": False,
                    "errors": [failed_row["errors"]],
                    "runtime_model": failed_row["runtime_model"],
                    "runtime_model_provider": failed_row["runtime_model_provider"],
                },
                "turns": [],
                "final_profile": {},
                "final_artifacts": {},
                "final_itinerary": None,
                "evaluation": {
                    "rule_metrics": None,
                    "independent_judge": None,
                    "human_review": None,
                },
            }
            append_case_output(output)
            completed_cases.add(case.case_id)
            state["completed_case_ids"] = sorted(completed_cases)
            state["rows"] = rows
            continue
        try_models_for_case(case)

    summary = aggregate_product_rows(rows, remediation_complete=args.remediation_complete)
    summary["relay_mode"] = True
    summary["real_multi_agent"] = True
    strict_values = [row.get("multi_agent_strict_success") for row in rows]
    strict_values = [value for value in strict_values if isinstance(value, bool)]
    summary["multi_agent_strict_success_rate"] = (
        sum(strict_values) / len(strict_values) if strict_values else None
    )
    summary["model_attempts_by_case"] = state.get("attempts_by_case", {})
    summary["model_switch_log"] = state.get("model_switch_log", [])
    summary["model_switch_count"] = len(state.get("model_switch_log", []))

    suite_result = HarnessSuiteResult(
        suite="agent-product",
        mode="product",
        environment="real_agent",
        case_count=len(selected_cases),
        attempted_count=attempts_total,
        metrics={
            key: value for key, value in summary.items() if key not in {"cases"}
        },
        rows=rows,
        artifacts={
            "dataset_version": PRODUCTION_DATASET_VERSION,
            "dataset_sha256": _sha256(Path(args.product_cases)),
            "code_revision": _git_revision(),
            "model_provider": "quota-relay",
            "model": "quota-relay",
            "model_temperature": settings.llm.temperature,
            "model_thinking_enabled": settings.llm.thinking_enabled,
            "model_requests_per_second": settings.llm.requests_per_second,
            "prompt_fingerprint": _prompt_fingerprint(),
            "relay_models": state["models"],
            "real_multi_agent": True,
            "required_agents": list(REQUIRED_AGENTS),
            "relay_model_usage": model_usage,
            "relay_summary": {
                "attempts_total": attempts_total,
                "current_model_index": current_model_index,
            },
            "_case_outputs": case_outputs,
        },
        benchmarks=[],
        stopped_early=False,
        stop_reason=None,
    )

    if args.write_report:
        summary_report = write_product_run(suite_result, output_root, run_id=run_id)
    else:
        summary_report = {
            "metrics": suite_result.metrics,
            "artifacts": {key: value for key, value in suite_result.artifacts.items() if key != "_case_outputs"},
            "rows": rows,
        }

    state["attempts_total"] = attempts_total
    state["current_model_index"] = current_model_index
    state["rows"] = rows
    state["completed_case_ids"] = sorted(completed_cases)
    save_state(state_path, state)
    if args.json:
        print(json.dumps(summary_report, ensure_ascii=False, indent=2))
    else:
        print(f"run_id={run_id}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
