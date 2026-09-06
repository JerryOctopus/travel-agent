"""Run production_v1.1 (192 cases) using multiple LLMs with automatic relay."""

from __future__ import annotations

import argparse
import fcntl
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
    DEFAULT_PRODUCT_DEV_CASES,
    PRODUCTION_DATASET_VERSION,
    _execution_case,
    _product_case_output,
    _product_row,
    _git_revision,
    _prompt_fingerprint,
    _sha256,
    _artifact_contract_fingerprint,
    _code_fingerprint,
    _configuration_fingerprint,
    _evaluator_fingerprint,
    _evaluator_version,
    _tooling_fingerprint,
    _tool_snapshot_fingerprint,
    aggregate_product_rows,
    write_product_run,
)
from travel_agent.harness.result import HarnessSuiteResult
from travel_agent.settings import load_settings
from travel_agent.evaluation.frozen_release_acceptance import (
    FROZEN_RELEASE_SCHEMA_VERSION,
    artifact_contract_implementation_fingerprint,
    load_frozen_split_cases,
    load_release_manifest,
    reject_unofficial_frozen_request,
    validate_frozen_run_request,
    validate_runtime_against_manifest,
    validate_shadow_prerequisites,
)

ALL_AGENT_ROLES = ["attraction", "hotel", "restaurant", "transport", "planner"]


def acquire_run_lock(run_dir: Path):
    """Hold an exclusive process lock for one run-id until the handle closes."""
    lock_path = run_dir / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(
            f"run_id is already active (lock owner pid={owner}): {run_dir.name}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


DEFAULT_RELAY_MODELS = [
    "qwen-max",
    "qwen3.6-flash",
    "qwen3.7-flash-2026-07-15",
    "qwen-long",
    "qwen3.5-flash-2026-02-23",
    "qwen-plus",
    "glm:glm-5",
]

DEFAULT_BASE_URL_BY_PROVIDER = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "silicon-flow": "https://api.siliconflow.cn/v1",
    "silicon_flow": "https://api.siliconflow.cn/v1",
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


def relay_mode_enabled(models: list[Any]) -> bool:
    """A one-model execution uses the relay runner but is not model relay."""
    return len(models) > 1


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


def recorded_model_identity(models: list[dict[str, Any]]) -> tuple[str, str]:
    """Record the actual fixed model unless a run truly contains a relay pool."""
    if len(models) == 1:
        return str(models[0]["provider"]), str(models[0]["model"])
    return "quota-relay", "quota-relay"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Dev34 or an official single-candidate frozen release stage."
    )
    parser.add_argument("--product-cases", default=str(DEFAULT_PRODUCT_DEV_CASES))
    parser.add_argument(
        "--product-split",
        choices=["dev", "core_frozen", "challenge_frozen", "shadow_frozen", "all"],
        default="dev",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the selected case id(s) after split filtering; repeatable.",
    )
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
        "--tool-provider",
        choices=["configured", "local"],
        default="configured",
        help="Use configured live tools or the frozen local seed provider.",
    )
    parser.add_argument(
        "--max-model-tokens",
        type=int,
        default=0,
        help="Optional token cap per model; 0 disables the local cap.",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "eval" / "product"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--official-frozen",
        action="store_true",
        help="Enforce the pre-registered single-candidate frozen release contract.",
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="Frozen release manifest required by --official-frozen.",
    )
    parser.add_argument("--core-acceptance", type=Path)
    parser.add_argument("--challenge-acceptance", type=Path)
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
        "model_preflight": [],
        "model_switch_log": [],
        "rows": [],
    }


def multi_agent_fields(result) -> dict[str, Any]:
    """Artifact-scoped proof that every semantically required role completed."""
    items = [item for turn in result.turns for item in (turn.agent_trace or [])]
    if not items:
        payload = result.final_artifacts.get("agent_trace") or {}
        items = list(payload.get("items") or [])
    successful_agents = {
        str(item.get("agent"))
        for item in items
        if item.get("kind") == "subagent"
        and item.get("status") in ("completed", "completed_with_warnings")
    }
    last = result.last_turn
    from travel_agent.evaluation.artifact_contract import required_agents_for_delivery

    expected = str((result.metrics or {}).get("expected_artifact_type") or "full_itinerary")
    required_agents = required_agents_for_delivery(
        expected,
        task_brief=str(last.user_message if last else ""),
        profile=(last.profile if last else result.final_profile),
    )
    missing_agents = [agent for agent in required_agents if agent not in successful_agents]
    real_delivery = bool(last and last.used_real_agent)
    artifact_delivered = bool((result.metrics or {}).get("artifact_type_match"))
    return {
        "multi_agent_mode": True,
        "multi_agent_required_agents": required_agents,
        "multi_agent_successful_agents": sorted(successful_agents),
        "multi_agent_missing_agents": missing_agents,
        "multi_agent_all_agents_completed": not missing_agents,
        "multi_agent_strict_success": bool(
            real_delivery and artifact_delivered and not missing_agents
        ),
    }


def provider_reported_models(result) -> list[str]:
    """Collect upstream response model identities without replacing requested identity."""
    return sorted({
        str(call.get("model") or "").strip()
        for turn in result.turns
        for call in (turn.model_calls or [])
        if str(call.get("model") or "").strip()
    })


def provider_tool_quota_error(result) -> str | None:
    """Return an explicit configured-tool quota failure, excluding LLM quota errors."""
    messages = [str(item) for item in (result.errors or []) if item]
    for turn in result.turns:
        messages.extend(
            str(item)
            for item in (turn.error, turn.raw_failure)
            if item
        )
    markers = (
        "providerratelimiterror",
        "amap rate limit",
        "user_daily_query_over_limit",
    )
    return next(
        (message for message in messages if any(marker in message.lower() for marker in markers)),
        None,
    )


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(state_path)


def save_pending_case_output(pending_dir: Path, case_output: dict[str, Any]) -> Path:
    """Persist a completed case before advancing relay resume state."""
    case_id = str(case_output["case"]["case_id"])
    repeat = int(case_output["execution"]["repeat"])
    path = pending_dir / f"{case_id}__repeat-{repeat}.json"
    pending_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(case_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def load_pending_case_outputs(pending_dir: Path) -> list[dict[str, Any]]:
    if not pending_dir.is_dir():
        return []
    outputs: list[dict[str, Any]] = []
    for path in sorted(pending_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"pending case output must be a JSON object: {path}")
        outputs.append(payload)
    return outputs


def build_failed_row(case, reason: str) -> dict[str, Any]:
    from travel_agent.evaluation.artifact_contract import (
        expected_artifact_type,
        required_agents_for_delivery,
    )

    expected = expected_artifact_type(case)
    required_agents = required_agents_for_delivery(
        expected,
        task_brief=str(case.turns[-1] if case.turns else ""),
        profile={"constraint_state": dict(case.hard_constraints or {})},
    )
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
        "multi_agent_required_agents": required_agents,
        "multi_agent_successful_agents": [],
        "multi_agent_missing_agents": required_agents,
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

    if args.official_frozen and args.release_manifest is None:
        raise ValueError("--official-frozen requires --release-manifest")
    model_entries = args.model or (
        ["deepseek:deepseek-v4-flash"] if args.official_frozen else DEFAULT_RELAY_MODELS
    )
    model_specs = parse_models(model_entries, default_provider)
    if not model_specs:
        raise ValueError("No model specified.")

    release_manifest = None
    case_source_path = Path(args.product_cases)
    if args.official_frozen:
        release_manifest = load_release_manifest(args.release_manifest)
        request_failures = validate_frozen_run_request(
            args, settings, model_specs, release_manifest
        )
        request_failures.extend(
            validate_runtime_against_manifest(
                release_manifest, settings, args.tool_provider
            )
        )
        if _git_revision() != release_manifest.get("commit_sha"):
            request_failures.append("current git revision differs from release candidate")
        if args.product_split == "shadow_frozen":
            request_failures.extend(
                validate_shadow_prerequisites(
                    release_manifest,
                    args.core_acceptance,
                    args.challenge_acceptance,
                )
            )
        if request_failures:
            raise RuntimeError(
                "official frozen run rejected before case loading: "
                + "; ".join(request_failures)
            )
        case_source_path, cases = load_frozen_split_cases(
            release_manifest, args.product_split
        )
        selected_cases = list(cases)
    else:
        reject_unofficial_frozen_request(args.product_split)
        cases = load_cases_json(args.product_cases)
        if len(cases) != 34 or any(case.split != "dev" for case in cases):
            raise RuntimeError("nonofficial runner requires the standalone 34-row dev.jsonl")
        selected_cases = list(cases)
    if args.case_id:
        wanted = set(args.case_id)
        selected_cases = [case for case in selected_cases if case.case_id in wanted]
        missing = wanted - {case.case_id for case in selected_cases}
        if missing:
            raise ValueError(f"case ids not found in selected split: {sorted(missing)}")
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
    pending_cases_dir = run_dir / "pending_cases"
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_lock = acquire_run_lock(run_dir)

    base_url = str(settings.llm.base_url)
    state = load_state(state_path)
    if not state:
        state = build_state_template(run_id, model_specs, args.base_url, base_url)
        state["case_count"] = len(selected_cases)
        state["split"] = args.product_split
        state["limit"] = args.limit
        state["tool_provider_mode"] = args.tool_provider
    else:
        if args.resume:
            recorded_tool_provider = str(state.get("tool_provider_mode") or "configured")
            if recorded_tool_provider != args.tool_provider:
                raise RuntimeError(
                    "resume tool-provider mismatch: "
                    f"recorded={recorded_tool_provider} requested={args.tool_provider}"
                )
            state["run_id"] = run_id
            state["case_count"] = len(selected_cases)
            state["split"] = args.product_split
            state["limit"] = args.limit
        else:
            raise RuntimeError("run already exists. Use --resume or choose different --run-id.")

    if args.preflight and not args.resume:
        print("Preflighting model list...")
        usable: list[tuple[str, str, str]] = []
        state["model_preflight"] = []
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
            state["model_preflight"].append(result)
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
                state["model_unavailable"].append(
                    {
                        "provider": provider,
                        "model": model,
                        "reason": "preflight_failed",
                        "detail": result,
                    }
                )
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
        tool_provider=args.tool_provider,
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
    case_outputs = load_pending_case_outputs(pending_cases_dir)
    attempts_total = int(state.get("attempts_total", 0))
    model_usage = {
        key: {"tokens": int(value.get("tokens", 0)), "cases": int(value.get("cases", 0))}
        for key, value in state.get("model_usage", {}).items()
    }
    current_model_index = int(state.get("current_model_index", 0))

    def append_case_output(case_output: dict[str, Any]) -> None:
        save_pending_case_output(pending_cases_dir, case_output)
        case_outputs.append(case_output)
        state["rows"] = rows
        save_state(state_path, state)

    fatal_tool_quota_error: str | None = None

    def try_models_for_case(case) -> bool:
        nonlocal current_model_index, attempts_total, fatal_tool_quota_error
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
            # Isolate L3 memory across distinct eval runs. Reusing the same
            # case-scoped user id makes a prior Smoke/Mini run leak preferences
            # into a later run even though artifact persistence is disabled.
            result = harness.run_case(
                _execution_case(case, 1, run_namespace=run_id)
            )
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

            fatal_tool_quota_error = provider_tool_quota_error(result)
            errors = [item for item in row.get("model_call_errors", [])]
            if errors and fatal_tool_quota_error is None:
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
            output["execution"]["requested_model"] = model
            output["execution"]["provider_reported_models"] = provider_reported_models(result)
            append_case_output(output)
            row["runtime_model"] = model
            rows.append(row)
            if fatal_tool_quota_error is not None:
                state["rows"] = rows
                state["fatal_tool_quota_error"] = fatal_tool_quota_error
                save_state(state_path, state)
                return True
            completed_cases.add(case_key)
            state["completed_case_ids"] = sorted(completed_cases)
            state["rows"] = rows
            state["current_model_index"] = current_model_index
            save_state(state_path, state)
            return True

        # if all remaining models fail on this case, write an explicit failure row
        if last_reason == "local token cap reached":
            return False
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
        return True

    stopped_early = False
    stop_reason: str | None = None
    for index, case in enumerate(selected_cases, start=1):
        if case.case_id in completed_cases:
            print(f"[skip] {case.case_id} already done")
            continue
        print(f"[case {index}/{len(selected_cases)}] {case.case_id}")
        if current_model_index >= len(model_specs):
            stopped_early = True
            stop_reason = f"relay pool exhausted after {len(completed_cases)} completed cases"
            print(f"[stop] {stop_reason}")
            break
        if not try_models_for_case(case):
            stopped_early = True
            stop_reason = (
                f"local model token cap reached after {len(completed_cases)} completed cases"
            )
            print(f"[stop] {stop_reason}")
            break
        if fatal_tool_quota_error is not None:
            stopped_early = True
            stop_reason = (
                "configured tool provider quota exhausted after "
                f"{len(completed_cases)} completed cases: {fatal_tool_quota_error}"
            )
            print(f"[stop] {stop_reason}")
            break

    summary = aggregate_product_rows(rows, remediation_complete=args.remediation_complete)
    relay_enabled = relay_mode_enabled(state["models"])
    summary["relay_mode"] = relay_enabled
    summary["real_multi_agent"] = True
    strict_values = [row.get("multi_agent_strict_success") for row in rows]
    strict_values = [value for value in strict_values if isinstance(value, bool)]
    summary["multi_agent_strict_success_rate"] = (
        sum(strict_values) / len(strict_values) if strict_values else None
    )
    summary["model_attempts_by_case"] = state.get("attempts_by_case", {})
    summary["model_switch_log"] = state.get("model_switch_log", [])
    summary["model_switch_count"] = len(state.get("model_switch_log", []))
    summary["environment_abort_reason"] = fatal_tool_quota_error

    recorded_provider, recorded_model = recorded_model_identity(state["models"])
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
            "dataset_sha256": _sha256(case_source_path),
            "code_revision": _git_revision(),
            "model_provider": recorded_provider,
            "model": recorded_model,
            "model_temperature": settings.llm.temperature,
            "model_thinking_enabled": settings.llm.thinking_enabled,
            "model_requests_per_second": settings.llm.requests_per_second,
            "hybrid_flags": {
                "enable_llm_intent_normalizer": settings.hybrid_planning.enable_llm_intent_normalizer,
                "enable_llm_preference_resolver": settings.hybrid_planning.enable_llm_preference_resolver,
                "enable_structured_duration_estimator": settings.hybrid_planning.enable_structured_duration_estimator,
            },
            "prompt_fingerprint": _prompt_fingerprint(),
            "code_fingerprint": _code_fingerprint(),
            "evaluator_fingerprint": _evaluator_fingerprint(),
            "tooling_fingerprint": _tooling_fingerprint(),
            "configuration_fingerprint": _configuration_fingerprint(
                settings, args.tool_provider
            ),
            "tool_snapshot_fingerprint": _tool_snapshot_fingerprint(
                settings, args.tool_provider
            ),
            "evaluator_version": _evaluator_version(),
            "artifact_contract_fingerprint": _artifact_contract_fingerprint(selected_cases),
            "artifact_contract_implementation_fingerprint": artifact_contract_implementation_fingerprint(),
            "relay_models": state["models"],
            "model_execution_mode": (
                "model_relay" if relay_enabled else "fixed_single_model"
            ),
            "requested_models": sorted({item["model"] for item in state["models"]}),
            "provider_reported_models": sorted({
                reported
                for output in case_outputs
                for reported in (
                    (output.get("execution") or {}).get("provider_reported_models") or []
                )
            }),
            "model_preflight": list(state.get("model_preflight") or []),
            "real_multi_agent": True,
            "tool_provider_mode": args.tool_provider,
            "required_agents": sorted({
                agent
                for row in rows
                for agent in (row.get("multi_agent_required_agents") or [])
            }),
            "relay_model_usage": model_usage,
            "relay_summary": {
                "attempts_total": attempts_total,
                "current_model_index": current_model_index,
            },
            "environment_abort_reason": fatal_tool_quota_error,
            "_case_outputs": case_outputs,
        },
        benchmarks=[],
        stopped_early=stopped_early,
        stop_reason=stop_reason,
    )

    if release_manifest is not None:
        suite_result.artifacts.update(
            {
                "release_manifest_schema_version": FROZEN_RELEASE_SCHEMA_VERSION,
                "release_candidate_id": release_manifest["candidate_id"],
                "release_manifest_fingerprint": release_manifest["manifest_fingerprint"],
                "frozen_split": args.product_split,
                "frozen_split_sha256": release_manifest["splits"][args.product_split]["sha256"],
            }
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
