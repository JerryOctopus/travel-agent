#!/usr/bin/env python3
"""Low-cost Intent/Preference contract runner.

No network is touched unless ``--live`` is explicit.  The runner invokes only
the two hybrid modules: it never constructs the Router, Planner, subagents,
Reviewer, Judge, or an Agent session.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent.hybrid_planning.intent_normalizer import (
    INTENT_NORMALIZER_PROMPT_VERSION,
    INTENT_NORMALIZER_SCHEMA_VERSION,
    IntentNormalizer,
)
from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.hybrid_planning.preference_resolver import (
    PREFERENCE_RESOLVER_PROMPT_VERSION,
    PREFERENCE_RESOLVER_SCHEMA_VERSION,
    PreferenceResolver,
)
from travel_agent.hybrid_planning.structured_client import ProductionStructuredLLMClient
from travel_agent.hybrid_planning.taxonomy import INTEREST_TAXONOMY_VERSION
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog
from travel_agent.settings import load_settings


MODEL_PROVIDER = "deepseek"
MODEL_NAME = "deepseek-chat"
MODEL_BASE_URL = "https://api.deepseek.com/v1"
TEMPERATURE = 0.0
MAX_BUSINESS_CALLS = 12
MAX_REPAIR_CALLS = 4
MAX_TOTAL_CALLS = 16


INTENT_FIXTURES = (
    ("intent_1_local_life", "老城区烟火气"),
    ("intent_2_architecture", "有年代感的建筑"),
    ("intent_3_industrial_renewal", "工业遗迹和城市更新"),
    ("intent_4_avoid_commercial", "不要太商业化"),
    ("intent_5_coast_cafe", "海边发呆喝咖啡"),
    ("intent_6_unknown", "喜欢一种难以可靠归类的未来城市气质"),
)

INTENT_STRUCTURED_PROBES = (
    ("intent_probe_1_local_life", "我喜欢能观察街坊日常生活的街区"),
    ("intent_probe_2_architecture", "我偏好能看出不同时代层次的街区设计"),
    ("intent_probe_3_reuse", "我想体验旧厂房改造成公共空间的地方"),
    ("intent_probe_4_commercialized", "我不喜欢连锁店和游客消费占主导的街区"),
    ("intent_probe_5_slow_drink", "我喜欢临水慢坐，再找适合手冲饮品的安静小店"),
)


BASE_CANDIDATES = (
    {
        "candidate_id": "candidate_a",
        "price": 500,
        "commute_time": 35,
        "walking_load": 45,
        "transfer_count": 2,
        "location": "outer",
        "pace": "compact",
        "accessibility": 0.4,
        "activity_density": 0.9,
        "user_interest_match": 0.7,
        "hard_constraints_passed": True,
    },
    {
        "candidate_id": "candidate_b",
        "price": 620,
        "commute_time": 10,
        "walking_load": 15,
        "transfer_count": 0,
        "location": "central",
        "pace": "relaxed",
        "accessibility": 0.9,
        "activity_density": 0.5,
        "user_interest_match": 0.85,
        "hard_constraints_passed": True,
    },
)


PREFERENCE_FIXTURES = (
    ("preference_1_price", "价格优先", BASE_CANDIDATES, {"budget_max_cny": 1000}),
    ("preference_2_commute", "通勤优先", BASE_CANDIDATES, {"budget_max_cny": 1000}),
    (
        "preference_3_elderly",
        "带老人，减少换乘和步行，优先无障碍",
        BASE_CANDIDATES,
        {"budget_max_cny": 1000},
    ),
    ("preference_4_child", "带孩子，节奏轻松", BASE_CANDIDATES, {"budget_max_cny": 1000}),
    ("preference_5_none", None, BASE_CANDIDATES, {"budget_max_cny": 1000}),
    (
        "preference_6_budget_conflict",
        "位置好愿意贵20%",
        (
            {**BASE_CANDIDATES[0], "price": 500},
            {**BASE_CANDIDATES[1], "price": 1200},
        ),
        {"budget_max_cny": 600},
    ),
)


@dataclass
class CallAudit:
    raw_outputs: list[str]
    raw_output_summary: dict[str, Any]
    payload: dict[str, Any] | None
    parse_status: str | None
    parse_reason_code: str | None
    token_usage: dict[str, int]
    latency_ms: float | None
    attempts: int
    error: str | None = None


class RecordingClient:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[CallAudit] = []

    def complete_json(self, **kwargs: Any) -> StructuredLLMResponse:
        try:
            response = self.delegate.complete_json(**kwargs)
        except Exception as exc:
            self.calls.append(
                CallAudit(
                    raw_outputs=[str(getattr(exc, "raw_output", ""))],
                    raw_output_summary=dict(getattr(exc, "raw_output_summary", {}) or {}),
                    payload=None,
                    parse_status=getattr(exc, "parse_status", None),
                    parse_reason_code=getattr(exc, "reason_code", type(exc).__name__),
                    token_usage=dict(getattr(exc, "token_usage", {}) or {}),
                    latency_ms=getattr(exc, "latency_ms", None),
                    attempts=int(getattr(exc, "attempts", 1) or 1),
                    error=type(exc).__name__,
                )
            )
            raise
        self.calls.append(
            CallAudit(
                raw_outputs=list(response.raw_outputs or (response.raw_output,)),
                raw_output_summary=dict(response.raw_output_summary),
                payload=response.payload,
                parse_status=response.parse_status,
                parse_reason_code=response.parse_reason_code,
                token_usage=dict(response.token_usage),
                latency_ms=response.latency_ms,
                attempts=response.attempts,
            )
        )
        return response


def _live_clients() -> tuple[RecordingClient, RecordingClient]:
    settings = load_settings()
    configured_key = (
        settings.llm.api_key if settings.llm.provider == MODEL_PROVIDER else None
    )
    api_key = os.getenv("TRAVEL_AGENT_DEEPSEEK_API_KEY") or configured_key
    if not api_key:
        raise RuntimeError(
            "--live requires a DeepSeek key via config.toml or TRAVEL_AGENT_DEEPSEEK_API_KEY"
        )
    llm = replace(
        settings.llm,
        provider=MODEL_PROVIDER,
        api_key=api_key,
        base_url=MODEL_BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        thinking_enabled=False,
    )
    fixed = replace(settings, llm=llm)
    return (
        RecordingClient(
            ProductionStructuredLLMClient(
                fixed, phase="intent_contract", max_output_tokens=512
            )
        ),
        RecordingClient(
            ProductionStructuredLLMClient(
                fixed, phase="preference_contract", max_output_tokens=768
            )
        ),
    )


def _trace_detail(trace: AgentTraceLog, kind: str) -> dict[str, Any]:
    entries = [item for item in trace.snapshot() if item["kind"] == kind]
    return dict(entries[-1]["detail"]) if entries else {}


def _scenario_success(detail: dict[str, Any], *, module: str) -> bool:
    if not detail.get("called"):
        return detail.get("skipped_reason") in {
            "deterministic_match",
            "no_soft_preference",
            "fewer_than_two_legal_candidates",
            "feature_disabled",
        }
    if detail.get("attempted_hard_constraint_change"):
        return False
    if detail.get("unknown_candidate_reference_count"):
        return False
    if detail.get("out_of_taxonomy_label_count"):
        return False
    if module == "intent":
        return detail.get("parse_status") in {
            "valid", "format_repaired", "low_confidence"
        } and detail.get("fallback_reason") in {None, "low_confidence"}
    return detail.get("validation_status") == "valid" and not detail.get("fallback_reason")


def run_contract_suite(
    *, live: bool, supplemental_intent_probes: bool = False
) -> dict[str, Any]:
    intent_client: RecordingClient | None = None
    preference_client: RecordingClient | None = None
    if live:
        intent_client, preference_client = _live_clients()

    scenarios: list[dict[str, Any]] = []
    intent_fixtures = (
        INTENT_STRUCTURED_PROBES if supplemental_intent_probes else INTENT_FIXTURES
    )
    for scenario_id, text in intent_fixtures:
        before = len(intent_client.calls) if intent_client else 0
        trace = AgentTraceLog(f"contract:{scenario_id}")
        result = IntentNormalizer(
            intent_client,
            enabled=live,
            request_scope=f"contract:{scenario_id}",
        ).normalize(
            text,
            source_turn="contract_turn_1",
            request_type="full_itinerary",
            delivery_intent="rebuild_now",
            trace=trace,
        )
        detail = _trace_detail(trace, "hybrid_intent_normalizer")
        audit = intent_client.calls[-1] if intent_client and len(intent_client.calls) > before else None
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "module": "intent",
                "input": {"raw_interests": [text]},
                "raw_model_outputs": audit.raw_outputs if audit else [],
                "raw_output_summary": audit.raw_output_summary if audit else {},
                "structured_result": result.to_dict(),
                "parse_status": detail.get("parse_status"),
                "validation_status": detail.get("validation_status"),
                "parse_reason_code": detail.get("parse_reason_code"),
                "fallback_reason": result.fallback_reason,
                "called": detail.get("called", False),
                "token_usage": detail.get("token_usage", {}),
                "latency_ms": detail.get("latency_ms"),
                "attempts": detail.get("attempts", 0),
                "contract_success": _scenario_success(detail, module="intent"),
            }
        )

    preference_fixtures = () if supplemental_intent_probes else PREFERENCE_FIXTURES
    for scenario_id, preference, candidates, hard in preference_fixtures:
        before = len(preference_client.calls) if preference_client else 0
        trace = AgentTraceLog(f"contract:{scenario_id}")
        result = PreferenceResolver(
            preference_client,
            enabled=live,
            request_scope=f"contract:{scenario_id}",
        ).resolve(
            task_type="full_itinerary",
            candidates=list(candidates),
            soft_preferences=preference,
            hard_constraints={**hard, "_constraint_revision": 1},
            source_turn="contract_turn_1",
            hard_filter_complete=True,
            evidence_sufficient=True,
            delivery_intent="rebuild_now",
            trace=trace,
        )
        detail = _trace_detail(trace, "hybrid_preference_resolver")
        audit = (
            preference_client.calls[-1]
            if preference_client and len(preference_client.calls) > before
            else None
        )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "module": "preference",
                "input": {
                    "soft_preferences": preference,
                    "candidates": list(candidates),
                    "hard_constraints": hard,
                },
                "raw_model_outputs": audit.raw_outputs if audit else [],
                "raw_output_summary": audit.raw_output_summary if audit else {},
                "structured_result": result.to_dict(),
                "parse_status": detail.get("parse_status"),
                "validation_status": detail.get("validation_status"),
                "parse_reason_code": detail.get("parse_reason_code"),
                "fallback_reason": result.fallback_reason,
                "called": detail.get("called", False),
                "token_usage": detail.get("token_usage", {}),
                "latency_ms": detail.get("latency_ms"),
                "attempts": detail.get("attempts", 0),
                "contract_success": _scenario_success(detail, module="preference"),
            }
        )

    called = [item for item in scenarios if item["called"]]
    business_calls = len(called)
    repair_calls = sum(max(0, int(item["attempts"] or 0) - 1) for item in called)
    total_calls = business_calls + repair_calls
    by_module = {
        module: {
            "scenarios": sum(item["module"] == module for item in scenarios),
            "contract_success": sum(
                item["module"] == module and item["contract_success"] for item in scenarios
            ),
            "business_calls": sum(item["module"] == module and item["called"] for item in scenarios),
        }
        for module in ("intent", "preference")
    }
    summary = {
        "scenario_count": len(scenarios),
        "scenario_contract_success": sum(item["contract_success"] for item in scenarios),
        "business_calls": business_calls,
        "repair_calls": repair_calls,
        "total_model_calls": total_calls,
        "structured_success": sum(
            item["called"]
            and item["validation_status"] in {"valid", "valid_with_unresolved"}
            and item["fallback_reason"] is None
            for item in scenarios
        ),
        "fallback_count": sum(item["called"] and item["fallback_reason"] is not None for item in scenarios),
        "input_tokens": sum(int(item["token_usage"].get("input_tokens", 0)) for item in scenarios),
        "output_tokens": sum(int(item["token_usage"].get("output_tokens", 0)) for item in scenarios),
        "total_tokens": sum(int(item["token_usage"].get("total_tokens", 0)) for item in scenarios),
        "total_latency_ms": round(sum(float(item["latency_ms"] or 0) for item in scenarios), 2),
        "by_module": by_module,
        "limits_respected": (
            business_calls <= MAX_BUSINESS_CALLS
            and repair_calls <= MAX_REPAIR_CALLS
            and total_calls <= MAX_TOTAL_CALLS
        ),
    }
    if not summary["limits_respected"]:
        raise RuntimeError(f"contract call cap exceeded: {summary}")
    return {
        "run_type": (
            "hybrid_module_contract_intent_supplemental"
            if supplemental_intent_probes
            else "hybrid_module_contract"
        ),
        "mode": "live" if live else "offline_no_network",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixed_contract": {
            "provider": MODEL_PROVIDER,
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "intent_prompt_version": INTENT_NORMALIZER_PROMPT_VERSION,
            "intent_schema_version": INTENT_NORMALIZER_SCHEMA_VERSION,
            "preference_prompt_version": PREFERENCE_RESOLVER_PROMPT_VERSION,
            "preference_schema_version": PREFERENCE_RESOLVER_SCHEMA_VERSION,
            "taxonomy_version": INTEREST_TAXONOMY_VERSION,
            "business_call_cap": MAX_BUSINESS_CALLS,
            "repair_call_cap": MAX_REPAIR_CALLS,
            "total_call_cap": MAX_TOTAL_CALLS,
        },
        "summary": summary,
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly allow bounded DeepSeek network calls.",
    )
    parser.add_argument(
        "--supplemental-intent-probes",
        action="store_true",
        help="Run five additional eligible Intent structured-output probes only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "travel-agent-hybrid-contract.json",
    )
    args = parser.parse_args()
    result = run_contract_suite(
        live=args.live,
        supplemental_intent_probes=args.supplemental_intent_probes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
