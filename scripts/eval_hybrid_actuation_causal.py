#!/usr/bin/env python3
"""Six-cell deterministic causal A/B for soft-preference actuation.

This runner isolates actuation from Router/Worker/Reviewer/model variance.  It
uses production toolkit entrypoints with fixed candidates and a fixture-backed
structured policy.  It never calls a network model, Judge, Dev34, or Frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass, replace
from typing import Any

from travel_agent.agent import toolkit
from travel_agent.agent.serde import poi_to_dict
from travel_agent.agent.session import build_session
from travel_agent.artifact_policy import constraint_version
from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.hybrid_planning.soft_preference_actuation import stable_fingerprint
from travel_agent.schemas import POI, TravelProfile
from travel_agent.settings import HybridPlanningSettings, load_settings


SEED = 20260816


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    profile: TravelProfile
    pois: tuple[POI, ...]
    policy: dict[str, Any]
    target_candidate_id: str
    hard_invalid_candidate_ids: tuple[str, ...] = ()
    routes: tuple[dict[str, Any], ...] = ()
    budget: dict[str, Any] | None = None


class FixturePolicyClient:
    """Deterministic structured boundary; records calls without network I/O."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> StructuredLLMResponse:
        self.calls.append({
            "schema_version": kwargs.get("schema_version"),
            "input_fingerprint": stable_fingerprint({
                "system_prompt": kwargs.get("system_prompt"),
                "user_prompt": kwargs.get("user_prompt"),
            }),
        })
        return StructuredLLMResponse(
            self.payload,
            model="deterministic-policy-fixture",
            latency_ms=0.0,
            token_usage={},
        )


def _poi(
    poi_id: str,
    *,
    tag: str,
    rating: float,
    popularity: float,
    average_cost: float | None = None,
) -> POI:
    return POI(
        poi_id=poi_id,
        name=f"候选{poi_id}",
        city="测试城",
        category="attraction",
        lat=30.0,
        lng=120.0,
        rating=rating,
        popularity=popularity,
        tags=[tag],
        estimated_duration_min=120,
        price_level="low",
        average_cost=average_cost,
        source="seed",
        source_poi_id=poi_id,
        verification_status="verified",
    )


def _policy(
    dimension: str,
    direction: str,
    source: str,
    reason: str,
    candidate_order: list[str],
) -> dict[str, Any]:
    return {
        "priorities": [{
            "dimension": dimension,
            "direction": direction,
            "importance": 1.0,
            "source": source,
            "reason": reason,
        }],
        "pace": "normal",
        "acceptable_tradeoffs": {},
        "candidate_order": candidate_order,
        "unresolved": [],
        "confidence": 0.95,
    }


def _scenarios() -> tuple[Scenario, ...]:
    interest_context = {
        "normalized_interests": [{
            "label": "industrial_heritage",
            "source_text": "旧厂房改造成公共空间",
            "polarity": "prefer",
            "confidence": 0.95,
            "source_turn": "turn_1",
            "basis": "fixture",
        }]
    }
    return (
        Scenario(
            scenario_id="interest_tradeoff",
            profile=TravelProfile(
                destination="测试城",
                days=1,
                pace="relaxed",
                interests=["industrial_heritage"],
                constraint_state={"normalized_interest_context": interest_context},
            ),
            pois=(
                _poi("interest", tag="industrial_heritage", rating=0.0, popularity=0.0),
                _poi("popular-a", tag="nature", rating=5.0, popularity=1.0),
                _poi("popular-b", tag="culture", rating=4.9, popularity=0.98),
            ),
            policy=_policy(
                "user_interest_match",
                "maximize",
                "旧厂房改造成公共空间",
                "兴趣匹配优先",
                ["interest", "popular-a", "popular-b"],
            ),
            target_candidate_id="interest",
        ),
        Scenario(
            scenario_id="commute_tradeoff",
            profile=TravelProfile(
                destination="测试城",
                days=1,
                constraint_state={"preference": "通勤优先"},
            ),
            pois=(
                _poi("near", tag="nature", rating=4.0, popularity=0.7),
                _poi("far", tag="culture", rating=5.0, popularity=1.0),
            ),
            routes=(
                {
                    "origin_poi_id": "hotel-anchor",
                    "destination_poi_id": "near",
                    "duration_min": 8,
                    "walking_distance_km": 0.5,
                    "transfer_count": 0,
                    "evidence_status": "provider_verified",
                },
                {
                    "origin_poi_id": "hotel-anchor",
                    "destination_poi_id": "far",
                    "duration_min": 35,
                    "walking_distance_km": 0.5,
                    "transfer_count": 2,
                    "evidence_status": "provider_verified",
                },
            ),
            policy=_policy(
                "commute_time",
                "minimize",
                "通勤优先",
                "减少通勤",
                ["near", "far"],
            ),
            target_candidate_id="near",
        ),
        Scenario(
            scenario_id="budget_hard_gate",
            profile=TravelProfile(
                destination="测试城",
                days=1,
                party_size=2,
                budget_limit=500,
                constraint_state={"preference": "价格优先"},
            ),
            pois=(
                _poi("cheap-a", tag="nature", rating=4.9, popularity=0.98, average_cost=100),
                _poi("cheap-b", tag="culture", rating=4.8, popularity=0.95, average_cost=120),
                _poi("over-budget", tag="history", rating=5.0, popularity=1.0, average_cost=400),
            ),
            policy=_policy(
                "price",
                "minimize",
                "价格优先",
                "控制预算",
                ["cheap-a", "cheap-b"],
            ),
            target_candidate_id="cheap-a",
            hard_invalid_candidate_ids=("over-budget",),
            budget={
                "currency": "CNY",
                "hotel": 0,
                "inner_city_transport": 0,
                "tickets": 440,
                "meals": 0,
                "fixed_event_cost": 0,
                "contingency": 0,
                "total_low": 440,
                "total_high": 490,
            },
        ),
    )


def _run_cell(scenario: Scenario, *, actuation_enabled: bool) -> dict[str, Any]:
    random.seed(SEED)
    ctx = build_session(persist=False)
    ctx.profile = copy.deepcopy(scenario.profile)
    ctx.active_task_type = "full_itinerary"
    ctx.active_delivery_intent = "rebuild_now"
    ctx.hybrid_request_scope = f"causal:{scenario.scenario_id}"
    ctx.runtime_settings = replace(
        load_settings(),
        hybrid_planning=HybridPlanningSettings(
            enable_llm_intent_normalizer=False,
            enable_llm_preference_resolver=actuation_enabled,
            enable_structured_duration_estimator=False,
        ),
    )
    client = FixturePolicyClient(scenario.policy)
    if actuation_enabled:
        ctx.hybrid_llm_client = client

    candidate_id = ctx.store.put(
        "candidates",
        {"city": "测试城", "pois": [poi_to_dict(poi) for poi in scenario.pois]},
    )
    ctx.remember_pois(list(scenario.pois))
    route_ids = [ctx.store.put("routes", route) for route in scenario.routes]
    budget_ids = [ctx.store.put("budget", scenario.budget)] if scenario.budget else []
    bound_ids = [candidate_id, *route_ids, *budget_ids]
    input_fingerprint = stable_fingerprint({
        "seed": SEED,
        "profile": ctx.profile,
        "pois": [poi_to_dict(poi) for poi in scenario.pois],
        "routes": scenario.routes,
        "budget": scenario.budget,
    })

    started = time.perf_counter()
    ranked_result = toolkit.recommend_candidates(ctx, artifact_ids=bound_ids)
    if ranked_result.get("isError"):
        return {
            "scenario_id": scenario.scenario_id,
            "variant": "actuation_on" if actuation_enabled else "baseline",
            "input_fingerprint": input_fingerprint,
            "error": ranked_result,
        }
    ranked_id = str(ranked_result["artifact_id"])
    ranked_payload = ctx.store.get(ranked_id)
    planned_result = toolkit.plan_and_critique(
        ctx, artifact_ids=[*bound_ids, ranked_id]
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if planned_result.get("isError"):
        return {
            "scenario_id": scenario.scenario_id,
            "variant": "actuation_on" if actuation_enabled else "baseline",
            "input_fingerprint": input_fingerprint,
            "error": planned_result,
        }
    plan_payload = ctx.store.get(str(planned_result["artifact_id"]))
    selected = [
        str(stop.get("poi", {}).get("poi_id") or "")
        for day in (plan_payload.get("itinerary") or {}).get("days") or []
        for stop in day.get("stops") or []
    ]
    ranked_ids = [
        str(item.get("poi", {}).get("poi_id") or "")
        for item in ranked_payload.get("pois") or []
    ]
    invalid_selected = sorted(
        set(selected).intersection(scenario.hard_invalid_candidate_ids)
    )
    active_version = constraint_version(ctx.profile)
    state_version = plan_payload.get("state_version") or {}
    contract = plan_payload.get("soft_preference_actuation") or {}
    fingerprints = contract.get("fingerprints") or {}
    selection_provenance = contract.get("final_selection_provenance") or {}
    validation_passed = (plan_payload.get("validation_result") or {}).get("passed")
    artifact_status = plan_payload.get("artifact_status")
    return {
        "scenario_id": scenario.scenario_id,
        "variant": "actuation_on" if actuation_enabled else "baseline",
        "seed": SEED,
        "input_fingerprint": input_fingerprint,
        "ranked_candidate_ids": ranked_ids,
        "selected_candidate_ids": selected,
        "target_candidate_id": scenario.target_candidate_id,
        "target_ranked_first": bool(ranked_ids and ranked_ids[0] == scenario.target_candidate_id),
        "target_selected_first": bool(selected and selected[0] == scenario.target_candidate_id),
        "hard_invalid_selected": invalid_selected,
        "hard_gate_pass": not invalid_selected,
        "artifact_status": artifact_status,
        "validation_passed": validation_passed,
        "artifact_status_fail_closed": (
            (validation_passed is True and artifact_status == "candidate")
            or (validation_passed is not True and artifact_status == "validation_failure")
        ),
        "critic_passed": (plan_payload.get("critic") or {}).get("passed"),
        "constraint_revision": state_version.get("constraint_revision"),
        "constraint_hash": state_version.get("constraint_hash"),
        "active_constraint_hash": active_version.get("constraint_hash"),
        "state_version_match": (
            state_version.get("constraint_hash") == active_version.get("constraint_hash")
        ),
        "actuation_contract_present": bool(contract),
        "planner_input_fingerprint_present": bool(
            fingerprints.get("planner_input_fingerprint")
        ),
        "final_artifact_fingerprint_present": bool(
            fingerprints.get("final_artifact_fingerprint")
        ),
        "all_selected_from_legal_ranking": selection_provenance.get(
            "all_selected_from_legal_ranking"
        ),
        "funnel": contract.get("funnel"),
        "fixture_policy_calls": len(client.calls),
        "real_llm_calls": 0,
        "judge_calls": 0,
        "elapsed_ms": elapsed_ms,
    }


def run() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for scenario in _scenarios():
        cells.append(_run_cell(scenario, actuation_enabled=False))
        cells.append(_run_cell(scenario, actuation_enabled=True))

    comparisons: list[dict[str, Any]] = []
    for scenario in _scenarios():
        baseline = next(
            cell for cell in cells
            if cell["scenario_id"] == scenario.scenario_id
            and cell["variant"] == "baseline"
        )
        treatment = next(
            cell for cell in cells
            if cell["scenario_id"] == scenario.scenario_id
            and cell["variant"] == "actuation_on"
        )
        comparisons.append({
            "scenario_id": scenario.scenario_id,
            "same_input": baseline.get("input_fingerprint")
            == treatment.get("input_fingerprint"),
            "target_ranked_first_delta": int(treatment.get("target_ranked_first", False))
            - int(baseline.get("target_ranked_first", False)),
            "target_selected_first_delta": int(treatment.get("target_selected_first", False))
            - int(baseline.get("target_selected_first", False)),
            "hard_invalid_selected_delta": len(treatment.get("hard_invalid_selected") or [])
            - len(baseline.get("hard_invalid_selected") or []),
            "artifact_validation_regression": bool(baseline.get("validation_passed"))
            and not bool(treatment.get("validation_passed")),
            "critic_regression": bool(baseline.get("critic_passed"))
            and not bool(treatment.get("critic_passed")),
        })

    treatments = [cell for cell in cells if cell.get("variant") == "actuation_on"]
    hard_gate = all(cell.get("hard_gate_pass") is True for cell in treatments)
    lifecycle_gate = all(
        cell.get("state_version_match") is True
        and cell.get("artifact_status_fail_closed") is True
        and cell.get("actuation_contract_present") is True
        and cell.get("planner_input_fingerprint_present") is True
        and cell.get("final_artifact_fingerprint_present") is True
        and cell.get("all_selected_from_legal_ranking") is True
        for cell in treatments
    )
    quality_gate = (
        sum(item["target_selected_first_delta"] > 0 for item in comparisons) >= 2
        and all(item["target_selected_first_delta"] >= 0 for item in comparisons)
        and all(not item["artifact_validation_regression"] for item in comparisons)
        and all(not item["critic_regression"] for item in comparisons)
    )
    cost_gate = all(
        cell.get("fixture_policy_calls") == 1
        and cell.get("real_llm_calls") == 0
        and cell.get("judge_calls") == 0
        for cell in treatments
    )
    defaults = HybridPlanningSettings()
    default_flags_off = not any((
        defaults.enable_llm_intent_normalizer,
        defaults.enable_llm_preference_resolver,
        defaults.enable_structured_duration_estimator,
    ))
    passed = hard_gate and lifecycle_gate and quality_gate and cost_gate and default_flags_off
    return {
        "schema_version": "hybrid-actuation-causal-ab-v1",
        "seed": SEED,
        "cell_count": len(cells),
        "scope": {
            "real_llm": False,
            "judge": False,
            "dev34": False,
            "frozen": False,
            "isolated_variable": "enable_llm_preference_resolver",
        },
        "cells": cells,
        "comparisons": comparisons,
        "gates": {
            "hard_gate": hard_gate,
            "artifact_lifecycle_gate": lifecycle_gate,
            "quality_gate": quality_gate,
            "cost_gate": cost_gate,
            "default_flags_off": default_flags_off,
        },
        "conclusion": (
            "READY_FOR_TARGETED_DEV" if passed else "KEEP_FLAGS_OFF_AND_FIX"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(result["conclusion"])
        print(json.dumps(result["gates"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
