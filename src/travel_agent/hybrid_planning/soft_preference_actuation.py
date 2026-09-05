"""Deterministic actuation of normalized interests and preference policies.

The LLM-facing v2 schemas stay interpretive.  This module owns retrieval
queries, evidence, numeric utility, ranking, fingerprints, and provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import replace
from typing import Any, Iterable, Mapping

from travel_agent.hybrid_planning.preference_resolver import PlanningPolicy
from travel_agent.hybrid_planning.taxonomy import (
    CANONICAL_INTERESTS,
    INTEREST_ALIAS_PAIRS,
    INTEREST_RETRIEVAL_DESCRIPTORS,
)
from travel_agent.poi_evidence import poi_avoid_match
from travel_agent.schemas import POI, ScoredPOI, TravelProfile


SOFT_PREFERENCE_ACTUATION_VERSION = "soft-preference-actuation-v1"
DEFAULT_QUERY_BUDGET = 4
DEFAULT_PER_INTEREST_QUERY_BUDGET = 2
DEFAULT_DIMENSION_DELTA_CAP = 0.12
DEFAULT_TOTAL_DELTA_CAP = 0.25


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", text)


_PRICE_LEVEL = {"free": 0.0, "low": 1.0, "mid": 2.0, "high": 3.0}
_ACCESSIBILITY_TAGS = frozenset({
    "无障碍", "轮椅友好", "无台阶", "电梯", "accessible", "wheelchair_accessible",
})
_TAG_TO_INTEREST = {
    _normalize_text(alias): label for alias, label in INTEREST_ALIAS_PAIRS
}
_TAG_TO_INTEREST.update({_normalize_text(label): label for label in CANONICAL_INTERESTS})


def stable_fingerprint(value: Any, *, length: int = 20) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def effective_normalized_interests(
    normalized_context: Mapping[str, Any] | None,
    profile_interests: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return controlled interests without inventing evidence or labels."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in (normalized_context or {}).get("normalized_interests") or []:
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("label") or "").strip()
        polarity = str(raw.get("polarity") or "prefer").strip()
        source_text = str(raw.get("source_text") or "").strip()
        if label not in CANONICAL_INTERESTS or polarity not in {"prefer", "avoid"}:
            continue
        key = (label, polarity, source_text)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "label": label,
            "polarity": polarity,
            "source_text": source_text,
            "confidence": _bounded_number(raw.get("confidence"), default=1.0),
            "source_turn": raw.get("source_turn"),
            "basis": str(raw.get("basis") or "normalized"),
        })
    for value in profile_interests:
        label = str(value or "").strip()
        if label not in CANONICAL_INTERESTS:
            continue
        key = (label, "prefer", label)
        if key in seen or any(item[0] == label and item[1] == "prefer" for item in seen):
            continue
        seen.add(key)
        result.append({
            "label": label,
            "polarity": "prefer",
            "source_text": label,
            "confidence": 1.0,
            "source_turn": None,
            "basis": "profile",
        })
    return result


def build_retrieval_plan(
    normalized_context: Mapping[str, Any] | None,
    *,
    destination: str,
    request_id: str,
    turn_id: str,
    constraint_revision: int | None,
    constraint_hash: str | None,
    query_budget: int = DEFAULT_QUERY_BUDGET,
    per_interest_query_budget: int = DEFAULT_PER_INTEREST_QUERY_BUDGET,
) -> dict[str, Any]:
    interests = effective_normalized_interests(normalized_context)
    total_budget = max(0, min(DEFAULT_QUERY_BUDGET, int(query_budget)))
    per_budget = max(1, min(DEFAULT_PER_INTEREST_QUERY_BUDGET, int(per_interest_query_budget)))
    raw_queries: list[dict[str, Any]] = []
    taxonomy_queries: list[dict[str, Any]] = []
    avoid_semantics: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_query(bucket: list[dict[str, Any]], text: str, interest: Mapping[str, Any], kind: str) -> bool:
        query = _bounded_query(text)
        normalized = _normalize_text(query)
        if not query or not normalized or normalized in seen:
            return False
        if len(raw_queries) + len(taxonomy_queries) >= total_budget:
            return False
        seen.add(normalized)
        bucket.append({
            "query": query,
            "kind": kind,
            "raw_interest": str(interest.get("source_text") or ""),
            "taxonomy_label": str(interest.get("label") or ""),
            "query_fingerprint": stable_fingerprint({
                "destination": destination,
                "query": normalized,
                "label": interest.get("label"),
                "kind": kind,
            }),
        })
        return True

    for interest in interests:
        if interest["polarity"] == "avoid":
            avoid_semantics.append({
                "raw_interest": interest["source_text"],
                "taxonomy_label": interest["label"],
                "retrieval_target": False,
            })
            continue
        used = 0
        if add_query(raw_queries, interest["source_text"], interest, "raw_interest"):
            used += 1
        for descriptor in INTEREST_RETRIEVAL_DESCRIPTORS.get(interest["label"], ()):
            if used >= per_budget:
                break
            if add_query(taxonomy_queries, descriptor, interest, "taxonomy_descriptor"):
                used += 1

    plan = {
        "destination": destination,
        "raw_queries": raw_queries,
        "taxonomy_queries": taxonomy_queries,
        "avoid_semantics": avoid_semantics,
        "query_budget": total_budget,
        "per_interest_query_budget": per_budget,
        "request_id": request_id,
        "turn_id": turn_id,
        "constraint_revision": constraint_revision,
        "constraint_hash": constraint_hash,
        "actuation_version": SOFT_PREFERENCE_ACTUATION_VERSION,
    }
    plan["fingerprint"] = stable_fingerprint(plan)
    return plan


def retrieval_queries(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for key in ("raw_queries", "taxonomy_queries")
        for item in (plan.get(key) or [])
        if isinstance(item, Mapping)
    ]


def build_candidate_evidence_matrix(
    ranked: list[ScoredPOI],
    records: list[Mapping[str, Any]],
    profile: TravelProfile,
    *,
    request_id: str,
    turn_id: str,
    constraint_revision: int | None,
    constraint_hash: str | None,
) -> dict[str, Any]:
    context = (profile.constraint_state or {}).get("normalized_interest_context") or {}
    interests = effective_normalized_interests(context, profile.interests)
    preferred = [item["label"] for item in interests if item["polarity"] == "prefer"]
    avoided = [item["label"] for item in interests if item["polarity"] == "avoid"]
    sources, retrieval = _candidate_sources(records)
    route_dimensions = _route_dimensions(records, {item.poi.poi_id for item in ranked})
    candidates: dict[str, Any] = {}
    hard_filter: dict[str, Any] = {}
    for item in ranked:
        poi = item.poi
        evidence_ids = sources.get(poi.poi_id, [])
        hard_gate = candidate_hard_gate(poi, profile, evidence_ids)
        hard_filter[poi.poi_id] = hard_gate
        if hard_gate["passed"] is not True:
            continue
        dimensions: dict[str, Any] = {}
        interest_dimension = _interest_dimension(
            poi, preferred, avoided, retrieval.get(poi.poi_id, []), evidence_ids
        )
        dimensions["user_interest_match"] = interest_dimension
        dimensions["price"] = _price_dimension(poi, evidence_ids)
        dimensions["pace"] = _pace_dimension(poi, profile, evidence_ids)
        dimensions["accessibility"] = _accessibility_dimension(poi, evidence_ids)
        dimensions["location"] = _location_dimension(poi, profile, evidence_ids)
        dimensions["activity_density"] = _unknown_dimension("candidate_density_not_evidenced")
        dimensions.update(route_dimensions.get(poi.poi_id, {}))
        for name in ("commute_time", "walking_load", "transfer_count"):
            dimensions.setdefault(name, _unknown_dimension("common_route_anchor_not_evidenced"))
        candidates[poi.poi_id] = {
            "candidate_id": poi.poi_id,
            "hard_constraints_passed": True,
            "hard_constraint_status": hard_gate["status"],
            "base_score": item.score,
            "dimensions": dimensions,
        }
    _fail_closed_on_mixed_units(candidates)
    matrix = {
        "request_id": request_id,
        "turn_id": turn_id,
        "constraint_revision": constraint_revision,
        "constraint_hash": constraint_hash,
        "normalized_interests": interests,
        "retrieved_candidate_ids": [item.poi.poi_id for item in ranked],
        "hard_filtered_candidate_ids": list(candidates),
        "hard_filter": hard_filter,
        "candidates": candidates,
        "actuation_version": SOFT_PREFERENCE_ACTUATION_VERSION,
    }
    matrix["fingerprint"] = stable_fingerprint(matrix)
    return matrix


def preference_candidates_from_matrix(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for candidate_id, raw in (matrix.get("candidates") or {}).items():
        if not isinstance(raw, Mapping) or raw.get("hard_constraints_passed") is not True:
            continue
        item: dict[str, Any] = {
            "candidate_id": str(candidate_id),
            "hard_constraints_passed": True,
        }
        for dimension, evidence in (raw.get("dimensions") or {}).items():
            if not isinstance(evidence, Mapping):
                continue
            if evidence.get("evidence_status") == "unknown" or evidence.get("value") is None:
                continue
            item[str(dimension)] = evidence.get("value")
        compact.append(item)
    return compact


def apply_preference_score(
    ranked: list[ScoredPOI],
    matrix: Mapping[str, Any],
    policy: PlanningPolicy,
    *,
    dimension_delta_cap: float = DEFAULT_DIMENSION_DELTA_CAP,
    total_delta_cap: float = DEFAULT_TOTAL_DELTA_CAP,
) -> tuple[list[ScoredPOI], dict[str, Any]]:
    evidence_by_id = matrix.get("candidates") or {}
    ranked = [item for item in ranked if item.poi.poi_id in evidence_by_id]
    base_index = {item.poi.poi_id: index for index, item in enumerate(ranked)}
    policy_index = {candidate_id: index for index, candidate_id in enumerate(policy.candidate_order)}
    values_by_dimension: dict[str, dict[str, float]] = {}
    for priority in policy.priorities:
        values: dict[str, float] = {}
        for item in ranked:
            evidence = (
                (evidence_by_id.get(item.poi.poi_id) or {}).get("dimensions") or {}
            ).get(priority.dimension) or {}
            value = evidence.get("value") if isinstance(evidence, Mapping) else None
            if evidence.get("evidence_status") != "unknown" and _is_number(value):
                values[item.poi.poi_id] = float(value)
        values_by_dimension[priority.dimension] = values

    trace: dict[str, Any] = {}
    adjusted: list[ScoredPOI] = []
    for item in ranked:
        candidate_id = item.poi.poi_id
        adjustments: list[dict[str, Any]] = []
        total = 0.0
        for priority in policy.priorities:
            values = values_by_dimension.get(priority.dimension) or {}
            if candidate_id not in values or len(values) < 2:
                continue
            low, high = min(values.values()), max(values.values())
            if high <= low:
                continue
            normalized = (values[candidate_id] - low) / (high - low)
            utility = 1.0 - normalized if priority.direction == "minimize" else normalized
            delta = min(
                dimension_delta_cap,
                max(0.0, dimension_delta_cap * priority.importance * utility),
            )
            remaining = max(0.0, total_delta_cap - total)
            delta = min(delta, remaining)
            evidence = evidence_by_id[candidate_id]["dimensions"][priority.dimension]
            adjustments.append({
                "dimension": priority.dimension,
                "direction": priority.direction,
                "raw_value": values[candidate_id],
                "unit": evidence.get("unit"),
                "normalized_utility": round(utility, 6),
                "importance": priority.importance,
                "delta": round(delta, 6),
                "evidence_ids": list(evidence.get("evidence_ids") or []),
            })
            total += delta
        total = min(total_delta_cap, total)
        adjusted_score = round(item.score + total, 6)
        reasons = list(item.reasons)
        if total > 0:
            reasons.append(f"软偏好确定性调整 +{total:.3f}")
        adjusted.append(replace(item, score=adjusted_score, reasons=reasons))
        trace[candidate_id] = {
            "candidate_id": candidate_id,
            "base_score": item.score,
            "dimension_adjustments": adjustments,
            "total_soft_delta": round(total, 6),
            "adjusted_score": adjusted_score,
        }
    adjusted.sort(key=lambda item: (
        -item.score,
        policy_index.get(item.poi.poi_id, len(policy_index)),
        base_index[item.poi.poi_id],
        item.poi.poi_id,
    ))
    return adjusted, trace


def build_actuation_contract(
    *,
    matrix: Mapping[str, Any],
    retrieval_plans: list[Mapping[str, Any]],
    policy: PlanningPolicy,
    score_trace: Mapping[str, Any],
    base_ranking: list[str],
    final_ranking: list[str],
) -> dict[str, Any]:
    merged_retrieval = _merge_retrieval_plans(retrieval_plans)
    candidate_evidence = dict(matrix.get("candidates") or {})
    hard_filter = dict(matrix.get("hard_filter") or {})
    retrieved_ids = list(
        matrix.get("retrieved_candidate_ids") or hard_filter or candidate_evidence
    )
    hard_legal_ids = list(
        matrix.get("hard_filtered_candidate_ids") or candidate_evidence
    )
    policy_payload = policy.to_dict()
    policy_dimensions = [item.dimension for item in policy.priorities]
    evidence_complete_ids = [
        candidate_id
        for candidate_id, candidate in candidate_evidence.items()
        if all(
            _dimension_is_known((candidate.get("dimensions") or {}).get(dimension))
            for dimension in policy_dimensions
        )
    ]
    preference_eligible_ids = [
        candidate_id
        for candidate_id in evidence_complete_ids
        if candidate_id in policy.candidate_order
    ]
    tool_input_fingerprints = sorted({
        str(item.get("query_fingerprint"))
        for key in ("raw_queries", "taxonomy_queries")
        for item in merged_retrieval.get(key) or []
        if isinstance(item, Mapping) and item.get("query_fingerprint")
    })
    fingerprints = {
        "intent_input_fingerprint": stable_fingerprint([
            {
                "source_text": item.get("source_text"),
                "polarity": item.get("polarity"),
                "source_turn": item.get("source_turn"),
                "basis": item.get("basis"),
            }
            for item in matrix.get("normalized_interests") or []
        ]),
        "normalized_interest_fingerprint": stable_fingerprint(
            matrix.get("normalized_interests") or []
        ),
        "retrieval_plan_fingerprint": stable_fingerprint(merged_retrieval),
        "tool_input_fingerprints": tool_input_fingerprints,
        "retrieved_candidate_set_fingerprint": stable_fingerprint(retrieved_ids),
        "hard_filtered_candidate_set_fingerprint": stable_fingerprint(hard_legal_ids),
        "evidence_matrix_fingerprint": matrix.get("fingerprint")
        or stable_fingerprint(matrix),
        "preference_policy_fingerprint": stable_fingerprint(policy_payload),
        "score_trace_fingerprint": stable_fingerprint(score_trace),
        "adjusted_ranking_fingerprint": stable_fingerprint(final_ranking),
        "planner_input_fingerprint": stable_fingerprint({
            "candidate_ids": final_ranking,
            "score_trace": score_trace,
        }),
        "final_artifact_fingerprint": None,
    }
    contract = {
        "actuation_version": SOFT_PREFERENCE_ACTUATION_VERSION,
        "request_id": matrix.get("request_id"),
        "turn_id": matrix.get("turn_id"),
        "constraint_revision": matrix.get("constraint_revision"),
        "constraint_hash": matrix.get("constraint_hash"),
        "normalized_interests": list(matrix.get("normalized_interests") or []),
        "retrieval_plan": merged_retrieval,
        "candidate_evidence": candidate_evidence,
        "hard_filter": hard_filter,
        "evidence_matrix_fingerprint": matrix.get("fingerprint"),
        "planning_policy": policy_payload,
        "score_trace": dict(score_trace),
        "base_ranking": list(base_ranking),
        "final_ranking": list(final_ranking),
        "selected_candidate_ids": [],
        "fingerprints": fingerprints,
        "funnel": {
            "retrieval_queries": len(merged_retrieval.get("raw_queries") or [])
            + len(merged_retrieval.get("taxonomy_queries") or []),
            "retrieved_candidates": len(retrieved_ids),
            "hard_legal_candidates": len(hard_legal_ids),
            "evidence_complete_candidates": len(evidence_complete_ids),
            "preference_eligible_candidates": len(preference_eligible_ids),
            "planner_candidates": len(final_ranking),
            "selected_candidates": 0,
        },
        "causal_diagnostic": {
            "preference_actuated": True,
            "ranking_changed": list(base_ranking) != list(final_ranking),
            "rank_changes": _rank_changes(base_ranking, final_ranking),
            "unknown_dimensions_received_positive_delta": False,
            "hard_constraint_candidates_revived": False,
        },
    }
    contract["fingerprint"] = stable_fingerprint(contract)
    return contract


def finalize_selection_provenance(
    contract: Mapping[str, Any],
    selected_candidate_ids: list[str],
    *,
    source_ranked_artifact_id: str | None,
    itinerary_payload: Mapping[str, Any] | None = None,
    state_version: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = json.loads(json.dumps(contract, ensure_ascii=False, default=str))
    selected = list(dict.fromkeys(str(item) for item in selected_candidate_ids if item))
    final_ranking = list(result.get("final_ranking") or [])
    base_ranking = list(result.get("base_ranking") or [])
    result["selected_candidate_ids"] = selected
    result["final_selection_provenance"] = {
        "source_ranked_artifact_id": source_ranked_artifact_id,
        "selected_ranks": {
            item: final_ranking.index(item) + 1
            for item in selected if item in final_ranking
        },
        "all_selected_from_legal_ranking": all(item in final_ranking for item in selected),
    }
    diagnostic = dict(result.get("causal_diagnostic") or {})
    diagnostic["selection_changed_from_base_top_n"] = (
        selected != [item for item in base_ranking if item in selected][:len(selected)]
    )
    result["causal_diagnostic"] = diagnostic
    funnel = dict(result.get("funnel") or {})
    funnel["selected_candidates"] = len(selected)
    result["funnel"] = funnel
    fingerprints = dict(result.get("fingerprints") or {})
    fingerprints["final_artifact_fingerprint"] = stable_fingerprint({
        "itinerary": itinerary_payload or {},
        "state_version": state_version or {},
        "selected_candidate_ids": selected,
        "source_ranked_artifact_id": source_ranked_artifact_id,
        "planner_input_fingerprint": fingerprints.get("planner_input_fingerprint"),
    })
    result["fingerprints"] = fingerprints
    result.pop("fingerprint", None)
    result["fingerprint"] = stable_fingerprint(result)
    return result


def hard_legal_ranked_from_matrix(
    ranked: list[ScoredPOI], matrix: Mapping[str, Any]
) -> list[ScoredPOI]:
    """Return only candidates whose candidate-level hard gate did not fail."""
    legal_ids = set((matrix.get("candidates") or {}).keys())
    return [item for item in ranked if item.poi.poi_id in legal_ids]


def with_planner_input_fingerprint(
    contract: Mapping[str, Any], ranked: list[ScoredPOI]
) -> dict[str, Any]:
    """Bind the contract to the exact ranked payload consumed by Planner."""
    result = json.loads(json.dumps(contract, ensure_ascii=False, default=str))
    planner_input = [
        {
            "candidate_id": item.poi.poi_id,
            "score": item.score,
            "reasons": list(item.reasons),
        }
        for item in ranked
    ]
    fingerprints = dict(result.get("fingerprints") or {})
    fingerprints["planner_input_fingerprint"] = stable_fingerprint(planner_input)
    result["fingerprints"] = fingerprints
    funnel = dict(result.get("funnel") or {})
    funnel["planner_candidates"] = len(planner_input)
    result["funnel"] = funnel
    result.pop("fingerprint", None)
    result["fingerprint"] = stable_fingerprint(result)
    return result


def ranked_from_contract(
    ranked: list[ScoredPOI], contract: Mapping[str, Any]
) -> list[ScoredPOI]:
    """Restore adjusted scores/order for Planner and same-constraint rework."""
    order = {candidate_id: index for index, candidate_id in enumerate(contract.get("final_ranking") or [])}
    trace = contract.get("score_trace") or {}
    restored: list[ScoredPOI] = []
    for item in ranked:
        score_item = trace.get(item.poi.poi_id) if isinstance(trace, Mapping) else None
        score = score_item.get("adjusted_score") if isinstance(score_item, Mapping) else None
        restored.append(replace(item, score=float(score) if _is_number(score) else item.score))
    restored.sort(key=lambda item: (order.get(item.poi.poi_id, len(order)), -item.score, item.poi.poi_id))
    return restored


def retrieval_provenance_for_pois(
    query: Mapping[str, Any], pois: Iterable[POI]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for poi in pois:
        result.setdefault(poi.poi_id, []).append({
            "raw_interest": query.get("raw_interest"),
            "taxonomy_label": query.get("taxonomy_label"),
            "query_fingerprint": query.get("query_fingerprint"),
            "tool_evidence_id": f"{poi.source}:{poi.source_poi_id or poi.poi_id}",
        })
    return result


def _candidate_sources(
    records: list[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    sources: dict[str, list[str]] = {}
    retrieval: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        artifact_id = str(record.get("artifact_id") or "")
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        for raw in (payload or {}).get("pois") or (payload or {}).get("restaurants") or []:
            if isinstance(raw, Mapping) and raw.get("poi_id"):
                sources.setdefault(str(raw["poi_id"]), []).append(artifact_id)
        for candidate_id, entries in ((payload or {}).get("retrieval_provenance") or {}).items():
            if isinstance(entries, list):
                retrieval.setdefault(str(candidate_id), []).extend(
                    dict(item) for item in entries if isinstance(item, Mapping)
                )
    return (
        {key: list(dict.fromkeys(value)) for key, value in sources.items()},
        retrieval,
    )


def _route_dimensions(
    records: list[Mapping[str, Any]], candidate_ids: set[str]
) -> dict[str, dict[str, Any]]:
    by_anchor: dict[str, dict[str, tuple[Mapping[str, Any], str]]] = {}
    for record in records:
        if record.get("kind") != "routes":
            continue
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        if payload.get("evidence_status") not in {"provider_verified", "deterministic_estimate"}:
            continue
        origin, destination = str(payload.get("origin_poi_id") or ""), str(payload.get("destination_poi_id") or "")
        if destination in candidate_ids and origin not in candidate_ids:
            by_anchor.setdefault(origin, {})[destination] = (payload, str(record.get("artifact_id") or ""))
        elif origin in candidate_ids and destination not in candidate_ids:
            by_anchor.setdefault(destination, {})[origin] = (payload, str(record.get("artifact_id") or ""))
    complete = [(anchor, values) for anchor, values in by_anchor.items() if candidate_ids.issubset(values)]
    if not complete:
        return {}
    _, routes = sorted(complete, key=lambda item: item[0])[0]
    result: dict[str, dict[str, Any]] = {}
    for candidate_id, (route, artifact_id) in routes.items():
        result[candidate_id] = {
            "commute_time": _dimension(
                route.get("duration_min"), "minutes", route.get("evidence_status"),
                [artifact_id], "common_anchor_route", 1.0,
            ),
            "walking_load": (
                _dimension(
                    float(route["walking_distance_km"]) * 12.0, "minutes",
                    route.get("evidence_status"), [artifact_id],
                    "provider_walking_distance_normalized", 0.9,
                ) if _is_number(route.get("walking_distance_km"))
                else _unknown_dimension("walking_distance_not_returned")
            ),
            "transfer_count": (
                _dimension(
                    route.get("transfer_count", route.get("transfers")), "count",
                    route.get("evidence_status"), [artifact_id], "provider_route", 1.0,
                ) if _is_number(route.get("transfer_count", route.get("transfers")))
                else _unknown_dimension("transfer_count_not_returned")
            ),
        }
    return result


def _interest_dimension(
    poi: POI,
    preferred: list[str],
    avoided: list[str],
    retrieval: list[dict[str, Any]],
    evidence_ids: list[str],
) -> dict[str, Any]:
    controlled_tags = {
        label for raw in [poi.category, *poi.tags]
        if (label := _TAG_TO_INTEREST.get(_normalize_text(raw))) is not None
    }
    retrieved_labels = {
        str(item.get("taxonomy_label")) for item in retrieval
        if item.get("taxonomy_label") in CANONICAL_INTERESTS
    }
    evidenced = controlled_tags | retrieved_labels
    if not evidenced or (not preferred and not avoided):
        return _unknown_dimension("controlled_interest_evidence_missing")
    positive = (
        len(set(preferred).intersection(evidenced)) / len(set(preferred))
        if preferred else 1.0
    )
    negative = len(set(avoided).intersection(evidenced)) / max(1, len(set(avoided))) if avoided else 0.0
    ids = [*evidence_ids, *(str(item.get("tool_evidence_id")) for item in retrieval if item.get("tool_evidence_id"))]
    return _dimension(
        max(0.0, min(1.0, positive - negative)), "ratio", "verified",
        list(dict.fromkeys(ids)),
        "verified_retrieval_provenance" if retrieval else "controlled_taxonomy_overlap",
        1.0,
    )


def _price_dimension(poi: POI, evidence_ids: list[str]) -> dict[str, Any]:
    if _is_number(poi.average_cost):
        return _dimension(poi.average_cost, "cny_per_person", "provider_verified", evidence_ids, "provider_average_cost", 1.0)
    if poi.price_level in _PRICE_LEVEL:
        return _dimension(_PRICE_LEVEL[poi.price_level], "price_level_ordinal", "verified", evidence_ids, "controlled_price_level", 0.75)
    return _unknown_dimension("price_not_evidenced")


def candidate_hard_gate(
    poi: POI, profile: TravelProfile, evidence_ids: list[str]
) -> dict[str, Any]:
    """Audit candidate-local hard constraints without treating missing data as failure."""
    reasons: list[str] = []
    passed = True
    if poi.verification_status != "verified":
        passed = False
        reasons.append("candidate_not_verified")
    if profile.destination and poi.city and poi.city != profile.destination:
        passed = False
        reasons.append("destination_mismatch")
    if poi_avoid_match(poi, profile) is not None:
        passed = False
        reasons.append("hard_avoid_match")

    party_size = max(1, int(profile.party_size or 1))
    known_candidate_cost = None
    if _is_number(poi.average_cost):
        known_candidate_cost = round(float(poi.average_cost) * party_size, 2)
        if (
            _is_number(profile.budget_limit)
            and known_candidate_cost > float(profile.budget_limit)
        ):
            passed = False
            reasons.append("candidate_cost_exceeds_total_budget")
    elif _is_number(profile.budget_limit):
        reasons.append("candidate_cost_unknown")

    if not passed:
        status = "failed"
    elif "candidate_cost_unknown" in reasons:
        status = "partial_pass"
    else:
        status = "verified_pass"
    return {
        "passed": passed,
        "status": status,
        "reason_codes": reasons,
        "evidence_ids": list(evidence_ids),
        "known_candidate_cost_cny": known_candidate_cost,
        "budget_limit_cny": profile.budget_limit,
        "party_size": party_size,
    }


def _dimension_is_known(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evidence_status") != "unknown"
        and value.get("value") is not None
    )


def _pace_dimension(poi: POI, profile: TravelProfile, evidence_ids: list[str]) -> dict[str, Any]:
    if not _is_number(poi.estimated_duration_min) or poi.estimated_duration_min <= 0:
        return _unknown_dimension("duration_not_evidenced")
    limit = {"relaxed": 150.0, "standard": 180.0, "intensive": 240.0}.get(profile.pace, 180.0)
    fit = max(0.0, min(1.0, 1.0 - abs(float(poi.estimated_duration_min) - limit) / limit))
    return _dimension(fit, "ratio", "verified", evidence_ids, "deterministic_duration_fit", 0.9)


def _accessibility_dimension(poi: POI, evidence_ids: list[str]) -> dict[str, Any]:
    tags = {_normalize_text(item) for item in poi.tags}
    if any(_normalize_text(item) in tags for item in _ACCESSIBILITY_TAGS):
        return _dimension(1.0, "ratio", "verified", evidence_ids, "controlled_accessibility_tag", 0.9)
    return _unknown_dimension("accessibility_not_evidenced")


def _location_dimension(poi: POI, profile: TravelProfile, evidence_ids: list[str]) -> dict[str, Any]:
    area = str(profile.hotel_area or (profile.constraint_state or {}).get("lodging_area") or "").strip()
    if area and poi.address and _normalize_text(area) in _normalize_text(poi.address):
        return _dimension(1.0, "ratio", "provider_verified", evidence_ids, "provider_address_area_match", 0.9)
    return _unknown_dimension("preferred_area_match_not_evidenced")


def _dimension(
    value: Any,
    unit: str | None,
    status: Any,
    evidence_ids: Iterable[str],
    source: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "evidence_status": str(status or "verified"),
        "evidence_ids": [item for item in dict.fromkeys(str(item) for item in evidence_ids) if item],
        "source": source,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
    }


def _unknown_dimension(reason: str) -> dict[str, Any]:
    return {
        "value": None,
        "unit": None,
        "evidence_status": "unknown",
        "evidence_ids": [],
        "source": reason,
        "confidence": 0.0,
    }


def _fail_closed_on_mixed_units(candidates: dict[str, Any]) -> None:
    dimensions = {
        name
        for candidate in candidates.values()
        for name in (candidate.get("dimensions") or {})
    }
    for name in dimensions:
        known = [
            candidate["dimensions"][name]
            for candidate in candidates.values()
            if name in candidate.get("dimensions", {})
            and candidate["dimensions"][name].get("evidence_status") != "unknown"
            and candidate["dimensions"][name].get("value") is not None
        ]
        units = {str(item.get("unit") or "") for item in known}
        if len(units) <= 1:
            continue
        for candidate in candidates.values():
            if name in candidate.get("dimensions", {}):
                candidate["dimensions"][name] = _unknown_dimension(
                    "incompatible_units_not_normalized"
                )


def _merge_retrieval_plans(plans: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not plans:
        return {"raw_queries": [], "taxonomy_queries": [], "retrieval_misses": [], "fingerprints": []}
    return {
        "raw_queries": _unique_dicts(item for plan in plans for item in plan.get("raw_queries") or []),
        "taxonomy_queries": _unique_dicts(item for plan in plans for item in plan.get("taxonomy_queries") or []),
        "avoid_semantics": _unique_dicts(item for plan in plans for item in plan.get("avoid_semantics") or []),
        "retrieval_misses": _unique_dicts(item for plan in plans for item in plan.get("retrieval_misses") or []),
        "fingerprints": list(dict.fromkeys(str(plan.get("fingerprint")) for plan in plans if plan.get("fingerprint"))),
    }


def _unique_dicts(values: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        key = stable_fingerprint(value)
        if key not in seen:
            seen.add(key)
            result.append(dict(value))
    return result


def _rank_changes(before: list[str], after: list[str]) -> list[dict[str, Any]]:
    old = {candidate_id: index + 1 for index, candidate_id in enumerate(before)}
    return [
        {"candidate_id": candidate_id, "base_rank": old.get(candidate_id), "final_rank": index + 1,
         "rank_delta": (old.get(candidate_id) or index + 1) - (index + 1)}
        for index, candidate_id in enumerate(after)
        if old.get(candidate_id) != index + 1
    ]


def _bounded_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:120].strip()


def _bounded_number(value: Any, *, default: float) -> float:
    if not _is_number(value):
        return default
    return max(0.0, min(1.0, float(value)))


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
