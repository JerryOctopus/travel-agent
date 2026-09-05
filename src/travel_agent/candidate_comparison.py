"""Evidence contract for non-planner candidate comparisons.

The comparison artifact is deliberately derived from current-turn worker tool
artifacts.  Worker prose, prior-turn artifacts, empty payloads and metadata-only
``artifact_id`` placeholders are never accepted as comparison evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from statistics import fmean
from typing import Any, Iterable, Mapping


DIMENSION_ALIASES = {
    "accessibility": "accessibility",
    "transport": "accessibility",
    "commute": "accessibility",
    "通勤": "accessibility",
    "交通": "accessibility",
    "accessibility_needs": "accessibility_needs",
    "accessible": "accessibility_needs",
    "适老": "accessibility_needs",
    "无障碍": "accessibility_needs",
    "cost": "cost",
    "price": "cost",
    "价格": "cost",
    "成本": "cost",
    "ambience": "ambience",
    "area_suitability": "area_suitability",
    "area suitability": "area_suitability",
    "区域适配": "area_suitability",
    "overall_fit": "overall_fit",
}

ROUTE_DIMENSIONS = frozenset({"accessibility", "accessibility_needs"})
AREA_DIMENSIONS = frozenset({"ambience", "area_suitability", "overall_fit"})
SUCCESS_STATUSES = frozenset({"completed", "completed_with_warnings"})


def comparison_candidates(state: Mapping[str, Any]) -> list[str]:
    values = (
        state.get("comparison_candidates")
        or state.get("compare_lodging_areas")
        or state.get("candidate_attractions")
        or []
    )
    return list(
        dict.fromkeys(
            text
            for value in values
            if (text := str(value).strip())
        )
    )


def comparison_dimensions(state: Mapping[str, Any]) -> list[str]:
    raw = state.get("comparison_dimensions") or ["overall_fit"]
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    normalized = [
        DIMENSION_ALIASES.get(str(value).strip().casefold(), str(value).strip().casefold())
        for value in values
        if str(value).strip()
    ]
    return list(dict.fromkeys(normalized or ["overall_fit"]))


def comparison_core_dimensions(state: Mapping[str, Any]) -> list[str]:
    """Return hard comparison dimensions, defaulting to the first requested one.

    The ordered dimension list comes from the user's wording.  Callers that
    have an explicit product contract may provide ``comparison_core_dimensions``;
    otherwise the leading requested dimension is the primary decision axis and
    later dimensions are informative/secondary.
    """
    dimensions = comparison_dimensions(state)
    raw = state.get("comparison_core_dimensions") or state.get("core_comparison_dimensions")
    if raw in (None, "", [], {}):
        return dimensions[:1]
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    normalized = [
        DIMENSION_ALIASES.get(str(value).strip().casefold(), str(value).strip().casefold())
        for value in values
        if str(value).strip()
    ]
    return [value for value in dict.fromkeys(normalized) if value in dimensions] or dimensions[:1]


def agents_for_comparison(task_brief: str, state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only evidence workers justified by the requested dimensions."""
    text = task_brief.casefold()
    if state.get("comparison_dimensions"):
        dimensions = set(comparison_dimensions(state))
    else:
        inferred: list[str] = []
        if re.search(r"交通|路线|通勤|耗时|距离|可达|方便", text):
            inferred.append("accessibility")
        if re.search(r"适老|老人|父母|无障碍|少走|步行|换乘", text):
            inferred.append("accessibility_needs")
        if re.search(r"价格|预算|成本|便宜|贵", text):
            inferred.append("cost")
        if re.search(r"安静|热闹|氛围|环境|区域适合", text):
            inferred.append("ambience")
        dimensions = set(inferred or ["overall_fit"])
    lodging = bool(state.get("compare_lodging_areas") or re.search(r"住宿|酒店|住哪|住在", text))
    restaurant = bool(
        state.get("specific_restaurant_recommendation")
        or re.search(r"餐厅|饭店|用餐|吃饭|早餐|午餐|晚餐|聚餐", text)
    )
    no_inventory = bool(state.get("no_live_inventory_required"))

    agents: list[str] = []
    if lodging and not no_inventory:
        agents.append("hotel")
    elif restaurant:
        agents.append("restaurant")
    # Area/POI evidence is the safe source for ambience and suitability.  It
    # also gives accessibility workers grounded endpoints/features without
    # turning an area comparison into a hotel-inventory recommendation.
    if (
        dimensions.intersection(AREA_DIMENSIONS)
        or "accessibility_needs" in dimensions
    ) and not (restaurant or (lodging and not no_inventory)):
        agents.append("attraction")
    if dimensions.intersection(ROUTE_DIMENSIONS):
        agents.append("transport")
    if (
        any(state.get(key) not in (None, "", [], {}) for key in ("location", "location_anchor"))
        and (
            state.get("walking_time_max_min") is not None
            or re.search(r"附近|周边|一带|步行|距离", text)
        )
    ):
        # Discovery requests are still anchored comparisons even before the
        # candidate names exist.  Transport must therefore be eligible for a
        # later wave after attraction/restaurant evidence supplies endpoints.
        agents.append("transport")
    if "cost" in dimensions:
        if restaurant:
            agents.append("restaurant")
        elif lodging and not no_inventory:
            agents.append("hotel")
        else:
            agents.append("attraction")
    if not agents:
        agents.append("restaurant" if restaurant else "attraction")
    # Planner is intentionally not representable in this contract.
    return tuple(dict.fromkeys(agent for agent in agents if agent != "planner"))


@dataclass
class ComparisonEvidenceSet:
    evidence: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    metrics: dict[str, dict[str, list[tuple[float, str, str]]]] = field(default_factory=dict)
    dimension_strategies: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.missing

    def covered_cells(self) -> set[tuple[str, str]]:
        return {
            (str(item.get("candidate") or ""), str(item.get("dimension") or ""))
            for item in self.evidence
            if item.get("candidate") and item.get("dimension")
        }

    def complete_dimensions(
        self,
        candidates: Iterable[str],
        dimensions: Iterable[str],
    ) -> list[str]:
        names = list(candidates)
        covered = self.covered_cells()
        return [
            dimension
            for dimension in dimensions
            if names and all((candidate, dimension) in covered for candidate in names)
        ]

    def comparable_dimensions(
        self,
        candidates: Iterable[str],
        dimensions: Iterable[str],
    ) -> list[str]:
        # A shared numeric metric permits ranking, but structured qualitative
        # facts are still directly comparable and can support a bounded
        # "no winner" conclusion.  Completeness remains candidate-wide.
        return self.complete_dimensions(candidates, dimensions)


def collect_comparison_evidence(
    store: Any,
    results: Iterable[Any],
    state: Mapping[str, Any],
    *,
    request_id: str | None = None,
) -> ComparisonEvidenceSet:
    """Bind valid current-turn worker artifacts to candidate × dimension."""
    candidates = comparison_candidates(state)
    dimensions = comparison_dimensions(state)
    target = str(state.get("target_anchor") or "").strip()
    bound: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for result in results:
        if str(getattr(result, "status", "")) not in SUCCESS_STATUSES:
            continue
        result_request = str(getattr(result, "request_id", "") or "")
        result_task = str(getattr(result, "task_id", "") or "")
        result_agent = str(getattr(result, "agent", "") or "")
        result_payload = getattr(result, "payload", None) or {}
        scope = (
            result_payload.get("comparison_scope")
            if isinstance(result_payload, Mapping)
            else None
        ) or {}
        scoped_candidate = str(scope.get("candidate") or "").strip()
        scoped_dimensions = {
            str(value).strip()
            for value in scope.get("dimensions") or []
            if str(value).strip()
        }
        if request_id and result_request != request_id:
            continue
        for item in getattr(result, "evidence", None) or []:
            if not isinstance(item, Mapping):
                continue
            artifact_id = str(item.get("artifact_id") or "")
            record = store.get_record(artifact_id) if artifact_id else None
            if not _record_owned_by_result(
                record, result_request, result_task, result_agent, current_request_id=request_id
            ):
                continue
            payload = record.get("payload") or {}
            kind = str(record.get("kind") or "")
            if not isinstance(payload, Mapping) or not _meaningful_payload(kind, payload):
                continue
            for candidate in candidates:
                # A valid unanchored route is evidence for both declared
                # endpoints.  Non-route artifacts remain owned by the scoped
                # candidate so a broad provider response cannot fill cells by
                # accident.
                if scoped_candidate and candidate != scoped_candidate and kind != "routes":
                    continue
                facts = _candidate_facts(kind, payload, candidate, target, candidates)
                if not facts:
                    continue
                for dimension in dimensions:
                    if scoped_dimensions and dimension not in scoped_dimensions:
                        continue
                    if not _facts_support_dimension(kind, facts, dimension, target=target):
                        continue
                    key = (candidate, dimension, artifact_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence = {
                        "candidate": candidate,
                        "dimension": dimension,
                        "source_artifact_id": artifact_id,
                        "source_kind": kind,
                        "source_agent": result_agent,
                        "source_task_id": result_task,
                        "facts": facts,
                    }
                    bound.append(evidence)
    bound, strategies = normalize_comparison_evidence_rows(
        bound,
        candidates,
        dimensions,
        target_anchor=target,
    )
    metrics: dict[str, dict[str, list[tuple[float, str, str]]]] = {
        candidate: {dimension: [] for dimension in dimensions}
        for candidate in candidates
    }
    for item in bound:
        candidate = str(item["candidate"])
        dimension = str(item["dimension"])
        facts = item.get("facts") or {}
        metric = _comparison_metric(dimension, facts, source_kind=str(item.get("source_kind") or ""))
        if metric is not None:
            metric_name = _metric_name(dimension, facts, source_kind=str(item.get("source_kind") or ""))
            metrics[candidate][dimension].append(
                (metric, str(item["source_artifact_id"]), metric_name)
            )

    covered = {(item["candidate"], item["dimension"]) for item in bound}
    missing = [
        f"候选「{candidate}」的 {dimension} 证据"
        for candidate in candidates
        for dimension in dimensions
        if (candidate, dimension) not in covered
    ]
    if not candidates:
        missing.append("比较候选项")
    return ComparisonEvidenceSet(
        evidence=bound,
        missing=missing,
        metrics=metrics,
        dimension_strategies=strategies,
    )


def supported_recommendation(
    candidates: list[str],
    dimensions: list[str],
    evidence_set: ComparisonEvidenceSet,
    *,
    core_dimensions: list[str] | None = None,
) -> dict[str, Any] | None:
    """Rank only on dimensions that are complete and comparable for all candidates."""
    core = list(core_dimensions or dimensions[:1])
    complete = set(evidence_set.complete_dimensions(candidates, dimensions))
    if not candidates or any(dimension not in complete for dimension in core):
        return None
    points = {candidate: 0.0 for candidate in candidates}
    comparable: dict[str, dict[str, dict[str, Any]]] = {}
    usable_dimensions = evidence_set.comparable_dimensions(candidates, dimensions)
    for dimension in usable_dimensions:
        entries_by_candidate: dict[str, dict[str, list[tuple[float, str]]]] = {}
        for candidate in candidates:
            entries = evidence_set.metrics.get(candidate, {}).get(dimension) or []
            if not entries:
                entries_by_candidate = {}
                break
            grouped: dict[str, list[tuple[float, str]]] = {}
            for value, artifact_id, metric_name in entries:
                grouped.setdefault(metric_name, []).append((value, artifact_id))
            entries_by_candidate[candidate] = grouped
        if len(entries_by_candidate) != len(candidates):
            continue
        shared_metrics = set.intersection(
            *(set(grouped) for grouped in entries_by_candidate.values())
        )
        metric_name = next(
            (
                name
                for name in _PREFERRED_COMPARISON_METRICS.get(dimension, ())
                if name in shared_metrics
            ),
            next(iter(sorted(shared_metrics)), None),
        )
        if not metric_name:
            continue
        values: dict[str, dict[str, Any]] = {}
        for candidate, grouped in entries_by_candidate.items():
            metric_entries = grouped[metric_name]
            values[candidate] = {
                "value": fmean(value for value, _artifact_id in metric_entries),
                "metric": (
                    f"average_{metric_name}"
                    if evidence_set.dimension_strategies.get(dimension) == "candidate_route_matrix"
                    else metric_name
                ),
                "source_artifact_ids": list(dict.fromkeys(
                    artifact_id for _value, artifact_id in metric_entries
                )),
            }
        comparable[dimension] = values
        reverse = _metric_higher_is_better(metric_name, dimension)
        ordered = sorted(values, key=lambda name: values[name]["value"], reverse=reverse)
        for rank, candidate in enumerate(ordered):
            points[candidate] += len(candidates) - rank
    if not comparable or any(dimension not in comparable for dimension in core):
        source_ids = list(dict.fromkeys(
            item["source_artifact_id"]
            for item in evidence_set.evidence
            if item.get("dimension") in usable_dimensions
        ))
        return {
            "candidate": None,
            "decision": "完整维度只有定性事实，有限结论为暂不排序",
            "dimensions_used": usable_dimensions,
            "comparable_metrics": [],
            "basis": [],
            "source_artifact_ids": source_ids,
        }
    winner = max(candidates, key=lambda name: points[name])
    basis = []
    for dimension, values in comparable.items():
        winner_value = values[winner]
        artifact_ids = winner_value["source_artifact_ids"]
        basis.append(
            {
                "dimension": dimension,
                "candidate": winner,
                "metric": winner_value["metric"],
                "value": winner_value["value"],
                "source_artifact_id": artifact_ids[0],
                "source_artifact_ids": artifact_ids,
            }
        )
    source_ids = list(dict.fromkeys(
        artifact_id
        for values in comparable.values()
        for candidate_value in values.values()
        for artifact_id in candidate_value["source_artifact_ids"]
    ))
    return {
        "candidate": winner,
        "decision": f"基于 {len(comparable)} 个完整可比维度有限推荐 {winner}",
        "dimensions_used": list(comparable),
        "comparable_metrics": list(dict.fromkeys(
            item["metric"] for item in basis
        )),
        "basis": basis,
        "source_artifact_ids": source_ids,
    }


def comparison_hard_missing(
    state: Mapping[str, Any],
    evidence_set: ComparisonEvidenceSet,
) -> list[str]:
    """Return only fail-closed gaps; secondary gaps remain limitations."""
    candidates = comparison_candidates(state)
    dimensions = comparison_dimensions(state)
    core = comparison_core_dimensions(state)
    covered = evidence_set.covered_cells()
    missing = [
        f"核心维度：候选「{candidate}」的 {dimension} 证据"
        for dimension in core
        for candidate in candidates
        if (candidate, dimension) not in covered
    ]
    if candidates and not evidence_set.comparable_dimensions(candidates, dimensions):
        missing.append("至少一个覆盖全部候选的完整可比维度")
    if not candidates:
        missing.append("比较候选项")
    return list(dict.fromkeys(missing))


def entity_matches(expected: str, actual: str) -> bool:
    left = _canonical_entity(expected)
    right = _canonical_entity(actual)
    return bool(left and right and (left in right or right in left))


def accessibility_mode(target_anchor: str | None) -> str:
    return "anchored" if str(target_anchor or "").strip() else "unanchored"


def payload_has_candidate_transit_fact(
    kind: str,
    payload: Mapping[str, Any],
    candidate: str,
) -> bool:
    """Return whether one entity artifact supplies local transit facts."""
    if kind == "routes" or not isinstance(payload, Mapping):
        return False
    facts = _candidate_facts(kind, payload, candidate, "", [candidate])
    return bool(facts and _facts_support_dimension(
        kind,
        facts,
        "accessibility",
        target="",
    ))


def normalize_comparison_evidence_rows(
    rows: Iterable[Mapping[str, Any]],
    candidates: Iterable[str],
    dimensions: Iterable[str],
    *,
    target_anchor: str = "",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Apply the shared comparison evidence contract.

    This function intentionally works on artifact rows rather than store
    records so both the evidence collector and artifact validator can apply
    exactly the same anchored/unanchored endpoint and comparability rules.
    """
    names = [str(item).strip() for item in candidates if str(item).strip()]
    requested = {str(item).strip() for item in dimensions if str(item).strip()}
    target = str(target_anchor or "").strip()
    valid: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        candidate = str(row.get("candidate") or "").strip()
        dimension = str(row.get("dimension") or "").strip()
        kind = str(row.get("source_kind") or "")
        facts = row.get("facts")
        if candidate not in names or dimension not in requested or not isinstance(facts, Mapping):
            continue
        if not _row_semantically_supports(
            candidate,
            dimension,
            kind,
            facts,
            candidates=names,
            target=target,
        ):
            continue
        if kind == "routes":
            pair = _route_candidate_pair(facts, names)
            pair_key = tuple(sorted(_canonical_entity(item) for item in pair)) if pair else ()
            key = (candidate, dimension, "route", pair_key, str(facts.get("mode") or ""))
        else:
            key = (candidate, dimension, "entity", str(row.get("source_artifact_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        valid.append(row)

    strategies: dict[str, str] = {}
    for dimension in requested.intersection(ROUTE_DIMENSIONS):
        dimension_rows = [row for row in valid if row.get("dimension") == dimension]
        route_rows = _single_mode_route_rows(
            [row for row in dimension_rows if row.get("source_kind") == "routes"],
            names,
        )
        node_rows = [row for row in dimension_rows if row.get("source_kind") != "routes"]
        if target and dimension == "accessibility":
            strategy = "candidate_to_anchor"
            chosen = route_rows
        else:
            route_covered = {str(row.get("candidate") or "") for row in route_rows}
            node_covered = {str(row.get("candidate") or "") for row in node_rows}
            if names and all(name in node_covered for name in names):
                strategy = "candidate_transit_nodes"
                chosen = node_rows
            elif names and all(name in route_covered for name in names):
                strategy = "candidate_route_matrix"
                chosen = route_rows
            elif len(node_covered) >= len(route_covered):
                strategy = "candidate_transit_nodes"
                chosen = node_rows
            else:
                strategy = "candidate_route_matrix"
                chosen = route_rows
        strategies[dimension] = strategy
        valid = [row for row in valid if row.get("dimension") != dimension] + chosen
    return valid, strategies


def accessibility_contract(
    candidates: Iterable[str],
    evidence_set: ComparisonEvidenceSet,
    *,
    target_anchor: str = "",
) -> dict[str, Any]:
    """Describe the accessibility evidence strategy used by an artifact."""
    names = [str(item).strip() for item in candidates if str(item).strip()]
    mode = accessibility_mode(target_anchor)
    strategy = evidence_set.dimension_strategies.get("accessibility")
    contract: dict[str, Any] = {
        "mode": mode,
        "target_anchor": str(target_anchor or "").strip() or None,
        "evidence_strategy": strategy,
    }
    if strategy == "candidate_route_matrix":
        pairs: list[list[str]] = []
        modes: list[str] = []
        for row in evidence_set.evidence:
            if row.get("dimension") != "accessibility" or row.get("source_kind") != "routes":
                continue
            pair = _route_candidate_pair(row.get("facts") or {}, names)
            if pair and list(pair) not in pairs and list(reversed(pair)) not in pairs:
                pairs.append(list(pair))
            route_mode = str((row.get("facts") or {}).get("mode") or "").strip()
            if route_mode and route_mode not in modes:
                modes.append(route_mode)
        complete_pair_count = len(names) * (len(names) - 1) // 2
        contract["route_matrix"] = {
            "scope": "complete" if len(pairs) == complete_pair_count else "bounded",
            "pairs": pairs,
            "modes": modes,
        }
    return contract


def _record_owned_by_result(
    record: Mapping[str, Any] | None,
    result_request_id: str,
    task_id: str,
    agent: str,
    *,
    current_request_id: str | None = None,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    return bool(
        result_request_id
        and task_id
        and agent
        and str(record.get("request_id") or "") == result_request_id
        and str(record.get("task_id") or "") == task_id
        and str(record.get("agent") or "") == agent
        and (
            current_request_id is None
            or str(record.get("request_id") or "") == current_request_id
        )
        and agent not in {"engine", "planner"}
    )


def _meaningful_payload(kind: str, payload: Mapping[str, Any]) -> bool:
    if kind == "routes":
        return bool(
            payload.get("origin_name")
            and payload.get("destination_name")
            and any(payload.get(key) is not None for key in ("duration_min", "distance_km", "walking_distance_km"))
        )
    collections = _collections(payload)
    if collections:
        return True
    if kind == "budget":
        return any(payload.get(key) is not None for key in _COST_KEYS) and bool(
            payload.get("candidate") or payload.get("candidate_name") or payload.get("area")
        )
    return False


def _candidate_facts(
    kind: str,
    payload: Mapping[str, Any],
    candidate: str,
    target: str,
    candidates: Iterable[str],
) -> dict[str, Any]:
    if kind == "routes":
        endpoints = [
            str(payload.get("origin_name") or ""),
            str(payload.get("destination_name") or ""),
        ]
        if not any(entity_matches(candidate, endpoint) for endpoint in endpoints):
            return {}
        if target:
            if not any(entity_matches(target, endpoint) for endpoint in endpoints):
                return {}
        elif not _route_candidate_pair(payload, candidates):
            # In unanchored mode both endpoints must be declared candidates;
            # an unrelated POI cannot manufacture candidate coverage.
            return {}
        return _select_fields(payload, _ROUTE_FACT_KEYS)

    matched = [item for item in _collections(payload) if _item_matches(candidate, item)]
    scoped = any(
        entity_matches(candidate, str(payload.get(key) or ""))
        for key in ("candidate", "candidate_name", "area")
    )
    if not matched and not scoped:
        return {}
    facts: dict[str, Any] = {}
    if matched:
        facts["entities"] = [
            _select_fields(item, _ENTITY_FACT_KEYS)
            for item in matched[:5]
            if _select_fields(item, _ENTITY_FACT_KEYS)
        ]
    facts.update(_select_fields(payload, _PAYLOAD_FACT_KEYS))
    return facts if any(value not in (None, "", [], {}) for value in facts.values()) else {}


def _facts_support_dimension(
    kind: str,
    facts: Mapping[str, Any],
    dimension: str,
    *,
    target: str,
) -> bool:
    flat = list(_walk_items(facts))
    if dimension == "accessibility":
        if kind == "routes":
            return any(key in flat for key in ("duration_min", "distance_km"))
        return bool(_transport_entities(facts))
    if dimension == "accessibility_needs":
        if kind == "routes":
            return any(key in flat for key in ("walking_distance_km", "transfers"))
        return any(
            any(key in item for key in ("wheelchair_accessible", "step_free", "walking_distance_km", "transfers"))
            or any(marker in " ".join(map(str, item.get("tags") or [])) for marker in ("无障碍", "适老", "少走", "wheelchair", "step-free"))
            for item in facts.get("entities") or []
            if isinstance(item, Mapping)
        )
    if dimension == "cost":
        return any(key in flat for key in (*_COST_KEYS, "price_level"))
    if dimension in AREA_DIMENSIONS:
        return bool(facts.get("entities")) and any(
            key in flat for key in ("tags", "rating", "popularity", "category", "address", "ambience", "suitability")
        )
    return False


def _comparison_metric(
    dimension: str,
    facts: Mapping[str, Any],
    *,
    source_kind: str = "",
) -> float | None:
    values = dict(_numeric_values(facts))
    if dimension == "accessibility":
        if source_kind != "routes":
            nodes = _transport_entities(facts)
            if nodes:
                return float(sum(_transport_node_weight(item) for item in nodes))
        return _first_numeric(values, "duration_min", "distance_km")
    if dimension == "accessibility_needs":
        return _first_numeric(values, "walking_distance_km", "transfers")
    if dimension == "cost":
        direct = _first_numeric(values, *_COST_KEYS)
        if direct is not None:
            return direct
        levels = [str(value).casefold() for key, value in _walk_pairs(facts) if key == "price_level"]
        return {"free": 0.0, "low": 1.0, "mid": 2.0, "high": 3.0}.get(levels[0]) if levels else None
    if dimension in AREA_DIMENSIONS:
        return _first_numeric(values, "rating", "popularity", "suitability_score")
    return None


def _metric_name(
    dimension: str,
    facts: Mapping[str, Any],
    *,
    source_kind: str = "",
) -> str:
    if dimension == "accessibility" and source_kind != "routes" and _transport_entities(facts):
        return "transit_node_count"
    preferred = {
        "accessibility": ("duration_min", "distance_km"),
        "accessibility_needs": ("walking_distance_km", "transfers"),
        "cost": _COST_KEYS,
        "ambience": ("rating", "popularity", "suitability_score"),
        "area_suitability": ("rating", "popularity", "suitability_score"),
        "overall_fit": ("rating", "popularity", "suitability_score"),
    }.get(dimension, ())
    keys = {key for key, _value in _walk_pairs(facts)}
    return next((key for key in preferred if key in keys), "derived_score")


def _row_semantically_supports(
    candidate: str,
    dimension: str,
    kind: str,
    facts: Mapping[str, Any],
    *,
    candidates: Iterable[str],
    target: str,
) -> bool:
    if not _facts_support_dimension(kind, facts, dimension, target=target):
        return False
    if kind != "routes":
        return True
    endpoints = (
        str(facts.get("origin_name") or ""),
        str(facts.get("destination_name") or ""),
    )
    if not any(entity_matches(candidate, endpoint) for endpoint in endpoints):
        return False
    if target:
        return any(entity_matches(target, endpoint) for endpoint in endpoints)
    return bool(_route_candidate_pair(facts, candidates))


def _route_candidate_pair(
    facts: Mapping[str, Any],
    candidates: Iterable[str],
) -> tuple[str, str] | None:
    names = [str(item).strip() for item in candidates if str(item).strip()]
    endpoints = (
        str(facts.get("origin_name") or ""),
        str(facts.get("destination_name") or ""),
    )
    matched: list[str] = []
    for endpoint in endpoints:
        matches = [name for name in names if entity_matches(name, endpoint)]
        if len(matches) != 1:
            return None
        matched.append(matches[0])
    if matched[0] == matched[1]:
        return None
    return matched[0], matched[1]


def _single_mode_route_rows(
    rows: Iterable[dict[str, Any]],
    candidates: Iterable[str],
) -> list[dict[str, Any]]:
    """Choose one route mode; mixed modes are not a comparable matrix."""
    names = [str(item).strip() for item in candidates if str(item).strip()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mode = str((row.get("facts") or {}).get("mode") or "unspecified").strip()
        groups.setdefault(mode, []).append(row)
    if not groups:
        return []

    def coverage(mode_rows: list[dict[str, Any]]) -> set[str]:
        return {str(row.get("candidate") or "") for row in mode_rows}

    selected = next(
        (
            mode_rows
            for mode_rows in groups.values()
            if names and all(name in coverage(mode_rows) for name in names)
        ),
        None,
    )
    if selected is not None:
        return selected
    return max(groups.values(), key=lambda mode_rows: len(coverage(mode_rows)))


def _transport_entities(facts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in facts.get("entities") or []
        if isinstance(item, Mapping)
        and (
            str(item.get("category") or item.get("entity_type") or "").casefold()
            in {"transport", "transit", "station"}
            or any(
                marker in " ".join(map(str, item.get("tags") or [])).casefold()
                for marker in ("地铁", "公交", "交通", "station", "metro", "transit")
            )
        )
    ]


def _transport_node_weight(item: Mapping[str, Any]) -> int:
    for key in ("line_count", "lines_count", "route_count"):
        try:
            return max(1, int(item.get(key)))
        except (TypeError, ValueError):
            continue
    lines = item.get("lines") or item.get("transit_lines") or []
    return max(1, len(lines)) if isinstance(lines, (list, tuple, set)) else 1


def _metric_higher_is_better(metric_name: str, dimension: str) -> bool:
    return metric_name in {
        "transit_node_count", "line_count", "lines_count", "route_count",
        "rating", "popularity", "suitability_score",
    } or dimension in AREA_DIMENSIONS


def _collections(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for key in ("pois", "endpoint_pois", "items", "hotels", "restaurants", "areas"):
        for item in payload.get(key) or []:
            if isinstance(item, Mapping):
                values.append(item)
    return values


def _item_matches(candidate: str, item: Mapping[str, Any]) -> bool:
    values = [
        item.get("name"), item.get("canonical_name"), item.get("area"), item.get("address"),
        *(item.get("aliases") or []), *(item.get("tags") or []),
    ]
    return any(entity_matches(candidate, str(value or "")) for value in values)


def _select_fields(payload: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in keys
        if key in payload and payload[key] not in (None, "", [], {})
    }


def _canonical_entity(value: str) -> str:
    text = re.sub(r"[\s\-—–_·•:：,，。/\\()（）\[\]【】]+", "", str(value or "").casefold())
    return re.sub(r"(?:附近|周边|一带|商圈|区域)$", "", text)


def _walk_items(value: Any) -> Iterable[str]:
    for key, _item in _walk_pairs(value):
        yield key


def _walk_pairs(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_pairs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_pairs(item)


def _numeric_values(value: Any) -> Iterable[tuple[str, float]]:
    for key, item in _walk_pairs(value):
        if isinstance(item, bool):
            continue
        try:
            yield key, float(item)
        except (TypeError, ValueError):
            continue


def _first_numeric(values: Mapping[str, float], *keys: str) -> float | None:
    return next((values[key] for key in keys if key in values), None)


_ROUTE_FACT_KEYS = (
    "origin_name", "destination_name", "duration_min", "distance_km", "mode",
    "walking_distance_km", "transfers", "estimated_cost", "source", "evidence_status",
)
_ENTITY_FACT_KEYS = (
    "name", "canonical_name", "area", "address", "category", "entity_type", "tags",
    "rating", "popularity", "price_level", "average_cost", "cost", "price_cny",
    "wheelchair_accessible", "step_free", "walking_distance_km", "transfers",
    "line_count", "lines_count", "route_count", "lines", "transit_lines",
    "ambience", "suitability", "suitability_score", "source",
)
_PAYLOAD_FACT_KEYS = (
    "candidate", "candidate_name", "area", "ambience", "suitability", "suitability_score",
    "average_cost", "cost", "price_cny", "estimated_cost", "total_low", "total_high",
)
_COST_KEYS = (
    "average_cost", "cost", "price_cny", "estimated_cost", "total_low", "total_high",
)
_PREFERRED_COMPARISON_METRICS = {
    "accessibility": ("duration_min", "distance_km", "transit_node_count"),
    "accessibility_needs": ("walking_distance_km", "transfers", "duration_min"),
    "cost": _COST_KEYS,
    "ambience": ("rating", "popularity", "suitability_score"),
    "area_suitability": ("rating", "popularity", "suitability_score"),
    "overall_fit": ("rating", "popularity", "suitability_score"),
}
