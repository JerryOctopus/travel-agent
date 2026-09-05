"""Deterministic, evidence-aware activity and transport duration estimates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Mapping

from travel_agent.schemas import POI, RouteInfo, ScoredPOI, TravelProfile


DURATION_BASELINE_VERSION = "activity-duration-v1"
DurationConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CategoryBaseline:
    minimum: int
    typical: int
    maximum: int


CATEGORY_BASELINES: dict[str, CategoryBaseline] = {
    "museum": CategoryBaseline(90, 150, 240),
    "park": CategoryBaseline(60, 120, 180),
    "historic_district": CategoryBaseline(90, 150, 240),
    "history": CategoryBaseline(60, 120, 180),
    "culture": CategoryBaseline(60, 120, 180),
    "architecture": CategoryBaseline(45, 90, 150),
    "industrial_heritage": CategoryBaseline(60, 120, 180),
    "nature": CategoryBaseline(90, 180, 300),
    "attraction": CategoryBaseline(60, 120, 180),
    "food": CategoryBaseline(45, 75, 120),
    "cafe": CategoryBaseline(30, 60, 90),
}
DEFAULT_BASELINE = CategoryBaseline(45, 90, 150)
_CATEGORY_ALIASES = {
    "博物馆": "museum",
    "美术馆": "museum",
    "公园": "park",
    "历史街区": "historic_district",
    "古镇": "historic_district",
    "历史": "history",
    "文化": "culture",
    "自然": "nature",
    "景点": "attraction",
    "餐饮": "food",
    "餐厅": "food",
}
_ALLOWED_LLM_ADJUSTMENTS = frozenset(
    {"elderly", "child", "accessibility", "photography", "queue_risk"}
)
_PACE_FACTORS = {"relaxed": 1.2, "normal": 1.0, "standard": 1.0, "compact": 0.8, "intensive": 0.8}
_ADJUSTMENT_FACTORS = {
    "elderly": 1.15,
    "child": 1.10,
    "accessibility": 1.20,
    "photography": 1.20,
    "queue_risk": 1.15,
}
MIN_ADJUSTMENT_FACTOR = 0.65
MAX_ADJUSTMENT_FACTOR = 1.75
MIN_ACTIVITY_MINUTES = 20
MAX_ACTIVITY_MINUTES = 480


@dataclass(frozen=True)
class ActivityDurationEstimate:
    estimated_minutes: int
    range_minutes: tuple[int, int]
    source: str
    adjustments: list[str]
    confidence: DurationConfidence
    category: str
    baseline_version: str = DURATION_BASELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["range_minutes"] = list(self.range_minutes)
        return payload


@dataclass(frozen=True)
class TransportDurationEstimate:
    estimated_minutes: int
    range_minutes: tuple[int, int]
    source: str
    confidence: DurationConfidence
    is_estimate: bool
    route_evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["range_minutes"] = list(self.range_minutes)
        return payload


class DeterministicDurationEstimator:
    """Compute final minutes in code; an optional LLM hint may classify only."""

    def estimate_activity(
        self,
        *,
        category: str | None,
        metadata_minutes: int | None = None,
        metadata_source: str | None = None,
        pace: str = "normal",
        elderly: bool = False,
        child: bool = False,
        accessibility: bool = False,
        photography: bool = False,
        queue_risk: bool = False,
        fixed_reservation_buffer_min: int = 0,
        llm_hint: Mapping[str, Any] | None = None,
    ) -> ActivityDurationEstimate:
        normalized_category = _normalize_category(category)
        llm_adjustments: set[str] = set()
        if normalized_category not in CATEGORY_BASELINES and isinstance(llm_hint, Mapping):
            hinted = _normalize_category(str(llm_hint.get("category") or ""))
            if hinted in CATEGORY_BASELINES:
                normalized_category = hinted
            raw_adjustments = llm_hint.get("adjustment_factors") or []
            if isinstance(raw_adjustments, list):
                llm_adjustments = {
                    str(item) for item in raw_adjustments
                    if str(item) in _ALLOWED_LLM_ADJUSTMENTS
                }

        if metadata_minutes is not None and int(metadata_minutes) > 0:
            typical = int(metadata_minutes)
            baseline = CategoryBaseline(
                max(MIN_ACTIVITY_MINUTES, _round_minutes(typical * 0.75)),
                _round_minutes(typical),
                min(MAX_ACTIVITY_MINUTES, _round_minutes(typical * 1.35)),
            )
            source = metadata_source or "poi_metadata"
            confidence: DurationConfidence = "high" if source in {"official", "tool"} else "medium"
        else:
            baseline = CATEGORY_BASELINES.get(normalized_category, DEFAULT_BASELINE)
            source = "category_baseline"
            confidence = "medium" if normalized_category in CATEGORY_BASELINES else "low"

        active: list[str] = []
        pace_key = pace if pace in _PACE_FACTORS else "normal"
        if pace_key != "normal" and pace_key != "standard":
            active.append(f"{pace_key}_pace")
        flags = {
            "elderly": elderly,
            "child": child,
            "accessibility": accessibility,
            "photography": photography,
            "queue_risk": queue_risk,
        }
        for name in _ADJUSTMENT_FACTORS:
            if flags[name] or name in llm_adjustments:
                active.append(name)

        factor = _PACE_FACTORS[pace_key]
        for name in active:
            if name in _ADJUSTMENT_FACTORS:
                factor *= _ADJUSTMENT_FACTORS[name]
        factor = max(MIN_ADJUSTMENT_FACTOR, min(MAX_ADJUSTMENT_FACTOR, factor))
        buffer = max(0, min(120, int(fixed_reservation_buffer_min or 0)))
        if buffer:
            active.append("fixed_reservation_buffer")

        lower = _clamp_minutes(_round_minutes(baseline.minimum * factor) + buffer)
        typical = _clamp_minutes(_round_minutes(baseline.typical * factor) + buffer)
        upper = _clamp_minutes(_round_minutes(baseline.maximum * factor) + buffer)
        lower = min(lower, typical)
        upper = max(upper, typical)
        return ActivityDurationEstimate(
            estimated_minutes=typical,
            range_minutes=(lower, upper),
            source=source,
            adjustments=active,
            confidence=confidence,
            category=normalized_category or "unknown",
        )

    def estimate_transport(
        self,
        *,
        route: RouteInfo | None = None,
        fallback_distance_km: float | None = None,
        mode: str = "public_transport",
    ) -> TransportDurationEstimate:
        if route is not None and route.duration_min > 0:
            provider_verified = route.evidence_status == "provider_verified"
            estimated = int(route.duration_min)
            margin = 0.10 if provider_verified else 0.25
            return TransportDurationEstimate(
                estimated_minutes=estimated,
                range_minutes=(
                    max(1, _round_minutes(estimated * (1 - margin))),
                    max(1, _round_minutes(estimated * (1 + margin))),
                ),
                source=route.source,
                confidence="high" if provider_verified else "medium",
                is_estimate=not provider_verified,
                route_evidence={
                    "origin_poi_id": route.origin_poi_id,
                    "destination_poi_id": route.destination_poi_id,
                    "distance_km": route.distance_km,
                    "mode": route.mode,
                    "source": route.source,
                    "evidence_status": route.evidence_status,
                },
            )
        distance = max(0.0, float(fallback_distance_km or 0.0))
        speed_kmh = {"walk": 4.5, "public_transport": 18.0, "taxi": 24.0, "drive": 28.0}.get(mode, 18.0)
        overhead = {"walk": 0, "public_transport": 12, "taxi": 8, "drive": 8}.get(mode, 12)
        estimated = max(1, int(math.ceil(distance / speed_kmh * 60 + overhead)))
        return TransportDurationEstimate(
            estimated_minutes=estimated,
            range_minutes=(max(1, _round_minutes(estimated * 0.75)), _round_minutes(estimated * 1.4)),
            source="fallback_estimate",
            confidence="low",
            is_estimate=True,
            route_evidence={
                "distance_km": distance,
                "mode": mode,
                "source": "deterministic_speed_fallback",
                "evidence_status": "deterministic_estimate",
            },
        )

    def estimate_poi(self, poi: POI, profile: TravelProfile) -> ActivityDurationEstimate:
        state = profile.constraint_state or {}
        companions = str(profile.companions or "").lower()
        tags = {str(item).lower() for item in poi.tags}
        pace = {"standard": "normal", "intensive": "compact"}.get(profile.pace, profile.pace)
        return self.estimate_activity(
            category=poi.category,
            metadata_minutes=poi.estimated_duration_min if poi.estimated_duration_min > 0 else None,
            metadata_source="poi_metadata" if poi.estimated_duration_min > 0 else None,
            pace=pace,
            elderly=bool(state.get("elderly")) or "elderly" in companions or "老人" in companions,
            child=bool(state.get("child_age") is not None) or "child" in companions or "孩子" in companions,
            accessibility=bool(state.get("wheelchair_user") or state.get("accessibility_priority")),
            photography="photography" in tags,
            queue_risk=bool(state.get("queue_risk")) or "queue_risk" in tags,
            fixed_reservation_buffer_min=0,
        )

    def apply_to_ranked(
        self,
        ranked: list[ScoredPOI],
        profile: TravelProfile,
    ) -> tuple[list[ScoredPOI], dict[str, dict[str, Any]]]:
        """Return copied POIs with deterministic visit minutes and an audit map."""
        updated: list[ScoredPOI] = []
        audit: dict[str, dict[str, Any]] = {}
        for item in ranked:
            estimate = self.estimate_poi(item.poi, profile)
            updated.append(
                replace(item, poi=replace(item.poi, estimated_duration_min=estimate.estimated_minutes))
            )
            audit[item.poi.poi_id] = estimate.to_dict()
        return updated, audit


def _normalize_category(category: str | None) -> str:
    value = str(category or "").strip().lower()
    return _CATEGORY_ALIASES.get(value, value)


def _round_minutes(value: float) -> int:
    return int(round(float(value) / 5.0) * 5)


def _clamp_minutes(value: int) -> int:
    return max(MIN_ACTIVITY_MINUTES, min(MAX_ACTIVITY_MINUTES, int(value)))
