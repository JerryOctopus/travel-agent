from __future__ import annotations

import pytest

from travel_agent.hybrid_planning.duration_estimator import (
    CATEGORY_BASELINES,
    DeterministicDurationEstimator,
)
from travel_agent.schemas import POI, RouteInfo, ScoredPOI, TravelProfile


def poi(category: str = "museum", minutes: int = 0) -> POI:
    return POI(
        poi_id="p1",
        name="Generic Venue",
        city="Test City",
        category=category,
        lat=0,
        lng=0,
        rating=4.5,
        popularity=0.8,
        tags=[],
        estimated_duration_min=minutes,
        price_level="mid",
    )


def test_provider_route_time_has_priority_and_preserves_evidence() -> None:
    route = RouteInfo(
        "a", "b", 8.2, 27, "public_transport",
        source="amap", evidence_status="provider_verified",
    )
    estimate = DeterministicDurationEstimator().estimate_transport(
        route=route, fallback_distance_km=999
    )

    assert estimate.estimated_minutes == 27
    assert estimate.source == "amap"
    assert estimate.confidence == "high"
    assert estimate.is_estimate is False
    assert estimate.route_evidence["evidence_status"] == "provider_verified"


def test_transport_fallback_is_explicitly_marked_as_estimate() -> None:
    estimate = DeterministicDurationEstimator().estimate_transport(
        fallback_distance_km=9, mode="public_transport"
    )

    assert estimate.source == "fallback_estimate"
    assert estimate.is_estimate is True
    assert estimate.confidence == "low"
    assert estimate.route_evidence["source"] == "deterministic_speed_fallback"


@pytest.mark.parametrize(
    ("category", "expected"),
    [("museum", 150), ("park", 120), ("historic_district", 150)],
)
def test_category_baselines_are_explainable_ranges(category, expected) -> None:
    estimate = DeterministicDurationEstimator().estimate_activity(category=category)

    assert estimate.estimated_minutes == expected
    assert estimate.range_minutes == (
        CATEGORY_BASELINES[category].minimum,
        CATEGORY_BASELINES[category].maximum,
    )
    assert estimate.source == "category_baseline"
    assert estimate.confidence == "medium"


def test_relaxed_and_compact_pace_adjust_deterministically() -> None:
    estimator = DeterministicDurationEstimator()
    normal = estimator.estimate_activity(category="museum")
    relaxed = estimator.estimate_activity(category="museum", pace="relaxed")
    compact = estimator.estimate_activity(category="museum", pace="compact")

    assert compact.estimated_minutes < normal.estimated_minutes < relaxed.estimated_minutes
    assert relaxed.adjustments == ["relaxed_pace"]
    assert compact.adjustments == ["compact_pace"]


def test_elderly_child_and_accessibility_add_bounded_buffer() -> None:
    estimate = DeterministicDurationEstimator().estimate_activity(
        category="park", elderly=True, child=True, accessibility=True
    )

    assert estimate.estimated_minutes > CATEGORY_BASELINES["park"].typical
    assert estimate.adjustments == ["elderly", "child", "accessibility"]
    assert estimate.range_minutes[0] <= estimate.estimated_minutes <= estimate.range_minutes[1]


def test_adjustment_factor_and_fixed_buffer_are_clamped() -> None:
    estimate = DeterministicDurationEstimator().estimate_activity(
        category="nature",
        pace="relaxed",
        elderly=True,
        child=True,
        accessibility=True,
        photography=True,
        queue_risk=True,
        fixed_reservation_buffer_min=999,
    )

    assert estimate.estimated_minutes <= 480
    assert estimate.range_minutes[1] <= 480
    assert "fixed_reservation_buffer" in estimate.adjustments


def test_poi_metadata_precedes_category_baseline() -> None:
    estimate = DeterministicDurationEstimator().estimate_activity(
        category="museum", metadata_minutes=80, metadata_source="official"
    )

    assert estimate.estimated_minutes == 80
    assert estimate.source == "official"
    assert estimate.confidence == "high"


def test_llm_hint_can_only_classify_and_cannot_supply_minutes() -> None:
    estimate = DeterministicDurationEstimator().estimate_activity(
        category="unknown",
        llm_hint={
            "category": "museum",
            "adjustment_factors": ["photography", "invented"],
            "estimated_minutes": 999,
            "range_minutes": [900, 1100],
        },
    )

    assert estimate.category == "museum"
    assert estimate.estimated_minutes == 180
    assert estimate.estimated_minutes != 999
    assert estimate.adjustments == ["photography"]


def test_profile_adjustments_and_audit_are_applied_to_copies() -> None:
    original = poi("museum", 100)
    ranked = [ScoredPOI(original, 0.9, ["test"])]
    profile = TravelProfile(
        pace="relaxed",
        companions="elderly",
        constraint_state={"wheelchair_user": True},
    )
    updated, audit = DeterministicDurationEstimator().apply_to_ranked(ranked, profile)

    assert original.estimated_duration_min == 100
    assert updated[0].poi.estimated_duration_min > 100
    assert audit["p1"]["source"] == "poi_metadata"
    assert audit["p1"]["range_minutes"][0] <= audit["p1"]["estimated_minutes"]


def test_output_shape_contains_source_range_confidence_and_version() -> None:
    payload = DeterministicDurationEstimator().estimate_activity(category="park").to_dict()

    assert payload["range_minutes"] == [60, 180]
    assert payload["source"] == "category_baseline"
    assert payload["confidence"] == "medium"
    assert payload["baseline_version"] == "activity-duration-v1"
