"""Single-source delivery semantics derived from final ArtifactStore state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from travel_agent.route_evidence import canonical_route_evidence_status


DELIVERABLE_CURRENT = "deliverable_current"
PARTIAL_CURRENT_WITH_LIMITATIONS = "partial_current_with_limitations"
CANDIDATE_REJECTED = "candidate_rejected"
REBUILD_PENDING = "rebuild_pending"
CLARIFICATION = "clarification"
NO_DELIVERABLE = "no_deliverable"
DELIVERABLE_SPECIALIZED = "deliverable_specialized"
PARTIAL_SPECIALIZED_WITH_LIMITATIONS = "partial_specialized_with_limitations"


@dataclass(frozen=True)
class DeliverySnapshot:
    status: str
    artifact_id: str | None
    route_evidence_status: str
    sidebar_available: bool
    map_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_delivery_snapshot(
    ctx: Any,
    *,
    attempted_artifact_id: str | None = None,
    specialized_artifact_id: str | None = None,
    clarification: bool = False,
) -> DeliverySnapshot:
    """Resolve user-visible delivery only from final store/validation state."""
    if clarification:
        return DeliverySnapshot(CLARIFICATION, None, "unavailable", False, False)

    if specialized_artifact_id:
        record = ctx.store.get_record(specialized_artifact_id) or {}
        payload = record.get("payload") or {}
        kind = str(record.get("kind") or "")
        from travel_agent.evaluation.artifact_contract import artifact_content_valid

        if (
            kind in {
                "route_plan",
                "candidate_comparison",
                "local_adjustment_advice",
                "itinerary_patch",
            }
            and isinstance(payload, dict)
            and payload.get("artifact_type") == kind
            and artifact_content_valid(kind, payload)
        ):
            status = (
                PARTIAL_SPECIALIZED_WITH_LIMITATIONS
                if payload.get("limitations")
                else DELIVERABLE_SPECIALIZED
            )
            return DeliverySnapshot(
                status, specialized_artifact_id, "unavailable", False, False
            )

    current_id = ctx.store.latest_current_id("itinerary")
    current = ctx.store.get(current_id) if current_id else None
    if current_id and isinstance(current, dict):
        from travel_agent.plan_invariants import validate_plan_artifact

        validation = validate_plan_artifact(current, ctx.profile)
        limitations = list(current.get("limitations") or [])
        lodging = current.get("lodging_plan") or {}
        explicit_lodging_unverified = bool(
            isinstance(lodging, dict)
            and lodging.get("explicit_requirement") is True
            and lodging.get("status") == "evidence_unavailable"
        )
        status = (
            DELIVERABLE_CURRENT
            if validation.get("passed") is True
            and (current.get("critic") or {}).get("passed") is True
            and not limitations
            and not current.get("unresolved_changes")
            and not explicit_lodging_unverified
            else PARTIAL_CURRENT_WITH_LIMITATIONS
        )
        route_status = _route_status(current)
        return DeliverySnapshot(
            status,
            current_id,
            route_status,
            True,
            route_status != "unavailable",
        )

    attempted = ctx.store.get_record(attempted_artifact_id) if attempted_artifact_id else None
    attempted_status = str((attempted or {}).get("artifact_status") or "")
    if attempted_status in {
        "review_failed", "rework_failed", "validation_failure", "rejected"
    }:
        status = CANDIDATE_REJECTED
    elif (ctx.profile.constraint_state or {}).get("_plan_status") == "rebuild_pending":
        status = REBUILD_PENDING
    else:
        status = NO_DELIVERABLE
    return DeliverySnapshot(status, None, "unavailable", False, False)


def _route_status(payload: dict[str, Any]) -> str:
    statuses: list[str] = []
    for day in (payload.get("itinerary") or {}).get("days") or []:
        for stop in day.get("stops") or []:
            route = stop.get("route_from_previous")
            if isinstance(route, dict):
                statuses.append(canonical_route_evidence_status(route))
    anchors = payload.get("required_route_anchors") or {}
    for leg in anchors.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        rows = list(leg.get("routes") or [])
        if isinstance(leg.get("route"), dict):
            rows.append(leg["route"])
        statuses.extend(canonical_route_evidence_status(row) for row in rows)
    if "provider_verified" in statuses:
        return "provider_verified"
    if any(status in {"deterministic_estimate", "haversine_estimate"} for status in statuses):
        return "estimated"
    return "unavailable"
