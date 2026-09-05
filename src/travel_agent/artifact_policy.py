"""Constraint-aware artifact reuse and deterministic invalidation."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping


MATERIAL_PROFILE_FIELDS = frozenset({
    "destination", "days", "start_date", "budget_limit", "hotel_area",
    "budget_level", "must_visit", "avoid", "transport_mode", "food_preference",
    "interests", "pace", "companions", "party_size",
})

# These fields are deliberately semantic rather than phrase-based.  New POIs,
# cities and wording variants therefore follow the same invalidation policy.
MATERIAL_CONSTRAINT_FIELDS = frozenset({
    "date_start", "date_end", "duration_days", "destination", "destinations",
    "destination_city",
    "must_visit", "removed", "fixed_events",
    "budget_max_cny", "budget_total_cny", "budget_remaining_cny",
    "budget_per_person_cny", "prepaid_cost", "prepaid_lodging_cny",
    "hotel_budget_per_night_cny", "lodging_area", "return_deadline",
    "return_deadline_local_time", "return_deadline_day",
    "return_location", "return_day_index", "activity_end_deadline",
    "activity_end_target",
    "transport_mode", "transport_modes", "self_driving_allowed",
    "public_transport_required", "taxi_backup", "fallback_transport",
    "dietary", "food_preference", "mobility", "wheelchair_user",
    "accessibility_priority", "walking_time_max_min", "max_single_walk_min",
    "max_walking_km_per_day", "max_transfers_per_day", "avoid", "exclude",
    "night_activity_allowed", "pace", "traveler_count",
})

VERSION_META_KEYS = frozenset({
    "_constraint_events", "_constraint_revision", "_constraint_hash",
    "_constraint_snapshot", "_plan_status",
})
NON_MATERIAL_CONSTRAINT_FIELDS = frozenset({"referenced_day_index"})


KIND_DEPENDENCIES: dict[str, frozenset[str]] = {
    "candidates": frozenset({"destination", "destinations", "destination_city", "location_anchor", "comparison_candidates", "compare_lodging_areas", "must_visit", "candidate_attractions", "removed", "interests", "accessibility_priority"}),
    "pois": frozenset({"destination", "destinations", "must_visit", "candidate_attractions", "removed", "interests"}),
    "ranked": frozenset({"must_visit", "removed", "interests", "avoid", "mobility", "accessibility_priority"}),
    "routes": frozenset({"destination", "destination_city", "origin", "location_anchor", "target_anchor", "comparison_candidates", "lodging_area", "transport_mode", "transport_modes", "self_driving_allowed", "public_transport_required", "mobility", "removed"}),
    "hotels": frozenset({"destination", "lodging_area", "hotel_budget_per_night_cny", "accessibility_priority", "date_start", "date_end", "traveler_count"}),
    "restaurants": frozenset({"destination", "lodging_area", "dietary", "budget_per_person_cny"}),
    "budget": frozenset({"duration_days", "traveler_count", "budget_max_cny", "budget_per_person_cny", "prepaid_cost", "prepaid_lodging_cny", "lodging_area"}),
    "weather": frozenset({"destination", "date_start", "date_end"}),
}


def constraint_basis(
    kind: str,
    state: dict[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dependencies = set(KIND_DEPENDENCIES.get(kind, ()))
    if _payload_has_temporal_evidence(kind, payload):
        dependencies.update({"date_start", "date_end", "weekday"})
    return {
        key: state.get(key)
        for key in sorted(dependencies)
        if state.get(key) not in (None, "", [], {})
    }


def constraint_fingerprint(
    kind: str,
    state: dict[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> str:
    encoded = json.dumps(
        constraint_basis(kind, state, payload),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def active_constraint_snapshot(profile: Any) -> dict[str, Any]:
    """Return the canonical plan-affecting state, independent of provenance metadata."""
    state = getattr(profile, "constraint_state", {}) or {}
    snapshot = {
        key: value
        for key, value in state.items()
        if key not in VERSION_META_KEYS and not key.startswith("_")
        and value not in (None, "", [], {})
    }
    for key in MATERIAL_PROFILE_FIELDS:
        value = getattr(profile, key, None)
        if value not in (None, "", [], {}):
            snapshot[f"profile.{key}"] = value
    return json.loads(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str))


def active_constraint_hash(profile: Any) -> str:
    encoded = json.dumps(
        active_constraint_snapshot(profile), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def material_changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    return sorted(
        key for key in changed
        if (
            key.startswith("profile.") and key[8:] in MATERIAL_PROFILE_FIELDS
        ) or (
            not key.startswith("profile.")
            and not key.startswith("_")
            and key not in NON_MATERIAL_CONSTRAINT_FIELDS
            and key in MATERIAL_CONSTRAINT_FIELDS
        )
    )


def update_constraint_version(profile: Any, before: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist a monotonic revision and return its material-change audit."""
    state = getattr(profile, "constraint_state", None)
    if not isinstance(state, dict):
        state = {}
        profile.constraint_state = state
    previous = before if before is not None else dict(state.get("_constraint_snapshot") or {})
    current = active_constraint_snapshot(profile)
    changed_fields = material_changed_fields(previous, current)
    revision = int(state.get("_constraint_revision") or 0)
    if revision <= 0:
        revision = 1
    elif changed_fields:
        revision += 1
    digest = active_constraint_hash(profile)
    state["_constraint_revision"] = revision
    state["_constraint_hash"] = digest
    state["_constraint_snapshot"] = current
    return {
        "revision": revision,
        "constraint_hash": digest,
        "constraint_snapshot": current,
        "material_changed": bool(changed_fields),
        "changed_fields": changed_fields,
    }


def constraint_version(profile: Any) -> dict[str, Any]:
    state = getattr(profile, "constraint_state", {}) or {}
    current_hash = active_constraint_hash(profile)
    if (
        not state.get("_constraint_revision")
        or not state.get("_constraint_hash")
        or str(state.get("_constraint_hash")) != current_hash
    ):
        return update_constraint_version(
            profile, dict(state.get("_constraint_snapshot") or {})
        )
    return {
        "revision": int(state["_constraint_revision"]),
        "constraint_hash": current_hash,
        "constraint_snapshot": active_constraint_snapshot(profile),
    }


def reusable_artifact_ids(store: Any, state: dict[str, Any], *, ttl_seconds: float = 86400.0) -> tuple[list[str], list[dict[str, Any]]]:
    """Return latest valid artifact per kind plus auditable skip reasons."""
    now = time.time()
    selected: dict[str, tuple[float, str]] = {}
    audit: list[dict[str, Any]] = []
    for artifact_id, record in store.snapshot_records().items():
        kind = str(record.get("kind") or "")
        if kind not in KIND_DEPENDENCIES:
            continue
        if now - float(record.get("created_at") or 0) > ttl_seconds:
            audit.append({"artifact_id": artifact_id, "reason": "expired"})
            continue
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        recorded = record.get("constraint_fingerprint")
        expected = constraint_fingerprint(kind, state, payload)
        if not recorded:
            audit.append({"artifact_id": artifact_id, "reason": "missing_constraint_fingerprint"})
            continue
        if recorded != expected:
            audit.append({"artifact_id": artifact_id, "reason": "constraint_changed"})
            continue
        status = str(record.get("artifact_status") or "")
        if status in {"stale", "historical"} and not _verified_identity_payload(kind, payload):
            audit.append({"artifact_id": artifact_id, "reason": "stale_non_identity_artifact"})
            continue
        current = selected.get(kind)
        stamp = float(record.get("created_at") or 0)
        if current is None or stamp > current[0]:
            if current is not None:
                audit.append({"artifact_id": current[1], "reason": "superseded_cache_entry"})
            selected[kind] = (stamp, artifact_id)
    ids = [item[1] for _, item in sorted(selected.items())]
    for artifact_id in ids:
        record = store.get_record(artifact_id) or {}
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        status = str(record.get("artifact_status") or "")
        lineage = [artifact_id]
        for value in (
            record.get("parent_artifact_id"),
            *(record.get("revision_lineage") or []),
            *(payload.get("source_artifact_ids") or []),
        ):
            text = str(value or "").strip()
            if text and text not in lineage:
                lineage.append(text)
        audit.append({
            "artifact_id": artifact_id,
            "reason": (
                "stale_artifact_reuse"
                if status in {"stale", "historical"}
                else "artifact_reuse"
            ),
            "reuse_reason": "kind_specific_fingerprint_match",
            "source_kind": record.get("kind"),
            "source_lineage": lineage,
            "constraint_basis": record.get("constraint_basis")
            or constraint_basis(str(record.get("kind") or ""), state, payload),
        })
    return ids, audit


def _payload_has_temporal_evidence(
    kind: str,
    payload: Mapping[str, Any] | None,
) -> bool:
    if kind in {"weather", "hotels"}:
        return True
    if kind not in {"candidates", "pois", "restaurants"} or not payload:
        return False

    def has_temporal(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                key in {"opening_hours", "availability", "available", "inventory_status", "reservation_status"}
                and item not in (None, "", [], {})
                or has_temporal(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(has_temporal(item) for item in value)
        return False

    return has_temporal(payload)


def _verified_identity_payload(kind: str, payload: Mapping[str, Any]) -> bool:
    if kind not in {"candidates", "pois"}:
        return False
    for key in ("pois", "items", "endpoint_pois"):
        for item in payload.get(key) or []:
            if (
                isinstance(item, Mapping)
                and item.get("verification_status") == "verified"
                and item.get("source")
                and (item.get("source_poi_id") or item.get("poi_id"))
            ):
                return True
    return False
