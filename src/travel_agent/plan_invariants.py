"""Deterministic invariants for a final, renderable TravelPlan artifact."""

from __future__ import annotations

from datetime import date
from typing import Any

from travel_agent.schemas import TravelProfile
from travel_agent.artifact_policy import constraint_version
from travel_agent.route_evidence import (
    canonical_route_evidence_status,
    route_evidence_reason_codes,
)


CRITICAL = "error"


def validate_plan_artifact(payload: dict[str, Any], profile: TravelProfile) -> dict[str, Any]:
    """Validate the assembled artifact, including supplemental plans and lineage."""
    issues: list[dict[str, str]] = []
    itinerary = payload.get("itinerary") or {}
    days = itinerary.get("days") or []
    state = profile.constraint_state or {}
    if state.get("return_deadline"):
        from travel_agent.datetime_semantics import normalize_datetime_value

        deadline_normalization = normalize_datetime_value(
            state.get("return_deadline"),
            context=state,
            prefer_trip_end=True,
        )
        stored_deadline_normalization = (
            (state.get("_datetime_normalizations") or {}).get("return_deadline") or {}
        )
        relative_last_day = bool(
            state.get("return_deadline_day") == "last_day"
            and state.get("return_deadline_local_time")
            and str(state.get("return_deadline"))
            == str(state.get("return_deadline_local_time"))
        )
        if (
            not deadline_normalization.comparable
            and stored_deadline_normalization.get("comparable") is not True
            and not relative_last_day
        ):
            _issue(
                issues,
                "return_deadline_not_comparable",
                "返程截止时间缺少可唯一推导的日期或时区，不能进入可交付 Artifact。",
            )
    if state.get("activity_end_target"):
        from travel_agent.constraint_events import clock_minutes

        try:
            target_minutes = clock_minutes(state.get("activity_end_target"))
            final_stop = (
                (days[-1].get("stops") or [])[-1]
                if days and (days[-1].get("stops") or []) else None
            )
            final_end = (
                clock_minutes(final_stop.get("start_time"))
                + int(final_stop.get("duration_min") or 0)
                if isinstance(final_stop, dict) else None
            )
        except (TypeError, ValueError):
            target_minutes = final_end = None
        if (
            target_minutes is None
            or final_end is None
            or abs(final_end - target_minutes) > 30
        ):
            _issue(
                issues,
                "activity_end_target_missed",
                f"最终活动结束时间未对齐用户明确要求的 {state.get('activity_end_target')}。",
            )
    active_version = constraint_version(profile)
    artifact_version = payload.get("state_version") or {}
    if artifact_version and (
        artifact_version.get("constraint_revision") != active_version["revision"]
        or artifact_version.get("constraint_hash") != active_version["constraint_hash"]
    ):
        _issue(
            issues,
            "artifact_constraint_version_mismatch",
            "计划 Artifact 的约束版本与当前 active constraints 不一致。",
        )

    expected_days = _expected_days(profile)
    indexes = [day.get("day_index") for day in days if isinstance(day, dict)]
    if expected_days is not None and len(days) != expected_days:
        _issue(issues, "trip_day_count_mismatch", f"计划包含{len(days)}天，但日期/时长要求为{expected_days}天。")
    if indexes != list(range(1, len(days) + 1)):
        _issue(issues, "day_index_not_contiguous", "day_index 必须从1开始连续且不重复。")

    for day in days:
        stops = day.get("stops") or []
        for index, stop in enumerate(stops):
            poi = stop.get("poi") or {}
            if not poi.get("poi_id") or not poi.get("source") or poi.get("verification_status") not in (None, "verified"):
                _issue(issues, "poi_evidence_binding_invalid", f"第{day.get('day_index')}天 POI 缺少有效实体证据绑定。")
            route = stop.get("route_from_previous")
            if index > 0 and (
                not isinstance(route, dict)
                or not route.get("origin_poi_id")
                or not route.get("destination_poi_id")
                or route.get("evidence_status") in (None, "unavailable")
            ):
                _issue(issues, "route_evidence_binding_invalid", f"第{day.get('day_index')}天相邻地点缺少有效路线证据绑定。")
            if isinstance(route, dict):
                for code in route_evidence_reason_codes(route):
                    _issue(issues, code, "路线声明为 provider_verified，但缺少匹配、可引用的实际工具证据。")
            if index > 0 and isinstance(route, dict):
                try:
                    from travel_agent.constraint_events import clock_minutes

                    previous = stops[index - 1]
                    previous_end = (
                        clock_minutes(previous.get("start_time"))
                        + int(previous.get("duration_min") or 0)
                    )
                    current_start = clock_minutes(stop.get("start_time"))
                    transfer = int(route.get("duration_min") or 0)
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if previous_end + transfer > current_start:
                        _issue(
                            issues,
                            "insufficient_transfer_time",
                            f"第{day.get('day_index')}天前往 {poi.get('name') or '下一站'} 的预留时间不足。",
                        )
            _validate_stop_opening_hours(
                issues,
                stop=stop,
                profile=profile,
                day_index=int(day.get("day_index") or 0),
            )

    for required in state.get("must_visit") or profile.must_visit or []:
        if not any(
            _stop_covers_requirement(stop, required)
            for day in days if isinstance(day, dict)
            for stop in (day.get("stops") or []) if isinstance(stop, dict)
        ):
            _issue(issues, "must_visit_missing", f"当前计划未包含必去地点 {required}。")
    for removed in state.get("removed") or []:
        if any(
            _stop_covers_requirement(stop, removed)
            for day in days if isinstance(day, dict)
            for stop in (day.get("stops") or []) if isinstance(stop, dict)
        ):
            _issue(issues, "removed_place_present", f"已删除地点 {removed} 仍出现在当前计划中。")

    lodging = payload.get("lodging_plan") or {}
    expected_nights = _expected_nights(profile, expected_days)
    if lodging and expected_nights is not None and lodging.get("nights") != expected_nights:
        _issue(issues, "lodging_nights_mismatch", f"住宿为{lodging.get('nights')}晚，但到达/离开语义要求{expected_nights}晚。")
    if lodging.get("status") == "recommended_not_booked":
        hotel = lodging.get("hotel") or {}
        if not hotel.get("hotel_id") or not hotel.get("source"):
            _issue(issues, "hotel_evidence_binding_invalid", "推荐住宿缺少有效 hotel_id/source 证据绑定。")

    fixed = payload.get("fixed_event_plan") or {}
    fixed_items = fixed.get("events") or []
    source_events = [event for event in (state.get("fixed_events") or []) if isinstance(event, dict)]
    if len(fixed_items) != len(source_events):
        _issue(issues, "fixed_event_artifact_missing", "最终 Artifact 未完整保留全部固定事件。")
    for source in source_events:
        found = next((item for item in fixed_items if _same_event(item, source)), None)
        if found is None:
            _issue(issues, "fixed_event_changed", f"固定事件 {source.get('location') or ''} 的日期或时间被修改/丢失。")
            continue
        if found.get("route_evidence_status") in {"provider_verified", "deterministic_estimate"}:
            if not found.get("recommended_departure") or int(found.get("required_buffer_min") or 0) <= 0:
                _issue(issues, "fixed_event_buffer_missing", f"固定事件 {source.get('location') or ''} 缺少可执行的出发时间和缓冲。")
        elif str(source.get("location") or "").strip() not in {
            "自由活动", "休息", "自由时间"
        }:
            _issue(
                issues,
                "fixed_event_route_missing",
                f"固定事件 {source.get('location') or ''} 缺少可验证的到达路线与缓冲。",
            )

    verification = payload.get("candidate_verification") or {}
    for result in verification.get("results") or []:
        if not isinstance(result, dict) or result.get("status") in {
            "suitable", "verified", "provider_verified"
        }:
            continue
        requested = result.get("requested_name")
        matched = result.get("matched_name")
        if any(
            _stop_covers_requirement(stop, requested)
            or (matched and _stop_covers_requirement(stop, matched))
            for day in days if isinstance(day, dict)
            for stop in (day.get("stops") or []) if isinstance(stop, dict)
        ):
            _issue(
                issues,
                "rejected_candidate_scheduled",
                f"候选 {requested or matched or ''} 的核验状态为 {result.get('status')}，不能进入最终日程。",
            )

    route_anchors = payload.get("required_route_anchors") or {}
    anchor_legs = [
        item for item in (route_anchors.get("legs") or []) if isinstance(item, dict)
    ]
    for leg in anchor_legs:
        declared = str(leg.get("evidence_status") or "unavailable")
        route_rows = (
            list(leg.get("routes") or [])
            if leg.get("kind") == "fixed_event_transfer"
            else [leg.get("route")] if isinstance(leg.get("route"), dict) else []
        )
        canonical_rows = [canonical_route_evidence_status(row) for row in route_rows]
        if declared == "provider_verified" and not route_rows:
            _issue(
                issues,
                "provider_verified_routes_empty",
                "required route anchor 声称 provider_verified，但 routes 为空。",
            )
        if declared == "provider_verified" and "provider_verified" not in canonical_rows:
            _issue(
                issues,
                "provider_verified_route_evidence_missing",
                "required route anchor 的 provider_verified 声明没有匹配的实际路线证据。",
            )
        if declared == "unavailable" and any(status == "provider_verified" for status in canonical_rows):
            _issue(
                issues,
                "route_evidence_status_conflict",
                "required route anchor 同时包含 unavailable 状态和 provider_verified 路线。",
            )

    for event in fixed_items:
        location = str(event.get("location") or "")
        anchor = next((
            leg for leg in anchor_legs
            if leg.get("kind") == "fixed_event_transfer"
            and _entity_matches(leg.get("required_name"), location)
        ), None)
        event_status = str(event.get("route_evidence_status") or "unavailable")
        anchor_status = str((anchor or {}).get("evidence_status") or "unavailable")
        if event_status != anchor_status:
            _issue(
                issues,
                "fixed_event_route_status_mismatch",
                f"固定事件 {location} 在 fixed_event_plan 与 required_route_anchors 中的路线证据状态冲突。",
            )
        if event_status == "provider_verified" and canonical_route_evidence_status(
            event.get("route_evidence_reference")
        ) != "provider_verified":
            _issue(
                issues,
                "fixed_event_provider_evidence_missing",
                f"固定事件 {location} 声称 provider_verified，但没有匹配的可引用路线。",
            )

    trip_origin = str(state.get("origin") or "").strip()
    if trip_origin:
        origin_leg = next(
            (
                item for item in anchor_legs
                if item.get("kind") == "trip_origin_to_first_stop"
            ),
            None,
        )
        origin_route = (origin_leg or {}).get("route")
        origin_status = canonical_route_evidence_status(origin_route)
        if not origin_leg or origin_status == "unavailable":
            _issue(
                issues,
                "trip_origin_route_missing",
                f"出发地 {trip_origin} 到行程首站缺少可执行路线证据。",
            )
        else:
            first_stop_id = str(
                ((days[0].get("stops") or [])[0].get("poi") or {}).get("poi_id")
                if days and (days[0].get("stops") or [])
                else ""
            )
            origin_id = str(origin_leg.get("origin_poi_id") or "")
            destination_id = str(origin_leg.get("destination_poi_id") or "")
            if (
                not first_stop_id
                or destination_id != first_stop_id
                or str(origin_route.get("origin_poi_id") or "") != origin_id
                or str(origin_route.get("destination_poi_id") or "") != destination_id
            ):
                _issue(
                    issues,
                    "trip_origin_route_endpoint_mismatch",
                    "出发路线未绑定到 Planner 当前行程的真实首站。",
                )

    return_location = str(state.get("return_location") or "").strip()
    if return_location and state.get("return_deadline"):
        anchors = route_anchors.get("legs") or []
        leg = route_anchors.get("last_stop_to_return_location") or next(
            (
                item for item in anchors
                if item.get("kind") == "last_stop_to_return_location"
            ),
            None,
        )
        if not leg or leg.get("evidence_status") not in {"provider_verified", "deterministic_estimate"}:
            _issue(issues, "return_route_missing", f"最后一天缺少到返程地点 {return_location} 的可信显式路线。")
        else:
            final_stop_id = str(
                ((days[-1].get("stops") or [])[-1].get("poi") or {}).get("poi_id")
                if days and (days[-1].get("stops") or [])
                else ""
            )
            required_fields = (
                "origin_poi_id", "destination_poi_id", "mode", "duration_min",
                "distance_km", "source", "recommended_latest_departure",
                "required_buffer_min",
            )
            if any(leg.get(field) in (None, "") for field in required_fields):
                _issue(
                    issues,
                    "return_route_artifact_incomplete",
                    "返程路线缺少端点、方式、耗时/距离、来源或最晚出发缓冲。",
                )
            if not final_stop_id or str(leg.get("origin_poi_id") or "") != final_stop_id:
                _issue(
                    issues,
                    "return_route_origin_mismatch",
                    "返程路线起点未绑定 Planner 最终日的真实最后一站。",
                )
            if (
                not _positive_number(leg.get("duration_min"))
                or not _positive_number(leg.get("distance_km"))
                or not _positive_number(leg.get("required_buffer_min"))
            ):
                _issue(
                    issues,
                    "return_route_feasibility_incomplete",
                    "返程路线没有可用于截止时间判断的正数耗时、距离和缓冲。",
                )
        return_plan = payload.get("return_plan") or {}
        terminal_transfer = return_plan.get("terminal_transfer")
        if isinstance(terminal_transfer, dict):
            final_stop_id = str(
                ((days[-1].get("stops") or [])[-1].get("poi") or {}).get("poi_id")
                if days and (days[-1].get("stops") or [])
                else ""
            )
            if (
                not final_stop_id
                or str(terminal_transfer.get("origin_poi_id") or "") != final_stop_id
            ):
                _issue(
                    issues,
                    "return_plan_origin_mismatch",
                    "return_plan 的接驳路线不是从 Planner 最终日的真实最后一站出发。",
                )

    budget_required = _budget_required(profile)
    budget = payload.get("budget_plan")
    if budget_required and not isinstance(budget, dict):
        _issue(issues, "budget_plan_missing", "用户要求预算约束，但最终 Artifact 没有预算计划。")
    elif budget_required and isinstance(budget, dict):
        if (
            budget.get("constraint_revision") != active_version["revision"]
            or budget.get("constraint_hash") != active_version["constraint_hash"]
        ):
            _issue(
                issues,
                "budget_constraint_version_mismatch",
                "预算计划不是按当前约束 revision/hash 生成。",
            )
        expected_people = int(state.get("traveler_count") or profile.party_size or 1)
        if int(budget.get("people") or 0) != expected_people:
            _issue(issues, "budget_people_mismatch", "预算人数与当前出行人数不一致。")
        if expected_days is not None and int(budget.get("days") or 0) != expected_days:
            _issue(issues, "budget_days_mismatch", "预算天数与当前行程天数不一致。")
        unknown = list(budget.get("unknown_items") or [])
        required = {"lodging", "transport", "tickets", "meals", "fixed_event_cost", "contingency"}
        breakdown = budget.get("breakdown_cny") or {}
        absent = sorted(required - set(breakdown) - set(unknown))
        if absent:
            _issue(issues, "budget_components_missing", "预算成本项未覆盖且未标 unknown：" + "、".join(absent))
        if budget.get("expected_total") is None and not unknown:
            _issue(issues, "budget_expected_total_missing", "预算缺少 expected_total。")
        if budget.get("high_total") is None and not unknown:
            _issue(issues, "budget_high_total_missing", "预算缺少 high_total。")

    critic = payload.get("critic") or {}
    for item in critic.get("issues") or []:
        if str(item.get("severity") or "").lower() in {"error", "critical"}:
            _issue(issues, str(item.get("code") or "critic_error"), str(item.get("message") or "确定性 critic 未通过。"))

    unresolved = [str(item) for item in (payload.get("unresolved_changes") or []) if str(item).strip()]
    for item in unresolved:
        _issue(issues, "repair_unresolved", item)
    return {
        "passed": not issues,
        "status": "passed" if not issues else "validation_failure",
        "issues": issues,
        "validated_constraint_revision": active_version["revision"],
        "validated_constraint_hash": active_version["constraint_hash"],
        "checks_run": [
            "date_duration_days", "day_index", "lodging_nights", "fixed_events",
            "schedule_route_feasibility", "must_visit_removed_candidates",
            "trip_origin_route", "return_route",
            "budget_completeness", "evidence_binding", "repair_consistency", "render_gate",
            "datetime_semantics", "activity_end_target",
            "applicable_opening_hours",
        ],
    }


def _validate_stop_opening_hours(
    issues: list[dict[str, str]],
    *,
    stop: dict[str, Any],
    profile: TravelProfile,
    day_index: int,
) -> None:
    """Recheck provider hours after every reviewer revision.

    Scheduling is not the final authority: a reviewer may move a stop after the
    planner checked it.  Only check evidenced, timed POIs here so legacy imports
    without provider hours remain governed by their existing evidence policy.
    """
    poi_data = stop.get("poi") or {}
    if not isinstance(poi_data, dict) or not poi_data.get("opening_hours"):
        return
    if stop.get("start_time") in (None, "") or stop.get("duration_min") in (None, ""):
        return
    try:
        from travel_agent.agent.serde import poi_from_dict
        from travel_agent.constraint_events import clock_minutes
        from travel_agent.planning import (
            _last_admission_for_trip_day,
            _opening_window_for_trip_day,
            poi_open_on_trip_day,
        )

        poi = poi_from_dict(poi_data)
        start = clock_minutes(stop.get("start_time"))
        duration = int(stop.get("duration_min") or 0)
    except (KeyError, TypeError, ValueError):
        return
    if day_index <= 0 or not poi_open_on_trip_day(poi, profile, day_index):
        _issue(
            issues,
            "poi_closed_on_trip_day",
            f"第{day_index}天 {poi.name} 的工具证据显示当天闭馆。",
        )
        return
    window = _opening_window_for_trip_day(poi, profile, day_index)
    if window is not None:
        opens, closes = window
        if start < opens or start + duration > closes:
            _issue(
                issues,
                "outside_applicable_opening_hours",
                f"第{day_index}天 {poi.name} 的安排超出当天适用营业时间。",
            )
    last_admission = _last_admission_for_trip_day(poi, profile, day_index)
    if last_admission is not None and start > last_admission:
        _issue(
            issues,
            "after_last_admission",
            f"第{day_index}天 {poi.name} 的到达时间晚于工具证据中的停止入场时间。",
        )


def _expected_days(profile: TravelProfile) -> int | None:
    state = profile.constraint_state or {}
    start = profile.start_date or state.get("date_start")
    end = state.get("date_end")
    if start and end:
        try:
            return (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days + 1
        except ValueError:
            pass
    value = state.get("duration_days") or profile.days
    try:
        return max(1, int(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _expected_nights(profile: TravelProfile, expected_days: int | None) -> int | None:
    state = profile.constraint_state or {}
    if state.get("lodging_nights") is not None:
        try:
            return max(0, int(state["lodging_nights"]))
        except (TypeError, ValueError):
            return None
    # A day-range trip departs on the last date, hence one fewer overnight.
    return max(0, expected_days - 1) if expected_days is not None else None


def _budget_required(profile: TravelProfile) -> bool:
    state = profile.constraint_state or {}
    return profile.budget_limit is not None or any(
        state.get(key) is not None for key in (
            "budget_max_cny", "budget_total_cny", "budget_remaining_cny",
            "budget_per_person_cny", "prepaid_lodging_cny",
        )
    )


def _positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _same_event(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(str(left.get(key) or "") == str(right.get(key) or "") for key in ("date", "day", "start", "end", "location"))


def _stop_names(stop: dict[str, Any]) -> list[str]:
    poi = stop.get("poi") or {}
    return [
        str(stop.get("name") or ""), str(poi.get("name") or ""),
        str(poi.get("canonical_name") or ""),
        *[str(item) for item in (poi.get("aliases") or [])],
    ]


def _entity_matches(left: Any, right: Any) -> bool:
    a, b = str(left).strip().lower(), str(right).strip().lower()
    return bool(a and b and (a == b or a in b or b in a))


def _stop_covers_requirement(stop: dict[str, Any], required: Any) -> bool:
    poi = stop.get("poi") or {}
    if isinstance(poi, dict) and poi.get("poi_id") and poi.get("name"):
        try:
            from travel_agent.agent.serde import poi_from_dict
            from travel_agent.poi_evidence import (
                normalize_candidate_requirement,
                poi_covers_requirement,
            )

            return poi_covers_requirement(
                poi_from_dict(poi), normalize_candidate_requirement(required)
            )
        except (KeyError, TypeError, ValueError):
            pass
    # Legacy imported plans may not carry the full POI schema.  Keep exact
    # canonical/alias equality, but never use unconstrained short containment.
    required_name = str(required).strip().casefold()
    return bool(
        required_name
        and any(required_name == str(name).strip().casefold() for name in _stop_names(stop))
    )


def _issue(target: list[dict[str, str]], code: str, message: str) -> None:
    if not any(item["code"] == code and item["message"] == message for item in target):
        target.append({"code": code, "message": message, "severity": CRITICAL})
