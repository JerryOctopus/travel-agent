"""规划完成后的用户可见摘要：短、结构化，详情交给右侧卡片。"""

from __future__ import annotations

import re

from travel_agent.agent.session import SessionContext
from travel_agent.workflow_rules import format_interests_display, format_pace_display

_HALLUCINATION_SIGNALS = (
    "人工介入",
    "人工修正",
    "人工校准",
    "本地知识库",
    "权威信源",
    "Critic 检测到关键问题",
    "修正后 ·",
    "### ✅",
    "Day 1",
    "Day 2",
    "第 1 天",
    "第1天",
)


def looks_like_hallucinated_itinerary(text: str) -> bool:
    """检测模型是否在手写逐日行程或编造补救话术。"""
    if not text:
        return False
    hits = sum(1 for signal in _HALLUCINATION_SIGNALS if signal in text)
    day_headers = len(re.findall(r"(Day\s*\d|第\s*\d+\s*天)", text, re.IGNORECASE))
    if hits >= 2 or day_headers >= 2:
        return True
    if len(text) > 400 and hits >= 1:
        return True
    return False


def build_plan_reply_text(ctx: SessionContext, plan_artifact_id: str | None = None) -> str:
    """从 artifact 生成简短摘要；逐日时刻表由右侧卡片展示。"""
    from travel_agent.delivery_contract import (
        DELIVERABLE_CURRENT,
        PARTIAL_CURRENT_WITH_LIMITATIONS,
        resolve_delivery_snapshot,
    )

    delivery = resolve_delivery_snapshot(ctx, attempted_artifact_id=plan_artifact_id)
    payload = ctx.store.get(delivery.artifact_id) if delivery.artifact_id else None
    if not payload:
        return "当前没有通过最终校验并处于 current 状态的可交付行程。"

    itinerary = payload.get("itinerary", {})
    critic = payload.get("critic", {})
    profile = ctx.profile

    record = ctx.store.get_record(delivery.artifact_id) if delivery.artifact_id else None
    artifact_status = str(
        (record or {}).get("artifact_status") or payload.get("artifact_status") or ""
    )
    from travel_agent.plan_invariants import validate_plan_artifact

    current_validation = validate_plan_artifact(payload, ctx.profile)
    validation_passed = artifact_status == "current" and current_validation.get("passed") is True
    critic_passed = (payload.get("critic") or {}).get("passed") is True
    deliverable = validation_passed and critic_passed and not payload.get("unresolved_changes")
    lines: list[str] = []
    title = itinerary.get("summary") or f"{profile.destination or ''}{profile.days or ''}天行程"
    if delivery.status == DELIVERABLE_CURRENT and deliverable:
        route_suffix = (
            "每日安排和已核验地图路线请查看右侧面板。"
            if delivery.route_evidence_status == "provider_verified"
            else "每日安排请查看右侧面板；路线为估算结果，未声明地图已核验。"
            if delivery.route_evidence_status == "estimated"
            else "每日安排请查看右侧面板；当前没有可用路线证据，不提供地图路线声明。"
        )
        lines.append(f"{title} 已排好。{route_suffix}")
    elif delivery.status == PARTIAL_CURRENT_WITH_LIMITATIONS:
        lines.append(
            f"{title} 是当前可查看的部分草案，仍有明确限制，尚不可交付为完整计划；"
            "草案可在右侧面板查看。"
        )
    else:
        lines.append("当前没有可交付行程。")

    meta: list[str] = []
    if profile.destination:
        meta.append(profile.destination)
    if profile.days:
        meta.append(f"{profile.days}天")
    if profile.interests:
        meta.append("偏好 " + format_interests_display(profile.interests[:5]))
    if profile.pace != "standard":
        meta.append(f"节奏 {format_pace_display(profile.pace)}")
    if meta:
        lines.append("")
        lines.append(" · ".join(meta))

    day_lines: list[str] = []
    for day in itinerary.get("days", []):
        stops = day.get("stops") or []
        names = "、".join(s.get("name", "") for s in stops[:3] if s.get("name"))
        if not names:
            continue
        theme = day.get("theme") or ""
        label = f"第{day.get('day_index')}天"
        if theme:
            label += f"（{theme}）"
        day_lines.append(f"- {label}：{names}")

    if day_lines:
        lines.append("")
        lines.append("行程概览：")
        lines.extend(day_lines)

    state = profile.constraint_state or {}
    explicit_meals = bool(profile.food_preference or "food" in profile.interests or state.get("dietary"))
    has_meal_stop = any(
        str(stop.get("category") or "") == "food"
        for day in itinerary.get("days", [])
        for stop in (day.get("stops") or [])
    )
    scheduled_meals = [
        item
        for item in ((payload.get("meal_strategy") or {}).get("scheduled_meals") or [])
        if isinstance(item, dict) and item.get("start_time")
    ]
    if scheduled_meals:
        meal_windows = []
        for item in scheduled_meals:
            day_index = item.get("day_index")
            start = str(item.get("start_time") or "")
            end = str(item.get("end_time") or "")
            label = str(item.get("name") or "用餐时段").split("（", 1)[0]
            window = f"{start}–{end}" if end else start
            prefix = f"第{day_index}天 " if day_index else ""
            meal_windows.append(f"{prefix}{window} {label}")
        lines.append("用餐：" + "；".join(meal_windows) + "。")
    elif not explicit_meals and not has_meal_stop:
        lines.append("用餐：每天约 12:00–13:00 预留午餐时间，可在当日活动区域就近自行安排。")

    orig = payload.get("original_issue_count", 0)
    final = payload.get("final_issue_count", 0)
    passed = critic.get("passed", True)
    budget_exceeded = bool(
        isinstance(payload.get("budget_plan"), dict)
        and payload["budget_plan"].get("user_limit_cny") is not None
        and payload["budget_plan"].get("within_user_limit") is False
    )
    notes = [
        str(note)
        for note in (payload.get("applied_changes") or payload.get("revision_notes") or [])
        if not str(note).startswith("已携带 Reviewer")
    ]
    lines.append("")
    if passed and validation_passed and final == 0 and not notes and not budget_exceeded:
        lines.append("约束检查已通过。")
    else:
        status = "预算超限，尚不可交付" if budget_exceeded else ("已通过" if passed and validation_passed else "尚不可交付")
        lines.append(f"约束检查：{orig} → {final} 项（{status}）。")
        if notes:
            lines.append("自动修正：" + "；".join(notes[:3]))
    warnings = [
        str(issue.get("message") or "").strip()
        for issue in (critic.get("issues") or [])
        if issue.get("severity") == "warning" and str(issue.get("message") or "").strip()
    ]
    if warnings:
        lines.append("需复核：" + "；".join(warnings[:3]))

    return_plan = payload.get("return_plan")
    if isinstance(return_plan, dict) and return_plan.get("required"):
        lines.append(
            "返程计划：最晚 "
            f"{return_plan.get('activity_cutoff')} 结束杭州活动，"
            f"选择 {return_plan.get('arrival_deadline')} 前抵达"
            f"{return_plan.get('to_location')}的公共交通班次；"
            "实时车次、余票与检票时间需在购票平台复核。"
        )

    lodging_plan = payload.get("lodging_plan")
    if isinstance(lodging_plan, dict) and lodging_plan.get("status") == "recommended_not_booked":
        hotel = lodging_plan.get("hotel") or {}
        lines.append(
            f"住宿推荐（未预订）：{hotel.get('name')}，"
            f"{lodging_plan.get('rooms')}间 × {lodging_plan.get('nights')}晚，"
            f"估算小计 ¥{lodging_plan.get('lodging_subtotal_cny')}。"
        )

    budget_plan = payload.get("budget_plan")
    if isinstance(budget_plan, dict):
        budget_line = (
            f"预算估算：明细合计 ¥{budget_plan.get('total_expected_cny')}，"
            f"风险区间 ¥{budget_plan.get('total_low_cny')}–¥{budget_plan.get('total_high_cny')}"
        )
        if budget_plan.get("user_limit_cny") is not None:
            verdict = "不超过" if budget_plan.get("within_user_limit") else "可能超过"
            budget_line += f"，{verdict}上限 ¥{budget_plan.get('user_limit_cny')}"
        lines.append(budget_line + "；不含项见预算卡说明。")

    fixed_event_plan = payload.get("fixed_event_plan")
    if isinstance(fixed_event_plan, dict):
        for event in (fixed_event_plan.get("events") or [])[:2]:
            if not isinstance(event, dict):
                continue
            text = (
                f"固定预约：{event.get('date') or '指定日期'} "
                f"{event.get('start')}–{event.get('end')} {event.get('location')}。"
            )
            if event.get("recommended_departure") and event.get("recommended_return_arrival"):
                if event.get("route_evidence_status") == "provider_verified":
                    text += (
                        f"按地图工具核验路线单程约 {event.get('transfer_duration_min')} 分钟，"
                        f"建议约 {event.get('recommended_departure')} 出发，"
                        f"返程约 {event.get('recommended_return_arrival')} 抵达城区。"
                    )
                else:
                    text += "当前路线未获 provider 核验，只能作为估算，不能据此保证准时到达。"
            lines.append(text)

    mobility_plan = payload.get("mobility_plan")
    if isinstance(mobility_plan, dict):
        lines.append(
            f"步行约束：每日不超过 {mobility_plan.get('max_walking_km_per_day')} 公里；"
            "公交接驳步行未知或可能超限的区段改用点到点出租车，并在出发前复核。"
        )

    candidate_verification = payload.get("candidate_verification")
    if isinstance(candidate_verification, dict):
        results = candidate_verification.get("results") or []
        suitable = [item.get("requested_name") for item in results if item.get("status") == "suitable"]
        excluded = [item for item in results if item.get("status") != "suitable"]
        if suitable:
            lines.append("候选核验可安排：" + "、".join(str(item) for item in suitable) + "。")
        if excluded:
            lines.append(
                "候选核验未排入：" + "；".join(
                    f"{item.get('requested_name')}（{item.get('reason') or '未核实'}）"
                    for item in excluded
                ) + "。"
            )

    return "\n".join(lines)


def fallback_no_artifact_message() -> str:
    return (
        "抱歉，刚才的回复格式不合适（不应手写逐日行程）。"
        "请再说一次需求，或回复「重新规划」，我会通过工具重新生成。"
    )
