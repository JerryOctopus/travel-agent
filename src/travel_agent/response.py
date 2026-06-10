from __future__ import annotations

from travel_agent.schemas import WorkflowResult


INTEREST_LABELS = {
    "nature": "自然风光",
    "food": "美食",
    "culture": "文化",
    "history": "历史",
    "museum": "博物馆",
    "couple": "情侣友好",
}

PACE_LABELS = {
    "relaxed": "轻松",
    "standard": "标准",
    "intensive": "紧凑",
}


def render_markdown_response(result: WorkflowResult) -> str:
    """把结构化 workflow 结果渲染成中文 Markdown 回复。"""
    if result.clarification_question:
        return result.clarification_question
    if not result.itinerary:
        return "暂时无法生成行程，我还需要更多出行信息。"

    lines = [
        f"# {result.itinerary.summary}",
        "",
        _render_profile_summary(result),
        "",
    ]

    if result.revised and result.revision_notes:
        lines.extend(["## 自动修正", ""])
        lines.extend(f"- {note}" for note in result.revision_notes)
        lines.append("")

    context_lines = _render_context_section(result)
    if context_lines:
        lines.extend(context_lines)
        lines.append("")

    lines.extend(["## 每日行程", ""])
    for day in result.itinerary.days:
        lines.append(f"### 第{day.day_index}天：{day.theme}")
        if not day.stops:
            lines.append("- 暂无合适安排")
            lines.append("")
            continue
        for stop in day.stops:
            route_text = _render_route_prefix(stop.route_from_previous)
            lines.append(
                f"- {stop.start_time}｜{route_text}{stop.poi.name}｜"
                f"建议停留 {stop.duration_min} 分钟｜{stop.note}"
            )
        lines.append("")

    lines.extend(_render_quality_section(result))
    return "\n".join(lines).strip()


def _render_profile_summary(result: WorkflowResult) -> str:
    profile = result.profile
    parts = []
    if profile.destination:
        parts.append(f"目的地：{profile.destination}")
    if profile.days:
        parts.append(f"天数：{profile.days}天")
    parts.append(f"节奏：{PACE_LABELS[profile.pace]}")
    if profile.interests:
        interests = "、".join(INTEREST_LABELS.get(item, item) for item in profile.interests)
        parts.append(f"偏好：{interests}")
    if profile.companions:
        parts.append(f"同行：{profile.companions}")
    return "；".join(parts)


def _render_quality_section(result: WorkflowResult) -> list[str]:
    if not result.critic_result:
        return []
    if result.critic_result.passed:
        return ["## 约束检查", "", "- 当前行程已通过 critic 检查。"]

    lines = ["## 约束检查", ""]
    for issue in result.critic_result.issues:
        lines.append(f"- [{issue.severity}] {issue.message}")
    return lines


def _render_context_section(result: WorkflowResult) -> list[str]:
    lines = []
    if result.weather:
        lines.extend(
            [
                "## 出行上下文",
                "",
                f"- 天气：{_weather_label(result.weather.condition)}，约 {result.weather.temperature_c}°C（{result.weather.source}）",
            ]
        )
    if result.knowledge_chunks:
        if not lines:
            lines.extend(["## 出行上下文", ""])
        best_chunk = result.knowledge_chunks[0]
        lines.append(f"- 攻略依据：{best_chunk.text}")
    return lines


def _render_route_prefix(route) -> str:
    if route is None:
        return ""
    return (
        f"上一站通勤约 {route.duration_min} 分钟 / {route.distance_km} 公里｜"
    )


def _weather_label(condition: str) -> str:
    return {
        "sunny": "晴",
        "cloudy": "多云",
        "rain": "雨",
        "unknown": "未知",
    }.get(condition, condition)
