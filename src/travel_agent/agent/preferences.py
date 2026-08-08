"""出行偏好引导：必要信息齐全后，引导用户补充兴趣、节奏等（可跳过）。"""

from __future__ import annotations

import re

from travel_agent.schemas import TravelProfile
from travel_agent.workflow_rules import (
    extract_profile_rule_based,
    format_budget_display,
    format_interests_display,
    format_pace_display,
)

_SKIP_RE = re.compile(
    r"(随便|都可以|都行|你推荐|你定|你安排|无所谓|直接规划|默认|看着办|帮我选)",
)

_CONFIRM_L3_RE = re.compile(r"(沿用|跟上次|和上次|一样|照旧|老样子)")

_PREFERENCE_KEYWORDS = (
    "美食",
    "自然",
    "风景",
    "历史",
    "文化",
    "博物馆",
    "购物",
    "夜生活",
    "轻松",
    "慢游",
    "不要太累",
    "紧凑",
    "特种兵",
    "多打卡",
    "预算",
    "省钱",
    "奢华",
    "情侣",
    "亲子",
    "独自",
    "必去",
    "避开",
    "自驾",
    "步行",
)


def user_skips_preference_prompt(user_message: str) -> bool:
    return bool(_SKIP_RE.search(user_message.strip()))


def user_confirms_l3_preferences(user_message: str) -> bool:
    return bool(_CONFIRM_L3_RE.search(user_message.strip()))


def turn_expresses_preferences(user_message: str) -> bool:
    """本轮输入是否已包含可规划的偏好信号。"""
    extracted = extract_profile_rule_based(user_message)
    if extracted.interests or extracted.budget_level or extracted.companions:
        return True
    if extracted.pace != "standard" or extracted.must_visit or extracted.avoid:
        return True
    text = user_message.strip()
    return any(keyword in text for keyword in _PREFERENCE_KEYWORDS)


def needs_preference_guidance(profile: TravelProfile, user_message: str) -> bool:
    """偏好缺失不是阻断条件；保留接口供 UI 做非阻断提示。"""
    del profile, user_message
    return False


def build_preference_guide_text(profile: TravelProfile, l3: TravelProfile | None = None) -> str:
    dest = profile.destination or "目的地"
    days = profile.days or "?"
    lines = [f"好的，{dest} {days} 天行程。为了排得更合你心意，可以告诉我："]

    lines.append("· 想玩什么：自然风景、美食探店、历史文化、博物馆、购物、夜生活……")
    lines.append("· 节奏怎样：轻松慢游、适中去几个点、还是特种兵多打卡")
    lines.append("· 其他：预算档次、同行人（独自/情侣/亲子/带老人）、必去或想避开的地方")

    if l3 and (l3.interests or l3.pace != "standard" or l3.budget_level):
        hint_parts: list[str] = []
        if l3.interests:
            hint_parts.append(f"偏好 {format_interests_display(l3.interests[:4])}")
        if l3.pace != "standard":
            hint_parts.append(f"节奏 {format_pace_display(l3.pace)}")
        if l3.budget_level:
            hint_parts.append(f"预算 {format_budget_display(l3.budget_level)}")
        lines.append(f"\n你以往常选：{'；'.join(hint_parts)}。这次要沿用吗？")
    lines.append("\n如果不挑食，回复「随便」或「你推荐」，我就按大众偏好来排。")
    return "\n".join(lines)
