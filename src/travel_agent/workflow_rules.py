from __future__ import annotations

import re

from travel_agent.schemas import Pace, TravelProfile


CITY_ALIASES = {
    "杭州": "杭州",
    "Hangzhou": "杭州",
    "hangzhou": "杭州",
    "北京": "北京",
    "Beijing": "北京",
    "beijing": "北京",
    "上海": "上海",
    "Shanghai": "上海",
    "shanghai": "上海",
    "成都": "成都",
    "Chengdu": "成都",
    "chengdu": "成都",
    "西安": "西安",
    "Xi'an": "西安",
    "xian": "西安",
    "XiAn": "西安",
}

INTEREST_KEYWORDS = {
    "自然": "nature",
    "风景": "nature",
    "风光": "nature",
    "美食": "food",
    "吃": "food",
    "文化": "culture",
    "历史": "history",
    "博物馆": "museum",
    "情侣": "couple",
    "城市漫步": "citywalk",
    "citywalk": "citywalk",
    "亲子": "family",
    "夜生活": "nightlife",
    "购物": "shopping",
}


def extract_profile_rule_based(user_message: str) -> TravelProfile:
    return TravelProfile(
        destination=_extract_destination(user_message),
        days=_extract_days(user_message),
        budget_level=_extract_budget_level(user_message),
        interests=_extract_interests(user_message),
        companions=_extract_companions(user_message),
        pace=_extract_pace(user_message),
    )


def _extract_destination(text: str) -> str | None:
    for alias, city in CITY_ALIASES.items():
        if alias in text:
            return city
    return None


def _extract_days(text: str) -> int | None:
    match = re.search(r"(\d+)\s*天", text)
    if match:
        return int(match.group(1))
    chinese_digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
    }
    for char, value in chinese_digits.items():
        if f"{char}天" in text:
            return value
    return None


def _extract_budget_level(text: str):
    if "低预算" in text or "省钱" in text or "便宜" in text:
        return "low"
    if "高预算" in text or "奢华" in text or "豪华" in text:
        return "high"
    if "预算" in text or "中等" in text:
        return "mid"
    return None


def _extract_interests(text: str) -> list[str]:
    interests = []
    for keyword, tag in INTEREST_KEYWORDS.items():
        if keyword in text and tag not in interests:
            interests.append(tag)
    return interests


def _extract_companions(text: str) -> str | None:
    if "情侣" in text:
        return "couple"
    if "亲子" in text or "小孩" in text or "孩子" in text:
        return "family"
    if "老人" in text or "父母" in text:
        return "elderly"
    if "一个人" in text or "独自" in text:
        return "solo"
    return None


def _extract_pace(text: str) -> Pace:
    if "轻松" in text or "慢" in text or "不要太累" in text:
        return "relaxed"
    if "紧凑" in text or "多玩" in text or "尽量多" in text:
        return "intensive"
    return "standard"
