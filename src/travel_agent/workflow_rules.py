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
    "重庆": "重庆",
    "武汉": "武汉",
    "南京": "南京",
    "苏州": "苏州",
    "广州": "广州",
    "深圳": "深圳",
    "厦门": "厦门",
    "长沙": "长沙",
    "昆明": "昆明",
    "大理": "大理",
    "青岛": "青岛",
    "天津": "天津",
    "洛阳": "洛阳",
    "桂林": "桂林",
    "三亚": "三亚",
}

INTEREST_KEYWORDS = {
    "自然": "nature",
    "风景": "nature",
    "风光": "nature",
    "自然风光": "nature",
    "海边": "nature",
    "园林": "nature",
    "美食": "food",
    "吃": "food",
    "咖啡店": "food",
    "文化": "culture",
    "历史文化": "culture",
    "历史": "history",
    "历史景点": "history",
    "主要历史景点": "history",
    "博物馆": "museum",
    "情侣": "couple",
    "城市漫步": "citywalk",
    "城市景观": "citywalk",
    "经典城市景观": "citywalk",
    "经典景观": "citywalk",
    "citywalk": "citywalk",
    "亲子": "family",
    "夜生活": "nightlife",
    "购物": "shopping",
}

CANONICAL_INTERESTS = frozenset(INTEREST_KEYWORDS.values())

INTEREST_DISPLAY_LABELS: dict[str, str] = {
    "nature": "自然",
    "food": "美食",
    "culture": "文化",
    "history": "历史",
    "museum": "博物馆",
    "couple": "情侣",
    "citywalk": "城市漫步",
    "family": "亲子",
    "nightlife": "夜生活",
    "shopping": "购物",
}

PACE_DISPLAY_LABELS: dict[str, str] = {
    "relaxed": "轻松",
    "standard": "适中",
    "intensive": "紧凑",
}

BUDGET_DISPLAY_LABELS: dict[str, str] = {
    "low": "经济",
    "mid": "中等",
    "high": "高档",
}


def normalize_interest(tag: str) -> str:
    text = tag.strip()
    if not text:
        return text
    if text in INTEREST_KEYWORDS:
        return INTEREST_KEYWORDS[text]
    lower = text.lower()
    if lower in CANONICAL_INTERESTS:
        return lower
    return text


def format_interests_display(interests: list[str]) -> str:
    """兴趣标签 → 面向用户的中文展示。"""
    return "、".join(INTEREST_DISPLAY_LABELS.get(item, item) for item in interests)


def format_pace_display(pace: str) -> str:
    return PACE_DISPLAY_LABELS.get(pace, pace)


def format_budget_display(budget: str) -> str:
    return BUDGET_DISPLAY_LABELS.get(budget, budget)


def normalize_interests(interests: list[str]) -> list[str]:
    """中英文兴趣标签去重归一（自然/美食 → nature/food）。"""
    merged: list[str] = []
    for item in interests:
        canonical = normalize_interest(item)
        if canonical and canonical not in merged:
            merged.append(canonical)
    return merged


def extract_profile_rule_based(user_message: str) -> TravelProfile:
    return TravelProfile(
        destination=_extract_destination(user_message),
        days=_extract_days(user_message),
        start_date=_extract_start_date(user_message),
        budget_level=_extract_budget_level(user_message),
        budget_limit=_extract_budget_limit(user_message),
        interests=_extract_interests(user_message),
        companions=_extract_companions(user_message),
        party_size=_extract_party_size(user_message),
        pace=_extract_pace(user_message),
        hotel_area=_extract_hotel_area(user_message),
        food_preference=_extract_food_preferences(user_message),
        must_visit=_extract_must_visit(user_message),
        avoid=_extract_avoid(user_message),
        transport_mode=_extract_transport_mode(user_message),
    )


_NEGATION_MARKERS = ("不喜欢", "不想", "不要", "不再喜欢", "避开", "排除")


def extract_preference_removals(user_message: str) -> list[str]:
    """提取对已知兴趣标签的明确否定，不调用 LLM。"""
    removed: list[str] = []
    for keyword, tag in INTEREST_KEYWORDS.items():
        for match in re.finditer(re.escape(keyword), user_message, re.IGNORECASE):
            prefix = user_message[max(0, match.start() - 6) : match.start()]
            if any(marker in prefix for marker in _NEGATION_MARKERS):
                if tag not in removed:
                    removed.append(tag)
                break
    return removed


_SKIP_DESTINATIONS = frozenset({"哪", "哪儿", "哪里", "那", "什么", "啥", "一次"})
_DESTINATION_STOPWORDS = (
    "三天",
    "两天",
    "一天",
    "二天",
    "四天",
    "五天",
    "六天",
    "七天",
    "行程",
    "旅行",
    "旅游",
    "游玩",
    "攻略",
    "路线",
    "安排",
    "计划",
    "自由行",
)


def _extract_destination(text: str) -> str | None:
    metadata_match = re.search(r"目标位置\s*([^\],，,\s]+)", text)
    if metadata_match:
        dest = _clean_destination(metadata_match.group(1))
        if _valid_destination_candidate(dest):
            return dest
    for alias, city in CITY_ALIASES.items():
        if alias in text:
            return city
    patterns = (
        r"(?:规划|安排|做|制定|生成)\s*([^\s，,。！!？?；;]+?)(?:\d+\s*天|[一二两三四五六七]\s*天|行程|旅行|旅游|游玩|攻略|路线)",
        r"([一-龥A-Za-z][一-龥A-Za-z·'’]{1,20}?)(?:\d+\s*天|[一二两三四五六七]\s*天)(?:行程|旅行|旅游|游玩|攻略|路线)?",
        r"想去\s*([^\s，,。！!？?；;]+)",
        r"去\s*([^\s，,。！!？?；;]{2,10}?)(?:玩|旅|游|出差|看看|度假)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        dest = match.group(1).strip()
        dest = _clean_destination(dest)
        if _valid_destination_candidate(dest):
            return dest
    return None


def _valid_destination_candidate(destination: str | None) -> bool:
    if not destination or destination in _SKIP_DESTINATIONS:
        return False
    return not destination.startswith(("哪", "哪里", "哪儿", "什么", "一次"))


def _clean_destination(destination: str) -> str:
    cleaned = destination.strip(" ，,。！!？?；;的")
    changed = True
    while changed:
        changed = False
        for word in _DESTINATION_STOPWORDS:
            if cleaned.endswith(word):
                cleaned = cleaned[: -len(word)].strip(" 的")
                changed = True
    return cleaned


def _extract_days(text: str) -> int | None:
    metadata_match = re.search(r"旅行天数\s*(\d+)", text)
    if metadata_match:
        return int(metadata_match.group(1))
    match = re.search(r"(\d+)\s*天", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*日游", text)
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
        if f"{char}天" in text or f"{char}日游" in text:
            return value
    return None


def _extract_budget_level(text: str):
    limit = _extract_budget_limit(text)
    if "低预算" in text or "省钱" in text or "便宜" in text or (limit is not None and limit <= 3500):
        return "low"
    if "高预算" in text or "奢华" in text or "豪华" in text or "预算充足" in text:
        return "high"
    if "预算" in text or "中等" in text or limit is not None:
        return "mid"
    return None


def _extract_budget_limit(text: str) -> float | None:
    patterns = (
        r"预算(?:控制)?(?:在)?\s*(\d+(?:\.\d+)?)\s*元?(?:以内|以下|封顶)",
        r"(?:不超过|最多|上限)\s*(\d+(?:\.\d+)?)\s*元",
        r"总预算\s*(\d+(?:\.\d+)?)\s*元",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def _extract_start_date(text: str) -> str | None:
    match = re.search(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = re.search(r"(\d{1,2}月\d{1,2}[日号])", text)
    if match:
        return match.group(1)
    for token in ("今天", "明天", "后天", "本周末", "这周末", "下周末"):
        if token in text:
            return token
    return None


def _extract_interests(text: str) -> list[str]:
    removed = set(extract_preference_removals(text))
    raw: list[str] = []
    for keyword, tag in INTEREST_KEYWORDS.items():
        if keyword in text and tag not in raw and tag not in removed:
            raw.append(tag)
    return normalize_interests(raw)


def _extract_companions(text: str) -> str | None:
    if "情侣" in text:
        return "couple"
    child_match = re.search(r"一个([一二三四五六七八九十\d]+岁)孩子", text)
    if child_match:
        return f"一个{child_match.group(1)}孩子"
    if "亲子" in text or "小孩" in text or "孩子" in text:
        return "孩子"
    if "老人" in text or "父母" in text:
        return "父母" if "父母" in text else "老人"
    if "伴侣" in text:
        return "伴侣"
    if "一个人" in text or "独自" in text:
        return "solo"
    if re.search(r"(?:两|2)个人", text):
        return "两个人"
    if "朋友" in text:
        return "朋友"
    if "一家三口" in text:
        return "一家三口"
    return None


def _extract_party_size(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:个人|人)", text)
    if match:
        return max(1, int(match.group(1)))
    chinese = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
    match = re.search(r"([一两二三四五六])\s*个人", text)
    if match:
        return chinese[match.group(1)]
    if any(word in text for word in ("情侣", "夫妻")):
        return 2
    if "一家三口" in text:
        return 3
    if any(word in text for word in ("一个人", "独自")):
        return 1
    return None


def _extract_hotel_area(text: str) -> str | None:
    match = re.search(r"(?:住|住宿(?:在)?)([^，,。；;]{2,14}?(?:附近|一带|周边))", text)
    return match.group(1).strip() if match else None


def _extract_food_preferences(text: str) -> list[str]:
    values: list[str] = []
    if re.search(r"(?:不能(?:吃)?|不吃|忌)\s*(?:太)?辣", text):
        values.append("不辣")
    for cuisine in ("本帮菜", "川菜", "粤菜", "素食", "清真", "海鲜", "小吃"):
        if cuisine in text and not re.search(
            rf"(?:不吃|不能吃|不要吃|忌(?:口)?)\s*{re.escape(cuisine)}",
            text,
        ):
            values.append(cuisine)
    return values


def _extract_must_visit(text: str) -> list[str]:
    match = re.search(r"([^，,。；;]+?)\s*(?:仍然|还是)必须保留", text)
    if match:
        value = re.sub(r"^(?:但|不过|另外)", "", match.group(1)).strip()
        return [item.strip() for item in re.split(r"[、和与]", value) if item.strip()]
    match = re.search(r"(?:必须|务必)(?:安排|保留)\s*([^，,。；;]+)", text)
    if match:
        return [item.strip() for item in re.split(r"[、和与]", match.group(1)) if item.strip()]
    match = re.search(r"(?:必去|一定要去|必须去)\s*([^，,。；;]+)", text)
    if match:
        return [item.strip() for item in re.split(r"[、和与]", match.group(1)) if item.strip()]
    match = re.search(r"([^，,。；;]+?)\s*必须去", text)
    if match:
        prefix = re.sub(r"^.*?行程[，,]?", "", match.group(1)).strip()
        return [item.strip() for item in re.split(r"[、和与]", prefix) if item.strip()]
    match = re.search(
        r"(?:想去|想看)\s*([^，,。；;]+?)(?:，|,|。|；|;|一日游|两日游|三日游|$)",
        text,
    )
    if match:
        generic = {"海边", "咖啡店", "园林", "历史景点", "主要历史景点"}
        values = [item.strip() for item in re.split(r"[、和与]", match.group(1)) if item.strip()]
        named = [item for item in values if item not in generic and len(item) >= 2]
        if named:
            return named
    return []


def _extract_avoid(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"(?:避开|不要去|不去)\s*([^，,。；;]+)", text):
        values.extend(
            re.sub(r"(?:了|吧)$", "", item.strip()).strip()
            for item in re.split(r"[、和与]", match.group(1))
            if re.sub(r"(?:了|吧)$", "", item.strip()).strip()
            and len(re.sub(r"(?:了|吧)$", "", item.strip()).strip()) >= 2
            # 过滤「苏州不去了」误抽的单字尾巴
        )
    return values


def _extract_transport_mode(text: str):
    if re.search(r"不自驾|不能自驾|取消自驾|驾照没带", text):
        return "public_transport"
    if any(word in text for word in ("自驾", "开车")):
        return "drive"
    if re.search(r"公共交通|公交|地铁", text):
        return "public_transport"
    if any(word in text for word in ("打车", "出租车")):
        return "taxi"
    if re.search(r"全程步行|以步行为主|步行游|徒步|步行(?:路线|怎么走|前往|去|到)", text):
        return "walk"
    return "public_transport"


def _extract_pace(text: str) -> Pace:
    if "轻松" in text or "慢" in text or "不要太累" in text:
        return "relaxed"
    if "紧凑" in text or "多玩" in text or "尽量多" in text or ("节奏充实" in text and "朋友" in text):
        return "intensive"
    return "standard"
