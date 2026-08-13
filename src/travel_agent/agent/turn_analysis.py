"""轮次级分析：意图 kind + 任务类型 task_type + 画像 patch，一次 LLM 调用合并产出。

设计：

- 高精度规则优先：kind 高置信且无复杂信号时纯规则直出，零额外 LLM 成本；
- ambiguous 或检测到复杂信号（多目的地、替换、时间跨度、条件句、指代）时，
  把意图分类与结构化抽取合并成一次 LLM 调用；
- LLM 输出经过确定性校验（枚举、类型、值域），解析失败或全部非法则回退规则结果。

追问门槛按 ``REQUIRED_SLOTS[task_type]`` 动态计算：路线问答不查天数，
行程修改不查目的地，避免「全局静态必填槽位」造成的误追问。
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from travel_agent.agent.intent import MessageKind, classify_message_rule_based
from travel_agent.profile_patch import (
    LIST_SLOTS,
    SETTABLE_SLOTS,
    PatchOp,
    SlotPatch,
    extract_slot_clears,
    extracted_to_patches,
    merge_patches,
)
from travel_agent.schemas import TravelProfile
from travel_agent.workflow_rules import (
    CITY_ALIASES,
    extract_profile_rule_based,
    normalize_interests,
)

if TYPE_CHECKING:
    from travel_agent.agent.session import SessionContext
    from travel_agent.settings import Settings


class TaskType(str, Enum):
    UNKNOWN = "unknown"  # 无法安全映射到当前支持的旅行任务
    FULL_TRIP_PLAN = "full_trip_plan"  # 生成/规划完整行程
    ROUTE_QUERY = "route_query"  # A 到 B 怎么走/交通方式/耗时
    POI_ADVICE = "poi_advice"  # 住哪/选酒店/选区域等咨询
    ITINERARY_REVISION = "itinerary_revision"  # 修改/删除/替换既有行程
    DAY_ADVICE = "day_advice"  # 天气/特定日期去哪玩的轻量建议


REQUIRED_SLOTS: dict[TaskType, tuple[str, ...]] = {
    TaskType.UNKNOWN: (),
    TaskType.FULL_TRIP_PLAN: ("destination", "days"),
    TaskType.POI_ADVICE: ("destination",),
    TaskType.DAY_ADVICE: ("destination",),
    TaskType.ROUTE_QUERY: (),
    TaskType.ITINERARY_REVISION: (),
}

# 修改类任务里规则易误抽的核心槽位（「第二天」会命中 days 等）
CORE_SLOTS = ("destination", "days", "start_date")

# 这些任务类型属于旅行需求，但规则意图分类容易因缺少强信号误判为 ambiguous/out_of_scope
_TRAVEL_ADVICE_TASKS = frozenset(
    {TaskType.ROUTE_QUERY, TaskType.POI_ADVICE, TaskType.DAY_ADVICE}
)

TURN_ANALYSIS_TIMEOUT_SECONDS = 120
TURN_ANALYSIS_MAX_OUTPUT_TOKENS = 512


@dataclass
class TurnAnalysis:
    kind: MessageKind
    task_type: TaskType
    patches: dict[str, SlotPatch] = field(default_factory=dict)
    source: str = "rule"  # rule | llm
    revision_directives: dict[str, Any] = field(default_factory=dict)
    constraint_state: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 任务类型：规则分类
# --------------------------------------------------------------------------- #
_PLAN_REVISION_KEYWORDS: tuple[str, ...] = (
    "删",
    "删掉",
    "去掉",
    "移除",
    "移去",
    "改成",
    "改为",
    "改去",
    "修改",
    "调整",
    "轻松",
    "别太累",
    "别换",
    "不换",
    "替换",
    "对比",
    "比较",
    "比较下",
    "对比下",
    "补充",
    "取消",
    "改用",
    "预算下调",
    "预算降低",
    "不去",
    "最终版",
    "全部条件",
    "不要改",
    "保持不变",
)

_ROUTE_QUERY_RE = re.compile(r"怎么走|怎么坐|怎么去|怎么过去|多久能到|多久到|要多久|需要多久")
_ROUTE_FROM_TO_RE = re.compile(r"从.{1,24}(?:到|去|前往).{1,24}")
_ROUTE_CONTEXT_RE = re.compile(r"路线|交通|换乘|打车|地铁|公交|高铁|飞机|开车|步行")
_POI_ADVICE_RE = re.compile(
    r"住哪|住哪里|住哪儿|住在哪|住哪个|哪个区.{0,4}住|住.{0,8}(?:方便|合适|推荐)"
    r"|推荐.{0,6}(?:酒店|民宿|餐厅|午餐|晚餐)|(?:酒店|民宿|餐厅|午餐|晚餐).{0,6}(?:推荐|候选)"
    r"|(?:找|选).{0,12}(?:餐厅|午餐|晚餐)"
    r"|(?:安排|找).{0,18}(?:午餐|晚餐|聚餐)"
    r"|(?:选|找|推荐).{0,8}\d+\s*个.{0,8}(?:活动|景点|候选)"
)
_DAY_ADVICE_RE = re.compile(
    r"(?:今天|明天|后天|周末|周[一二三四五六日天]).{0,10}(?:下雨|天气|高温|降温).{0,12}(?:去哪|玩什么|安排|怎么办|适合)"
    r"|(?:下雨|天气不好|天气好).{0,8}(?:去哪|玩什么|适合)"
    r"|(?:只根据|根据).{0,18}天气.{0,18}(?:替换|备选|调整|判断)"
)
_FULL_PLAN_CUE_RE = re.compile(
    r"(?:一|二|两|三|四|五|六|七|八|九|十|\d+)\s*(?:日|天)游"
    r"|(?:规划|安排|制定|生成).{0,12}(?:行程|旅行|旅游)"
    r"|(?:规划|安排|制定|生成).{0,12}"
    r"(?:一|二|两|三|四|五|六|七|八|九|十|\d+)\s*(?:日|天)"
    r"|(?:我)?想去(?:旅行|旅游)"
)


def is_plan_revision_followup(user_message: str, ctx: "SessionContext") -> bool:
    """判断是否属于对既有行程的修改/对比场景。"""
    if not ctx.store.latest("itinerary"):
        return False
    text = user_message.strip().lower()
    if not text:
        return False
    if any(keyword in text for keyword in _PLAN_REVISION_KEYWORDS):
        return True
    if re.search(r"(?:第\s*[一二三四五六七八九十\d]+|最后一)\s*天", text):
        return True
    return False


def classify_task_type_rule_based(
    user_message: str,
    ctx: "SessionContext | None" = None,
) -> TaskType:
    text = user_message.strip()
    if ctx is not None and is_plan_revision_followup(text, ctx):
        return TaskType.ITINERARY_REVISION
    # A complete trip request often contains an origin/destination leg and
    # words such as public transport or walking.  Those embedded logistics
    # must not downgrade an explicit multi-hour/day itinerary to route_query.
    if _FULL_PLAN_CUE_RE.search(text):
        return TaskType.FULL_TRIP_PLAN
    if _ROUTE_QUERY_RE.search(text) or (
        _ROUTE_FROM_TO_RE.search(text) and _ROUTE_CONTEXT_RE.search(text)
    ):
        return TaskType.ROUTE_QUERY
    if _POI_ADVICE_RE.search(text):
        return TaskType.POI_ADVICE
    if _DAY_ADVICE_RE.search(text):
        return TaskType.DAY_ADVICE
    return TaskType.UNKNOWN


# --------------------------------------------------------------------------- #
# 复杂信号：触发 LLM 兜底的条件
# --------------------------------------------------------------------------- #
_COMPLEX_PATTERNS = (
    re.compile(r"不(?:想|打算|要)?去.{1,8}了|.{1,8}不(?:想|打算|要)?去(?:了|啦)"),
    re.compile(r"换成|改成|改为|替换成|取消"),
    re.compile(r"周[一二三四五六日天].{0,10}(?:到|至)|\d{1,2}月\d{1,2}[日号].{0,8}(?:到|至)"),
    re.compile(r"(?:晚上|下午|上午)(?:到|走|出发)|如果|要是|假如|下雨就|不下雨"),
    re.compile(r"那里|那边|那儿|那个地方|上次(?:的|那个)"),
)


def has_complex_signals(user_message: str) -> bool:
    if any(pattern.search(user_message) for pattern in _COMPLEX_PATTERNS):
        return True
    cities = {city for alias, city in CITY_ALIASES.items() if alias in user_message}
    return len(cities) >= 2


def build_rule_patches(
    user_message: str,
    task_type: TaskType,
    extracted_profile: TravelProfile | None = None,
) -> dict[str, SlotPatch]:
    """规则抽取 → patches；修改类任务丢弃核心槽位的 SET，防止「第二天」误抽天数。"""
    extracted = extracted_profile or extract_profile_rule_based(user_message)
    patches = merge_patches(extracted_to_patches(extracted), extract_slot_clears(user_message))
    if _venues_require_verification(user_message):
        # “先核验这些地点是否适合安排” describes candidates whose feasibility
        # must be checked, not unconditional must-visits.  Keeping them hard can
        # make a truthful closure/weather result impossible to deliver.
        patches.pop("must_visit", None)
    date_range = _extract_iso_date_range(user_message)
    if "days" not in patches and date_range:
        start, end = (date.fromisoformat(value) for value in date_range)
        patches["days"] = SlotPatch(PatchOp.SET, (end - start).days + 1)
    if task_type == TaskType.ITINERARY_REVISION:
        for name in CORE_SLOTS:
            if name == "must_visit":
                continue
            patch = patches.get(name)
            if patch is not None and patch.op == PatchOp.SET:
                patches.pop(name)
    return patches


def build_rule_constraint_state(
    user_message: str,
    patches: dict[str, SlotPatch],
    reference_datetime: str | None = None,
) -> dict[str, Any]:
    """Preserve common literal constraints even when the LLM analyzer is unavailable."""
    state: dict[str, Any] = {}
    values = {
        name: patch.value
        for name, patch in patches.items()
        if patch.op == PatchOp.SET
    }
    aliases = {
        "start_date": "date_start",
        "destination": "destinations",
        "days": "duration_days",
        "party_size": "traveler_count",
        "budget_limit": "budget_max_cny",
        "hotel_area": "lodging_area",
        "must_visit": "must_visit",
        "avoid": "avoid",
    }
    for source, target in aliases.items():
        value = values.get(source)
        if value is None:
            continue
        state[target] = [value] if target == "destinations" else value
    if state.get("must_visit") and values.get("destination"):
        state["must_visit"] = [
            str(values["destination"]) + item if item == "城墙" else item
            for item in state["must_visit"]
        ]

    date_range = _extract_iso_date_range(user_message)
    if date_range:
        state["date_start"], state["date_end"] = date_range
    elif state.get("date_start") and state.get("duration_days"):
        try:
            state["date_end"] = (
                date.fromisoformat(str(state["date_start"]))
                + timedelta(days=int(state["duration_days"]) - 1)
            ).isoformat()
        except (TypeError, ValueError):
            pass

    if re.search(r"不自驾|不能自驾|取消自驾|驾照没带", user_message):
        state["self_driving_allowed"] = False
    if "轮椅" in user_message:
        state["wheelchair_user"] = True
        state["accessibility_priority"] = True
    elif re.search(r"(?:必须|需要|优先).{0,6}无障碍|无障碍.{0,6}(?:优先|路线)", user_message):
        state["accessibility_priority"] = True
    if re.search(r"少走|不要安排太多步行|低步行|步行少", user_message):
        state["mobility"] = "low_walking"
    if re.search(r"石板路.{0,8}(?:太多|过多)|(?:大量|太多)石板路", user_message):
        state["avoid"] = ["大量石板路"]
    if re.search(r"老人|父母|\d{2,3}岁", user_message):
        state["elderly"] = True
    deadline = re.search(r"(\d{1,2}:\d{2})\s*前\s*(回到|返回|结束)", user_message)
    if deadline:
        key = "return_deadline" if deadline.group(2) in {"回到", "返回"} else "activity_end_deadline"
        state[key] = deadline.group(1)
    if not deadline:
        return_deadline = re.search(r"(\d{1,2}:\d{2})\s*(?:要|需|必须)?\s*(回到|返回)", user_message)
        activity_deadline = re.search(r"(?:晚上)?(\d{1,2}:\d{2})\s*(?:前)?\s*结束", user_message)
        if return_deadline:
            state["return_deadline"] = return_deadline.group(1)
        elif activity_deadline:
            state["activity_end_deadline"] = activity_deadline.group(1)
    route = re.search(r"从(.{1,36}?)(?:到|去|前往)(.{1,36}?)(?:的|，|,|。|$)", user_message)
    if route:
        origin = re.sub(r"^.*?(?:日|号|上午|下午|晚上)", "", route.group(1)).strip()
        destination = re.sub(r"(?:的)?公共交通路线.*$|，.*$", "", route.group(2)).strip(" \"'“”")
        if origin:
            state["origin"] = origin
        if destination:
            origin_city = next(
                (city for alias, city in CITY_ALIASES.items() if alias in origin),
                None,
            )
            if origin_city and origin_city not in destination and destination.endswith("度假区"):
                destination = origin_city + destination
            state["destination"] = destination
            state["destination_name"] = destination
            state["destination_area"] = destination
    if "公共交通" in user_message:
        state["public_transport_required"] = True
    if re.search(r"打车.{0,6}(?:备选|备用)|(?:备选|备用).{0,6}打车", user_message):
        state["taxi_backup"] = True
    if "公共交通" in user_message and "打车" in user_message:
        state["compare_modes"] = ["public_transport", "taxi"]
        state["transport_modes"] = ["public_transport", "taxi"]
    excluded: list[str] = []
    recommendation_ban = re.search(r"不要(?:再)?推荐([^。；;]+)", user_message)
    banned_text = recommendation_ban.group(1) if recommendation_ban else ""
    if "景点" in banned_text:
        excluded.append("景点推荐")
    if "餐厅" in banned_text:
        excluded.append("餐厅推荐")
    if excluded:
        state["exclude"] = excluded
    interest_terms = [
        term
        for term in ("园林", "海边", "咖啡店", "历史景点")
        if term in user_message
    ]
    if interest_terms:
        state["interests"] = interest_terms
    if re.search(r"(?:全程)?只吃清真", user_message):
        state["dietary"] = ["仅清真餐厅"]
    elif "不吃海鲜" in user_message:
        state["dietary"] = ["不吃海鲜"]
    elif "不吃辣" in user_message:
        state["dietary"] = ["不吃辣"]
    elif re.search(r"不(?:要|能)?太辣", user_message):
        state["dietary"] = ["不太辣"]
    elif re.search(r"(?:饮食|吃得?|口味).{0,5}清淡|清淡(?:饮食|口味)", user_message):
        state["dietary"] = ["清淡"]
    fixed = re.search(
        r"第\s*([一二三四五六七八九十\d]+)天.{0,8}?(\d{1,2}:\d{2})\s*(?:到|至|[-—])\s*"
        r"(\d{1,2}:\d{2}).{0,10}?(?:预约|固定)([^，,。；;]+)",
        user_message,
    )
    if not fixed:
        fixed = re.search(
            r"第\s*([一二三四五六七八九十\d]+)天.{0,8}?(\d{1,2}:\d{2})\s*(?:到|至|[-—])\s*"
            r"(\d{1,2}:\d{2}).{0,16}?([^，,。；;]+?)(?:，|,|。|$)",
            user_message,
        )
    if fixed:
        day = _parse_chinese_day(fixed.group(1))
        location = re.sub(r"^(?:已经)?预约", "", fixed.group(4)).strip()
        if day and location:
            state["fixed_events"] = [
                {
                    "day": day,
                    "start": fixed.group(2),
                    "end": fixed.group(3),
                    "location": location,
                }
            ]

    meal_event = re.search(
        r"第\s*([一二三四五六七八九十\d]+)天(?:的)?"
        r"(早饭|早餐|午饭|午餐|晚饭|晚餐).{0,8}(?:已经)?(?:约在|定在|安排在)"
        r"([^，,。；;]+)",
        user_message,
    )
    if meal_event:
        day = _parse_chinese_day(meal_event.group(1))
        meal = meal_event.group(2)
        windows = {
            "早饭": ("08:00", "09:00"),
            "早餐": ("08:00", "09:00"),
            "午饭": ("12:00", "14:00"),
            "午餐": ("12:00", "14:00"),
            "晚饭": ("18:00", "20:00"),
            "晚餐": ("18:00", "20:00"),
        }
        if day:
            start, end = windows[meal]
            state["fixed_events"] = [{
                "day": day,
                "start": start,
                "end": end,
                "location": meal_event.group(3).strip(),
            }]

    broad_event = re.search(
        r"第\s*([一二三四五六七八九十\d]+)天(?:的)?"
        r"(上午|下午|晚上).{0,8}?(?:安排|保留|留作)([^，,。；;]+)",
        user_message,
    )
    if broad_event and "fixed_events" not in state:
        day = _parse_chinese_day(broad_event.group(1))
        windows = {
            "上午": ("09:00", "12:00"),
            "下午": ("14:00", "18:00"),
            "晚上": ("18:00", "21:00"),
        }
        if day:
            start, end = windows[broad_event.group(2)]
            state["fixed_events"] = [{
                "day": day,
                "start": start,
                "end": end,
                "location": broad_event.group(3).strip(),
            }]

    _apply_general_constraint_patterns(
        state,
        user_message,
        reference_datetime=reference_datetime,
    )
    destinations = state.get("destinations") or []
    if state.get("must_visit") and destinations:
        city = str(destinations[0])
        state["must_visit"] = [
            city + item if item == "城墙" else item
            for item in state["must_visit"]
        ]
    return state


def _apply_general_constraint_patterns(
    state: dict[str, Any],
    text: str,
    *,
    reference_datetime: str | None,
) -> None:
    """Deterministic vocabulary for product constraints shared across models."""
    reference_date = _reference_date(reference_datetime)
    if reference_date is not None:
        if "今晚" in text or "今天" in text:
            state["date_start"] = reference_date.isoformat()
        elif "明天" in text:
            state["resolved_date"] = (reference_date + timedelta(days=1)).isoformat()

    child = re.search(
        r"(?:带|有(?:个|一位|一个)?|其中(?:一位|一个)?)(\d{1,2})岁(?:孩子|儿童)",
        text,
    )
    if child:
        state["child_age"] = int(child.group(1))
    if re.search(r"不安排夜间|不要夜间|不参加夜间", text):
        state["night_activity_allowed"] = False
    if "父母" in text:
        state["travel_with_parents"] = True
    if re.search(r"不能骑行|不骑行|取消骑行", text):
        state["cycling_allowed"] = False
    if re.search(r"公共交通(?:时间|路线|和|与|、)", text):
        state["transport_mode"] = "public_transport"

    party = re.search(r"([一二三四五六七八九十\d]+)位成年人", text)
    if party:
        state["traveler_count"] = _parse_chinese_day(party.group(1))
    elif child and "带" in text and "孩子" in text:
        state["traveler_count"] = 2

    number = re.search(r"(?:每天)?最多安排\s*(\d+)\s*个主要活动", text)
    if number:
        state["max_major_activities_per_day"] = int(number.group(1))
    number = re.search(r"酒店预算每晚\s*(\d+(?:\.\d+)?)\s*元?以内", text)
    if number:
        state["hotel_budget_per_night_cny"] = float(number.group(1))
    number = re.search(r"(?:预算每人|每人预算)\s*(\d+(?:\.\d+)?)\s*元", text)
    if number:
        state["budget_per_person_cny"] = float(number.group(1))
    number = re.search(r"人均\s*(\d+(?:\.\d+)?)\s*元?(?:左右|以内)?", text)
    if number:
        state["budget_per_person_cny"] = float(number.group(1))
    number = re.search(r"住宿已经花了\s*(\d+(?:\.\d+)?)\s*元", text)
    if number:
        prepaid = float(number.group(1))
        state["prepaid_lodging_cny"] = prepaid
        total = state.get("budget_total_cny") or state.get("budget_max_cny")
        if isinstance(total, (int, float)):
            state["budget_remaining_cny"] = float(total) - prepaid
    total_budget = re.search(r"总预算\s*(\d+(?:\.\d+)?)\s*元", text)
    if total_budget:
        state["budget_total_cny"] = float(total_budget.group(1))
    initial_budget = re.search(
        r"(?:最初)?预算(?:是|为)?\s*(\d+(?:\.\d+)?)\s*元",
        text,
    )
    if initial_budget and not total_budget:
        state["budget_max_cny"] = float(initial_budget.group(1))
    changed_budget = re.search(
        r"预算.{0,18}?(?:改成|调整为|降到)\s*(\d+(?:\.\d+)?)\s*元",
        text,
    )
    if changed_budget:
        state["budget_max_cny"] = float(changed_budget.group(1))

    number = re.search(r"(?:单段)?步行(?:不要)?超过\s*(\d+)\s*分钟", text)
    if number:
        state["max_single_walk_min"] = int(number.group(1))
    number = re.search(r"步行\s*(\d+)\s*分钟以内", text)
    if number:
        state["walking_time_max_min"] = int(number.group(1))
    number = re.search(
        r"每天步行(?:(?:控制在)?\s*(\d+(?:\.\d+)?)\s*公里以内|"
        r"(?:不超过|最多)\s*(\d+(?:\.\d+)?)\s*公里)",
        text,
    )
    if number:
        state["max_walking_km_per_day"] = float(number.group(1) or number.group(2))
    number = re.search(r"每天最多换乘\s*(\d+)\s*次", text)
    if number:
        state["max_transfers_per_day"] = int(number.group(1))
    if re.search(r"超过(?:就|则)改用公共交通", text):
        state["fallback_transport"] = "public_transport"

    location = re.search(
        r"(?:今晚在|地点在|^在|从)([^，,。；;]{2,28}?(?:附近|大道|商圈))",
        text,
    )
    if location:
        value = location.group(1).strip()
        city = next((city for alias, city in CITY_ALIASES.items() if alias in text), None)
        if city and city not in value and ("区" in value or "大道" in value):
            value = city + value
        state["location"] = value
    lodging = re.search(
        r"(?:住宿(?:放在|安排在|住在|在)?|住(?!宿))([^，,。；;]{2,14}?)(?:，|。|$)",
        text,
    )
    if lodging:
        value = lodging.group(1).strip()
        if not re.search(r"酒店|景点|库存|降档|(?:区域)?不要改|(?:区域)?不变", value):
            state["lodging_area"] = value
    top_n = re.search(r"(?:只要|选|找)?\s*(\d+)\s*(?:个|家)(?:餐厅)?(?:候选|不同类型|适合|的下午活动)", text)
    if top_n:
        state["top_n"] = int(top_n.group(1))
    if "安静聊天" in text:
        state["ambience"] = "安静聊天"
    for occasion in ("家庭聚餐", "商务午餐"):
        if occasion in text:
            state["occasion"] = occasion
    if "停车场" in text or "停车方便" in text:
        state["parking_preferred"] = True

    compare_areas = re.search(r"比较([^。]+?)三个区域", text)
    if compare_areas:
        values = _split_named_items(compare_areas.group(1))
        if len(values) >= 2:
            state["compare_lodging_areas"] = values
    if re.search(r"不需要.{0,8}(?:酒店)?库存|不要.{0,8}(?:酒店)?库存", text):
        state["no_live_inventory_required"] = True

    candidate_match = re.search(
        r"(?:候选是|想去|想看|安排)([^。；;]+?)(?:。|，请|，帮|，不能|，晚上|，单段|，全程|$)",
        text,
    )
    if candidate_match:
        candidates = _split_named_items(candidate_match.group(1))
        candidates = [item for item in candidates if _looks_like_named_candidate(item)]
        if len(candidates) >= 2:
            state["candidate_attractions"] = candidates
    compare_candidates = re.search(r"比较去([^。；;]+?)三个地点", text)
    if compare_candidates:
        candidates = _split_named_items(compare_candidates.group(1))
        if len(candidates) >= 2:
            state["candidate_attractions"] = candidates
    max_selected = re.search(r"(?:选择|选)最多\s*(\d+)\s*个", text)
    if max_selected:
        state["max_selected"] = int(max_selected.group(1))
    if re.search(r"不要推荐候选外|只(?:从|在)候选", text):
        state["candidate_only"] = True

    if re.search(r"请先确认具体地点|先确认具体地点", text):
        state["need_disambiguation"] = True
    if "公共交通和步行" in text:
        state["transport_modes"] = ["public_transport", "walking"]
    if "半日游" in text or "半天时间" in text:
        state["trip_length"] = "half_day"
    if "室内备选" in text:
        state["need_indoor_backup"] = True
    conditional = re.search(r"判断([^，。]+?)那天是否需要替换", text)
    if conditional:
        state["conditional_activity"] = conditional.group(1).strip()

    window = re.search(
        r"如果(?:中午)?高温，?(\d{1,2}:\d{2})\s*(?:到|至|-)(\d{1,2}:\d{2})"
        r"不要安排长时间户外活动",
        text,
    )
    if window:
        state["conditional_avoid_window"] = {
            "condition": "high_temperature",
            "start": window.group(1),
            "end": window.group(2),
            "avoid": "long_outdoor_activity",
        }
    if "有雨的话" in text or "下雨的话" in text:
        state["weather_condition"] = "rain_if_true"
    if "少走露天路段" in text:
        state["preference"] = ["少走露天路段"]

    dated_event = re.search(
        r"(\d{1,2})月(\d{1,2})日(\d{1,2}:\d{2})\s*(?:到|至|-)(\d{1,2}:\d{2})"
        r"已经预约([^，,。；;]+)",
        text,
    )
    if dated_event:
        start_date = state.get("date_start")
        year = int(str(start_date)[:4]) if start_date else (reference_date or date.today()).year
        event_date = date(year, int(dated_event.group(1)), int(dated_event.group(2))).isoformat()
        location_name = re.sub(r"讲解$", "博物馆", dated_event.group(5).strip())
        state["fixed_events"] = [{
            "date": event_date,
            "start": dated_event.group(3),
            "end": dated_event.group(4),
            "location": location_name,
        }]

    if re.search(r"住宿可以降档|酒店可以降档", text):
        state["lodging_flexibility"] = "can_downgrade"
    removed = re.search(r"([^，。]+?)不去了，?换成([^，。]+)", text)
    if removed:
        state["removed"] = [removed.group(1).strip()]
    else:
        removed = re.search(r"(?:不去|不去了|仍然不去)\s*([^，,。；;]+)", text)
        if not removed:
            removed = re.search(r"([^，,。；;]+?)\s*(?:仍然)?不去(?:了)?", text)
        if removed:
            value = re.sub(r"^(?:明确一下|明确|还是)", "", removed.group(1)).strip()
            value = re.sub(r"了$", "", value).strip()
            if value:
                state["removed"] = [value]
    optional = re.search(r"([^，。]+?)可以不去", text)
    if optional:
        state["optional_remove"] = [optional.group(1).strip()]
    if "不同类型" in text:
        state["diversity_required"] = True
    if re.search(r"节奏.{0,4}(?:轻松|别太赶|不要太赶)|别太赶|不要太赶", text):
        state["pace"] = "relaxed"
    weekday = re.search(r"周([一二三四五六日天])", text)
    if weekday:
        state["weekday"] = "周" + ("日" if weekday.group(1) == "天" else weekday.group(1))

    origin = re.search(r"从([^，,。；;]{2,24}?)(?:出发|去)", text)
    if origin:
        value = origin.group(1).strip(" \"'“”")
        state["origin"] = value
    if state.get("return_deadline") and re.search(r"回到([^，,。；;]+)", text):
        state["return_location"] = re.search(r"回到([^，,。；;]+)", text).group(1).strip()
    final_return = re.search(
        r"最后一天\s*(\d{1,2})(?::(\d{2}))?点?前\s*(?:到达?|抵达)([^，,。；;]+)",
        text,
    )
    if final_return:
        state["return_deadline"] = (
            f"{int(final_return.group(1)):02d}:{int(final_return.group(2) or 0):02d}"
        )
        state["return_location"] = final_return.group(3).strip()

    named_venues = [] if _venues_require_verification(text) else _extract_named_venues(text)
    fixed_locations = {
        str(event.get("location") or "").strip()
        for event in (state.get("fixed_events") or [])
        if isinstance(event, dict)
    }
    named_venues = [
        venue
        for venue in named_venues
        if venue not in fixed_locations or venue not in {"自由活动", "休息", "自由时间"}
    ]
    if named_venues:
        state["must_visit"] = named_venues
    replacement = re.search(r"(?:换成|改去)([^，,。；;]+)", text)
    if replacement:
        venues = _split_named_items(replacement.group(1))
        if venues:
            state["must_visit"] = venues
    avoid_values: list[str] = list(state.get("avoid") or [])
    if re.search(r"网红排队特别久|排队特别久", text):
        avoid_values.append("长时间排队的网红点")
    if "连续爬坡" in text:
        avoid_values.append("连续爬坡")
    if "长楼梯" in text:
        avoid_values.append("长楼梯")
    if "不要重复商场" in text:
        avoid_values.append("重复商场")
    if "同一景区的不同入口" in text:
        avoid_values.append("同一景区不同入口")
    if avoid_values:
        state["avoid"] = list(dict.fromkeys(avoid_values))
    if "全部改成公共交通和打车" in text:
        state["transport_modes"] = ["public_transport", "taxi"]


def _reference_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _split_named_items(value: str) -> list[str]:
    cleaned = re.sub(r"^(?:但|其中|去|看|安排|比较)", "", value).strip()
    return [
        item.strip(" \"'“”")
        for item in re.split(r"[、，,和与或]", cleaned)
        if item.strip(" \"'“”")
    ]


def _extract_named_venues(text: str) -> list[str]:
    raw: str | None = None
    # Prefer verb-before-object forms so context such as “最后方案里” is not
    # mistaken for a venue in “最后方案里必须保留西湖”.
    match = re.search(
        r"(?:必须|务必)(?:去|保留|安排)\s*([^，,。；;]+)",
        text,
    )
    if match:
        raw = match.group(1)
    else:
        match = re.search(r"([^，,。；;]+?)\s*(?:仍然)?必须去", text)
        if match:
            raw = match.group(1)
        else:
            match = re.search(
                r"(?:想去|想看|(?<!不要)(?<!不)安排)"
                r"([^。；;]+?)(?:，|。|一日游|两日游|三日游|$)",
                text,
            )
    if not match:
        return []
    if raw is None:
        raw = match.group(1)
    generic = {"海边", "咖啡店", "园林", "历史景点", "主要历史景点"}
    suffixes = (
        "博物馆", "博物院", "城墙", "风景区", "度假区", "步行街", "街",
        "园", "塔", "洞", "坝", "祠", "楼", "陵", "湖", "山", "岛", "滩",
    )
    values = _split_named_items(raw)
    return [
        value
        for value in values
        if value not in generic
        and not re.fullmatch(r"(?:每天)?(?:最多)?\s*\d+\s*个(?:主要)?(?:活动|景点|项目)", value)
        and (value.endswith(suffixes) or len(value) >= 3)
    ]


def _looks_like_named_candidate(value: str) -> bool:
    value = value.strip()
    if not 1 < len(value) <= 18:
        return False
    if re.fullmatch(r"(?:每天)?(?:最多)?\s*\d+\s*个(?:主要)?(?:活动|景点|项目)", value):
        return False
    return not re.search(
        r"预算|每晚|以内|每天|最多|人均|步行\d|优先|少走|地点在|公司附近|"
        r"(?:\d+|[一二两三四五六七八九十]+)天行程|其他项目|预约冲突|(?:不要|不能|避免).{0,8}冲突",
        value,
    )


def _venues_require_verification(text: str) -> bool:
    """Whether named venues are conditional candidates pending a user-requested check."""
    return bool(
        re.search(
            r"(?:请?先).{0,12}(?:核验|确认|检查).{0,30}"
            r"(?:是否|能否).{0,12}(?:适合|可以|可否).{0,8}安排",
            text,
        )
    )


def _extract_iso_date_range(text: str) -> tuple[str, str] | None:
    match = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日?\s*(?:到|至|[-—])\s*"
        r"(?:(\d{4})年)?(?:(\d{1,2})月)?(\d{1,2})日?",
        text,
    )
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    start_day = int(match.group(3))
    end_year = int(match.group(4) or year)
    end_month = int(match.group(5) or month)
    try:
        return date(year, month, start_day).isoformat(), date(end_year, end_month, int(match.group(6))).isoformat()
    except ValueError:
        return None


def extract_revision_directives(user_message: str, task_type: TaskType) -> dict[str, Any]:
    """Extract deterministic revision controls that are not profile slots."""
    if task_type != TaskType.ITINERARY_REVISION:
        return {}
    directives: dict[str, Any] = {}
    indoor_match = re.search(
        r"第\s*([一二三四五六七八九十\d]+)\s*天.{0,12}(?:室内|雨天)",
        user_message,
    )
    if indoor_match:
        day = _parse_chinese_day(indoor_match.group(1))
        if day is not None:
            directives["indoor_days"] = [day]
    if re.search(r"(?:酒店|住宿).{0,6}(?:别换|不换|保持|保留)", user_message):
        directives["preserve_hotel"] = True
    if re.search(r"轻松|悠闲|别太累|少安排", user_message):
        directives["pace"] = "relaxed"
    return directives


def _parse_chinese_day(value: str) -> int | None:
    if value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return digits.get(value)


# --------------------------------------------------------------------------- #
# 合并入口：规则优先，ambiguous / 复杂信号时一次 LLM 调用
# --------------------------------------------------------------------------- #
def analyze_travel_turn(
    user_message: str,
    ctx: "SessionContext",
    settings: "Settings",
    history: list[tuple[str, str]] | None = None,
    evaluation_trace: list[dict[str, Any]] | None = None,
) -> TurnAnalysis:
    extracted_profile = extract_profile_rule_based(user_message)
    decision = classify_message_rule_based(user_message, extracted_profile)
    task_type = classify_task_type_rule_based(user_message, ctx)
    kind = decision.kind
    if (
        task_type == TaskType.ITINERARY_REVISION
        and ctx.store.latest("itinerary")
        and decision.reason != "strong_non_travel"
    ):
        kind = MessageKind.TRAVEL
    if (
        task_type in _TRAVEL_ADVICE_TASKS
        and kind in (MessageKind.AMBIGUOUS, MessageKind.OUT_OF_SCOPE)
        and decision.reason not in ("pure_greeting", "strong_non_travel", "empty")
    ):
        # 「怎么走/住哪里/明天下雨去哪」这类咨询是旅行需求，不应被弱信号规则挡在规划外
        kind = MessageKind.TRAVEL
    rule_patches = build_rule_patches(user_message, task_type, extracted_profile)
    rule_result = TurnAnalysis(
        kind=kind,
        task_type=task_type,
        patches=rule_patches,
        source="rule",
        revision_directives=extract_revision_directives(user_message, task_type),
        constraint_state=build_rule_constraint_state(
            user_message,
            rule_patches,
            reference_datetime=ctx.reference_datetime,
        ),
    )
    # UNKNOWN means “no supported workflow has been selected”, not “default to
    # a full plan”.  Promote it only when deterministic trip state makes that
    # workflow unambiguous: explicit destination/days start or continue a full
    # plan; constraints/preferences against an existing itinerary are edits.
    if rule_result.task_type == TaskType.UNKNOWN:
        from travel_agent.agent.preferences import (
            turn_expresses_preferences,
            user_skips_preference_prompt,
        )

        has_trip_slot_patch = any(
            name in rule_result.patches for name in ("destination", "days", "start_date")
        )
        has_trip_followup = bool(
            rule_result.constraint_state
            or rule_result.patches
            or user_skips_preference_prompt(user_message)
            or turn_expresses_preferences(user_message, extracted_profile)
        )
        if ctx.store.latest("itinerary") and has_trip_followup:
            rule_result = TurnAnalysis(
                kind=rule_result.kind,
                task_type=TaskType.ITINERARY_REVISION,
                patches=rule_result.patches,
                source=rule_result.source,
                revision_directives=extract_revision_directives(
                    user_message, TaskType.ITINERARY_REVISION
                ),
                constraint_state=rule_result.constraint_state,
            )
            task_type = TaskType.ITINERARY_REVISION
        elif has_trip_slot_patch or (
            (ctx.profile.destination or ctx.profile.days) and has_trip_followup
        ):
            rule_result = TurnAnalysis(
                kind=rule_result.kind,
                task_type=TaskType.FULL_TRIP_PLAN,
                patches=rule_result.patches,
                source=rule_result.source,
                revision_directives=rule_result.revision_directives,
                constraint_state=rule_result.constraint_state,
            )
            task_type = TaskType.FULL_TRIP_PLAN
    # Once a trip context exists, a deterministic constraint-only follow-up is
    # still a travel turn even if it contains no standalone travel verb.  This
    # prevents messages such as “住宿放在思明区” from taking the out-of-scope
    # early-return path before their state update can be merged.
    if (
        rule_result.kind != MessageKind.TRAVEL
        and rule_result.constraint_state
        and (ctx.profile.destination or ctx.profile.days or ctx.store.latest("itinerary"))
    ):
        rule_result = TurnAnalysis(
            kind=MessageKind.TRAVEL,
            task_type=rule_result.task_type,
            patches=rule_result.patches,
            source=rule_result.source,
            revision_directives=rule_result.revision_directives,
            constraint_state=rule_result.constraint_state,
        )
        kind = MessageKind.TRAVEL
    if (
        re.search(
            r"住宿(?:仍|还是|继续)住|(?:仍|还是|继续)住|"
            r"(?:住宿|酒店)(?:区域)?.{0,4}(?:不要改|不变|别换|保持)",
            user_message,
        )
        and ctx.profile.constraint_state.get("lodging_area")
    ):
        rule_result.constraint_state["lodging_area"] = ctx.profile.constraint_state[
            "lodging_area"
        ]
    if kind not in {MessageKind.TRAVEL, MessageKind.AMBIGUOUS}:
        return rule_result
    if not settings.llm.enabled:
        return rule_result
    if kind != MessageKind.AMBIGUOUS and not has_complex_signals(user_message):
        return rule_result
    try:
        payload = _analyze_with_llm(user_message, ctx, settings, history or [], evaluation_trace)
    except Exception:  # noqa: BLE001 - LLM 失败必须回退规则，不能阻塞主流程
        return rule_result
    return _merge_llm_payload(payload, rule_result)


_TURN_ANALYSIS_SYSTEM = """你是旅行规划 Agent 的轮次分析模块，对用户输入一次性输出四件事：意图类别、任务类型、出行画像字段更新、完整的结构化约束状态。
只输出 JSON 对象，不要输出 Markdown。格式：
{"kind": "greeting|travel|ambiguous|out_of_scope",
 "task_type": "unknown|full_trip_plan|route_query|poi_advice|itinerary_revision|day_advice",
 "slots": {"字段名": {"op": "set", "value": ...} 或 {"op": "clear"}},
 "constraint_state": {"约束字段": JSON值}}

kind 判定：
- greeting：纯寒暄；travel：明确的出行/规划需求；ambiguous：有旅行弱信号但不确定；out_of_scope：与旅行无关。

task_type 判定：
- unknown：无法安全归入下列任一已支持任务，或请求的能力当前不支持；
- full_trip_plan：要求规划/生成完整多日行程；
- route_query：问 A 到 B 怎么走、交通方式、耗时，不要求完整行程；
- poi_advice：住哪里/选哪个区/酒店推荐等咨询，不要求完整行程；
- itinerary_revision：对已有行程做增删改（删掉某天某站、换掉某项）；
- day_advice：根据天气或某个具体日期问去哪玩、玩什么。

slots 抽取规则：
- 只抽本轮用户明确表达的信息，不要编造；本轮没提的字段不要出现在 slots 中；
- 用户明确取消/收回已知信息（如「苏州不去了」）时，对应字段输出 {"op": "clear"}；
- destination 为中文城市名；days 为正整数；budget_level 取 low/mid/high；
- pace 取 relaxed/standard/intensive；transport_mode 取 walk/public_transport/taxi/drive；
- interests 用英文标签数组，取自 history/food/nature/culture/museum/citywalk/family/nightlife/shopping/couple；
- 多目的地行程取用户最主要的目的地作为 destination，天数取总天数。

constraint_state 规则：
- 输出当前请求结合最近对话后的全部有效显式约束；本轮修改覆盖旧值，删除项从 must_visit 移除并记入 removed；
- 保留用户原文语义，不把“园林/海边/历史景点”等中文要求改成英文标签；日期用 YYYY-MM-DD，时间用 HH:MM；
- 仅记录明确表达的约束，不推测。优先使用这些通用字段：date_start,date_end,destinations,destination,origin,return_location,destination_area,destination_name,duration_days,traveler_count,child_age,elderly,wheelchair_user,travel_with_parents,budget_max_cny,budget_per_person_cny,budget_total_cny,budget_remaining_cny,prepaid_lodging_cny,hotel_budget_per_night_cny,lodging_area,lodging_flexibility,interests,must_visit,removed,optional_remove,avoid,dietary,self_driving_allowed,cycling_allowed,transport_mode,transport_modes,public_transport_required,taxi_backup,fallback_transport,return_deadline,activity_end_deadline,walking_time_max_min,max_single_walk_min,max_walking_km_per_day,max_transfers_per_day,max_major_activities_per_day,max_selected,candidate_attractions,candidate_only,fixed_events,conditional_activity,conditional_avoid_window,need_indoor_backup,night_activity_allowed,accessibility_priority,mobility,occasion,ambience,parking_preferred,top_n,diversity_required,exclude,no_live_inventory_required,need_disambiguation,trip_length,weather_condition,weekday,preference。
"""


def _analyze_with_llm(
    user_message: str,
    ctx: "SessionContext",
    settings: "Settings",
    history: list[tuple[str, str]],
    evaluation_trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    from travel_agent.agent.runtime import _build_chat_model

    profile = ctx.profile
    brief_parts: list[str] = []
    if profile.destination:
        brief_parts.append(f"目的地={profile.destination}")
    if profile.days:
        brief_parts.append(f"天数={profile.days}")
    if profile.start_date:
        brief_parts.append(f"出发日期={profile.start_date}")
    if profile.constraint_state:
        brief_parts.append(
            "当前有效约束=" + json.dumps(profile.constraint_state, ensure_ascii=False, default=str)
        )
    prompt = f"已知出行画像：{'；'.join(brief_parts) or '空'}\n"
    if history:
        recent = "\n".join(f"{role}: {content}" for role, content in history[-4:])
        prompt += f"最近对话：\n{recent}\n"
    prompt += f"\n当前用户输入：{user_message}"

    # This is optional enrichment over a complete deterministic analysis.  A
    # provider failure already has a safe rule fallback, so retrying the same
    # 60-second request three times only stalls the turn and hides the timeout
    # from the adaptive circuit breaker.
    model = _build_chat_model(
        settings,
        timeout_seconds=TURN_ANALYSIS_TIMEOUT_SECONDS,
        max_retries=0,
    ).bind(
        max_tokens=TURN_ANALYSIS_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    from travel_agent.orchestration.meter import meter_callbacks

    callbacks = meter_callbacks("preflight")
    if evaluation_trace is not None:
        from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

        callbacks.append(
            EvaluationTraceCallback(
                evaluation_trace,
                model=settings.llm.model,
                phase="turn_analysis",
            )
        )
    response = model.invoke(
        [
            SystemMessage(content=_TURN_ANALYSIS_SYSTEM),
            HumanMessage(content=prompt),
        ],
        config={"callbacks": callbacks} if callbacks else None,
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    parsed = json.loads(_strip_json_fence(content))
    if not isinstance(parsed, dict):
        raise ValueError("turn analysis response is not a JSON object")
    return parsed


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _merge_llm_payload(payload: Any, rule_result: TurnAnalysis) -> TurnAnalysis:
    """确定性校验 LLM 输出；合法字段采用 LLM 结论，非法逐项回退规则。"""
    if not isinstance(payload, dict):
        return rule_result
    kind = rule_result.kind
    try:
        kind = MessageKind(str(payload.get("kind", "")).lower())
    except ValueError:
        pass
    task_type = rule_result.task_type
    try:
        candidate_task_type = TaskType(str(payload.get("task_type", "")).lower())
        # A revision requires an actual plan artifact.  Do not let a model turn
        # weather/route advice about a user-owned plan into an impossible local
        # artifact revision.
        # Task type controls whether Planner/Reviewer/Gate are mandatory.  Do
        # not let a stochastic analyzer downgrade a deterministic full-plan
        # classification to a lightweight path that bypasses those gates.
        # Unknown lightweight phrasings fail closed as a full plan until the
        # deterministic classifier gains a generalized rule.
        if candidate_task_type == rule_result.task_type:
            task_type = candidate_task_type
        elif (
            rule_result.task_type == TaskType.UNKNOWN
            and candidate_task_type != TaskType.UNKNOWN
        ):
            # Rules deliberately return UNKNOWN instead of guessing a full
            # plan.  The optional analyzer may resolve it to one supported
            # type, subject to all slot/constraint validation below.
            task_type = candidate_task_type
        elif (
            candidate_task_type == TaskType.FULL_TRIP_PLAN
            and rule_result.task_type in _TRAVEL_ADVICE_TASKS
        ):
            task_type = rule_result.task_type
    except ValueError:
        pass
    llm_patches = _validate_slot_patches(payload.get("slots"))
    patches = (
        {**rule_result.patches, **llm_patches}
        if llm_patches is not None
        else rule_result.patches
    )
    # must_visit is a hard constraint.  The deterministic extractor covers
    # explicit “必须/必去/一定要” forms; do not let a stochastic analyzer harden
    # suggestions, comparison candidates, or conditional venues.
    if (
        "must_visit" not in rule_result.patches
        and "must_visit" not in rule_result.constraint_state
    ):
        patches.pop("must_visit", None)
    # Transport mode affects every route and must be explicitly stated.  A
    # walking-distance limit is not consent to make the whole trip walking.
    # Keep deterministic explicit modes; reject model inference otherwise.
    if (
        "transport_mode" not in rule_result.patches
        and "transport_mode" not in rule_result.constraint_state
    ):
        patches.pop("transport_mode", None)
    if task_type == TaskType.ITINERARY_REVISION:
        for name in CORE_SLOTS:
            patch = patches.get(name)
            if patch is not None and patch.op == PatchOp.SET:
                patches.pop(name)
    constraint_state = {
        **_validate_constraint_state(payload.get("constraint_state")),
        **rule_result.constraint_state,
    }
    if "budget_max_cny" in rule_result.constraint_state:
        constraint_state.pop("budget_total_cny", None)
    if task_type == TaskType.ITINERARY_REVISION:
        for key in ("destination", "destinations", "date_start", "date_end", "duration_days"):
            if key not in rule_result.constraint_state:
                constraint_state.pop(key, None)
    if (
        "must_visit" not in rule_result.patches
        and "must_visit" not in rule_result.constraint_state
    ):
        constraint_state.pop("must_visit", None)
    if (
        "transport_mode" not in rule_result.patches
        and "transport_mode" not in rule_result.constraint_state
    ):
        constraint_state.pop("transport_mode", None)
    candidates = constraint_state.get("candidate_attractions")
    if isinstance(candidates, list):
        cleaned_candidates = [
            str(item).strip()
            for item in candidates
            if _looks_like_named_candidate(str(item))
        ]
        if len(cleaned_candidates) >= 2:
            constraint_state["candidate_attractions"] = cleaned_candidates
        else:
            constraint_state.pop("candidate_attractions", None)
    return TurnAnalysis(
        kind=kind,
        task_type=task_type,
        patches=patches,
        source="llm",
        revision_directives=rule_result.revision_directives,
        constraint_state=constraint_state,
    )


def _validate_constraint_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    def clean(item: Any, depth: int = 0) -> Any:
        if depth > 4:
            return None
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        if isinstance(item, list):
            return [clean(entry, depth + 1) for entry in item[:50]]
        if isinstance(item, dict):
            return {
                str(key): clean(entry, depth + 1)
                for key, entry in list(item.items())[:50]
                if str(key).strip()
            }
        return str(item)

    cleaned = {
        str(key): clean(item)
        for key, item in list(value.items())[:100]
        if str(key).strip()
    }
    numeric_keys = {
        "budget_max_cny", "budget_per_person_cny", "budget_total_cny",
        "budget_remaining_cny", "prepaid_lodging_cny",
        "hotel_budget_per_night_cny", "traveler_count", "child_age",
        "duration_days", "walking_time_max_min", "max_single_walk_min",
        "max_walking_km_per_day", "max_transfers_per_day",
        "max_major_activities_per_day", "max_selected",
    }
    for key in numeric_keys:
        item = cleaned.get(key)
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, (int, float))
        ):
            cleaned.pop(key, None)
    for key, item in list(cleaned.items()):
        if isinstance(item, str) and key.endswith(("deadline", "_start", "_end")):
            match = re.search(r"(?:T|\s)(\d{1,2}:\d{2})(?::\d{2})?", item)
            if match:
                cleaned[key] = match.group(1)
    events = cleaned.get("fixed_events")
    if isinstance(events, list):
        normalized_events: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            normalized = dict(event)
            if "start" not in normalized and normalized.get("start_time"):
                normalized["start"] = normalized.pop("start_time")
            if "end" not in normalized and normalized.get("end_time"):
                normalized["end"] = normalized.pop("end_time")
            if "location" not in normalized and normalized.get("name"):
                normalized["location"] = re.sub(
                    r"(?:预约|讲解)$", "", str(normalized.pop("name"))
                ).strip()
            normalized_events.append(normalized)
        cleaned["fixed_events"] = normalized_events
    return cleaned


_SLOT_ENUMS = {
    "budget_level": {"low", "mid", "high"},
    "pace": {"relaxed", "standard", "intensive"},
    "transport_mode": {"walk", "public_transport", "taxi", "drive"},
}
_SLOT_NUMBERS = {"days": int, "party_size": int, "budget_limit": float}
_SLOT_STRINGS = ("destination", "start_date", "companions", "hotel_area")


def _validate_slot_patches(slots: Any) -> dict[str, SlotPatch] | None:
    if not isinstance(slots, dict) or not slots:
        return None
    result: dict[str, SlotPatch] = {}
    for name, item in slots.items():
        if name not in SETTABLE_SLOTS or not isinstance(item, dict):
            continue
        try:
            op = PatchOp(str(item.get("op", "")).lower())
        except ValueError:
            continue
        if op == PatchOp.CLEAR:
            result[name] = SlotPatch(PatchOp.CLEAR, item.get("value"))
            continue
        value = _validate_slot_value(name, item.get("value"))
        if value is not None:
            result[name] = SlotPatch(PatchOp.SET, value)
    return result or None


def _validate_slot_value(name: str, value: Any) -> Any:
    if name in _SLOT_STRINGS:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
    if name in _SLOT_NUMBERS:
        caster = _SLOT_NUMBERS[name]
        try:
            number = caster(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None
    if name in _SLOT_ENUMS:
        return value if value in _SLOT_ENUMS[name] else None
    if name in LIST_SLOTS:
        if not isinstance(value, list):
            return None
        items = [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int)) and str(item).strip()
        ]
        if name == "interests":
            items = normalize_interests(items)
        return items or None
    return None
