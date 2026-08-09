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
from travel_agent.workflow_rules import (
    CITY_ALIASES,
    extract_profile_rule_based,
    normalize_interests,
)

if TYPE_CHECKING:
    from travel_agent.agent.session import SessionContext
    from travel_agent.settings import Settings


class TaskType(str, Enum):
    FULL_TRIP_PLAN = "full_trip_plan"  # 生成/规划完整行程
    ROUTE_QUERY = "route_query"  # A 到 B 怎么走/交通方式/耗时
    POI_ADVICE = "poi_advice"  # 住哪/选酒店/选区域等咨询
    ITINERARY_REVISION = "itinerary_revision"  # 修改/删除/替换既有行程
    DAY_ADVICE = "day_advice"  # 天气/特定日期去哪玩的轻量建议


REQUIRED_SLOTS: dict[TaskType, tuple[str, ...]] = {
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


@dataclass
class TurnAnalysis:
    kind: MessageKind
    task_type: TaskType
    patches: dict[str, SlotPatch] = field(default_factory=dict)
    source: str = "rule"  # rule | llm
    revision_directives: dict[str, Any] = field(default_factory=dict)


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
)

_ROUTE_QUERY_RE = re.compile(r"怎么走|怎么坐|怎么去|怎么过去|多久能到|多久到|要多久|需要多久")
_ROUTE_FROM_TO_RE = re.compile(r"从.{1,12}到.{1,12}")
_ROUTE_CONTEXT_RE = re.compile(r"路线|交通|换乘|打车|地铁|公交|高铁|飞机|开车|步行")
_POI_ADVICE_RE = re.compile(
    r"住哪|住哪里|住哪儿|住在哪|住哪个|哪个区.{0,4}住|住.{0,8}(?:方便|合适|推荐)"
    r"|推荐.{0,6}(?:酒店|民宿)|(?:酒店|民宿).{0,4}推荐"
)
_DAY_ADVICE_RE = re.compile(
    r"(?:今天|明天|后天|周末|周[一二三四五六日天]).{0,10}(?:下雨|天气|高温|降温).{0,12}(?:去哪|玩什么|安排|怎么办|适合)"
    r"|(?:下雨|天气不好|天气好).{0,8}(?:去哪|玩什么|适合)"
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
    if re.search(r"第\s*[一二三四五六七八九十\d]+\s*天", text):
        return True
    return False


def classify_task_type_rule_based(
    user_message: str,
    ctx: "SessionContext | None" = None,
) -> TaskType:
    text = user_message.strip()
    if ctx is not None and is_plan_revision_followup(text, ctx):
        return TaskType.ITINERARY_REVISION
    if _ROUTE_QUERY_RE.search(text) or (
        _ROUTE_FROM_TO_RE.search(text) and _ROUTE_CONTEXT_RE.search(text)
    ):
        return TaskType.ROUTE_QUERY
    if _POI_ADVICE_RE.search(text):
        return TaskType.POI_ADVICE
    if _DAY_ADVICE_RE.search(text):
        return TaskType.DAY_ADVICE
    return TaskType.FULL_TRIP_PLAN


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


def build_rule_patches(user_message: str, task_type: TaskType) -> dict[str, SlotPatch]:
    """规则抽取 → patches；修改类任务丢弃核心槽位的 SET，防止「第二天」误抽天数。"""
    extracted = extract_profile_rule_based(user_message)
    patches = merge_patches(extracted_to_patches(extracted), extract_slot_clears(user_message))
    if task_type == TaskType.ITINERARY_REVISION:
        for name in CORE_SLOTS:
            patch = patches.get(name)
            if patch is not None and patch.op == PatchOp.SET:
                patches.pop(name)
    return patches


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
    decision = classify_message_rule_based(user_message)
    task_type = classify_task_type_rule_based(user_message, ctx)
    kind = decision.kind
    if (
        task_type in _TRAVEL_ADVICE_TASKS
        and kind in (MessageKind.AMBIGUOUS, MessageKind.OUT_OF_SCOPE)
        and decision.reason not in ("pure_greeting", "strong_non_travel", "empty")
    ):
        # 「怎么走/住哪里/明天下雨去哪」这类咨询是旅行需求，不应被弱信号规则挡在规划外
        kind = MessageKind.TRAVEL
    rule_result = TurnAnalysis(
        kind=kind,
        task_type=task_type,
        patches=build_rule_patches(user_message, task_type),
        source="rule",
        revision_directives=extract_revision_directives(user_message, task_type),
    )
    if kind != MessageKind.AMBIGUOUS and not has_complex_signals(user_message):
        return rule_result
    if not settings.llm.enabled:
        return rule_result
    try:
        payload = _analyze_with_llm(user_message, ctx, settings, history or [], evaluation_trace)
    except Exception:  # noqa: BLE001 - LLM 失败必须回退规则，不能阻塞主流程
        return rule_result
    return _merge_llm_payload(payload, rule_result)


_TURN_ANALYSIS_SYSTEM = """你是旅行规划 Agent 的轮次分析模块，对用户输入一次性输出三件事：意图类别、任务类型、出行画像字段更新。
只输出 JSON 对象，不要输出 Markdown。格式：
{"kind": "greeting|travel|ambiguous|out_of_scope",
 "task_type": "full_trip_plan|route_query|poi_advice|itinerary_revision|day_advice",
 "slots": {"字段名": {"op": "set", "value": ...} 或 {"op": "clear"}}}

kind 判定：
- greeting：纯寒暄；travel：明确的出行/规划需求；ambiguous：有旅行弱信号但不确定；out_of_scope：与旅行无关。

task_type 判定：
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
    prompt = f"已知出行画像：{'；'.join(brief_parts) or '空'}\n"
    if history:
        recent = "\n".join(f"{role}: {content}" for role, content in history[-4:])
        prompt += f"最近对话：\n{recent}\n"
    prompt += f"\n当前用户输入：{user_message}"

    model = _build_chat_model(settings)
    callbacks = None
    if evaluation_trace is not None:
        from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

        callbacks = [
            EvaluationTraceCallback(
                evaluation_trace,
                model=settings.llm.model,
                phase="turn_analysis",
            )
        ]
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
        task_type = TaskType(str(payload.get("task_type", "")).lower())
    except ValueError:
        pass
    llm_patches = _validate_slot_patches(payload.get("slots"))
    patches = llm_patches if llm_patches is not None else rule_result.patches
    if task_type == TaskType.ITINERARY_REVISION:
        for name in CORE_SLOTS:
            patch = patches.get(name)
            if patch is not None and patch.op == PatchOp.SET:
                patches.pop(name)
    return TurnAnalysis(
        kind=kind,
        task_type=task_type,
        patches=patches,
        source="llm",
        revision_directives=rule_result.revision_directives,
    )


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
