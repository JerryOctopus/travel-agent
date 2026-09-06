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
    FULL_ITINERARY = "full_itinerary"
    ROUTE_PLAN = "route_plan"
    CANDIDATE_COMPARISON = "candidate_comparison"
    ITINERARY_PATCH = "itinerary_patch"
    LOCAL_ADJUSTMENT_ADVICE = "local_adjustment_advice"
    CLARIFICATION = "clarification"
    CONSTRAINT_NEGOTIATION = "constraint_negotiation"
    SAFE_DECLINE = "safe_decline"

    # Source-compatible aliases for callers that still use the pre-routing-goals
    # vocabulary.  Their serialized value is deliberately the canonical goal.
    FULL_TRIP_PLAN = FULL_ITINERARY
    ROUTE_QUERY = ROUTE_PLAN
    POI_ADVICE = CANDIDATE_COMPARISON
    ITINERARY_REVISION = ITINERARY_PATCH
    LOCAL_ADJUSTMENT = LOCAL_ADJUSTMENT_ADVICE
    DAY_ADVICE = CANDIDATE_COMPARISON
    UNKNOWN = CLARIFICATION

    @classmethod
    def _missing_(cls, value: object) -> "TaskType | None":
        legacy = {
            "full_trip_plan": cls.FULL_ITINERARY,
            "route_query": cls.ROUTE_PLAN,
            "poi_advice": cls.CANDIDATE_COMPARISON,
            "itinerary_revision": cls.ITINERARY_PATCH,
            "local_adjustment": cls.LOCAL_ADJUSTMENT_ADVICE,
            "day_advice": cls.CANDIDATE_COMPARISON,
            "unknown": cls.CLARIFICATION,
        }
        return legacy.get(str(value).lower())


class DeliveryIntent(str, Enum):
    """How this turn should deliver its result, independently of task taxonomy."""

    STATE_UPDATE_ONLY = "state_update_only"
    REBUILD_NOW = "rebuild_now"
    LOCAL_PATCH = "local_patch"
    LIGHTWEIGHT_ADVICE = "lightweight_advice"


REQUIRED_SLOTS: dict[TaskType, tuple[str, ...]] = {
    TaskType.CLARIFICATION: (),
    TaskType.FULL_ITINERARY: ("destination", "days"),
    TaskType.CANDIDATE_COMPARISON: ("destination",),
    # Endpoints are carried in constraint_state rather than TravelProfile
    # attributes and are validated by the route worker. Duration is irrelevant.
    TaskType.ROUTE_PLAN: (),
    TaskType.ITINERARY_PATCH: (),
    TaskType.LOCAL_ADJUSTMENT_ADVICE: (),
    TaskType.CONSTRAINT_NEGOTIATION: (),
    TaskType.SAFE_DECLINE: (),
}


def required_slot_satisfied(profile: Any, task_type: TaskType, slot: str) -> bool:
    """Return whether a workflow slot is supplied by compact or structured state.

    Lightweight discovery requests often provide a self-contained geographic
    scope (for example “在某湖附近找餐厅”) without a separately extracted city.
    That scope is sufficient to start grounded search and must not be upgraded
    into a full-trip destination clarification.
    """
    if getattr(profile, slot, None):
        return True
    if slot != "destination" or task_type != TaskType.CANDIDATE_COMPARISON:
        return False
    state = getattr(profile, "constraint_state", {}) or {}
    return any(
        state.get(field) not in (None, "", [], {})
        for field in ("destination_city", "location", "location_anchor")
    )

# 修改类任务里规则易误抽的核心槽位（「第二天」会命中 days 等）
CORE_SLOTS = ("destination", "days", "start_date")

# 这些任务类型属于旅行需求，但规则意图分类容易因缺少强信号误判为 ambiguous/out_of_scope
_TRAVEL_ADVICE_TASKS = frozenset(
    {
        TaskType.ROUTE_PLAN,
        TaskType.CANDIDATE_COMPARISON,
        TaskType.ITINERARY_PATCH,
        TaskType.LOCAL_ADJUSTMENT_ADVICE,
    }
)

TURN_ANALYSIS_TIMEOUT_SECONDS = 120
TURN_ANALYSIS_MAX_OUTPUT_TOKENS = 512


@dataclass
class TurnAnalysis:
    kind: MessageKind
    task_type: TaskType
    delivery_intent: DeliveryIntent | None = None
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
    r"|(?:比较|对比).{0,18}(?:住宿区域|区域|酒店|餐厅|景点|地点)"
    r"|(?:候选).{0,80}(?:排序|比较|对比|推荐|选择|选一个|选最多)"
    r"|[^，。；;]{1,18}(?:和|与|还是|vs\.?)[^，。；;]{1,18}(?:哪个|哪一个|更方便|更合适|更好)"
)
_DAY_ADVICE_RE = re.compile(
    r"(?:今天|明天|后天|周末|周[一二三四五六日天]).{0,10}(?:下雨|天气|高温|降温).{0,12}(?:去哪|玩什么|安排|怎么办|适合)"
    r"|(?:下雨|天气不好|天气好).{0,8}(?:去哪|玩什么|适合)"
    r"|(?:只根据|根据).{0,18}天气.{0,18}(?:替换|备选|调整|判断)"
    r"|(?:今天|明天|后天|周末|周[一二三四五六日天])?.{0,4}天气.{0,8}(?:怎么样|如何|预报|情况|要注意什么)"
    r"|(?:天气|下雨|高温|大风).{0,16}(?:怎么调整|怎么办|是否要调整)"
    r"|(?:天气|气候|温度|冷不冷|热不热).{0,16}(?:穿什么|怎么穿|穿衣|衣服|带什么)"
    r"|(?:穿什么|怎么穿|穿衣建议|带什么衣服|(?:需要|要不要|是否要|要)带外套)"
    r"|(?:当地|那边|目的地).{0,8}(?:常见|通常|一般)?.{0,4}(?:天气|气候)"
)


def is_pure_weather_advice_request(
    user_message: str,
    state: dict[str, Any] | None = None,
) -> bool:
    """Return true when weather itself, not a plan operation, is requested."""
    if not _DAY_ADVICE_RE.search(user_message):
        return False
    if re.search(
        r"餐厅|酒店|住宿|路线|怎么走|交通方式|景点|活动|"
        r"替换|保留|取消|调整|备选|候选|排行|比较|对比",
        user_message,
    ):
        return False
    state = state or {}
    return not any(
        state.get(field) not in (None, "", [], {})
        for field in (
            "specific_restaurant_recommendation",
            "compare_lodging_areas",
            "conditional_activity",
            "referenced_day_index",
            "need_indoor_backup",
            "top_n",
        )
    )
_FULL_PLAN_CUE_RE = re.compile(
    r"(?:一|二|两|三|四|五|六|七|八|九|十|\d+)\s*(?:日|天)游"
    r"|(?:规划|安排|制定|生成).{0,12}(?:行程|旅行|旅游)"
    r"|(?:规划|安排|制定|生成).{0,12}"
    r"(?:一|二|两|三|四|五|六|七|八|九|十|\d+)\s*(?:日|天)"
    r"|(?:我)?想去(?:旅行|旅游)"
)
_SAFE_DECLINE_RE = re.compile(
    r"(?:替我|帮我).{0,8}(?:购票|付款|支付|预订|取消预订|联系商家|冒充)"
)
_CONSTRAINT_NEGOTIATION_RE = re.compile(
    r"(?:无法|不能|冲突|超出预算|来不及).{0,16}(?:改成|放弃|取舍|怎么办|可以吗)"
    r"|(?:必须|一定要).{0,12}(?:但|可是|同时).{0,12}(?:不能|不允许|预算)"
)
_EXPLICIT_LOCAL_ADJUSTMENT_RE = re.compile(
    r"(?:已有|已经有|现有|当前).{0,12}(?:行程|计划).{0,30}"
    r"(?:不要重写|不重写|只.{0,12}(?:替换|调整|修改)|局部调整)"
    r"|(?:不要重写|不重写).{0,18}(?:行程|计划)"
    r"|(?:原|原有)(?:行程|计划).{0,12}(?:不变|保留).{0,20}(?:只|仅).{0,12}(?:建议|备选|调整)"
)
_EXPLICIT_COMPARISON_RE = re.compile(
    r"比较|对比|候选.{0,12}(?:排序|比较|对比|推荐|选)"
    r"|[^，。；;]{1,18}(?:和|与|还是|vs\.?)[^，。；;]{1,18}(?:哪个|哪一个|更方便|更合适|更好)"
)
_TRANSPORT_MODE_COMPARISON_RE = re.compile(
    r"(?:比较|对比).{0,20}(?:公共交通|公交|地铁|打车|出租车|驾车|步行|高铁|飞机)"
    r".{0,12}(?:和|与|还是|vs\.?).{0,12}"
    r"(?:公共交通|公交|地铁|打车|出租车|驾车|步行|高铁|飞机)"
)
_LOCAL_ADJUSTMENT_ADVICE_RE = re.compile(
    r"(?:判断|看看|评估).{0,24}(?:是否|要不要|需不需要).{0,12}(?:替换|保留|取消|改成)"
    r"|(?:某天|当天|第\s*[一二三四五六七八九十\d]+\s*天|周[一二三四五六日天]).{0,20}"
    r"(?:户外|活动|景点).{0,16}(?:替换|保留|调整|备选)"
    r"|(?:天气|下雨|高温|大风).{0,20}(?:替换|保留|调整|室内备选)"
)
_FULL_REBUILD_REQUEST_RE = re.compile(
    r"(?:(?:最终|最后)(?:版|计划|方案|行程)|完整(?:版|行程)|更新完整行程|重新(?:生成|规划|安排)|重排(?:整份)?行程)"
    r"|(?:按|基于).{0,12}(?:全部|所有|当前).{0,8}(?:条件|约束).{0,12}(?:出|生成|更新|规划)"
    r"|再确认一次"
)
_IMMEDIATE_REBUILD_RE = re.compile(
    r"(?:现在|立即|马上|本轮).{0,10}(?:更新|调整|重建|重排|生成|规划).{0,8}(?:行程|方案|计划)"
    r"|(?:请|帮我).{0,4}(?:更新|调整|重建|重排|重新规划|改一下).{0,8}(?:现有|当前|整体|完整)?(?:行程|方案|计划)"
    r"|(?:更新|调整|重建|重排|重新规划).{0,8}(?:现有|当前|整体|完整)(?:行程|方案|计划)"
)
_DEFERRED_REBUILD_RE = re.compile(
    r"(?:先记住|记下来|先补充|暂时不用(?:出|生成|规划)|先别(?:出|生成|规划))"
    r"|(?:之后|稍后|等我说).{0,12}(?:再出|再规划|最终版|开始)"
)

# These constraints affect the plan as a whole.  They may trigger a full
# rebuild (or a deferred state update), but never an itinerary_patch artifact.
_GLOBAL_REBUILD_CONSTRAINT_FIELDS = frozenset({
    "date_start", "date_end", "duration_days", "destination", "destinations",
    "destination_city", "budget_max_cny", "budget_total_cny",
    "budget_remaining_cny", "budget_per_person_cny", "prepaid_cost",
    "prepaid_lodging_cny", "hotel_budget_per_night_cny", "lodging_area",
    "must_visit", "removed", "optional_remove", "avoid", "exclude", "dietary",
    "food_preference", "mobility", "wheelchair_user", "accessibility_priority",
    "walking_time_max_min", "max_single_walk_min", "max_walking_km_per_day",
    "max_transfers_per_day", "fixed_events", "return_deadline",
    "return_deadline_local_time", "return_deadline_day", "return_location",
    "return_day_index", "activity_end_deadline", "activity_end_target",
    "transport_mode",
    "transport_modes", "self_driving_allowed", "public_transport_required",
    "pace", "traveler_count",
})
_GLOBAL_REBUILD_PATCH_FIELDS = frozenset({
    "destination", "days", "start_date", "budget_limit", "budget_level",
    "hotel_area", "must_visit", "avoid", "food_preference", "transport_mode",
    "pace", "party_size",
})

_ADDITIVE_CONSTRAINT_FIELDS = frozenset({
    "must_visit", "candidate_attractions", "fixed_events", "interests",
    "dietary", "transport_modes", "avoid", "exclude", "optional_remove",
})
_STATE_PROFILE_ALIASES: dict[str, str] = {
    "date_start": "start_date",
    "duration_days": "days",
    "destination": "destination",
    "destinations": "destination",
    "destination_city": "destination",
    "budget_max_cny": "budget_limit",
    "budget_total_cny": "budget_limit",
    "lodging_area": "hotel_area",
    "food_preference": "food_preference",
    "transport_mode": "transport_mode",
    "pace": "pace",
    "traveler_count": "party_size",
}
_BUDGET_CONSTRAINT_FIELDS = frozenset({
    "budget_max_cny", "budget_total_cny", "budget_remaining_cny",
    "budget_per_person_cny", "prepaid_cost", "prepaid_lodging_cny",
    "hotel_budget_per_night_cny",
})


def _has_plan_context(ctx: "SessionContext | None") -> bool:
    if ctx is None:
        return False
    return bool(ctx.store.latest_revisable_id("itinerary"))


def _has_current_plan(ctx: "SessionContext | None") -> bool:
    return bool(ctx is not None and ctx.store.latest_current_id("itinerary"))


def _comparable_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return _comparable_value(value[0])
    if isinstance(value, tuple):
        return [_comparable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _comparable_value(item) for key, item in sorted(value.items())}
    return value


def _existing_constraint_value(
    ctx: "SessionContext",
    field: str,
) -> tuple[bool, Any]:
    """Resolve a field against active state and its compact-profile mirror."""
    active = ctx.profile.constraint_state or {}
    if field in active and active.get(field) not in (None, "", [], {}):
        return True, active[field]
    if field in _BUDGET_CONSTRAINT_FIELDS:
        for candidate in _BUDGET_CONSTRAINT_FIELDS:
            if active.get(candidate) not in (None, "", [], {}):
                return True, active[candidate]
        if ctx.profile.budget_limit is not None:
            return True, ctx.profile.budget_limit
    profile_field = _STATE_PROFILE_ALIASES.get(field)
    if profile_field:
        value = getattr(ctx.profile, profile_field, None)
        if value not in (None, "", [], {}):
            return True, value
    return False, None


def _has_structured_global_replacement(
    ctx: "SessionContext | None",
    patches: dict[str, SlotPatch],
    state: dict[str, Any],
) -> bool:
    """Detect replace/remove from structured deltas, independent of wording.

    Additive declarations remain deferred.  Scalar changes, explicit clears,
    and remove tombstones against a current itinerary require an immediate
    rebuild so the deliverable cannot remain validated against an old value.
    """
    if ctx is None or not _has_current_plan(ctx):
        return False
    for field, value in state.items():
        if field not in _GLOBAL_REBUILD_CONSTRAINT_FIELDS:
            continue
        if field == "removed":
            if value not in (None, "", [], {}):
                return True
            continue
        if field in _ADDITIVE_CONSTRAINT_FIELDS:
            continue
        exists, previous = _existing_constraint_value(ctx, field)
        if exists and _comparable_value(previous) != _comparable_value(value):
            return True
    for field, patch in patches.items():
        if field not in _GLOBAL_REBUILD_PATCH_FIELDS:
            continue
        profile_value = getattr(ctx.profile, field, None)
        if patch.op == PatchOp.CLEAR:
            if profile_value not in (None, "", [], {}):
                return True
            continue
        if field in {"must_visit", "avoid", "food_preference"}:
            continue
        if (
            profile_value not in (None, "", [], {})
            and _comparable_value(profile_value) != _comparable_value(patch.value)
        ):
            return True
    return False


def _has_local_patch_target(text: str, state: dict[str, Any]) -> bool:
    if state.get("referenced_day_index") or state.get("conditional_activity"):
        return True
    return bool(re.search(
        r"第\s*[一二三四五六七八九十\d]+\s*天|最后一天|"
        r"\d{4}年\d{1,2}月\d{1,2}日|"
        r"(?:当天|当日|上午|下午|晚上|早上|午餐|晚餐|\d{1,2}:\d{2}).{0,18}"
        r"(?:站点|景点|活动|安排|时段|替换|调整|修改)",
        text,
    ))


def classify_delivery_intent(
    user_message: str,
    ctx: "SessionContext | None",
    task_type: TaskType,
    patches: dict[str, SlotPatch] | None = None,
    state: dict[str, Any] | None = None,
) -> DeliveryIntent:
    """Classify delivery scope without changing the semantic task type.

    Declarative, plan-wide constraints accumulate as state while a stale parent
    remains available.  Planner admission is reserved for an explicit rebuild
    request (or the initial full-plan request), while narrow schedule edits and
    advice keep their own delivery contracts.
    """
    text = user_message.strip()
    state = state or {}
    patches = patches or {}
    has_plan = _has_plan_context(ctx)
    rebuild_pending = bool(
        ctx is not None
        and (ctx.profile.constraint_state or {}).get("_plan_status")
        == "rebuild_pending"
    )
    deferred = bool(_DEFERRED_REBUILD_RE.search(text))
    explicit_rebuild = bool(
        _IMMEDIATE_REBUILD_RE.search(text)
        or (_FULL_REBUILD_REQUEST_RE.search(text) and not deferred)
    )

    # A direct final-delivery request is the only operation that may interrupt
    # a pending rebuild, and it also beats lightweight task taxonomy.
    if explicit_rebuild and (has_plan or rebuild_pending):
        return DeliveryIntent.REBUILD_NOW
    if task_type in {TaskType.ROUTE_PLAN, TaskType.CANDIDATE_COMPARISON}:
        return DeliveryIntent.LIGHTWEIGHT_ADVICE
    if task_type == TaskType.LOCAL_ADJUSTMENT_ADVICE:
        return DeliveryIntent.LIGHTWEIGHT_ADVICE
    if rebuild_pending:
        return DeliveryIntent.STATE_UPDATE_ONLY
    if task_type == TaskType.ITINERARY_PATCH and _has_local_patch_target(text, state):
        return DeliveryIntent.LOCAL_PATCH
    if task_type == TaskType.FULL_ITINERARY:
        if (
            not has_plan
            or (
                _has_structured_global_replacement(ctx, patches, state)
                and not deferred
            )
            or (_has_local_patch_target(text, state) and not deferred)
        ):
            return DeliveryIntent.REBUILD_NOW
        if state or patches:
            return DeliveryIntent.STATE_UPDATE_ONLY
        return DeliveryIntent.REBUILD_NOW
    if has_plan and (state or patches):
        return (
            DeliveryIntent.REBUILD_NOW
            if _has_structured_global_replacement(ctx, patches, state) and not deferred
            else DeliveryIntent.STATE_UPDATE_ONLY
        )
    return DeliveryIntent.LIGHTWEIGHT_ADVICE


def _with_delivery_intent(
    analysis: TurnAnalysis,
    user_message: str,
    ctx: "SessionContext | None",
) -> TurnAnalysis:
    if analysis.task_type != TaskType.CANDIDATE_COMPARISON:
        # Words such as “交通方便” and “费用控制” are ordinary full-trip
        # constraints.  They must not manufacture a comparison contract when
        # the selected delivery is not a comparison.  A full itinerary may
        # still retain ``candidate_attractions`` as an allowed shortlist.
        for field in ("comparison_candidates", "comparison_dimensions"):
            analysis.constraint_state.pop(field, None)
    intent = classify_delivery_intent(
        user_message,
        ctx,
        analysis.task_type,
        analysis.patches,
        analysis.constraint_state,
    )
    if (
        intent == DeliveryIntent.REBUILD_NOW
        and analysis.task_type == TaskType.ITINERARY_PATCH
        and not _has_local_patch_target(user_message, analysis.constraint_state)
    ):
        analysis.task_type = TaskType.FULL_ITINERARY
    analysis.delivery_intent = intent
    return analysis


def _enforce_itinerary_scope(
    user_message: str,
    ctx: "SessionContext | None",
    proposed: TaskType,
    patches: dict[str, SlotPatch],
    state: dict[str, Any],
) -> TaskType:
    """Keep patch delivery local and route plan-wide changes to rebuild."""
    if is_pure_weather_advice_request(user_message, state):
        return proposed
    has_context = _has_plan_context(ctx)
    current_id = ctx.store.latest_current_id("itinerary") if ctx is not None else None
    explicit_full = bool(_FULL_REBUILD_REQUEST_RE.search(user_message))
    global_change = bool(
        _GLOBAL_REBUILD_CONSTRAINT_FIELDS.intersection(state)
        or _GLOBAL_REBUILD_PATCH_FIELDS.intersection(patches)
    )
    locally_scoped_global_fields = {
        "must_visit", "removed", "optional_remove", "avoid", "exclude",
    }
    locally_scoped_patch_fields = {"must_visit", "avoid"}
    planwide_global_change = bool(
        (_GLOBAL_REBUILD_CONSTRAINT_FIELDS - locally_scoped_global_fields).intersection(state)
        or (_GLOBAL_REBUILD_PATCH_FIELDS - locally_scoped_patch_fields).intersection(patches)
    )
    if has_context and explicit_full:
        return TaskType.FULL_ITINERARY
    if has_context and planwide_global_change:
        return TaskType.FULL_ITINERARY
    if proposed in {TaskType.ITINERARY_PATCH, TaskType.LOCAL_ADJUSTMENT_ADVICE}:
        # A stale/historical plan is revision context for a rebuild only; it
        # can never be the source of a newly deliverable local patch.
        if has_context and not current_id:
            return TaskType.FULL_ITINERARY
        if current_id and _has_local_patch_target(user_message, state):
            return TaskType.ITINERARY_PATCH
        if current_id and not _has_local_patch_target(user_message, state):
            return TaskType.FULL_ITINERARY
    if has_context and global_change:
        return TaskType.FULL_ITINERARY
    return proposed


def is_plan_revision_followup(user_message: str, ctx: "SessionContext") -> bool:
    """判断是否属于对既有行程的修改/对比场景。"""
    if not ctx.store.latest_revisable_id("itinerary"):
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
    if _SAFE_DECLINE_RE.search(text):
        return TaskType.SAFE_DECLINE
    if _CONSTRAINT_NEGOTIATION_RE.search(text):
        return TaskType.CONSTRAINT_NEGOTIATION
    if _EXPLICIT_LOCAL_ADJUSTMENT_RE.search(text) or _LOCAL_ADJUSTMENT_ADVICE_RE.search(text):
        if ctx is not None and ctx.store.latest_current_id("itinerary"):
            return TaskType.ITINERARY_PATCH
        return TaskType.LOCAL_ADJUSTMENT_ADVICE
    # Explicit scope words beat the generic verb “规划”. This is crucial for
    # requests such as “只需要规划 9 月 5 日从机场到酒店区的路线”.
    if re.search(r"(?:只需要|只要|仅|只(?:需)?规划).{0,40}(?:路线|怎么走|公共交通|打车)", text):
        return TaskType.ROUTE_PLAN
    if _FULL_PLAN_CUE_RE.search(text):
        return TaskType.FULL_TRIP_PLAN
    # Comparing multiple destination candidates by their travel time is a
    # candidate decision, while comparing transport modes for one A->B leg is
    # still a route plan.  Check this distinction before the broad route regex,
    # which intentionally matches both forms because both mention transport.
    if (
        _EXPLICIT_COMPARISON_RE.search(text)
        and not _TRANSPORT_MODE_COMPARISON_RE.search(text)
    ):
        return TaskType.CANDIDATE_COMPARISON
    # A complete trip request often contains an origin/destination leg and
    # words such as public transport or walking.  Those embedded logistics
    # must not downgrade an explicit multi-hour/day itinerary to route_query.
    if _ROUTE_QUERY_RE.search(text) or (
        _ROUTE_FROM_TO_RE.search(text) and _ROUTE_CONTEXT_RE.search(text)
    ):
        return TaskType.ROUTE_QUERY
    if ctx is not None and is_plan_revision_followup(text, ctx):
        return TaskType.ITINERARY_PATCH
    if _POI_ADVICE_RE.search(text):
        return TaskType.POI_ADVICE
    if _DAY_ADVICE_RE.search(text):
        return TaskType.CANDIDATE_COMPARISON
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
    # A cancelled venue is never a destination update. This must hold even
    # before an itinerary exists; otherwise a long-horizon dialogue can turn
    # “不去迪士尼了” into the bogus city destination “迪士”.
    replacement = re.search(r"(?:换成|改成|改为|改去|换去)([^，,。；;]+)", user_message)
    replacement_is_city = bool(
        replacement
        and any(alias in replacement.group(1) for alias in CITY_ALIASES)
    )
    if re.search(r"不(?:想|打算|要)?去|仍然不去|不去了", user_message) and not (
        replacement_is_city
        or re.search(
            r"(?:目的地|城市).{0,8}(?:改|换)|(?:改去|换去|改成去|改为去)",
            user_message,
        )
    ):
        patches.pop("destination", None)
    if _DAY_ADVICE_RE.search(user_message) and not _FULL_PLAN_CUE_RE.search(user_message):
        # Relative dates in a weather question identify the advice horizon;
        # they do not silently change the trip's start date or duration.
        patches.pop("start_date", None)
        patches.pop("days", None)
    if task_type == TaskType.LOCAL_ADJUSTMENT_ADVICE:
        # A fragment supplied from a user-owned itinerary is context for a
        # local recommendation, not a request to create durable trip slots.
        # Broad profile extraction can otherwise mistake phrases such as
        # “只判断第二天…” for a destination and a trip duration.
        for slot in ("destination", "start_date", "days"):
            patches.pop(slot, None)
    _remove_destination_city_from_patch_must_visits(patches)
    if _venues_require_verification(user_message):
        # “先核验这些地点是否适合安排” describes candidates whose feasibility
        # must be checked, not unconditional must-visits.  Keeping them hard can
        # make a truthful closure/weather result impossible to deliver.
        patches.pop("must_visit", None)
    date_range = _extract_iso_date_range(user_message)
    if "days" not in patches and date_range:
        start, end = (date.fromisoformat(value) for value in date_range)
        patches["days"] = SlotPatch(PatchOp.SET, (end - start).days + 1)
    # Ordinal day references describe a schedule slot, not trip duration.
    # Only explicit duration/range language is allowed to set ``days``.
    ordinal_day = re.search(r"第\s*[一二三四五六七八九十\d]+\s*天|最后一天", user_message)
    duration_cue = re.search(
        r"(?:共|总共|一共|行程|旅行|旅游|玩|去[^，。]{0,12})\s*"
        r"(?:一|二|两|三|四|五|六|七|八|九|十|\d+)\s*天"
        r"|(?:一|二|两|三|四|五|六|七|八|九|十|\d+)\s*(?:日|天)游",
        user_message,
    )
    if ordinal_day and not duration_cue and not date_range:
        patches.pop("days", None)
    if task_type == TaskType.ITINERARY_REVISION:
        explicit_core_change = {
            "destination": bool(re.search(
                r"(?:目的地|城市).{0,8}(?:改|换)|(?:改去|换去|改成去|改为去)",
                user_message,
            )),
            "days": bool(duration_cue),
            "start_date": bool(re.search(
                r"(?:日期|出发时间|出发日期).{0,8}(?:改|换|提前|推迟)|"
                r"(?:改到|改为|提前到|推迟到).{0,8}(?:月|日|号|周)",
                user_message,
            )),
        }
        for name in CORE_SLOTS:
            patch = patches.get(name)
            if (
                patch is not None
                and patch.op == PatchOp.SET
                and not explicit_core_change.get(name, False)
            ):
                patches.pop(name)
    return patches


def build_rule_constraint_state(
    user_message: str,
    patches: dict[str, SlotPatch],
    reference_datetime: str | None = None,
) -> dict[str, Any]:
    """Preserve common literal constraints even when the LLM analyzer is unavailable."""
    state: dict[str, Any] = {}
    if _DAY_ADVICE_RE.search(user_message):
        state["weather_condition"] = "query"
        state["advice_topic"] = (
            "clothing"
            if re.search(r"穿|衣服|外套", user_message)
            else "climate"
            if re.search(r"气候|通常|一般|常见", user_message)
            else "weather"
        )
        if re.search(r"下雨|雨天|降雨", user_message):
            state["need_indoor_backup"] = True
    referenced_day = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*天", user_message)
    if referenced_day:
        parsed_day = _parse_chinese_day(referenced_day.group(1))
        if parsed_day:
            state["referenced_day_index"] = parsed_day
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
        activity_deadline = re.search(
            r"(?:晚上)?(\d{1,2}:\d{2})\s*(前)?\s*结束", user_message
        )
        if return_deadline:
            state["return_deadline"] = return_deadline.group(1)
        elif activity_deadline:
            state["activity_end_deadline"] = activity_deadline.group(1)
            if activity_deadline.group(2) is None:
                state["activity_end_target"] = activity_deadline.group(1)
    route = re.search(r"从(.{1,36}?)(?:到|去|前往)(.{1,36}?)(?:的|，|,|。|$)", user_message)
    if route:
        origin = re.sub(r"^.*?(?:日|号|上午|下午|晚上)", "", route.group(1)).strip()
        destination = re.sub(r"(?:的)?公共交通路线.*$|，.*$", "", route.group(2)).strip(" \"'“”")
        if origin:
            state["origin"] = origin
        if destination:
            destination = re.sub(
                r"(?:[一二两三四五六七八九十\d]+(?:日|天)游|(?:一日|两日|多日)游)$",
                "",
                destination,
            ).strip()
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
    if re.search(
        r"(?:推荐|找|选择?|安排|列出|给出|具体|哪家).{0,12}(?:餐厅|饭店|馆子|午餐|晚餐|用餐)"
        r"|(?:餐厅|饭店|馆子).{0,12}(?:推荐|哪家|具体|名单)",
        user_message,
    ):
        state["specific_restaurant_recommendation"] = True
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
            # Keep the canonical fixed-event object backward compatible with
            # frozen state contracts.  This companion field records that the
            # user supplied an area-level reservation block, not a verified
            # restaurant entity that the system may fabricate.
            state["user_owned_unspecified_fixed_event_locations"] = [
                meal_event.group(3).strip()
            ]

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
    from travel_agent.constraint_events import normalize_return_deadline

    normalize_return_deadline(
        state,
        start_date=str(state.get("date_start") or "") or None,
        duration_days=state.get("duration_days"),
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

    # Keep the enclosing city separate from the local anchor.  Downstream
    # search uses destination_city; route and proximity checks use the anchor.
    city_match = next(
        (
            (alias, city)
            for alias, city in sorted(CITY_ALIASES.items(), key=lambda item: -len(item[0]))
            if alias in text
        ),
        None,
    )
    if city_match:
        city_alias, destination_city = city_match
        state["destination_city"] = destination_city
        anchor_match = re.search(
            rf"(?:在|位于|靠近|住在|从)?{re.escape(city_alias)}(?:市|城区|城里)?"
            rf"(?:的|内|中|里)?([^，,。；;]{{2,20}}?)(?:附近|周边|一带)",
            text,
        )
        if anchor_match:
            anchor = re.sub(r"^(?:的|内|中|里)", "", anchor_match.group(1)).strip()
            anchor = re.sub(r"(?:区域|商圈)$", "", anchor).strip() or anchor
            if anchor and anchor not in {city_alias, destination_city}:
                state["location_anchor"] = anchor
                state["location"] = f"{destination_city}{anchor}附近"

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
        r"预算.{0,18}?(?:改成|改为|调整为|降到)\s*(\d+(?:\.\d+)?)\s*元",
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
        r"(?:今晚在|地点在|(?:^|[，,])在|从)([^，,。；;]{2,28}?(?:附近|大道|商圈))",
        text,
    )
    if location:
        value = location.group(1).strip()
        city = next((city for alias, city in CITY_ALIASES.items() if alias in text), None)
        if city and city not in value and ("区" in value or "大道" in value):
            value = city + value
        if not state.get("location_anchor"):
            state["location"] = value
    lodging = re.search(
        r"(?:住宿(?:放在|安排在|住在|在)?|住(?!宿))([^，,。；;]{2,14}?)(?:，|。|$)",
        text,
    )
    lodging_question = bool(
        _EXPLICIT_COMPARISON_RE.search(text)
        or re.search(r"住(?:在)?哪(?:里|儿|边|个)?", text)
    )
    if lodging and not lodging_question:
        value = lodging.group(1).strip()
        if not re.search(
            r"酒店|景点|库存|降档|花了|已花|支付|预付|\d+\s*元|"
            r"(?:区域)?不要改|(?:区域)?不变",
            value,
        ):
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

    compare_areas = re.search(
        r"(?:只)?(?:比较|对比)([^。；;]+?)(?:[两二三四五六七八九十\d]+个)?(?:住宿)?区域",
        text,
    )
    if compare_areas:
        values = _split_named_items(compare_areas.group(1))
        if len(values) >= 2:
            state["compare_lodging_areas"] = values
            state["comparison_candidates"] = values
    if re.search(r"不需要.{0,8}(?:酒店)?库存|不要.{0,8}(?:酒店)?库存", text):
        state["no_live_inventory_required"] = True

    candidate_match = re.search(
        r"(?:候选是|想去|想看|(?<!不)安排)([^。；;]+?)(?:。|，请|，帮|，不能|，晚上|，单段|，全程|$)",
        text,
    )
    if candidate_match:
        candidates = _split_named_items(candidate_match.group(1))
        candidates = [item for item in candidates if _looks_like_named_candidate(item)]
        if len(candidates) >= 2:
            state["candidate_attractions"] = candidates
            state["comparison_candidates"] = candidates
            if _venues_require_verification(text):
                state["candidate_only"] = True
    compare_candidates = re.search(r"比较去([^。；;]+?)三个地点", text)
    if compare_candidates:
        candidates = _split_named_items(compare_candidates.group(1))
        if len(candidates) >= 2:
            state["candidate_attractions"] = candidates
            state["comparison_candidates"] = candidates
    anchored_comparison = re.search(
        r"(?:比较|对比)([^。；;]+?)到[^，,。；;]{2,24}?(?:通勤|交通|路线|耗时|时间|距离|可达性)",
        text,
    )
    if anchored_comparison and not state.get("comparison_candidates"):
        values = _split_named_items(anchored_comparison.group(1))
        if len(values) >= 2 and all(_looks_like_named_candidate(item) for item in values):
            state["comparison_candidates"] = values
    binary_comparison = re.search(
        r"(?:比较|对比)?([^，。；;]{1,18}?)\s*(?:和|与|还是|vs\.?)\s*"
        r"([^，。；;]{1,18}?)[，,]?\s*(?:"
        r"住(?:在)?(?:哪(?:里|儿|边|个)?)?(?:更)?(?:方便|合适|好)"
        r"|哪个|哪一个|谁|更方便|更合适|更好|的差别|的区别)",
        text,
        flags=re.IGNORECASE,
    )
    if binary_comparison and not state.get("comparison_candidates"):
        values = [_clean_comparison_candidate(item) for item in binary_comparison.groups()]
        if all(_looks_like_named_candidate(item) for item in values):
            state["comparison_candidates"] = values

    target_anchor = re.search(
        r"(?:到|前往|去)([^，,。；;]{2,24}?)(?:的)?"
        r"(?:通勤|交通|路线|耗时|时间|距离|可达性)",
        text,
    )
    if target_anchor:
        anchor_value = target_anchor.group(1).strip()
        if not re.search(
            r"\d{1,2}:\d{2}|不要|避免|安排|活动|高温|时段",
            anchor_value,
        ):
            state["target_anchor"] = anchor_value
    # “从 X 出发，比较去 A/B/C” is an anchored candidate comparison from X.
    # The broad target regex above otherwise consumes the whole candidate list
    # (and even the adjective before “交通”), which makes every route endpoint
    # fail the evidence contract.  The origin is the shared route anchor.
    shared_origin = re.search(
        r"从([^，,。；;]{1,30}?)(?:出发|启程)[，,]?(?:比较|对比)去", text
    )
    if state.get("comparison_candidates") and shared_origin:
        state["target_anchor"] = shared_origin.group(1).strip()
        candidates = [str(item) for item in state.get("comparison_candidates") or []]
        for field in ("destination", "destination_name", "destination_area"):
            value = str(state.get(field) or "")
            if sum(candidate in value for candidate in candidates) >= 2:
                state.pop(field, None)
    dimensions: list[str] = []
    if re.search(r"交通|路线|通勤|耗时|距离|可达|方便|步行", text):
        dimensions.append("accessibility")
    if re.search(r"价格|预算|成本|便宜|贵|人均|元(?:以内|左右)", text):
        dimensions.append("cost")
    if re.search(r"安静|热闹|氛围|环境", text):
        dimensions.append("ambience")
    if re.search(r"适老|老人|父母|无障碍|少走", text):
        dimensions.append("accessibility_needs")
    if dimensions:
        state["comparison_dimensions"] = list(dict.fromkeys(dimensions))
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
    if re.search(
        r"(?:有雨|下雨)(?:的话|时|是否|就)|是否.{0,8}(?:有雨|下雨)|因(?:为)?下雨",
        text,
    ):
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
        year_match = re.match(r"(\d{4})-\d{2}-\d{2}$", str(start_date or ""))
        year = (
            int(year_match.group(1))
            if year_match
            else (reference_date or date.today()).year
        )
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
    if state.get("comparison_candidates") and state.get("origin") and not state.get("target_anchor"):
        # Candidate ranking from a shared departure point is an anchored
        # accessibility comparison even when “比较去” is not adjacent to the
        # origin phrase (for example “根据从 X 出发、18:00 结束”).
        state["target_anchor"] = state["origin"]
        dimensions = list(state.get("comparison_dimensions") or [])
        if "accessibility" not in dimensions:
            state["comparison_dimensions"] = ["accessibility", *dimensions]
    if state.get("return_deadline") and re.search(r"回到([^，,。；;]+)", text):
        state["return_location"] = re.search(r"回到([^，,。；;]+)", text).group(1).strip()
    final_return = re.search(
        r"最后一天\s*(\d{1,2})(?::(\d{2}))?点?前\s*(?:到达?|抵达|回到)([^，,。；;]+)",
        text,
    )
    if final_return:
        state["return_deadline"] = (
            f"{int(final_return.group(1)):02d}:{int(final_return.group(2) or 0):02d}"
        )
        state["return_location"] = final_return.group(3).strip()
        state["return_day_index"] = state.get("duration_days") or "last"

    named_venues = [] if _venues_require_verification(text) else _extract_named_venues(text)
    fixed_locations = {
        str(event.get("location") or "").strip()
        for event in (state.get("fixed_events") or [])
        if isinstance(event, dict)
    }
    named_venues = [
        venue
        for venue in named_venues
        if venue not in fixed_locations and venue not in {"自由活动", "休息", "自由时间"}
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

    _remove_destination_cities_from_state_must_visits(state)


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


def _clean_comparison_candidate(value: str) -> str:
    """Remove request/action prefixes while preserving the named entity itself."""
    cleaned = value.strip(" \"'“”，,")
    prefix = re.compile(
        r"^(?:帮我|请|麻烦(?:帮我)?|能否|可以(?:帮我)?|"
        r"比较(?:一下|下)?|对比(?:一下|下)?|住在|住|选)\s*"
    )
    while True:
        updated = prefix.sub("", cleaned, count=1).strip(" \"'“”，,")
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _canonical_destination_city(value: Any) -> str | None:
    """Return a canonical supported city only for an exact city/alias value."""
    text = str(value or "").strip()
    if not text:
        return None
    without_suffix = text[:-1] if text.endswith("市") else text
    for alias, city in CITY_ALIASES.items():
        if without_suffix.casefold() == alias.casefold() or without_suffix == city:
            return city
    return None


def _remove_destination_city_from_patch_must_visits(
    patches: dict[str, SlotPatch],
) -> None:
    destination = patches.get("destination")
    required = patches.get("must_visit")
    if (
        destination is None
        or destination.op != PatchOp.SET
        or required is None
        or required.op != PatchOp.SET
    ):
        return
    destination_city = _canonical_destination_city(destination.value)
    if destination_city is None:
        return
    values = [
        item
        for item in list(required.value or [])
        if _canonical_destination_city(item) != destination_city
    ]
    if values:
        patches["must_visit"] = SlotPatch(PatchOp.SET, values)
    else:
        patches.pop("must_visit", None)


def _remove_destination_cities_from_state_must_visits(state: dict[str, Any]) -> None:
    required = state.get("must_visit")
    if not isinstance(required, list):
        return
    destinations = [
        *list(state.get("destinations") or []),
        state.get("destination"),
        state.get("destination_city"),
    ]
    destination_cities = {
        city for value in destinations if (city := _canonical_destination_city(value))
    }
    if not destination_cities:
        return
    values = [
        item
        for item in required
        if _canonical_destination_city(item) not in destination_cities
    ]
    if values:
        state["must_visit"] = values
    else:
        state.pop("must_visit", None)


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
        and not re.search(r"(?:完整|最终|全部)?(?:行程|方案|计划)$", value)
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
        r"(?:完整|最终|全部)?(?:行程|方案|计划)|夜间项目|餐饮不要|"
        r"^(?:但|不过|而且)?(?:不要|不能|避免)|"
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
        or re.search(
            r"(?:判断|看看|确认).{0,8}(?:能否|是否|可不可以).{0,6}(?:都|全部)?(?:去|安排)"
            r".{0,16}(?:不能|不行|不可行).{0,8}(?:删减|删除|取舍|少去)",
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
    if (
        task_type == TaskType.CANDIDATE_COMPARISON
        and rule_result.constraint_state.get("specific_restaurant_recommendation")
    ):
        # Meal-request phrases such as “安排公司附近晚餐” describe the
        # requested output, not a named must-visit venue.  Keeping them in
        # must_visit would later force an impossible POI identity check.
        meal_markers = re.compile(r"(?:早餐|午餐|晚餐|用餐|聚餐|餐厅|饭店|馆子)")
        rule_result.constraint_state["must_visit"] = [
            item
            for item in rule_result.constraint_state.get("must_visit") or []
            if not meal_markers.search(str(item))
        ]
        if not rule_result.constraint_state["must_visit"]:
            rule_result.constraint_state.pop("must_visit", None)
        patch = rule_result.patches.get("must_visit")
        if patch is not None and patch.op == PatchOp.SET:
            values = [
                item for item in list(patch.value or [])
                if not meal_markers.search(str(item))
            ]
            if values:
                rule_result.patches["must_visit"] = SlotPatch(PatchOp.SET, values)
            else:
                rule_result.patches.pop("must_visit", None)
    if (
        rule_result.constraint_state.get("fixed_events")
        and _extract_iso_date_range(user_message) is None
        and not re.search(r"(?:出发|启程|行程开始|旅行开始)", user_message)
    ):
        # A dated appointment is an event date, not a replacement trip start.
        rule_result.patches.pop("start_date", None)
        rule_result.constraint_state.pop("date_start", None)
    if (
        rule_result.constraint_state.get("lodging_area")
        and task_type != TaskType.CANDIDATE_COMPARISON
        and not re.search(
            r"(?:必去|必须去|务必去|必须保留|想去|想看|景点|参观|游览)",
            user_message,
        )
    ):
        # Lodging declarations occasionally look like named-venue lists to the
        # broad profile extractor.  The structured lodging field is dominant.
        rule_result.patches.pop("must_visit", None)
        for field in (
            "must_visit", "candidate_attractions", "comparison_candidates",
        ):
            rule_result.constraint_state.pop(field, None)
    if (
        ctx.store.latest_revisable_id("itinerary")
        and _FULL_REBUILD_REQUEST_RE.search(user_message)
        and not re.search(
            r"(?:目的地|城市).{0,8}(?:改|换)|(?:改去|换去|改成去|改为去)",
            user_message,
        )
        and not any(alias in user_message for alias in CITY_ALIASES)
    ):
        # Final-delivery wording must not be misread as a new destination.
        rule_result.patches.pop("destination", None)
        for field in ("destination", "destinations", "destination_city"):
            rule_result.constraint_state.pop(field, None)
    scoped_task_type = _enforce_itinerary_scope(
        user_message,
        ctx,
        rule_result.task_type,
        rule_result.patches,
        rule_result.constraint_state,
    )
    if scoped_task_type != rule_result.task_type:
        rule_result = TurnAnalysis(
            kind=MessageKind.TRAVEL,
            task_type=scoped_task_type,
            patches=rule_result.patches,
            source=rule_result.source,
            # Local targeting directives remain useful rebuild inputs when a
            # plan-wide constraint in the same turn widens the task to full.
            revision_directives=rule_result.revision_directives,
            constraint_state=rule_result.constraint_state,
        )
        task_type = scoped_task_type
        kind = MessageKind.TRAVEL
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
        return _with_delivery_intent(rule_result, user_message, ctx)
    if not settings.llm.enabled:
        return _with_delivery_intent(rule_result, user_message, ctx)
    if kind != MessageKind.AMBIGUOUS and not has_complex_signals(user_message):
        return _with_delivery_intent(rule_result, user_message, ctx)
    try:
        payload = _analyze_with_llm(user_message, ctx, settings, history or [], evaluation_trace)
    except Exception:  # noqa: BLE001 - LLM 失败必须回退规则，不能阻塞主流程
        return _with_delivery_intent(rule_result, user_message, ctx)
    return _with_delivery_intent(_merge_llm_payload(payload, rule_result), user_message, ctx)


_TURN_ANALYSIS_SYSTEM = """你是旅行规划 Agent 的轮次分析模块，对用户输入一次性输出四件事：意图类别、任务类型、出行画像字段更新、完整的结构化约束状态。
只输出 JSON 对象，不要输出 Markdown。格式：
{"kind": "greeting|travel|ambiguous|out_of_scope",
 "task_type": "full_itinerary|route_plan|candidate_comparison|itinerary_patch|local_adjustment_advice|clarification|constraint_negotiation|safe_decline",
 "slots": {"字段名": {"op": "set", "value": ...} 或 {"op": "clear"}},
 "constraint_state": {"约束字段": JSON值}}

kind 判定：
- greeting：纯寒暄；travel：明确的出行/规划需求；ambiguous：有旅行弱信号但不确定；out_of_scope：与旅行无关。

task_type 判定：
- full_itinerary：要求规划/生成完整行程；
- route_plan：问 A 到 B 怎么走、交通方式、耗时，不要求完整行程；
- candidate_comparison：比较地点、住宿区域、酒店或餐厅候选；
- itinerary_patch：系统当前确实持有结构化行程，对其中局部时段做 patch；
- local_adjustment_advice：没有系统持有的完整行程，但局部对象、日期或条件足够时给保留/替换建议；
- clarification：任务目标本身无法安全识别；
- constraint_negotiation：约束冲突，需要用户取舍；
- safe_decline：请求代执行预订、支付等超出只读建议边界的操作。

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
- 仅记录明确表达的约束，不推测。城市与局部锚点必须分别写 destination_city 与 location_anchor。优先使用这些通用字段：date_start,date_end,destinations,destination,destination_city,location_anchor,target_anchor,comparison_candidates,comparison_dimensions,origin,return_location,destination_area,destination_name,duration_days,traveler_count,child_age,elderly,wheelchair_user,travel_with_parents,budget_max_cny,budget_per_person_cny,budget_total_cny,budget_remaining_cny,prepaid_lodging_cny,hotel_budget_per_night_cny,lodging_area,lodging_flexibility,interests,must_visit,removed,optional_remove,avoid,dietary,self_driving_allowed,cycling_allowed,transport_mode,transport_modes,public_transport_required,taxi_backup,fallback_transport,return_deadline,activity_end_deadline,activity_end_target,walking_time_max_min,max_single_walk_min,max_walking_km_per_day,max_transfers_per_day,max_major_activities_per_day,max_selected,candidate_attractions,candidate_only,fixed_events,conditional_activity,conditional_avoid_window,need_indoor_backup,night_activity_allowed,accessibility_priority,mobility,occasion,ambience,parking_preferred,top_n,diversity_required,exclude,no_live_inventory_required,need_disambiguation,trip_length,weather_condition,weekday,preference。
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
            if (
                patch is not None
                and patch.op == PatchOp.SET
                and name not in rule_result.patches
            ):
                patches.pop(name)
    constraint_state = {
        **_validate_constraint_state(payload.get("constraint_state")),
        **rule_result.constraint_state,
    }
    if (
        "budget_max_cny" in rule_result.constraint_state
        and "budget_total_cny" not in rule_result.constraint_state
    ):
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
    # Downgrading accommodation changes both spend and location.  The rule
    # extractor covers explicit user consent, so a model may not infer this
    # permission merely from a low/total budget.
    if "lodging_flexibility" not in rule_result.constraint_state:
        constraint_state.pop("lodging_flexibility", None)
    _remove_destination_city_from_patch_must_visits(patches)
    _remove_destination_cities_from_state_must_visits(constraint_state)
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
