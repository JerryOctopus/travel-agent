"""V1 确定性派工规则（Step 1 基础框架）。

V1 不调用 LLM 路由，也不对所有请求固定调全部 Subagent，而是按集中式
``task_type → required_subagents + dependencies`` 映射查表派工。

安全回退规则（严格执行）：
- 明确完整行程请求（``FULL_TRIP_PLAN``）→ 领域批次 + planner；
- 类型不明且信息不足（task_type 为 None 或不在映射表）→ clarification_required，
  **不允许统一回退 full_itinerary**；
- 一般旅行问答（POI_ADVICE / DAY_ADVICE / ROUTE_QUERY）→ 安全单任务路径；
- 不允许一句简单问答触发全部 Agent。

V1 与 V2 复用完全相同的 Subagent、工具、Schema 与 Planner，仅派工方式不同。
"""

from __future__ import annotations

import re

from travel_agent.agent.turn_analysis import TaskType
from travel_agent.candidate_comparison import agents_for_comparison
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    SubagentTask,
    new_task_id,
)

# task_type → 执行批次；同一批次内并行，批次间按顺序（依赖前一批全部完成）。
TASK_TYPE_SUBAGENT_MAP: dict[TaskType, tuple[tuple[str, ...], ...]] = {
    # A 到 B 怎么走 / 交通方式 / 耗时
    TaskType.ROUTE_QUERY: (("transport",),),
    # 住哪 / 选酒店 / 选区域等咨询：先取得 POI/酒店候选，再做动线比较。
    TaskType.CANDIDATE_COMPARISON: (("attraction",), ("transport",)),
    # 天气 / 特定日期去哪玩的轻量建议
    # 局部调整只派完成该调整所需的领域 worker；不自动升级为完整重规划。
    TaskType.ITINERARY_PATCH: (("attraction",), ("transport",)),
    TaskType.LOCAL_ADJUSTMENT_ADVICE: (("attraction",), ("transport",)),
    # 完整行程：领域并行 → 交通 → 规划
    TaskType.FULL_ITINERARY: (
        ("attraction",),
        ("transport",),
        ("planner",),
    ),
}


def specific_restaurant_recommendation_requested(
    task_brief: str,
    profile: dict | None = None,
    state: dict | None = None,
) -> bool:
    """Distinguish concrete dining output from dietary-only constraints."""
    text = str(task_brief or "").lower()
    profile = dict(profile or {})
    state = dict(state or {})
    if state.get("specific_restaurant_recommendation") is True:
        return True
    explicit_venue = bool(re.search(
        r"(?:推荐|找|选择?|安排|保留|列出|给出|具体|哪家).{0,12}(?:餐厅|饭店|馆子|早餐|午餐|晚餐|早饭|午饭|晚饭|用餐)"
        r"|(?:餐厅|饭店|馆子).{0,12}(?:推荐|哪家|具体|名单)",
        text,
    ))
    cuisine_intent = bool(re.search(r"(?:想吃|品尝|探店|美食|小吃|菜系|咖啡馆)", text))
    negative_only = bool(re.search(r"(?:饮食)?清淡|不吃|忌口|过敏|少盐|少油|低糖", text))
    if negative_only and not explicit_venue and not cuisine_intent:
        return False
    if "food" in (profile.get("interests") or []):
        return True
    preferences = [str(item) for item in (profile.get("food_preference") or []) if str(item).strip()]
    dietary_only = ("清淡", "不吃", "忌", "过敏", "少盐", "少油", "低糖", "无糖")
    if preferences and any(not any(marker in item for marker in dietary_only) for item in preferences):
        return True
    return explicit_venue or (cuisine_intent and not negative_only)


def agents_required_for_turn(
    task_type: TaskType | None,
    *,
    task_brief: str = "",
    inputs: dict | None = None,
) -> tuple[str, ...]:
    """Return only agents justified by explicit needs or hard constraints."""
    if task_type in {None, TaskType.CLARIFICATION, TaskType.CONSTRAINT_NEGOTIATION, TaskType.SAFE_DECLINE}:
        return ()
    text = task_brief.lower()
    profile = dict((inputs or {}).get("profile") or {})
    state = dict(profile.get("constraint_state") or {})
    exclusions = " ".join(str(item) for item in (state.get("exclude") or []))
    self_arranged_hotel = bool(re.search(r"(?:酒店|住宿).{0,8}(?:已订|订好|自己安排|自行安排)", text))
    self_arranged_food = bool(re.search(r"(?:餐厅|用餐|吃饭).{0,8}(?:已订|订好|自己安排|自行安排)", text))
    hotel_explicit = bool(
        re.search(r"(?:推荐|找|选|比较|对比|安排).{0,10}(?:酒店|住宿|住哪里|住宿区域)", text)
        or re.search(r"(?:酒店|住宿).{0,8}(?:降档|经济|便宜|省预算)", text)
        or state.get("compare_lodging_areas")
        or state.get("hotel_budget_per_night_cny")
        or state.get("lodging_area")
        or profile.get("hotel_area")
    ) and not self_arranged_hotel and "酒店推荐" not in exclusions
    restaurant_explicit = specific_restaurant_recommendation_requested(
        task_brief, profile, state
    )
    restaurant_explicit = (
        restaurant_explicit
        and not self_arranged_food
        and "餐厅推荐" not in exclusions
    )

    if task_type == TaskType.ROUTE_PLAN:
        return ("transport",)
    if task_type == TaskType.CANDIDATE_COMPARISON:
        return agents_for_comparison(task_brief, state)
    if task_type in {TaskType.ITINERARY_PATCH, TaskType.LOCAL_ADJUSTMENT_ADVICE}:
        base = ["attraction"]
        if re.search(r"路线|交通|步行|距离|通勤|耗时", text) and any(
            state.get(key) not in (None, "", [], {})
            for key in ("target_anchor", "location_anchor", "origin")
        ):
            base.append("transport")
        if hotel_explicit:
            base.append("hotel")
        if restaurant_explicit:
            base.append("restaurant")
        return tuple(base)
    agents = ["attraction", "transport", "planner"]
    if hotel_explicit:
        agents.insert(1, "hotel")
    if restaurant_explicit:
        agents.insert(1, "restaurant")
    return tuple(agents)


def resolve_task_batches(task_type: TaskType | None) -> tuple[tuple[str, ...], ...] | None:
    """查表得到执行批次；无法识别的任务类型返回 None（= 需要澄清）。"""
    if task_type is None:
        return None
    return TASK_TYPE_SUBAGENT_MAP.get(task_type)


def needs_clarification(task_type: TaskType | None) -> bool:
    return resolve_task_batches(task_type) is None


def build_fixed_tasks(
    request_id: str,
    task_type: TaskType | None,
    *,
    task_brief: str = "",
    inputs: dict | None = None,
    constraints: dict | None = None,
) -> list[SubagentTask] | str:
    """把映射表展开为带 ``depends_on`` 的 SubagentTask 列表。

    返回任务列表；类型不明时返回 ``STATUS_CLARIFICATION_REQUIRED`` 哨兵，
    由调用方转为追问，绝不回退 full_itinerary。
    """
    batches = resolve_task_batches(task_type)
    if batches is None:
        return STATUS_CLARIFICATION_REQUIRED

    allowed = set(agents_required_for_turn(task_type, task_brief=task_brief, inputs=inputs))
    if task_type == TaskType.FULL_ITINERARY:
        optional_domains = tuple(
            agent for agent in ("hotel", "restaurant") if agent in allowed
        )
        if optional_domains:
            batches = (("attraction", *optional_domains), *batches[1:])
    elif task_type == TaskType.CANDIDATE_COMPARISON:
        primary = tuple(agent for agent in ("attraction", "hotel", "restaurant") if agent in allowed)
        if primary:
            batches = (primary, *batches[1:])
    batches = tuple(tuple(agent for agent in batch if agent in allowed) for batch in batches)
    batches = tuple(batch for batch in batches if batch)
    tasks: list[SubagentTask] = []
    previous_ids: list[str] = []
    for batch in batches:
        tasks.extend(
            SubagentTask(
                request_id=request_id,
                task_id=new_task_id(agent),
                agent=agent,
                instruction=task_brief or f"按 {agent} 职责完成本轮任务并输出结构化结论。",
                inputs=dict(inputs or {}),
                constraints=dict(constraints or {}),
                # 累积依赖：等待所有已完成批次，保证下游（尤其 planner）能通过
                # depends_on 拿到全部上游 evidence 的 artifact_id。
                depends_on=list(previous_ids),
            )
            for agent in batch
        )
        previous_ids.extend(task.task_id for task in tasks[-len(batch) :])
    return tasks
