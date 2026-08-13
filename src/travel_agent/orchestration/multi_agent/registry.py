"""SubagentRegistry：只注册 5 个执行型 Subagent（Step 1 基础框架）。

- attraction / hotel / restaurant / transport / planner；
- **Reviewer 不在 Registry**——它由 Engine 在 Planner 完成后直接发起一次
  无工具 LLM 调用（见 ``review.py``），不经过 SubagentRunner；
- 所有执行型 Subagent：独立 Prompt、只持有工具白名单、有 max_steps /
  max_tool_calls / timeout 预算、不持有 ``dispatch_subagent``、不允许再创建
  新的 Agent（工具白名单物理上不含 dispatch 工具）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubagentDefinition:
    """一个执行型 Subagent 的静态定义。"""

    name: str
    description: str  # 给 Orchestrator 派工决策看的职责说明
    system_prompt: str
    tool_names: tuple[str, ...]  # 工具白名单（与 lc_tools 注册名一致）
    max_steps: int = 8  # create_agent recursion_limit
    max_tool_calls: int = 4  # 单次执行内工具调用上限（超限时结果带 warning）
    timeout_seconds: float = 90.0  # Step 2 起生效的执行超时预算


_ATTRACTION_PROMPT = """你是 attraction 领域 Subagent，只负责景点调研。
职责：景点搜索、兴趣匹配、天气适配、适老性与无障碍提示。
可用工具仅限：search_poi、check_weather。
纪律：
1. search_poi 成功一次即可，不要重复检索同一城市；
2. 涉及户外安排时查询一次天气，并给出雨天适配建议；
3. 有老人/儿童同行信号时，标注适老性与无障碍注意事项；
4. 结束时输出结构化结论：候选景点（含 poi_id）、匹配理由、天气影响、未解决事项。
禁止：调用白名单以外的工具、编造不存在的景点。"""

_HOTEL_PROMPT = """你是 hotel 领域 Subagent，只负责住宿调研。
职责：酒店候选、住宿区域比较、价格档位与商圈适配。
可用工具仅限：search_hotel。
纪律：
1. search_hotel 成功一次即可，可按区域参数补充检索一次，不要反复刷候选；
2. 输出结构化结论：候选酒店/区域（含 poi_id）、价格档位、与行程动线的匹配度、未解决事项；
3. 用户未明确预算时按画像 budget_level 推断，不要编造价格。
禁止：调用白名单以外的工具。"""

_RESTAURANT_PROMPT = """你是 restaurant 领域 Subagent，只负责餐饮调研。
职责：餐厅搜索、口味与饮食限制匹配、人均预算评估。
可用工具仅限：search_restaurant、estimate_budget。
纪律：
1. search_restaurant 成功一次即可，不要重复检索同一城市；
2. 注意画像中的 food_preference / avoid（忌口与饮食限制）并显式说明匹配情况；
3. 需要预算口径时可调用一次 estimate_budget；
4. 输出结构化结论：候选餐厅（含 poi_id）、口味匹配、人均档位、未解决事项。
禁止：调用白名单以外的工具。"""

_TRANSPORT_PROMPT = """你是 transport 领域 Subagent，只负责交通调研。
职责：市内与城际路线、换乘与步行、耗时估算、返程时间提醒。
可用工具仅限：search_poi、plan_route、estimate_budget。
纪律：
1. plan_route 的 poi_id 必须来自任务输入；若只有地点名称，先用 search_poi 分别解析起终点，禁止把名称冒充 poi_id；
2. 多段路线逐段估算，给出总耗时与换乘建议；
3. 输出结构化结论：路线段、耗时、方式、返程/截止时间风险提示、未解决事项。
禁止：调用白名单以外的工具、编造班次时刻。"""

_PLANNER_PROMPT = """你是 planner Subagent，唯一负责生成完整 TravelPlan 的执行者。
职责：读取明确的领域 Artifact，聚合约束与证据，生成完整行程并执行 plan_and_critique。
可用工具仅限：build_constraints、recommend_candidates、plan_and_critique。
纪律：
1. 只使用任务指令中给出的 artifact_id 读取领域结果，禁止依赖“最新结果”猜测；
2. 不得重新发起大规模外部搜索（你没有搜索工具）；
3. 标准顺序：build_constraints → recommend_candidates → plan_and_critique；
4. plan_and_critique 成功后输出 artifact_id、critic 是否通过、遗留告警与未解决事项。
禁止：调用白名单以外的工具、自行编造行程内容。"""


SUBAGENT_REGISTRY: dict[str, SubagentDefinition] = {
    "attraction": SubagentDefinition(
        name="attraction",
        description="景点搜索、兴趣匹配、天气适配、适老性与无障碍；工具：search_poi、check_weather。",
        system_prompt=_ATTRACTION_PROMPT,
        tool_names=("search_poi", "check_weather"),
        max_steps=8,
        max_tool_calls=4,
    ),
    "hotel": SubagentDefinition(
        name="hotel",
        description="酒店与住宿区域、价格和商圈比较；工具：search_hotel。",
        system_prompt=_HOTEL_PROMPT,
        tool_names=("search_hotel",),
        max_steps=6,
        max_tool_calls=3,
    ),
    "restaurant": SubagentDefinition(
        name="restaurant",
        description="餐厅搜索、口味与饮食限制、人均预算；工具：search_restaurant、estimate_budget。",
        system_prompt=_RESTAURANT_PROMPT,
        tool_names=("search_restaurant", "estimate_budget"),
        max_steps=6,
        max_tool_calls=4,
    ),
    "transport": SubagentDefinition(
        name="transport",
        description="市内与城际交通、换乘、步行、耗时与返程截止；工具：search_poi、plan_route、estimate_budget。",
        system_prompt=_TRANSPORT_PROMPT,
        tool_names=("search_poi", "plan_route", "estimate_budget"),
        max_steps=10,
        max_tool_calls=6,
    ),
    "planner": SubagentDefinition(
        name="planner",
        description="聚合领域 Artifact 与约束，生成完整 TravelPlan 并执行 plan_and_critique；"
        "工具：build_constraints、recommend_candidates、plan_and_critique。",
        system_prompt=_PLANNER_PROMPT,
        tool_names=("build_constraints", "recommend_candidates", "plan_and_critique"),
        # Three required tools commonly need AI -> tool graph transitions plus
        # a final answer.  Eight graph steps could terminate after the second
        # tool even though max_tool_calls had not been reached.
        max_steps=12,
        max_tool_calls=4,
        timeout_seconds=120.0,
    ),
}


def get_subagent(name: str) -> SubagentDefinition | None:
    return SUBAGENT_REGISTRY.get(name)


def list_subagents() -> list[SubagentDefinition]:
    return list(SUBAGENT_REGISTRY.values())


def all_subagent_tool_names() -> frozenset[str]:
    """全部执行型 Subagent 工具白名单的并集。"""
    names: set[str] = set()
    for definition in SUBAGENT_REGISTRY.values():
        names.update(definition.tool_names)
    return frozenset(names)
