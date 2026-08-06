"""Multi-Agent Engine：统一引擎骨架与能力预设（Step 1 基础框架）。

核心约定（严格执行）：

- **Production = Full = V3**：三个名字是**同一配置对象的别名**；
- V0–V3 共用同一个 Engine，由 ``EngineCapabilities`` 降级产生：

  - V0：mode=single（单 Agent 持全部业务工具，无 Subagent、无 Reviewer）；
  - V1：mode=orchestrated + dispatch=fixed（task_type 查表派工，无 LLM 路由）；
  - V2：mode=orchestrated + dispatch=dynamic（LLM Orchestrator 动态派工）；
  - V3/Full/Production：V2 + Semantic Reviewer + 最多一次修复周期。

- 生产入口固定使用 ``PRODUCTION_CONFIG``，**不读取** settings 里的 variant；
  评测入口才允许显式选择 ``VARIANT_PRESETS`` 中的消融配置；
- Reviewer 启停只由 ``EngineCapabilities.reviewer_enabled`` 决定，
  且仅对"生成/修改完整 TravelPlan"的任务生效（Engine 判定，不靠 Prompt 自觉）。

Step 1 阶段 ``MultiAgentEngine.run`` 尚未接入生产路由（Step 3/4），
调用会抛出明确的 NotImplementedError。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent.schemas import EngineCapabilities

# --- V0–V3 能力预设 --------------------------------------------------------- #

V0_CONFIG = EngineCapabilities(
    mode="single",
    dispatch="none",
    reviewer_enabled=False,
    max_rework=0,
)

V1_CONFIG = EngineCapabilities(
    mode="orchestrated",
    dispatch="fixed",
    reviewer_enabled=False,
    max_rework=0,
)

V2_CONFIG = EngineCapabilities(
    mode="orchestrated",
    dispatch="dynamic",
    reviewer_enabled=False,
    max_rework=0,
)

FULL_CONFIG = EngineCapabilities(
    mode="orchestrated",
    dispatch="dynamic",
    reviewer_enabled=True,
    max_rework=1,
)

# 对象别名（不是拷贝、不是 is 表达式赋值）：Production = Full = V3。
V3_CONFIG = FULL_CONFIG
PRODUCTION_CONFIG = FULL_CONFIG

# 评测入口显式选择消融配置用的映射（生产入口不得使用）。
VARIANT_PRESETS: dict[str, EngineCapabilities] = {
    "v0": V0_CONFIG,
    "v1": V1_CONFIG,
    "v2": V2_CONFIG,
    "v3": V3_CONFIG,
}

# 只有生成/修改完整 TravelPlan 的任务才触发 Semantic Reviewer（V3）。
REVIEW_REQUIRED_TASK_TYPES: frozenset[TaskType] = frozenset(
    {TaskType.FULL_TRIP_PLAN, TaskType.ITINERARY_REVISION}
)


def capabilities_for_variant(variant: str) -> EngineCapabilities:
    """评测入口专用：variant 名 → 能力预设。非法名字直接报错，不做隐式回退。"""
    key = variant.strip().lower()
    if key not in VARIANT_PRESETS:
        raise ValueError(f"unknown variant: {variant!r} (allowed: {sorted(VARIANT_PRESETS)})")
    return VARIANT_PRESETS[key]


def requires_semantic_review(capabilities: EngineCapabilities, task_type: TaskType | None) -> bool:
    """Engine 判定是否执行 Reviewer：能力开启 且 任务会生成/修改完整 TravelPlan。

    路线查询、单餐厅/酒店/景点问答等不产生完整计划的任务 → 不 Review。
    """
    if not capabilities.reviewer_enabled:
        return False
    return task_type in REVIEW_REQUIRED_TASK_TYPES


@dataclass
class TurnOutcome:
    """一轮执行的最终交付状态（Step 3/4 填充真实字段）。"""

    status: str
    reply: str = ""
    plan_artifact_id: str | None = None
    rework_used: int = 0
    results: list[Any] = field(default_factory=list)


class MultiAgentEngine:
    """统一 Engine：V0–V3 与生产链路共用。

    Step 1 仅持有配置与协作者句柄；``run`` 的真实编排（准入层之后的
    Orchestrator / fixed dispatch / Reviewer / 修复周期）在 Step 3/4 接入。
    """

    def __init__(self, capabilities: EngineCapabilities = PRODUCTION_CONFIG) -> None:
        self.capabilities = capabilities

    @property
    def is_production(self) -> bool:
        """是否为生产 Full 配置（对象别名同一性判定）。"""
        return self.capabilities is PRODUCTION_CONFIG

    def run(self, *args: Any, **kwargs: Any) -> TurnOutcome:
        raise NotImplementedError(
            "MultiAgentEngine.run lands in Step 3/4; Step 1 only provides the skeleton."
        )
