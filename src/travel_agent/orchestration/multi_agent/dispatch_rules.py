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

from travel_agent.agent.turn_analysis import TaskType
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_CLARIFICATION_REQUIRED,
    SubagentTask,
    new_task_id,
)

# task_type → 执行批次；同一批次内并行，批次间按顺序（依赖前一批全部完成）。
TASK_TYPE_SUBAGENT_MAP: dict[TaskType, tuple[tuple[str, ...], ...]] = {
    # A 到 B 怎么走 / 交通方式 / 耗时
    TaskType.ROUTE_QUERY: (("transport",),),
    # 住哪 / 选酒店 / 选区域等咨询（酒店区域比较需要动线参考）
    TaskType.POI_ADVICE: (("hotel", "transport"),),
    # 天气 / 特定日期去哪玩的轻量建议
    TaskType.DAY_ADVICE: (("attraction",),),
    # 修改既有行程：由 planner 基于既有 artifact 重新规划
    TaskType.ITINERARY_REVISION: (("planner",),),
    # 完整行程：领域并行 → 交通 → 规划
    TaskType.FULL_TRIP_PLAN: (
        ("attraction", "hotel", "restaurant"),
        ("transport",),
        ("planner",),
    ),
}


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
