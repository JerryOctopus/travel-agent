"""Main Orchestrator 工具面与 dispatch 工具骨架（Step 1 基础框架）。

架构约束（严格执行）：

- Main Orchestrator **不得直接调用业务重工具**：只允许
  ``ORCHESTRATOR_ALLOWED_TOOLS`` 内的编排/交互/渲染工具；
- 业务重工具（搜索/路线/预算/规划）只出现在领域 Subagent 白名单中；
- Subagent 不持有 ``dispatch_subagent``，不允许再创建 Agent；
- ``dispatch_subagent`` 的执行全程不持有共享 session 锁（锁仅由业务工具
  内部的 ``serialized()`` 短暂持有，见 lc_tools 审计结论）。

Step 1 阶段只提供：工具面常量、校验函数与 dispatch 闭包骨架；
StructuredTool 注册与 Orchestrator ReAct agent 构建在 Step 3 接入。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from travel_agent.orchestration.multi_agent.runner import SubagentRunner
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    SubagentResult,
    SubagentTask,
    new_request_id,
    new_task_id,
)

# Main Orchestrator 可见的全部工具（编排 + 交互 + 渲染）。
ORCHESTRATOR_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "update_travel_profile",
        "request_travel_info",
        "request_preference_guide",
        "dispatch_subagent",
        "render_itinerary",
        "render_map",
    }
)

# 业务重工具：禁止暴露给 Main Orchestrator，只属于领域 Subagent 白名单。
BUSINESS_HEAVY_TOOLS: frozenset[str] = frozenset(
    {
        "search_poi",
        "check_weather",
        "search_hotel",
        "search_restaurant",
        "plan_route",
        "estimate_budget",
        "build_constraints",
        "recommend_candidates",
        "plan_and_critique",
    }
)


def assert_orchestrator_toolset(tool_names: set[str] | frozenset[str] | list[str]) -> None:
    """校验 Orchestrator 工具面：不含业务重工具，且不越出允许清单。"""
    names = set(tool_names)
    leaked_business = names & BUSINESS_HEAVY_TOOLS
    if leaked_business:
        raise ValueError(f"orchestrator 工具面泄漏业务重工具: {sorted(leaked_business)}")
    unknown = names - ORCHESTRATOR_ALLOWED_TOOLS
    if unknown:
        raise ValueError(f"orchestrator 工具面包含未授权工具: {sorted(unknown)}")


def filter_orchestrator_tools(all_tools: list[Any]) -> list[Any]:
    """从全量 StructuredTool 列表中筛出 Orchestrator 可见工具（Step 3 使用）。"""
    filtered = [tool for tool in all_tools if tool.name in ORCHESTRATOR_ALLOWED_TOOLS]
    assert_orchestrator_toolset({tool.name for tool in filtered})
    return filtered


DispatchFn = Callable[[str, str, dict | None, list | None], str]


def build_dispatch_tool(runner: SubagentRunner, request_id: str | None = None) -> DispatchFn:
    """构建 ``dispatch_subagent`` 的执行闭包（Step 3 包装为 StructuredTool）。

    参数语义：``agent``（subagent 类型）、``instruction``（任务描述）、
    ``inputs``（可选结构化输入，可含 artifact_id 列表）、``depends_on``。

    返回 JSON 字符串（结构化 ``SubagentResult``），供 Orchestrator 消费
    payload / evidence / warnings / unresolved，而非只读 summary。

    并发契约：本函数**不获取任何 session 锁**；锁由 runner 内部业务工具
    短暂持有（Step 2 用死锁测试验证）。
    """
    rid = request_id or new_request_id()
    results_by_task: dict[str, SubagentResult] = {}

    def dispatch_subagent(
        agent: str,
        instruction: str,
        inputs: dict | None = None,
        depends_on: list | None = None,
    ) -> str:
        task_id = new_task_id(agent)
        merged_inputs = dict(inputs or {})
        dependency_ids: list[str] = []
        unresolved: list[str] = []
        for dependency_task_id in depends_on or []:
            dependency = results_by_task.get(str(dependency_task_id))
            if dependency is None or dependency.status not in {
                STATUS_COMPLETED,
                STATUS_COMPLETED_WITH_WARNINGS,
            }:
                unresolved.append(str(dependency_task_id))
                continue
            for artifact_id in _evidence_artifact_ids(dependency):
                if artifact_id not in dependency_ids:
                    dependency_ids.append(artifact_id)

        explicit_ids = [str(aid) for aid in (merged_inputs.get("artifact_ids") or []) if aid]
        for artifact_id in dependency_ids:
            if artifact_id not in explicit_ids:
                explicit_ids.append(artifact_id)
        if explicit_ids:
            merged_inputs["artifact_ids"] = explicit_ids

        task = SubagentTask(
            request_id=rid,
            task_id=task_id,
            agent=agent,
            instruction=instruction,
            inputs=merged_inputs,
            depends_on=list(depends_on or []),
        )
        if unresolved:
            result = SubagentResult(
                request_id=rid,
                task_id=task_id,
                agent=agent,
                status=STATUS_FAILED,
                error=f"unresolved dependencies: {unresolved}",
                unresolved=[f"dependency:{item}" for item in unresolved],
            )
        else:
            result = runner.run_subagent(task)
        results_by_task[result.task_id] = result
        return json.dumps(result.to_dict(), ensure_ascii=False, default=str)

    return dispatch_subagent


def _evidence_artifact_ids(result: SubagentResult) -> list[str]:
    artifact_ids: list[str] = []
    for item in result.evidence:
        artifact_id = item.get("artifact_id") if isinstance(item, dict) else None
        if artifact_id and artifact_id not in artifact_ids:
            artifact_ids.append(str(artifact_id))
    return artifact_ids
