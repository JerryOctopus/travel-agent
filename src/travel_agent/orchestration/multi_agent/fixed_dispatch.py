"""V1 确定性派工执行（Step 3）。

- 不使用 LLM 路由：按 ``dispatch_rules`` 的 task_type → 批次映射查表；
- 批次间串行、按 ``depends_on`` 把上游 evidence 的 artifact_id 注入下游
  ``inputs["artifact_ids"]``（planner 因此只消费明确指定的领域结果）；
- 未识别任务返回 clarification_required，不回退 full_itinerary；
- V1 与 V2 复用完全相同的 Subagent、工具、Schema 与 Planner（同一 Runner）。

注：同一批次内的并行执行属于 Step 2 已验证的并发能力，V1 这里按受控串行
执行（确定性优先），并行为可选优化。
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from travel_agent.orchestration.multi_agent.dispatch_rules import (
    STATUS_CLARIFICATION_REQUIRED,
    build_fixed_tasks,
)
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    SubagentResult,
    SubagentTask,
)


def run_fixed_dispatch(
    runner: Any,
    request_id: str,
    task_type: Any | None,
    *,
    task_brief: str,
    inputs: dict | None = None,
    constraints: dict | None = None,
    deadline: Any | None = None,
) -> tuple[str, list[SubagentResult]]:
    """执行 V1 固定派工，返回 ``(status, results)``。

    status 为 clarification_required 时 results 为空；否则为各任务结果，
    整体 status 为 completed（个别失败记录在各自 result.status 中）。
    """
    tasks_or_sentinel = build_fixed_tasks(
        request_id,
        task_type,
        task_brief=task_brief,
        inputs=inputs,
        constraints=constraints,
    )
    if tasks_or_sentinel == STATUS_CLARIFICATION_REQUIRED:
        return STATUS_CLARIFICATION_REQUIRED, []

    tasks: list[SubagentTask] = tasks_or_sentinel  # type: ignore[assignment]
    results_by_id: dict[str, SubagentResult] = {}
    pending = list(tasks)
    while pending:
        ready = [task for task in pending if all(dep in results_by_id for dep in task.depends_on)]
        if not ready:  # malformed fixed DAG; fail closed without spinning
            for task in pending:
                results_by_id[task.task_id] = _dependency_failure(
                    task,
                    [dep for dep in task.depends_on if dep not in results_by_id],
                )
            break

        executable: list[SubagentTask] = []
        for task in ready:
            upstream_artifacts: list[str] = []
            unresolved_dependencies: list[str] = []
            for dep_id in task.depends_on:
                dep_result = results_by_id[dep_id]
                if dep_result.status not in {
                    STATUS_COMPLETED,
                    STATUS_COMPLETED_WITH_WARNINGS,
                }:
                    unresolved_dependencies.append(dep_id)
                    continue
                artifact_ids = _evidence_artifact_ids(dep_result)
                if not artifact_ids:
                    unresolved_dependencies.append(dep_id)
                    continue
                upstream_artifacts.extend(artifact_ids)
            if unresolved_dependencies:
                results_by_id[task.task_id] = _dependency_failure(
                    task, unresolved_dependencies
                )
                continue
            if upstream_artifacts:
                task.inputs["artifact_ids"] = list(
                    dict.fromkeys(
                        list(task.inputs.get("artifact_ids") or []) + upstream_artifacts
                    )
                )
            executable.append(task)

        if executable:
            timeout = _fixed_batch_timeout(executable, deadline)
            with ThreadPoolExecutor(
                max_workers=len(executable), thread_name_prefix="fixed-wave"
            ) as pool:
                futures = {}
                for task in executable:
                    from travel_agent.orchestration.meter import current_turn_meter

                    meter = current_turn_meter()
                    if meter is not None:
                        meter.record_dispatch(task.agent)
                    context = contextvars.copy_context()
                    futures[task.task_id] = pool.submit(
                        context.run,
                        _run_with_optional_timeout,
                        runner,
                        task,
                        timeout,
                    )
                for task in executable:
                    results_by_id[task.task_id] = futures[task.task_id].result()

        ready_ids = {task.task_id for task in ready}
        pending = [task for task in pending if task.task_id not in ready_ids]

    return STATUS_COMPLETED, [results_by_id[task.task_id] for task in tasks]


def _fixed_batch_timeout(tasks: list[SubagentTask], deadline: Any | None) -> float | None:
    if deadline is None:
        return None
    if any(task.agent == "planner" for task in tasks):
        from travel_agent.orchestration.multi_agent.registry import SUBAGENT_REGISTRY

        component = max(SUBAGENT_REGISTRY[task.agent].timeout_seconds for task in tasks)
        return deadline.planner_timeout(component)
    from travel_agent.orchestration.multi_agent.registry import SUBAGENT_REGISTRY

    component = max(SUBAGENT_REGISTRY[task.agent].timeout_seconds for task in tasks)
    return deadline.effective_preplanner_timeout(component)


def _run_with_optional_timeout(
    runner: Any,
    task: SubagentTask,
    timeout: float | None,
) -> SubagentResult:
    if timeout is None:
        return runner.run_subagent(task)
    try:
        return runner.run_subagent(task, timeout_seconds=timeout)
    except TypeError as exc:
        if "timeout_seconds" not in str(exc):
            raise
        return runner.run_subagent(task)


def _dependency_failure(task: SubagentTask, unresolved: list[str]) -> SubagentResult:
    return SubagentResult(
        request_id=task.request_id,
        task_id=task.task_id,
        agent=task.agent,
        status=STATUS_FAILED,
        error=f"unresolved dependencies: {unresolved}",
        unresolved=[f"dependency:{item}" for item in unresolved],
    )


def _evidence_artifact_ids(result: SubagentResult) -> list[str]:
    ids: list[str] = []
    for item in result.evidence:
        artifact_id = item.get("artifact_id") if isinstance(item, dict) else None
        if artifact_id and artifact_id not in ids:
            ids.append(artifact_id)
    return ids
