"""SubagentRunner：统一 Subagent 执行器（Step 1 基础框架）。

职责边界（与最终架构对齐）：

- 按 ``SubagentDefinition.tool_names`` 白名单过滤工具，构建受限执行环境；
- 执行单个 ``SubagentTask``，组装结构化 ``SubagentResult``；
- **异常一律包装为 ``status="failed"`` 的结果，绝不向调用方上抛**；
- 不在 Registry 中的 agent 名直接返回 failed（不抛异常）；
- ``dispatch_subagent`` 永远不进入 Subagent 工具集（白名单物理隔离，
  ``assert_no_dispatch_leak`` 兜底校验）。

Step 1 阶段：``executor`` 参数支持注入 Mock 执行器（单测用）；真实
LLM 执行路径（create_react_agent + token/duration 采集）在 Step 3 接入。

并发纪律（Step 2 验证项，这里先声明契约）：

- ``run_subagent`` 的整个执行过程**不持有**共享 session 锁；
- 锁仅由各业务工具内部的 ``serialized()`` 在短暂读写共享状态时持有；
- Subagent LLM 推理与等待外部 API 期间不持锁。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from travel_agent.orchestration.multi_agent.registry import (
    SUBAGENT_REGISTRY,
    SubagentDefinition,
)
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    SubagentResult,
    SubagentTask,
)

# Mock/真实执行器统一返回结构：至少包含 summary，其余字段可选。
SubagentExecutor = Callable[[SubagentDefinition, SubagentTask, Any], dict[str, Any]]


def assert_no_dispatch_leak(tool_names: tuple[str, ...] | list[str]) -> None:
    """白名单兜底校验：任何 Subagent 工具集不得包含派工/编排类工具。"""
    forbidden = {"dispatch_subagent", "render_itinerary", "render_map"}
    leaked = forbidden.intersection(tool_names)
    if leaked:
        raise ValueError(f"subagent 工具白名单泄漏编排工具: {sorted(leaked)}")


class SubagentRunner:
    """统一执行器。

    在 executor 前后承担三件确定性的事（不依赖 LLM 自觉）：

    1. 设置任务元数据上下文（工具写 artifact 自动带 request/task/agent）；
    2. planner 任务执行前，按 inputs["artifact_ids"] 物化指定领域结果，
       避免 planner 依赖并发下不可靠的 latest()；
    3. 执行后按 store diff 归因本任务新产出的 evidence 与 payload。
    """

    def __init__(self, ctx: Any, executor: SubagentExecutor | None = None) -> None:
        self._ctx = ctx
        self._executor = executor

    def run_subagent(self, task: SubagentTask) -> SubagentResult:
        """执行单个任务并返回结构化结果；任何异常都包装为 failed，不上抛。"""
        from travel_agent.agent.session import (
            reset_task_meta,
            set_current_task_meta,
        )

        start = time.monotonic()
        definition = SUBAGENT_REGISTRY.get(task.agent)
        if definition is None:
            return SubagentResult(
                request_id=task.request_id,
                task_id=task.task_id,
                agent=task.agent,
                status=STATUS_FAILED,
                attempt=task.attempt,
                error=f"unknown subagent: {task.agent}",
                duration_ms=_elapsed_ms(start),
            )

        assert_no_dispatch_leak(definition.tool_names)

        if self._executor is None:
            return SubagentResult(
                request_id=task.request_id,
                task_id=task.task_id,
                agent=task.agent,
                status=STATUS_FAILED,
                attempt=task.attempt,
                error="no executor configured (real LLM path lands in Step 3)",
                duration_ms=_elapsed_ms(start),
            )

        store = getattr(self._ctx, "store", None)
        if task.agent == "planner":
            self._stage_planner_inputs(task)
        before_ids = store.artifact_ids() if store is not None else set()

        token = set_current_task_meta(
            {"request_id": task.request_id, "task_id": task.task_id, "agent": task.agent}
        )
        try:
            raw = self._executor(definition, task, self._ctx)
        except Exception as exc:  # noqa: BLE001 — 统一包装为 failed，不上抛
            return SubagentResult(
                request_id=task.request_id,
                task_id=task.task_id,
                agent=task.agent,
                status=STATUS_FAILED,
                attempt=task.attempt,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=_elapsed_ms(start),
            )
        finally:
            reset_task_meta(token)

        new_ids = sorted(store.artifact_ids() - before_ids) if store is not None else []
        return _build_result(task, raw, start, store, new_ids)

    def _stage_planner_inputs(self, task: SubagentTask) -> None:
        """把任务显式指定的 artifact 物化为新记录，planner 的既有工具
        （内部读 latest）因此只会消费指定结果，而非并发写入的其他产物。"""
        store = getattr(self._ctx, "store", None)
        artifact_ids = [aid for aid in (task.inputs.get("artifact_ids") or []) if aid]
        if store is None or not artifact_ids:
            return
        for artifact_id in artifact_ids:
            record = store.get_record(artifact_id)
            if record is None:
                continue
            store.put(
                record.get("kind") or "staged",
                record.get("payload") or {},
                request_id=task.request_id,
                task_id=task.task_id,
                agent="planner",
                data_source=record.get("data_source"),
            )


def _build_result(
    task: SubagentTask,
    raw: Any,
    start: float,
    store: Any = None,
    new_artifact_ids: list[str] | None = None,
) -> SubagentResult:
    raw = raw if isinstance(raw, dict) else {}
    evidence = list(raw.get("evidence") or [])
    payload = raw.get("payload")
    # 按 store diff 归因本任务新产出：evidence 与 payload 自动补全。
    if store is not None and new_artifact_ids:
        artifacts_by_kind: dict[str, dict] = {}
        for artifact_id in new_artifact_ids:
            record = store.get_record(artifact_id)
            if record is None:
                continue
            evidence.append(
                {
                    "artifact_id": artifact_id,
                    "kind": record.get("kind"),
                    "agent": record.get("agent") or task.agent,
                    "request_id": record.get("request_id") or task.request_id,
                    "task_id": record.get("task_id") or task.task_id,
                    "data_source": record.get("data_source"),
                }
            )
            kind = record.get("kind")
            if kind:
                artifacts_by_kind[kind] = record.get("payload") or {}
        if not isinstance(payload, dict) or not payload:
            payload = {"artifacts": artifacts_by_kind}
    return SubagentResult(
        request_id=task.request_id,
        task_id=task.task_id,
        agent=task.agent,
        status=str(raw.get("status") or STATUS_COMPLETED),
        attempt=task.attempt,
        summary=str(raw.get("summary") or ""),
        payload=payload if isinstance(payload, dict) else {},
        evidence=evidence,
        constraints_used=list(raw.get("constraints_used") or []),
        warnings=list(raw.get("warnings") or []),
        unresolved=list(raw.get("unresolved") or []),
        tool_trace=list(raw.get("tool_trace") or []),
        token_usage=dict(raw.get("token_usage") or {}),
        duration_ms=_elapsed_ms(start),
    )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
