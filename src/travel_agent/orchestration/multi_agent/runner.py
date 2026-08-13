"""SubagentRunner：统一 Subagent 执行器（Step 1 基础框架）。

职责边界（与最终架构对齐）：

- 按 ``SubagentDefinition.tool_names`` 白名单过滤工具，构建受限执行环境；
- 执行单个 ``SubagentTask``，组装结构化 ``SubagentResult``；
- **异常一律包装为 ``status="failed"`` 的结果，绝不向调用方上抛**；
- 不在 Registry 中的 agent 名直接返回 failed（不抛异常）；
- ``dispatch_subagent`` 永远不进入 Subagent 工具集（白名单物理隔离，
  ``assert_no_dispatch_leak`` 兜底校验）。

Step 1 阶段：``executor`` 参数支持注入 Mock 执行器（单测用）；真实
LLM 执行路径（create_agent + token/duration 采集）在 Step 3 接入。

并发纪律（Step 2 验证项，这里先声明契约）：

- ``run_subagent`` 的整个执行过程**不持有**共享 session 锁；
- 锁仅由各业务工具内部的 ``serialized()`` 在短暂读写共享状态时持有；
- Subagent LLM 推理与等待外部 API 期间不持锁。
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import time
from typing import Any, Callable

from travel_agent.orchestration.multi_agent.registry import (
    SUBAGENT_REGISTRY,
    SubagentDefinition,
)
from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_COMPLETED,
    STATUS_BUDGET_EXHAUSTED,
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
    2. planner 任务只绑定 inputs["artifact_ids"] 中的明确领域结果；
    3. 执行后按 request_id/task_id/agent 元数据归因本任务产物。
    """

    def __init__(self, ctx: Any, executor: SubagentExecutor | None = None) -> None:
        self._ctx = ctx
        self._executor = executor

    @property
    def context(self) -> Any:
        return self._ctx

    def run_subagent(
        self,
        task: SubagentTask,
        *,
        timeout_seconds: float | None = None,
    ) -> SubagentResult:
        result = self._run_subagent(task, timeout_seconds=timeout_seconds)
        from travel_agent.orchestration.multi_agent.trace import current_trace

        trace = current_trace()
        if trace is not None:
            trace.append(
                "subagent",
                agent=result.agent,
                task_id=result.task_id,
                status=result.status,
                attempt=result.attempt,
                error=result.error,
                duration_ms=result.duration_ms,
                detail={
                    "effective_timeout_ms": (
                        int(timeout_seconds * 1000)
                        if timeout_seconds is not None
                        else None
                    ),
                    "tool_trace": list(result.tool_trace),
                    "evidence": list(result.evidence),
                    "warnings": list(result.warnings),
                    "unresolved": list(result.unresolved),
                },
            )
        return result

    def _run_subagent(
        self,
        task: SubagentTask,
        *,
        timeout_seconds: float | None = None,
    ) -> SubagentResult:
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

        effective_timeout = min(
            definition.timeout_seconds,
            float(timeout_seconds)
            if timeout_seconds is not None
            else definition.timeout_seconds,
        )
        if effective_timeout <= 0:
            return SubagentResult(
                request_id=task.request_id,
                task_id=task.task_id,
                agent=task.agent,
                status=STATUS_BUDGET_EXHAUSTED,
                attempt=task.attempt,
                error="effective timeout exhausted before dispatch",
                warnings=["未启动 Subagent：阶段 deadline 已无可用时间"],
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
        artifact_ids = [str(aid) for aid in (task.inputs.get("artifact_ids") or []) if aid]
        if task.agent == "planner":
            invalid_ids = [
                artifact_id
                for artifact_id in artifact_ids
                if store is None or store.get_record(artifact_id) is None
            ]
            if not artifact_ids or invalid_ids:
                reason = "planner requires explicit artifact_ids"
                if invalid_ids:
                    reason += f"; missing={invalid_ids}"
                return SubagentResult(
                    request_id=task.request_id,
                    task_id=task.task_id,
                    agent=task.agent,
                    status=STATUS_FAILED,
                    attempt=task.attempt,
                    error=reason,
                    duration_ms=_elapsed_ms(start),
                )
        elif artifact_ids and store is not None:
            task.inputs["artifact_inputs"] = _compact_dependency_inputs(store, artifact_ids)

        isolated_ctx = (
            self._ctx.clone_isolated() if hasattr(self._ctx, "clone_isolated") else self._ctx
        )
        isolated_store = getattr(isolated_ctx, "store", None)
        base_ids = isolated_store.artifact_ids() if isolated_store is not None else set()

        def _execute() -> Any:
            # Preserve our task/trace/meter context, but do not inherit the
            # parent LangChain RunnableConfig.  Otherwise Orchestrator
            # callbacks observe every nested worker model call as if it were
            # an Orchestrator call, double-counting tokens, calls and latency.
            from langchain_core.runnables.config import var_child_runnable_config

            runnable_token = var_child_runnable_config.set({})
            token = set_current_task_meta(
                {
                    "request_id": task.request_id,
                    "task_id": task.task_id,
                    "agent": task.agent,
                    "artifact_ids": artifact_ids,
                    "revision_directives": dict(
                        task.inputs.get("revision_directives") or {}
                    ),
                }
            )
            try:
                return self._executor(definition, task, isolated_ctx)
            finally:
                reset_task_meta(token)
                var_child_runnable_config.reset(runnable_token)

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"subagent-{task.agent}")
        future = pool.submit(contextvars.copy_context().run, _execute)
        try:
            raw = future.result(timeout=effective_timeout)
        except FutureTimeoutError:
            future.cancel()
            from travel_agent.orchestration.meter import current_turn_meter

            meter = current_turn_meter()
            if meter is not None:
                role = "planner" if task.agent == "planner" else f"worker:{task.agent}"
                meter.fail_pending_llm(
                    role, f"subagent timeout after {effective_timeout:g}s"
                )
            return SubagentResult(
                request_id=task.request_id,
                task_id=task.task_id,
                agent=task.agent,
                status=STATUS_BUDGET_EXHAUSTED,
                attempt=task.attempt,
                error=f"timeout after {effective_timeout:g}s",
                warnings=["Subagent 执行达到 timeout_seconds 硬上限"],
                duration_ms=_elapsed_ms(start),
            )
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
            pool.shutdown(wait=False, cancel_futures=True)

        if isolated_ctx is not self._ctx and hasattr(self._ctx, "merge_task_from"):
            try:
                self._ctx.merge_task_from(isolated_ctx, base_ids)
            except Exception as exc:  # cancelled request or failed atomic publish
                return SubagentResult(
                    request_id=task.request_id,
                    task_id=task.task_id,
                    agent=task.agent,
                    status=STATUS_FAILED,
                    attempt=task.attempt,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=_elapsed_ms(start),
                )

        new_ids = (
            store.artifact_ids_for_task(
                task.request_id,
                task.task_id,
                agent=task.agent,
            )
            if store is not None
            else []
        )
        return _build_result(task, raw, start, store, new_ids)


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


def _compact_dependency_inputs(store: Any, artifact_ids: list[str]) -> list[dict[str, Any]]:
    """给非 Planner 下游提供可执行的紧凑依赖数据；Planner 仍只按 ID 自取。"""
    compact: list[dict[str, Any]] = []
    # Revision plans can reference the previous plan and its whole ancestry.
    # Workers need a bounded snapshot, not recursively embedded plan payloads.
    selected_ids = list(dict.fromkeys(artifact_ids))
    if len(selected_ids) > 20:
        selected_ids = [selected_ids[0], *selected_ids[-19:]]
    for artifact_id in selected_ids:
        record = store.get_record(artifact_id)
        payload = store.get(artifact_id)
        if record is None or not isinstance(payload, dict):
            continue
        kind = str(record.get("kind") or "")
        value: dict[str, Any] = {"city": payload.get("city")}
        if kind == "candidates":
            value["pois"] = [
                {"poi_id": item.get("poi_id"), "name": item.get("name")}
                for item in (payload.get("pois") or [])[:8]
                if isinstance(item, dict)
            ]
        elif kind == "ranked":
            value["pois"] = [
                {
                    "poi_id": (item.get("poi") or {}).get("poi_id"),
                    "name": (item.get("poi") or {}).get("name"),
                }
                for item in (payload.get("pois") or [])[:8]
                if isinstance(item, dict) and isinstance(item.get("poi"), dict)
            ]
        elif kind == "hotels":
            value["hotels"] = [
                {"poi_id": item.get("hotel_id"), "name": item.get("name")}
                for item in (payload.get("hotels") or [])[:8]
                if isinstance(item, dict)
            ]
        elif kind == "restaurants":
            value["restaurants"] = [
                {"poi_id": item.get("poi_id"), "name": item.get("name")}
                for item in (payload.get("restaurants") or [])[:8]
                if isinstance(item, dict)
            ]
        elif kind == "itinerary":
            plan = payload.get("itinerary") or {}
            value = {
                "city": plan.get("city"),
                "summary": plan.get("summary"),
                "days": [
                    {
                        "day_index": day.get("day_index"),
                        "stops": [
                            {
                                "poi_id": (stop.get("poi") or {}).get("poi_id"),
                                "name": (stop.get("poi") or {}).get("name"),
                                "start_time": stop.get("start_time"),
                            }
                            for stop in (day.get("stops") or [])[:4]
                            if isinstance(stop, dict)
                        ],
                    }
                    for day in (plan.get("days") or [])[:14]
                    if isinstance(day, dict)
                ],
            }
        elif kind == "routes":
            value = {
                key: payload.get(key)
                for key in (
                    "origin_poi_id",
                    "destination_poi_id",
                    "origin_name",
                    "destination_name",
                    "distance_km",
                    "duration_min",
                    "mode",
                    "walking_distance_km",
                    "source",
                )
                if payload.get(key) is not None
            }
        else:
            value = {
                key: _bounded_dependency_value(payload.get(key))
                for key in sorted(payload)[:12]
                if key not in {
                    "domain_inputs",
                    "original_itinerary",
                    "source_artifact_ids",
                }
            }
        compact.append({"artifact_id": artifact_id, "kind": kind, "payload": value})
    return compact


def _bounded_dependency_value(value: Any, depth: int = 0) -> Any:
    if depth >= 2:
        if isinstance(value, (dict, list)):
            return f"<{type(value).__name__}:{len(value)}>"
        return str(value)[:160] if value is not None else None
    if isinstance(value, str):
        return value[:240]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [_bounded_dependency_value(item, depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        return {
            str(key): _bounded_dependency_value(item, depth + 1)
            for key, item in list(value.items())[:12]
        }
    return str(value)[:160]
