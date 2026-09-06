"""Semantic Reviewer：Engine 直调的单层无工具 LLM 复核（Step 1 基础框架）。

架构约束（严格执行）：

- Reviewer **不属于 SubagentRegistry**、**不经过 SubagentRunner**；
- 不挂载任何业务工具，不调用 ``plan_and_critique``，不重新生成整份计划；
- 输入：Planner 产出的 TravelPlan、Profile、领域 ``SubagentResult`` 与 Evidence；
- 输出：结构化 ``ReviewResult``（issue 含 severity / repair_target /
  repair_instruction / evidence）；
- 触发由 Engine 决定：仅"生成/修改完整 TravelPlan"的任务在 V3 固定执行一次。

修复周期规则（由 Engine 执行，此处只提供判定原语）：

- critical：不得自动交付 → incomplete / clarification_required / failed；
- recoverable：最多触发**一次**修复周期（定向重派领域 Subagent → Planner
  重新规划），修复后不再二轮 Review，整个周期只计一次 rework；
- noncritical：可 completed_with_warnings 交付。

Step 1 阶段提供：ReviewContext 组装、无工具的 prompt 构建、
Mock 可注入的 ``run_semantic_review`` 骨架与修复周期判定函数。
真实 LLM 调用在 Step 3 接入。
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from travel_agent.orchestration.multi_agent.schemas import (
    SEVERITIES,
    SEVERITY_CRITICAL,
    SEVERITY_NONCRITICAL,
    SEVERITY_RECOVERABLE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    VERDICT_FAILED,
    VERDICT_PASS,
    VERDICT_REWORK,
    ReviewIssue,
    ReviewResult,
    SubagentResult,
)

@dataclass
class ReviewContext:
    """Engine 组装给 Reviewer 的只读输入快照（不含任何工具句柄）。"""

    request_id: str
    plan: dict[str, Any]  # TravelPlan artifact 的 dict 形式
    profile_brief: dict[str, Any] = field(default_factory=dict)
    subagent_results: list[SubagentResult] = field(default_factory=list)
    task_brief: str = ""

    def to_prompt_text(self) -> str:
        """渲染为 Reviewer prompt 的上下文段（纯文本，无工具）。"""
        plan = _review_artifact_snapshot(self.plan, self.profile_brief)
        if isinstance(plan.get("itinerary"), dict):
            plan["itinerary"] = _compact_itinerary(plan["itinerary"])
        payload = {
            "request_id": self.request_id,
            "task_brief": self.task_brief,
            "profile": self.profile_brief,
            "plan": plan,
            "subagent_results": [
                {
                    "task_id": result.task_id,
                    "agent": result.agent,
                    "status": result.status,
                    "evidence": result.evidence,
                    "warnings": result.warnings,
                    "unresolved": result.unresolved,
                }
                for result in self.subagent_results
            ],
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


# Mock/真实调用统一签名：接收 ReviewContext，返回 ReviewResult 的 dict 形式。
ReviewCallable = Callable[[ReviewContext], dict[str, Any]]
REVIEWER_MAX_OUTPUT_TOKENS = 1024
REVIEWER_TIMEOUT_SECONDS = 90

_REVIEWER_INSTRUCTION = """你是行程语义复核者（Semantic Reviewer）。你不拥有任何工具，也不重新生成计划。
必须完整读取 artifact 中的 itinerary、budget_plan、lodging_plan、route_evidence、
meal_strategy、validation_result 与 active_constraints，再结合领域调研证据检查：软偏好匹配、行程节奏、
适老性/无障碍、跨领域一致性（交通耗时与景点安排、预算与档位等）。
对每个问题输出 JSON：issue_type / severity(critical|recoverable|noncritical) /
description / evidence / repair_target(attraction|hotel|restaurant|transport|planner) /
repair_instruction。严重度必须遵循：
- critical：计划中存在有明确证据的硬约束冲突或安全风险，且一次有界修复无法可靠解决；verdict 必须为 failed；
- recoverable：计划或证据有具体缺口，能由一次定向领域补充或 Planner 重排解决；verdict 为 rework；
- noncritical：实时开放、无障碍设施、班次等工具证据未覆盖而需出发前/现场复核，或仅是优化建议；不得标成 critical；
- 不要求无关领域都提供结果；确定性 critic 已通过的项目不要在没有相反证据时判为冲突；
- profile.constraint_state.candidate_attractions 是待核验/比较的候选集合，不等同 must_visit；
  候选闭馆或证据不支持时，计划省略并明确说明限制是有效处理，不得按“遗漏必去”判 critical；
- transport_modes 或 taxi_backup 表示允许/备选方式，不要求每种方式都必须出现在最终路线；
  已满足 public_transport_required 且未使用禁用方式时，缺少 taxi 实际路段最多是优化建议；
- 轻松节奏中的空档、相邻地点重复、未提供无障碍/实时班次信息等，如无明确硬约束或安全证据，
  只能判 noncritical；不得把“可能”“疑似”或常识猜测作为 critical/recoverable 的相反证据；
- 只以 profile 中的显式约束判定硬冲突，不要把 task_brief 的自然语言候选或领域缺席自行升级为硬约束；
- 饮食限制约束的是已安排餐饮必须合规；某天没有餐厅/餐饮 stop 本身不代表吃了不合规食物，
  未明确要求具体餐厅时只检查是否合理预留用餐时间，缺少具体餐厅不是问题、不得触发返工。
  只有明确餐厅需求/饮食硬约束缺少可靠证据，或计划明确安排了冲突餐厅，才能判硬冲突；
- 用户自有但未指定具体场所的用餐预约，以 fixed_event_plan 中保留的日期、时间和区域为准；
  其区域记录在 user_owned_unspecified_fixed_event_locations 时不得要求虚构餐厅 POI，但仍应检查到该区域的路线证据；
- 住宿由 hotel 领域 evidence、住宿区域与补充卡片承载，不要求把酒店塞进 itinerary.days[].stops；
  用户未明确要求具体酒店时，住宿区域即可；不得强制推荐具体酒店或要求 Planner 添加酒店 stop；
- noncritical/warning 只作交付注释，不得改变 verdict/outcome，也不得阻止交付；
- 每个问题的 evidence 必须绑定 artifact 的具体字段路径（如 budget_plan.expected_total）
  或给定的工具/Artifact evidence id；没有绑定证据的推测只能是 noncritical；
- validation_result/critic 的确定性通过结论优先于无相反工具证据的推测性建议；
- 时间衔接只需满足“上一站开始时间 + 上一站停留时长 + 路线耗时 <= 下一站开始时间”；
  不得再从这个衔接间隔中扣除下一站自身的停留时长；
- budget_plan 中带 source_artifact_id 的金额是预算领域给出的区间估算；没有相反价格证据、算术错误或
  用户预算超限时，不得仅因 itinerary 没有逐景点票价字段要求返工；
- verdict=pass 时 issues 必须为空；verdict=rework 时不得含 critical；存在 critical 时 verdict=failed。
整体给出 verdict：pass | rework | failed。"""


def run_semantic_review(
    review_ctx: ReviewContext,
    *,
    review_callable: ReviewCallable | None = None,
    timeout_seconds: float | None = None,
) -> ReviewResult:
    """Engine 直调的一次无工具 Review。

    Step 1：``review_callable`` 为空时返回 failed（真实 LLM 路径 Step 3 接入）；
    任何异常包装为 ``verdict="failed"``，不上抛。
    """
    start = time.monotonic()
    if review_callable is None:
        return ReviewResult(
            verdict=VERDICT_FAILED,
            error="no review callable configured (real LLM path lands in Step 3)",
            duration_ms=_elapsed_ms(start),
        )
    pool = None
    try:
        if timeout_seconds is None:
            raw = review_callable(review_ctx)
        else:
            if timeout_seconds <= 0:
                raise TimeoutError("reviewer admission denied")
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-review")
            future = pool.submit(
                contextvars.copy_context().run,
                review_callable,
                review_ctx,
            )
            try:
                raw = future.result(timeout=timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                from travel_agent.orchestration.meter import current_turn_meter

                meter = current_turn_meter()
                if meter is not None:
                    meter.fail_pending_llm(
                        "reviewer", f"reviewer timeout after {timeout_seconds:g}s"
                    )
                raise TimeoutError(
                    f"reviewer timeout after {timeout_seconds:g}s"
                ) from exc
    except Exception as exc:  # noqa: BLE001 — 包装为 failed，不上抛
        return ReviewResult(
            verdict=VERDICT_FAILED,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=_elapsed_ms(start),
        )
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    return _calibrate_review_result(_build_review_result(raw, start), review_ctx)


def reviewer_prompt(review_ctx: ReviewContext) -> str:
    """组装完整 Reviewer prompt（Step 3 真实调用复用）。"""
    return f"{_REVIEWER_INSTRUCTION}\n\n输入（JSON）：\n{review_ctx.to_prompt_text()}\n\n只输出 JSON，格式：{{\"verdict\": \"pass|rework|failed\", \"issues\": [...]}}"


def extract_json_payload(text: str) -> dict[str, Any]:
    """从 LLM 回复中提取 JSON 对象（容错 markdown 围栏与前后缀文本）。"""
    candidate = (text or "").strip()
    if "```" in candidate:
        fenced = candidate.split("```")
        for part in fenced:
            stripped = part.strip()
            if stripped.startswith("{"):
                candidate = stripped.strip("`").strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                break
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_review_callable(
    settings: Any,
    model: Any | None = None,
    *,
    timeout_seconds: float | None = None,
    evaluation_trace: list[dict[str, Any]] | None = None,
) -> ReviewCallable | None:
    """构建真实的单层无工具 LLM Review 调用；LLM 不可用时返回 None。

    返回的 callable 只读 ReviewContext，不挂载任何工具。仅当首次响应为空或
    缺少合法 verdict/issues 时，允许一次格式纠正重试；有效业务 verdict 不重试。
    """
    llm = getattr(settings, "llm", None)
    if model is None and (llm is None or not getattr(llm, "enabled", False)):
        return None

    def review_callable(review_ctx: ReviewContext) -> dict[str, Any]:
        active_model = model
        if active_model is None:
            from travel_agent.agent.runtime import _build_chat_model

            # Give one complete generation window, but do not turn one loaded
            # provider failure into ChatOpenAI's implicit three-attempt wait.
            active_model = _build_chat_model(
                settings,
                timeout_seconds=(
                    REVIEWER_TIMEOUT_SECONDS
                    if timeout_seconds is None
                    else timeout_seconds
                ),
                max_retries=0,
            )
        from langchain_core.messages import HumanMessage

        from travel_agent.orchestration.meter import meter_callbacks

        # Reviewer emits a small JSON verdict. Bounding the output prevents a
        # provider from spending the full timeout on verbose reasoning while
        # preserving the review rubric and fail-closed delivery gate.
        max_output_tokens = int(
            getattr(
                getattr(settings, "orchestration", None),
                "reviewer_max_output_tokens",
                REVIEWER_MAX_OUTPUT_TOKENS,
            )
            or REVIEWER_MAX_OUTPUT_TOKENS
        )
        bounded_model = active_model.bind(max_tokens=max_output_tokens)
        callbacks = meter_callbacks("reviewer")
        if evaluation_trace is not None:
            from travel_agent.agent.evaluation_trace import EvaluationTraceCallback

            callbacks.append(
                EvaluationTraceCallback(
                    evaluation_trace,
                    model=getattr(settings.llm, "model", ""),
                    phase="reviewer",
                )
            )
        prompt = reviewer_prompt(review_ctx)
        for attempt in range(2):
            if attempt:
                prompt = (
                    reviewer_prompt(review_ctx)
                    + "\n\n上一次响应为空或 JSON 结构非法。请仅返回一个 JSON object，"
                    "必须包含合法 verdict 与 issues 数组，不要输出解释文字。"
                )
            response = bounded_model.invoke(
                [HumanMessage(content=prompt)],
                config={"callbacks": callbacks},
            )
            content = response.content if isinstance(response.content, str) else str(response.content)
            parsed = extract_json_payload(content)
            if _review_contract_shape_valid(parsed):
                return parsed
        return parsed

    return review_callable


def _review_contract_shape_valid(raw: Any) -> bool:
    """Cheap preflight so one malformed business verdict gets the format retry."""
    if not isinstance(raw, dict):
        return False
    verdict = raw.get("verdict")
    issues = raw.get("issues")
    if verdict not in {VERDICT_PASS, VERDICT_REWORK, VERDICT_FAILED} or not isinstance(issues, list):
        return False
    if verdict == VERDICT_PASS:
        return not issues
    allowed_targets = {
        "",
        "none",
        "null",
        "n/a",
        "无",
        "attraction",
        "hotel",
        "restaurant",
        "transport",
        "planner",
    }
    for item in issues:
        if not isinstance(item, dict):
            return False
        severity = str(item.get("severity") or "")
        repair_target = str(item.get("repair_target") or "").strip().lower()
        if (
            not str(item.get("issue_type") or "").strip()
            or not str(item.get("description") or "").strip()
            or severity not in SEVERITIES
            or repair_target not in allowed_targets
        ):
            return False
        if severity == SEVERITY_RECOVERABLE and (
            repair_target not in {
                "attraction", "hotel", "restaurant", "transport", "planner"
            }
            or not str(item.get("repair_instruction") or "").strip()
        ):
            return False
    severities = {
        str(item.get("severity") or "")
        for item in issues
        if isinstance(item, dict)
    }
    if verdict == VERDICT_REWORK:
        return SEVERITY_RECOVERABLE in severities and SEVERITY_CRITICAL not in severities
    return SEVERITY_CRITICAL in severities


def _review_artifact_snapshot(
    plan: dict[str, Any],
    profile_brief: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete semantic artifact contract without map-only bloat."""
    snapshot = {
        key: plan.get(key)
        for key in (
            "itinerary",
            "budget_plan",
            "lodging_plan",
            "meal_strategy",
            "validation_result",
            "critic",
            "return_plan",
            "fixed_event_plan",
            "mobility_plan",
            "candidate_verification",
            "revision_notes",
            "final_issue_count",
            "source_artifact_ids",
            "revision_directives",
            "state_version",
        )
        if key in plan
    }
    active_constraints = (
        profile_brief.get("constraint_state")
        if isinstance(profile_brief.get("constraint_state"), dict)
        else {}
    )
    snapshot["active_constraints"] = active_constraints
    snapshot["route_evidence"] = {
        "required_route_anchors": plan.get("required_route_anchors"),
        "lodging_route_anchors": plan.get("lodging_route_anchors"),
        "return_plan": plan.get("return_plan"),
        "transport_artifacts": [
            {
                "artifact_id": item.get("artifact_id"),
                "payload": item.get("payload"),
            }
            for item in ((plan.get("domain_inputs") or {}).get("transport") or [])
            if isinstance(item, dict)
        ],
    }
    if not snapshot.get("meal_strategy"):
        snapshot["meal_strategy"] = {
            "dietary_constraints": active_constraints.get("dietary"),
            "restaurant_artifacts": [
                {
                    "artifact_id": item.get("artifact_id"),
                    "payload": item.get("payload"),
                }
                for item in ((plan.get("domain_inputs") or {}).get("restaurants") or [])
                if isinstance(item, dict)
            ],
        }
    return snapshot


def _compact_itinerary(itinerary: dict[str, Any]) -> dict[str, Any]:
    """Keep reviewer-relevant semantics without repeating bulky map fields."""
    compact_days: list[dict[str, Any]] = []
    for raw_day in itinerary.get("days") or []:
        if not isinstance(raw_day, dict):
            continue
        stops: list[dict[str, Any]] = []
        for raw_stop in raw_day.get("stops") or []:
            if not isinstance(raw_stop, dict):
                continue
            poi = raw_stop.get("poi") if isinstance(raw_stop.get("poi"), dict) else {}
            route = (
                raw_stop.get("route_from_previous")
                if isinstance(raw_stop.get("route_from_previous"), dict)
                else None
            )
            stop = {
                key: raw_stop.get(key)
                for key in ("start_time", "duration_min", "note")
                if raw_stop.get(key) is not None
            }
            stop["poi"] = {
                key: poi.get(key)
                for key in (
                    "name",
                    "city",
                    "category",
                    "tags",
                    "indoor",
                    "opening_hours",
                    "average_cost",
                    "address",
                    "source",
                )
                if poi.get(key) is not None
            }
            if route is not None:
                stop["route_from_previous"] = {
                    key: route.get(key)
                    for key in (
                        "duration_min",
                        "distance_km",
                        "walking_distance_km",
                        "mode",
                        "source",
                    )
                    if route.get(key) is not None
                }
            stops.append(stop)
        compact_days.append(
            {
                key: raw_day.get(key)
                for key in ("day_index", "theme")
                if raw_day.get(key) is not None
            }
            | {"stops": stops}
        )
    return {
        key: itinerary.get(key)
        for key in ("city", "summary")
        if itinerary.get(key) is not None
    } | {"days": compact_days}


def resolve_delivery_status(
    review: ReviewResult,
    *,
    reviewer_enabled: bool,
    max_rework: int,
    rework_used: int,
) -> tuple[str, bool]:
    """按 severity 规则推导交付状态与是否触发修复周期。

    返回 ``(status, start_repair_cycle)``：

    - 未启用 Reviewer → completed；
    - 存在 critical → incomplete，禁止修复周期（不得未经再次验证自动交付）；
    - verdict=rework 且仅 recoverable 且 ``rework_used < max_rework``
      → 触发一次修复周期（status 由修复结果决定，这里返回 incomplete）；
    - 仅 noncritical → completed_with_warnings；
    - pass → completed。
    """
    if not reviewer_enabled:
        return STATUS_COMPLETED, False
    if review.error:
        # Review 自身失败按 critical 对待：不得声称计划已完成。
        return STATUS_INCOMPLETE, False
    if review.critical_issues():
        return STATUS_INCOMPLETE, False
    if review.verdict == VERDICT_FAILED:
        return STATUS_FAILED, False
    if review.verdict == VERDICT_REWORK:
        if rework_used < max_rework and any(
            issue.severity == SEVERITY_RECOVERABLE for issue in review.issues
        ):
            return STATUS_INCOMPLETE, True
        return STATUS_INCOMPLETE, False
    if any(issue.severity == SEVERITY_NONCRITICAL for issue in review.issues):
        return STATUS_COMPLETED_WITH_WARNINGS, False
    return STATUS_COMPLETED, False


def repair_targets(review: ReviewResult) -> list[str]:
    """提取一次修复周期需要定向重派的领域 Subagent（去重、保序，planner 总在最后）。"""
    targets: list[str] = []
    for issue in review.issues:
        if issue.severity != SEVERITY_RECOVERABLE or not issue.repair_target:
            continue
        if issue.repair_target not in targets:
            targets.append(issue.repair_target)
    ordered = [name for name in targets if name != "planner"]
    if "planner" in targets:
        ordered.append("planner")
    return ordered


def _build_review_result(raw: Any, start: float) -> ReviewResult:
    if not isinstance(raw, dict):
        return _invalid_review("review payload 不是 JSON object", start)
    verdict = str(raw.get("verdict") or "")
    if verdict not in (VERDICT_PASS, VERDICT_REWORK, VERDICT_FAILED):
        return _invalid_review(f"非法 verdict: {verdict!r}", start)
    raw_issues = raw.get("issues")
    if not isinstance(raw_issues, list):
        return _invalid_review("issues 必须是 list", start)
    issues: list[ReviewIssue] = []
    for index, item in enumerate(raw_issues):
        if not isinstance(item, dict):
            return _invalid_review(f"issues[{index}] 不是 object", start)
        severity = str(item.get("severity") or "")
        if severity not in SEVERITIES:
            return _invalid_review(f"issues[{index}] severity 非法: {severity!r}", start)
        issue_type = str(item.get("issue_type") or "").strip()
        description = str(item.get("description") or "").strip()
        evidence = item.get("evidence", [])
        repair_target = str(item.get("repair_target") or "").strip()
        repair_instruction = str(item.get("repair_instruction") or "").strip()
        if not issue_type or not description:
            return _invalid_review(f"issues[{index}] 缺少 issue_type/description", start)
        if evidence is None:
            evidence = []
        elif not isinstance(evidence, list):
            evidence = [evidence]
        if repair_target.lower() in {"none", "null", "n/a", "无"}:
            repair_target = ""
        if repair_target and repair_target not in {
            "attraction",
            "hotel",
            "restaurant",
            "transport",
            "planner",
        }:
            return _invalid_review(f"issues[{index}] repair_target 非法", start)
        if severity == SEVERITY_RECOVERABLE and (not repair_target or not repair_instruction):
            return _invalid_review(f"issues[{index}] recoverable 缺少修复目标或指令", start)
        issues.append(
            ReviewIssue(
                issue_type=issue_type,
                severity=severity,
                description=description,
                evidence=[str(entry) for entry in evidence],
                repair_target=repair_target,
                repair_instruction=repair_instruction,
            )
        )
    if verdict == VERDICT_REWORK and not any(
        issue.severity == SEVERITY_RECOVERABLE for issue in issues
    ):
        return _invalid_review("rework verdict 缺少 recoverable issue", start)
    return ReviewResult(
        verdict=verdict,
        issues=issues,
        token_usage=dict(raw.get("token_usage") or {}),
        duration_ms=_elapsed_ms(start),
    )


def _calibrate_review_result(review: ReviewResult, review_ctx: ReviewContext) -> ReviewResult:
    """Downgrade unsupported semantic guesses before they reach the fail-closed gate.

    The semantic reviewer may notice useful quality concerns, but phrases such as
    “可能/疑似/常识” are not evidence for an automatic repair or a hard failure.
    Likewise, omission of a verification candidate is not omission of a must-visit.
    Deterministic Critic remains authoritative for actual hard-plan violations.
    """
    if review.error:
        return review
    profile = review_ctx.profile_brief or {}
    state = profile.get("constraint_state") if isinstance(profile.get("constraint_state"), dict) else {}
    candidates = [
        str(item).strip()
        for item in (state.get("candidate_attractions") or [])
        if str(item).strip()
    ]
    must_visit = [
        str(item).strip()
        for item in [*list(profile.get("must_visit") or []), *list(state.get("must_visit") or [])]
        if str(item).strip()
    ]
    uncertainty_markers = (
        "可能",
        "疑似",
        "根据领域常识",
        "根据常识",
        "推断",
        "缺乏明确证据",
        "未提供任何证据",
        "未指定",
        "需考虑",
    )
    omission_markers = ("未包含", "未安排", "遗漏", "缺失", "省略", "未提供替代")
    fixed_events = [
        event for event in (state.get("fixed_events") or []) if isinstance(event, dict)
    ]
    plan = review_ctx.plan or {}
    source_ids = {
        str(item)
        for item in plan.get("source_artifact_ids") or []
        if str(item).strip()
    }
    for result in review_ctx.subagent_results:
        for evidence in result.evidence:
            if isinstance(evidence, dict) and evidence.get("artifact_id"):
                source_ids.add(str(evidence["artifact_id"]))
    known_fields = {
        key for key, value in plan.items() if value not in (None, "", [], {})
    } | {"active_constraints", "profile", "constraint_state"}
    contract_fields = {
        "itinerary", "budget_plan", "lodging_plan", "route_evidence",
        "meal_strategy", "validation_result", "critic", "active_constraints",
        "constraint_state", "profile", "return_deadline", "fixed_events",
        "must_visit", "budget_max_cny", "lodging_area", "dietary",
        "transport_mode", "public_transport_required", "mobility",
    }
    mobility_plan = plan.get("mobility_plan") or {}
    mobility_days = [
        item for item in mobility_plan.get("days") or [] if isinstance(item, dict)
    ]
    mobility_limit = mobility_plan.get("max_walking_km_per_day")
    try:
        mobility_limit_value = float(mobility_limit)
    except (TypeError, ValueError):
        mobility_limit_value = None
    bounded_taxi_fallback = bool(
        mobility_plan.get("status") == "bounded_with_taxi_fallback"
        and mobility_limit_value is not None
        and mobility_days
        and all(
            float(day.get("known_walking_km") or 0) <= mobility_limit_value
            and (
                not day.get("unknown_walking_legs")
                or day.get("taxi_fallback_required") is True
            )
            for day in mobility_days
        )
        and (plan.get("critic") or {}).get("passed") is True
    )
    hard_timed_route_state = bool(
        state.get("return_deadline") or state.get("fixed_events")
    )
    required_route_anchors = plan.get("required_route_anchors") or {}
    return_anchor = (
        required_route_anchors.get("last_stop_to_return_location")
        if isinstance(required_route_anchors, dict)
        else None
    ) or {}
    return_anchor_route = (
        return_anchor.get("route")
        if isinstance(return_anchor, dict)
        else None
    ) or {}
    verified_return_anchor = bool(
        isinstance(return_anchor, dict)
        and return_anchor.get("evidence_status") == "provider_verified"
        and return_anchor.get("recommended_latest_departure")
        and isinstance(return_anchor_route, dict)
        and return_anchor_route.get("hard_feasibility_proven") is True
    )
    itinerary_days = [
        day for day in (plan.get("itinerary") or {}).get("days") or []
        if isinstance(day, dict)
    ]
    internal_routes_bound = all(
        isinstance(stop.get("route_from_previous"), dict)
        and stop["route_from_previous"].get("evidence_status") not in {None, "unavailable"}
        for day in itinerary_days
        for index, stop in enumerate(day.get("stops") or [])
        if index > 0 and isinstance(stop, dict)
    )
    origin_anchor = (
        required_route_anchors.get("trip_origin_to_first_stop")
        if isinstance(required_route_anchors, dict)
        else None
    ) or {}
    origin_route_bound = bool(
        not state.get("origin")
        or (
            isinstance(origin_anchor, dict)
            and origin_anchor.get("evidence_status") in {
                "provider_verified", "deterministic_estimate"
            }
            and origin_anchor.get("origin_poi_id")
            and origin_anchor.get("destination_poi_id")
        )
    )
    return_route_bound = bool(
        not state.get("return_location")
        or not state.get("return_deadline")
        or verified_return_anchor
    )
    deterministic_route_bindings_pass = bool(
        (plan.get("critic") or {}).get("passed") is True
        and (plan.get("validation_result") or {}).get("passed") is True
        and internal_routes_bound
        and origin_route_bound
        and return_route_bound
    )
    raw_transport_modes = state.get("transport_modes") or []
    if isinstance(raw_transport_modes, str):
        raw_transport_modes = [raw_transport_modes]
    allowed_transport_modes = {
        str(mode).strip().casefold()
        for mode in raw_transport_modes
        if str(mode).strip()
    }
    validation_checks = {
        str(check).strip()
        for check in (plan.get("validation_result") or {}).get("checks_run") or []
        if str(check).strip()
    }
    hard_mobility_evidence_required = bool(
        state.get("wheelchair_user")
        or state.get("accessibility_priority")
        or state.get("elderly")
        or state.get("max_walking_km_per_day") is not None
        or state.get("walking_time_max_min") is not None
        or state.get("max_single_walk_min") is not None
        or any(
            marker in str(value).casefold()
            for value in (state.get("avoid") or [], state.get("mobility") or [])
            for marker in (
                "台阶", "楼梯", "爬坡", "上坡", "stairs", "hills", "wheelchair"
            )
        )
    )
    hard_internal_accessibility_required = bool(
        state.get("wheelchair_user")
        or state.get("accessibility_priority")
        or any(
            marker in str(value).casefold()
            for value in (state.get("avoid") or [], state.get("mobility") or [])
            for marker in (
                "台阶", "楼梯", "爬坡", "上坡", "stairs", "hills", "wheelchair"
            )
        )
    )

    def evidence_bound(issue: ReviewIssue) -> bool:
        for raw in issue.evidence:
            item = str(raw).strip()
            lowered = item.lower()
            if not item:
                continue
            if any(source_id in item for source_id in source_ids):
                return True
            if lowered.startswith(("tool:", "artifact:", "artifact_id:")):
                return True
            field_binding = re.match(
                r"^(?:profile\.|active_constraints\.|constraint_state\.)?"
                r"([a-z][a-z0-9_]*)(?:\[[^]]+\]|\.[a-z0-9_]+)*\s*(?:=|:)",
                lowered,
            )
            if field_binding and field_binding.group(1) in contract_fields | known_fields:
                return True
            if any(
                re.search(rf"(?:^|[.\[\s]){re.escape(field)}(?:$|[.\[=:\s])", item)
                for field in known_fields
            ):
                return True
            if (plan.get("itinerary") or "days" in plan) and re.search(
                r"(?:itinerary|day\s*\d+|day\d+|第\d+天)", lowered
            ):
                return True
        return False

    for issue in review.issues:
        combined = " ".join([issue.description, *issue.evidence])
        combined_lower = combined.casefold()
        unsupported_guess = any(marker in combined for marker in uncertainty_markers)
        omitted_candidate_only = (
            any(candidate in combined for candidate in candidates)
            and not any(required in combined for required in must_visit)
            and any(marker in combined for marker in omission_markers)
        )
        fixed_event_claim = (
            bool(fixed_events)
            and any(term in combined for term in ("预约", "固定事件", "晚饭", "晚餐", "午饭", "午餐"))
            and any(marker in combined for marker in omission_markers)
            and not any(
                marker in combined_lower
                for marker in ("route_evidence", "路线证据", "transport evidence")
            )
        )
        soft_gap_claim = any(
            marker in combined_lower
            for marker in (
                "空档", "空白", "过于稀疏", "仅安排", "时间浪费",
                "unaccounted gap", "long gap", "six-hour gap", "sparse",
            )
        )
        lodging_claim = any(
            marker in combined_lower
            for marker in ("住宿", "酒店", "lodging", "hotel", "accommodation")
        )
        explicit_lodging = bool(
            profile.get("hotel_area")
            or state.get("lodging_area")
            or state.get("hotel_area")
            or state.get("hotel_budget_per_night_cny") is not None
        )
        unrequested_lodging_gap = lodging_claim and not explicit_lodging
        hard_route_state = any(
            state.get(key) not in (None, "", [], {})
            for key in (
                "return_deadline", "walking_time_max_min",
                "max_single_walk_min", "max_walking_km_per_day",
                "max_transfers_per_day", "accessibility_priority", "wheelchair_user",
            )
        ) or (
            bool(state.get("fixed_events"))
            and any(marker in combined_lower for marker in ("fixed_event", "固定", "预约"))
        )
        hard_timed_route_claim = bool(
            hard_timed_route_state
            and any(
                marker in combined_lower
                for marker in (
                    "return", "deadline", "返程", "返回", "截止",
                    "fixed_event", "固定事件", "固定预约", "预约会场",
                    "required_route_anchors.fixed_event",
                    "required_route_anchors.last_stop_to_return",
                )
            )
        )
        critic_warnings = {
            str(item.get("code") or "")
            for item in (plan.get("critic") or {}).get("issues") or []
            if isinstance(item, dict)
            and str(item.get("severity") or "warning").lower() == "warning"
        }
        issue_type_lower = str(issue.issue_type or "").casefold()
        route_pace_claim = bool(
            "route_too_long" in combined_lower
            or ("route" in issue_type_lower and "pace" in issue_type_lower)
            or any(
                marker in combined_lower
                for marker in (
                    "节奏建议上限", "pace suggested limit", "pace limit",
                )
            )
        )
        soft_route_pace_claim = bool(
            route_pace_claim
            and "route_too_long" in critic_warnings
            and state.get("pace") in (None, "")
            and not hard_timed_route_claim
        )
        soft_route_evidence_gap = (
            (
                "route_evidence" in str(issue.issue_type or "").casefold()
                or any(
                marker in combined_lower
                for marker in ("route_evidence", "路线证据", "transport evidence")
                )
            )
            and (
                not hard_route_state
                or (
                    bounded_taxi_fallback
                    and not hard_timed_route_claim
                )
            )
            and (plan.get("critic") or {}).get("passed") is True
        )
        default_transport_fallback_claim = bool(
            not state.get("public_transport_required")
            and state.get("transport_mode") in (None, "")
            and any(marker in combined_lower for marker in ("taxi", "出租车", "打车"))
            and any(
                marker in combined_lower
                for marker in ("public_transport", "公共交通", "公交")
            )
            and any(
                marker in combined_lower
                for marker in (
                    "违反", "冲突", "禁止", "明确要求",
                    "prohibit", "conflict", "explicitly require",
                )
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
        )
        explicitly_allowed_taxi_claim = bool(
            "taxi" in allowed_transport_modes
            and state.get("taxi_backup") is True
            and any(
                marker in combined_lower
                for marker in ("taxi", "出租车", "打车")
            )
            and any(
                marker in str(issue.issue_type or "").casefold()
                for marker in ("transport", "route")
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
        )
        bounded_mobility_evidence_gap = bool(
            bounded_taxi_fallback
            and not state.get("wheelchair_user")
            and not state.get("accessibility_priority")
            and any(
                token in str(issue.issue_type or "").casefold()
                for token in ("accessibility", "mobility", "elderly", "walking")
            )
            and any(
                marker in combined_lower
                for marker in (
                    "known_walking_km", "unknown_walking_legs", "步行距离未知",
                    "无法确认", "内部步行", "步行距离证据",
                )
            )
        )
        nonrequired_internal_accessibility_claim = bool(
            not hard_internal_accessibility_required
            and (
                "accessibility" in str(issue.issue_type or "").casefold()
                or any(
                    marker in combined_lower
                    for marker in ("无障碍", "台阶", "电梯", "适老", "内部步行")
                )
            )
        )
        deterministic_schedule_advisory = bool(
            any(
                marker in str(issue.issue_type or "").casefold()
                for marker in ("schedule_feasibility", "schedule_conflict", "schedule_gap")
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
            and (
                not hard_timed_route_claim
                or (
                    verified_return_anchor
                    and any(
                        marker in combined_lower
                        for marker in ("返程", "返回", "return", "截止", "车站")
                    )
                    and not any(
                        marker in combined_lower
                        for marker in ("固定预约", "fixed event", "预约冲突")
                    )
                )
            )
        )
        verified_return_feasibility_advisory = bool(
            verified_return_anchor
            and "return" in str(issue.issue_type or "").casefold()
            and "feasibility" in str(issue.issue_type or "").casefold()
            and any(
                marker in combined_lower
                for marker in (
                    "满足", "可在截止前", "截止前到", "before the deadline",
                    "meets the deadline", "within the deadline",
                )
            )
            and not any(
                marker in combined_lower
                for marker in (
                    "无法满足", "不能满足", "超过截止", "晚于截止",
                    "misses the deadline", "after the deadline",
                )
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
        )
        soft_schedule_rhythm_advisory = bool(
            (
                any(
                    marker in str(issue.issue_type or "").casefold()
                    for marker in (
                        "schedule_rhythm", "activity_sparsity", "schedule_density",
                    )
                )
                or any(
                    marker in combined_lower
                    for marker in (
                        "仅包含", "仅安排", "安排较稀疏", "普通节奏不符",
                        "under-filled", "underfilled", "schedule rhythm",
                    )
                )
            )
            and not any(
                marker in combined_lower
                for marker in (
                    "冲突", "重叠", "来不及", "迟到", "超过截止",
                    "conflict", "overlap", "miss the", "deadline violation",
                )
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
        )
        false_missing_route_claim = bool(
            deterministic_route_bindings_pass
            and any(
                marker in combined_lower
                for marker in ("路线证据", "route evidence", "route_evidence", "交通证据")
            )
            and any(
                marker in combined_lower
                for marker in (
                    "缺少", "缺失", "没有", "未提供", "missing",
                    "no route evidence",
                )
            )
        )
        missing_hours_quality_advisory = bool(
            any(
                marker in str(issue.issue_type or "").casefold()
                for marker in ("attraction_quality", "poi_quality", "venue_quality")
            )
            and any(
                marker in combined_lower
                for marker in ("opening_hours", "opening hours", "开放时间", "开放状态")
            )
            and "applicable_opening_hours" in validation_checks
            and not any(
                marker in combined_lower
                for marker in (
                    "明确闭馆", "确认闭馆", "当天闭馆", "不开放",
                    "known closed", "is closed", "closed on",
                )
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
        )
        soft_interest_gap = (
            "interest" in str(issue.issue_type or "").casefold()
            and not any(required in combined for required in must_visit)
        )
        soft_preference_gap = (
            "preference" in str(issue.issue_type or "").casefold()
            and not any(required in combined for required in must_visit)
        )
        budget_plan = plan.get("budget_plan") or {}
        uncertain_budget_band_only = bool(
            budget_plan.get("within_user_limit") is True
            and budget_plan.get("risk_high_exceeds_limit") is True
            and any(
                marker in combined_lower
                for marker in ("risk_high_exceeds_limit", "high risk band", "风险区间", "超预算风险")
            )
        )
        deterministic_budget_advisory = bool(
            "budget" in str(issue.issue_type or "").casefold()
            and budget_plan.get("source_artifact_id")
            and budget_plan.get("within_user_limit") is not False
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
            and not any(
                marker in combined_lower
                for marker in (
                    "超过预算", "超出预算", "合计错误", "算术错误", "未计入",
                    "漏算", "遗漏固定", "exceeds the budget", "arithmetic error",
                    "omitted fixed", "missing fixed cost",
                )
            )
        )
        missing_meal_only = any(
            marker in combined
            for marker in (
                "未安排午餐",
                "未安排晚餐",
                "未安排任何餐饮",
                "没有安排任何餐厅",
                "没有包含任何餐厅",
                "没有任何清真餐厅",
                "缺少餐厅",
            )
        ) and not any(
            marker in combined
            for marker in ("安排了非清真", "安排非清真", "与饮食限制冲突的餐厅")
        )
        false_composite_omission = any((
            bool(plan.get("budget_plan"))
            and any(term in combined for term in ("缺少预算", "未提供预算", "没有预算")),
            bool(plan.get("lodging_plan"))
            and any(term in combined for term in ("缺少住宿", "未提供住宿", "没有住宿", "未安排酒店")),
            bool(
                plan.get("required_route_anchors")
                or plan.get("lodging_route_anchors")
                or plan.get("return_plan")
                or (plan.get("domain_inputs") or {}).get("transport")
            )
            and any(term in combined for term in ("缺少路线", "未提供路线", "没有路线", "缺少交通")),
        ))
        unbound_material_issue = (
            issue.severity in {SEVERITY_CRITICAL, SEVERITY_RECOVERABLE}
            and not evidence_bound(issue)
        )
        self_acknowledged_correctness = bool(
            any(
                marker in combined_lower
                for marker in ("是正确的", "为正确", "is correct", "correct, but")
            )
            and (plan.get("critic") or {}).get("passed") is True
            and (plan.get("validation_result") or {}).get("passed") is True
        )
        if issue.severity != SEVERITY_NONCRITICAL and (
            unsupported_guess
            or omitted_candidate_only
            or fixed_event_claim
            or soft_gap_claim
            or unrequested_lodging_gap
            or soft_route_pace_claim
            or soft_route_evidence_gap
            or default_transport_fallback_claim
            or explicitly_allowed_taxi_claim
            or bounded_mobility_evidence_gap
            or nonrequired_internal_accessibility_claim
            or deterministic_schedule_advisory
            or verified_return_feasibility_advisory
            or soft_schedule_rhythm_advisory
            or false_missing_route_claim
            or missing_hours_quality_advisory
            or soft_interest_gap
            or soft_preference_gap
            or uncertain_budget_band_only
            or deterministic_budget_advisory
            or missing_meal_only
            or false_composite_omission
            or unbound_material_issue
            or self_acknowledged_correctness
        ):
            issue.severity = SEVERITY_NONCRITICAL
            issue.repair_target = ""
            issue.repair_instruction = ""
    if any(issue.severity == SEVERITY_CRITICAL for issue in review.issues):
        review.verdict = VERDICT_FAILED
    elif any(issue.severity == SEVERITY_RECOVERABLE for issue in review.issues):
        review.verdict = VERDICT_REWORK
    else:
        review.verdict = VERDICT_PASS
    return review


def _invalid_review(reason: str, start: float) -> ReviewResult:
    return ReviewResult(
        verdict=VERDICT_FAILED,
        error=f"invalid reviewer output: {reason}",
        duration_ms=_elapsed_ms(start),
    )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
