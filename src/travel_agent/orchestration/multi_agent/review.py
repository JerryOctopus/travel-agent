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
        plan = {
            key: self.plan.get(key)
            for key in (
                "itinerary",
                "critic",
                "revision_notes",
                "final_issue_count",
                "source_artifact_ids",
                "revision_directives",
            )
            if key in self.plan
        }
        if isinstance(plan.get("itinerary"), dict):
            plan["itinerary"] = _compact_itinerary(plan["itinerary"])
        payload = {
            "request_id": self.request_id,
            "task_brief": self.task_brief,
            "profile": self.profile_brief,
            # Do not duplicate original_itinerary or full domain payloads.  The
            # reviewer needs the final plan plus evidence identities, and some
            # providers enforce a ~30k input-character boundary.
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
只基于给定的 TravelPlan、用户画像与领域调研证据，检查：软偏好匹配、行程节奏、
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
  最多是行程完整性优化。只有计划明确安排了与 dietary 冲突的餐厅或菜系，才能判硬冲突；
- 住宿由 hotel 领域 evidence、住宿区域与补充卡片承载，不要求把酒店塞进 itinerary.days[].stops；
  酒店没有作为每日游玩停靠点本身不是缺失，也不得据此要求 Planner 添加酒店 stop；
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
    severities = {
        str(item.get("severity") or "")
        for item in issues
        if isinstance(item, dict)
    }
    if verdict == VERDICT_REWORK:
        return SEVERITY_RECOVERABLE in severities and SEVERITY_CRITICAL not in severities
    return SEVERITY_CRITICAL in severities


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
        "未指定",
        "需考虑",
    )
    omission_markers = ("未包含", "未安排", "遗漏", "缺失", "未提供替代")
    fixed_events = [
        event for event in (state.get("fixed_events") or []) if isinstance(event, dict)
    ]
    for issue in review.issues:
        combined = " ".join([issue.description, *issue.evidence])
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
        )
        soft_gap_claim = "空档" in combined or "时间浪费" in combined
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
        if issue.severity != SEVERITY_NONCRITICAL and (
            unsupported_guess
            or omitted_candidate_only
            or fixed_event_claim
            or soft_gap_claim
            or missing_meal_only
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
