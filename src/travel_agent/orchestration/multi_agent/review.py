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
        payload = {
            "request_id": self.request_id,
            "task_brief": self.task_brief,
            "profile": self.profile_brief,
            "plan": self.plan,
            "subagent_results": [
                {
                    "task_id": result.task_id,
                    "agent": result.agent,
                    "status": result.status,
                    "payload": result.payload,
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

_REVIEWER_INSTRUCTION = """你是行程语义复核者（Semantic Reviewer）。你不拥有任何工具，也不重新生成计划。
只基于给定的 TravelPlan、用户画像与领域调研证据，检查：软偏好匹配、行程节奏、
适老性/无障碍、跨领域一致性（交通耗时与景点安排、预算与档位等）。
对每个问题输出 JSON：issue_type / severity(critical|recoverable|noncritical) /
description / evidence / repair_target(attraction|hotel|restaurant|transport|planner) /
repair_instruction。整体给出 verdict：pass | rework | failed。"""


def run_semantic_review(
    review_ctx: ReviewContext,
    *,
    review_callable: ReviewCallable | None = None,
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
    try:
        raw = review_callable(review_ctx)
    except Exception as exc:  # noqa: BLE001 — 包装为 failed，不上抛
        return ReviewResult(
            verdict=VERDICT_FAILED,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=_elapsed_ms(start),
        )
    return _build_review_result(raw, start)


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


def build_review_callable(settings: Any, model: Any | None = None) -> ReviewCallable | None:
    """构建真实的单层无工具 LLM Review 调用；LLM 不可用时返回 None。

    返回的 callable 只读 ReviewContext，直接调用模型一次，不挂载任何工具。
    """
    if model is None and not settings.llm.enabled:
        return None

    def review_callable(review_ctx: ReviewContext) -> dict[str, Any]:
        active_model = model
        if active_model is None:
            from travel_agent.agent.runtime import _build_chat_model

            active_model = _build_chat_model(settings)
        from langchain_core.messages import HumanMessage

        from travel_agent.orchestration.meter import meter_callbacks

        response = active_model.invoke(
            [HumanMessage(content=reviewer_prompt(review_ctx))],
            config={"callbacks": meter_callbacks("reviewer")},
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return extract_json_payload(content)

    return review_callable


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
        if not isinstance(evidence, list):
            return _invalid_review(f"issues[{index}] evidence 必须是 list", start)
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


def _invalid_review(reason: str, start: float) -> ReviewResult:
    return ReviewResult(
        verdict=VERDICT_FAILED,
        error=f"invalid reviewer output: {reason}",
        duration_ms=_elapsed_ms(start),
    )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
