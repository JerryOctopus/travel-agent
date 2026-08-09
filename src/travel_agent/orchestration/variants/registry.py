"""V0–V3 架构消融实验：变体注册表与统一入口（Step 4 收敛版）。

四个版本**共用同一个** ``MultiAgentEngine``，差异只体现在
``EngineCapabilities``（编排模式/派工方式/Reviewer 开关/返工上限）：

- v0：单 Agent（mode=single，复用 runtime._run_react 全工具链路）；
- v1：确定性规则派工（dispatch=fixed，无 LLM 路由）；
- v2：动态 Main Orchestrator，无 Reviewer；
- v3 = Production Full：动态 Orchestrator + Reviewer + 最多一次修复周期。

本模块不再保留任何独立 StateGraph / Supervisor / Worker / Critic 实现；
Evaluation/Harness 只能通过 ``run_variant_turn`` 显式选择版本，
生产入口（``runtime.run_production_turn``）固定使用 PRODUCTION_CONFIG。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from travel_agent.agent.runtime import AgentReply
from travel_agent.agent.session import SessionContext
from travel_agent.orchestration.multi_agent import (
    VARIANT_PRESETS,
    EngineCapabilities,
)
from travel_agent.settings import Settings


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    capabilities: EngineCapabilities


_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "v0": "单 Agent 基线：一个 ReAct agent 自主编排全部工具，判断多智能体是否真的带来提升。",
    "v1": "确定性规则派工多 Agent：task_type 查表生成批次任务，无 LLM 路由，"
    "判断简单任务拆分是否已经足够。",
    "v2": "动态 Main Orchestrator（无 Reviewer）：由 Orchestrator 动态决定派工与停止，"
    "验证动态路由与任务分解本身的收益。",
    "v3": "完整多智能体系统（= Production Full）：动态 Orchestrator + 语义 Reviewer，"
    "最多一次定向修复周期。",
}

VARIANTS: dict[str, VariantSpec] = {
    name: VariantSpec(
        name=name,
        description=_VARIANT_DESCRIPTIONS[name],
        capabilities=capabilities,
    )
    for name, capabilities in VARIANT_PRESETS.items()
}


def run_variant_turn(
    variant: str,
    user_message: str,
    ctx: SessionContext,
    settings: Settings,
    history: list[tuple[str, str]],
    user_id: str,
    **kwargs: Any,
) -> AgentReply:
    """按 variant 执行一轮对话；所有版本共用同一 Engine 与同一 Token 硬上限审计。"""
    key = str(variant or "").strip().lower()
    spec = VARIANTS.get(key)
    if spec is None:
        raise ValueError(
            f"未知的消融实验变体: {variant!r}，可用: {sorted(VARIANTS)}"
        )
    trace_start = len(ctx.evaluation_trace) if ctx.evaluation_trace_enabled else 0
    reply = _run_variant_turn(spec, user_message, ctx, settings, history, user_id)
    _audit_token_budget(ctx, settings, key, trace_start, reply)
    return reply


def _run_variant_turn(
    spec: VariantSpec,
    user_message: str,
    ctx: SessionContext,
    settings: Settings,
    history: list[tuple[str, str]],
    user_id: str,
) -> AgentReply:
    """薄适配：只依赖已发布 runtime 接口，然后交给同一 Engine。"""
    from travel_agent.agent.runtime import run_architecture_turn

    reply = run_architecture_turn(
        spec.capabilities,
        user_message,
        ctx,
        history,
        settings,
        user_id,
    )
    _record_reply_metrics(ctx, spec, reply)
    return reply


def _record_reply_metrics(ctx: SessionContext, spec: VariantSpec, reply: AgentReply) -> None:
    previous = ctx.store.latest("variant_metrics") or {}
    ctx.store.put(
        "variant_metrics",
        {
            **previous,
            "variant": spec.name,
            "mode": spec.capabilities.mode,
            "dispatch": spec.capabilities.dispatch,
            "status": getattr(reply, "status", None),
        },
    )


def _record_engine_metrics(ctx: SessionContext, spec: VariantSpec, outcome: Any) -> None:
    """把本轮 Engine 编排结论写入 variant_metrics，供消融报告使用。"""
    previous = ctx.store.latest("variant_metrics") or {}
    metrics = {
        **previous,
        "variant": spec.name,
        "mode": spec.capabilities.mode,
        "dispatch": spec.capabilities.dispatch,
        "status": outcome.status,
        "rework_used": outcome.rework_used,
    }
    if outcome.review is not None:
        metrics["review"] = {
            "verdict": outcome.review.verdict,
            "delivery": outcome.status,
            "rework_used": outcome.rework_used,
        }
    ctx.store.put("variant_metrics", metrics)


def _audit_token_budget(
    ctx: SessionContext,
    settings: Settings,
    variant: str,
    trace_start: int,
    reply: AgentReply,
) -> None:
    """四版本统一的 Token 硬上限审计。

    各版本在编排内部已做硬停止；这里做轮末总量审计：超限时把该 case
    标记 budget_exceeded，保证成本边界对所有版本同样“硬”。
    """
    budget = settings.orchestration.variant_token_budget
    previous = ctx.store.latest("variant_metrics") or {}
    metrics = {
        **previous,
        "variant": variant,
    }
    if not budget or budget <= 0:
        ctx.store.put("variant_metrics", metrics)
        return
    tokens_used = 0
    for record in ctx.evaluation_trace[trace_start:]:
        if record.get("kind") == "model" and record.get("total_tokens") is not None:
            tokens_used += int(record["total_tokens"])
    exceeded = tokens_used >= budget
    metrics.update(
        {
            "tokens_used": tokens_used,
            "token_budget": budget,
            "budget_exceeded": exceeded,
        }
    )
    ctx.store.put("variant_metrics", metrics)
    if exceeded and not getattr(reply, "failure_reason", None):
        # AgentReply 在较老 runtime 快照中没有该字段；dataclass 非 slots，兼容附加。
        setattr(reply, "failure_reason", "budget_exceeded")
