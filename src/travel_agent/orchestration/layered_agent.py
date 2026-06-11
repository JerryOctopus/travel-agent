"""五层编排：requirement → research → planning → risk → render（M5）。

默认关闭；开启后按层执行 toolkit、层校验失败则回滚 checkpoint 重试，
并产出 layer_hit_rate / rollback_rate 等指标落盘为 artifact。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from travel_agent.agent import toolkit
from travel_agent.agent.runtime import AgentReply, _reply_from_store
from travel_agent.agent.session import SessionContext
from travel_agent.settings import Settings
from travel_agent.workflow_rules import extract_profile_rule_based

LayerName = str

LAYER_ORDER: list[LayerName] = ["requirement", "research", "planning", "risk", "render"]

LAYER_TOOLS: dict[LayerName, list[str]] = {
    "requirement": ["update_travel_profile", "request_travel_info"],
    "research": ["search_poi", "check_weather", "plan_route"],
    "planning": ["recommend_candidates", "plan_and_critique"],
    "risk": ["plan_and_critique"],
    "render": ["render_itinerary", "render_map"],
}


@dataclass
class LayerMetrics:
    layer_attempts: dict[str, int] = field(default_factory=dict)
    layer_successes: dict[str, int] = field(default_factory=dict)
    rollbacks: int = 0
    completed: bool = False

    def record_attempt(self, layer: str) -> None:
        self.layer_attempts[layer] = self.layer_attempts.get(layer, 0) + 1

    def record_success(self, layer: str) -> None:
        self.layer_successes[layer] = self.layer_successes.get(layer, 0) + 1

    def record_rollback(self) -> None:
        self.rollbacks += 1

    @property
    def layer_hit_rate(self) -> float | None:
        if not self.layer_attempts:
            return None
        hits = sum(self.layer_successes.get(layer, 0) for layer in LAYER_ORDER)
        attempts = sum(self.layer_attempts.values())
        return round(hits / attempts, 4) if attempts else None

    @property
    def rollback_rate(self) -> float | None:
        attempts = sum(self.layer_attempts.values())
        if attempts == 0:
            return None
        return round(self.rollbacks / attempts, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_hit_rate": self.layer_hit_rate,
            "rollback_rate": self.rollback_rate,
            "rollbacks": self.rollbacks,
            "completed": self.completed,
            "layer_attempts": self.layer_attempts,
            "layer_successes": self.layer_successes,
        }


def run_layered_turn(
    user_message: str,
    ctx: SessionContext,
    settings: Settings,
) -> AgentReply:
    metrics = LayerMetrics()
    trace: list[str] = []
    checkpoint = _capture_checkpoint(ctx)
    clarification = False
    text = ""

    for attempt in range(settings.orchestration.max_layer_retries + 1):
        failed_layer: str | None = None
        for layer in LAYER_ORDER:
            metrics.record_attempt(layer)
            ok, layer_trace, layer_text, layer_clarify = _run_layer(layer, ctx, user_message)
            trace.extend(layer_trace)
            if layer_text:
                text = layer_text
            if layer_clarify:
                clarification = True
            if ok:
                metrics.record_success(layer)
                continue
            failed_layer = layer
            metrics.record_rollback()
            _restore_checkpoint(ctx, checkpoint)
            break
        else:
            metrics.completed = True
            break
        if failed_layer is None:
            break
    else:
        text = text or "分层编排未能在重试次数内完成，请简化需求后重试。"

    ctx.store.put("layer_metrics", metrics.to_dict())
    if metrics.completed and not clarification:
        return _reply_from_store(ctx, text or "分层编排已完成。", trace, used_real_agent=False)

    if clarification:
        return AgentReply(
            text=text,
            tool_trace=trace,
            used_real_agent=False,
            clarification=True,
            profile=toolkit._profile_brief(ctx.profile),
        )
    return _reply_from_store(ctx, text, trace, used_real_agent=False)


def _run_layer(
    layer: LayerName,
    ctx: SessionContext,
    user_message: str,
) -> tuple[bool, list[str], str, bool]:
    trace: list[str] = []
    text = ""
    clarify = False

    if layer == "requirement":
        extracted = extract_profile_rule_based(user_message)
        toolkit.update_travel_profile(
            ctx,
            destination=extracted.destination,
            days=extracted.days,
            interests=extracted.interests,
            budget_level=extracted.budget_level,
            pace=extracted.pace,
            companions=extracted.companions,
        )
        trace.append("update_travel_profile")
        missing = ctx.profile.missing_required_fields()
        if missing:
            info = toolkit.request_travel_info(ctx, missing)
            trace.append("request_travel_info")
            return True, trace, info["question"], True
        return True, trace, "", False

    if layer == "research":
        search = toolkit.search_poi(ctx)
        trace.append("search_poi")
        if search.get("isError"):
            return False, trace, search["summary"], False
        weather = toolkit.check_weather(ctx)
        trace.append("check_weather")
        if weather.get("isError"):
            return False, trace, weather["summary"], False
        return True, trace, "", False

    if layer == "planning":
        rec = toolkit.recommend_candidates(ctx)
        trace.append("recommend_candidates")
        if rec.get("isError"):
            return False, trace, rec["summary"], False
        plan = toolkit.plan_and_critique(ctx)
        trace.append("plan_and_critique")
        if plan.get("isError"):
            return False, trace, plan["summary"], False
        text = plan["summary"]
        return True, trace, text, False

    if layer == "risk":
        pack = ctx.store.latest("itinerary")
        if not pack:
            return False, trace, "缺少行程，无法做 risk 层校验。", False
        critic = pack.get("critic", {})
        issues = critic.get("issues", [])
        errors = [i for i in issues if i.get("severity") == "error"]
        if errors:
            return False, trace, f"risk 层未通过：{errors[0].get('message', '存在 error 级违规')}", False
        return True, trace, "", False

    if layer == "render":
        cards = toolkit.render_itinerary(ctx)
        trace.append("render_itinerary")
        if cards.get("isError"):
            return False, trace, cards["summary"], False
        maps = toolkit.render_map(ctx)
        trace.append("render_map")
        if maps.get("isError"):
            return False, trace, maps["summary"], False
        pack = ctx.store.latest("itinerary") or {}
        text = pack.get("itinerary", {}).get("summary", "行程已渲染。")
        return True, trace, text, False

    return False, trace, f"未知层：{layer}", False


def _capture_checkpoint(ctx: SessionContext) -> dict[str, Any]:
    return {
        "profile": copy.deepcopy(ctx.profile),
        "pois_by_id": copy.deepcopy(ctx.pois_by_id),
        "store_items": copy.deepcopy(ctx.store._items),
    }


def _restore_checkpoint(ctx: SessionContext, checkpoint: dict[str, Any]) -> None:
    ctx.profile = checkpoint["profile"]
    ctx.pois_by_id = checkpoint["pois_by_id"]
    ctx.store._items = copy.deepcopy(checkpoint["store_items"])
