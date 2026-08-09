"""Renderer Gate：Engine 统一渲染入口（Step 3）。

规则（严格执行）：

- Main Orchestrator 不得自行生成行程后直接渲染；渲染只能由 Engine 在
  Planner 产出（并经 Reviewer 判定）后发起；
- ``render_itinerary`` / ``render_map`` 必须显式绑定 planner 产出的
  ``plan_artifact_id``（toolkit.gated_render_* 内做存在性/类型/产出者校验）；
- critic 通过 → 正常渲染；仅非关键 warning → completed_with_warnings；
  critic 未通过或存在未处理 critical → 只渲染 incomplete，不标记成功。
"""

from __future__ import annotations

from typing import Any

from travel_agent.orchestration.multi_agent.schemas import (
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_INCOMPLETE,
)

# 允许发起渲染的交付状态；incomplete 只能以 incomplete 标记渲染。
_RENDERABLE_STATUSES = {
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_INCOMPLETE,
}


def render_plan_outcome(
    ctx: Any,
    plan_artifact_id: str | None,
    delivery_status: str,
    *,
    allowed_agents: frozenset[str] = frozenset({"planner"}),
) -> dict[str, Any]:
    """按交付状态经 Gate 渲染行程与地图；返回渲染结果包。

    - 无 plan / 状态不可渲染 → 不渲染；
    - incomplete → gated_render_*(mark_incomplete=True)；
    - completed / completed_with_warnings → 正常渲染（critic 未过时
      toolkit 层仍会自动降级为 rendered_incomplete）。
    """
    from travel_agent.agent import toolkit

    outcome: dict[str, Any] = {
        "rendered": False,
        "cards": [],
        "map_payload": None,
        "gate_status": "skipped",
        "reason": "",
    }
    if not plan_artifact_id:
        outcome["reason"] = "本轮未产生完整 TravelPlan，跳过渲染"
        return outcome
    if delivery_status not in _RENDERABLE_STATUSES:
        outcome["reason"] = f"交付状态 {delivery_status} 不允许渲染"
        return outcome

    mark_incomplete = delivery_status == STATUS_INCOMPLETE
    itinerary_result = toolkit.gated_render_itinerary(
        ctx,
        plan_artifact_id,
        mark_incomplete=mark_incomplete,
        allowed_agents=allowed_agents,
    )
    if itinerary_result.get("isError"):
        outcome["reason"] = str(itinerary_result.get("summary") or "渲染被 Gate 拒绝")
        outcome["gate_status"] = "rejected"
        return outcome

    map_result = toolkit.gated_render_map(
        ctx,
        plan_artifact_id,
        mark_incomplete=mark_incomplete,
        allowed_agents=allowed_agents,
    )
    outcome.update(
        {
            "rendered": True,
            "cards": itinerary_result.get("cards") or [],
            "map_payload": (
                {
                    key: value
                    for key, value in map_result.items()
                    if key not in ("summary", "isError")
                }
                if not map_result.get("isError")
                else None
            ),
            "gate_status": itinerary_result.get("gate_status") or "rendered",
        }
    )
    return outcome
