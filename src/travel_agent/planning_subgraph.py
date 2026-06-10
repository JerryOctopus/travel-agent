"""可控规划子图 ``plan_and_critique``（项目差异化核心，对应 PROJECT_PLAN M2）。

外层 ReAct agent 决定「何时」规划；本子图保证「规划质量可控、可评估、可复现」：
内部用 LangGraph ``StateGraph`` 把现有 planner / critic / reviser 编排成

    plan -> critic -> [passed? END : revise -> critic -> ...]

的有限闭环（最多 ``max_iters`` 轮），输出约束满足的结构化行程，并保留
「原始稿 vs 修正稿」对比，量化 critic 的价值。

这里刻意保持确定性（不调用 LLM）：什么样的行程算可行、约束有没有满足，必须
可控、可评估，而不是交给 LLM 自由发挥。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from travel_agent.critic import critique_itinerary
from travel_agent.planning import build_simple_itinerary
from travel_agent.reviser import revise_itinerary
from travel_agent.schemas import (
    CriticResult,
    Itinerary,
    ScoredPOI,
    TravelProfile,
)

try:
    from langgraph.graph import END, START, StateGraph

    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - 依赖未安装时回退为纯函数实现
    _HAS_LANGGRAPH = False


DEFAULT_MAX_ITERS = 3


class PlanState(TypedDict, total=False):
    ranked_pois: list[ScoredPOI]
    profile: TravelProfile
    route_estimator: Any
    itinerary: Itinerary
    original_itinerary: Itinerary
    critic_result: CriticResult
    revision_notes: list[str]
    iteration: int
    max_iters: int


@dataclass(frozen=True)
class PlanAndCritiqueResult:
    itinerary: Itinerary
    original_itinerary: Itinerary
    critic_result: CriticResult
    original_issue_count: int = 0
    revision_notes: list[str] = field(default_factory=list)
    iterations: int = 0

    @property
    def revised(self) -> bool:
        return bool(self.revision_notes)

    @property
    def final_issue_count(self) -> int:
        return len(self.critic_result.issues)


def _plan_node(state: PlanState) -> PlanState:
    itinerary = build_simple_itinerary(
        ranked_pois=state["ranked_pois"],
        profile=state["profile"],
        route_estimator=state.get("route_estimator"),
    )
    return {"itinerary": itinerary, "original_itinerary": itinerary, "iteration": 0}


def _critic_node(state: PlanState) -> PlanState:
    result = critique_itinerary(state["itinerary"], state["profile"])
    return {"critic_result": result}


def _revise_node(state: PlanState) -> PlanState:
    revised, result, notes = revise_itinerary(
        itinerary=state["itinerary"],
        ranked_pois=state["ranked_pois"],
        profile=state["profile"],
        critic_result=state["critic_result"],
    )
    accumulated = list(state.get("revision_notes", [])) + notes
    return {
        "itinerary": revised,
        "critic_result": result,
        "revision_notes": accumulated,
        "iteration": state.get("iteration", 0) + 1,
    }


def _should_revise(state: PlanState) -> str:
    critic_result = state.get("critic_result")
    if critic_result is not None and critic_result.passed:
        return "done"
    if state.get("iteration", 0) >= state.get("max_iters", DEFAULT_MAX_ITERS):
        return "done"
    return "revise"


def _build_graph():
    graph = StateGraph(PlanState)
    graph.add_node("plan", _plan_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("revise", _revise_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "critic")
    graph.add_conditional_edges(
        "critic",
        _should_revise,
        {"revise": "revise", "done": END},
    )
    graph.add_edge("revise", "critic")
    return graph.compile()


_COMPILED_GRAPH = _build_graph() if _HAS_LANGGRAPH else None


def _run_plain(state: PlanState) -> PlanState:
    """无 langgraph 时的等价纯函数实现，保证行为一致、可离线测试。"""
    state = {**state, **_plan_node(state)}
    state = {**state, **_critic_node(state)}
    while _should_revise(state) == "revise":
        # revise_node 内部已重算 critic_result
        state = {**state, **_revise_node(state)}
    return state


def plan_and_critique(
    ranked_pois: list[ScoredPOI],
    profile: TravelProfile,
    route_estimator: Any | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> PlanAndCritiqueResult:
    """运行可控规划子图，返回带 critic 闭环结果的结构化行程。"""
    initial: PlanState = {
        "ranked_pois": ranked_pois,
        "profile": profile,
        "route_estimator": route_estimator,
        "revision_notes": [],
        "iteration": 0,
        "max_iters": max_iters,
    }
    if _COMPILED_GRAPH is not None:
        final: PlanState = _COMPILED_GRAPH.invoke(initial)
    else:
        final = _run_plain(initial)

    original = final["original_itinerary"]
    original_issues = critique_itinerary(original, profile).issues
    return PlanAndCritiqueResult(
        itinerary=final["itinerary"],
        original_itinerary=original,
        critic_result=final["critic_result"],
        original_issue_count=len(original_issues),
        revision_notes=list(final.get("revision_notes", [])),
        iterations=final.get("iteration", 0),
    )
