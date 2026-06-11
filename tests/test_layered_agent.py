from __future__ import annotations

from travel_agent.agent.session import build_session
from travel_agent.orchestration.layered_agent import LayerMetrics, run_layered_turn
from travel_agent.settings import Settings, OrchestrationSettings


def _layered_settings() -> Settings:
    base = Settings()
    return Settings(
        llm=base.llm,
        amap=base.amap,
        agent=base.agent,
        memory=base.memory,
        orchestration=OrchestrationSettings(layered_enabled=True, max_layer_retries=1),
        skills=base.skills,
    )


def test_layered_turn_completes_pipeline(offline_settings):
    ctx = build_session(persist=False)
    reply = run_layered_turn(
        "帮我规划杭州两天，喜欢自然和美食，轻松一点",
        ctx,
        _layered_settings(),
    )
    assert "plan_and_critique" in reply.tool_trace
    assert "render_map" in reply.tool_trace
    assert reply.map_payload is not None
    metrics = ctx.store.latest("layer_metrics")
    assert metrics is not None
    assert metrics["completed"] is True


def test_layer_metrics_rates():
    m = LayerMetrics()
    m.record_attempt("requirement")
    m.record_success("requirement")
    m.record_attempt("research")
    m.record_rollback()
    assert m.layer_hit_rate == 0.5
    assert m.rollback_rate == 0.5
