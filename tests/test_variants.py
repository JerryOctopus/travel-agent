"""V0–V3 架构消融收敛测试（Step 4）。

四个版本共用同一个 ``MultiAgentEngine``，variants 只是薄适配器：

- registry 四版本齐备、描述非空；非法 variant 直接报错；
- run_variant_turn 按 variant 选择 EngineCapabilities，但构造的是同一个 Engine；
- 生产入口固定 PRODUCTION_CONFIG，不读取任何 variant；
- variants 包不再保留独立 StateGraph / Supervisor / Worker / Critic 实现；
- 离线场景与生产共用 deterministic fallback；
- Harness 能通过 environment.variant 显式运行 V0–V3。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from travel_agent.agent.runtime import AgentReply, _run_multi_agent
from travel_agent.agent.session import build_session
from travel_agent.agent.turn_analysis import MessageKind, TaskType
from travel_agent.harness import AgentHarness, HarnessCase, HarnessEnvironment
from travel_agent.orchestration.multi_agent import (
    FULL_CONFIG,
    PRODUCTION_CONFIG,
    V3_CONFIG,
    VARIANT_PRESETS,
    EngineCapabilities,
    MultiAgentEngine,
)
from travel_agent.orchestration.variants import VARIANTS, run_variant_turn
from travel_agent.settings import Settings, OrchestrationSettings, LLMSettings


def _llm_settings() -> Settings:
    """启用 LLM 的最小配置，保证 variant 路径不会走离线兜底。"""
    return Settings(llm=LLMSettings(provider="openai", api_key="test-key"))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_contains_all_four_variants_with_descriptions():
    assert set(VARIANTS) == {"v0", "v1", "v2", "v3"}
    for name, spec in VARIANTS.items():
        assert spec.name == name
        assert spec.description.strip(), f"{name} 描述不能为空"
        assert isinstance(spec.capabilities, EngineCapabilities)
        assert spec.capabilities is VARIANT_PRESETS[name]


def test_variant_capabilities_match_step4_semantics():
    v0, v1, v2, v3 = (VARIANTS[name].capabilities for name in ("v0", "v1", "v2", "v3"))
    # V0 单 Agent；V1 确定性规则派工；V2 动态编排无 Reviewer；V3 = Production Full。
    assert v0.mode == "single"
    assert v1.dispatch == "fixed" and not v1.reviewer_enabled
    assert v2.dispatch == "dynamic" and not v2.reviewer_enabled
    assert v3 is PRODUCTION_CONFIG is FULL_CONFIG is V3_CONFIG
    assert v3.dispatch == "dynamic" and v3.reviewer_enabled and v3.max_rework == 1


def test_run_variant_turn_rejects_unknown_variant():
    ctx = build_session(persist=False)
    with pytest.raises(ValueError, match="未知的消融实验变体"):
        run_variant_turn("v9", "你好", ctx, Settings(), [], "test_user")


# --------------------------------------------------------------------------- #
# 同一 Engine + 生产入口固定
# --------------------------------------------------------------------------- #
def _fake_analysis():
    return SimpleNamespace(kind=MessageKind.TRAVEL, task_type=TaskType.FULL_TRIP_PLAN)


class _CapturingEngine:
    """记录构造时收到的 capabilities，run_turn 返回最小可用 outcome。"""

    captured: list[EngineCapabilities] = []

    def __init__(self, capabilities: EngineCapabilities | None = None) -> None:
        self.capabilities = capabilities
        _CapturingEngine.captured.append(capabilities)

    def run_turn(self, *args, **kwargs):
        return SimpleNamespace(status="completed", rework_used=0, review=None)


def test_all_variants_route_through_the_same_engine_class():
    _CapturingEngine.captured = []
    for variant in ("v0", "v1", "v2", "v3"):
        ctx = build_session(persist=False)
        with (
            patch(
                "travel_agent.agent.turn_analysis.analyze_travel_turn",
                return_value=_fake_analysis(),
            ),
            patch(
                "travel_agent.agent.runtime._reply_from_outcome",
                return_value=AgentReply(text="ok"),
            ),
            patch(
                "travel_agent.orchestration.multi_agent.MultiAgentEngine",
                _CapturingEngine,
            ),
        ):
            reply = run_variant_turn(
                variant, "杭州三天", ctx, _llm_settings(), [], "test_user"
            )
        assert reply.text == "ok"

    # 四版本构造的都是同一 Engine 类，仅 capabilities 不同。
    assert len(_CapturingEngine.captured) == 4
    assert _CapturingEngine.captured == [
        VARIANT_PRESETS[name] for name in ("v0", "v1", "v2", "v3")
    ]


def test_production_entry_is_fixed_to_production_config():
    # 编排配置只剩消融共用的 Token 预算字段，产品入口无法切到 V0–V2。
    field_names = {field.name for field in dataclasses.fields(OrchestrationSettings)}
    assert field_names == {"variant_token_budget"}

    _CapturingEngine.captured = []
    ctx = build_session(persist=False)
    with (
        patch(
            "travel_agent.orchestration.multi_agent.MultiAgentEngine",
            _CapturingEngine,
        ),
        patch(
            "travel_agent.agent.runtime._reply_from_outcome",
            return_value=AgentReply(text="ok"),
        ),
    ):
        reply = _run_multi_agent("杭州三天", ctx, [], Settings(), _fake_analysis())

    assert reply.text == "ok"
    assert _CapturingEngine.captured == [PRODUCTION_CONFIG]


def test_variants_package_has_no_independent_graphs():
    """V0–V3 不得保留独立 StateGraph / Supervisor / Worker / Critic 实现。"""
    import travel_agent.orchestration.variants as variants_pkg

    package_dir = Path(variants_pkg.__file__).resolve().parent
    sources = "".join(
        path.read_text(encoding="utf-8") for path in sorted(package_dir.glob("*.py"))
    )
    # 检查的是实现特征（导入/构造/定义），而非文档措辞。
    forbidden = (
        "from langgraph",
        "import langgraph",
        "StateGraph(",
        "class Supervisor",
        "class Worker",
        "class Critic",
    )
    for pattern in forbidden:
        assert pattern not in sources, f"variants 包仍包含独立编排实现: {pattern}"
    import re

    assert not re.search(r"def build_v\d_graph", sources), "variants 包仍存在独立 Graph 构建函数"


# --------------------------------------------------------------------------- #
# 离线 fallback 与 Harness 显式版本选择
# --------------------------------------------------------------------------- #
def test_run_variant_turn_offline_shares_production_fallback(offline_settings):
    for variant in ("v0", "v3"):
        ctx = build_session(persist=False)
        reply = run_variant_turn(
            variant, "帮我规划杭州三天", ctx, offline_settings, [], "test_user"
        )
        # 离线与生产同一条确定性兜底链路：能交付行程，且不宣称真实 Agent。
        assert reply.text
        assert reply.used_real_agent is False
        assert ctx.store.latest("itinerary") is not None


def test_harness_runs_explicit_variants_offline(offline_settings):
    case = HarnessCase(
        case_id="variant_case",
        turns=["杭州三天随便"],
        expected_city="杭州",
        expected_days=3,
        expect_itinerary=True,
    )
    for variant in ("v0", "v1", "v2", "v3"):
        harness = AgentHarness(
            settings=offline_settings,
            environment=HarnessEnvironment(mode="offline", variant=variant),
        )
        result = harness.run_case(case)
        assert result.errors == [], f"{variant}: {result.errors}"
        assert "itinerary" in result.final_artifacts


def test_harness_unknown_variant_surfaces_error(offline_settings):
    harness = AgentHarness(
        settings=offline_settings,
        environment=HarnessEnvironment(mode="offline", variant="v9"),
    )
    result = harness.run_case(
        HarnessCase(case_id="bad_variant", turns=["杭州三天随便"])
    )
    assert result.errors
    assert "未知的消融实验变体" in result.errors[0]


def test_multi_agent_engine_default_is_production_config():
    # Engine 未显式指定配置时即生产 Full（V3），与显式 PRODUCTION_CONFIG 等价。
    assert MultiAgentEngine().capabilities is PRODUCTION_CONFIG
