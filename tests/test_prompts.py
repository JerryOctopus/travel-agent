from __future__ import annotations

from types import SimpleNamespace

from travel_agent.agent.prompts import (
    SYSTEM_BASE,
    build_system_prompt,
    build_system_prompt_sections,
    record_system_prompt_trace,
    sections_to_text,
)
from travel_agent.schemas import TravelProfile
from travel_agent.storage.memory_framework import MemoryFramework


def _memory() -> MemoryFramework:
    return MemoryFramework(
        mode="compressed",
        history_for_prompt=[],
        compressed_summary="用户想去杭州玩 2 天",
        l2_snapshot="- 天气：杭州 晴 28°C",
        l3_snapshot="用户ID：u_001\n长期偏好：美食",
    )


def test_sections_match_build_system_prompt_text():
    """分段装配的拼接结果必须与原 build_system_prompt 完全一致。"""
    profile = TravelProfile(destination="杭州", days=2)
    memory = _memory()

    sections = build_system_prompt_sections(profile, memory, "## 技能清单")

    assert sections_to_text(sections) == build_system_prompt(profile, memory, "## 技能清单")


def test_section_names_and_order():
    profile = TravelProfile(destination="杭州", days=2)

    sections = build_system_prompt_sections(profile, _memory(), "## 技能清单")
    names = [name for name, _ in sections]

    assert names == [
        "base",
        "profile",
        "memory_mode",
        "l1_summary",
        "l2_snapshot",
        "l3_snapshot",
        "skills",
    ]
    assert sections[0][1] == SYSTEM_BASE


def test_record_system_prompt_trace_appends_section_stats():
    ctx = SimpleNamespace(evaluation_trace_enabled=True, evaluation_trace=[])
    sections = build_system_prompt_sections(TravelProfile(), _memory())

    record_system_prompt_trace(ctx, sections, phase="react")

    assert len(ctx.evaluation_trace) == 1
    record = ctx.evaluation_trace[0]
    assert record["kind"] == "prompt"
    assert record["phase"] == "react"
    assert record["content"] == sections_to_text(sections)
    assert record["total_chars"] == len(record["content"])
    assert record["est_tokens"] > 0
    by_name = {item["name"]: item for item in record["sections"]}
    assert by_name["base"]["chars"] == len(SYSTEM_BASE)
    assert all(item["est_tokens"] > 0 for item in record["sections"])


def test_record_system_prompt_trace_noop_when_disabled():
    ctx = SimpleNamespace(evaluation_trace_enabled=False, evaluation_trace=[])
    sections = build_system_prompt_sections(TravelProfile(), None)

    record_system_prompt_trace(ctx, sections, phase="react")

    assert ctx.evaluation_trace == []
