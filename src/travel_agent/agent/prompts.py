"""Agent 系统提示词（动态注入会话画像快照）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from travel_agent.schemas import TravelProfile

if TYPE_CHECKING:
    from travel_agent.storage.memory_framework import MemoryFramework

SYSTEM_BASE = """你是一个专业的中文旅行规划助手，通过自主调用工具来完成任务。

可用工具与典型编排顺序：
1. update_travel_profile：把用户提到的目的地/天数/偏好等记录进画像；
2. 若缺少目的地或天数等必要信息，调用 request_travel_info 向用户追问，并停止后续规划；
3. search_poi：检索目的地候选 POI；check_weather：查询天气；
4. recommend_candidates：对候选做打分与多样性重排；
5. plan_and_critique：运行可控规划子图，产出约束满足的行程（这是规划质量的保证）；
6. render_itinerary 和 render_map：生成前端可渲染的卡片与地图数据。

原则：
- 你自己决定何时调用哪个工具、是否需要追问，不要凭空编造 POI 或行程；
- 信息齐全时，务必依次完成 search_poi → recommend_candidates → plan_and_critique → render_*；
- 规划必须通过 plan_and_critique 产出，不要自己手写行程文本冒充规划结果；
- 最终用简洁中文向用户总结：目的地、天数、行程亮点，以及 critic 是否通过/做了哪些修正。
"""


def build_system_prompt(
    profile: TravelProfile,
    memory: MemoryFramework | None = None,
    skills_section: str = "",
) -> str:
    snapshot = _profile_snapshot(profile)
    parts = [SYSTEM_BASE, f"\n当前已知出行画像（多轮累积）：\n{snapshot}"]
    if memory is not None:
        parts.append(f"\n【记忆模式：{memory.mode}】")
        if memory.compressed_summary:
            parts.append(f"\nL1 历史摘要：\n{memory.compressed_summary}")
        parts.append(f"\nL2 本会话工具快照：\n{memory.l2_snapshot}")
        parts.append(f"\nL3 跨会话用户画像：\n{memory.l3_snapshot}")
    if skills_section:
        parts.append(f"\n{skills_section}")
    return "\n".join(parts)


def _profile_snapshot(profile: TravelProfile) -> str:
    if not profile.destination and not profile.days and not profile.interests:
        return "- 暂无，需要从用户输入中抽取。"
    lines = []
    if profile.destination:
        lines.append(f"- 目的地：{profile.destination}")
    if profile.days:
        lines.append(f"- 天数：{profile.days}")
    if profile.interests:
        lines.append(f"- 偏好：{', '.join(profile.interests)}")
    if profile.budget_level:
        lines.append(f"- 预算：{profile.budget_level}")
    if profile.pace:
        lines.append(f"- 节奏：{profile.pace}")
    if profile.must_visit:
        lines.append(f"- 必去：{', '.join(profile.must_visit)}")
    if profile.avoid:
        lines.append(f"- 避开：{', '.join(profile.avoid)}")
    missing = profile.missing_required_fields()
    if missing:
        lines.append(f"- 仍缺少：{', '.join(missing)}")
    return "\n".join(lines)
