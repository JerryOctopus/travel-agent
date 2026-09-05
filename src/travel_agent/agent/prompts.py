"""Agent 系统提示词（动态注入会话画像快照）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from travel_agent.schemas import TravelProfile
from travel_agent.workflow_rules import (
    format_budget_display,
    format_interests_display,
    format_pace_display,
)

if TYPE_CHECKING:
    from travel_agent.storage.memory_framework import MemoryFramework

SYSTEM_BASE = """你是一个专业的中文旅行规划助手，通过自主调用工具来完成任务。

可用工具与典型编排顺序：
1. update_travel_profile：把用户提到的目的地/天数/偏好等记录进画像；
2. 若缺少目的地或天数等必要信息，通常调用 request_travel_info 向用户追问，并停止后续规划；
2b. 若会话已有已完成行程（有 itinerary 快照）且用户是在修改/对比该行程（如删改某天、比较地点），可复用历史上下文进行修改，不要求用户再次补充 destination/days。
3. 目的地与天数已齐但用户未说明偏好时，调用 request_preference_guide 引导兴趣/节奏/预算等（用户说「随便」可跳过）；
4. build_constraints：把画像转成可验证约束；search_poi：检索目的地候选 POI；check_weather：查询天气；
5. 若用户提到餐饮/菜系/酒店/预算，分别调用 search_restaurant / search_hotel / estimate_budget；
6. recommend_candidates：对候选做打分与多样性重排；
7. plan_and_critique：运行可控规划子图，产出约束满足的行程并结束本轮工具调用；
8. 系统会根据规划结果自动生成卡片与地图，不需要模型继续调用渲染工具。

原则：
- 你自己决定何时调用哪个工具、是否需要追问，不要凭空编造 POI 或行程；
- **若用户仅打招呼、闲聊、或问与出行无关的问题，只友好回应，禁止调用 search_poi / plan_and_critique / render_*；**
- **即使 L3 跨会话画像里有历史目的地/天数，也不能当作本轮已确认信息；用户只说目的地没说天数时，必须先追问天数；**
- 信息齐全且用户明确要规划时，务必依次完成 build_constraints → search_poi → recommend_candidates → plan_and_critique；
- 核心工具返回 `isError=false` 即表示成功；成功后禁止用相同或不同参数重复调用该工具；
- search_poi 成功后的下一步必须是 recommend_candidates，不要为了增加候选而反复搜索；recommend_candidates 成功后的下一步必须是 plan_and_critique，调用后本轮结束；
- 工具返回 `isError=true, retryable=true` 时才允许修正参数后重试，单个工具最多重试一次；`retryable=false` 时禁止重试并如实报告；
- 遇到“吃/餐厅/菜系/美食”要补充 search_restaurant；遇到“住/酒店/住宿区域”要补充 search_hotel；遇到“预算/费用/花多少钱”要补充 estimate_budget；
- 规划必须通过 plan_and_critique 产出，不要自己手写行程文本冒充规划结果；
- plan_and_critique 成功后系统会自动生成结构化摘要、卡片与地图，无需再生成逐日文本。

回复格式（聊天界面右侧已有行程卡片与地图）：
- **禁止**在正文中输出 Day1/Day2、逐日上午/下午/傍晚、大量 emoji、或「人工介入/本地知识库/权威信源」等编造话术；
- **禁止**重复 render_itinerary 已展示的逐日时刻表；最多用一行列出每天去了哪几个点；
- critic 有问题时如实说明，建议放宽偏好或换目的地重试，不要自己「修正」POI 列表；
- 工具数据异常时建议重试，不要掩盖。
"""


def build_system_prompt_sections(
    profile: TravelProfile,
    memory: MemoryFramework | None = None,
    skills_section: str = "",
) -> list[tuple[str, str]]:
    """按段装配 system prompt，返回 (段名, 内容) 列表。

    顺序即注入顺序：静态主干在前、逐轮变化的动态段在后，对 provider 的
    prefix cache 友好；段名同时服务于 prompt dump 的逐段成本统计。"""
    snapshot = _profile_snapshot(profile)
    sections: list[tuple[str, str]] = [
        ("base", SYSTEM_BASE),
        ("profile", f"\n当前已知出行画像（多轮累积）：\n{snapshot}"),
    ]
    if memory is not None:
        sections.append(("memory_mode", f"\n【记忆模式：{memory.mode}】"))
        if memory.compressed_summary:
            sections.append(("l1_summary", f"\nL1 历史摘要：\n{memory.compressed_summary}"))
        sections.append(("l2_snapshot", f"\nL2 本会话工具快照：\n{memory.l2_snapshot}"))
        sections.append(("l3_snapshot", f"\nL3 跨会话用户画像：\n{memory.l3_snapshot}"))
    if skills_section:
        sections.append(("skills", f"\n{skills_section}"))
    return sections


def sections_to_text(sections: list[tuple[str, str]]) -> str:
    return "\n".join(content for _, content in sections)


def build_system_prompt(
    profile: TravelProfile,
    memory: MemoryFramework | None = None,
    skills_section: str = "",
) -> str:
    return sections_to_text(build_system_prompt_sections(profile, memory, skills_section))


def record_system_prompt_trace(
    ctx: Any,
    sections: list[tuple[str, str]],
    *,
    phase: str,
) -> None:
    """把本轮真实发送的 system prompt 与各段成本写入 evaluation_trace。

    对应 prompt 可观测性：复现 eval case、排查模型行为跑偏、分析哪个
    记忆段吃 token 最多时，都有一手证据可查。"""
    if not getattr(ctx, "evaluation_trace_enabled", False):
        return
    from travel_agent.storage.memory_framework import estimate_text_tokens

    content = sections_to_text(sections)
    ctx.evaluation_trace.append(
        {
            "kind": "prompt",
            "phase": phase,
            "sections": [
                {
                    "name": name,
                    "chars": len(section),
                    "est_tokens": estimate_text_tokens(section),
                }
                for name, section in sections
            ],
            "total_chars": len(content),
            "est_tokens": estimate_text_tokens(content),
            "content": content,
        }
    )


def _profile_snapshot(profile: TravelProfile) -> str:
    if not profile.destination and not profile.days and not profile.interests:
        return "- 暂无，需要从用户输入中抽取。"
    lines = []
    if profile.destination:
        lines.append(f"- 目的地：{profile.destination}")
    if profile.days:
        lines.append(f"- 天数：{profile.days}")
    if profile.interests:
        lines.append(f"- 偏好：{format_interests_display(profile.interests)}")
    if profile.budget_level:
        lines.append(f"- 预算：{format_budget_display(profile.budget_level)}")
    if profile.pace:
        lines.append(f"- 节奏：{format_pace_display(profile.pace)}")
    if profile.must_visit:
        lines.append(f"- 必去：{', '.join(profile.must_visit)}")
    if profile.avoid:
        lines.append(f"- 避开：{', '.join(profile.avoid)}")
    missing = profile.missing_required_fields()
    if missing:
        lines.append(f"- 仍缺少：{', '.join(missing)}")
    return "\n".join(lines)


CONVERSATION_SYSTEM = """你是旅行规划助手。当前轮用户尚未提出需要动用工具的出行规划需求。

请用简短、自然的中文回应：
- 可以寒暄、接话；
- 对无关话题（算术、编程、新闻、情感倾诉等）礼貌说明自己主要擅长旅行规划，不要一本正经地当通用百科助手；
- 若用户似乎有出行兴趣但未说清，只轻量追问目的地、天数、偏好；
- 不要编造具体 POI、天气或完整行程；不要假装已经调用了工具；
- 回复保持简短易读，少用 Markdown 装饰。
"""
