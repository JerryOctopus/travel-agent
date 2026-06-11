"""按阈值在 full / compressed / profile_only 间切换记忆注入策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from travel_agent.agent.session import SessionContext
from travel_agent.schemas import TravelProfile
from travel_agent.settings import MemorySettings
from travel_agent.workflow import merge_profile
from travel_agent.storage.memory_compressor import MemoryCompressor
from travel_agent.storage.user_profile import UserProfileStore

MemoryMode = Literal["full", "compressed", "profile_only"]


def choose_memory_mode(
    history: list[tuple[str, str]],
    settings: MemorySettings,
) -> MemoryMode:
    est_tokens = sum(len(content) for _, content in history) // 3
    if est_tokens >= settings.profile_only_token_threshold:
        return "profile_only"
    if len(history) >= settings.compress_message_threshold:
        return "compressed"
    return "full"


@dataclass
class MemoryFramework:
    mode: MemoryMode
    history_for_prompt: list[tuple[str, str]]
    compressed_summary: str
    l2_snapshot: str
    l3_snapshot: str

    @classmethod
    def build(
        cls,
        ctx: SessionContext,
        history: list[tuple[str, str]],
        user_id: str,
        memory_settings: MemorySettings,
        llm_enabled: bool,
    ) -> MemoryFramework:
        from travel_agent.settings import get_settings

        mode = choose_memory_mode(history, memory_settings)
        settings = get_settings()
        compressor = MemoryCompressor(
            llm_settings=settings.llm if llm_enabled else None,
            keep_recent_turns=memory_settings.keep_recent_turns,
        )

        if mode == "profile_only":
            hist: list[tuple[str, str]] = []
            summary = compressor._rule_summarize(history) if history else ""
        elif mode == "compressed":
            hist, summary = compressor.compress(history)
        else:
            hist, summary = history, ""

        l3_profile = UserProfileStore(memory_settings.profile_dir).load(user_id)
        ctx.profile = merge_profile(l3_profile, ctx.profile)

        return cls(
            mode=mode,
            history_for_prompt=hist,
            compressed_summary=summary,
            l2_snapshot=ctx.store.build_prompt_snapshot(),
            l3_snapshot=_format_l3(l3_profile, user_id),
        )


def _format_l3(profile: TravelProfile, user_id: str) -> str:
    parts = [f"用户ID：{user_id}"]
    if profile.interests:
        parts.append(f"长期偏好：{', '.join(profile.interests)}")
    if profile.pace and profile.pace != "standard":
        parts.append(f"常用节奏：{profile.pace}")
    if profile.budget_level:
        parts.append(f"常用预算：{profile.budget_level}")
    if profile.avoid:
        parts.append(f"长期避开：{', '.join(profile.avoid)}")
    if len(parts) == 1:
        return f"用户ID：{user_id}（暂无跨会话画像）"
    return "\n".join(parts)
