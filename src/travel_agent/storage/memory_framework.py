"""按阈值在 full / compressed / profile_only 间切换记忆注入策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from travel_agent.agent.session import SessionContext
from travel_agent.schemas import TravelProfile
from travel_agent.settings import LLMSettings, MemorySettings
from travel_agent.workflow import merge_profile
from travel_agent.storage.memory_compressor import MemoryCompressor
from travel_agent.storage.user_memory import RecentTrip, get_user_memory_service

MemoryMode = Literal["full", "compressed", "profile_only"]

# 为压缩摘要输出 + 下一轮回复预留的 token（借鉴 Claude Code Auto-Compact 的预算扣减）。
RESERVED_OUTPUT_TOKENS = 20_000

# 无法在线探测窗口时的离线估算表；前缀匹配即返回，未命中走默认值。
_KNOWN_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("qwen-long", 10_000_000),
    ("gemini-2.5", 1_048_576),
    ("deepseek-v4", 1_000_000),
    ("glm-4.6", 200_000),
    ("glm-4.5", 128_000),
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_000_000),
    ("deepseek-chat", 64_000),
    ("deepseek-reasoner", 64_000),
    ("glm-4", 128_000),
    ("qwen", 128_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-3.5", 16_000),
)
DEFAULT_CONTEXT_WINDOW = 128_000


def estimate_text_tokens(content: str) -> int:
    """保守估算中英文混合文本的 token 数。"""
    ascii_chars = sum(char.isascii() for char in content)
    non_ascii_chars = len(content) - ascii_chars
    return (ascii_chars + 3) // 4 + non_ascii_chars


def estimate_history_tokens(history: list[tuple[str, str]]) -> int:
    """估算历史消息 token 数；上线后应结合模型 usage 持续校准。"""
    return sum(estimate_text_tokens(content) for _, content in history)


def estimate_model_context_window(model: str) -> int:
    """按模型名离线估算上下文窗口；显式配置 ``LLMSettings.context_window`` 优先。"""
    name = (model or "").lower()
    for keyword, window in _KNOWN_CONTEXT_WINDOWS:
        if keyword in name:
            return window
    return DEFAULT_CONTEXT_WINDOW


def effective_profile_only_threshold(
    settings: MemorySettings,
    llm_settings: LLMSettings | None = None,
) -> int:
    """生效阈值 = min(配置阈值, 模型窗口 - 预留输出)。

    换用大窗口模型时无需手调配置；小窗口模型则避免历史先撞模型上限。
    """
    configured = settings.profile_only_token_threshold
    if llm_settings is None:
        return configured
    window = llm_settings.context_window or estimate_model_context_window(llm_settings.model)
    return max(min(configured, window - RESERVED_OUTPUT_TOKENS), 0)


def choose_memory_mode(
    history: list[tuple[str, str]],
    settings: MemorySettings,
    llm_settings: LLMSettings | None = None,
) -> MemoryMode:
    est_tokens = estimate_history_tokens(history)
    if est_tokens >= effective_profile_only_threshold(settings, llm_settings):
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

        settings = get_settings()
        mode = choose_memory_mode(history, memory_settings, settings.llm)
        compressor = MemoryCompressor(
            llm_settings=settings.llm if llm_enabled else None,
            keep_recent_turns=memory_settings.keep_recent_turns,
        )

        if mode == "profile_only":
            max_messages = memory_settings.keep_recent_turns * 2
            hist = history[-max_messages:]
            old_history = history[:-max_messages]
            # summarize_old 内部带缓存与熔断：LLM 不可用时自动回退规则摘要。
            summary = compressor.summarize_old(old_history)
        elif mode == "compressed":
            hist, summary = compressor.compress(history)
        else:
            hist, summary = history, ""

        memory_service = get_user_memory_service(memory_settings)
        l3_profile = memory_service.load_stable_profile(user_id)
        recent_trips = memory_service.list_recent_trips(user_id, 3)
        # L3 仅注入 prompt 快照，不自动写入 ctx.profile，避免用户只说「你好」就按旧画像开规划。

        return cls(
            mode=mode,
            history_for_prompt=hist,
            compressed_summary=summary,
            l2_snapshot=ctx.store.build_prompt_snapshot(),
            l3_snapshot=_format_l3(l3_profile, user_id, recent_trips),
        )


def _format_l3(
    profile: TravelProfile,
    user_id: str,
    recent_trips: list[RecentTrip] | None = None,
) -> str:
    parts = [f"用户ID：{user_id}"]
    if profile.interests:
        parts.append(f"长期偏好：{', '.join(profile.interests[:8])}")
    if profile.food_preference:
        parts.append(f"饮食偏好：{', '.join(profile.food_preference[:8])}")
    if profile.pace and profile.pace != "standard":
        parts.append(f"常用节奏：{profile.pace}")
    if profile.budget_level:
        parts.append(f"常用预算：{profile.budget_level}")
    if profile.transport_mode and profile.transport_mode != "public_transport":
        parts.append(f"常用交通：{profile.transport_mode}")
    if profile.avoid:
        parts.append(f"长期避开：{', '.join(profile.avoid[:8])}")
    for trip in (recent_trips or [])[:3]:
        detail = " ".join(
            value
            for value in (
                trip.start_date or "",
                trip.destination or "",
                f"{trip.days}天" if trip.days else "",
            )
            if value
        )
        summary = trip.itinerary_summary[:160]
        parts.append(f"最近行程：{detail}；{summary}".rstrip("；"))
    if len(parts) == 1:
        return f"用户ID：{user_id}（暂无跨会话画像）"
    snapshot = "\n".join(parts)
    # L3 prompt 采用约 2K token 的保守上限；中文按 1 字符/token 估算。
    return snapshot[:2000]
