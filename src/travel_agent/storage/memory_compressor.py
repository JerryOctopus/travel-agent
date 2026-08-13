"""L1：对话历史压缩（超阈值时摘要旧消息）。

借鉴 Claude Code Auto-Compact 的两条工程实践：

- 压缩结果缓存：同一进程内相同的旧历史只做一次 LLM 摘要，避免每轮重复付费；
- 失败熔断：LLM 摘要连续失败达阈值后，冷却期内直接走规则压缩，不再徒劳重试。
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from travel_agent.settings import LLMSettings

MAX_SUMMARY_CACHE_SIZE = 128
MAX_CONSECUTIVE_SUMMARY_FAILURES = 3
SUMMARY_FAILURE_COOLDOWN_SECONDS = 600.0

_LOCK = threading.Lock()
_SUMMARY_CACHE: "OrderedDict[str, str]" = OrderedDict()
_consecutive_failures = 0
_blocked_until = 0.0


def _cache_key(history: list[tuple[str, str]], model: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    for role, content in history:
        digest.update(f"\n{role}\x00{content}".encode("utf-8"))
    return digest.hexdigest()


def _llm_summary_blocked(now: float | None = None) -> bool:
    moment = now if now is not None else time.monotonic()
    with _LOCK:
        return moment < _blocked_until


def _record_summary_failure(now: float | None = None) -> None:
    global _consecutive_failures, _blocked_until
    moment = now if now is not None else time.monotonic()
    with _LOCK:
        _consecutive_failures += 1
        if _consecutive_failures >= MAX_CONSECUTIVE_SUMMARY_FAILURES:
            # 触发熔断：冷却期内短路 LLM 摘要，随后重新计数。
            _blocked_until = moment + SUMMARY_FAILURE_COOLDOWN_SECONDS
            _consecutive_failures = 0


def _record_summary_success() -> None:
    global _consecutive_failures, _blocked_until
    with _LOCK:
        _consecutive_failures = 0
        _blocked_until = 0.0


def reset_summarizer_guard() -> None:
    """测试钩子：清空摘要缓存与熔断状态。"""
    global _consecutive_failures, _blocked_until
    with _LOCK:
        _consecutive_failures = 0
        _blocked_until = 0.0
        _SUMMARY_CACHE.clear()


class MemoryCompressor:
    def __init__(
        self,
        llm_settings: LLMSettings | None = None,
        keep_recent_turns: int = 10,
    ) -> None:
        self.llm_settings = llm_settings
        self.keep_recent_turns = keep_recent_turns

    def compress(
        self,
        history: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], str]:
        """返回 (保留的近期历史, 旧消息摘要)。"""
        max_messages = self.keep_recent_turns * 2
        if len(history) <= max_messages:
            return history, ""

        old = history[:-max_messages]
        recent = history[-max_messages:]
        return recent, self.summarize_old(old)

    def summarize_old(self, old: list[tuple[str, str]]) -> str:
        """对旧消息做带缓存与熔断的 LLM 摘要，不可用时回退规则摘要。"""
        if not old:
            return ""
        if self.llm_settings and self.llm_settings.enabled:
            key = _cache_key(old, self.llm_settings.model)
            with _LOCK:
                cached = _SUMMARY_CACHE.get(key)
                if cached is not None:
                    _SUMMARY_CACHE.move_to_end(key)
            if cached:
                return cached
            if not _llm_summary_blocked():
                summary = self._llm_summarize(old)
                if summary:
                    _record_summary_success()
                    with _LOCK:
                        _SUMMARY_CACHE[key] = summary
                        _SUMMARY_CACHE.move_to_end(key)
                        while len(_SUMMARY_CACHE) > MAX_SUMMARY_CACHE_SIZE:
                            _SUMMARY_CACHE.popitem(last=False)
                    return summary
                _record_summary_failure()
        return self._rule_summarize(old)

    def _rule_summarize(self, history: list[tuple[str, str]]) -> str:
        parts: list[str] = []
        for role, content in history:
            label = "用户" if role == "user" else "助手"
            text = content.strip().replace("\n", " ")
            if len(text) > 120:
                text = text[:120] + "…"
            parts.append(f"{label}：{text}")
        return "历史对话摘要：\n" + "\n".join(parts)

    def _llm_summarize(self, history: list[tuple[str, str]]) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            if not self.llm_settings or not self.llm_settings.api_key:
                return ""
            model = ChatOpenAI(
                model=self.llm_settings.model,
                api_key=self.llm_settings.api_key,
                base_url=self.llm_settings.base_url,
                temperature=0.1,
                timeout=self.llm_settings.timeout_seconds,
            )
            transcript = "\n".join(
                f"{'用户' if r == 'user' else '助手'}: {c}" for r, c in history
            )
            resp = model.invoke(
                [
                    SystemMessage(
                        content="你是旅行助手记忆压缩器。用中文 3-6 条要点概括对话中的目的地、天数、偏好、已做决策，不要编造。"
                    ),
                    HumanMessage(content=transcript),
                ]
            )
            content = resp.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            pass
        return ""
