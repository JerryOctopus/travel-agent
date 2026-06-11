"""L1：对话历史压缩（超阈值时摘要旧消息）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from travel_agent.settings import LLMSettings


class MemoryCompressor:
    def __init__(
        self,
        llm_settings: LLMSettings | None = None,
        keep_recent_turns: int = 4,
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
        if self.llm_settings and self.llm_settings.enabled:
            summary = self._llm_summarize(old)
            if summary:
                return recent, summary
        return recent, self._rule_summarize(old)

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
