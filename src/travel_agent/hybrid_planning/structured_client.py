"""Adapter over the production chat-model factory for bounded JSON modules."""

from __future__ import annotations

import json
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

from travel_agent.hybrid_planning.llm_types import (
    StructuredLLMResponse,
    StructuredOutputError,
)


@dataclass(frozen=True)
class ProductionStructuredLLMClient:
    settings: Any
    phase: str
    max_output_tokens: int = 768

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_version: str,
    ) -> StructuredLLMResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        # Reuse the same provider/model construction, callbacks, quotas and
        # timeout behavior as the existing production agent.
        from travel_agent.agent.runtime import _build_chat_model
        from travel_agent.orchestration.meter import meter_callbacks

        model = _build_chat_model(
            self.settings,
            timeout_seconds=int(self.settings.llm.timeout_seconds),
            max_retries=0,
        ).bind(
            max_tokens=self.max_output_tokens,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_outputs: list[str] = []
        total_usage: dict[str, int] = {}
        total_latency_ms = 0.0
        repair_attempted = False

        def invoke(messages: list[Any]) -> str:
            nonlocal total_latency_ms
            started = time.perf_counter()
            try:
                response = model.invoke(
                    messages,
                    config={"callbacks": meter_callbacks(self.phase)} or None,
                )
            except Exception as exc:
                total_latency_ms += round((time.perf_counter() - started) * 1000, 2)
                status = "timeout" if _is_timeout(exc) else "provider_error"
                reason = "timeout" if status == "timeout" else "provider_error"
                raise StructuredOutputError(
                    reason,
                    parse_status=status,
                    raw_output=raw_outputs[-1] if raw_outputs else "",
                    raw_output_summary=_safe_output_summary(raw_outputs[-1] if raw_outputs else ""),
                    latency_ms=round(total_latency_ms, 2),
                    token_usage=total_usage,
                    repair_attempted=repair_attempted,
                    attempts=len(raw_outputs) + 1,
                ) from exc
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            total_latency_ms += elapsed
            _merge_usage(total_usage, _usage_dict(response))
            content = response.content if isinstance(response.content, str) else str(response.content)
            raw_outputs.append(content)
            return content

        content = invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed, fenced = _try_parse_object(content)
        parse_status = "format_repaired" if fenced else "valid"
        parse_reason = "markdown_code_fence" if fenced else None
        if parsed is None:
            primary_reason = _json_failure_reason(content)
            if _repairable_json_shape(content):
                repair_attempted = True
                repair_prompt = (
                    "只修复下面文本的 JSON 语法/围栏格式。不得添加、删除、重命名或改写业务字段；"
                    "如果无法仅靠格式修复，输出原文。只输出一个 JSON object。\n"
                    f"schema_version={schema_version}\n原文：\n{content}"
                )
                repaired = invoke(
                    [
                        SystemMessage(content="你是 JSON 格式修复器，不解释、不推断业务语义。"),
                        HumanMessage(content=repair_prompt),
                    ]
                )
                parsed, _ = _try_parse_object(repaired)
                if parsed is not None:
                    parse_status = "format_repaired"
                    parse_reason = "repair_success"
            if parsed is None:
                raise StructuredOutputError(
                    "repair_failed" if repair_attempted else primary_reason,
                    parse_status="schema_invalid",
                    raw_output=raw_outputs[-1],
                    raw_output_summary=_safe_output_summary(raw_outputs[-1]),
                    latency_ms=round(total_latency_ms, 2),
                    token_usage=total_usage,
                    repair_attempted=repair_attempted,
                    attempts=len(raw_outputs),
                )
        return StructuredLLMResponse(
            payload=parsed,
            model=str(self.settings.llm.model),
            latency_ms=round(total_latency_ms, 2),
            token_usage=total_usage,
            raw_output=raw_outputs[-1],
            raw_outputs=tuple(raw_outputs),
            raw_output_summary=_safe_output_summary(raw_outputs[-1]),
            parse_status=parse_status,
            parse_reason_code=parse_reason,
            repair_attempted=repair_attempted,
            attempts=len(raw_outputs),
        )


def _strip_json_fence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    fenced = False
    if stripped.startswith("```"):
        fenced = True
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped, fenced


def _try_parse_object(text: str) -> tuple[dict[str, Any] | None, bool]:
    stripped, fenced = _strip_json_fence(text)
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return None, fenced
    return (parsed if isinstance(parsed, dict) else None), fenced


def _json_failure_reason(text: str) -> str:
    stripped, _ = _strip_json_fence(text)
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        return "top_level_not_object"
    if not stripped.startswith(("{", "```")):
        return "non_json_text"
    return "invalid_json_syntax"


def _repairable_json_shape(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("{", "```"))


def _safe_output_summary(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="replace")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "char_count": len(text),
        "looks_like_json": text.lstrip().startswith("{"),
        "has_markdown_fence": text.strip().startswith("```"),
    }


def _merge_usage(target: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        target[key] = target.get(key, 0) + int(value)


def _is_timeout(exc: Exception) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def _usage_dict(response: Any) -> dict[str, int]:
    raw = getattr(response, "usage_metadata", None) or {}
    if not isinstance(raw, dict):
        return {}
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    result: dict[str, int] = {}
    for target, names in aliases.items():
        value = next((raw.get(name) for name in names if raw.get(name) is not None), None)
        if value is not None:
            try:
                result[target] = int(value)
            except (TypeError, ValueError):
                continue
    return result
