from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class EvaluationTraceCallback(BaseCallbackHandler):
    """Collect per-model-call timing and token usage without prompt contents."""

    def __init__(self, trace: list[dict[str, Any]], *, model: str, phase: str) -> None:
        self.trace = trace
        self.model = model
        self.phase = phase
        self._started: dict[str, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[str(run_id)] = time.perf_counter()

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        usage, model = _response_usage(response, self.model)
        self.trace.append(
            {
                "kind": "model",
                "phase": self.phase,
                "model": model,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "duration_ms": self._duration(run_id),
                "status": "ok",
                "error": None,
            }
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.trace.append(
            {
                "kind": "model",
                "phase": self.phase,
                "model": self.model,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "duration_ms": self._duration(run_id),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def _duration(self, run_id: UUID) -> float | None:
        started = self._started.pop(str(run_id), None)
        if started is None:
            return None
        return round((time.perf_counter() - started) * 1000, 2)


def _response_usage(response: Any, default_model: str) -> tuple[dict[str, Any], str]:
    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") or {}
    usage = {
        "input_tokens": token_usage.get("input_tokens", token_usage.get("prompt_tokens")),
        "output_tokens": token_usage.get("output_tokens", token_usage.get("completion_tokens")),
        "total_tokens": token_usage.get("total_tokens"),
    }
    model = llm_output.get("model_name") or default_model
    generations = getattr(response, "generations", None) or []
    for group in generations:
        for generation in group:
            message = getattr(generation, "message", None)
            message_usage = getattr(message, "usage_metadata", None) or {}
            metadata = getattr(message, "response_metadata", None) or {}
            if message_usage:
                usage = {
                    "input_tokens": message_usage.get("input_tokens"),
                    "output_tokens": message_usage.get("output_tokens"),
                    "total_tokens": message_usage.get("total_tokens"),
                }
            model = metadata.get("model_name") or model
            return usage, str(model)
    return usage, str(model)
