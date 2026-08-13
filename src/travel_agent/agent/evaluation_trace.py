from __future__ import annotations

import json
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
        self._estimated_inputs: dict[str, int] = {}
        self._input_shapes: dict[str, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[str(run_id)] = time.perf_counter()
        message_chars = sum(
            len(str(getattr(message, "content", message)))
            for group in messages or []
            for message in (group if isinstance(group, list) else [group])
        )
        invocation = kwargs.get("invocation_params") or {}
        tool_chars = len(json.dumps(invocation.get("tools") or [], default=str))
        key = str(run_id)
        self._estimated_inputs[key] = max(1, (message_chars + tool_chars) // 4)
        self._input_shapes[key] = {
            "message_chars": message_chars,
            "tool_schema_chars": tool_chars,
            "message_count": sum(
                len(group) if isinstance(group, list) else 1 for group in messages or []
            ),
        }

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[str(run_id)] = time.perf_counter()
        self._estimated_inputs[str(run_id)] = sum(
            max(1, len(str(prompt)) // 4) for prompt in prompts
        )
        self._input_shapes[str(run_id)] = {
            "message_chars": sum(len(str(prompt)) for prompt in prompts),
            "tool_schema_chars": 0,
            "message_count": len(prompts),
        }

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        usage, model = _response_usage(response, self.model)
        self.trace.append(
            {
                "kind": "model",
                "phase": self.phase,
                "model": model,
                "input_tokens": usage.get("input_tokens"),
                "estimated_input_tokens": self._estimated_inputs.pop(str(run_id), None),
                "input_shape": self._input_shapes.pop(str(run_id), None),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "visible_output_tokens": usage.get("visible_output_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "provider_completion_tokens": usage.get(
                    "provider_completion_tokens"
                ),
                "finish_reason": usage.get("finish_reason"),
                "truncated": usage.get("truncated"),
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
                "estimated_input_tokens": self._estimated_inputs.pop(str(run_id), None),
                "input_shape": self._input_shapes.pop(str(run_id), None),
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
    completion = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
    completion_details = token_usage.get("completion_tokens_details") or {}
    reasoning = completion_details.get("reasoning_tokens")
    usage = {
        "input_tokens": token_usage.get("input_tokens", token_usage.get("prompt_tokens")),
        "output_tokens": completion,
        "total_tokens": token_usage.get("total_tokens"),
        "provider_completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "visible_output_tokens": (
            max(0, int(completion) - int(reasoning or 0))
            if completion is not None
            else None
        ),
        "finish_reason": None,
        "truncated": False,
    }
    model = llm_output.get("model_name") or default_model
    generations = getattr(response, "generations", None) or []
    for group in generations:
        for generation in group:
            message = getattr(generation, "message", None)
            message_usage = getattr(message, "usage_metadata", None) or {}
            metadata = getattr(message, "response_metadata", None) or {}
            metadata_usage = metadata.get("token_usage") or {}
            if message_usage:
                output_details = message_usage.get("output_token_details") or {}
                provider_completion = metadata_usage.get(
                    "completion_tokens", message_usage.get("output_tokens")
                )
                reasoning_tokens = (
                    (metadata_usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    or output_details.get("reasoning")
                )
                usage = {
                    "input_tokens": message_usage.get("input_tokens"),
                    "output_tokens": message_usage.get("output_tokens"),
                    "total_tokens": message_usage.get("total_tokens"),
                    "provider_completion_tokens": provider_completion,
                    "reasoning_tokens": reasoning_tokens,
                    "visible_output_tokens": (
                        max(
                            0,
                            int(provider_completion) - int(reasoning_tokens or 0),
                        )
                        if provider_completion is not None
                        else message_usage.get("output_tokens")
                    ),
                    "finish_reason": metadata.get("finish_reason"),
                    "truncated": metadata.get("finish_reason")
                    in {"length", "max_tokens"},
                }
            else:
                usage["finish_reason"] = metadata.get("finish_reason")
                usage["truncated"] = metadata.get("finish_reason") in {
                    "length",
                    "max_tokens",
                }
            model = metadata.get("model_name") or model
            return usage, str(model)
    return usage, str(model)
