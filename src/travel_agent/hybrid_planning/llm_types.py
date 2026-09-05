"""Small injectable structured-output boundary shared by hybrid modules.

The production adapter owns JSON decoding and the single format-only repair
attempt.  Business validators stay in the individual modules so taxonomy and
candidate permissions cannot be relaxed by a generic parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


StructuredParseStatus = Literal[
    "valid",
    "format_repaired",
    "schema_invalid",
    "taxonomy_invalid",
    "low_confidence",
    "timeout",
    "provider_error",
]


class StructuredOutputError(RuntimeError):
    """Transport/format failure with safe, machine-readable diagnostics."""

    def __init__(
        self,
        reason_code: str,
        *,
        parse_status: StructuredParseStatus,
        raw_output: str = "",
        raw_output_summary: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        token_usage: dict[str, int] | None = None,
        repair_attempted: bool = False,
        attempts: int = 1,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.parse_status = parse_status
        self.raw_output = raw_output
        self.raw_output_summary = dict(raw_output_summary or {})
        self.latency_ms = latency_ms
        self.token_usage = dict(token_usage or {})
        self.repair_attempted = repair_attempted
        self.attempts = attempts


@dataclass(frozen=True)
class StructuredLLMResponse:
    payload: dict[str, Any]
    model: str = ""
    latency_ms: float | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_output: str = ""
    raw_outputs: tuple[str, ...] = ()
    raw_output_summary: dict[str, Any] = field(default_factory=dict)
    parse_status: StructuredParseStatus = "valid"
    parse_reason_code: str | None = None
    repair_attempted: bool = False
    attempts: int = 1


class StructuredLLMClient(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_version: str,
    ) -> StructuredLLMResponse:
        """Return one parsed JSON object or raise on transport/schema failure."""
