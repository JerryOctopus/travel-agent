"""Turn-scoped, architecture-neutral usage and hard-budget accounting."""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TurnBudgetExhausted(RuntimeError):
    """Raised before an LLM call that cannot fit the remaining turn budget."""


@dataclass
class _RoleUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    dispatch_count: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    failures: int = 0


@dataclass
class TurnMeter:
    request_id: str
    token_budget: int = 0
    llm_call_budget: int = 0
    tool_call_budget: int = 0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    started_at: float = field(default_factory=time.time)
    _started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    _roles: dict[str, _RoleUsage] = field(default_factory=dict, repr=False)
    _pending: dict[str, tuple[str, int, float]] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _finished_at: float | None = field(default=None, repr=False)
    budget_exhausted: bool = False

    def begin_llm(self, role: str, estimated_input_tokens: int = 0) -> str:
        """Reserve a call before network I/O; reject if no token budget remains."""
        reserve = max(1, int(estimated_input_tokens or 0))
        with self._lock:
            used = sum(item.total_tokens for item in self._roles.values())
            pending = sum(item[1] for item in self._pending.values())
            llm_calls = sum(item.llm_calls for item in self._roles.values())
            if self.llm_call_budget > 0 and llm_calls >= self.llm_call_budget:
                self.budget_exhausted = True
                raise TurnBudgetExhausted(
                    f"turn LLM-call budget exhausted before {role} call: "
                    f"used={llm_calls}, budget={self.llm_call_budget}"
                )
            if self.token_budget > 0 and used + pending + reserve > self.token_budget:
                self.budget_exhausted = True
                raise TurnBudgetExhausted(
                    f"turn token budget exhausted before {role} call: "
                    f"used={used}, reserved={pending}, requested={reserve}, budget={self.token_budget}"
                )
            call_id = f"llm_{uuid.uuid4().hex[:12]}"
            self._pending[call_id] = (role, reserve, time.monotonic())
            self._role(role).llm_calls += 1
            return call_id

    def finish_llm(
        self,
        call_id: str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            pending = self._pending.pop(call_id, None)
            if pending is None:
                return
            role, reserve, started = pending
            usage = self._role(role)
            # Some compatible providers omit usage metadata.  Preserve a
            # conservative, comparable count instead of silently charging zero.
            input_value = max(0, int(input_tokens if input_tokens is not None else reserve))
            output_value = max(0, int(output_tokens or 0))
            total_value = max(
                input_value + output_value,
                int(total_tokens if total_tokens is not None else input_value + output_value),
            )
            usage.input_tokens += input_value
            usage.output_tokens += output_value
            usage.total_tokens += total_value
            usage.latency_ms += (time.monotonic() - started) * 1000
            usage.cost_usd += (
                input_value * self.input_cost_per_million
                + output_value * self.output_cost_per_million
            ) / 1_000_000
            if error:
                usage.failures += 1
            if self.token_budget > 0 and self.total_tokens >= self.token_budget:
                self.budget_exhausted = True

    def begin_tool(self, role: str) -> float:
        with self._lock:
            tool_calls = sum(item.tool_calls for item in self._roles.values())
            if self.tool_call_budget > 0 and tool_calls >= self.tool_call_budget:
                self.budget_exhausted = True
                raise TurnBudgetExhausted(
                    f"turn tool-call budget exhausted before {role} call: "
                    f"used={tool_calls}, budget={self.tool_call_budget}"
                )
            usage = self._role(role)
            usage.tool_calls += 1
            return time.monotonic()

    def finish_tool(self, role: str, started: float, *, error: bool = False) -> None:
        with self._lock:
            usage = self._role(role)
            usage.latency_ms += max(0.0, (time.monotonic() - started) * 1000)
            if error:
                usage.failures += 1

    def record_tool(self, role: str, *, error: bool = False, duration_ms: float = 0.0) -> None:
        """Record deterministic/non-wrapped tools while preserving pre-call checks."""
        started = self.begin_tool(role)
        with self._lock:
            usage = self._role(role)
            usage.latency_ms += max(0.0, duration_ms)
            if error:
                usage.failures += 1

    def record_dispatch(self, agent: str) -> None:
        with self._lock:
            self._role(f"dispatch:{agent}").dispatch_count += 1

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(item.total_tokens for item in self._roles.values())

    def finish(self) -> None:
        with self._lock:
            if self._finished_at is None:
                self._finished_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            roles = {
                role: {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "llm_calls": usage.llm_calls,
                    "tool_calls": usage.tool_calls,
                    "dispatch_count": usage.dispatch_count,
                    "latency_ms": round(usage.latency_ms, 2),
                    "cost_usd": round(usage.cost_usd, 8),
                    "failures": usage.failures,
                }
                for role, usage in sorted(self._roles.items())
            }
            totals = {
                key: sum(float(value[key]) for value in roles.values())
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "llm_calls",
                    "tool_calls",
                    "dispatch_count",
                    "latency_ms",
                    "cost_usd",
                    "failures",
                )
            }
            for key in ("input_tokens", "output_tokens", "total_tokens", "llm_calls", "tool_calls", "dispatch_count", "failures"):
                totals[key] = int(totals[key])
            totals["latency_ms"] = round(
                ((self._finished_at or time.time()) - self.started_at) * 1000, 2
            )
            totals["cost_usd"] = round(totals["cost_usd"], 8)
            return {
                "request_id": self.request_id,
                "token_budget": self.token_budget,
                "llm_call_budget": self.llm_call_budget,
                "tool_call_budget": self.tool_call_budget,
                "budget_exhausted": self.budget_exhausted,
                "roles": roles,
                "totals": totals,
            }

    def _role(self, role: str) -> _RoleUsage:
        return self._roles.setdefault(role, _RoleUsage())


_CURRENT_METER: ContextVar[TurnMeter | None] = ContextVar("travel_agent_turn_meter", default=None)


def current_turn_meter() -> TurnMeter | None:
    return _CURRENT_METER.get()


@contextmanager
def turn_meter_scope(meter: TurnMeter):
    token = _CURRENT_METER.set(meter)
    try:
        yield meter
    finally:
        _CURRENT_METER.reset(token)


class TurnMeterCallback(BaseCallbackHandler):
    """LangChain callback that enforces pre-call budget and records actual usage."""

    raise_error = True

    def __init__(self, role: str) -> None:
        self.role = role
        self._calls: dict[str, str] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._start(str(run_id), _estimate_messages(messages))

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        estimate = sum(max(1, len(str(prompt)) // 4) for prompt in prompts)
        self._start(str(run_id), estimate)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        from travel_agent.agent.evaluation_trace import _response_usage

        usage, _model = _response_usage(response, "")
        self._finish(str(run_id), usage, None)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._finish(str(run_id), {}, f"{type(error).__name__}: {error}")

    def _start(self, run_id: str, estimate: int) -> None:
        meter = current_turn_meter()
        if meter is not None:
            self._calls[run_id] = meter.begin_llm(self.role, estimate)

    def _finish(self, run_id: str, usage: dict[str, Any], error: str | None) -> None:
        meter = current_turn_meter()
        call_id = self._calls.pop(run_id, None)
        if meter is not None and call_id is not None:
            meter.finish_llm(
                call_id,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                error=error,
            )


def meter_callbacks(role: str) -> list[BaseCallbackHandler]:
    return [TurnMeterCallback(role)] if current_turn_meter() is not None else []


def _estimate_messages(messages: Any) -> int:
    total = 0
    for group in messages or []:
        for message in group if isinstance(group, list) else [group]:
            total += max(1, len(str(getattr(message, "content", message))) // 4)
    return total
