from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HarnessTurnResult:
    user_message: str
    reply_text: str
    tool_trace: list[str]
    used_real_agent: bool
    clarification: bool
    profile: dict[str, Any]
    artifacts: dict[str, Any]
    error: str | None = None
    duration_ms: float | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    planner_status: str | None = None
    recovery_state: str | None = None
    critical_slots_matched: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    status: str | None = None
    plan_artifact_id: str | None = None
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    request_id: str | None = None
    turn_metrics: dict[str, Any] = field(default_factory=dict)
    raw_failure: str | None = None
    fallback_triggered: bool = False
    final_outcome: str | None = None


@dataclass(frozen=True)
class HarnessCaseResult:
    case_id: str
    turns: list[HarnessTurnResult]
    final_profile: dict[str, Any]
    final_artifacts: dict[str, Any]
    passed: bool | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def last_turn(self) -> HarnessTurnResult | None:
        return self.turns[-1] if self.turns else None


@dataclass(frozen=True)
class HarnessOperationResult:
    case_id: str
    operation: str
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float | None = None


@dataclass(frozen=True)
class HarnessBenchmarkResult:
    benchmark: str
    suite: str
    mode: str
    metrics: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HarnessSuiteResult:
    suite: str
    mode: str
    environment: str
    case_count: int
    attempted_count: int
    metrics: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    benchmarks: list[HarnessBenchmarkResult] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None

    @property
    def passed(self) -> bool | None:
        values = [
            value for value in self.metrics.values()
            if isinstance(value, bool)
        ]
        if not values:
            return None
        return all(values)
