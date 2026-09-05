"""Monotonic turn deadlines and admission control for multi-agent execution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


def _positive(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True)
class DeadlineConfig:
    turn_hard_cap: float = 300.0
    base_timeout: float = 210.0
    planner_reserve: float = 25.0
    reviewer_reserve: float = 85.0
    deterministic_gate_reserve: float = 5.0
    render_reserve: float = 5.0
    router_timeout: float = 80.0
    router_useful: float = 35.0
    worker_timeout: float = 60.0
    planner_timeout: float = 60.0
    recovery_worker_useful: float = 25.0
    reviewer_timeout: float = 85.0
    repair_worker_timeout: float = 30.0
    repair_planner_timeout: float = 45.0
    admission_guard: float = 5.0
    latency_sla: float = 180.0

    @classmethod
    def from_settings(cls, settings: Any) -> "DeadlineConfig":
        agent = getattr(settings, "agent", None)
        orchestration = getattr(settings, "orchestration", None)
        return cls(
            turn_hard_cap=_positive(
                getattr(agent, "request_timeout_seconds", None), 300.0
            ),
            base_timeout=_positive(
                getattr(orchestration, "base_timeout_seconds", None), 210.0
            ),
            planner_reserve=_positive(
                getattr(orchestration, "planner_reserve_seconds", None), 25.0
            ),
            reviewer_reserve=_positive(
                getattr(orchestration, "reviewer_reserve_seconds", None), 85.0
            ),
            deterministic_gate_reserve=_positive(
                getattr(orchestration, "deterministic_gate_reserve_seconds", None), 5.0
            ),
            render_reserve=_positive(
                getattr(orchestration, "render_reserve_seconds", None), 5.0
            ),
            router_timeout=_positive(
                getattr(orchestration, "router_timeout_seconds", None), 80.0
            ),
            router_useful=_positive(
                getattr(orchestration, "router_useful_seconds", None), 35.0
            ),
            worker_timeout=_positive(
                getattr(orchestration, "worker_timeout_seconds", None), 60.0
            ),
            planner_timeout=_positive(
                getattr(orchestration, "planner_timeout_seconds", None), 60.0
            ),
            recovery_worker_useful=_positive(
                getattr(orchestration, "recovery_worker_useful_seconds", None),
                25.0,
            ),
            reviewer_timeout=_positive(
                getattr(orchestration, "reviewer_timeout_seconds", None), 85.0
            ),
            repair_worker_timeout=_positive(
                getattr(orchestration, "repair_worker_timeout_seconds", None), 30.0
            ),
            repair_planner_timeout=_positive(
                getattr(orchestration, "repair_planner_timeout_seconds", None), 45.0
            ),
            admission_guard=_positive(
                getattr(orchestration, "admission_guard_seconds", None), 5.0
            ),
            latency_sla=_positive(
                getattr(orchestration, "latency_sla_seconds", None), 180.0
            ),
        )


@dataclass(frozen=True)
class TurnDeadline:
    """Absolute deadlines derived once from the turn's monotonic start."""

    started_at: float
    config: DeadlineConfig

    @classmethod
    def start(cls, settings: Any, *, started_at: float | None = None) -> "TurnDeadline":
        if started_at is None:
            try:
                from travel_agent.orchestration.meter import current_turn_meter

                meter = current_turn_meter()
                started_at = meter.started_monotonic if meter is not None else None
            except Exception:  # pragma: no cover - defensive import boundary
                started_at = None
        return cls(
            started_at=time.monotonic() if started_at is None else started_at,
            config=DeadlineConfig.from_settings(settings),
        )

    @property
    def turn_deadline(self) -> float:
        return self.started_at + self.config.turn_hard_cap

    @property
    def base_deadline(self) -> float:
        return min(
            self.started_at + self.config.base_timeout,
            self.turn_deadline
            - self.config.reviewer_reserve
            - self.config.deterministic_gate_reserve
            - self.config.render_reserve
            - self.config.admission_guard,
        )

    def turn_remaining(self, now: float | None = None) -> float:
        return max(0.0, self.turn_deadline - (time.monotonic() if now is None else now))

    def remaining_base(self, now: float | None = None) -> float:
        return max(0.0, self.base_deadline - (time.monotonic() if now is None else now))

    def usable_preplanner_time(self, now: float | None = None) -> float:
        return max(
            0.0,
            self.remaining_base(now)
            - self.config.planner_reserve
            - self.config.admission_guard,
        )

    def effective_preplanner_timeout(
        self,
        component_timeout: float,
        now: float | None = None,
    ) -> float:
        timestamp = time.monotonic() if now is None else now
        return max(
            0.0,
            min(
                float(component_timeout),
                self.usable_preplanner_time(timestamp),
                self.turn_remaining(timestamp),
            ),
        )

    def admits_recovery_wave(self, now: float | None = None) -> bool:
        return self.usable_preplanner_time(now) >= (
            self.config.router_useful + self.config.recovery_worker_useful
        )

    def admits_router(self, now: float | None = None) -> bool:
        return self.usable_preplanner_time(now) >= (
            self.config.router_useful + self.config.recovery_worker_useful
        )

    def router_timeout_preserving_worker(self, now: float | None = None) -> float:
        """Bound Router without consuming the minimum useful Worker window."""
        timestamp = time.monotonic() if now is None else now
        return max(
            0.0,
            min(
                self.config.router_timeout,
                self.usable_preplanner_time(timestamp)
                - self.config.recovery_worker_useful,
                self.turn_remaining(timestamp),
            ),
        )

    def admits_planner(self, now: float | None = None) -> bool:
        return self.remaining_base(now) >= (
            self.config.planner_reserve + self.config.admission_guard
        )

    def planner_timeout(self, component_timeout: float, now: float | None = None) -> float:
        timestamp = time.monotonic() if now is None else now
        return max(
            0.0,
            min(
                float(component_timeout),
                self.config.planner_timeout,
                self.remaining_base(timestamp) - self.config.admission_guard,
                self.turn_remaining(timestamp) - self.config.admission_guard,
            ),
        )

    def reviewer_timeout(self, now: float | None = None) -> float:
        return max(
            0.0,
            min(
                self.config.reviewer_timeout,
                self.turn_remaining(now)
                - self.config.deterministic_gate_reserve
                - self.config.render_reserve
                - self.config.admission_guard,
            ),
        )

    def admits_repair(self, *, domain_worker: bool, now: float | None = None) -> bool:
        required = self.config.repair_planner_timeout + self.config.admission_guard
        if domain_worker:
            required += self.config.repair_worker_timeout
        return self.turn_remaining(now) >= required

    def repair_worker_timeout(self, now: float | None = None) -> float:
        """Keep repair Planner time and guard unavailable to the repair worker."""
        return max(
            0.0,
            min(
                self.config.repair_worker_timeout,
                self.turn_remaining(now)
                - self.config.repair_planner_timeout
                - self.config.admission_guard,
            ),
        )

    def repair_planner_timeout(self, now: float | None = None) -> float:
        return max(
            0.0,
            min(
                self.config.repair_planner_timeout,
                self.turn_remaining(now) - self.config.admission_guard,
            ),
        )

    def sla_exceeded(self, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        return timestamp - self.started_at > self.config.latency_sla

    def trace_detail(self, now: float | None = None) -> dict[str, int | bool]:
        timestamp = time.monotonic() if now is None else now
        return {
            "remaining_ms": int(self.turn_remaining(timestamp) * 1000),
            "remaining_base_ms": int(self.remaining_base(timestamp) * 1000),
            "usable_preplanner_ms": int(self.usable_preplanner_time(timestamp) * 1000),
            "planner_reserved_ms": int(self.config.planner_reserve * 1000),
            "reviewer_reserved_ms": int(self.config.reviewer_reserve * 1000),
            "deterministic_gate_reserved_ms": int(self.config.deterministic_gate_reserve * 1000),
            "render_reserved_ms": int(self.config.render_reserve * 1000),
            "sla_180_exceeded": self.sla_exceeded(timestamp),
        }
