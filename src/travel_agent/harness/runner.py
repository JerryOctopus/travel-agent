from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import SessionContext, build_session
from travel_agent.harness.cases import HarnessCase
from travel_agent.harness.environments import HarnessEnvironment
from travel_agent.harness.faults import FaultInjectingProvider
from travel_agent.harness.result import HarnessCaseResult, HarnessTurnResult
from travel_agent.harness.validators import validate_case_result
from travel_agent.settings import Settings, get_settings
from travel_agent.storage.user_profile import _profile_to_dict

_ARTIFACT_KINDS = (
    "constraints",
    "candidates",
    "weather",
    "restaurants",
    "hotels",
    "budget",
    "ranked",
    "route_plan",
    "candidate_comparison",
    "itinerary_patch",
    "local_adjustment_advice",
    "itinerary",
    "duration_diagnostic",
    "agent_trace",
    "variant_metrics",
)


class AgentHarness:
    """Drive `run_production_turn` (or a variant) with controlled case, history, and artifact capture."""

    def __init__(
        self,
        settings: Settings | None = None,
        environment: HarnessEnvironment | None = None,
    ) -> None:
        self.environment = environment or HarnessEnvironment()
        self.settings = self.environment.apply(settings or get_settings())

    def run_case(self, case: HarnessCase) -> HarnessCaseResult:
        session_ids = _session_ids(case)
        contexts: dict[str, SessionContext] = {}
        histories: dict[str, list[tuple[str, str]]] = {}
        turns: list[HarnessTurnResult] = []
        errors: list[str] = []
        user_ids = _user_ids(case, self.environment.user_id)
        if not case.turns:
            ctx = build_session(
                session_id=f"eval_{case.case_id}",
                poi_path=self.environment.poi_path,
                persist=self.environment.persist,
            )
            _apply_tool_provider_mode(ctx, self.environment)
            _prepare_context(ctx, case)
            return _finalize_case(case, [], ctx, [], variant=_architecture_variant(self.environment))

        for index, user_message in enumerate(case.turns):
            session_id = session_ids[index]
            if session_id not in contexts:
                ctx = build_session(
                    session_id=session_id,
                    poi_path=self.environment.poi_path,
                    persist=self.environment.persist,
                )
                _apply_tool_provider_mode(ctx, self.environment)
                _prepare_context(ctx, case)
                contexts[session_id] = ctx
                histories[session_id] = []
            ctx = contexts[session_id]
            turn = self._run_turn(
                user_message,
                ctx,
                histories[session_id],
                user_ids[index],
            )
            turns.append(turn)
            if turn.error:
                errors.append(turn.error)
                break

        final_ctx = contexts[session_ids[len(turns) - 1]]
        return _finalize_case(
            case, turns, final_ctx, errors, variant=_architecture_variant(self.environment)
        )

    def run_case_with_context(
        self,
        case: HarnessCase,
        ctx: SessionContext,
    ) -> HarnessCaseResult:
        _prepare_context(ctx, case)
        history: list[tuple[str, str]] = []
        turns: list[HarnessTurnResult] = []
        errors: list[str] = []

        for user_message in case.turns:
            turn = self._run_turn(
                user_message,
                ctx,
                history,
                str(case.metadata.get("user_id") or self.environment.user_id),
            )
            turns.append(turn)
            if turn.error:
                errors.append(turn.error)
                break
        return _finalize_case(
            case, turns, ctx, errors, variant=_architecture_variant(self.environment)
        )

    def _run_turn(
        self,
        user_message: str,
        ctx: SessionContext,
        history: list[tuple[str, str]],
        user_id: str,
    ) -> HarnessTurnResult:
        started = time.perf_counter()
        trace_start = len(ctx.evaluation_trace)
        try:
            reply = _dispatch_turn(
                self.environment.variant,
                user_message,
                ctx=ctx,
                history=list(history),
                settings=self.settings,
                user_id=user_id,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            trace = ctx.evaluation_trace[trace_start:]
            tool_calls = [dict(item) for item in trace if item.get("kind") == "tool"]
            recorded_names = {item.get("name") for item in tool_calls}
            tool_calls.extend(
                {"name": name, "arguments": {}, "status": "ok", "duration_ms": None}
                for name in reply.tool_trace
                if name not in recorded_names
            )
            history.append(("user", user_message))
            history.append(("assistant", reply.text))
            return HarnessTurnResult(
                user_message=user_message,
                reply_text=reply.text,
                tool_trace=list(reply.tool_trace),
                used_real_agent=reply.used_real_agent,
                clarification=reply.clarification,
                profile=dict(reply.profile),
                artifacts=_artifact_snapshot(ctx),
                duration_ms=duration_ms,
                tool_calls=tool_calls,
                model_calls=[dict(item) for item in trace if item.get("kind") == "model"],
                memory_snapshot=_memory_snapshot(self.settings, user_id),
                status=getattr(reply, "status", None),
                plan_artifact_id=getattr(reply, "plan_artifact_id", None),
                delivery_artifact_id=getattr(reply, "delivery_artifact_id", None),
                agent_trace=list(getattr(reply, "agent_trace", None) or []),
                request_id=getattr(reply, "request_id", None),
                turn_metrics=dict(getattr(reply, "turn_metrics", None) or {}),
                planner_status=getattr(reply, "planner_status", None),
                recovery_state=getattr(reply, "recovery_state", None),
                critical_slots_matched=list(
                    getattr(reply, "critical_slots_matched", None) or []
                ),
                failure_reason=getattr(reply, "failure_reason", None),
                raw_failure=getattr(reply, "raw_failure", None),
                fallback_triggered=bool(getattr(reply, "fallback_triggered", False)),
                final_outcome=getattr(reply, "final_outcome", None),
                delivery_status=getattr(reply, "delivery_status", None),
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            message = f"{type(exc).__name__}: {exc}"
            trace = ctx.evaluation_trace[trace_start:]
            return HarnessTurnResult(
                user_message=user_message,
                reply_text="",
                tool_trace=[],
                used_real_agent=False,
                clarification=False,
                profile=_profile_to_dict(ctx.profile),
                artifacts=_artifact_snapshot(ctx),
                error=message,
                duration_ms=duration_ms,
                tool_calls=[dict(item) for item in trace if item.get("kind") == "tool"],
                model_calls=[dict(item) for item in trace if item.get("kind") == "model"],
                memory_snapshot=_memory_snapshot(self.settings, user_id),
            )

    def run_cases(self, cases: list[HarnessCase]) -> list[HarnessCaseResult]:
        return [self.run_case(case) for case in cases]


def _artifact_snapshot(ctx: SessionContext) -> dict[str, Any]:
    return {
        kind: payload
        for kind in _ARTIFACT_KINDS
        if (payload := (
            ctx.store.latest_current(kind)
            if kind == "itinerary"
            else ctx.store.latest(kind)
        )) is not None
    }


def _dispatch_turn(
    variant: str | None,
    user_message: str,
    *,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Settings,
    user_id: str,
):
    """Route a harness turn: production entry by default, explicit V0-V3 for ablation."""
    if variant:
        from travel_agent.orchestration.variants import run_variant_turn

        return run_variant_turn(
            variant,
            user_message,
            ctx,
            settings,
            history,
            user_id,
        )
    return run_production_turn(
        user_message,
        ctx=ctx,
        history=history,
        settings=settings,
        user_id=user_id,
    )


def _architecture_variant(environment: HarnessEnvironment) -> str:
    """Explicit harness variant (v0-v3) -> external architecture V0-V3 label.

    None means the production fixed chain (Multi-Agent Full = V3).
    """
    variant = environment.variant
    return str(variant).upper() if variant else "V3"


def _prepare_context(ctx: SessionContext, case: HarnessCase) -> None:
    ctx.evaluation_trace_enabled = True
    ctx.reference_datetime = case.snapshot_date
    if case.failure_injection:
        ctx.provider = FaultInjectingProvider(ctx.provider, case.failure_injection)


def _apply_tool_provider_mode(
    ctx: SessionContext,
    environment: HarnessEnvironment,
) -> None:
    if environment.tool_provider != "local":
        return
    from travel_agent.data_loader import load_seed_pois
    from travel_agent.providers import LocalToolProvider

    ctx.provider = LocalToolProvider(load_seed_pois(environment.poi_path))


def _session_ids(case: HarnessCase) -> list[str]:
    if not case.turns:
        return []
    if case.session_ids and len(case.session_ids) != len(case.turns):
        raise ValueError(f"case {case.case_id}: session_ids must match turns")
    if case.session_ids:
        return case.session_ids
    session_id = str(case.metadata.get("session_id") or f"eval_{case.case_id}")
    return [session_id] * len(case.turns)


def _user_ids(case: HarnessCase, default_user_id: str) -> list[str]:
    if case.user_ids and len(case.user_ids) != len(case.turns):
        raise ValueError(f"case {case.case_id}: user_ids must match turns")
    if case.user_ids:
        return case.user_ids
    user_id = str(case.metadata.get("user_id") or default_user_id)
    return [user_id] * len(case.turns)


def _memory_snapshot(settings: Settings, user_id: str) -> dict[str, Any]:
    try:
        from travel_agent.storage.user_memory import get_user_memory_service

        service = get_user_memory_service(settings.memory)
        profile = _profile_to_dict(service.load_stable_profile(user_id))
        trips = [asdict(item) for item in service.list_recent_trips(user_id, limit=3)]
        return {"stable_profile": profile, "recent_trips": trips}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _finalize_case(
    case: HarnessCase,
    turns: list[HarnessTurnResult],
    ctx: SessionContext,
    errors: list[str],
    *,
    variant: str = "V3",
) -> HarnessCaseResult:
    result = HarnessCaseResult(
        case_id=case.case_id,
        turns=turns,
        final_profile=_profile_to_dict(ctx.profile),
        final_artifacts=_artifact_snapshot(ctx),
        errors=errors,
    )
    metrics = validate_case_result(case, result, variant=variant)
    return HarnessCaseResult(
        case_id=result.case_id,
        turns=result.turns,
        final_profile=result.final_profile,
        final_artifacts=result.final_artifacts,
        passed=_all_known_metrics_pass(metrics, production=bool(case.gold_outcome)),
        metrics=metrics,
        errors=result.errors,
    )


# production_v1 专属布尔指标；旧 schema case 无 gold，不参与 passed 合取。
_PRODUCTION_BOOLEAN_KEYS = {
    "strict_task_success",
    "expected_outcome_match",
    "grounding_ok",
    "feasibility_ok",
    "authorization_ok",
    "architecture_policy_ok",
    "gating_passed",
}


def _all_known_metrics_pass(metrics: dict[str, Any], *, production: bool) -> bool | None:
    known = [
        value
        for key, value in metrics.items()
        if isinstance(value, bool) and (production or key not in _PRODUCTION_BOOLEAN_KEYS)
    ]
    if not known:
        return None
    return all(known)
