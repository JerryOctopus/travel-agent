"""A bounded LLM policy layer for already-legal planning candidates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from travel_agent.hybrid_planning.call_cache import HybridCallCache
from travel_agent.hybrid_planning.llm_types import (
    StructuredLLMClient,
    StructuredOutputError,
)


PREFERENCE_RESOLVER_SCHEMA_VERSION = "preference-resolver-v2"
PREFERENCE_RESOLVER_PROMPT_VERSION = "preference-resolver-prompt-v2"
_FULL_PLAN_TYPES = frozenset({"full_itinerary", "full_trip_plan"})
_ALLOWED_DIMENSIONS = frozenset(
    {
        "commute_time",
        "walking_load",
        "transfer_count",
        "price",
        "location",
        "pace",
        "accessibility",
        "activity_density",
        "user_interest_match",
    }
)
_ALLOWED_DIRECTIONS = frozenset({"minimize", "maximize", "prefer"})
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "hard_constraints",
        "hard_constraint_changes",
        "budget_max_cny",
        "budget_limit",
        "must_visit",
        "avoid",
        "deadline",
        "return_deadline",
        "fixed_event",
        "fixed_events",
        "transport_prohibition",
        "itinerary",
        "artifact",
        "artifact_status",
        "publish",
        "new_candidates",
        "pois",
        "hotels",
        "restaurants",
    }
)
_MAX_TRADEOFFS = 2
_MAX_PRICE_PREMIUM_PERCENT = 20.0

_DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "commute_time": ("commute_time", "commute_minutes", "travel_time_min"),
    "walking_load": (
        "walking_load",
        "walking_minutes",
        "walking_distance",
        "walking_distance_km",
    ),
    "transfer_count": ("transfer_count", "transfers"),
    "price": ("price", "price_cny", "price_level", "nightly_price"),
    "location": ("location", "area", "location_score"),
    "pace": ("pace", "pace_fit"),
    "accessibility": ("accessibility", "accessibility_score", "step_free"),
    "activity_density": ("activity_density", "activities_per_day", "density_score"),
    "user_interest_match": (
        "user_interest_match",
        "interest_match_score",
        "deterministic_score",
        "score",
    ),
}

_PREFERENCE_PATTERNS: dict[str, re.Pattern[str]] = {
    "price": re.compile(r"价格|便宜|省钱|预算|性价比|贵.{0,5}(?:一点|一些|\d+%)"),
    "commute_time": re.compile(r"通勤|少坐车|就近|路上|车程|交通时间"),
    "walking_load": re.compile(r"少走|步行|爬坡|楼梯|别太累|体力"),
    "transfer_count": re.compile(r"换乘|倒车"),
    "location": re.compile(r"位置|地段|靠近|附近|中心|区域"),
    "pace": re.compile(r"节奏|轻松|悠闲|紧凑|高强度|慢慢"),
    "accessibility": re.compile(r"无障碍|轮椅|老人|父母|长辈|行动不便"),
    "activity_density": re.compile(r"多安排|少安排|活动密度|丰富|不要太满"),
    "user_interest_match": re.compile(
        r"兴趣|喜欢|偏好|体验|自然|风景|文化|历史|博物馆|亲子|咖啡|"
        r"interests|nature|food|culture|history|museum|family|cafe|citywalk"
    ),
}


@dataclass(frozen=True)
class PolicyPriority:
    dimension: str
    direction: Literal["minimize", "maximize", "prefer"]
    importance: float
    source: str | None
    reason: str


@dataclass(frozen=True)
class PlanningPolicy:
    priorities: list[PolicyPriority]
    pace: Literal["relaxed", "normal", "compact"] = "normal"
    acceptable_tradeoffs: dict[str, float] = field(default_factory=dict)
    candidate_order: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: Literal["llm", "deterministic_default"] = "deterministic_default"
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "priorities": [asdict(item) for item in self.priorities],
            "pace": self.pace,
            "acceptable_tradeoffs": dict(self.acceptable_tradeoffs),
            "candidate_order": list(self.candidate_order),
            "unresolved": list(self.unresolved),
            "confidence": self.confidence,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class PreferenceEligibility:
    eligible: bool
    reason_code: str
    relevant_dimensions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreferenceResolver:
    """Resolve soft trade-offs without changing the legal candidate set."""

    llm_client: StructuredLLMClient | None = None
    enabled: bool = False
    confidence_threshold: float = 0.65
    cache: HybridCallCache = field(default_factory=HybridCallCache, repr=False)
    request_scope: str = "standalone"

    def resolve(
        self,
        *,
        task_type: Any,
        candidates: list[Mapping[str, Any]],
        soft_preferences: str | list[str] | Mapping[str, Any] | None,
        hard_constraints: Mapping[str, Any] | None = None,
        source_turn: str | None = None,
        hard_filter_complete: bool = True,
        evidence_sufficient: bool = True,
        delivery_intent: Any | None = None,
        trace: Any | None = None,
    ) -> PlanningPolicy:
        hard = dict(hard_constraints or {})
        legal = _legal_candidates(candidates, hard)
        candidate_ids = [_candidate_id(item) for item in legal]
        preference_text = _preference_text(soft_preferences)
        tradeoff_dimensions = _candidate_tradeoff_dimensions(legal)
        relevant_dimensions = _relevant_preference_dimensions(preference_text)
        eligibility = preference_eligibility(
            task_type=str(getattr(task_type, "value", task_type)),
            candidates=legal,
            candidate_ids=candidate_ids,
            preferences=preference_text,
            differences=tradeoff_dimensions,
            relevant_dimensions=relevant_dimensions,
            hard_filter_complete=hard_filter_complete,
            evidence_sufficient=evidence_sufficient,
            delivery_intent=str(getattr(delivery_intent, "value", delivery_intent or "")),
        )
        eligible = eligibility.eligible
        default = _deterministic_default_policy(
            preference_text,
            candidate_ids,
            source_turn,
            fallback_reason=None if eligible else eligibility.reason_code,
        )
        fingerprint = _policy_fingerprint(
            legal,
            soft_preferences,
            hard,
            eligibility.relevant_dimensions,
        )
        detail: dict[str, Any] = {
            "eligible": eligible,
            "eligibility_reason": eligibility.reason_code,
            "called": False,
            "skipped_reason": None if eligible else eligibility.reason_code,
            "model": "",
            "prompt_version": PREFERENCE_RESOLVER_PROMPT_VERSION,
            "schema_version": PREFERENCE_RESOLVER_SCHEMA_VERSION,
            "source_turn": source_turn,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "structured_output": None,
            "confidence": None,
            "fallback_reason": None,
            "latency_ms": None,
            "token_usage": {},
            "out_of_taxonomy_label_count": 0,
            "unknown_candidate_reference_count": 0,
            "attempted_hard_constraint_change": False,
            "tradeoff_dimensions": tradeoff_dimensions,
            "relevant_dimensions": list(eligibility.relevant_dimensions),
            "candidate_set_fingerprint": fingerprint,
            "cache_reason": None,
            "parse_status": None,
            "validation_status": None,
            "parse_reason_code": None,
            "raw_output_summary": {},
            "repair_attempted": False,
            "attempts": 0,
        }
        if not eligible:
            detail["fallback_reason"] = eligibility.reason_code
            _append_trace(trace, detail)
            return default
        if not self.enabled:
            policy = _replace_fallback(default, "feature_disabled")
            detail["skipped_reason"] = "feature_disabled"
            detail["fallback_reason"] = "feature_disabled"
            _append_trace(trace, detail)
            return policy
        if self.llm_client is None:
            policy = _replace_fallback(default, "llm_unavailable")
            detail["skipped_reason"] = "llm_unavailable"
            detail["fallback_reason"] = "llm_unavailable"
            _append_trace(trace, detail)
            return policy

        reservation, cached = self.cache.reserve(
            self.request_scope, "preference", fingerprint
        )
        if reservation == "in_progress":
            ready, cached = self.cache.wait(cached, timeout_seconds=65.0)
            reservation = "cache_hit" if ready else "dedup_wait_timeout"
        if reservation == "cache_hit" and isinstance(cached, PlanningPolicy):
            detail["cache_reason"] = "valid_cache_hit" if not cached.fallback_reason else "dedup_after_fallback"
            detail["skipped_reason"] = detail["cache_reason"]
            detail["fallback_reason"] = cached.fallback_reason
            detail["validation_status"] = "cached"
            _append_trace(trace, detail)
            return cached
        if reservation == "dedup_wait_timeout":
            policy = _replace_fallback(default, "dedup_wait_timeout")
            detail["cache_reason"] = "dedup_wait_timeout"
            detail["fallback_reason"] = "dedup_wait_timeout"
            _append_trace(trace, detail)
            return policy

        detail["called"] = True
        fallback_reason: str | None = None
        policy: PlanningPolicy | None = None
        try:
            compact_candidates = _compact_candidates(
                legal, eligibility.relevant_dimensions
            )
            response = self.llm_client.complete_json(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=json.dumps(
                    {
                        "candidate_ids": candidate_ids,
                        "candidates": compact_candidates,
                        "soft_preferences": _compact_preferences(soft_preferences),
                        "allowed_dimensions": list(eligibility.relevant_dimensions),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
                schema_version=PREFERENCE_RESOLVER_SCHEMA_VERSION,
            )
            detail["model"] = response.model
            detail["latency_ms"] = response.latency_ms
            detail["token_usage"] = dict(response.token_usage)
            detail["structured_output"] = response.payload
            detail["parse_status"] = response.parse_status
            detail["parse_reason_code"] = response.parse_reason_code
            detail["raw_output_summary"] = dict(response.raw_output_summary)
            detail["repair_attempted"] = response.repair_attempted
            detail["attempts"] = response.attempts
            policy, unknown_count, attempted_hard_change = self._validate_output(
                response.payload,
                candidate_ids,
                eligibility.relevant_dimensions,
                legal,
                hard,
                source_turn,
            )
            detail["unknown_candidate_reference_count"] = unknown_count
            detail["attempted_hard_constraint_change"] = attempted_hard_change
            detail["confidence"] = policy.confidence if policy is not None else None
            if attempted_hard_change:
                fallback_reason = "attempted_hard_constraint_change"
                detail["validation_status"] = "schema_invalid"
                detail["parse_reason_code"] = fallback_reason
            elif unknown_count:
                fallback_reason = "unknown_candidate_id"
                detail["validation_status"] = "schema_invalid"
                detail["parse_reason_code"] = fallback_reason
            elif policy is None:
                fallback_reason = "schema_invalid"
                detail["validation_status"] = "schema_invalid"
                detail["parse_reason_code"] = detail["parse_reason_code"] or fallback_reason
            elif policy.confidence < self.confidence_threshold:
                fallback_reason = "low_confidence"
                detail["parse_status"] = "low_confidence"
                detail["validation_status"] = "low_confidence"
                detail["parse_reason_code"] = "confidence_below_threshold"
            else:
                detail["validation_status"] = "valid"
        except StructuredOutputError as exc:
            fallback_reason = exc.reason_code
            detail["parse_status"] = exc.parse_status
            detail["validation_status"] = exc.parse_status
            detail["parse_reason_code"] = exc.reason_code
            detail["raw_output_summary"] = dict(exc.raw_output_summary)
            detail["latency_ms"] = exc.latency_ms
            detail["token_usage"] = dict(exc.token_usage)
            detail["repair_attempted"] = exc.repair_attempted
            detail["attempts"] = exc.attempts
        except Exception as exc:
            fallback_reason, status, reason = _validation_failure(exc)
            detail["parse_status"] = status
            detail["validation_status"] = status
            detail["parse_reason_code"] = reason

        if fallback_reason is not None or policy is None:
            policy = _replace_fallback(default, fallback_reason or "provider_error")
        detail["fallback_reason"] = policy.fallback_reason
        self.cache.publish(self.request_scope, "preference", fingerprint, policy)
        _append_trace(trace, detail)
        return policy

    def _validate_output(
        self,
        payload: dict[str, Any],
        candidate_ids: list[str],
        relevant_dimensions: tuple[str, ...],
        legal_candidates: list[Mapping[str, Any]],
        hard_constraints: Mapping[str, Any],
        source_turn: str | None,
    ) -> tuple[PlanningPolicy | None, int, bool]:
        if not isinstance(payload, dict):
            raise ValueError("schema:not_object")
        attempted_hard_change = bool(_FORBIDDEN_OUTPUT_KEYS.intersection(payload))
        if attempted_hard_change:
            return None, 0, True
        required = {"priorities", "candidate_order", "confidence"}
        if not required.issubset(payload):
            raise ValueError("schema:missing_field")
        if not isinstance(payload["candidate_order"], list):
            raise TypeError("schema:candidate_order_type_mismatch")
        if not all(isinstance(item, str) for item in payload["candidate_order"]):
            raise TypeError("schema:candidate_id_type_mismatch")
        order = list(payload["candidate_order"])
        if not order:
            raise ValueError("schema:empty_candidate_order")
        unknown = sum(item not in candidate_ids for item in order)
        if len(order) != len(set(order)):
            raise ValueError("schema:duplicate_candidate_id")

        raw_priorities = payload["priorities"]
        if not isinstance(raw_priorities, list):
            raise TypeError("schema:priorities_type_mismatch")
        if not raw_priorities:
            raise ValueError("schema:empty_priorities")
        if len(raw_priorities) > len(relevant_dimensions):
            raise ValueError("schema:too_many_priorities")
        priorities: list[PolicyPriority] = []
        for raw in raw_priorities:
            if not isinstance(raw, dict):
                raise TypeError("schema:priority_type_mismatch")
            required_priority = {"dimension", "direction", "importance"}
            if not required_priority.issubset(raw):
                raise ValueError("schema:missing_priority_field")
            if not isinstance(raw["dimension"], str) or not isinstance(raw["direction"], str):
                raise TypeError("schema:priority_string_type_mismatch")
            dimension = raw["dimension"]
            direction = raw["direction"]
            if dimension not in _ALLOWED_DIMENSIONS or dimension not in relevant_dimensions:
                raise ValueError("schema:dimension_enum_mismatch")
            if direction not in _ALLOWED_DIRECTIONS:
                raise ValueError("schema:direction_enum_mismatch")
            importance_value = raw["importance"]
            if isinstance(importance_value, bool) or not isinstance(importance_value, (int, float)):
                raise TypeError("schema:importance_type_mismatch")
            importance = float(importance_value)
            if not 0.0 <= importance <= 1.0:
                raise ValueError("schema:importance_out_of_range")
            source = raw.get("source")
            reason = raw.get("reason")
            if source is not None and not isinstance(source, str):
                raise TypeError("schema:source_type_mismatch")
            if reason is not None and not isinstance(reason, str):
                raise TypeError("schema:reason_type_mismatch")
            priorities.append(
                PolicyPriority(
                    dimension=dimension,
                    direction=direction,  # type: ignore[arg-type]
                    importance=importance,
                    source=source or source_turn,
                    reason=reason or "用户表达的软偏好",
                )
            )

        pace = payload.get("pace", "normal")
        if not isinstance(pace, str):
            raise TypeError("schema:pace_type_mismatch")
        if pace not in {"relaxed", "normal", "compact"}:
            raise ValueError("schema:pace_enum_mismatch")
        unresolved = payload.get("unresolved", [])
        if not isinstance(unresolved, list) or not all(isinstance(item, str) for item in unresolved):
            raise TypeError("schema:unresolved_type_mismatch")

        confidence_value = payload["confidence"]
        if isinstance(confidence_value, bool) or not isinstance(confidence_value, (int, float)):
            raise TypeError("schema:confidence_type_mismatch")
        confidence = float(confidence_value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("schema:confidence_out_of_range")

        tradeoffs = _validate_tradeoffs(
            payload.get("acceptable_tradeoffs", {}),
            legal_candidates,
            hard_constraints,
            order,
        )
        return (
            PlanningPolicy(
                priorities=priorities,
                pace=pace,  # type: ignore[arg-type]
                acceptable_tradeoffs=tradeoffs,
                candidate_order=order,
                unresolved=list(unresolved),
                confidence=confidence,
                source="llm",
            ),
            unknown,
            attempted_hard_change,
        )


def preference_eligibility(
    *,
    task_type: str,
    candidates: list[Mapping[str, Any]],
    candidate_ids: list[str],
    preferences: str,
    differences: list[str],
    relevant_dimensions: list[str],
    hard_filter_complete: bool,
    evidence_sufficient: bool,
    delivery_intent: str = "",
) -> PreferenceEligibility:
    if task_type not in _FULL_PLAN_TYPES:
        return PreferenceEligibility(False, "not_full_itinerary")
    if delivery_intent == "state_update_only":
        return PreferenceEligibility(False, "rebuild_pending_without_final_request")
    if not hard_filter_complete:
        return PreferenceEligibility(False, "hard_filter_incomplete")
    if len(candidates) < 2 or len(set(candidate_ids)) < 2:
        return PreferenceEligibility(False, "fewer_than_two_legal_candidates")
    if not preferences:
        return PreferenceEligibility(False, "no_soft_preference")
    if not relevant_dimensions:
        return PreferenceEligibility(False, "no_relevant_soft_preference")
    if not evidence_sufficient:
        return PreferenceEligibility(False, "candidate_evidence_insufficient")
    relevant_differences = tuple(
        dimension for dimension in relevant_dimensions if dimension in differences
    )
    if not relevant_differences:
        missing = any(
            not _dimension_has_complete_evidence(candidates, dimension)
            for dimension in relevant_dimensions
        )
        return PreferenceEligibility(
            False,
            "candidate_evidence_insufficient" if missing else "no_relevant_candidate_difference",
        )
    return PreferenceEligibility(True, "eligible", relevant_differences)


def _legal_candidates(
    candidates: list[Mapping[str, Any]], hard_constraints: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    budget = _budget_max(hard_constraints)
    legal: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = _candidate_id(candidate)
        if not candidate_id or candidate_id in seen:
            continue
        if candidate.get("hard_constraints_passed", True) is not True:
            continue
        price = _candidate_numeric_price(candidate)
        if budget is not None and price is not None and price > budget:
            continue
        seen.add(candidate_id)
        legal.append(candidate)
    return legal


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("candidate_id") or candidate.get("poi_id") or candidate.get("id") or "").strip()


def _candidate_dimension_value(candidate: Mapping[str, Any], dimension: str) -> Any:
    return next(
        (candidate.get(key) for key in _DIMENSION_ALIASES[dimension] if key in candidate),
        None,
    )


def _candidate_tradeoff_dimensions(candidates: list[Mapping[str, Any]]) -> list[str]:
    differences: list[str] = []
    for dimension in sorted(_ALLOWED_DIMENSIONS):
        values = [_candidate_dimension_value(candidate, dimension) for candidate in candidates]
        if any(value is None for value in values):
            continue
        encoded = [json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) for value in values]
        if len(set(encoded)) > 1:
            differences.append(dimension)
    return differences


def _dimension_has_complete_evidence(
    candidates: list[Mapping[str, Any]], dimension: str
) -> bool:
    return bool(candidates) and all(
        _candidate_dimension_value(candidate, dimension) is not None
        for candidate in candidates
    )


def _relevant_preference_dimensions(text: str) -> list[str]:
    return [
        dimension for dimension, pattern in _PREFERENCE_PATTERNS.items()
        if pattern.search(text)
    ]


def _preference_text(value: str | list[str] | Mapping[str, Any] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        meaningful = {key: item for key, item in value.items() if item not in (None, "", [], {})}
        return json.dumps(meaningful, ensure_ascii=False, sort_keys=True, default=str) if meaningful else ""
    return " ".join(str(item) for item in value if str(item).strip()).strip()


def _compact_preferences(
    value: str | list[str] | Mapping[str, Any] | None,
) -> str | list[str] | dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return value
    allowed = {
        "interests",
        "pace",
        "companions",
        "budget_level",
        "preference",
        "mobility",
        "accessibility_priority",
        "soft_avoid_interests",
    }
    return {
        key: value[key]
        for key in sorted(allowed.intersection(value))
        if value[key] not in (None, "", [], {})
    }


def _compact_candidates(
    candidates: list[Mapping[str, Any]], dimensions: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": _candidate_id(candidate),
            **{
                dimension: _candidate_dimension_value(candidate, dimension)
                for dimension in dimensions
            },
        }
        for candidate in candidates
    ]


def _policy_fingerprint(
    candidates: list[Mapping[str, Any]],
    preferences: Any,
    hard_constraints: Mapping[str, Any],
    relevant_dimensions: tuple[str, ...],
) -> str:
    compact = sorted(
        _compact_candidates(candidates, relevant_dimensions),
        key=lambda item: item["candidate_id"],
    )
    revision = hard_constraints.get("_constraint_revision") or hard_constraints.get("constraint_revision") or 0
    payload = {
        "candidates": compact,
        "preferences": _compact_preferences(preferences),
        "constraint_revision": revision,
        "budget_max_cny": _budget_max(hard_constraints),
        "schema": PREFERENCE_RESOLVER_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _budget_max(hard_constraints: Mapping[str, Any]) -> float | None:
    for key in ("budget_max_cny", "budget_limit"):
        value = hard_constraints.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and float(value) >= 0:
            return float(value)
    return None


def _candidate_numeric_price(candidate: Mapping[str, Any]) -> float | None:
    for key in ("price", "price_cny", "nightly_price"):
        value = candidate.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and float(value) >= 0:
            return float(value)
    return None


def _validate_tradeoffs(
    raw: Any,
    candidates: list[Mapping[str, Any]],
    hard_constraints: Mapping[str, Any],
    order: list[str],
) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise TypeError("schema:tradeoffs_type_mismatch")
    if len(raw) > _MAX_TRADEOFFS:
        raise ValueError("schema:too_many_tradeoffs")
    if set(raw) - {"hotel_price_premium_percent"}:
        raise ValueError("schema:tradeoff_key_enum_mismatch")
    if not raw:
        return {}
    value = raw.get("hotel_price_premium_percent")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("schema:tradeoff_value_type_mismatch")
    premium = float(value)
    if not 0.0 <= premium <= _MAX_PRICE_PREMIUM_PERCENT:
        raise ValueError("schema:tradeoff_out_of_range")
    if premium == 0:
        return {"hotel_price_premium_percent": 0.0}
    prices = {
        _candidate_id(candidate): price
        for candidate in candidates
        if (price := _candidate_numeric_price(candidate)) is not None
    }
    budget = _budget_max(hard_constraints)
    if len(prices) < 2 or budget is None:
        return {}
    if order and order[0] in prices and prices[order[0]] > budget:
        raise ValueError("schema:budget_tradeoff_conflict")
    within_budget = [price for price in prices.values() if price <= budget]
    if len(within_budget) < 2:
        return {}
    return {"hotel_price_premium_percent": premium}


def _deterministic_default_policy(
    text: str,
    candidate_ids: list[str],
    source_turn: str | None,
    *,
    fallback_reason: str | None,
) -> PlanningPolicy:
    rules = (
        (r"价格|便宜|省钱|预算", "price", "minimize", "优先控制价格"),
        (r"通勤|少坐车|就近|路上", "commute_time", "minimize", "优先减少通勤"),
        (r"少换乘|换乘|老人|父母", "transfer_count", "minimize", "优先减少换乘"),
        (r"少走|步行|轮椅|无障碍|爬坡|楼梯", "walking_load", "minimize", "优先减少步行负担"),
        (r"位置|地段|靠近|附近", "location", "prefer", "优先匹配位置偏好"),
        (r"兴趣|喜欢|偏好|体验|自然|文化|历史|咖啡", "user_interest_match", "maximize", "优先匹配用户兴趣"),
    )
    priorities = [
        PolicyPriority(dimension, direction, 0.8, source_turn, reason)  # type: ignore[arg-type]
        for pattern, dimension, direction, reason in rules
        if re.search(pattern, text)
    ]
    if not priorities:
        priorities = [
            PolicyPriority("user_interest_match", "maximize", 1.0, source_turn, "保持确定性候选排序")
        ]
    pace: Literal["relaxed", "normal", "compact"] = (
        "relaxed" if re.search(r"孩子|儿童|老人|父母|轻松|悠闲|别太累", text)
        else "compact" if re.search(r"紧凑|多安排|高强度", text)
        else "normal"
    )
    return PlanningPolicy(
        priorities=priorities,
        pace=pace,
        candidate_order=list(candidate_ids),
        confidence=1.0,
        source="deterministic_default",
        fallback_reason=fallback_reason,
    )


def _replace_fallback(policy: PlanningPolicy, reason: str) -> PlanningPolicy:
    return PlanningPolicy(
        priorities=policy.priorities,
        pace=policy.pace,
        acceptable_tradeoffs=policy.acceptable_tradeoffs,
        candidate_order=policy.candidate_order,
        unresolved=policy.unresolved,
        confidence=policy.confidence,
        source="deterministic_default",
        fallback_reason=reason,
    )


def _validation_failure(exc: Exception) -> tuple[str, str, str]:
    name = type(exc).__name__.lower()
    if "timeout" in name or "timed out" in str(exc).lower():
        return "timeout", "timeout", "timeout"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        reason = str(exc)
        code = reason.removeprefix("schema:") if reason.startswith("schema:") else "schema_invalid"
        return "schema_invalid", "schema_invalid", code
    return "provider_error", "provider_error", "provider_error"


def _append_trace(trace: Any | None, detail: dict[str, Any]) -> None:
    if trace is None:
        try:
            from travel_agent.orchestration.multi_agent.trace import current_trace

            trace = current_trace()
        except Exception:
            trace = None
    if trace is not None:
        trace.append(
            "hybrid_preference_resolver",
            agent="preference_resolver",
            status="completed" if not detail.get("fallback_reason") else "fallback",
            duration_ms=detail.get("latency_ms"),
            detail=detail,
        )


_SYSTEM_PROMPT = """你是旅行候选软偏好排序器。只输出一个 JSON object，不要 markdown。
只能引用输入 candidate_id；dimension 只能逐字选择 allowed_dimensions；不得新建候选。
硬约束已完成过滤且不会提供给你：不得输出或修改预算上限、must_visit、avoid、deadline、
fixed_event、交通禁令、itinerary 或 Artifact。软价格溢价绝不代表提高预算上限。
固定结构：{\"priorities\":[{\"dimension\":\"允许维度\",\"direction\":\"minimize|maximize|prefer\",
\"importance\":0.0,\"source\":\"偏好原文\",\"reason\":\"简短依据\"}],
\"pace\":\"relaxed|normal|compact\",\"acceptable_tradeoffs\":{},
\"candidate_order\":[\"candidate_id\"],\"unresolved\":[],\"confidence\":0.0}。"""
