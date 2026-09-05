from __future__ import annotations

from dataclasses import dataclass

import pytest

from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.hybrid_planning.preference_resolver import PreferenceResolver
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog


@dataclass
class FakeClient:
    payload: dict | None = None
    error: Exception | None = None
    calls: int = 0

    def complete_json(self, **_kwargs) -> StructuredLLMResponse:
        self.calls += 1
        if self.error:
            raise self.error
        if self.payload is None:
            raise ValueError("invalid JSON")
        return StructuredLLMResponse(
            self.payload,
            model="fake-model",
            latency_ms=8,
            token_usage={"input_tokens": 20, "output_tokens": 8},
        )


def candidates() -> list[dict]:
    return [
        {"candidate_id": "a", "price": 500, "commute_minutes": 35, "transfers": 2},
        {"candidate_id": "b", "price": 620, "commute_minutes": 10, "transfers": 0},
    ]


def valid_payload(**overrides) -> dict:
    payload = {
        "priorities": [
            {
                "dimension": "commute_time",
                "direction": "minimize",
                "importance": 0.9,
                "source": "turn_3",
                "reason": "减少市内通勤",
            }
        ],
        "pace": "relaxed",
        "acceptable_tradeoffs": {"hotel_price_premium_percent": 20},
        "candidate_order": ["b", "a"],
        "unresolved": [],
        "confidence": 0.86,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("preference", "dimension", "pace", "fallback"),
    [
        ("价格优先", "price", "normal", "feature_disabled"),
        ("尽量减少通勤", "commute_time", "normal", "feature_disabled"),
        ("带老人，减少换乘", "transfer_count", "relaxed", "feature_disabled"),
        ("带孩子，节奏轻松", "user_interest_match", "relaxed", "candidate_evidence_insufficient"),
    ],
)
def test_deterministic_fallback_expresses_common_soft_preferences(preference, dimension, pace, fallback) -> None:
    policy = PreferenceResolver(enabled=False).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences=preference,
    )

    assert policy.priorities[0].dimension == dimension
    assert policy.pace == pace
    assert policy.source == "deterministic_default"
    assert policy.fallback_reason == fallback


def test_valid_llm_policy_only_reorders_known_legal_candidates() -> None:
    client = FakeClient(valid_payload())
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences="少通勤，可以贵一点",
        hard_constraints={"budget_max_cny": 5000},
        source_turn="turn_3",
    )

    assert policy.source == "llm"
    assert policy.candidate_order == ["b", "a"]
    assert policy.acceptable_tradeoffs == {"hotel_price_premium_percent": 20.0}
    assert client.calls == 1


def test_no_soft_preference_does_not_call() -> None:
    client = FakeClient(valid_payload())
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary", candidates=candidates(), soft_preferences=None
    )

    assert client.calls == 0
    assert policy.fallback_reason == "no_soft_preference"


def test_only_one_legal_candidate_does_not_call() -> None:
    client = FakeClient(valid_payload())
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary", candidates=candidates()[:1], soft_preferences="少通勤"
    )

    assert client.calls == 0
    assert policy.fallback_reason == "fewer_than_two_legal_candidates"


@pytest.mark.parametrize("task_type", ["route_plan", "weather_only", "restaurant_only", "itinerary_patch"])
def test_non_full_itinerary_requests_do_not_call(task_type) -> None:
    client = FakeClient(valid_payload())
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type=task_type, candidates=candidates(), soft_preferences="少通勤"
    )

    assert client.calls == 0
    assert policy.fallback_reason == "not_full_itinerary"


def test_unknown_candidate_reference_falls_back_and_is_traced() -> None:
    client = FakeClient(valid_payload(candidate_order=["b", "invented"]))
    trace = AgentTraceLog("req-1")
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences="少通勤",
        trace=trace,
    )

    assert policy.source == "deterministic_default"
    assert policy.candidate_order == ["a", "b"]
    assert policy.fallback_reason == "unknown_candidate_id"
    assert trace.snapshot()[0]["detail"]["unknown_candidate_reference_count"] == 1


@pytest.mark.parametrize(
    "forbidden",
    [
        {"hard_constraints": {"budget_max_cny": 999999}},
        {"must_visit": []},
        {"itinerary": {"days": []}},
        {"new_candidates": [{"candidate_id": "new"}]},
    ],
)
def test_attempt_to_modify_hard_boundary_or_create_content_falls_back(forbidden) -> None:
    client = FakeClient(valid_payload(**forbidden))
    trace = AgentTraceLog("req-2")
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences="预算优先",
        hard_constraints={"budget_max_cny": 1000},
        trace=trace,
    )

    assert policy.source == "deterministic_default"
    assert policy.fallback_reason == "attempted_hard_constraint_change"
    assert trace.snapshot()[0]["detail"]["attempted_hard_constraint_change"] is True


def test_soft_price_premium_never_changes_hard_budget() -> None:
    hard = {"budget_max_cny": 1000}
    payload = valid_payload(
        priorities=[{
            "dimension": "price", "direction": "minimize", "importance": 0.8,
            "source": "位置好可以贵20%", "reason": "价格软权衡",
        }]
    )
    policy = PreferenceResolver(FakeClient(payload), enabled=True).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences="位置好可以贵20%",
        hard_constraints=hard,
    )

    assert hard == {"budget_max_cny": 1000}
    assert policy.acceptable_tradeoffs["hotel_price_premium_percent"] == 20


@pytest.mark.parametrize(
    ("payload", "error", "fallback"),
    [
        (valid_payload(confidence=0.4), None, "low_confidence"),
        (None, TimeoutError("slow"), "timeout"),
        (None, ValueError("invalid JSON"), "schema_invalid"),
    ],
)
def test_llm_failure_or_low_confidence_uses_deterministic_default(payload, error, fallback) -> None:
    client = FakeClient(payload=payload, error=error)
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary", candidates=candidates(), soft_preferences="通勤优先"
    )

    assert policy.source == "deterministic_default"
    assert policy.priorities[0].dimension == "commute_time"
    assert policy.fallback_reason == fallback


def test_feature_flag_off_never_calls_llm() -> None:
    client = FakeClient(valid_payload())
    trace = AgentTraceLog("req-3")
    PreferenceResolver(client, enabled=False).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences="少通勤",
        trace=trace,
    )

    assert client.calls == 0
    assert trace.snapshot()[0]["detail"]["called"] is False
    assert trace.snapshot()[0]["detail"]["skipped_reason"] == "feature_disabled"


@pytest.mark.parametrize(
    ("preference", "candidate_override", "expected_reason"),
    [
        ("想要安静一点", {}, "no_relevant_soft_preference"),
        ("少走路", {}, "candidate_evidence_insufficient"),
        ("通勤优先", {"commute_minutes": 20}, "no_relevant_candidate_difference"),
    ],
)
def test_preference_eligibility_requires_relevant_comparable_evidence(
    preference, candidate_override, expected_reason
) -> None:
    client = FakeClient(valid_payload())
    source = [{**item, **candidate_override} for item in candidates()]

    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary", candidates=source, soft_preferences=preference
    )

    assert client.calls == 0
    assert policy.fallback_reason == expected_reason


def test_rebuild_pending_without_final_request_does_not_call() -> None:
    client = FakeClient(valid_payload())
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary",
        delivery_intent="state_update_only",
        candidates=candidates(),
        soft_preferences="通勤优先",
    )

    assert client.calls == 0
    assert policy.fallback_reason == "rebuild_pending_without_final_request"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"priorities": []}, "empty_priorities"),
        ({"priorities": [{"dimension": "cost", "direction": "minimize", "importance": 0.9}]}, "dimension_enum_mismatch"),
        ({"priorities": [{"dimension": "commute_time", "direction": "lower", "importance": 0.9}]}, "direction_enum_mismatch"),
        ({"priorities": [{"dimension": "commute_time", "direction": "minimize", "importance": "0.9"}]}, "importance_type_mismatch"),
        ({"confidence": 1.2}, "confidence_out_of_range"),
    ],
)
def test_preference_schema_reason_codes(overrides, reason) -> None:
    trace = AgentTraceLog("schema")
    policy = PreferenceResolver(FakeClient(valid_payload(**overrides)), enabled=True).resolve(
        task_type="full_itinerary",
        candidates=candidates(),
        soft_preferences="通勤优先",
        trace=trace,
    )

    if reason is None:
        assert policy.source == "llm"
    else:
        assert policy.fallback_reason == "schema_invalid"
        assert trace.snapshot()[0]["detail"]["parse_reason_code"] == reason


def test_budget_filter_prevents_soft_premium_from_admitting_over_budget_candidate() -> None:
    client = FakeClient(valid_payload(candidate_order=["b", "a"]))
    source = [
        {"candidate_id": "a", "price": 500, "commute_minutes": 35},
        {"candidate_id": "b", "price": 1200, "commute_minutes": 10},
    ]
    policy = PreferenceResolver(client, enabled=True).resolve(
        task_type="full_itinerary",
        candidates=source,
        soft_preferences="位置好愿意贵20%",
        hard_constraints={"budget_max_cny": 600},
    )

    assert client.calls == 0
    assert policy.candidate_order == ["a"]
    assert policy.acceptable_tradeoffs == {}
    assert policy.fallback_reason == "fewer_than_two_legal_candidates"


def test_same_candidate_set_is_called_once_including_after_fallback() -> None:
    client = FakeClient(valid_payload(confidence=0.2))
    resolver = PreferenceResolver(client, enabled=True, request_scope="req-1")
    kwargs = {
        "task_type": "full_itinerary",
        "candidates": candidates(),
        "soft_preferences": "通勤优先",
        "hard_constraints": {"_constraint_revision": 3},
    }

    first = resolver.resolve(**kwargs)
    second = resolver.resolve(**kwargs)

    assert client.calls == 1
    assert first == second
    assert second.fallback_reason == "low_confidence"


def test_changed_constraint_revision_invalidates_preference_cache() -> None:
    client = FakeClient(valid_payload())
    resolver = PreferenceResolver(client, enabled=True, request_scope="req-1")
    resolver.resolve(
        task_type="full_itinerary", candidates=candidates(), soft_preferences="通勤优先",
        hard_constraints={"_constraint_revision": 1},
    )
    resolver.resolve(
        task_type="full_itinerary", candidates=candidates(), soft_preferences="通勤优先",
        hard_constraints={"_constraint_revision": 2},
    )

    assert client.calls == 2
