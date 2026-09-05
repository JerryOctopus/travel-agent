from __future__ import annotations

from dataclasses import dataclass

import pytest

from travel_agent.hybrid_planning.intent_normalizer import IntentNormalizer
from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog


@dataclass
class FakeClient:
    payload: dict | None = None
    error: Exception | None = None
    calls: int = 0

    def complete_json(self, **_kwargs) -> StructuredLLMResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.payload is None:
            raise ValueError("invalid JSON")
        return StructuredLLMResponse(
            payload=self.payload,
            model="fake-model",
            latency_ms=12.5,
            token_usage={"input_tokens": 10, "output_tokens": 5},
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("老城区烟火气", {("local_life", "prefer")}),
        ("有年代感的建筑", {("architecture", "prefer")}),
        ("工业遗迹和城市更新", {("industrial_heritage", "prefer"), ("urban_renewal", "prefer")}),
        ("不要太商业化", {("commercialized", "avoid")}),
        ("适合建筑摄影", {("architecture", "prefer"), ("photography", "prefer")}),
        ("带孩子动手体验", {("family", "prefer"), ("hands_on", "prefer")}),
        ("海边发呆喝咖啡", {("nature", "prefer"), ("cafe", "prefer")}),
    ],
)
def test_deterministic_aliases_are_bounded_and_do_not_call_llm(text, expected) -> None:
    client = FakeClient(payload={})
    result = IntentNormalizer(client, enabled=True).normalize(text, source_turn="turn_3")

    assert {(item.label, item.polarity) for item in result.normalized_interests} == expected
    assert all(item.source_turn == "turn_3" for item in result.normalized_interests)
    assert result.unresolved == []
    assert client.calls == 0


def test_multiple_raw_interests_mix_fast_path_and_llm() -> None:
    client = FakeClient(
        payload={
            "normalized_interests": [
                {
                    "label": "culture",
                    "source_text": "看点有人文味的地方",
                    "confidence": 0.82,
                    "reason": "人文体验",
                    "polarity": "prefer",
                }
            ],
            "unresolved": [],
        }
    )
    result = IntentNormalizer(client, enabled=True).normalize(
        ["老城区烟火气", "看点有人文味的地方"]
    )

    assert {item.label for item in result.normalized_interests} == {"local_life", "culture"}
    assert client.calls == 1


def test_negative_interest_is_not_promoted_to_positive() -> None:
    result = IntentNormalizer(enabled=False).normalize("喜欢历史，但不喜欢博物馆")
    by_label = {item.label: item.polarity for item in result.normalized_interests}

    assert by_label["history"] == "prefer"
    assert by_label["museum"] == "avoid"


def test_unknown_interest_is_preserved_when_feature_is_disabled() -> None:
    result = IntentNormalizer(enabled=False).normalize("想体验一种说不清的城市气质")

    assert result.normalized_interests == []
    assert result.unresolved == ["想体验一种说不清的城市气质"]
    assert result.fallback_reason == "feature_disabled"


def test_out_of_taxonomy_label_is_rejected_fail_closed_and_traced() -> None:
    client = FakeClient(
        payload={
            "normalized_interests": [
                {
                    "label": "invented_secret_label",
                    "source_text": "想找很特别的体验",
                    "confidence": 0.99,
                    "reason": "invented",
                    "polarity": "prefer",
                }
            ],
            "unresolved": [],
        }
    )
    trace = AgentTraceLog("request-1")
    result = IntentNormalizer(client, enabled=True).normalize("想找很特别的体验", trace=trace)

    assert result.normalized_interests == []
    assert result.unresolved == ["想找很特别的体验"]
    assert result.fallback_reason == "taxonomy_invalid"
    assert trace.snapshot()[0]["detail"]["out_of_taxonomy_label_count"] == 1


def test_low_confidence_does_not_become_a_preference() -> None:
    client = FakeClient(
        payload={
            "normalized_interests": [
                {
                    "label": "culture",
                    "source_text": "想找很特别的体验",
                    "confidence": 0.4,
                    "reason": "不确定",
                    "polarity": "prefer",
                }
            ],
            "unresolved": [],
        }
    )
    result = IntentNormalizer(client, enabled=True).normalize("想找很特别的体验")

    assert result.normalized_interests == []
    assert result.unresolved == ["想找很特别的体验"]
    assert result.fallback_reason == "low_confidence"


def test_attempted_hard_constraint_change_is_rejected_and_traced() -> None:
    client = FakeClient(
        payload={
            "normalized_interests": [
                {
                    "label": "culture",
                    "source_text": "看点人文气息",
                    "confidence": 0.9,
                    "reason": "人文体验",
                    "polarity": "prefer",
                }
            ],
            "unresolved": [],
            "must_visit": ["invented venue"],
        }
    )
    trace = AgentTraceLog("request-hard")
    result = IntentNormalizer(client, enabled=True).normalize("看点人文气息", trace=trace)

    assert result.normalized_interests == []
    assert result.unresolved == ["看点人文气息"]
    assert result.fallback_reason == "attempted_hard_constraint_change"
    assert trace.snapshot()[0]["detail"]["attempted_hard_constraint_change"] is True


@pytest.mark.parametrize(
    ("error", "fallback"),
    [(TimeoutError("slow"), "timeout"), (ValueError("invalid JSON"), "schema_invalid")],
)
def test_llm_failures_fall_back_without_losing_raw_text(error, fallback) -> None:
    client = FakeClient(error=error)
    result = IntentNormalizer(client, enabled=True).normalize("想找很特别的体验")

    assert result.raw_interests == ["想找很特别的体验"]
    assert result.unresolved == result.raw_interests
    assert result.fallback_reason == fallback


def test_deterministic_result_has_versioned_explainable_shape() -> None:
    payload = IntentNormalizer(enabled=False).normalize("老城区烟火气").to_dict()

    assert payload["taxonomy_version"] == "travel-interest-v1"
    assert payload["normalized_interests"][0] == {
        "label": "local_life",
        "source_text": "老城区烟火气",
        "confidence": 1.0,
        "reason": "社区日常与本地生活氛围",
        "source_turn": None,
        "polarity": "prefer",
        "basis": "deterministic_alias",
    }


@pytest.mark.parametrize(
    ("text", "request_type", "reason"),
    [
        ("", "full_itinerary", "empty_interest"),
        ("预算1000元，两个人", "full_itinerary", "constraints_only"),
        ("想去西湖", "full_itinerary", "must_visit_only"),
        ("西湖门票多少钱", "full_itinerary", "fact_query"),
        ("想体验当地人的生活气质", "route_plan", "request_type_not_eligible"),
    ],
)
def test_intent_ineligible_reason_codes_do_not_call(text, request_type, reason) -> None:
    client = FakeClient(payload={})
    trace = AgentTraceLog("eligibility")
    IntentNormalizer(client, enabled=True).normalize(
        text, request_type=request_type, trace=trace
    )

    assert client.calls == 0
    assert trace.snapshot()[0]["detail"]["skipped_reason"] == reason


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"normalized_interests": []}, "missing_field"),
        ({"normalized_interests": "bad", "unresolved": []}, "type_mismatch"),
        (
            {
                "normalized_interests": [{
                    "label": "culture", "source_text": "人文气质", "confidence": "0.9",
                    "polarity": "prefer",
                }],
                "unresolved": [],
            },
            "confidence_type_mismatch",
        ),
        (
            {
                "normalized_interests": [{
                    "label": "culture", "source_text": "人文气质", "confidence": 1.2,
                    "polarity": "prefer",
                }],
                "unresolved": [],
            },
            "confidence_out_of_range",
        ),
        (
            {
                "normalized_interests": [{
                    "label": "culture", "source_text": "模型改写的文本", "confidence": 0.9,
                    "polarity": "prefer",
                }],
                "unresolved": [],
            },
            "source_text_mismatch",
        ),
    ],
)
def test_intent_schema_failures_have_exact_reason_codes(payload, reason) -> None:
    trace = AgentTraceLog("schema")
    result = IntentNormalizer(FakeClient(payload=payload), enabled=True).normalize(
        "喜欢人文气质", trace=trace
    )

    assert result.fallback_reason == "schema_invalid"
    assert result.raw_interests == ["喜欢人文气质"]
    assert result.unresolved == result.raw_interests
    assert trace.snapshot()[0]["detail"]["parse_reason_code"] == reason


def test_traceable_source_substring_is_accepted_and_partial_low_confidence_is_unresolved() -> None:
    payload = {
        "normalized_interests": [
            {"label": "culture", "source_text": "人文气质", "confidence": 0.9, "polarity": "prefer"},
            {"label": "history", "source_text": "人文气质", "confidence": 0.3, "polarity": "prefer"},
        ],
        "unresolved": [],
    }
    result = IntentNormalizer(FakeClient(payload=payload), enabled=True).normalize(
        "喜欢有层次的人文气质"
    )

    assert [item.label for item in result.normalized_interests] == ["culture"]
    assert result.unresolved == ["喜欢有层次的人文气质"]
    assert result.fallback_reason is None


def test_same_intent_fingerprint_is_called_once_and_cached() -> None:
    client = FakeClient(payload={
        "normalized_interests": [
            {"label": "culture", "source_text": "人文气质", "confidence": 0.9, "polarity": "prefer"}
        ],
        "unresolved": [],
    })
    normalizer = IntentNormalizer(client, enabled=True, request_scope="req-1")

    first = normalizer.normalize("喜欢人文气质")
    second = normalizer.normalize("喜欢人文气质")

    assert client.calls == 1
    assert first == second


def test_unknown_interest_can_be_validly_reported_unresolved() -> None:
    trace = AgentTraceLog("unknown")
    result = IntentNormalizer(
        FakeClient(payload={
            "normalized_interests": [],
            "unresolved": ["未来城市气质"],
        }),
        enabled=True,
    ).normalize("喜欢一种难以归类的未来城市气质", trace=trace)

    assert result.normalized_interests == []
    assert result.unresolved == ["喜欢一种难以归类的未来城市气质"]
    assert result.fallback_reason is None
    assert trace.snapshot()[0]["detail"]["validation_status"] == "valid_with_unresolved"


def test_llm_fallback_never_overwrites_deterministic_interest() -> None:
    client = FakeClient(error=ValueError("bad schema"))
    result = IntentNormalizer(client, enabled=True).normalize(
        ["老城区烟火气", "喜欢一种难以归类的未来城市气质"]
    )

    assert [(item.label, item.basis) for item in result.normalized_interests] == [
        ("local_life", "deterministic_alias")
    ]
    assert result.raw_interests == ["老城区烟火气", "喜欢一种难以归类的未来城市气质"]
    assert result.unresolved == ["喜欢一种难以归类的未来城市气质"]
    assert result.fallback_reason == "schema_invalid"
