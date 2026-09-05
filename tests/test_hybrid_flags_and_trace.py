from __future__ import annotations

import textwrap
from dataclasses import dataclass

from travel_agent.hybrid_planning.intent_normalizer import IntentNormalizer
from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.hybrid_planning.preference_resolver import PreferenceResolver
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog
from travel_agent.settings import load_settings


@dataclass
class CountingClient:
    calls: int = 0

    def complete_json(self, **_kwargs) -> StructuredLLMResponse:
        self.calls += 1
        return StructuredLLMResponse({"normalized_interests": []}, model="fake")


def test_hybrid_flags_default_off(tmp_path, monkeypatch) -> None:
    for name in (
        "ENABLE_LLM_INTENT_NORMALIZER",
        "ENABLE_LLM_PREFERENCE_RESOLVER",
        "ENABLE_STRUCTURED_DURATION_ESTIMATOR",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.enable_llm_intent_normalizer is False
    assert settings.enable_llm_preference_resolver is False
    assert settings.enable_structured_duration_estimator is False


def test_hybrid_flags_load_from_toml_and_env_wins(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [hybrid_planning]
            enable_llm_intent_normalizer = true
            enable_llm_preference_resolver = true
            enable_structured_duration_estimator = true
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENABLE_LLM_PREFERENCE_RESOLVER", "false")

    settings = load_settings(config)

    assert settings.enable_llm_intent_normalizer is True
    assert settings.enable_llm_preference_resolver is False
    assert settings.enable_structured_duration_estimator is True


def test_closed_llm_flags_never_call_and_emit_request_owned_trace() -> None:
    client = CountingClient()
    trace = AgentTraceLog("request-current")
    IntentNormalizer(client, enabled=False).normalize("无法识别的兴趣", trace=trace)
    PreferenceResolver(client, enabled=False).resolve(
        task_type="full_itinerary",
        candidates=[
            {"candidate_id": "a", "price": 1},
            {"candidate_id": "b", "price": 2},
        ],
        soft_preferences="价格优先",
        trace=trace,
    )

    assert client.calls == 0
    entries = trace.snapshot()
    assert [item["kind"] for item in entries] == [
        "hybrid_intent_normalizer",
        "hybrid_preference_resolver",
    ]
    assert all(item["request_id"] == "request-current" for item in entries)
    assert all(item["detail"]["called"] is False for item in entries)


def test_trace_records_structured_output_model_usage_and_latency() -> None:
    class Client:
        def complete_json(self, **_kwargs):
            return StructuredLLMResponse(
                {
                    "normalized_interests": [
                        {
                            "label": "culture",
                            "source_text": "看点人文气息",
                            "confidence": 0.9,
                            "reason": "人文体验",
                            "polarity": "prefer",
                        }
                        ]
                        , "unresolved": []
                    },
                model="model-x",
                latency_ms=23,
                token_usage={"input_tokens": 12, "output_tokens": 4},
            )

    trace = AgentTraceLog("request-observed")
    IntentNormalizer(Client(), enabled=True).normalize("看点人文气息", trace=trace)
    detail = trace.snapshot()[0]["detail"]

    assert detail["eligible"] is True
    assert detail["called"] is True
    assert detail["model"] == "model-x"
    assert detail["structured_output"]["normalized_interests"][0]["label"] == "culture"
    assert detail["confidence"] == 0.9
    assert detail["latency_ms"] == 23
    assert detail["token_usage"] == {"input_tokens": 12, "output_tokens": 4}
