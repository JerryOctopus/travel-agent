from __future__ import annotations

from types import SimpleNamespace

import pytest

from travel_agent.hybrid_planning.llm_types import StructuredOutputError
from travel_agent.hybrid_planning.structured_client import ProductionStructuredLLMClient


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def bind(self, **_kwargs):
        return self

    def invoke(self, _messages, config=None):
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(
            content=output,
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )


def client_for(monkeypatch, outputs):
    model = FakeModel(outputs)
    monkeypatch.setattr("travel_agent.agent.runtime._build_chat_model", lambda *_a, **_k: model)
    settings = SimpleNamespace(
        llm=SimpleNamespace(timeout_seconds=10, model="deepseek-chat")
    )
    return ProductionStructuredLLMClient(settings, "contract-test"), model


def test_valid_json_and_code_fence_are_parsed_without_business_repair(monkeypatch) -> None:
    client, model = client_for(monkeypatch, ['```json\n{"ok": true}\n```'])

    result = client.complete_json(system_prompt="s", user_prompt="u", schema_version="v")

    assert result.payload == {"ok": True}
    assert result.parse_status == "format_repaired"
    assert result.parse_reason_code == "markdown_code_fence"
    assert result.repair_attempted is False
    assert result.attempts == 1
    assert model.calls == 1
    assert result.raw_output_summary["has_markdown_fence"] is True


def test_non_json_text_fails_without_guessing_or_retry(monkeypatch) -> None:
    client, model = client_for(monkeypatch, ["I think candidate a is best"])

    with pytest.raises(StructuredOutputError) as caught:
        client.complete_json(system_prompt="s", user_prompt="u", schema_version="v")

    assert caught.value.reason_code == "non_json_text"
    assert caught.value.repair_attempted is False
    assert model.calls == 1


def test_valid_json_with_wrong_top_level_has_exact_reason(monkeypatch) -> None:
    client, model = client_for(monkeypatch, ["[]"])

    with pytest.raises(StructuredOutputError) as caught:
        client.complete_json(system_prompt="s", user_prompt="u", schema_version="v")

    assert caught.value.reason_code == "top_level_not_object"
    assert caught.value.repair_attempted is False
    assert model.calls == 1


def test_json_syntax_gets_at_most_one_format_only_repair(monkeypatch) -> None:
    client, model = client_for(monkeypatch, ['{"ok": true,}', '{"ok": true}'])

    result = client.complete_json(system_prompt="s", user_prompt="u", schema_version="v")

    assert result.payload == {"ok": True}
    assert result.parse_status == "format_repaired"
    assert result.parse_reason_code == "repair_success"
    assert result.repair_attempted is True
    assert result.attempts == 2
    assert model.calls == 2
    assert result.token_usage["total_tokens"] == 10


def test_failed_repair_is_bounded_to_two_attempts(monkeypatch) -> None:
    client, model = client_for(monkeypatch, ['{"ok": true,}', '{still invalid'])

    with pytest.raises(StructuredOutputError) as caught:
        client.complete_json(system_prompt="s", user_prompt="u", schema_version="v")

    assert caught.value.reason_code == "repair_failed"
    assert caught.value.repair_attempted is True
    assert caught.value.attempts == 2
    assert model.calls == 2


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (TimeoutError("slow"), "timeout", "timeout"),
        (RuntimeError("provider unavailable"), "provider_error", "provider_error"),
    ],
)
def test_transport_failures_have_exact_status(monkeypatch, error, status, reason) -> None:
    client, model = client_for(monkeypatch, [error])

    with pytest.raises(StructuredOutputError) as caught:
        client.complete_json(system_prompt="s", user_prompt="u", schema_version="v")

    assert caught.value.parse_status == status
    assert caught.value.reason_code == reason
    assert model.calls == 1
