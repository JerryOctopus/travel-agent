from __future__ import annotations

import json
from types import SimpleNamespace

from travel_agent.harness.preflight import preflight_llm


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_deepseek_v4_preflight_disables_thinking_and_records_reported_model(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_urlopen(request, *, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({"model": "DeepSeek-V4-Flash-0731"})

    monkeypatch.setattr("travel_agent.harness.preflight.urlopen", fake_urlopen)
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="deepseek",
            model="deepseek-v4-flash",
            api_key="secret",
            base_url="https://api.deepseek.com/v1",
            timeout_seconds=12,
            temperature=0.2,
            thinking_enabled=False,
        )
    )

    result = preflight_llm(settings)

    assert captured["payload"]["thinking"] == {"type": "disabled"}
    assert captured["payload"]["temperature"] == 0.2
    assert captured["timeout"] == 12
    assert result["provider_reported_model"] == "DeepSeek-V4-Flash-0731"
    assert result["thinking_enabled"] is False


def test_non_deepseek_preflight_does_not_send_provider_specific_thinking(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_urlopen(request, *, timeout):
        del timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response({"model": "other-model"})

    monkeypatch.setattr("travel_agent.harness.preflight.urlopen", fake_urlopen)
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            provider="other",
            model="other-model",
            api_key="secret",
            base_url="https://example.invalid/v1",
            timeout_seconds=12,
            temperature=0.2,
            thinking_enabled=False,
        )
    )

    result = preflight_llm(settings)

    assert "thinking" not in captured["payload"]
    assert result["provider_reported_model"] == "other-model"
