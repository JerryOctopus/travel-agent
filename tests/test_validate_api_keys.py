import json
from types import SimpleNamespace

from scripts import validate_api_keys
from travel_agent.harness import preflight


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _settings():
    return SimpleNamespace(
        amap=SimpleNamespace(
            rest_enabled=True,
            web_key="secret",
            base_url="https://restapi.amap.com",
            timeout_seconds=5,
        )
    )


def test_llm_preflight_uses_shared_provider_aware_probe(monkeypatch) -> None:
    settings = SimpleNamespace(llm=SimpleNamespace(enabled=True))
    expected = {"ok": True, "provider_reported_model": "deepseek-v4-flash"}
    monkeypatch.setattr(validate_api_keys, "preflight_llm", lambda value: expected)

    assert validate_api_keys._check_llm(settings) is expected


def test_amap_preflight_requires_place_search_quota(monkeypatch) -> None:
    def fake_urlopen(url: str, timeout: int):
        if "/v5/place/text" in url:
            return _Response({
                "status": "0",
                "info": "USER_DAILY_QUERY_OVER_LIMIT",
                "infocode": "10044",
            })
        return _Response({"status": "1", "lives": [{"weather": "晴"}]})

    monkeypatch.setattr(preflight, "urlopen", fake_urlopen)

    report = validate_api_keys._check_amap(_settings())

    assert report["ok"] is False
    assert report["checks"]["weather"]["ok"] is True
    assert report["checks"]["place_search"] == {
        "ok": False,
        "detail": "USER_DAILY_QUERY_OVER_LIMIT",
        "infocode": "10044",
    }


def test_amap_preflight_passes_only_when_weather_and_place_search_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "urlopen",
        lambda *_args, **_kwargs: _Response({"status": "1", "pois": []}),
    )

    report = validate_api_keys._check_amap(_settings())

    assert report["ok"] is True
    assert report["checks"]["weather"]["ok"] is True
    assert report["checks"]["place_search"]["ok"] is True


def test_combined_preflight_stops_before_llm_when_amap_fails(monkeypatch) -> None:
    calls: list[str] = []
    settings = object()
    monkeypatch.setattr(
        validate_api_keys,
        "_check_amap",
        lambda _settings: calls.append("amap") or {"ok": False},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_llm",
        lambda _settings: calls.append("llm") or {"ok": True},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_judge",
        lambda _settings: calls.append("judge") or {"ok": True},
    )

    report = validate_api_keys.validate_all(settings)

    assert calls == ["amap"]
    assert report["llm"]["skipped"] is True
    assert report["judge"]["skipped"] is True


def test_combined_preflight_treats_local_fallback_as_not_release_ready(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        validate_api_keys,
        "_check_amap",
        lambda _settings: {"ok": True, "skipped": True},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_llm",
        lambda _settings: calls.append("llm") or {"ok": True},
    )

    report = validate_api_keys.validate_all(object())

    assert calls == []
    assert report["llm"]["ok"] is False


def test_combined_preflight_stops_before_judge_when_llm_fails(monkeypatch) -> None:
    calls: list[str] = []
    settings = object()
    monkeypatch.setattr(
        validate_api_keys,
        "_check_amap",
        lambda _settings: calls.append("amap") or {"ok": True},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_llm",
        lambda _settings: calls.append("llm") or {"ok": False},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_judge",
        lambda _settings: calls.append("judge") or {"ok": True},
    )

    report = validate_api_keys.validate_all(settings)

    assert calls == ["amap", "llm"]
    assert report["judge"]["skipped"] is True


def test_combined_preflight_calls_all_services_in_spend_safe_order(monkeypatch) -> None:
    calls: list[str] = []
    settings = object()
    monkeypatch.setattr(
        validate_api_keys,
        "_check_amap",
        lambda _settings: calls.append("amap") or {"ok": True},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_llm",
        lambda _settings: calls.append("llm") or {"ok": True},
    )
    monkeypatch.setattr(
        validate_api_keys,
        "_check_judge",
        lambda _settings: calls.append("judge") or {"ok": True},
    )

    report = validate_api_keys.validate_all(settings)

    assert calls == ["amap", "llm", "judge"]
    assert all(item["ok"] for item in report.values())
