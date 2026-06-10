from __future__ import annotations

import textwrap

from travel_agent.settings import load_settings


def test_env_overrides_toml(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [llm]
            provider = "deepseek"
            api_key = "from_toml"

            [amap]
            web_key = "amap_toml"
            js_key = "js_toml"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "from_env")
    settings = load_settings(config)

    assert settings.llm.provider == "deepseek"
    assert settings.llm.api_key == "from_env"  # env 优先于 toml
    assert settings.llm.base_url == "https://api.deepseek.com/v1"
    assert settings.llm.enabled is True
    assert settings.amap.web_key == "amap_toml"
    assert settings.amap.rest_enabled is True
    assert settings.amap.js_key == "js_toml"


def test_defaults_disabled_without_keys(tmp_path, monkeypatch):
    for var in [
        "TRAVEL_AGENT_LLM_PROVIDER",
        "TRAVEL_AGENT_LLM_API_KEY",
        "TRAVEL_AGENT_AMAP_WEB_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    settings = load_settings(tmp_path / "missing.toml")
    assert settings.llm.enabled is False
    assert settings.amap.rest_enabled is False
    assert settings.agent.recursion_limit == 20
