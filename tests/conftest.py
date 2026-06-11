from __future__ import annotations

from pathlib import Path

import pytest

from travel_agent.settings import get_settings, load_settings

_MISSING_CONFIG = Path(__file__).resolve().parent / "_no_such_config.toml"


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch, tmp_path):
    """测试环境忽略本地 config.toml，避免真实 key / 落盘数据干扰。"""
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("travel_agent.settings.DEFAULT_CONFIG_PATH", _MISSING_CONFIG)
    monkeypatch.setattr("travel_agent.storage.user_profile.DEFAULT_PROFILE_DIR", profiles)
    monkeypatch.setattr("travel_agent.agent.session.DEFAULT_ARTIFACT_DIR", artifacts)
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "rule")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_AMAP_WEB_KEY", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_AMAP_JS_KEY", raising=False)
    monkeypatch.setenv("TRAVEL_AGENT_LAYERED_ENABLED", "false")
    get_settings.cache_clear()

    import travel_agent.server as server_module

    server_module.MANAGER = server_module.SessionLifecycleManager(
        artifact_dir=artifacts,
        profile_dir=profiles,
    )

    yield
    get_settings.cache_clear()


@pytest.fixture
def offline_settings(isolate_settings):
    return load_settings()
