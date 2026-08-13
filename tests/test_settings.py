from __future__ import annotations

import textwrap

from travel_agent.settings import load_settings
from travel_agent.storage.user_memory import get_user_memory_service


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
    monkeypatch.delenv("TRAVEL_AGENT_LLM_PROVIDER", raising=False)
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
    assert settings.orchestration.variant_token_budget == 0
    assert not hasattr(settings.orchestration, "variant")
    assert settings.memory.backend == "json"


def test_zhipu_provider_uses_glm_flash_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.llm.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert settings.llm.model == "glm-4.7-flash"
    assert settings.llm.enabled is True


def test_google_provider_uses_gemini_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "google")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "gemini-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.llm.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert settings.llm.model == "gemini-2.5-flash"
    assert settings.llm.enabled is True


def test_google_provider_override_uses_gemini_key_not_toml_key(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [llm]
            provider = "qwen"
            api_key = "qwen-secret"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_API_KEY", raising=False)

    settings = load_settings(config)

    assert settings.llm.api_key == "gemini-secret"
    assert settings.llm.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_siliconflow_provider_uses_deepseek_v4_flash_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "siliconflow")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "siliconflow-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.llm.api_key == "siliconflow-test-key"
    assert settings.llm.base_url == "https://api.siliconflow.cn/v1"
    assert settings.llm.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert settings.llm.enabled is True


def test_freellmapi_provider_uses_local_router_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "freellmapi")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "freellmapi-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.llm.base_url == "http://localhost:31415/v1"
    assert settings.llm.model == "auto"
    assert settings.llm.enabled is True


def test_freellmapi_docker_provider_uses_server_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "freellmapi-docker")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "freellmapi-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.llm.base_url == "http://localhost:3001/v1"
    assert settings.llm.model == "auto"
    assert settings.llm.enabled is True


def test_freellmapi_judge_uses_router_defaults_and_provider_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_JUDGE_PROVIDER", "freellmapi")
    monkeypatch.setenv("FREELLMAPI_API_KEY", "freellmapi-judge-key")
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.evaluation.judge.provider == "freellmapi"
    assert settings.evaluation.judge.api_key == "freellmapi-judge-key"
    assert settings.evaluation.judge.base_url == "http://localhost:31415/v1"
    assert settings.evaluation.judge.model == "auto"
    assert settings.evaluation.judge.enabled is True
    assert settings.evaluation.judge.requests_per_second == 0.0


def test_siliconflow_judge_uses_dedicated_provider_key(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [evaluation.judge]
            provider = "siliconflow"
            api_key = "stale-router-key"
            base_url = "https://api.siliconflow.cn/v1"
            model = "Qwen/Qwen3.5-397B-A17B"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SILICONFLOW_API_KEY", "siliconflow-judge-key")
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_API_KEY", raising=False)

    settings = load_settings(config)

    assert settings.evaluation.judge.provider == "siliconflow"
    assert settings.evaluation.judge.api_key == "siliconflow-judge-key"
    assert settings.evaluation.judge.base_url == "https://api.siliconflow.cn/v1"
    assert settings.evaluation.judge.model == "Qwen/Qwen3.5-397B-A17B"


def test_siliconflow_judge_accepts_explicit_toml_key(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [evaluation.judge]
            provider = "siliconflow"
            api_key = "siliconflow-config-key"
            """
        ),
        encoding="utf-8",
    )
    for name in [
        "TRAVEL_AGENT_JUDGE_API_KEY",
        "TRAVEL_AGENT_SILICONFLOW_API_KEY",
        "SILICONFLOW_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings(config)

    assert settings.evaluation.judge.api_key == "siliconflow-config-key"
    assert settings.evaluation.judge.enabled is True


def test_freellmapi_docker_judge_uses_server_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_JUDGE_PROVIDER", "freellmapi-docker")
    monkeypatch.setenv("TRAVEL_AGENT_JUDGE_API_KEY", "freellmapi-judge-key")
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_MODEL", raising=False)

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.evaluation.judge.base_url == "http://localhost:3001/v1"
    assert settings.evaluation.judge.model == "auto"
    assert settings.evaluation.judge.enabled is True


def test_provider_env_override_does_not_reuse_toml_endpoint(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [llm]
            provider = "qwen"
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            model = "qwen3.7-plus"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    settings = load_settings(config)

    assert settings.llm.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert settings.llm.model == "glm-4.7-flash"


def test_llm_thinking_mode_can_be_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_LLM_THINKING_ENABLED", "true")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_REQUESTS_PER_SECOND", "0.05")

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.llm.thinking_enabled is True
    assert settings.llm.requests_per_second == 0.05


def test_independent_judge_settings_do_not_reuse_agent_llm(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [llm]
            provider = "deepseek"
            api_key = "agent-key"
            model = "deepseek-v4-flash"

            [evaluation.judge]
            provider = "zhipu"
            model = "glm-4.7-flash"
            requests_per_second = 0.0125
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TRAVEL_AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TRAVEL_AGENT_JUDGE_API_KEY", "judge-key")

    settings = load_settings(config)

    assert settings.llm.model == "deepseek-v4-flash"
    assert settings.llm.api_key == "agent-key"
    assert settings.evaluation.judge.model == "glm-4.7-flash"
    assert settings.evaluation.judge.api_key == "judge-key"
    assert settings.evaluation.judge.enabled is True
    assert settings.evaluation.judge.requests_per_second == 0.0125


def test_independent_judge_defaults_to_gemini_36_flash(tmp_path, monkeypatch):
    for name in [
        "TRAVEL_AGENT_JUDGE_PROVIDER",
        "TRAVEL_AGENT_JUDGE_API_KEY",
        "TRAVEL_AGENT_JUDGE_BASE_URL",
        "TRAVEL_AGENT_JUDGE_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-judge-key")

    settings = load_settings(tmp_path / "missing.toml")

    assert settings.evaluation.judge.provider == "google"
    assert settings.evaluation.judge.api_key == "gemini-judge-key"
    assert settings.evaluation.judge.base_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert settings.evaluation.judge.model == "gemini-3.6-flash"
    assert settings.evaluation.judge.enabled is True
    assert settings.evaluation.judge.requests_per_second == 0.3


def test_judge_provider_override_does_not_reuse_toml_endpoint_or_model(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(
            """
            [evaluation.judge]
            provider = "zhipu"
            api_key = "zhipu-secret"
            base_url = "https://open.bigmodel.cn/api/paas/v4"
            model = "glm-4.7-flash"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAVEL_AGENT_JUDGE_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-judge-key")
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_JUDGE_MODEL", raising=False)

    settings = load_settings(config)

    assert settings.evaluation.judge.api_key == "gemini-judge-key"
    assert settings.evaluation.judge.base_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert settings.evaluation.judge.model == "gemini-3.6-flash"


def test_postgres_memory_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_MEMORY_BACKEND", "postgres")
    monkeypatch.delenv("TRAVEL_AGENT_DATABASE_URL", raising=False)
    settings = load_settings(tmp_path / "missing.toml")

    try:
        get_user_memory_service(settings.memory)
    except RuntimeError as exc:
        assert "TRAVEL_AGENT_DATABASE_URL" in str(exc)
    else:
        raise AssertionError("postgres memory must fail without database URL")
