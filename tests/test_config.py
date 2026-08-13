from travel_agent.config import load_llm_config


def test_load_llm_config_defaults_to_rule_provider(monkeypatch) -> None:
    monkeypatch.delenv("TRAVEL_AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_API_KEY", raising=False)

    config = load_llm_config()

    assert config.provider == "rule"
    assert config.api_key is None
    assert config.enabled is False


def test_load_llm_config_enables_non_rule_provider_with_key(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_BASE_URL", "https://api.deepseek.com/v1/")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_MODEL", "deepseek-chat")

    config = load_llm_config()

    assert config.provider == "deepseek"
    assert config.api_key == "test-key"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.model == "deepseek-chat"
    assert config.enabled is True


def test_load_llm_config_uses_qwen_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.model == "qwen-plus"
    assert config.enabled is True


def test_load_llm_config_uses_zhipu_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert config.model == "glm-4.7-flash"
    assert config.enabled is True


def test_load_llm_config_uses_gemini_alias_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "gemini-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert config.model == "gemini-2.5-flash"
    assert config.enabled is True


def test_load_llm_config_accepts_standard_gemini_api_key(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "google")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "standard-gemini-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.api_key == "standard-gemini-key"
    assert config.enabled is True


def test_load_llm_config_uses_siliconflow_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "silicon-flow")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "siliconflow-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.api_key == "siliconflow-test-key"
    assert config.base_url == "https://api.siliconflow.cn/v1"
    assert config.model == "deepseek-ai/DeepSeek-V4-Flash"
    assert config.enabled is True


def test_load_llm_config_uses_freellmapi_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "freellmapi")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "freellmapi-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.base_url == "http://localhost:31415/v1"
    assert config.model == "auto"
    assert config.enabled is True


def test_load_llm_config_uses_freellmapi_docker_defaults(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_AGENT_LLM_PROVIDER", "freellmapi-docker")
    monkeypatch.setenv("TRAVEL_AGENT_LLM_API_KEY", "freellmapi-test-key")
    monkeypatch.delenv("TRAVEL_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_LLM_MODEL", raising=False)

    config = load_llm_config()

    assert config.base_url == "http://localhost:3001/v1"
    assert config.model == "auto"
    assert config.enabled is True
