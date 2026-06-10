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
