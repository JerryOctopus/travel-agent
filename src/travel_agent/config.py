from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: int

    @property
    def enabled(self) -> bool:
        return self.provider != "rule" and bool(self.api_key)


@dataclass(frozen=True)
class ToolProviderConfig:
    provider: str
    amap_api_key: str | None
    amap_base_url: str
    timeout_seconds: int

    @property
    def amap_enabled(self) -> bool:
        return self.provider == "amap" and bool(self.amap_api_key)


def load_llm_config() -> LLMConfig:
    provider = os.getenv("TRAVEL_AGENT_LLM_PROVIDER", "rule").strip().lower()
    return LLMConfig(
        provider=provider,
        api_key=(
            _empty_to_none(os.getenv("TRAVEL_AGENT_LLM_API_KEY"))
            or _provider_api_key(provider)
        ),
        base_url=_env_or_default("TRAVEL_AGENT_LLM_BASE_URL", _default_base_url(provider)).rstrip("/"),
        model=_env_or_default("TRAVEL_AGENT_LLM_MODEL", _default_model(provider)),
        timeout_seconds=_int_env("TRAVEL_AGENT_LLM_TIMEOUT_SECONDS", 20),
    )


def load_tool_provider_config() -> ToolProviderConfig:
    return ToolProviderConfig(
        provider=os.getenv("TRAVEL_AGENT_TOOL_PROVIDER", "local").strip().lower(),
        amap_api_key=_empty_to_none(os.getenv("TRAVEL_AGENT_AMAP_API_KEY")),
        amap_base_url=_env_or_default(
            "TRAVEL_AGENT_AMAP_BASE_URL",
            "https://restapi.amap.com",
        ).rstrip("/"),
        timeout_seconds=_int_env("TRAVEL_AGENT_TOOL_TIMEOUT_SECONDS", 5),
    )


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _provider_api_key(provider: str) -> str | None:
    if provider in {"google", "gemini", "google-gemini"}:
        names = ("TRAVEL_AGENT_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    elif provider in {"siliconflow", "silicon-flow", "silicon_flow"}:
        names = ("TRAVEL_AGENT_SILICONFLOW_API_KEY", "SILICONFLOW_API_KEY")
    else:
        names = (f"TRAVEL_AGENT_{provider.upper().replace('-', '_')}_API_KEY",)
    for name in names:
        value = _empty_to_none(os.getenv(name))
        if value:
            return value
    return None


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _default_base_url(provider: str) -> str:
    if provider in {"freellmapi", "free_llm_api", "free-llm-api"}:
        return "http://localhost:31415/v1"
    if provider in {"freellmapi-docker", "freellmapi_server", "freellmapi-server"}:
        return "http://localhost:3001/v1"
    if provider in {"qwen", "dashscope", "aliyun"}:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "deepseek":
        return "https://api.deepseek.com/v1"
    if provider in {"zhipu", "glm", "zai"}:
        return "https://open.bigmodel.cn/api/paas/v4"
    if provider in {"google", "gemini", "google-gemini"}:
        return "https://generativelanguage.googleapis.com/v1beta/openai"
    if provider in {"siliconflow", "silicon-flow", "silicon_flow"}:
        return "https://api.siliconflow.cn/v1"
    return "https://api.openai.com/v1"


def _default_model(provider: str) -> str:
    if provider in {
        "freellmapi",
        "free_llm_api",
        "free-llm-api",
        "freellmapi-docker",
        "freellmapi_server",
        "freellmapi-server",
    }:
        return "auto"
    if provider in {"qwen", "dashscope", "aliyun"}:
        return "qwen-plus"
    if provider == "deepseek":
        return "deepseek-v4-flash"
    if provider in {"zhipu", "glm", "zai"}:
        return "glm-4.7-flash"
    if provider in {"google", "gemini", "google-gemini"}:
        return "gemini-2.5-flash"
    if provider in {"siliconflow", "silicon-flow", "silicon_flow"}:
        return "deepseek-ai/DeepSeek-V4-Flash"
    return "gpt-4o-mini"
