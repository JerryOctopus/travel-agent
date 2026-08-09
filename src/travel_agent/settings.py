"""统一配置加载。

设计目标（对应 PROJECT_PLAN「配置」一节）：

- 用一个 ``config.toml`` 集中管理 LLM key / 高德 Web 服务 key（REST）/
  高德 JS API key（前端地图），而不是散落的环境变量；
- 环境变量优先级高于 ``config.toml``，方便 CI / 临时覆盖；
- 在缺少任何 key 时仍能加载（返回 ``enabled=False``），保证「无 key 离线兜底」。

实现上为了兼容 Python 3.10（无 ``tomllib``），toml 解析做了优雅降级：
优先 ``tomllib``（3.11+），回退 ``tomli``，都没有则跳过 toml 仅用环境变量。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - 取决于运行环境
    try:
        import tomli as _toml  # type: ignore
    except ModuleNotFoundError:
        _toml = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "rule"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 30
    temperature: float = 0.2
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    @property
    def enabled(self) -> bool:
        """是否可以真正驱动 LLM（用于决定走 ReAct 还是离线兜底）。"""
        return self.provider != "rule" and bool(self.api_key)


@dataclass(frozen=True)
class AmapSettings:
    # 高德 Web 服务 key：REST（POI / 天气 / 路线）
    web_key: str | None = None
    base_url: str = "https://restapi.amap.com"
    # 高德 JS API key：前端地图渲染
    js_key: str | None = None
    # 高德 JS API 安全密钥（与 js_key 配对，需在加载地图脚本前注入）
    js_security_key: str | None = None
    timeout_seconds: int = 5

    @property
    def rest_enabled(self) -> bool:
        return bool(self.web_key)


@dataclass(frozen=True)
class AgentSettings:
    recursion_limit: int = 20
    max_tool_retries: int = 1
    request_timeout_seconds: int = 120


@dataclass(frozen=True)
class MemorySettings:
    compress_message_threshold: int = 8
    profile_only_token_threshold: int = 4000
    keep_recent_turns: int = 4
    profile_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "profiles")


@dataclass(frozen=True)
class McpSettings:
    use_mcp_tools: bool = False
    auto_start: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    server_name: str = "travel"
    timeout_seconds: int = 30


@dataclass(frozen=True)
class OrchestrationSettings:
    # V0–V3 消融实验四版本共用的 Token 硬上限；<=0 表示不设上限。
    # 生产入口固定使用 Multi-Agent Full（V3），不读取任何 variant 配置；
    # 版本选择只能通过 orchestration.variants.run_variant_turn 显式指定。
    variant_token_budget: int = 0
    variant_llm_call_budget: int = 0
    variant_tool_call_budget: int = 0


@dataclass(frozen=True)
class SkillsSettings:
    skills_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".storyline" / "skills")
    enabled: bool = True


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    amap: AmapSettings = field(default_factory=AmapSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    orchestration: OrchestrationSettings = field(default_factory=OrchestrationSettings)
    mcp: McpSettings = field(default_factory=McpSettings)
    skills: SkillsSettings = field(default_factory=SkillsSettings)


def _load_toml(path: Path) -> dict[str, Any]:
    if _toml is None or not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            return _toml.load(handle)
    except Exception:
        return {}


def _pick(env_name: str, toml_value: Any, default: Any) -> Any:
    raw = os.getenv(env_name)
    if raw is not None and raw.strip():
        return raw.strip()
    if toml_value is not None and toml_value != "":
        return toml_value
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out > 0 else default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _default_base_url(provider: str) -> str:
    provider = provider.lower()
    if provider in {"qwen", "dashscope", "aliyun"}:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "deepseek":
        return "https://api.deepseek.com/v1"
    return "https://api.openai.com/v1"


def _default_model(provider: str) -> str:
    provider = provider.lower()
    if provider in {"qwen", "dashscope", "aliyun"}:
        return "qwen-plus"
    if provider == "deepseek":
        return "deepseek-chat"
    return "gpt-4o-mini"


def load_settings(config_path: Path | str | None = None) -> Settings:
    """加载配置：环境变量优先，其次 ``config.toml``，最后内置默认值。"""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    data = _load_toml(path)
    llm_toml = data.get("llm", {}) if isinstance(data.get("llm"), dict) else {}
    amap_toml = data.get("amap", {}) if isinstance(data.get("amap"), dict) else {}
    agent_toml = data.get("agent", {}) if isinstance(data.get("agent"), dict) else {}
    memory_toml = data.get("memory", {}) if isinstance(data.get("memory"), dict) else {}
    orch_toml = data.get("orchestration", {}) if isinstance(data.get("orchestration"), dict) else {}
    mcp_toml = data.get("mcp", {}) if isinstance(data.get("mcp"), dict) else {}
    skills_toml = data.get("skills", {}) if isinstance(data.get("skills"), dict) else {}

    provider = str(_pick("TRAVEL_AGENT_LLM_PROVIDER", llm_toml.get("provider"), "rule")).lower()
    llm = LLMSettings(
        provider=provider,
        api_key=_empty_to_none(_pick("TRAVEL_AGENT_LLM_API_KEY", llm_toml.get("api_key"), None)),
        base_url=str(
            _pick("TRAVEL_AGENT_LLM_BASE_URL", llm_toml.get("base_url"), _default_base_url(provider))
        ).rstrip("/"),
        model=str(_pick("TRAVEL_AGENT_LLM_MODEL", llm_toml.get("model"), _default_model(provider))),
        timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_LLM_TIMEOUT_SECONDS", llm_toml.get("timeout_seconds"), 30), 30
        ),
        temperature=_as_float(
            _pick("TRAVEL_AGENT_LLM_TEMPERATURE", llm_toml.get("temperature"), 0.2), 0.2
        ),
        input_cost_per_million=_as_float(
            _pick(
                "TRAVEL_AGENT_LLM_INPUT_COST_PER_MILLION",
                llm_toml.get("input_cost_per_million"),
                0.0,
            ),
            0.0,
        ),
        output_cost_per_million=_as_float(
            _pick(
                "TRAVEL_AGENT_LLM_OUTPUT_COST_PER_MILLION",
                llm_toml.get("output_cost_per_million"),
                0.0,
            ),
            0.0,
        ),
    )

    amap = AmapSettings(
        web_key=_empty_to_none(_pick("TRAVEL_AGENT_AMAP_WEB_KEY", amap_toml.get("web_key"), None)),
        base_url=str(
            _pick("TRAVEL_AGENT_AMAP_BASE_URL", amap_toml.get("base_url"), "https://restapi.amap.com")
        ).rstrip("/"),
        js_key=_empty_to_none(_pick("TRAVEL_AGENT_AMAP_JS_KEY", amap_toml.get("js_key"), None)),
        js_security_key=_empty_to_none(
            _pick("TRAVEL_AGENT_AMAP_JS_SECURITY_KEY", amap_toml.get("js_security_key"), None)
        ),
        timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_AMAP_TIMEOUT_SECONDS", amap_toml.get("timeout_seconds"), 5), 5
        ),
    )

    agent = AgentSettings(
        recursion_limit=_as_int(
            _pick("TRAVEL_AGENT_RECURSION_LIMIT", agent_toml.get("recursion_limit"), 20), 20
        ),
        max_tool_retries=_as_int(
            _pick("TRAVEL_AGENT_MAX_TOOL_RETRIES", agent_toml.get("max_tool_retries"), 1), 1
        ),
        request_timeout_seconds=_as_int(
            _pick(
                "TRAVEL_AGENT_REQUEST_TIMEOUT_SECONDS",
                agent_toml.get("request_timeout_seconds"),
                120,
            ),
            120,
        ),
    )

    memory = MemorySettings(
        compress_message_threshold=_as_int(
            _pick(
                "TRAVEL_AGENT_MEMORY_COMPRESS_THRESHOLD",
                memory_toml.get("compress_message_threshold"),
                8,
            ),
            8,
        ),
        profile_only_token_threshold=_as_int(
            _pick(
                "TRAVEL_AGENT_MEMORY_PROFILE_ONLY_TOKEN_THRESHOLD",
                memory_toml.get("profile_only_token_threshold"),
                4000,
            ),
            4000,
        ),
        keep_recent_turns=_as_int(
            _pick("TRAVEL_AGENT_MEMORY_KEEP_RECENT_TURNS", memory_toml.get("keep_recent_turns"), 4),
            4,
        ),
        profile_dir=Path(
            str(_pick("TRAVEL_AGENT_PROFILE_DIR", memory_toml.get("profile_dir"), str(PROJECT_ROOT / "data" / "profiles")))
        ),
    )

    orchestration = OrchestrationSettings(
        variant_token_budget=_as_int(
            _pick(
                "TRAVEL_AGENT_VARIANT_TOKEN_BUDGET",
                orch_toml.get("variant_token_budget"),
                0,
            ),
            0,
        ),
        variant_llm_call_budget=_as_int(
            _pick(
                "TRAVEL_AGENT_VARIANT_LLM_CALL_BUDGET",
                orch_toml.get("variant_llm_call_budget"),
                0,
            ),
            0,
        ),
        variant_tool_call_budget=_as_int(
            _pick(
                "TRAVEL_AGENT_VARIANT_TOOL_CALL_BUDGET",
                orch_toml.get("variant_tool_call_budget"),
                0,
            ),
            0,
        ),
    )

    mcp = McpSettings(
        use_mcp_tools=str(
            _pick("TRAVEL_AGENT_MCP_USE_TOOLS", mcp_toml.get("use_mcp_tools"), "false")
        ).lower()
        in {"1", "true", "yes", "on"},
        auto_start=str(
            _pick("TRAVEL_AGENT_MCP_AUTO_START", mcp_toml.get("auto_start"), "false")
        ).lower()
        in {"1", "true", "yes", "on"},
        host=str(_pick("TRAVEL_AGENT_MCP_HOST", mcp_toml.get("host"), "127.0.0.1")),
        port=_as_int(_pick("TRAVEL_AGENT_MCP_PORT", mcp_toml.get("port"), 8765), 8765),
        server_name=str(_pick("TRAVEL_AGENT_MCP_SERVER_NAME", mcp_toml.get("server_name"), "travel")),
        timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_MCP_TIMEOUT", mcp_toml.get("timeout_seconds"), 30), 30
        ),
    )

    skills = SkillsSettings(
        skills_dir=Path(
            str(
                _pick(
                    "TRAVEL_AGENT_SKILLS_DIR",
                    skills_toml.get("skills_dir"),
                    str(PROJECT_ROOT / ".storyline" / "skills"),
                )
            )
        ),
        enabled=str(_pick("TRAVEL_AGENT_SKILLS_ENABLED", skills_toml.get("enabled"), "true")).lower()
        not in {"0", "false", "no", "off"},
    )

    return Settings(
        llm=llm,
        amap=amap,
        agent=agent,
        memory=memory,
        orchestration=orchestration,
        mcp=mcp,
        skills=skills,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
