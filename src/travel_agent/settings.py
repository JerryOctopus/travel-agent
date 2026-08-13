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
    thinking_enabled: bool = False
    requests_per_second: float = 0.0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    # 模型上下文窗口（token）；None 时由 memory 模块按模型名离线估算。
    context_window: int | None = None

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
    request_timeout_seconds: int = 300


@dataclass(frozen=True)
class MemorySettings:
    compress_message_threshold: int = 20
    profile_only_token_threshold: int = 64000
    keep_recent_turns: int = 10
    backend: str = "json"
    database_url: str | None = None
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
    base_timeout_seconds: int = 210
    planner_reserve_seconds: int = 25
    reviewer_reserve_seconds: int = 85
    router_timeout_seconds: int = 80
    router_useful_seconds: int = 35
    worker_timeout_seconds: int = 60
    planner_timeout_seconds: int = 60
    worker_max_output_tokens: int = 2048
    planner_max_output_tokens: int = 1024
    reviewer_max_output_tokens: int = 1024
    recovery_worker_useful_seconds: int = 25
    reviewer_timeout_seconds: int = 85
    repair_worker_timeout_seconds: int = 30
    repair_planner_timeout_seconds: int = 45
    admission_guard_seconds: int = 5
    latency_sla_seconds: int = 180
    routing_max_waves: int = 3
    routing_max_calls: int = 3
    routing_max_dispatches: int = 6
    routing_wave1_max_tasks: int = 4


@dataclass(frozen=True)
class SkillsSettings:
    skills_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".storyline" / "skills")
    enabled: bool = True


@dataclass(frozen=True)
class JudgeSettings:
    provider: str = "google"
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-3.6-flash"
    timeout_seconds: int = 60
    temperature: float = 0.0
    thinking_enabled: bool = False
    # AI Studio currently reports 20 RPM for gemini-3.6-flash on this project.
    # Keep 10% headroom so minute-window jitter does not cause avoidable 429s.
    requests_per_second: float = 0.3

    @property
    def enabled(self) -> bool:
        return self.provider != "rule" and bool(self.api_key)


@dataclass(frozen=True)
class EvaluationSettings:
    judge: JudgeSettings = field(default_factory=JudgeSettings)


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    amap: AmapSettings = field(default_factory=AmapSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    orchestration: OrchestrationSettings = field(default_factory=OrchestrationSettings)
    mcp: McpSettings = field(default_factory=McpSettings)
    skills: SkillsSettings = field(default_factory=SkillsSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


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


def _provider_api_key(provider: str) -> str | None:
    """读取 provider 专属环境变量，避免跨服务复用凭据。"""
    names: tuple[str, ...]
    if provider.lower() in {"freellmapi", "free_llm_api", "free-llm-api"}:
        names = ("TRAVEL_AGENT_FREELLMAPI_API_KEY", "FREELLMAPI_API_KEY")
    elif provider.lower() in {
        "freellmapi-docker",
        "freellmapi_server",
        "freellmapi-server",
    }:
        names = ("TRAVEL_AGENT_FREELLMAPI_API_KEY", "FREELLMAPI_API_KEY")
    elif provider.lower() in {"google", "gemini", "google-gemini"}:
        names = ("TRAVEL_AGENT_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    elif provider.lower() in {"siliconflow", "silicon-flow", "silicon_flow"}:
        names = ("TRAVEL_AGENT_SILICONFLOW_API_KEY", "SILICONFLOW_API_KEY")
    else:
        names = (f"TRAVEL_AGENT_{provider.upper().replace('-', '_')}_API_KEY",)
    for name in names:
        value = _empty_to_none(os.getenv(name))
        if value:
            return value
    return None


def _llm_api_key(provider: str, toml_value: Any, *, provider_overridden: bool) -> str | None:
    generic = _empty_to_none(os.getenv("TRAVEL_AGENT_LLM_API_KEY"))
    if generic:
        return generic
    provider_key = _provider_api_key(provider)
    if provider_key:
        return provider_key
    return None if provider_overridden else _empty_to_none(toml_value)


def _judge_api_key(provider: str, toml_value: Any, *, provider_overridden: bool) -> str | None:
    """Resolve a Judge credential without falling back to the Agent credential.

    ``TRAVEL_AGENT_JUDGE_API_KEY`` remains the highest-priority override. Provider-specific
    variables (including ``FREELLMAPI_API_KEY``) make it possible to switch the Judge without
    copying a unified router key into ``config.toml``.
    """
    generic = _empty_to_none(os.getenv("TRAVEL_AGENT_JUDGE_API_KEY"))
    if generic:
        return generic
    provider_key = _provider_api_key(provider)
    if provider_key:
        return provider_key
    return None if provider_overridden else _empty_to_none(toml_value)


def _default_base_url(provider: str) -> str:
    provider = provider.lower()
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
    provider = provider.lower()
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
        return "deepseek-chat"
    if provider in {"zhipu", "glm", "zai"}:
        return "glm-4.7-flash"
    if provider in {"google", "gemini", "google-gemini"}:
        return "gemini-2.5-flash"
    if provider in {"siliconflow", "silicon-flow", "silicon_flow"}:
        return "deepseek-ai/DeepSeek-V4-Flash"
    return "gpt-4o-mini"


def _default_judge_model(provider: str) -> str:
    if provider.lower() in {"google", "gemini", "google-gemini"}:
        return "gemini-3.6-flash"
    return _default_model(provider)


def _default_judge_requests_per_second(provider: str) -> float:
    # The public Gemini quota benefits from a conservative client-side limiter.
    # A local FreeLLMAPI router owns upstream quotas/fallbacks itself.
    if provider.lower() in {"google", "gemini", "google-gemini"}:
        return 0.3
    return 0.0


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
    evaluation_toml = (
        data.get("evaluation", {}) if isinstance(data.get("evaluation"), dict) else {}
    )
    judge_toml = (
        evaluation_toml.get("judge", {})
        if isinstance(evaluation_toml.get("judge"), dict)
        else {}
    )

    provider_env = _empty_to_none(os.getenv("TRAVEL_AGENT_LLM_PROVIDER"))
    provider = str(provider_env or llm_toml.get("provider") or "rule").lower()
    provider_overridden = bool(
        provider_env
        and provider_env.lower() != str(llm_toml.get("provider") or "rule").lower()
    )
    configured_base_url = None if provider_overridden else llm_toml.get("base_url")
    configured_model = None if provider_overridden else llm_toml.get("model")
    llm = LLMSettings(
        provider=provider,
        api_key=_llm_api_key(
            provider,
            llm_toml.get("api_key"),
            provider_overridden=provider_overridden,
        ),
        base_url=str(
            _pick("TRAVEL_AGENT_LLM_BASE_URL", configured_base_url, _default_base_url(provider))
        ).rstrip("/"),
        model=str(_pick("TRAVEL_AGENT_LLM_MODEL", configured_model, _default_model(provider))),
        timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_LLM_TIMEOUT_SECONDS", llm_toml.get("timeout_seconds"), 30), 30
        ),
        temperature=_as_float(
            _pick("TRAVEL_AGENT_LLM_TEMPERATURE", llm_toml.get("temperature"), 0.2), 0.2
        ),
        thinking_enabled=str(
            _pick(
                "TRAVEL_AGENT_LLM_THINKING_ENABLED",
                llm_toml.get("thinking_enabled"),
                "false",
            )
        ).lower()
        in {"1", "true", "yes", "on"},
        requests_per_second=_as_float(
            _pick(
                "TRAVEL_AGENT_LLM_REQUESTS_PER_SECOND",
                llm_toml.get("requests_per_second"),
                0.0,
            ),
            0.0,
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
        context_window=_as_int(
            _pick("TRAVEL_AGENT_LLM_CONTEXT_WINDOW", llm_toml.get("context_window"), 0), 0
        )
        or None,
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
                300,
            ),
            300,
        ),
    )

    memory = MemorySettings(
        compress_message_threshold=_as_int(
            _pick(
                "TRAVEL_AGENT_MEMORY_COMPRESS_THRESHOLD",
                memory_toml.get("compress_message_threshold"),
                20,
            ),
            20,
        ),
        profile_only_token_threshold=_as_int(
            _pick(
                "TRAVEL_AGENT_MEMORY_PROFILE_ONLY_TOKEN_THRESHOLD",
                memory_toml.get("profile_only_token_threshold"),
                64000,
            ),
            64000,
        ),
        keep_recent_turns=_as_int(
            _pick("TRAVEL_AGENT_MEMORY_KEEP_RECENT_TURNS", memory_toml.get("keep_recent_turns"), 10),
            10,
        ),
        backend=str(
            _pick("TRAVEL_AGENT_MEMORY_BACKEND", memory_toml.get("backend"), "json")
        ).lower(),
        database_url=_empty_to_none(
            _pick(
                "TRAVEL_AGENT_DATABASE_URL",
                memory_toml.get("database_url"),
                None,
            )
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
        base_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_BASE_TIMEOUT_SECONDS", orch_toml.get("base_timeout_seconds"), 210),
            210,
        ),
        planner_reserve_seconds=_as_int(
            _pick("TRAVEL_AGENT_PLANNER_RESERVE_SECONDS", orch_toml.get("planner_reserve_seconds"), 25),
            25,
        ),
        reviewer_reserve_seconds=_as_int(
            _pick("TRAVEL_AGENT_REVIEWER_RESERVE_SECONDS", orch_toml.get("reviewer_reserve_seconds"), 85),
            85,
        ),
        router_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_ROUTER_TIMEOUT_SECONDS", orch_toml.get("router_timeout_seconds"), 80),
            80,
        ),
        router_useful_seconds=_as_int(
            _pick("TRAVEL_AGENT_ROUTER_USEFUL_SECONDS", orch_toml.get("router_useful_seconds"), 35),
            35,
        ),
        worker_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_WORKER_TIMEOUT_SECONDS", orch_toml.get("worker_timeout_seconds"), 60),
            60,
        ),
        planner_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_PLANNER_TIMEOUT_SECONDS", orch_toml.get("planner_timeout_seconds"), 60),
            25,
        ),
        worker_max_output_tokens=_as_int(
            _pick("TRAVEL_AGENT_WORKER_MAX_OUTPUT_TOKENS", orch_toml.get("worker_max_output_tokens"), 2048),
            2048,
        ),
        planner_max_output_tokens=_as_int(
            _pick("TRAVEL_AGENT_PLANNER_MAX_OUTPUT_TOKENS", orch_toml.get("planner_max_output_tokens"), 1024),
            1024,
        ),
        reviewer_max_output_tokens=_as_int(
            _pick("TRAVEL_AGENT_REVIEWER_MAX_OUTPUT_TOKENS", orch_toml.get("reviewer_max_output_tokens"), 1024),
            1024,
        ),
        recovery_worker_useful_seconds=_as_int(
            _pick(
                "TRAVEL_AGENT_RECOVERY_WORKER_USEFUL_SECONDS",
                orch_toml.get("recovery_worker_useful_seconds"),
                25,
            ),
            25,
        ),
        reviewer_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_REVIEWER_TIMEOUT_SECONDS", orch_toml.get("reviewer_timeout_seconds"), 85),
            85,
        ),
        repair_worker_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_REPAIR_WORKER_TIMEOUT_SECONDS", orch_toml.get("repair_worker_timeout_seconds"), 30),
            30,
        ),
        repair_planner_timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_REPAIR_PLANNER_TIMEOUT_SECONDS", orch_toml.get("repair_planner_timeout_seconds"), 45),
            20,
        ),
        admission_guard_seconds=_as_int(
            _pick("TRAVEL_AGENT_ADMISSION_GUARD_SECONDS", orch_toml.get("admission_guard_seconds"), 5),
            5,
        ),
        latency_sla_seconds=_as_int(
            _pick("TRAVEL_AGENT_LATENCY_SLA_SECONDS", orch_toml.get("latency_sla_seconds"), 180),
            180,
        ),
        routing_max_waves=_as_int(
            _pick("TRAVEL_AGENT_ROUTING_MAX_WAVES", orch_toml.get("routing_max_waves"), 3),
            3,
        ),
        routing_max_calls=_as_int(
            _pick("TRAVEL_AGENT_ROUTING_MAX_CALLS", orch_toml.get("routing_max_calls"), 3),
            3,
        ),
        routing_max_dispatches=_as_int(
            _pick("TRAVEL_AGENT_ROUTING_MAX_DISPATCHES", orch_toml.get("routing_max_dispatches"), 6),
            6,
        ),
        routing_wave1_max_tasks=_as_int(
            _pick("TRAVEL_AGENT_ROUTING_WAVE1_MAX_TASKS", orch_toml.get("routing_wave1_max_tasks"), 4),
            4,
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

    judge_provider_env = _empty_to_none(os.getenv("TRAVEL_AGENT_JUDGE_PROVIDER"))
    judge_provider = str(judge_provider_env or judge_toml.get("provider") or "google").lower()
    judge_provider_overridden = bool(
        judge_provider_env
        and judge_provider_env.lower() != str(judge_toml.get("provider") or "google").lower()
    )
    configured_judge_base_url = (
        None if judge_provider_overridden else judge_toml.get("base_url")
    )
    configured_judge_model = None if judge_provider_overridden else judge_toml.get("model")
    judge = JudgeSettings(
        provider=judge_provider,
        api_key=_judge_api_key(
            judge_provider,
            judge_toml.get("api_key"),
            provider_overridden=judge_provider_overridden,
        ),
        base_url=str(
            _pick(
                "TRAVEL_AGENT_JUDGE_BASE_URL",
                configured_judge_base_url,
                _default_base_url(judge_provider),
            )
        ).rstrip("/"),
        model=str(
            _pick(
                "TRAVEL_AGENT_JUDGE_MODEL",
                configured_judge_model,
                _default_judge_model(judge_provider),
            )
        ),
        timeout_seconds=_as_int(
            _pick("TRAVEL_AGENT_JUDGE_TIMEOUT_SECONDS", judge_toml.get("timeout_seconds"), 60),
            60,
        ),
        temperature=_as_float(
            _pick("TRAVEL_AGENT_JUDGE_TEMPERATURE", judge_toml.get("temperature"), 0.0),
            0.0,
        ),
        thinking_enabled=str(
            _pick(
                "TRAVEL_AGENT_JUDGE_THINKING_ENABLED",
                judge_toml.get("thinking_enabled"),
                "false",
            )
        ).lower()
        in {"1", "true", "yes", "on"},
        requests_per_second=_as_float(
            _pick(
                "TRAVEL_AGENT_JUDGE_REQUESTS_PER_SECOND",
                judge_toml.get("requests_per_second"),
                _default_judge_requests_per_second(judge_provider),
            ),
            _default_judge_requests_per_second(judge_provider),
        ),
    )

    return Settings(
        llm=llm,
        amap=amap,
        agent=agent,
        memory=memory,
        orchestration=orchestration,
        mcp=mcp,
        skills=skills,
        evaluation=EvaluationSettings(judge=judge),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
