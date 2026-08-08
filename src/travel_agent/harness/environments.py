from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from travel_agent.agent.session import DEFAULT_POI_PATH
from travel_agent.settings import AmapSettings, LLMSettings, Settings

HarnessMode = Literal["offline", "real_agent"]


@dataclass(frozen=True)
class HarnessEnvironment:
    """Runtime policy for a harness run.

    ``variant`` 为消融实验显式版本选择（"v0"–"v3"）；None 表示生产
    固定链路（Multi-Agent Full = V3），经 ``run_production_turn`` 执行。
    """

    mode: HarnessMode = "offline"
    poi_path: Path | str = DEFAULT_POI_PATH
    persist: bool = False
    user_id: str = "harness_eval"
    variant: str | None = None

    def apply(self, settings: Settings) -> Settings:
        if self.mode == "real_agent":
            return settings
        if self.mode != "offline":
            raise ValueError(f"Unsupported harness mode: {self.mode}")
        updates = {
            "llm": LLMSettings(provider="rule", api_key=None),
            "amap": AmapSettings(),
        }
        # MCP 是可选扩展；Step 4 Harness 不得要求未进入快照的 MCP settings。
        if hasattr(settings, "mcp"):
            updates["mcp"] = replace(
                settings.mcp,
                use_mcp_tools=False,
                auto_start=False,
            )
        return replace(settings, **updates)
