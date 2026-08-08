"""多 Agent 编排：统一 Engine（生产 = Full = V3）与 V0–V3 架构消融实验入口。

- ``multi_agent``：V0–V3 共用的 ``MultiAgentEngine`` 与 Subagent 框架；
- ``variants``：消融实验薄适配器，只能通过 ``run_variant_turn`` 显式选择版本。

生产入口（``agent.runtime.run_production_turn``）固定使用
``multi_agent.PRODUCTION_CONFIG``，不读取任何 variant 配置。
"""

from travel_agent.orchestration.variants import VARIANTS, VariantSpec, run_variant_turn

__all__ = [
    "VARIANTS",
    "VariantSpec",
    "run_variant_turn",
]
