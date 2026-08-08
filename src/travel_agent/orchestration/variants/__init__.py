"""V0–V3 架构消融实验（Ablation Study）——多智能体是否“为了复杂而复杂”？

这是本项目的四版本对照实验模块。唯一自变量是 Agent 编排架构（由
``EngineCapabilities`` 表达）；主模型、工具集、工具数据、测试集
（production_v1 180 条）、Token 硬上限与基础业务规则在四个版本间完全共享。
完整实验设计见 ``docs/ABLATION_V0_V3.md``。

四个版本共用同一个 ``MultiAgentEngine``（见 ``orchestration.multi_agent``）：

- ``v0`` 单 Agent 基线：一个 ReAct agent 自主编排全部工具；
- ``v1`` 确定性规则派工：task_type 查表生成批次任务，无 LLM 路由；
- ``v2`` 动态 Main Orchestrator，无 Reviewer；
- ``v3`` = Production Full：动态 Orchestrator + 语义 Reviewer +
  最多一次定向修复周期。

四版本共用同一 Token 硬上限（``settings.orchestration.variant_token_budget``）。

用法：

- 评测：``python scripts/eval_ablation.py --variants v0,v1,v2,v3``；
- 编程入口：``run_variant_turn("v2", user_message, ctx, settings, history, user_id)``。

生产入口不经过本模块：``runtime.run_production_turn`` 固定使用
``PRODUCTION_CONFIG``（= FULL_CONFIG = V3_CONFIG）。
"""

from travel_agent.orchestration.variants.registry import (
    VARIANTS,
    VariantSpec,
    run_variant_turn,
)

__all__ = [
    "VARIANTS",
    "VariantSpec",
    "run_variant_turn",
]
