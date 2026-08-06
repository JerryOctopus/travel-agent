"""多 Agent 架构（生产 = Full = V3）基础框架包。

Step 1 新增，尚未接入生产路由；现有 runtime / variants 行为不受影响。

模块分工：

- ``schemas``：SubagentTask / SubagentResult / ReviewIssue / ReviewResult /
  EngineCapabilities 核心契约；
- ``registry``：5 个执行型 Subagent 定义（Reviewer 不在其中）；
- ``dispatch_rules``：V1 确定性 ``task_type → 批次`` 派工映射；
- ``runner``：统一 Subagent 执行器（异常包装为 failed，不上抛）；
- ``executor``：真实 LLM 受限 ReAct Subagent 执行器（Step 3）；
- ``fixed_dispatch``：V1 确定性派工执行（Step 3）；
- ``orchestrator_agent``：V2/V3 动态 Main Orchestrator（Step 3）；
- ``review``：Engine 直调的单层无工具 Semantic Reviewer 与修复周期判定；
- ``render_gate``：Renderer Artifact Gate，Engine 统一渲染入口（Step 3）；
- ``orchestrator``：Main Orchestrator 工具面约束与 dispatch 工具骨架；
- ``engine``：V0–V3 能力预设、配置别名与统一 Engine（Step 3 全编排）。
"""

from travel_agent.orchestration.multi_agent.engine import (
    FULL_CONFIG,
    PRODUCTION_CONFIG,
    V0_CONFIG,
    V1_CONFIG,
    V2_CONFIG,
    V3_CONFIG,
    VARIANT_PRESETS,
    MultiAgentEngine,
    TurnOutcome,
    capabilities_for_variant,
    requires_semantic_review,
)
from travel_agent.orchestration.multi_agent.executor import build_subagent_executor
from travel_agent.orchestration.multi_agent.fixed_dispatch import run_fixed_dispatch
from travel_agent.orchestration.multi_agent.orchestrator_agent import (
    build_orchestrator_prompt,
    build_orchestrator_tools,
    run_orchestrator,
)
from travel_agent.orchestration.multi_agent.render_gate import render_plan_outcome
from travel_agent.orchestration.multi_agent.review import (
    ReviewCallable,
    ReviewContext,
    build_review_callable,
    extract_json_payload,
    repair_targets,
    resolve_delivery_status,
    reviewer_prompt,
    run_semantic_review,
)
from travel_agent.orchestration.multi_agent.registry import (
    SUBAGENT_REGISTRY,
    SubagentDefinition,
    all_subagent_tool_names,
    get_subagent,
    list_subagents,
)
from travel_agent.orchestration.multi_agent.runner import SubagentRunner
from travel_agent.orchestration.multi_agent.schemas import (
    EngineCapabilities,
    ReviewIssue,
    ReviewResult,
    SubagentResult,
    SubagentTask,
    new_request_id,
    new_task_id,
)
from travel_agent.orchestration.multi_agent.store_view import (
    ArtifactReadResult,
    PlannerInputs,
    read_artifacts_for_planner,
)
from travel_agent.orchestration.multi_agent.trace import AgentTraceLog

__all__ = [
    "FULL_CONFIG",
    "PRODUCTION_CONFIG",
    "V0_CONFIG",
    "V1_CONFIG",
    "V2_CONFIG",
    "V3_CONFIG",
    "VARIANT_PRESETS",
    "MultiAgentEngine",
    "TurnOutcome",
    "capabilities_for_variant",
    "requires_semantic_review",
    "build_subagent_executor",
    "run_fixed_dispatch",
    "build_orchestrator_prompt",
    "build_orchestrator_tools",
    "run_orchestrator",
    "render_plan_outcome",
    "ReviewCallable",
    "ReviewContext",
    "build_review_callable",
    "extract_json_payload",
    "repair_targets",
    "resolve_delivery_status",
    "reviewer_prompt",
    "run_semantic_review",
    "SUBAGENT_REGISTRY",
    "SubagentDefinition",
    "all_subagent_tool_names",
    "get_subagent",
    "list_subagents",
    "SubagentRunner",
    "EngineCapabilities",
    "ReviewIssue",
    "ReviewResult",
    "SubagentResult",
    "SubagentTask",
    "new_request_id",
    "new_task_id",
    "ArtifactReadResult",
    "PlannerInputs",
    "read_artifacts_for_planner",
    "AgentTraceLog",
]
