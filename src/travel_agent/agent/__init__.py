"""旅行规划 Agent。

生产入口 ``runtime.run_production_turn`` 固定使用 Multi-Agent Full（V3）：
``orchestration.multi_agent.MultiAgentEngine(PRODUCTION_CONFIG)``，渲染统一
经 Renderer Gate，可观测性统一为 agent_trace。V0 单 Agent 基线仍复用
``runtime._run_react``（ReAct 全工具链路），内层规划走
``plan_and_critique``（见 ``travel_agent.planning_subgraph``）。
无 LLM key 时自动降级为确定性兜底，保证可离线演示。
"""

from travel_agent.agent.runtime import AgentReply, run_production_turn

__all__ = ["AgentReply", "run_production_turn"]
