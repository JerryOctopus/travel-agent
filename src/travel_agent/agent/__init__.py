"""LangGraph 旅行规划 Agent。

外层是真正自主 tool calling 的 ReAct agent（``runtime.build_react_agent``），
内层是可控规划子图 ``plan_and_critique``（见 ``travel_agent.planning_subgraph``）。
无 LLM key 时由 ``runtime.run_turn`` 自动降级为确定性兜底，保证可离线演示。
"""

from travel_agent.agent.runtime import AgentReply, run_turn

__all__ = ["AgentReply", "run_turn"]
