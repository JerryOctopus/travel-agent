# 真实模型 Agent Harness 评测

> 自动记录：使用 `python -m travel_agent.harness.cli --suite agent-real --env real_agent` 对候选模型做真实 LLM 评测。

## 候选模型

| 模型 | smoke 结果 | 主要结论 |
| --- | --- | --- |
| `qwen3.7-flash-2026-07-15` | 未通过 | preflight 通过；requirement 层触发 DashScope repetitive tool calls 400。 |
| `qwen-long` | 未通过 | preflight 通过；requirement 层返回 `input.messages.*.content` 缺失错误。 |
| `qwen3.5-35b-a3b` | 未通过 | preflight 通过；能进入 research/planning，但 planning 层未成功触发 `plan_and_critique`。 |
| `glm-4.5-air` | 未通过 | preflight 失败；模型要求 stream mode，当前非流式 ChatOpenAI 调用不适配。 |
| `qwen3-max-preview` | 通过 | smoke 通过；全量 `agent-real` 通过。 |
| `qwen3.5-flash-2026-02-23` | 未通过 | preflight 通过；能进入 research/planning，但 planning 层未成功触发 `plan_and_critique`。 |

## 选型结果

当前真实 Agent 评测推荐模型：

```text
qwen3-max-preview
```

全量命令：

```bash
TRAVEL_AGENT_LLM_MODEL=qwen3-max-preview PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite agent-real --env real_agent --json
```

全量结果：

```text
sample_size: 6
task_completion_rate: 1.0
tool_coverage_rate: 1.0
used_real_agent_rate: 1.0
agent_trace_rate: 1.0
research_agent_tool_rate: 1.0
planning_agent_tool_rate: 1.0
external_llm_error_rate: 0.0
final_pass_rate: 1.0
full_multi_agent_pass_rate: 1.0
```

## 备注

- `qwen3-max-preview` 的真实全量评测覆盖 `eval/cases.json` 中 6 条可完整规划的 travel case；寒暄、缺字段、只追问类 case 不纳入 real multi-agent 完整规划集。
- 评测器已把 `restaurants` artifact 计入 `food` 偏好覆盖，避免真实 Agent 已调用餐厅工具但 itinerary stop 未放餐厅时被误判为美食偏好未覆盖。
- ChinaTravel 官方 mini-dev 仍是上线质量阻塞项，见 `docs/EVALUATION_CHINATRAVEL.md` 和 `docs/EVALUATION_RELEASE.md`。
