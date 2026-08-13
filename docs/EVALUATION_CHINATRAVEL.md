# ChinaTravel 离线评测报告

> 自动生成：`PYTHONPATH=src python scripts/eval_chinatravel.py --suite ... --write-report`。

## 评测层级

- Mini-Dev：5-10 条黄金微型集，用于 Prompt / 多 Agent 协作流调试；
- Human-154：版本级回归集，用于保存重要算法版本；
- Human-1000：最终离线验收集，用于进入 Shadow Testing 前的完整跑分。

## 本次结果

- suite：`mini-dev`
- split：`human`
- mode：`real_multi_agent`
- llm：`qwen` / `qwen3.7-plus`
- layered_enabled：True（Step 4 前的历史基线；新架构已删除该开关，固定 Multi-Agent Full = V3）
- provider：`chinatravel_official_database`
- chinatravel_root：`/Users/carrier/run/projects/ChinaTravel`
- database_version：`files-30`
- case_count：8
- attempted_case_count：8
- evaluated_case_count：8
- prediction_file_count：5
- stopped_early：False
- stop_reason：None
- predictions：`/Users/carrier/run/projects/travel-agent/data/eval/chinatravel/mini-dev/predictions_real_multi_agent`
- diagnostics：`/Users/carrier/run/projects/travel-agent/data/eval/chinatravel/mini-dev/diagnostics_real_multi_agent`
- official_eval_available：True
- DR / Delivery Rate：1.0
- Schema Pass Rate：100.0
- Schema Error Count：None
- EPR micro：100.0
- LPR micro：100.0
- C-LPR micro：100.0
- FPR：1.0
- Preference Pass Rate：None
- All Pass Count：5

## 真实 Multi-Agent 诊断（本次命令）

- row_count：8
- strict_valid_count：5
- used_real_agent_rate：1.0
- agent_trace_rate（旧口径 layer_trace_rate）：1.0
- required_agents_rate（旧口径 required_layers_rate）：0.625
- itinerary_produced_rate：0.625
- fallback_used_rate：0.375
- external_llm_error_rate：0.375

## 真实 Multi-Agent 诊断（累计 diagnostics）

- row_count：8
- strict_valid_count：5
- strict_valid_rate：0.625
- used_real_agent_rate：1.0
- agent_trace_rate（旧口径 layer_trace_rate）：1.0
- required_agents_rate（旧口径 required_layers_rate）：0.625
- itinerary_produced_rate：0.625
- fallback_used_rate：0.375
- external_llm_error_rate：0.375

## 失败原因 TopN

### Commonsense / 环境约束

| reason | fail_rate |
| --- | ---: |
| 无 | 0 |

### Logical / 需求约束

| reason | fail_rate |
| --- | ---: |
| logic_py_0 | 1.0000 |
| logic_py_1 | 1.0000 |
| logic_py_2 | 1.0000 |
| logic_py_3 | 1.0000 |

## 未交付样例 Top20

| query_id | status |
| --- | --- |
| h20241029143456235562 | llm_error |
| h20241029143457494209 | llm_error |
| h20241029143458760702 | llm_error |
