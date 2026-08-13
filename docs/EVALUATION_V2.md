# 旅游 Agent 评测体系 V2

## 三层评测

评测结论由三套互相隔离的 suite 组成，避免把模型、规划代码和外部数据问题混在一起：

| suite | 规模 | 负责回答的问题 |
| --- | ---: | --- |
| Product-120 | 120 | Agent 是否满足线上用户需求，工具调用、澄清和长期记忆是否正确 |
| ChinaTravel Human-154 | 154 | 固定官方沙箱中的规划、约束满足和可复现能力 |
| Live Tools Shadow | 100 | 高德/沙箱工具的成功率、数据质量、降级行为和延迟 |

ChinaTravel 不代表实时线上 POI；Live Tools 不评价模型是否选对工具。只有 Product-120
会参与微调 go/no-go 归因，工具和适配器故障必须先排除。

## Product-120

数据位于 `data/eval/product/product_120.json`，由
`scripts/build_product_eval_cases.py` 确定性生成。生成后会验证总量、类别配额、
`30 dev / 90 frozen` 划分、ID 唯一性以及 30 条预先锁定的高风险冻结用例。

- 80 条正常规划：基础探索、明确兴趣、同行人与强度、预算餐饮住宿、必去 POI、
  日期天气路线、高复杂度可满足约束。
- 40 条诊断：澄清、跨会话记忆、冲突不可满足、工具异常各 10 条。
- 记忆用例覆盖正向召回、否定、重新激活、历史目的地/天数不继承和跨用户隔离。
- 当前 Agent 是单目的地模型，V2 不用多城市题目评判模型能力。

每条 case 可标注 `required_tools`、`allowed_tools`、`forbidden_tools`、
`tool_argument_assertions`、硬/软约束、记忆断言、`failure_injection`、数据快照日期、
逐轮 `session_ids` 和 `user_ids`。故障注入支持 POI、天气、路线的 empty、partial、
timeout、error 和 unavailable。

```bash
# 快速离线 smoke，不形成微调结论
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli \
  --suite agent-product --env offline --product-split dev

# qwen3.7-plus 开发集
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli \
  --suite agent-product --env real_agent --product-split dev --write-report

# 冻结集一次全量 + 30 条高风险用例额外两次，共 90 个独立样本、150 次执行
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli \
  --suite agent-product --env real_agent --product-split frozen \
  --repeat-high-risk --remediation-complete --write-report
```

真实 Product suite 每次固定并记录一个模型/provider，禁止中途混用模型。报告保存数据、Prompt、代码版本指纹，
每轮模型 token、工具名/参数/状态/数据来源/耗时、原始规划、critic 和最终规划。
评测 JSON 不输出完整用户消息，敏感工具参数会替换为 `[REDACTED]`。

使用 `--write-report` 时，每次执行写入独立目录
`data/eval/product/runs/<run_id>/`。`summary.json` 只保存指标与逐 case 文件索引；
`cases/<case_id>__repeat-<n>.json` 保存该次执行的需求、每轮回复、工具和模型轨迹、
最终 profile、完整 itinerary/critic artifact、规则评分，并预留独立 Judge 与人工复核字段。
历史 run 不会被后续执行覆盖。

主要指标包括任务完成、硬约束、软偏好、critic、澄清、工具 precision/recall/F1、
参数准确率、故障恢复、记忆准确率、p50/p95 延迟、token 和稳定性。比例指标同时输出
Wilson 95% 置信区间。重复执行只用于稳定性，不扩大独立样本数。

### 行程质量后处理

行程质量不由 Agent 内部 critic 单独决定。对保存后的 Product run 依次运行：

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py rules --run-dir <run>
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py judge --run-dir <run> --resume
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py sample --run-dir <run> --size 30
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py calibrate --run-dir <run> --reviews <csv>
```

规则层检查 POI 来源、日程结构、时间冲突、路线覆盖与衔接、必去/避开、预算配置、
住宿餐饮、天气适配和营业时间证据。缺失证据记为 `unknown` 并单独统计，不默认通过。
独立 Judge 默认使用 `GLM-4.7-Flash`，按 100 分固定 Rubric 输出结构化结果，缓存键绑定
行程、模型、Prompt 和 Rubric 版本。Judge 在人工校准前为 `experimental`。

人工模板包含 20 条预锁定代表样本、10 条边界/冲突诊断样本，以及代表样本中的
10 条盲重评。只有一致率、Kappa、MAE、严重问题召回和盲重评稳定性全部达到门槛，
Judge 才升级为正式指标并生成 `quality_adjusted_task_success_rate`；阈值扫描只报告，
不会自动修改默认 70 分阈值。

## 失败归因与微调

失败固定归入：`model_intent`、`model_tool_selection`、`model_tool_arguments`、
`model_clarification`、`planner_or_workflow`、`critic_or_validator`、
`memory_or_storage`、`tool_or_data`、`adapter_or_environment`、`external_api`。

离线规则路径固定输出 `offline_smoke_only`。只跑开发集或不完整冻结集时，结论必须是
`insufficient_frozen_evidence`。完整 qwen3.7-plus 冻结集上，
同类模型错误至少 10 个独立样本或错误率达到 20%，且开发集 Prompt、few-shot、工具
schema 和工作流修正已经完成，才输出 `recommend_sft`。冻结 90 条永远不能进入训练集。
其他情况输出 `no_fine_tuning` 或 `prompt_or_workflow_first`。

## ChinaTravel 与 Live Tools

ChinaTravel runner 默认根目录为 `/Users/carrier/run/projects/ChinaTravel`，也可通过
`--chinatravel-root` 或 `TRAVEL_AGENT_CHINATRAVEL_ROOT` 指定。运行前必须找到官方
schema、数据库和实体 CSV；正式评分禁止静默退回 synthetic provider，报告记录实际
root、provider 和数据库版本。先跑 Mini-Dev，再跑 Human-154：

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_chinatravel.py \
  --suite mini-dev --chinatravel-root /path/to/ChinaTravel --write-report
PYTHONPATH=src .venv/bin/python scripts/eval_chinatravel.py \
  --suite human154 --chinatravel-root /path/to/ChinaTravel \
  --real-multi-agent --resume --write-report
```

Live 模式调用真实 API 并写入递归脱敏快照；Replay 使用快照复现。报告包含快照 SHA-256
版本、API/空结果/timeout/fallback、城市污染、重复 POI、分类匹配、路线可用性和延迟。
