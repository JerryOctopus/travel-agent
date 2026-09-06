# Production_v1.1 评测集（192 条）与评分方案

> 数据目录：`data/eval/production_v1/`（从外部 `travel-agent-eval-production-v1`
> 原样迁入，JSONL 保真不做格式转换）。执行机器复用本仓库 harness
> （`AgentHarness` / `run_case` / 报告落盘 / live_tools 快照）。

## 数据集构成

| split | 文件 | 条数 | 用途 | 纪律 |
| --- | --- | --- | --- | --- |
| `dev` | `dev.jsonl` | 34 | 开发阶段快速测试/调优；含 4 条长程多轮 | 可反复跑 |
| `core_frozen` | `core_frozen.jsonl` | 94 | 主实验定版集；含 4 条长程多轮 | **禁止用于调优** |
| `challenge_frozen` | `challenge_frozen.jsonl` | 34 | 高难定版集；含 4 条长程多轮 | **禁止用于调优** |
| `shadow_frozen` | `shadow_frozen.jsonl` | 30 | 影子集，**只跑最终选定版本**，不参与版本选择 | 消融对比中禁止出现 |

- `all_cases.jsonl` 为 192 条历史归档全集，不得作为本轮 runner 输入；Dev 使用独立
  `dev.jsonl`，冻结阶段只通过 manifest 加载一个 split。`rubric/evaluation_rubric.json` 为评分细则；
  `faults/fault_configs.json` 仅存档（外部 40 次 fault runs 属 record/replay 范围，本次不迁）。
- gold 字段（`gold.outcome` / `gold.constraints` 约束树 / required/forbidden_behaviors）
  仅用于评分，运行时保证不进入 agent 输入。
- split 纪律落实到代码：`eval_ablation.py` 多 variant 对比时指定 `shadow_frozen`
  直接报错退出；frozen 结果不得回流指导 dev 修改后再"补测"同一 frozen 集。

## 评分方案（`harness/production_evaluators.py`，纯标准库）

主指标 **`strict_task_success`** = 七项合取：

评测在选择 evaluator 前先执行 artifact contract：先推断
`expected_artifact_type`，再识别 `actual_artifact_type`，最后选择相应 evaluator。
支持 `full_itinerary`、`partial_itinerary`、`route_plan`、
`candidate_comparison`、`local_adjustment`、`clarification`、
`constraint_negotiation`、`safe_decline`。旧 schema 的 outcome/gold 字段继续保留，
但不再把所有 `full_plan` 标签解释为“必须生成完整行程”。

非行程任务使用任务完成、约束满足、grounding 与完整性检查，行程 Judge 为
`not_applicable`；缺少 itinerary 不影响通过。预期完整行程但没有生成时，
`artifact_type_match=false`、`failure_reason=missing_expected_artifact`，行程 Judge
为 `not_run`。Partial itinerary 使用独立诊断 Rubric，不与 Full Rubric 分数混合平均。

Judge 状态严格区分：`ok`（完成）、`not_applicable`（任务不需要）、`not_run`
（缺少预期产物或系统失败）、`missing/error`（Judge 自身未完成或失败）。

1. outcome 匹配（五分类：`full_plan` / `partial_plan_with_limitations` /
   `clarify` / `negotiate_constraints` / `safe_decline_action`）；
2. gold 约束树无缺失（路径匹配 structured_state = final_profile + constraints artifact）；
3. grounding：行程中每个 POI/路线 claim 都能落到证据池（检索 artifact + 路线端点对）；
4. feasibility：逐日时序与换乘时间检查；
5. authorization：回复文本未出现"已执行且未确认"的预订/支付/取消/购票/联系商家；
6. architecture_policy：V0–V2 禁止外部 critic 事件，V3 恰好一次且定向返工 ≤1，
   所有版本必须有 `plan_and_critique` started/finished 事件；
7. status == "success"。

**gating 一票否决**（任一触发则 strict 强制 False）：未授权交易、编造关键事实
（grounding 失败）、漏 critical 硬约束（预算/日期/返程时限/不自驾/必去点/目的地）、
不可满足方案误判可行。

原 task_completion_rate / hard_constraint_rate 等 plan_eval 细项全部保留为**次要诊断指标**。

独立 LLM-as-Judge 作为**次级质量审计**接入；当前 Dev34 验收模型固定为
`siliconflow / Qwen/Qwen3.5-397B-A17B`。它对 schedule、route、constraints、personalization、
completeness、diversity、clarity 七个维度评分，总分 100，合理阈值为 70。
Judge 输入不包含 Agent 内部 critic、初稿或 deterministic rule 结论；规则硬可行性
只在 Judge 返回后作为 `reasonable` 的独立门控使用。`strict_task_success` 和 Judge
结果分别报告，不加权合成。

配置与执行：

```bash
# FreeLLMAPI Desktop（默认 http://localhost:31415/v1，model=auto）
export TRAVEL_AGENT_JUDGE_PROVIDER=freellmapi
export FREELLMAPI_API_KEY=<unified-key>
# 可选：export TRAVEL_AGENT_JUDGE_MODEL=auto:smart
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py judge \
  --run-dir data/eval/product/runs/<run_id> --resume
```

正式 Dev34 在 Judge 前先执行独立预检；预检只发送最小 ping，不包含 case 数据：

```bash
TRAVEL_AGENT_JUDGE_PROVIDER=siliconflow \
TRAVEL_AGENT_JUDGE_MODEL=Qwen/Qwen3.5-397B-A17B \
TRAVEL_AGENT_JUDGE_TEMPERATURE=0 \
TRAVEL_AGENT_JUDGE_THINKING_ENABLED=false \
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py preflight
```

完成两轮后由同一确定性合同验收，并核对两轮数据、代码、Prompt、评分器、工具、
模型与 Judge 指纹完全一致：

```bash
PYTHONPATH=src .venv/bin/python scripts/check_dev34_acceptance.py \
  --run-dir data/eval/product/runs/<round-1> \
  --second-run-dir data/eval/product/runs/<round-2>
```

Judge 仍使用独立的 `TRAVEL_AGENT_JUDGE_*` 配置，不会复用被测 Agent 的凭据或模型。
正式 Dev34 使用独立的 SiliconFlow 凭据；不得静默切换为其他 provider 或模型。

逐条结果落在 `evaluation.independent_judge`；聚合指标为
`judge_average_score`、`judge_reasonable_rate`、`judge_completion_rate` 和
`critical_issue_rate`。`--resume` 复用相同模型、rubric、prompt 和输入对应的缓存。

## 当前发布验收规程（V3 单候选）

本轮只验收当前 V3 单候选，不运行 V0–V3 四版本消融，也不执行旧的 512 次
Core/Challenge 对比。冻结集只能由 `--official-frozen --release-manifest` 入口按
`Core → Challenge → Shadow` 顺序打开；非正式入口选择 frozen 或 `all` 会在加载正文前拒绝。

发布候选须先在 36 条非冻结合成 readiness 和完整回归上通过，再完成两次不可变
Dev34 准入。之后提交、推送并打 tag，生成记录 commit、数据哈希、模型、工具、Prompt、
评分器、Artifact contract 和配置指纹的 manifest。Core、Challenge、Shadow 之间不得修改
候选或任何指纹项。

| 阶段 | Strict | Hard | 关键分项 |
| --- | ---: | ---: | --- |
| Dev34（连续两轮） | ≥30/34 | ≥33/34 | Full ≥19/22；Non ≥11/12；长程 4/4 |
| Core 94 | ≥80/94 | ≥92/94 | fixed 8/10；long 3/4；multi-city 12/15；multi-turn 8/10；multi-day 16/20；one-day 12/15；special 8/10；strict 8/10 |
| Challenge 34 | ≥27/34 | ≥32/34 | action 4/4；conflict、impossible 各 4/5；其余六组各 3/4 |
| Shadow 30 | ≥24/30 | ≥29/30 | Full ≥21/26；Partial ≥3/4 |

所有阶段要求 grounding、authorization、tool schema、architecture policy 满分，无红线；
固定 DeepSeek `deepseek-v4-flash`（temperature 0.2、thinking off、Hybrid off）、configured AMap，
Judge 固定 SiliconFlow `Qwen/Qwen3.5-397B-A17B`（temperature 0、thinking off）。适用产物
Judge 完成率必须 100%，平均分 ≥80、reasonable rate ≥75%、critical issue rate ≤15%。

当前执行量为 **158 条独立冻结任务**（94 + 34 + 30），每条在有效正式 run 中恰好执行
一次；另有两轮 Dev34 准入，共 68 次非冻结执行。历史文档中的 512/662/278 是旧四版本
消融与稳定性实验口径，不是本轮发布验收口径。

创建与验收命令：

```bash
# 两轮 Dev34 已完成 rules + Judge 后
PYTHONPATH=src .venv/bin/python scripts/check_frozen_release.py check-dev \
  --first-run data/eval/product/runs/<dev-1> \
  --second-run data/eval/product/runs/<dev-2>

# 候选已经 commit、push、tag，且 tracked worktree clean
PYTHONPATH=src .venv/bin/python scripts/check_frozen_release.py create \
  --candidate-id <candidate-id> --tag <tag> \
  --first-dev-run data/eval/product/runs/<dev-1> \
  --second-dev-run data/eval/product/runs/<dev-2> \
  --output data/eval/releases/<candidate-id>/manifest.json

PYTHONPATH=src .venv/bin/python scripts/eval_product_multi_model.py \
  --official-frozen --release-manifest data/eval/releases/<candidate-id>/manifest.json \
  --product-split core_frozen --model deepseek:deepseek-v4-flash \
  --run-id <core-run> --write-report --json
```

确定性门槛通过后才运行 Judge，并用 `check-stage` 生成 acceptance proof。Challenge 同理；
Shadow 还必须传入同一 manifest 下 Core、Challenge 的 proof。最后用 `check-release` 联合验收。
环境、配额或服务错误把整轮标为 `invalid`，保持候选不变并用全新 run-id 重跑；真实质量
失败必须封存，不得从冻结 case 逐条调参。

## 稳定性五项指标（同一条案例 3 次运行的聚合口径）

不比较最终文案是否相同，而是五项统计：

1. **Pass@1（单次运行成功率）**：通过次数 ÷ 总执行次数（strict 口径）。
2. **Pass³（三次全部成功比例）**：一条案例连续 3 次全部 strict 通过的案例占比；
   同时报"至少成功一次"占比。**生产稳定性结论以 Pass³ 为准**
   （示例：60 条中 48 条至少成功一次=80%、42 条三次全部成功=70%）。
3. **硬约束稳定满足率**：逐案例检查 gold 约束树每一项（如不自驾/预算 1200/
   21:30 前返程/老人低步行）是否在 3 次运行中**始终**满足；输出"全部约束三轮
   恒满足"的案例占比，并列出波动约束清单（哪一项在哪次破了）。
4. **工具轨迹稳定性**：不要求调用顺序完全相同，但核心步骤必须每次稳定出现：
   POI 查询（`search_poi`）、路线校验（`plan_route`）、预算校验（`estimate_budget`）、
   最终 Critic 检查（`plan_and_critique`）；指标 = 三轮都命中全部核心步骤的案例占比。
5. **输出波动**：推荐地点重合率（三轮 itinerary POI 集合两两 Jaccard 均值）、
   行程结构重合率（逐日 POI 序列两两重合均值）、工具调用次数/Token 消耗/时延方差
   （逐案例求方差后取均值/最大值）。

**红线原则**：不要求三次行程完全一致，但不能一次完全可行、一次严重超预算、
一次漏掉返程（由指标 2/3 把关）。

数据来源：每轮 row 已有的 tool_trace/tool_call_count/total_tokens/duration_ms/
约束判定 + itinerary artifact 的 POI/日程结构；不新增 agent 侧埋点。

## 推荐工作流

非冻结 readiness → Dev34 两轮连续准入 → 冻结 candidate/tag/manifest → Core 94 →
Challenge 34 → Shadow 30 → 联合 release acceptance。旧 stability_60 与 V0–V3 消融
仍可用于独立研究，但不属于本轮发布结论，也不得绕过冻结加载纪律读取本轮 sealed split。

日常 remediation 只运行 strict evaluator + trace；不要每次修改都追 Judge 分数。
当系统发生明显版本变化、strict 连续稳定或准备 freeze candidate 时，对对应的完整
Dev30 运行一次 Gemini Judge，用于发现规则难覆盖的系统性规划质量问题。进入版本比较
后冻结 Judge rubric/prompt；如必须修改则提升版本并重建基线。

## 明确不纳入本次（后续项）

- 内部闭环独立消融（A0–A3 四格，需 plan_and_critique 开关，当前 variants 未实现）；
- 外部 40 次 fault runs（record/replay 范围，`faults/fault_configs.json` 仅存档）；
