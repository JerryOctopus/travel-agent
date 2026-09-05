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

- `all_cases.jsonl` 为 192 条全集；`rubric/evaluation_rubric.json` 为评分细则；
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

## 三阶段执行规程（V0–V3 消融，对齐外部 version_matrix）

0. **开发调优**（不计入正式次数，可反复）：
   `python scripts/eval_ablation.py --variants v0,v1,v2,v3 --product-split dev`（冒烟加 `--limit 3`）。
1. **阶段一：四版本正式对比（512 次）**：Core 94 + Challenge 34 = 128 条冻结任务
   × 4 版本各跑一次：
   ```bash
   python scripts/eval_ablation.py --variants v0,v1,v2,v3 --product-split core_frozen
   python scripts/eval_ablation.py --variants v0,v1,v2,v3 --product-split challenge_frozen
   ```
   Shadow Set 不参与版本选择。
2. **阶段二：最终版本跑 Shadow（30 次）**：依据 128 条的 strict_success_rate
   选出最终版本，单版本跑影子集：
   ```bash
   python scripts/eval_ablation.py --variants <最终版> --product-split shadow_frozen
   ```
3. **阶段三：随机稳定性（追加 120 次）**：从冻结集选 60 条
   （Core 按 subset 分层抽 30 + Challenge 全部 30，清单固化为
   `data/eval/production_v1/stability_60.json`，seed=42，已提交入库不再重抽），
   最终版本共跑 3 遍；第一遍已含在阶段一的 512 次里，只补两遍：
   ```bash
   python scripts/eval_ablation.py --variants <最终版> --cases-file data/eval/production_v1/stability_60.json --repeat-index 2
   python scripts/eval_ablation.py --variants <最终版> --cases-file data/eval/production_v1/stability_60.json --repeat-index 3
   python scripts/eval_ablation.py --stability-merge --merge-runs <阶段一目录>,<repeat2目录>,<repeat3目录>
   ```

### 执行次数口径（必须准确表述）

| 阶段 | 计算 | 次数 |
| --- | --- | --- |
| 阶段一 | 128 条 × 4 版本 | 512 |
| 阶段二 | 30 条 × 1 最终版 | 30 |
| 阶段三 | 60 条 × 2 遍追加 | 120 |
| **正常环境总计** | 512 + 30 + 120 | **662** |
| 最终架构自身 | 128 + 30 + 120 | **278** |

共 **158 条独立冻结任务**（Core+Challenge 128 + Shadow 30），4 种架构累计完成
662 次正常环境执行。**禁止表述为"4 个版本一共只跑 278 次"。**

### 公平控制（五同，落盘 comparison.json 的 `fair_controls`）

同模型同温度、同工具 Schema 与快照、同 plan_and_critique 版本与规则集、
同 token/步数硬上限（`--token-budget`）、同评分器版本。
`execution_plan`（512/30/120/662/278/158）同步落盘。

### 架构纪律自动校验

每条 case 的 architecture_policy 评分器校验 V0–V2 无外部 critic 事件、
V3 恰好一次且返工 ≤1，违规直接 fail。

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

日常 dev（反复跑）→ 阶段一 core+challenge 512 定版对比 → 阶段二 shadow 30
仅最终版 → 阶段三 stability_60 补跑两遍 + 稳定性合并报告。

日常 remediation 只运行 strict evaluator + trace；不要每次修改都追 Judge 分数。
当系统发生明显版本变化、strict 连续稳定或准备 freeze candidate 时，对对应的完整
Dev30 运行一次 Gemini Judge，用于发现规则难覆盖的系统性规划质量问题。进入版本比较
后冻结 Judge rubric/prompt；如必须修改则提升版本并重建基线。

## 明确不纳入本次（后续项）

- 内部闭环独立消融（A0–A3 四格，需 plan_and_critique 开关，当前 variants 未实现）；
- 外部 40 次 fault runs（record/replay 范围，`faults/fault_configs.json` 仅存档）；
