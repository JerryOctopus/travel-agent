# V0–V3 架构消融实验（Ablation Study）

> 一句话定位：证明多智能体不是「为了复杂而复杂」。唯一自变量是 Agent 编排架构，
> 其余一切（模型、工具、数据、测试集、预算、业务规则）全部共享；每一层复杂度
> 必须用指标证明自己值得。

代码入口：[`src/travel_agent/orchestration/variants/`](../src/travel_agent/orchestration/variants/)
评测脚本：[`scripts/eval_ablation.py`](../scripts/eval_ablation.py)
单元测试：[`tests/test_variants.py`](../tests/test_variants.py)

## 四个版本定义

Step 4 收敛后，四个版本**共用同一个** `MultiAgentEngine`
（`orchestration/multi_agent/`），唯一自变量是 `EngineCapabilities`
（编排模式 / 派工方式 / Reviewer 开关 / 返工上限）：

| 版本 | 架构 | EngineCapabilities | 入口 |
| --- | --- | --- | --- |
| **V0** | 单 ReAct Agent 基线 | `mode=single, dispatch=none` | `run_variant_turn("v0", ...)` |
| **V1** | 确定性规则派工多 Agent | `mode=orchestrated, dispatch=fixed`，无 Reviewer | `run_variant_turn("v1", ...)` |
| **V2** | 动态 Main Orchestrator（无 Reviewer） | `mode=orchestrated, dispatch=dynamic` | `run_variant_turn("v2", ...)` |
| **V3** | 完整系统 = Production Full | `dispatch=dynamic, reviewer_enabled=True, max_rework=1` | `run_variant_turn("v3", ...)` |

```text
V0:  user → [单 ReAct Agent（全工具）] → reply

V1:  Orchestrator 按 task_type 查表生成批次任务 → 领域 Subagent → Planner → reply
     （确定性规则派工，无 LLM 路由）

V2:  Main Orchestrator 动态决定派工与停止 → 领域 Subagent → Planner → reply

V3:  V2 + Reviewer（无工具单次 LLM 语义审查）；
     不通过 → 定向返工一次 → 无论结果直接交付
```

生产入口（`runtime.run_production_turn`）固定使用 `PRODUCTION_CONFIG`
（= `FULL_CONFIG` = `V3_CONFIG`），不读取任何 variant；版本选择只能通过
`run_variant_turn` 或 Harness `environment.variant` 显式指定。

## 控制变量表（公平性保证）

| 维度 | 共享方式 |
| --- | --- |
| 主模型 | 四版本统一用 `settings.llm`（同一 provider/model）；评测不提供逐版本换模型入口 |
| 工具 | 同一份 toolkit；每个 Subagent 只暴露 `SUBAGENT_REGISTRY` 白名单内的工具子集，Orchestrator 不持有渲染/派工外重工具，工具实现零分叉 |
| 工具数据 | 同一 harness snapshot（`data/eval/live_tools/`） |
| 测试集 | production_v1.1 192 条（dev 34 调优 / core+challenge 128 定版 / shadow 30 仅最终版，`data/eval/production_v1/`，详见 docs/EVALUATION_PRODUCT.md） |
| 预算边界 | 同一 Token 硬上限（`--token-budget` / `variant_token_budget`），四版本同样"硬" |
| 业务规则 | Orchestrator / Subagent / Reviewer 的 prompt 由同一套构建函数生成，共享业务规则；V0 基线复用生产 `_run_react` 全工具链路，行为不变 |
| Subagent 实现 | V1–V3 复用同一份 `SUBAGENT_REGISTRY` 定义与 executor ReAct 循环，零分叉 |
| 确定性安全网 | 行程缺失时补齐强制尾链（`_complete_required_plan`）对四版本恒定施加——这是确定性工程控制，不算架构变量 |

## 预算规则（防死循环硬边界）

| 限额 | V0 | V1 | V2 | V3 |
| --- | --- | --- | --- | --- |
| 全局步数 | 20（recursion_limit） | Engine 内部编排硬停 | Engine 内部编排硬停 | Engine 内部编排硬停（多出的留给 Reviewer 与一次返工） |
| 单 Subagent 派工次数 | — | ≤3 | ≤3 | ≤3（返工计入额度） |
| 单次 Subagent 执行内工具调用 | — | ≤3 | ≤3 | ≤3 |
| 定向返工次数 | — | — | — | ≤1（返工后仍不通过则带告警交付，禁止全图重做） |
| Token 硬上限 | 轮末审计标记 `budget_exceeded` | 编排内部硬停 | 编排内部硬停 | 编排内部硬停 |

Token 口径与 `agent/evaluation_trace.py` 一致：累计 `kind == "model"` 记录的
`total_tokens`。Engine 在编排内部做硬停止；V0（单 ReAct）无法中途截断，由
`registry._audit_token_budget` 做轮末总量审计——超限 case 标记
`budget_exceeded`，保证成本边界对所有版本同样严格。

## V3 Reviewer 设计

Reviewer 不属于 `SUBAGENT_REGISTRY`，是 Engine 直接控制的**无工具单次 LLM
语义审查**（7 项 checklist：硬约束遗漏、时间冲突、预算超限、无证据地点/路线、
重复安排、未经授权操作、是否需要返工）。输出 `{verdict, target_agent, issues[]}`：

- `verdict=pass` → 直接交付；
- `verdict=rework` 且未返工过 → 定向派工给 `target_agent` 返工一次 → 重新审查；
- 返工后仍不通过 / 配额耗尽 / 预算耗尽 → `completed_with_warnings` 带告警交付。

绝不进入第二次返工循环（`max_rework=1`）。轻量任务（非完整行程规划）
不触发 Reviewer。

## 如何运行评测

```bash
# 小样本冒烟（需 LLM key）
.venv/bin/python scripts/eval_ablation.py --variants v0,v1,v2,v3 --product-split dev --limit 3

# dev 集调优（30 条，可反复跑）
.venv/bin/python scripts/eval_ablation.py --product-split dev

# 阶段一：定版集正式对比（core 94 + challenge 34，禁止用于调优）
.venv/bin/python scripts/eval_ablation.py --product-split core_frozen
.venv/bin/python scripts/eval_ablation.py --product-split challenge_frozen

# 阶段二/三：影子集与稳定性补跑仅允许最终选定版本（规程见 docs/EVALUATION_PRODUCT.md）
```

产物：`data/eval/ablation/<run_id>/` 下每版本一份独立报告（`v0/…v3/`，复用
harness reporting），外加跨版本 `comparison.json` / `comparison.md`。

交互会话中不提供版本切换开关：生产入口固定 V3；版本选择只能通过
`run_variant_turn("v0".."v3", ...)` 或 Harness `HarnessEnvironment(variant="v1")`
显式指定（如 `scripts/eval_ablation.py --variants v0,v1,v2,v3`）。

## 指标解读

对比表包含：strict 通过率、任务完成率、硬约束通过率、critic 通过率、追问准确率、
平均 token、平均步数、平均工具调用数、预算超限 case 数、未完成 case 数与原因、
时延 p50/p95、**成本归一化成功率**（任务完成率 / 万 token，防止「赢只是因为花得多」），
以及 V3 的返工触发率与交付分布。

逐对结论回答三个问题：

- **V0 → V1**：任务拆分本身值不值（规则派工多 Agent 相对单 Agent）；
- **V1 → V2**：动态路由的增量（动态 Orchestrator 相对确定性规则派工）；
- **V2 → V3**：Reviewer 的增量（语义审查 + 定向返工相对无验收）。

判定规则：完成率显著提升且成本未升 → 该层复杂度有收益；完成率提升但成本上升 →
看成本归一化成功率；完成率下降或持平而成本上升 → 该层复杂度没有收益。

## 已确认的两个公平性决策

1. **`plan_and_critique` 四版本完整保留**（plan → critic → revise 闭环）。它是
   确定性规划工具而非编排架构，不算本次消融的自变量；
2. **Token 采用硬上限**：超限即优雅终止并标记 incomplete / budget_exceeded，
   不做软警告。

## 与生产入口的关系

生产入口（`runtime.run_production_turn`）固定使用 Multi-Agent Full（V3），
不经过本模块。旧 M9 五层编排（`orchestration/layered_agent.py`）已在 Step 4
删除，历史代码保留在 git tag `m9-layered-final`。
