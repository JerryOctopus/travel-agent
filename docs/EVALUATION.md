# 评估报告（EVALUATION）

> 自动生成：`python scripts/eval_agent.py --write-report`。
> 评估在**离线确定性路径**下运行（强制 `provider=rule`），保证可复现。

## 样本与局限

- NL 用例：10 条人工构造小样本（`eval/cases.json`）；
- 闭环用例：4 条带约束的合成 profile；
- 指标用于自检与回归，**不代表真实分布上的泛化能力**，后续需用真实 query 扩充。

## 理解层 / Agent 层

- 目的地抽取准确率：1.0
- 天数抽取准确率：1.0
- 追问/澄清触达准确率：0.9
- 意图门控准确率（寒暄/缺信息不误规划）：0.6667
- 任务完成率（产出可用行程）：1.0
- 平均工具调用步数：4.7

## 规划层

- critic 通过率：0.8571

## ChinaTravel-style 计划质量层

这组指标按 ChinaTravel 的 planning eval 口径做轻量实现：
先看是否交付计划，再看环境可行性、逻辑约束与最终通过率。

- DR / Delivery Rate（成功交付行程）：1.0
- EPR / Environmental Pass Rate（环境可行性）：1.0
- LPR / Logical Pass Rate（硬约束满足）：1.0
- FPR / Final Pass Rate（环境 + 约束 + 偏好整体通过）：1.0
- C-LPR / Conditional Logical Pass Rate（环境通过后的逻辑通过）：1.0
- Preference Pass Rate（软偏好覆盖）：1.0
- 餐厅饭点有效率：1.0
- 路线可行率：1.0
- 单日负载有效率：1.0
- POI 城市一致率：1.0

## 闭环层（critic→reviser 价值，最重要）

- 修正前违规项合计：2
- 修正后违规项合计：0
- 违规项下降比例：1.0
- 修正后通过率：1.0

### 闭环明细

| 用例 | 修正前 | 修正后 | 闭环轮数 | 是否通过 |
| --- | --- | --- | --- | --- |
| hangzhou_1d_museum_uncovered | 1 | 0 | 1 | 是 |
| hangzhou_1d_multi_interest | 1 | 0 | 1 | 是 |
| shanghai_2d_multi_interest | 0 | 0 | 0 | 是 |
| chengdu_1d_food_museum | 0 | 0 | 0 | 是 |

## 多轮修改层

- 多轮用例数：1
- 多轮修改成功率：1.0
- 多轮最终通过率：1.0

| 用例 | 轮数 | 修改成功 | 最终通过 | 问题 |
| --- | --- | --- | --- | --- |
| food_then_nature_food | 2 | 是 | 是 | 无 |

## 三句话结论（作品集 / 面试）

1. 可控规划子图通过 critic→reviser 闭环，把合成用例违规项从 2 降到 0（下降 100%）。
2. 离线 Agent 路径任务完成率 1.0，澄清/追问触达率 0.9，平均 4.7 步工具调用。
3. 生产入口固定 Multi-Agent Full（V3）：Orchestrator 派工五个执行型 Subagent，Reviewer 最多一次修复；无 key 时用 deterministic fallback 复现指标。
