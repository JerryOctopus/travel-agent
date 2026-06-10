# 评估报告（EVALUATION）

> 自动生成：`python scripts/eval_agent.py --write-report`。
> 评估在离线确定性路径下运行（无 LLM key），保证可复现。

## 样本与局限

- NL 用例：6 条人工构造小样本（`eval/cases.json`）；
- 闭环用例：4 条带约束的合成 profile；
- 指标用于自检与回归，**不代表真实分布上的泛化能力**，后续需用真实 query 扩充。

## 理解层 / Agent 层

- 目的地抽取准确率：1.0
- 天数抽取准确率：1.0
- 追问触达准确率：1.0
- 任务完成率（产出可用行程）：1.0
- 平均工具调用步数：4.5

## 规划层

- critic 通过率：1.0

## 闭环层（critic→reviser 价值，最重要）

- 修正前违规项合计：5
- 修正后违规项合计：0
- 违规项下降比例：1.0
- 修正后通过率：1.0

### 闭环明细

| 用例 | 修正前 | 修正后 | 闭环轮数 | 是否通过 |
| --- | --- | --- | --- | --- |
| hangzhou_1d_museum_uncovered | 1 | 0 | 1 | 是 |
| hangzhou_1d_multi_interest | 2 | 0 | 1 | 是 |
| shanghai_2d_multi_interest | 1 | 0 | 1 | 是 |
| chengdu_1d_food_museum | 1 | 0 | 1 | 是 |
