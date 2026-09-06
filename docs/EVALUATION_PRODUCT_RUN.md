# Dev34 正式验收运行报告

- 数据集：`travel-agent-eval-production-v1.1`
- 被测模型：`deepseek / deepseek-v4-flash`
- 模型参数：temperature 0.2，thinking off，Hybrid flags off
- 工具：configured real tools（AMap）
- Judge：`siliconflow / Qwen/Qwen3.5-397B-A17B`，temperature 0，thinking off
- 正式结果：连续完整验收 **2/2 通过**

| 指标 | R13 | R14 | 门槛 |
| --- | ---: | ---: | ---: |
| Case 执行 | 34/34 | 34/34 | 34/34 |
| Strict | 28/34 | 28/34 | ≥26/34 |
| 完整行程 strict | 17/22 | 17/22 | ≥16/22 |
| 非行程 strict | 11/12 | 11/12 | ≥9/12 |
| 硬约束 | 34/34 | 34/34 | ≥31/34 |
| 长程 strict | 3/4 | 3/4 | ≥3/4 |
| Grounding / Authorization / Schema / Architecture | 34/34 | 34/34 | 34/34 |
| Judge 完成 | 17/17 | 17/17 | 100%，且 ≥16 |
| Judge 平均分 | 84.94 | 82.88 | ≥75 |
| Judge reasonable | 76.47% | 76.47% | ≥70% |
| Judge critical issue | 17.65% | 17.65% | ≤20% |

两轮各 case 恰好执行一次，系统错误、模型错误、工具错误和模型切换均为 0。双轮联合检查器返回 `passed=true`，两轮所有冻结指纹一致。

运行产物：

- [R13 summary](../data/eval/product/runs/dev34_acceptance_r13_20260905cz/summary.json)
- [R13 quality report](../data/eval/product/runs/dev34_acceptance_r13_20260905cz/quality_report.md)
- [R14 summary](../data/eval/product/runs/dev34_acceptance_r14_20260905da/summary.json)
- [R14 quality report](../data/eval/product/runs/dev34_acceptance_r14_20260905da/quality_report.md)
- [Dev34 逐 case 报告](DEV34_CASE_REPORT.md)
- [完整验收与基线对比](DEV34_ACCEPTANCE_REPORT.md)
