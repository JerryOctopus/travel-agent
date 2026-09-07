# Dev34 闭环验收与基线对比

> 历史记录：本文对应旧 `deepseek-chat` 与旧 Dev34 门槛，只保留作基线，不能作为当前
> DeepSeek V4 Flash 发布候选的准入证明。当前合同与命令见 `EVALUATION_PRODUCT.md`。

## 结论

Dev34 指标提升闭环于 2026-09-05 完成。正式检查器对两次连续完整运行 `dev34_acceptance_r13_20260905cz` 和 `dev34_acceptance_r14_20260905da` 均返回 `passed=true`；双轮联合检查也返回 `passed=true`，没有剩余验收失败项。

## 基线与提升口径

| 口径 | Strict | 完整行程 | 非行程 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 历史原始 run | 20/34 | — | — | 旧 evaluator 原始分数，仅保留历史记录 |
| 计划启动 Artifact 合同审计 | 14/34 | 10/22 | 0/12 | 合同重解释后的基线；evaluator correction 不算产品提升 |
| 正式 R13 | 28/34 | 17/22 | 11/12 | 全新真实执行 |
| 正式 R14 | 28/34 | 17/22 | 11/12 | 冻结候选后的独立复跑 |

相对计划启动合同基线，真实执行 Strict 增加 14 条、完整行程增加 7 条、非行程增加 11 条。历史 20/34 与合同审计 14/34 的 6 条差异属于 evaluator 口径变化，未计入产品能力提升。

## 两轮正式指标

| 指标 | 要求 | R13 | R14 |
| --- | ---: | ---: | ---: |
| 运行完整性 | 34/34，各一次 | 34/34 | 34/34 |
| Strict | ≥26/34 | 28/34 | 28/34 |
| Full-itinerary strict | ≥16/22 | 17/22 | 17/22 |
| Non-itinerary strict | ≥9/12 | 11/12 | 11/12 |
| Hard constraints | ≥31/34 | 34/34 | 34/34 |
| 4 个长程 case | ≥3/4 | 3/4 | 3/4 |
| Grounding | 34/34 | 34/34 | 34/34 |
| Authorization | 34/34 | 34/34 | 34/34 |
| Tool schema | 34/34 | 34/34 | 34/34 |
| Architecture policy | 34/34 | 34/34 | 34/34 |
| Judge 覆盖 | 100%，且 ≥16 | 17/17 | 17/17 |
| Judge 平均分 | ≥75 | 84.94 | 82.88 |
| Judge reasonable | ≥70% | 76.47% | 76.47% |
| Judge critical issue | ≤20% | 17.65% | 17.65% |

两轮红线检查均无失败项；系统、模型和工具环境错误均为 0。被测模型固定为 `deepseek-chat`，未发生模型接力；Judge 固定为 SiliconFlow `Qwen/Qwen3.5-397B-A17B`。

## 稳定性与逐 case 变化

- 两轮都未通过：`dev_001`、`dev_004`、`dev_005`、`dev_007`、`lh_12_001`。
- R13 未通过、R14 通过：`dev_018`。
- R13 通过、R14 未通过：`dev_023`。
- 其余 27 条两轮 strict 状态一致。

这说明总体与全部门槛稳定复现，但单 case 仍受生成随机性影响；验收结论是“达到冻结合同的产品交付门槛”，不是“34/34 strict”。

## 质量规则、成本与延迟

| 项目 | R13 | R14 |
| --- | ---: | ---: |
| 规则评估行程 | 18 | 18 |
| 规则质量通过 | 17/18 | 17/18 |
| Grounded / 无时序冲突 / 换乘可行 / 路线覆盖 / must-visit / 营业有效 | 100% | 100% |
| DeepSeek tokens | 1,306,430 | 1,299,477 |
| LLM calls | 237 | 236 |
| Tool calls | 301 | 298 |
| case 延迟 P50 | 22.15s | 20.87s |
| case 延迟 P95 | 47.94s | 51.72s |

两轮 DeepSeek 合计 2,605,907 tokens。Harness 没有配置供应商价格表，因此 `total_estimated_cost_usd` 为 0；这代表“未估价”，不代表真实调用免费。实际人民币消费应以 DeepSeek、SiliconFlow 与高德账单为准。

## 关键修复

- 统一 Dev34 的 Artifact 合同、任务路由、报告与评分语义，专业任务稳定交付 route、candidate comparison 或 local adjustment Artifact。
- 收紧 current Artifact 与 Reviewer 生命周期：修订原子应用、重新验证、拒绝 stale Artifact，并修复返程终点在 revision 中丢失的问题。
- 增加末日返程路线候选 fan-out，确保可选末站被裁剪后仍有真实返程路线证据。
- 为 AMap 增加全局 QPS 节流与瞬时限流短退避；日配额错误仍 fail-closed，不被重试掩盖。
- 用餐窗口为相邻活动保留 15 分钟衔接；饮食限制采用“确认或替换”策略，缺乏正向合规证据的具体餐厅不进入行程。
- 多轮约束采用 latest-write-wins、tombstone、revision/hash 与 stale rejection，完整保留预算、删除项、住宿、饮食、固定事件和返程状态。

## 验证与产物

- 本地测试：`1111 passed, 1 skipped`。
- `compileall`：通过。
- `git diff --check`：通过。
- R13：[summary.json](../data/eval/product/runs/dev34_acceptance_r13_20260905cz/summary.json)、[quality_report.md](../data/eval/product/runs/dev34_acceptance_r13_20260905cz/quality_report.md)。
- R14：[summary.json](../data/eval/product/runs/dev34_acceptance_r14_20260905da/summary.json)、[quality_report.md](../data/eval/product/runs/dev34_acceptance_r14_20260905da/quality_report.md)、[逐 case HTML](../data/eval/product/runs/dev34_acceptance_r14_20260905da/dev34_case_report.html)。
