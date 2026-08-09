# Multi-Agent 架构改造 —— Review 交接文档（Step 4 完成时点）

> 本文档用于切换账号后独立 Code Review。记录最终架构、分步交付、调用链、
> 关键设计约束、删除清单、测试现状与 Review 风险点。
> 生成时点：Step 4 完成、正式消融实验未开始。

---

## 1. 最终架构：Production = Full = V3

生产架构固定为一条链，不保留任何运行时开关：

```text
确定性准入层（turn_analysis + 槽位追问）
  → Main Orchestrator（动态派工，不持有渲染/业务重工具）
  → 五个执行型领域 Subagent（attraction / restaurant / hotel / transport / planner）
  → Planner Subagent（内部调用 plan_and_critique 确定性子图）
  → Engine 直接调用 Semantic Reviewer（无工具单次 LLM 调用，仅完整行程触发）
  → 最多一次定向修复周期（max_rework = 1）
  → Renderer Gate 统一渲染（render_itinerary 绑定 plan_artifact_id）
```

代码事实：

```python
PRODUCTION_CONFIG = FULL_CONFIG = V3_CONFIG   # 同一 EngineCapabilities 对象
```

- 生产入口 `runtime.run_production_turn()` 固定使用 `PRODUCTION_CONFIG`，
  **不读取任何 variant**；`OrchestrationSettings` 只剩 `variant_token_budget`
  （消融共用 Token 硬上限）。
- `Reviewer` 不属于 `SUBAGENT_REGISTRY`，由 Engine 直接执行单层无工具 LLM 调用。
- 技术栈：LangGraph `create_react_agent`（Orchestrator / Subagent ReAct 循环）、
  dataclass Schema、Local/MCP 双源工具接入，V0–V3 由统一 `MultiAgentEngine` 驱动。

## 2. V0–V3 的区别（唯一自变量：EngineCapabilities）

四个版本**共用同一个** `MultiAgentEngine`（`orchestration/multi_agent/`），
variants 包（`orchestration/variants/`）只是薄适配器，无独立 Graph：

| 版本 | 语义 | EngineCapabilities |
| --- | --- | --- |
| V0 | 单 Agent 基线（复用生产 `_run_react` 全工具链路） | `mode=single, dispatch=none` |
| V1 | 确定性规则派工（task_type 查表，无 LLM 路由） | `mode=orchestrated, dispatch=fixed`，无 Reviewer |
| V2 | 动态 Main Orchestrator，无 Reviewer | `mode=orchestrated, dispatch=dynamic` |
| V3 | Production Full = 动态 Orchestrator + Reviewer + 一次修复 | `dispatch=dynamic, reviewer_enabled=True, max_rework=1` |

版本选择**只能**通过 `run_variant_turn("v0".."v3", ...)` 或 Harness
`HarnessEnvironment(variant=...)` 显式指定；产品入口无法切到 V0–V2。
四版本共享同一 turn-scoped meter 与调用前硬预算策略。

## 3. Step 1–Step 4 分别完成了什么

| Step | Commit | 内容 |
| --- | --- | --- |
| Step 1 基础框架 | `2b6ef64` | `orchestration/multi_agent/` 骨架：schemas（EngineCapabilities/TurnOutcome）、engine、orchestrator、registry（SUBAGENT_REGISTRY）、dispatch_rules、review、runner + 401 行框架测试 |
| Step 2 并发与锁安全 | `37aa6bd` | ArtifactStore 内部 RLock + 元数据、agent_trace 原子追加、dispatch 锁隔离、store_view、并发测试 |
| Step 3 真实执行链 | `98b1a21` | 受限 ReAct Subagent 执行器（executor）、V1 规则派工（fixed_dispatch）、V2/V3 动态 Orchestrator（orchestrator_agent）、Reviewer 一次修复周期、Renderer Gate（render_gate）、turn_analysis / evaluation_trace 基线、486 行引擎测试 |
| Step 4 生产切换与收敛 | `dc476fa` | 生产入口切换 `run_production_turn`；V0–V3 收敛为同一 Engine 薄适配器；删除 M9/layered；外围与 agent_trace 迁移 |
| Review 修复 Batch 2 | `4edc338` | 多轮上下文、dispatch ledger、硬 timeout/step/tool 限制、HTTP 隔离取消、统一生命周期、evaluator 去标签泄漏 |
| Review 最终收口 | `1acc864` | 全链 meter、公平性、Local/MCP parity、短锁、Artifact/Runner/trace 安全、repair/evaluator 验收、clean snapshot 测试 |

## 4. 关键 Git commit 与 tag

- Step 3 commit：**`98b1a21`**（Step 4 的基线）
- M9 历史 tag：**`m9-layered-final`**（指向 `98b1a21`，保留删除前的完整 M9 五层代码）
- Step 2 commit：`37aa6bd`；Step 1 commit：`2b6ef64`
- Step 4 与 Review 修复：`dc476fa`、`4edc338`、`1acc864`

## 5. 调用链

### 生产调用链

```text
server / demo / chat_demo
  → run_production_turn()（不读 variant）
    → analyze_travel_turn → _prepare_travel_turn（按任务类型补槽位/追问）
    → _run_multi_agent → MultiAgentEngine(PRODUCTION_CONFIG)
        Orchestrator（工具白名单去掉 render_itinerary/render_map）
          → dispatch_subagent → 五 Subagent（各自 SUBAGENT_REGISTRY 工具白名单）
          → Planner（plan_and_critique 确定性子图，产出 plan artifact）
          → Reviewer（仅 FULL_TRIP_PLAN 触发；pass→交付 / rework→定向返工一次）
          → Renderer Gate（render_itinerary 绑定 plan_artifact_id + render_map）
    → 真实路径异常或离线：_run_fallback（与生产共享的确定性兜底，不宣称真实 Agent）
```

### 评测调用链

```text
eval_ablation.py --variants v0..v3 / harness.cli / eval_product_multi_model / eval_chinatravel
  → HarnessEnvironment(variant="v0".."v3")（None = 生产）
  → runner._dispatch_turn：有 variant → run_variant_turn(variant, ...)，否则 run_production_turn
  → run_variant_turn 查 VARIANTS 注册表 → 同一 MultiAgentEngine(VARIANT_PRESETS[name])
  → _record_engine_metrics（variant/mode/dispatch/status/rework_used/review{verdict,delivery,rework_used}）
  → _audit_token_budget（轮末总量审计，超限标记 budget_exceeded）
```

## 6. 五个执行型 Subagent、Reviewer、修复周期与 Renderer Gate

- **五个 Subagent**：`attraction` / `restaurant` / `hotel` / `transport` / `planner`，
  定义在 `SUBAGENT_REGISTRY`；每个只暴露白名单内工具（工具实现零分叉）。
  执行器是受限 ReAct 循环（`executor.py`，`create_react_agent` + 工具/步数限额）。
- **Reviewer**：Engine 直调、无工具、单次 LLM 语义审查（7 项 checklist：
  硬约束遗漏、时间冲突、预算超限、无证据地点/路线、重复安排、未经授权操作、
  是否需要返工），输出 `{verdict, target_agent, issues[]}`。
- **修复周期**：`max_rework=1`。`verdict=rework` 且未返工过 → 定向派工给
  `target_agent` 返工一次 → 重新审查；返工后仍不通过则 `completed_with_warnings`
  带告警交付，绝不进入第二次返工。轻量任务（非完整行程）不触发 Reviewer。
- **Renderer Gate**（`render_gate.py`）：渲染统一由 Engine 在 Reviewer 之后执行；
  `render_itinerary` 必须绑定 Planner 产出的 `plan_artifact_id`；
  `render_itinerary` / `render_map` 不暴露给 Main Orchestrator
  （`orchestrator_agent.py`：`allowed = ORCHESTRATOR_ALLOWED_TOOLS - {"render_itinerary", "render_map"}`）。

## 7. 锁、Artifact、agent_trace 设计

- **锁粒度**：LLM、Provider 与 MCP 网络 I/O 不持共享 session lock；
  Profile / Artifact / Trace 的实际读写由各对象内部短 RLock 保护。锁顺序不跨 I/O，
  dispatch 与 Subagent 不再形成嵌套锁链。
- **并发安全下沉到共享状态读写点**：ArtifactStore 内部 RLock + 记录元数据；
  `AgentTraceLog` 原子追加。
- **Artifact 纪律**：`dispatch_subagent` 不持有 session lock；Planner 只按明确
  `artifact_id`（`get_record` / `get_payloads`）读取并行结果，**禁止**
  `ctx.store.latest()` 依赖。
- **agent_trace**：统一可观测性（不再兼容 layer_trace）。条目结构
  `{request_id, task_id, agent, kind, status, detail, created_at}`；
  `kind ∈ {orchestration, subagent, review, render}`；subagent 的工具轨迹在
  `detail.tool_trace`；成功状态为 `completed` / `completed_with_warnings`。
  评测指标键：`agent_trace_rate`、`agent_completed_rate`、
  `required_agents_rate`、`research/planning_agent_tool_rate`；
  artifact 标记 `architecture: "multi_agent_full_v3"`。

## 8. 已删除的 M9/layered 内容

- `src/travel_agent/orchestration/layered_agent.py`（五层 `LayeredTravelAgent`、NodeManager）；
- `run_layered_turn` / `run_deterministic_layered_turn` 及 layered 相关导出；
- settings：`layered_enabled`、`max_layer_retries`、`max_react_retries`、
  仅服务 M9 的 strict_validation 部分；
- `layer_trace`（含 `build_layer_trace_card` 渲染卡片、前端 `.layer-row`）；
- M9 Harness 对照逻辑（`build_orchestration_comparison`）；
- M9 专属测试：`test_layered_agent.py`、`test_layered_orchestration.py`、
  `test_node_manager.py`；
- M9 指标脚本与文档入口：`scripts/eval_layer_metrics.py`、`debug_layer_trace.py`；
- 旧 variants 独立实现（v0_single / v1_fixed_dag / v2_supervisor /
  v3_supervisor_critic / state / prompts 各自的 StateGraph/Supervisor/Worker/Critic）；
- `_run_react_with_retries`（重试语义并入 Engine 修复周期）及对应测试。

历史代码可从 tag `m9-layered-final` 找回。

## 9. 当前测试结果

```text
混合工作区 pytest tests/  →  354 passed, 1 skipped
独立 clean checkout pytest tests/  →  163 passed
grep -rnE "layered|layer_trace" src/ tests/  →  零命中
```

Step 4 十项验收（`tests/test_variants.py` 等）：
Production/Full/V3 同一配置对象；产品入口无法切换 V0–V2；V0–V3 同一 Engine；
旧 variants 无独立 Graph；生产完整行程执行一次 Reviewer；简单任务不执行
Reviewer；渲染经 Renderer Gate；offline fallback 正常；layered 测试与入口已删；
Harness 能显式运行 V0–V3。

## 10. Review remediation 收口

- turn-scoped meter 统一统计 input/output/total tokens、LLM/tool/dispatch calls、
  role latency 与可选成本；token、LLM-call、tool-call 预算均在调用前阻断；
- Local/MCP 固定同一 14 工具逻辑契约、公开 Schema、权限过滤和结果信封；
  MCP 不兼容或缺工具时 fail-fast，不再静默回退不同语义；
- Provider/LLM/MCP 网络 I/O 不持 session lock；Profile、Artifact、Trace、Memory
  仅在实际读写处短锁；Artifact 和 trace 对外返回深拷贝；
- Runner 强制 session-scoped，trace 按 request_id 获取，repair 后执行 deterministic
  critic/Gate/critical 验收，失败统一为 incomplete；
- evaluator 将 raw failure、fallback triggered、final outcome 分离，actual outcome
  不读取 gold，required agents 按 task policy 计算。

已知并接受的运行时限制：Python 后台线程超时后无法安全强杀，底层 Provider 调用
可能短暂继续占用连接/线程资源；请求使用隔离 snapshot 和 cancellation guard，超时后
不得合并 Artifact/Profile/trace/final result。MCP 迟到 artifact 即使短暂留在远端 session，
也不会被后续 Planner 消费，因为 Planner 只接受本轮显式 artifact_id。

## 11. 尚未开始的工作

1. **独立 Code Review**（新账号进行，见下节风险点）；
2. **Dev 冒烟**：`scripts/eval_ablation.py --variants v0,v1,v2,v3 --product-split dev --limit 3`（需 LLM key）；
3. **正式消融实验**：dev 调优 → core_frozen/challenge_frozen 定版对比 →
   shadow 与稳定性补跑（规程见 `docs/EVALUATION_PRODUCT.md` 与
   `docs/ABLATION_V0_V3.md`）。用户已明确：Step 4 完成后停止，不自动开始。

## 12. 新账号 Review 时应重点检查的风险点

1. **生产入口唯一性**：确认除 `run_production_turn` 外无任何路径能绕过
   `PRODUCTION_CONFIG`；`server.py` / MCP / demo 是否全部走同一入口；
   交互会话不得存在 variant 切换开关。
2. **variants 薄适配纯度**：`orchestration/variants/` 只能依赖
   `MultiAgentEngine` 与 runtime 复用函数；不得复活独立 Graph/Supervisor。
3. **Reviewer 边界**：仅完整行程（FULL_TRIP_PLAN）触发；`max_rework=1` 硬边界；
   critical 类 ReviewIssue 必须阻断交付，不可仅带 warning 后标记成功。
4. **Renderer Gate**：`render_itinerary` 是否总是绑定本轮明确 plan_artifact_id；
   Orchestrator 工具白名单是否确实剔除渲染工具；V0/V1/V2 是否也经 Gate 渲染。
5. **锁与并发**：新增工具不得重新让 Provider/LLM/MCP I/O 持共享状态锁。
6. **Planner 读取纪律**：Planner 只能按 artifact_id 读取；Review 是否有
   `ctx.store.latest()` 回潮。
7. **离线 fallback 等价性**：`_run_fallback` 与 variant 离线路径共用同一链路，
   且 `used_real_agent=False`；离线不得宣称真实 Agent。
8. **评测口径一致性**：`build_agent_events`（review/orchestration 条目 →
   external_semantic_critic / targeted_rework 事件）与 architecture_policy
   （V0–V2 禁外部 critic；V3 恰好一次、rework ≤ 1）是否仍对齐。
9. **Token 预算硬边界**：四版本同一 `variant_token_budget`；超限 case 必须
   标记 `budget_exceeded` 而不是软警告。
10. **工作区混杂**：本次暂存刻意排除了与 Step 4 无关的既有改动
    （LaTeX 文件、Zhipu/GLM Provider 配置、planning/critic/provider/storage/
    workflow 等），Review 时只针对暂存内容；`git status` 中其余改动属于
    其他工作线，勿混入本次提交。
