# 项目计划

> ⚠️ **状态更新（Step 4）**：本文档是 M0–M9 阶段的历史规划记录。
> 文中 M9 五层 `LayeredTravelAgent` 编排已在多 Agent 架构改造 Step 4 中
> 删除，生产链路收敛为 `MultiAgentEngine`（V0–V3 共用同一 Engine，生产入口
> 固定 Multi-Agent Full = V3），可观测性统一为 agent_trace。
> M9 历史代码保留在 git tag `m9-layered-final`。
> 现行架构见 [ARCHITECTURE.md](ARCHITECTURE.md) 与
> [ABLATION_V0_V3.md](ABLATION_V0_V3.md)。

> 本文档描述「分层 Multi-Agent + 可控规划子图」的 LLM Travel Agent
> 目标架构与迁移路线图。
>
> **作品集主叙事（算法岗 demo）**：外层五层 `LayeredTravelAgent`
> Multi-Agent 负责需求、检索、规划、风险、渲染；内层
> `plan_and_critique` 确定性规划闭环保证行程质量可控、可回归。
>
> **评估支撑**：离线 eval、layer metrics、critic issue reduction 与前端
> 质量/层级 trace 展示。无 LLM key 时走 deterministic toolkit fallback，
> 不声称 Multi-Agent。
>
> 实现模式参考 `../../TravelAgent-AI-main`（借鉴架构与模式，不照搬功能集）。

## 项目定位

项目名称：

```text
Personalized Travel Planning Agent（基于 LangGraph 的分层 Multi-Agent 旅行规划系统）
```

目标岗位：

- AI Engineer
- LLM Application Engineer
- Agent Algorithm Engineer
- AI Algorithm Engineer

核心叙事（作品集 / 面试主版本）：

```text
我用 LangGraph 构建了一个分层 Multi-Agent 旅行规划系统：
外层 requirement → research → planning → risk → render 五层受限 ReAct Agent
按阶段协作，每层只暴露自己的工具白名单，并通过 layer trace / metrics 记录过程；
内层把 planner-critic-reviser 封装为确定性子图 plan_and_critique，
保证行程约束可回归、可量化，而不是让模型手写 Day1/Day2。
配套离线 eval、layer metrics、质量检查卡片与闭环指标报告，
解决「LLM 旅行规划不可控」这一算法岗核心问题。
```

一句话卖点：

```text
五层受限 Multi-Agent 编排 + 确定性 planner-critic-reviser 子图 +
可复现 eval，解决 LLM 旅行规划不可控问题。
```

## 需要避免的方向

不要把项目做成：

```text
1) 套了一层 LLM 的 POI 推荐系统；
2) 通用的单 ReAct 旅行 demo（与参考项目同质，丢掉自己的差异化）；
3) 无 LLM 的伪分层流水线（Python 硬编码调 toolkit，却声称是 Multi-Agent）。
```

应该始终保持主线为：

```text
五层受限 Multi-Agent 编排 + 内嵌 plan_and_critique 确定性规划核心。
```

## Agent 定位（重要）

旧版本是一个**写死的固定流水线**（extractor → recommend → planner →
critic → reviser）。当前目标定位是：

- **外层（作品集主路径）**：五层 `LayeredTravelAgent` Multi-Agent。
  requirement / research / planning / risk / render 每层创建受限 ReAct Agent，
  仅暴露该层工具白名单，并通过 layer trace / metrics 记录协作过程。
- **内层（可控规划核心，差异化）**：`plan_and_critique` 工具封装确定性子图
  plan → critic → revise 闭环；子图内不调 LLM，违规项与修正稿可量化。
- **意图路由（门控）**：`intent.py` 区分 GREETING / CASUAL / TRAVEL；
  非出行走轻量对话；出行缺字段 / 缺偏好时先追问，不提前开规划。
- **单 ReAct / 离线对照**：单 ReAct Agent 与无 LLM key 的确定性 toolkit
  流水线保留为 baseline / fallback，用于评估对照与无 key 演示；叙事上不冒充
  Multi-Agent。

为什么这是工程判断力的体现（面试讲述）：

```text
旅行规划天然分成需求澄清、信息检索、候选规划、风险复核和结果渲染；
这些阶段适合交给受限 Multi-Agent 协作，而不是一个大 Agent 自由发挥。
而 "一天能不能走完、约束有没有满足" 必须可控、可评估，不能交给 LLM 自由发挥。
所以我用五层 Multi-Agent 做编排，用确定性 plan_and_critique 子图做规划质量控制，
再用离线 eval、layer metrics 与质量卡片把价值量化出来。
```

**玩具陷阱（必须避免）**：

- 不要把规划核心做成黑盒大工具 `plan_everything`；
- 不要让五层共用一个无约束大 Agent（失去分层意义）；
- 不要把 M5 做成无 LLM 的伪分层流水线（当前实现偏差，M9 补齐）。

工具必须拆细（`search_poi` / `check_weather` / `plan_route` /
`recommend_candidates` / `plan_and_critique` / `request_travel_info` /
`request_preference_guide` / `search_restaurant` / `estimate_budget` /
`render_*`），让每层 ReAct 在受限工具集中自主编排。

## 目标架构

```text
┌────────┐
│  用户  │
└───┬────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ FastAPI + Web                                            │
│ WebSocket /chat · 地图 · A2UI 卡片 · 用户确认/导出        │
└───┬──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ IntentRouter                 │
│ 区分闲聊 / 缺信息 / 出行任务 │
└───┬───────────────────┬──────┘
    │ non_travel        │ travel
    ▼                   ▼
┌──────────────┐     ┌──────────────────────────────────────┐
│ 轻量对话     │     │ LayeredTravelAgent                   │
│ 不触发规划   │     │ 五层 Multi-Agent 主路径              │
└──────┬───────┘     └───┬──────────────────────────────────┘
       │                 │
       │                 ▼
       │     ┌──────────────────────────────────────────────┐
       │     │ FiveLayerMultiAgent                          │
       │     │                                              │
       │     │ requirement → research → planning → risk     │
       │     │      → render                                │
       │     │                                              │
       │     │ 每层 ReAct 只暴露本层工具白名单              │
       │     └───┬──────────────────────────────────────────┘
       │         │
       │         ▼
       │     ┌──────────────────────────────────────────────┐
       │     │ 工具层                                       │
       │     │                                              │
       │     │ 画像/追问：                                  │
       │     │   update_travel_profile                      │
       │     │   request_travel_info                        │
       │     │   request_preference_guide                   │
       │     │                                              │
       │     │ 调研检索：                                  │
       │     │   search_poi · search_restaurant             │
       │     │   search_hotel · check_weather · plan_route  │
       │     │                                              │
       │     │ 规划决策：                                  │
       │     │   recommend_candidates                       │
       │     │   estimate_budget                            │
       │     │   plan_and_critique 子图                     │
       │     │                                              │
       │     │ 渲染输出：                                  │
       │     │   render_itinerary · render_map              │
       │     │   quality / layer_trace / budget / cards     │
       │     └───┬───────────────────────┬──────────────────┘
       │         │                       │
       │         │                       ▼
       │         │            ┌──────────────────────┐
       │         │            │ MCP Server（可选）   │
       │         │            │ 复用同一套工具实现   │
       │         │            └──────────────────────┘
       │         │
       ▼         ▼
┌──────────────────────────────────────────────────────────┐
│ 三层记忆                                                  │
│ L1 对话压缩 · L2 ArtifactStore · L3 用户画像              │
└───┬──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ ArtifactStore / 会话持久化                                │
│ candidates · ranked · itinerary · weather · layer_metrics │
└──────────────────────────────────────────────────────────┘

旁路与评估：

LayeredTravelAgent
  ├─ fallback / baseline → 单 ReAct / deterministic toolkit
  ├─ trace / metrics     → layer trace / layer metrics
  └─ evaluation          → eval_agent.py / eval_layer_metrics.py
```

### 五层工具映射

| 层 | 允许工具 | 职责 |
|----|----------|------|
| requirement | `update_travel_profile`, `request_travel_info`, `request_preference_guide` | 抽取画像、追问缺失字段与偏好 |
| research | `search_poi`, `check_weather`, `plan_route` | 检索 POI、天气、路线 |
| planning | `recommend_candidates`, `plan_and_critique` | 候选打分 + 可控规划子图 |
| risk | `plan_and_critique` | 复验 / strict 模式下强制工具调用 |
| render | `render_itinerary`, `render_map` | A2UI 卡片与地图数据 |

### 场景标签与层校验

`NodeManager` 按场景标签（`family` / `senior` / `long_trip` / `couple` /
`budget` 等）动态筛选每层工具。`LayerValidator` 最小完备约束：

- **research**：至少一次检索类工具（`search_*` / `check_weather` / `plan_route`）；
  `family` / `senior` / `long_trip` 场景必须含 `check_weather`。
- **planning**：至少一次规划类工具（`recommend_candidates` / `plan_and_critique`）；
  `long_trip` 场景建议含 `plan_route`（可在 research 层已完成）。
- **risk / render**：`strict_validation=true` 时必须产生工具调用。

数据流（一次完整出行请求，Multi-Agent 主路径）：

1. `/chat` 收到用户请求；
2. `IntentRouter` 判断是否旅行任务，非旅行走轻量对话；
3. 旅行任务进入 `LayeredTravelAgent`；
4. requirement 层抽取画像、补齐缺失字段并决定是否追问；
5. research 层检索 POI / 天气 / 路线；
6. planning 层推荐候选并调用 `plan_and_critique`；
7. risk 层复核 critic / constraints，必要时触发回滚或重试；
8. render 层生成地图、行程、质量卡；
9. layer trace、tool trace、artifact、metrics 落盘。

## 现状 → 目标的模块改造映射

- `src/travel_agent/workflow.py`（写死流水线）→ **离线 fallback**；
  主路径为 `LayeredTravelAgent` 分层 Multi-Agent。
- `recommendation.py` → `recommend_candidates` 工具，保留多目标打分亮点。
- `planning.py` + `critic.py` + `reviser.py` → `plan_and_critique` 子图（差异化核心）；
  M9 在 `planning.py` 引入地理聚类（借鉴 `smart_plan_itinerary`）。
- `orchestration/layered_agent.py` → M9 升级为 per-layer `create_react_agent`（参考项目）。
- `nodes/node_manager.py` → M9 新增，工具分层 + 场景标签筛选。
- `providers.py` → 底层 HTTP 复用，上层 LangChain `@tool` / MCP 暴露。
- `dialogue.py` → 已由 `ArtifactStore` + `SessionLifecycleManager` 替代。
- `eval/` → agent 级 + 规划级 + 多 Agent 分层级评估。

## 迁移路线图

排序原则：**先补真实 Multi-Agent 证据，再补产品能力**。M0–M8 已基本完成；
**M9** 聚焦把分层 Multi-Agent 做成可演示、可评估的主路径；M10–M12
再做评估增强、可视化和餐厅/预算功能。

每个里程碑都应能独立演示并留存证据（截图/录屏/日志）。

### M0：技术栈切换 + 最小可跑的 ReAct Agent ✅

目标：先证明「LLM 真的能自主调用工具」，打通最小闭环。

状态：已完成。`create_react_agent` + `lc_tools` + 离线兜底可用。

DoD：一句中文 query 进去，能在日志里看到 LLM **自主**发起 ≥1 次 tool call。

### M1：真实高德工具落地 + 追问工具 ✅

目标：让工具打真实 API，坐实「真实工具调用」叙事。

状态：已完成。`search_poi` / `check_weather` / `plan_route` +
`request_travel_info` / `request_preference_guide` 可用。

DoD：在真实 key 下，POI / 天气 / 路线三类工具各至少一次真实请求成功。

### M2：约束感知 planner-critic-reviser 封装为可控规划子图（差异化核心）✅

目标：保留并强化项目最有价值的差异化能力。

状态：已完成。`planning_subgraph.py` + `plan_and_critique` 工具；
critic 检查约束违规；可展示 plan 原始稿 vs revise 修正稿。

DoD：规划子图后 critic 违规项为 0 或可解释；能展示修正前后差异。

### M3：MCP Server 化 ⚠️ 部分完成

目标：工具与 agent 解耦，向「可被任意 MCP 客户端复用」的工程形态靠拢。

状态：`mcp_server.py` 已注册全部工具；**偏差**：agent 主路径仍用进程内
`lc_tools`，未默认接 `MultiServerMCPClient` → **并入 M9.5**。

DoD：agent 启动日志显示「从 MCP Server 拉取到 N 个工具」，端到端行为与 M2 一致。

### M4：三层记忆 + 会话持久化 ✅

目标：支持多轮、跨会话记忆与可恢复的会话状态。

状态：已完成。L1 压缩 / L2 artifact / L3 画像 + `SessionLifecycleManager`；
意图门控下 L3 不自动注入 destination/days。

DoD：重启后能按 session_id 恢复历史工具结果；长对话不超 token 上限。

### M5：分层编排 + 量化指标 ⚠️ 骨架完成，M9 升级

目标：把「agent 行为」变得可度量、可回归。

状态：有 `run_layered_turn` 确定性流水线 + `layer_metrics` 落盘；
**偏差**：真实 per-layer ReAct 演示与 Multi-Agent / 单 ReAct / fallback
对照评估仍需在 M9 补齐。

DoD：能跑出分层指标，并用一句带数字的话说明分层编排带来的稳定性收益。

### M6：Skills ✅

目标：用声明式 Markdown 技能扩展能力。

状态：已完成。`.storyline/skills` + 动态加载。

DoD：新增 SKILL.md 即可被 agent 识别调用。

### M7：FastAPI + Web 前端 + 高德地图 + A2UI 卡片 ✅

目标：可演示入口。

状态：已完成。WebSocket `/chat`、Markdown 渲染、IME 回车修复、地图渲染；
灰度演示闭环已补齐：前端支持用户编辑路书 POI 名称与开始时间，确认修改后
才允许导出高德路书草稿；支持按人工修改后的路线重新生成；不自动下单、不自动支付。

DoD：浏览器里完成多轮对话 → 工具链 → critic 修正行程 → 地图展示 →
用户微调 POI/时间 → 确认 → 导出高德路书草稿 / 重新生成。

### M8：评估报告 + 作品集包装 ✅

目标：把「可评估」从口号变成带数字的结论。

状态：已完成。`docs/EVALUATION.md`、`docs/INTERVIEW_PITCH.md`、`eval/cases.json`。

DoD：能用 3 句带数字的话讲清项目价值。

### M9：分层 Multi-Agent 主路径（当前重点）

目标：将 M5 骨架升级为可演示、可评估的 per-layer ReAct Multi-Agent 主架构。

状态：⚠️ 骨架完成，需补真实 Multi-Agent 演示与评估对照。

| 子项 | 内容 | 状态 |
|------|------|------|
| M9.1 | `LayeredTravelAgent`（checkpoint 回滚、`LayerTrace`/`LayerMetrics`） | ✅ 骨架完成 |
| M9.2 | `nodes/node_manager.py` 工具分层 + 场景标签 | ✅ 骨架完成 |
| M9.3 | `LayerValidator`（`strict_validation` 可配置） | ✅ 骨架完成 |
| M9.4 | `runtime.py` 在 `layered_enabled=true` 时走分层路径 | ✅ 骨架完成 |
| M9.5 | `tool_source.py` 可选 MCP；server 可选 `mcp.auto_start` | ✅ 骨架完成 |
| M9.6 | `planning.py` 地理聚类增强 | ✅ 已完成 |
| M9.7 | 真实 / mock per-layer ReAct trace 验证 | ⚠️ 待补 |
| M9.8 | Multi-Agent / 单 ReAct / deterministic fallback 对照评估 | ⚠️ 待补 |

配置策略：

- 本地快速离线演示与 CI：`layered_enabled=false`，保证可复现；
- 正式 Multi-Agent demo：`layered_enabled=true`；
- 只有 M9 验收完成后，才考虑把 Multi-Agent 改为代码默认值。

DoD：

- 有 LLM key 时可走 `LayeredTravelAgent`；
- trace 能看到五层执行或合理跳过；
- 每层工具调用受白名单限制；
- `layer_hit_rate`、`rollback_rate`、每层工具分布落盘；
- 能和单 ReAct / deterministic fallback 做 eval 对照；
- README 明确无 key fallback 不是 Multi-Agent。

### M10：Multi-Agent 评估增强

目标：证明分层 Multi-Agent 的收益与代价，而不是只在架构图里写 Multi-Agent。

改造点：

- 扩充 `eval/cases.json` 至 **30–50 条**，覆盖：
  - 缺目的地、缺天数、缺偏好、寒暄闲聊；
  - 亲子 / 老人 / 情侣 / 低预算 / 雨天 / 多兴趣 / 必去点 / 避开项；
  - 北京、上海、杭州、成都、西安等 seed 城市；
  - 非 seed 目的地（如「日本」）标 `expected_clarification` 或 `expect_no_itinerary`。
- 统一 case schema：`expected_interests`、`expected_budget_level`、`expected_tools`、
  `expect_no_itinerary` 等。
- 扩展 `scripts/eval_agent.py` 指标：
  - 目的地 / 天数 / 兴趣 / 预算 / 节奏抽取准确率；
  - 追问准确率、非旅行意图拦截率；
  - **必要工具覆盖率**（`search_poi`、`plan_and_critique` 等是否调用；
    不对 ReAct 强求固定顺序）；
  - critic pass rate、issue reduction、兴趣覆盖率、路线超限率。
- 增加 Multi-Agent / 单 ReAct / deterministic fallback 对照输出：
  - 工具覆盖率；
  - rollback rate；
  - critic pass rate；
  - issue reduction；
  - 平均工具步数 / layer 步数。
- `docs/EVALUATION.md` 由 `--write-report` 自动生成上述指标，保留闭环明细表。

涉及文件：`eval/cases.json`、`scripts/eval_agent.py`、`docs/EVALUATION.md`。

DoD：`eval_agent.py --write-report` 产出完整指标；`eval_layer_metrics.py`
能聚合 layer metrics；报告如实标注样本量、成本与局限。

### M11：质量检查与层级 trace 可视化

目标：让浏览器 demo 直观看到每层 Agent 做了什么，以及 critic/reviser 修正了什么。

改造点：

- 在 `render_itinerary` / `build_itinerary_cards` 中新增结构化 `quality` 卡片
  （后端确定性生成，不让 LLM 手写判断）：
  - 兴趣覆盖：已覆盖 / 未覆盖兴趣；
  - 单日负载：是否超过 relaxed / standard / intensive 上限；
  - 路线通勤：是否存在单段 / 单日超限；
  - 天气风险：雨天是否优先室内（补 `critic` 规则或 quality 构建器逻辑）；
  - critic 状态：passed / warnings、`revision_notes`。
- 前端 `web/static/app.js` 渲染 `quality` 卡片；可与现有 `critic` 卡合并或并存。
- 新增 `layer_trace` 卡片：
  - requirement / research / planning / risk / render 的执行状态；
  - 每层调用工具；
  - retry / rollback 原因；
  - 最终完成状态。
- 不破坏现有 `weather` / `summary` / `day` 卡片行为。

涉及文件：`src/travel_agent/critic.py`（可选雨天规则）、
`src/travel_agent/agent/render.py`、`web/static/app.js`、`web/static/style.css`。

DoD：浏览器 demo 在行程旁展示质量卡和层级 trace 卡；单测覆盖 passed /
warning / revision / rollback 四种状态。

### M12：餐厅推荐与预算估算

目标：补两个高收益产品功能，但不抢 M9/M10 的 Multi-Agent 与评估主叙事。

#### M12.1 餐厅推荐（轻量，不泛化）

改造点：

- 新增工具 `search_restaurant(ctx, city, cuisine, area, budget_level, max_results)`；
- 离线：从 seed POI 筛选 `category=food`；有高德 key 时走 POI keyword/category 搜索；
- 输出：name、category、tags、price_level、rating、lat/lng、source；
- 行程展示：新增 `restaurants` 补充卡片，**不修改主 `Itinerary` schema**；
- 同步注册：`toolkit.py`、`lc_tools.py`、`mcp_server.py`、`nodes/node_manager.py`；
- 新增 eval cases：美食偏好、菜系、低预算餐厅。

涉及文件：`src/travel_agent/agent/toolkit.py`、`providers.py`（如需）、
`agent/lc_tools.py`、`mcp_server.py`、`agent/render.py`、`eval/cases.json`、
`tests/test_restaurant.py`（新建）。

DoD：离线可返回杭州/成都等城市餐厅候选；eval 中美食类 case 可命中 `search_restaurant`。

#### M12.2 预算估算（确定性决策辅助）

改造点：

- 新增工具 `estimate_budget(ctx, days, companions, budget_level, city, hotel_required)`；
- 纯规则：住宿 / 餐饮 / 门票 / 市内交通 / 总计区间，**不调用 LLM**；
- 前端新增 `budget` 卡片；`recommend_candidates` 继续用 `budget_level` 排序，
  预算工具只做解释与决策辅助。
- 同步 MCP / lc_tools 注册。

涉及文件：`src/travel_agent/agent/toolkit.py`、`agent/render.py`、
`web/static/app.js`、`tests/test_budget.py`（新建）。

DoD：不同天数 / 档位 / 人数输出合理区间；前端可见预算卡。

#### M12.3 README 与面试材料

改造点：

- README 首页改为算法岗叙事：分层 Multi-Agent + deterministic subgraph +
  eval + quality / layer trace card；
  架构图突出 Multi-Agent 主路径。
- 更新 `docs/INTERVIEW_PITCH.md` 3 分钟讲稿：问题背景 → 为何不让 LLM 直接规划 →
  为什么需要分层 Multi-Agent → planner/critic/reviser → 工具与记忆 →
  eval 指标 → 后续方向。
- 明确 `TravelAgent-AI-main` 为参考来源，当前项目非功能照搬。

涉及文件：`README.md`、`docs/INTERVIEW_PITCH.md`、`config.toml.example`。

DoD：面试官 3 分钟内能听懂差异化；README 含关键 eval 数字与 2–3 条 demo query。

#### M12 公共接口（不破坏现有行为）

```text
search_restaurant(ctx, city=None, cuisine=None, area=None, budget_level=None, max_results=10)
estimate_budget(ctx, days=None, companions=None, budget_level=None, city=None, hotel_required=False)
```

新增卡片类型：`quality`、`budget`、`restaurants`。

不破坏：`search_poi`、`plan_and_critique`、`render_itinerary`、`render_map`。

#### M12 测试与验收

- 单元：餐厅筛选、budget 区间、quality 卡三态渲染；
- 集成：离线链路 多轮 → POI → 餐厅 → 预算 → plan_and_critique → render；
- 评估：`eval_agent.py --write-report` + `pytest -q` 全绿；
- Demo query：杭州雨天、美食预算、北京历史亲子、成都三天美食。

#### M12 明确后置（本阶段不做）

- 酒店推荐、`smart_plan_itinerary` 替换子图、PDF 导出、城市间交通建议、
  TTS/ASR/数字人。

## 评估方案

评估是本项目对外可信度的核心，分为六层：

- **抽取/理解层**：目的地/天数/兴趣等字段抽取准确率；
- **规划层（最重要）**：约束满足率、单日 POI 超载率、通勤超限率、must_visit
  命中率、雨天户外冲突率、类型多样性；
- **闭环层**：plan 原始稿 vs revise 修正稿的违规项下降幅度，量化 critic 价值；
- **Agent 层**：工具调用成功率、平均 ReAct 步数、任务完成率、追问触达率；
- **分层编排层**：`layer_hit_rate` / `rollback_rate` / p90 / 高频失败层；
- **多 Agent 层（M9/M10）**：per-layer 工具分布、rollback 原因 TopN、
  Multi-Agent / 单 ReAct / deterministic fallback 对照；
- **作品集层（M11/M12 新增）**：quality / layer trace 卡字段完整性、
  餐厅/预算工具离线命中率。

评估数据来源：

- `eval/cases.json` 目标 30–50 条人工构造样本，报告中**如实标注样本量与局限**；
- `scripts/eval_agent.py` 主评估路径（离线可复现，并在 M10 增加 Multi-Agent 对照模式）；
- `scripts/eval_layer_metrics.py` 从落盘 artifact 聚合分层指标（正式 demo 开启分层后使用）；
- 后续从真实/半真实 query 扩充。

### ChinaTravel 离线三层评测 + 真实 API 上线路线

本项目最终评测路线分三大阶段：

```text
阶段 1：离线评测（ChinaTravel sandbox）
  1. Mini-Dev Set：5-10 条黄金微型集，用于 Prompt / 多 Agent 协作流快速调试；
  2. Human-154：版本级回归，用于保存重要算法版本；
  3. Human-1000：最终离线验收，用于进入真实 API Shadow Testing 前的跑分报告。

阶段 2：Shadow Testing
  接高德 API live/replay，评估真实工具链可靠性，不替代 ChinaTravel 指标。

阶段 3：灰度上线
  接高德生产 API，前端保留 Human-in-the-loop，用户确认/微调后导出高德路书。
```

新增评测命令：

```bash
PYTHONPATH=src python scripts/eval_chinatravel.py --suite mini-dev --write-report
PYTHONPATH=src python scripts/eval_chinatravel.py --suite human154 --resume --write-report
PYTHONPATH=src python scripts/eval_chinatravel.py --suite human1000 --smoke-limit 50 --write-report
PYTHONPATH=src python scripts/eval_chinatravel.py --suite human1000 --resume --write-report

PYTHONPATH=src python scripts/eval_live_tools.py --provider amap --suite shadow-full --mode live --write-report
PYTHONPATH=src python scripts/eval_live_tools.py --provider amap --suite shadow-full --mode replay --write-report
PYTHONPATH=src python scripts/eval_live_tools.py --provider chinatravel --suite shadow-full --mode live --write-report
PYTHONPATH=src python scripts/eval_live_tools.py --provider chinatravel --suite shadow-full --mode replay --write-report
```

当前离线进展：

- Human-154 已达到进入 Human-1000 的回归门槛：FPR `55.19%`、EPR macro
  `87.01%`、Schema Pass Rate `100%`；
- Human-1000 自然语言 CSV 已放入 `data/eval/chinatravel/human1000.csv`，
  完整 1000 条已跑通：DR `100%`、Schema Pass Rate `100%`；
- 当前这份 Human-1000 CSV 只有 `uid,nature_language`，没有官方 `hard_logic_py`，
  因此只能报告 delivery/schema，不能计算官方 LPR / C-LPR / FPR；
- 若后续拿到带 `hard_logic_py` 的 Human-1000 官方评测文件，可直接覆盖同一路径，
  再用 `--resume --write-report` 复用已生成 prediction 计算正式 FPR。

产物：

- `data/eval/chinatravel/*/predictions/`：ChinaTravel schema 预测文件；
- `docs/EVALUATION_CHINATRAVEL.md`：离线 benchmark 报告；
- `data/eval/live_tools/snapshots/`：高德 live API / ChinaTravel sandbox response snapshot；
- `docs/EVALUATION_LIVE_TOOLS.md`：Shadow Testing 报告。

阶段边界：

- 离线 ChinaTravel 评测默认不调用高德 API，保证可复现；
- 高德 API 与 ChinaTravel sandbox 用于 Shadow Testing / 灰度上线，评估 API /
  provider success rate、fallback、empty result、duplicate POI、route unavailable、
  餐厅/酒店/预算工具成功率、latency p50/p95 等工程指标；
- 灰度上线保留用户确认与人工微调，不自动下单、不自动支付。

当前 Shadow Testing 进展：

- `scripts/eval_live_tools.py` 已升级为操作级评测 runner，覆盖 POI / weather /
  route / restaurant / hotel / budget 六类工具；
- 支持 `--suite shadow-dev` / `--suite shadow-full`，其中 `shadow-full` 当前为
  100 条 case：10 个城市 * 8 类常规场景 + 20 条异常/边界场景；
- 支持 `--mode live` 保存 snapshot，支持 `--mode replay` 离线复现同一批结果；
- 当前 ChinaTravel `shadow-full` replay 基线：600 个操作、100 个 case，API /
  provider success rate `100%`、operation success rate `100%`、primary success
  rate `100%`、primary usable rate `91.67%`、schema valid rate `100%`；
- 数据质量指标：timeout rate `0%`、empty result rate `6.67%`、
  expected empty pass rate `100%`、duplicate POI rate `0%`、
  route unavailable rate `10%`、unexpected route unavailable rate `0%`、
  route degraded rate `0%`、avg coordinate valid rate `100%`、
  restaurant / hotel / budget success rate `100% / 100% / 100%`、
  avg city contamination rate `0%`；
- 当前 ChinaTravel fallback rate `8.33%`，共 50 个操作触发 fallback，原因均为
  `primary_unusable`。常规城市/正常 query 已无非预期路线无结果；剩余重点是
  进一步提高宽泛 POI query 的类目匹配率；
- latency p50 / p95：`0.02ms / 0.25ms`；
- 报告产物：`docs/EVALUATION_LIVE_TOOLS.md`；
  snapshot 产物：`data/eval/live_tools/snapshots/{amap,chinatravel}/shadow-full/*.json`。

## 技术栈与依赖

- Multi-Agent：`langgraph`（每层 `create_react_agent`）、`langchain`、`langchain-core`；
- LLM：`langchain-openai`（Qwen / DeepSeek 等 OpenAI 兼容服务）；
- 工具/MCP：`fastmcp`、`langchain-mcp-adapters`（`MultiServerMCPClient`）；
- Web：`fastapi`、`uvicorn[standard]`；
- 基础：`httpx`、`pydantic` v2、`pydantic-settings`；
- 测试：`pytest`；
- 外部 API：高德 Web 服务 key + 高德 JS API key。

## 参考项目映射（借模式，不照搬功能）

参考 `../../TravelAgent-AI-main`：

| 参考文件 | 借鉴模式 | 本项目对应 |
|----------|----------|------------|
| `orchestration/layered_agent.py` | 五层 per-layer ReAct、层校验、回滚、`LayerMetrics` | M5 骨架 ✅ / M9 对齐 |
| `nodes/node_manager.py` | 工具分层 + 场景标签筛选 | M9.2 |
| `agent.py` | `build_agent`、MCP 拉工具、三层记忆、`LayeredTravelAgent` 开关 | M9.4 / M9.5 |
| `mcp/register_tools.py` | 薄包装 + artifact + `isError` 约定 | M3 `mcp_server.py` ✅ |
| `storage/*` | 三层记忆与会话持久化 | M4 ✅ |
| `agent_fastapi.py` + `web/` | WebSocket、A2UI、地图 | M7 ✅ |
| `smart_plan_itinerary.py` | 地理聚类 + 最近邻排序 | M9.6 增强 `planning.py`（不替换子图） |
| `nodes/core_nodes/search_restaurant.py` | 餐厅 POI 筛选 | M12.1 |
| `nodes/core_nodes/estimate_budget.py` | 确定性预算区间 | M12.2 |
| `prompts/tasks/*.md` | 每层独立 task prompt | M9 可选 |
| `scripts/eval_layer_metrics.py` | 分层指标聚合 | M8/M9.7 |

注意：TTS/ASR/数字人不在范围内；`smart_plan_itinerary` 仅借鉴聚类算法，
不替换 `plan_and_critique` 差异化子图。

## 风险与未知项

工程/集成风险：

- **延迟**：五层 × 多轮 ReAct 响应比单 Agent 慢 → 缓解：`qwen-turbo`、
  层内 `recursion_limit`、requirement 层规则预填减少 LLM 步数；
- **成本**：正式 Multi-Agent demo 会放大 token → 可选 `orchestration.fast_requirement`
  （requirement 层走规则，research 起用 ReAct）；
- ReAct 死循环：必须设 `recursion_limit` 并对工具失败做有限重试；
- MCP 连接稳定性：端口冲突、超时需兜底回退 `lc_tools`；
- 高德双 key：Web 服务 key 与 JS API key 不同，需注意配额与合规。

产品/数据风险：

- seed 仅少数城市，无真实 key 时演示范围有限；
- 会话并发与持久化需 `SessionLifecycleManager` 正确隔离。

叙事风险：

- 无 LLM key 时不要声称「五层 Multi-Agent」——应说明是确定性 fallback；
- 不要把 M5 确定性流水线对外说成 Multi-Agent（M9 前的真实状态）；
- 避免把规划核心做成黑盒大工具。

## 各里程碑完成标准（DoD）汇总

- M0–M8：见各节（✅ 已完成或 ⚠️ 部分完成）；
- M9：per-layer ReAct 主路径可演示 + 指标落盘 + pytest（⚠️ 骨架已完成，证据待补）；
- M10：`eval/cases.json` ≥30 条，`eval_agent.py --write-report` 产出
  Multi-Agent / 单 ReAct / fallback 对照指标；
- M11：浏览器 demo 展示质量卡 + layer trace 卡 + 地图 + critic 修正行程；
- M12：补齐预算卡 + 餐厅卡，README / `INTERVIEW_PITCH.md` 叙事统一为
  「分层 Multi-Agent + 确定性子图 + eval」。
- 总验收：算法岗 3 分钟讲清「LLM 旅行规划不可控 → 分层 Multi-Agent 编排 →
  确定性子图控质量 → 量化 eval 证明收益」；无 key 时明确是 deterministic fallback。
