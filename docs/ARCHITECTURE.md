# 架构设计

> ⚠️ **状态更新（Step 4）**：本文档主体是项目初期的工作流设计记录。
> 现行生产链路已收敛为多 Agent 架构：`run_production_turn` 固定调用
> `MultiAgentEngine`（Main Orchestrator + 四个领域调研 Subagent + Planner
> Subagent + Reviewer +
> Renderer Gate，即 Multi-Agent Full = V3）；V0–V3 消融见
> [ABLATION_V0_V3.md](ABLATION_V0_V3.md)，可观测性统一为 agent_trace。

## 现行生产多 Agent 链路

生产请求的主调用链是：

```text
run_production_turn()        生产对外入口，固定使用 PRODUCTION_CONFIG
  -> run_architecture_turn() 准备会话与用户记忆
  -> run_turn_lifecycle()    单轮生命周期：预检、预算、降级和结果转换
  -> prepare_turn()          识别任务、更新画像、构造 turn_inputs
  -> MultiAgentEngine.run_turn()
  -> MultiAgentEngine._run_turn()
  -> run_dynamic_base()      V2/V3 动态路由与领域执行
```

`run_turn_lifecycle()` 是每轮请求的公共上层入口，
`MultiAgentEngine.run_turn()` 是多 Agent Engine 入口，`run_dynamic_base()`
才是生产 V3 动态路由的具体执行入口。如果预检需要追问，或
`settings.llm.enabled == false`，请求不会进入动态路由。

### 轮次结构化输入

`prepare_turn()` 使用“确定性规则优先、LLM 可选增强、失败回退规则”的方式
分析当前消息。结果会更新 `ctx.profile`，并构造所有 Subagent
共享的 `turn_inputs`（在派工时作为 `base_inputs`）：

```python
{
    "artifact_ids": artifact_ids,
    "plan_artifact_id": existing_plan_id,
    "revision_directives": dict(analysis.revision_directives),
    "profile": toolkit._profile_brief(ctx.profile),
}
```

- `profile`：当前结构化旅行画像，可包含历史轮次累积的有效偏好；
- `revision_directives`：本轮对旧行程的确定性修改指令；
- `artifact_ids`：允许本轮使用的产物引用，不是产物正文摘要；
- `plan_artifact_id`：行程修改场景中的旧行程 ID。

每次派工的专属 `inputs` 会覆盖同名的 `base_inputs` 普通字段；
`artifact_ids` 例外，两边会合并去重。

### 动态 Router 和 wave 预算

生产配置是 `mode="orchestrated"`、`dispatch="dynamic"`，因此进入
`run_dynamic_base()`。一个 wave 是一轮完整的“检查证据 → 调用 Router →
派工 → 收集结果”，不只是一次检查。

- `routing_max_waves`：最多允许的执行 wave 数，代码将其限制在 1–3；
- `routing_max_calls`：Router 模型的最多调用数，代码将其限制在 1–3；
- `routing_max_dispatches`：所有 wave 累计最多派发的领域任务数，默认 6；
- `routing_wave1_max_tasks`：首个 wave 最多的并行任务数，限制在 1–4；
- 后续 wave 每轮最多派发 2 个补充任务。

当前实现是“每个 wave 恰好调用一次 Router”，所以可执行的路由轮数
实际同时受 `min(routing_max_waves, routing_max_calls)` 约束。两个配置分别表达
编排深度与模型调用成本，但在当前 one-call-per-wave 实现中作用接近。

### Subagent 分层与职责

`SUBAGENT_REGISTRY` 一共注册 5 个执行型 Subagent，但动态 Router 只能从
4 个领域调研 Subagent 中按需选择：

| Subagent | 职责 | 工具白名单 | 调用方式 |
|---|---|---|---|
| `attraction` | 景点搜索、兴趣匹配、天气适配、适老与无障碍提示 | `search_poi`, `check_weather` | Router 动态派发 |
| `hotel` | 酒店候选、住宿区域、价格档位与商圈比较 | `search_hotel` | Router 动态派发 |
| `restaurant` | 餐厅搜索、口味与饮食限制、人均预算 | `search_restaurant`, `estimate_budget` | Router 动态派发 |
| `transport` | 市内/城际路线、换乘、步行、耗时和返程截止风险 | `search_poi`, `plan_route`, `estimate_budget` | Router 动态派发 |
| `planner` | 读取明确的领域 Artifact，聚合约束与证据，生成完整 `TravelPlan` | `build_constraints`, `recommend_candidates`, `plan_and_critique` | Engine 在证据阶段后确定性调用 |

Router 通过 Registry 中的 `description` 知道每个领域 Subagent 的职责和工具边界，
再根据用户请求与已有证据动态生成每次任务的 `instruction`、`objective`
和 `depends_on`。Router 不能派发 `planner`、Reviewer 或 repair；Planner 由 Engine
在领域证据就绪后调用，避免无证据规划或跳过规划阶段。

Reviewer 不在 `SUBAGENT_REGISTRY` 中，也不经过 `SubagentRunner`。它是 Engine
在 Planner 完成后直接发起的一次无工具 LLM 调用；生产 V3 最多允许一次
“定向重派领域 Subagent → Planner 重规划 → Reviewer 复审”修复周期。

因此数量口径是：

- Router 可动态派发的 Subagent：4 个；
- Registry 中的执行型 Subagent：5 个（上述 4 个 + `planner`）；
- 整个多 Agent 链路的独立角色：Main Router + 5 个执行型 Subagent +
  Reviewer；Renderer Gate 是确定性交付门禁，不是 Subagent。

## 系统形态

```text
Frontend
  -> FastAPI backend
  -> Agent orchestrator
  -> TravelProfileExtractor
  -> DialogueSession
  -> Travel state
  -> Tool layer
  -> RAG / Weather context
  -> Planner
  -> Critic
  -> Reviser
  -> Response generator
```

中文解释：

- `Frontend`：用户交互界面；
- `FastAPI backend`：后端服务；
- `Agent orchestrator`：Agent 编排器，决定下一步做什么；
- `TravelProfileExtractor`：需求抽取器，可由规则或 LLM structured output 实现；
- `DialogueSession`：会话管理器，维护 profile、message history 和 last_result；
- `Travel state`：结构化出行状态；
- `Tool layer`：外部工具和 API 封装；
- `RAG / Weather context`：目的地攻略检索和天气上下文；
- `Planner`：行程规划器；
- `Critic`：约束检查器；
- `Reviser`：根据 critic 结果自动修正行程；
- `Response generator`：最终回复生成器。

## Agent 节点

第一版可以先用普通 Python 函数实现。等工作流稳定后，再升级为 LangGraph。

```text
extract_profile
  -> clarify_or_continue
  -> retrieve_candidates
  -> score_candidates
  -> build_itinerary
  -> critique_itinerary
  -> revise_if_needed
  -> respond
```

节点说明：

- `extract_profile`：从用户输入中抽取结构化出行画像；
- `clarify_or_continue`：判断是否需要追问；
- `retrieve_candidates`：调用工具检索 POI 候选；
- `score_candidates`：对候选进行偏好感知打分；
- `build_itinerary`：生成行程草案；
- `critique_itinerary`：检查行程是否违反约束；
- `revise_if_needed`：必要时修正行程；
- `respond`：生成最终面向用户的回复。

## 当前 MVP 实现

当前已落地的规则版 workflow 位于：

```text
src/travel_agent/workflow.py
```

它包含：

- `TravelProfileExtractor`：抽取结构化 `TravelProfile` 的统一接口；
- `RuleBasedTravelProfileExtractor`：用关键词和简单规则抽取 `TravelProfile`；
- `FakeLLMTravelProfileExtractor`：测试用 fake provider，模拟 LLM structured output；
- `OpenAICompatibleTravelProfileExtractor`：通过 OpenAI-compatible chat completions 接真实模型；
- `DialogueSession`：提供 `send()` 和 `reset()`，封装多轮状态合并；
- `run_mvp_workflow`：串联 profile 抽取、多路 POI 召回、候选打分和行程生成；
- `merge_profile`：合并多轮对话中的旧画像和新画像；
- `critique_itinerary`：对生成行程做约束检查，返回结构化问题列表；
- `revise_itinerary`：根据 critic issue 尝试自动修正行程；
- `render_markdown_response`：把结构化结果渲染成中文 Markdown 回复；
- 缺少目的地或天数时，返回澄清问题，而不是直接编造行程。

这个版本先保证链路可测试。后续升级时，会把规则抽取替换成 LLM structured output，把简单 planner 替换成更强的约束规划器。

## 当前召回策略

当前 POI 候选采用两路召回后合并去重：

```text
偏好召回：根据用户兴趣 tags 找高匹配 POI
城市兜底召回：召回同城全部 seed POI，供 planner / reviser 使用
```

这样可以避免只召回“美食”时，reviser 无法补上“故宫”这类 `must_visit` 约束点。

## Stage 6 Context Tools

当前阶段 6 的第一版实现：

```text
src/travel_agent/rag.py
src/travel_agent/weather.py
data/guides/*.md
```

能力：

- 从本地 Markdown 目的地攻略中检索相关片段；
- 为 workflow 提供 `knowledge_chunks`；
- 使用 mock weather 为 workflow 提供 `weather`；
- 在中文回复里展示“出行上下文”。

后续升级方向：

- 本地 keyword overlap -> BM25 / embedding retrieval；
- mock weather -> Open-Meteo / 高德天气；
- 本地 POI seed -> OpenTripMap / OSM / 高德 POI。

## Eval

当前评估入口：

```text
scripts/run_eval.py
```

评估用例：

```text
eval/cases.json
```

指标包括城市识别、天数识别、行程生成率、澄清准确率、critic 通过率和兴趣覆盖率。

## Critic 结果

`critic` 返回 `CriticResult`：

```text
passed: 是否通过检查
issues: CriticIssue 列表
```

`CriticIssue` 包含：

```text
code: 问题编码
message: 中文问题说明
severity: info / warning / error
```

## Reviser 结果

`WorkflowResult` 中包含：

```text
revised: 是否发生过自动修正
revision_notes: 修正说明
critic_result: 修正后的二次 critic 结果
```

## Response Generator

当前回复生成器位于：

```text
src/travel_agent/response.py
```

它不调用 LLM，而是将结构化 `WorkflowResult` 渲染为中文 Markdown。这样做的好处是：

- 输出稳定，方便测试；
- 先把产品展示链路跑通；
- 后续可以替换为 LLM 生成自然语言回复，但仍保留结构化结果作为事实来源。

## Dialogue Session

多轮会话位于：

```text
src/travel_agent/dialogue.py
```

当前维护：

```text
profile: 当前合并后的 TravelProfile
last_result: 最近一轮 WorkflowResult
messages: user / assistant 消息历史
turns: 每轮输入、输出和结构化结果
```

调用方式：

```python
session = DialogueSession()
session.send("我想轻松一点，喜欢自然和美食")
session.send("杭州三天")
```

## LLM Extractor

LLM 结构化抽取位于：

```text
src/travel_agent/llm_extractor.py
```

设计原则：

- workflow 只依赖 `TravelProfileExtractor` 接口；
- 默认 provider 是规则抽取，保证本地可运行；
- fake provider 用于测试 LLM structured output 的工程链路；
- 真实 LLM provider 后续只需要把模型 JSON 输出转换成 `TravelProfile`。

## 模块边界

```text
travel_agent.schemas
  定义 profile、POI、itinerary、critique result 等数据模型。

travel_agent.tools
  定义工具接口和 provider 实现。

travel_agent.recommendation
  候选打分和 rerank。它是 planning support module，不是项目主线。

travel_agent.planning
  行程构建和约束处理。

travel_agent.agent
  工作流编排和状态转移。

travel_agent.evaluation
  测试 case、指标和评估脚本。
```

## 推荐叙事

建议这样讲：

```text
Agent 调用偏好感知候选工具，获得用于行程规划的 planning candidates。
```

避免这样讲：

```text
这个项目主要是一个 POI 推荐系统。
```
