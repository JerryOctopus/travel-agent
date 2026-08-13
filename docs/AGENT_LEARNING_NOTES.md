# Agent 架构学习笔记

## 当前架构总览

本项目当前不是“一个 ReAct Agent 加几个评测脚本”，而是生产运行时、统一多 Agent
引擎、状态存储和评测 Harness 四部分共同组成的系统：

```text
WebSocket / Python 调用
          |
          v
run_production_turn                 生产固定入口
          |
          v
run_architecture_turn               共用外层生命周期与 L3 持久化
          |
          v
run_turn_lifecycle                  意图/任务分析、槽位门禁、预算、失败降级
          |
          v
MultiAgentEngine(PRODUCTION_CONFIG) Production = Full = V3
          |
          +--> Main Orchestrator 动态派工
          +--> 5 个执行型 Subagent
          +--> Semantic Reviewer（仅完整规划/行程修改）
          +--> Renderer Artifact Gate
          |
          v
AgentReply + ArtifactStore + agent_trace + turn_metrics
```

生产代码不能通过配置切换 V0–V3。生产入口始终使用
`PRODUCTION_CONFIG`；V0–V3 只允许 Harness/消融评测通过
`orchestration.variants.run_variant_turn()` 显式选择。

四个能力版本共用同一个 `MultiAgentEngine`：

| 版本 | 编排方式 | Reviewer | 用途 |
| --- | --- | --- | --- |
| V0 | 单 ReAct Agent | 无 | 单 Agent 基线 |
| V1 | `task_type` 确定性派工 | 无 | 固定多 Agent 基线 |
| V2 | Main Orchestrator 动态派工 | 无 | 验证动态路由收益 |
| V3 | 动态派工 | 有，最多一次修复 | 生产 Full |

V3 的执行型 Subagent 固定为 `attraction`、`hotel`、`restaurant`、`transport`
和 `planner`。每个 Subagent 都有独立提示词、工具白名单、最大步骤、工具调用上限和
超时；Reviewer 不在 Subagent Registry 中，而是由 Engine 在 Planner 之后直接调用。

## Harness 是什么

`harness` 可以理解成围绕被运行对象搭起来的一层控制壳。

它不是业务逻辑本身，而是负责让系统可控地跑起来、可观测、可评估、可复现。

如果被运行对象是一个 Agent，那么 `Agent harness` 通常包括：

```text
执行环境
工具编排
上下文管理
循环控制
验证测试
错误处理
日志与评估
```

## Harness 包含哪些部分

### 执行环境

控制 Agent 在哪里运行。

例子：

```text
本地环境
沙箱环境
CI 环境
评测环境
线上灰度环境
```

执行环境会决定：

```text
是否能联网
是否有 API key
是否使用真实工具
是否使用 mock / replay 数据
```

### 输入驱动

负责给 Agent 喂任务。

输入来源可以是：

```text
用户请求
测试 case
benchmark 数据集
离线评测样本
线上 shadow traffic
```

比如旅行 Agent 的 harness 可以读取一批 case：

```text
帮我规划杭州三天，喜欢自然和美食
北京两天，带父母，不要太累
上海周末游，预算中等
```

然后逐条调用 Agent。

### 上下文管理

负责给 Agent 注入必要上下文。

常见上下文：

```text
系统提示词
历史对话
用户画像
memory
tool artifacts
RAG 检索结果
环境配置
```

在本项目里，用户长期画像由 `UserMemoryService` 保存，当前会话状态由
`SessionContext` 管理。Harness 会按 case 创建或复用 session，并在每轮后采集画像、
artifact 和 L3 快照。

### 工具编排

负责决定 Agent 可以用哪些工具，以及工具如何接入。

工具可能是真实调用，也可能是 mock、record、replay。

旅行项目里的工具包括：

```text
search_poi
check_weather
plan_route
recommend_candidates
plan_and_critique
render_map
```

工具的业务实现主要集中在：

```text
src/travel_agent/agent/toolkit.py
```

LangChain 工具适配和工具来源路由分别位于：

```text
src/travel_agent/agent/lc_tools.py
src/travel_agent/agent/tool_source.py
```

如果通过 MCP 暴露工具，则 `src/travel_agent/mcp_server.py` 是工具协议层，不是
Agent 的模型调用入口。

### 循环控制

控制 Agent 最多运行多少轮、什么时候停止。

常见控制项：

```text
最大工具调用轮数
最大递归深度
请求超时
是否允许重试
什么时候结束任务
什么时候进入降级路径
```

V0 ReAct Agent 和各 Subagent 都有循环上限。V0 的递归上限来自：

```python
config={"recursion_limit": settings.agent.recursion_limit}
```

WebSocket 请求也有超时保护：

```python
timeout=settings.agent.request_timeout_seconds
```

生产 V3 还通过 `TurnMeter` 统一限制每轮 token、模型调用数和工具调用数；Subagent
自身另有 `max_steps`、`max_tool_calls` 和 `timeout_seconds`。预算耗尽会返回
`incomplete`，不会继续发起新调用。

#### 为什么动态派工需要 `DispatchLedger`

V2/V3 的 Main Orchestrator 由模型动态决定调用哪些领域 Subagent。为了避免模型重复派工、
某个领域独占预算，或者在信息收益已经很低时仍持续调用，Engine 会为每个回合创建一个
`DispatchLedger`：

```python
ledger = DispatchLedger(
    max_total=max_dispatches,
    per_agent_limits={
        "attraction": 2,
        "hotel": 2,
        "restaurant": 2,
        "transport": 2,
        "planner": 0,
    },
)
```

这里同时设置了两层硬限制：

```text
max_total
    本回合动态领域派工的总尝试次数上限。默认 max_dispatches 为 6；即使各领域
    单独上限相加为 8，总尝试次数仍不能超过 6。

per_agent_limits
    每类领域 Subagent 的派工上限。景点、酒店、餐厅和交通各最多尝试 2 次，
    防止某一领域耗尽本回合全部预算。
```

`authorize()` 在真正执行 Subagent 前依次检查：Planner 是否已使账本进入终止状态、目标
是否重复、总预算是否耗尽、该领域预算是否耗尽。通过检查后会立即增加 `attempts` 和该
领域的 `counts`，因此这里统计的是“已授权的派工尝试”，即使后续执行失败也会占用预算；
完全相同的标准化目标会被提前拒绝，不消耗新额度。

限制派工次数主要有四个目的：

```text
终止保证    防止 Router 因信息不足或失败恢复而无限循环
成本控制    限制额外模型调用、工具调用、token 和外部 API 成本
延迟控制    为后续 Planner、Reviewer 和渲染阶段保留时间
结果多样性  避免单一领域反复搜索相似内容并挤占其他领域
```

这里的 `"planner": 0` 不表示 Planner 被禁用。它表示动态 Router 不能通过这一份
`DispatchLedger` 派发 Planner：Router 只负责收集领域证据，证据收集结束后由 Engine
在独立的规划阶段直接调用 Planner，并为它预留单独的 deadline。Reviewer 触发的修复
Planner 也由 Engine 直接调用，不占这份动态领域派工额度。这样能保证 Planner 只在输入
证据准备好后运行，并避免 Router 提前或重复调度 Planner。

`DispatchLedger` 只是动态派工的其中一道边界。Engine 还分别限制 Router 的调用次数、
派工波次数和首波任务数，并结合 deadline 判断是否允许恢复波；这些限制与 `TurnMeter`、
Subagent 自身的步骤/工具/超时预算共同生效，而不是互相替代。

#### 当前有哪些 Subagent

`SUBAGENT_REGISTRY` 当前只注册 5 个执行型 Subagent：

| Subagent | 职责 | 工具白名单 | `max_steps` | `max_tool_calls` | 定义超时 |
| --- | --- | --- | ---: | ---: | ---: |
| `attraction` | 景点搜索、兴趣匹配、天气适配、适老与无障碍提示 | `search_poi`, `check_weather` | 8 | 4 | 90 秒 |
| `hotel` | 酒店候选、住宿区域、价格和商圈比较 | `search_hotel` | 6 | 3 | 90 秒 |
| `restaurant` | 餐厅搜索、口味与饮食限制、人均预算 | `search_restaurant`, `estimate_budget` | 6 | 4 | 90 秒 |
| `transport` | 市内与城际路线、换乘、步行、耗时和返程风险 | `search_poi`, `plan_route`, `estimate_budget` | 10 | 6 | 90 秒 |
| `planner` | 聚合领域 Artifact 和约束，生成并检查完整 TravelPlan | `build_constraints`, `recommend_candidates`, `plan_and_critique` | 12 | 4 | 120 秒 |

这里的“定义超时”是 Registry 中的静态预算；实际运行还会受到 Engine 根据本回合剩余
deadline 计算出的有效超时约束。`max_steps` 是 LangGraph 的图步骤硬上限，不等于工具
调用次数；一次模型节点、工具节点或状态转换都可能消耗图步骤。

工具调用还有比 `max_tool_calls` 更细的单工具硬限制：

| Subagent | 单工具调用上限 | 受总上限约束后的效果 |
| --- | --- | --- |
| `attraction` | `search_poi: 1`, `check_weather: 1` | 实际最多调用 2 次工具 |
| `hotel` | `search_hotel: 2` | 实际最多调用 2 次工具 |
| `restaurant` | `search_restaurant: 1`, `estimate_budget: 1` | 实际最多调用 2 次工具 |
| `transport` | `search_poi: 2`, `estimate_budget: 1`；`plan_route` 无单独上限 | 所有工具合计最多 6 次 |
| `planner` | 三个白名单工具各 1 次 | 实际最多调用 3 次工具 |

每次 Subagent 执行都会创建新的共享计数器，所以这些上限按“单次 Subagent 任务”计算，
不是跨整个进程累计。工具必须先通过白名单过滤，再同时满足总工具上限和单工具上限；
超额的下一次调用会在工具真正执行前被拒绝。如果此前已经产生有效 Artifact，结果可以
保留为 `completed` 并附带预算告警，否则返回 `budget_exhausted`。

#### 各 Subagent 的系统 Prompt

以下内容来自 `src/travel_agent/orchestration/multi_agent/registry.py`，是当前传给各执行型
Subagent 的完整系统 Prompt。

##### `attraction`

```text
你是 attraction 领域 Subagent，只负责景点调研。
职责：景点搜索、兴趣匹配、天气适配、适老性与无障碍提示。
可用工具仅限：search_poi、check_weather。
纪律：
1. search_poi 成功一次即可，不要重复检索同一城市；
2. 涉及户外安排时查询一次天气，并给出雨天适配建议；
3. 有老人/儿童同行信号时，标注适老性与无障碍注意事项；
4. 结束时输出结构化结论：候选景点（含 poi_id）、匹配理由、天气影响、未解决事项。
禁止：调用白名单以外的工具、编造不存在的景点。
```

##### `hotel`

```text
你是 hotel 领域 Subagent，只负责住宿调研。
职责：酒店候选、住宿区域比较、价格档位与商圈适配。
可用工具仅限：search_hotel。
纪律：
1. search_hotel 成功一次即可，可按区域参数补充检索一次，不要反复刷候选；
2. 输出结构化结论：候选酒店/区域（含 poi_id）、价格档位、与行程动线的匹配度、未解决事项；
3. 用户未明确预算时按画像 budget_level 推断，不要编造价格。
禁止：调用白名单以外的工具。
```

##### `restaurant`

```text
你是 restaurant 领域 Subagent，只负责餐饮调研。
职责：餐厅搜索、口味与饮食限制匹配、人均预算评估。
可用工具仅限：search_restaurant、estimate_budget。
纪律：
1. search_restaurant 成功一次即可，不要重复检索同一城市；
2. 注意画像中的 food_preference / avoid（忌口与饮食限制）并显式说明匹配情况；
3. 需要预算口径时可调用一次 estimate_budget；
4. 输出结构化结论：候选餐厅（含 poi_id）、口味匹配、人均档位、未解决事项。
禁止：调用白名单以外的工具。
```

##### `transport`

```text
你是 transport 领域 Subagent，只负责交通调研。
职责：市内与城际路线、换乘与步行、耗时估算、返程时间提醒。
可用工具仅限：search_poi、plan_route、estimate_budget。
纪律：
1. plan_route 的 poi_id 必须来自任务输入；若只有地点名称，先用 search_poi 分别解析起终点，禁止把名称冒充 poi_id；
2. 多段路线逐段估算，给出总耗时与换乘建议；
3. 输出结构化结论：路线段、耗时、方式、返程/截止时间风险提示、未解决事项。
禁止：调用白名单以外的工具、编造班次时刻。
```

##### `planner`

```text
你是 planner Subagent，唯一负责生成完整 TravelPlan 的执行者。
职责：读取明确的领域 Artifact，聚合约束与证据，生成完整行程并执行 plan_and_critique。
可用工具仅限：build_constraints、recommend_candidates、plan_and_critique。
纪律：
1. 只使用任务指令中给出的 artifact_id 读取领域结果，禁止依赖“最新结果”猜测；
2. 不得重新发起大规模外部搜索（你没有搜索工具）；
3. 标准顺序：build_constraints → recommend_candidates → plan_and_critique；
4. plan_and_critique 成功后输出 artifact_id、critic 是否通过、遗留告警与未解决事项。
禁止：调用白名单以外的工具、自行编造行程内容。
```

Reviewer 不在 `SUBAGENT_REGISTRY` 中，也不通过 `SubagentRunner` 执行。它是在 Planner
完成后由 Engine 直接发起的无工具模型调用，因此不应被算作第 6 个 Subagent。

### 验证测试

负责检查 Agent 输出是否符合预期。

常见验证：

```text
输出是否符合 schema
是否识别出正确目的地
是否识别出正确天数
是否生成行程
是否调用了必要工具
critic 是否通过
是否出现重复 POI
是否出现不可执行路线
```

在本项目里有两层质量控制：

```text
规划内层：planner 调用 plan_and_critique，执行 critic/reviser
编排外层：V3 Semantic Reviewer 检查完整计划，最多触发一次定向修复
```

Renderer Artifact Gate 只渲染通过交付条件的计划。Harness 的 validators 再独立检查
城市、天数、追问、工具覆盖、参数、记忆、硬约束、grounding、可执行性、授权边界和
架构策略，避免用 Agent 自己的结论给自己打分。

### 错误处理

负责处理运行中的异常。

常见错误：

```text
模型调用失败
API key 无效
额度不足
工具调用失败
JSON 解析失败
超时
schema 不合法
外部服务返回空结果
```

本项目里有几个保守兜底策略：

```text
LLM 意图分类失败 -> 返回 ambiguous
生产多 Agent 调用失败 -> 统一降级到离线 fallback，并记录 raw_failure/fallback_triggered
单轮预算耗尽 -> 返回 incomplete，不再调用模型或工具
JSON 用户画像文件不存在 -> 返回空 TravelProfile；PostgreSQL 配置/连接失败 -> 启动失败
缺少目的地或天数 -> 追问用户
```

必填槽位不是全局固定为“目的地 + 天数”，而是由 `TaskType` 对应的
`REQUIRED_SLOTS` 决定；例如行程修改还必须先找到当前 session 的既有 itinerary。

### 日志与评估

负责记录运行过程，并生成指标。

常见记录项：

```text
tool trace
输入输出
token 用量
耗时
成功率
失败原因
critic issue
fallback 比例
```

当前统一 Harness 的主体位于：

```text
src/travel_agent/harness/cases.py          case schema 与 JSON/JSONL 加载
src/travel_agent/harness/environments.py   offline/real_agent 环境策略
src/travel_agent/harness/runner.py         驱动生产入口并采集每轮结果
src/travel_agent/harness/validators.py     case 级验证
src/travel_agent/harness/reporting.py      聚合指标、报告与 release gates
src/travel_agent/harness/cli.py            suite 统一命令入口
```

`chinatravel.py`、`live_tools.py`、`product.py` 是专项 suite 适配器。`scripts/eval_*.py`
仍承担兼容入口或专项实验，但不再是 Harness 架构的唯一主体；其中 `agent-nl` 和
`agent-real` 当前仍由统一 CLI 调用 `scripts/eval_agent.py` 的遗留实现。

它们不是 Agent 的核心业务逻辑，而是负责喂 case、运行 Agent、收集结果、打分和生成报告。

## Harness 和 Agent 核心的区别

Agent 核心负责解决任务：

```text
理解用户需求
决定是否调用工具
调用工具
整合结果
生成回复
```

Harness 负责控制和评估这个过程：

```text
准备输入
准备环境
限制运行边界
记录过程
验证输出
处理失败
统计指标
```

简单说：

```text
Agent = 做事的主体
Harness = 让 Agent 可控运行的外部框架
```

## 当前 Harness 执行链

默认 `AgentHarness` 不选择 variant，而是直接运行生产固定链路：

```text
HarnessCase
   |
   v
AgentHarness.run_case
   |
   +--> build_session / 注入 snapshot_date / 可选故障 Provider
   +--> 为每个 session 维护独立 history
   |
   v
run_production_turn                  默认固定 V3
   |                                显式 variant 时改走 run_variant_turn
   v
HarnessTurnResult
   +--> reply / clarification / status
   +--> tool_calls / model_calls / agent_trace
   +--> artifacts / profile / memory_snapshot
   +--> duration / turn_metrics / raw_failure / fallback
   |
   v
validate_case_result
   |
   v
HarnessCaseResult -> suite 聚合 -> release gates / Markdown / JSON
```

核心对象的职责是：

| 对象 | 职责 |
| --- | --- |
| `HarnessCase` | 单轮/多轮输入、期望结果、工具约束、故障注入、gold 与架构策略 |
| `HarnessEnvironment` | `offline`/`real_agent`、POI 数据、持久化、用户和消融 variant |
| `AgentHarness` | 驱动回合、隔离 session/history、捕获 trace、artifact、异常和耗时 |
| `HarnessTurnResult` | 保存单轮可观测数据 |
| `HarnessCaseResult` | 保存 case 结果、指标和错误 |
| `HarnessSuiteResult` | 保存 suite/benchmark 聚合结果与产物位置 |

`offline` 环境会强制使用 rule LLM、关闭 AMap，并在可用时关闭 MCP 自动启动，保证
测试可复现；`real_agent` 保留真实配置。Harness 默认 `persist=False`，避免评测污染
生产 session 数据。

## Suite 与统一入口

统一 CLI：

```text
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli \
  --suite agent-product --env real_agent --product-split dev --json
```

当前 suite：

| suite | 作用 |
| --- | --- |
| `agent-nl` | 离线自然语言回归，当前走 legacy adapter |
| `agent-real` | 真实 LLM、多 Agent trace 与工具白名单验证 |
| `agent-product` | production_v1 数据、严格任务成功和固定模型基线 |
| `chinatravel-*` | ChinaTravel mini/human benchmark |
| `live-tools-shadow` | AMap/ChinaTravel 工具 replay 或 live shadow |
| `all` | 按环境组合运行多套 suite |

case 支持项目旧 JSON schema，也支持 production_v1 JSONL schema。多轮 case 可以为每轮
指定 `session_ids` 和 `user_ids`，用来验证同 session 连续状态以及跨 session 的 L3
记忆。`gold` 只用于结束后的评估，不会注入 Agent。

## 代码导航

| 领域 | 入口 |
| --- | --- |
| 生产一轮调用 | `src/travel_agent/agent/runtime.py::run_production_turn` |
| 共用回合生命周期 | `src/travel_agent/agent/turn_lifecycle.py` |
| 多 Agent 引擎 | `src/travel_agent/orchestration/multi_agent/engine.py` |
| Subagent 注册表 | `src/travel_agent/orchestration/multi_agent/registry.py` |
| V0–V3 评测入口 | `src/travel_agent/orchestration/variants/registry.py` |
| WebSocket 服务 | `src/travel_agent/server.py` |
| 工具业务实现 | `src/travel_agent/agent/toolkit.py` |
| MCP 服务 | `src/travel_agent/mcp_server.py` |
| 当前 session/artifact | `src/travel_agent/agent/session.py` |
| L3 用户记忆 | `src/travel_agent/storage/user_memory.py` |
| 统一 Harness | `src/travel_agent/harness/` |

## 一句话总结

```text
harness = case + environment + runner + trace/artifact capture + validators + reporting + release gates。
```

它的价值是让 Agent 从“能跑一次”变成“能稳定、可复现、可评估地运行”。

## 项目的三层记忆系统

三层记忆要区分“存储能力”和“当前被哪条执行链消费”。下面这行代码是 V0 单 ReAct
Agent 构建本轮提示词记忆的入口：

```python
memory = MemoryFramework.build(
    ctx,
    history,
    user_id,
    settings.memory,
    settings.llm.enabled,
)
```

可以把 `MemoryFramework` 理解成“记忆整理器和 V0 提示词适配器”。它不是保存全部
记忆的数据库，而是从不同数据源读取记忆，按照长度决定压缩方式，再整理成适合注入
ReAct Agent 提示词的内容。

当前生产 V3 仍共用 `SessionContext`、`ArtifactStore` 和 `UserMemoryService`，并在
`run_architecture_turn()` 外层读取/持久化 L3；但 V3 的 Orchestrator/Subagent 提示词
尚未直接调用 `MemoryFramework.build()`，因此 L1 摘要和 L2 prompt snapshot 当前只在
V0 `_run_react()` 中完整注入。Harness 可以采集 L3 `memory_snapshot`，这不等于该快照
已经作为 V3 模型提示词发送。

三层记忆分别是：

```text
L1：当前会话的对话历史
L2：当前会话中各个工具产生的结构化结果
L3：同一用户跨会话保存的长期偏好画像
```

V0 的提示词流向如下：

```text
history ----------------------> L1 压缩/保留 -----------+
                                                           |
ctx.store（工具 artifacts）---> L2 工具快照 ------------+--> build_system_prompt
                                                           |          |
UserMemoryService(user_id) ---> L3 稳定偏好/近期行程 -----+          v
                                                                ReAct Agent
                                                                     |
                                                                     v
                                                       工具调用、生成回复
                                                                     |
                                                                     v
                           runtime 外层保存偏好事件和通过 critic 的紧凑行程
```

### `MemoryFramework.build` 的参数

```python
MemoryFramework.build(ctx, history, user_id, memory_settings, llm_enabled)
```

各参数的含义：

```text
ctx
    当前会话上下文。里面有当前旅行画像 ctx.profile、工具结果 ctx.store、
    provider 和已经检索过的 POI。

history
    当前 session 之前的用户消息和助手回复，形式是：
    [("user", "..."), ("assistant", "...")]

user_id
    用户标识。L3 使用它读取 JSON 或 PostgreSQL 中该用户的独立记录。

memory_settings
    记忆阈值、近期保留轮数、L3 backend、文件目录和数据库连接等配置。

llm_enabled
    是否允许 MemoryCompressor 使用 LLM 总结旧对话；不可用时使用规则摘要。
```

构建完成后，`memory` 大致是：

```python
MemoryFramework(
    mode="compressed",
    history_for_prompt=[
        ("user", "杭州有什么适合带父母去的地方？"),
        ("assistant", "你计划玩几天？"),
        ("user", "三天，节奏轻松一点。"),
    ],
    compressed_summary="历史对话摘要：\n用户：想去杭州……",
    l2_snapshot="- 天气：杭州 多云 25°C\n- 已检索 POI：杭州 共 18 个",
    l3_snapshot="用户ID：u_001\n长期偏好：美食, 历史\n常用节奏：relaxed",
)
```

### L1：对话历史记忆

L1 的原始数据是 `history: list[tuple[str, str]]`，例如：

```python
history = [
    ("user", "我想去杭州旅游。"),
    ("assistant", "计划玩几天？"),
    ("user", "三天，带父母。"),
    ("assistant", "更喜欢自然、历史还是美食？"),
]
```

`choose_memory_mode()` 根据历史长度选择三种模式：

```text
full
    消息少于 20 条，并且估算历史低于生效的强压缩阈值；全部原样传给 Agent。

compressed
    消息数量达到 20 条，但估算历史仍低于强压缩阈值；保留最近 10 轮，
    较早消息变成摘要。

profile_only
    估算历史达到生效的强压缩阈值；更早消息变成摘要，并继续保留最近 10 轮原文，
    同时注入当前画像、L2 和 L3。这个名称为兼容已有接口而保留，当前实现
    并不是“只传画像”。
```

选择顺序与边界条件是：

```python
est_tokens = estimate_history_tokens(history)
threshold = effective_profile_only_threshold(settings, llm_settings)
if est_tokens >= threshold:
    return "profile_only"
if len(history) >= settings.compress_message_threshold:
    return "compressed"
return "full"
```

强压缩阈值不是永远等于配置的 64K，而是：

```python
min(profile_only_token_threshold, model_context_window - 20_000)
```

模型窗口优先读取 `[llm].context_window`；未显式配置时按模型名前缀估算，未知模型默认
128K。20K 用于给摘要输出和下一轮回复预留空间。因此 token 判断优先：即使消息不到
20 条，只要估算历史达到生效阈值，也会选择 `profile_only`；两个模式阈值都使用
`>=`，达到阈值的当次构建就会切换。

默认配置是：

```toml
[memory]
compress_message_threshold = 20
profile_only_token_threshold = 64000
keep_recent_turns = 10
backend = "json"
profile_dir = "data/profiles"
```

这里的“轮”由一条用户消息和一条助手消息组成，因此 `keep_recent_turns = 10`
最多保留最近 20 条消息。`estimate_history_tokens()` 对每条历史消息采用以下规则：

```python
ascii_chars = sum(char.isascii() for char in content)
non_ascii_chars = len(content) - ascii_chars
estimated_tokens = (ascii_chars + 3) // 4 + non_ascii_chars
```

也就是 ASCII 文本约 4 字符/token，中文等非 ASCII 文本约 1 字符/token；例如
“杭州旅游规划”按 6 token 估算。这不是目标模型 tokenizer 的精确结果，代码、URL、表情
和特殊符号仍可能产生误差，生产监控还应记录模型 API 返回的实际输入 token，用于校准
阈值。

配置中的 64K 是历史预算上限，不是模型上下文上限；对于窗口小于 84K 的模型，实际
阈值会进一步下降。ReAct 在一次请求中可能多次调用模型和工具，每一步都可能重新携带
历史；提前压缩可以控制延迟和累计 token 成本，同时通过摘要、结构化旅行画像、L2
工具快照和近期原文保留真正影响规划的上下文。

在 `compressed` 模式下，摘要优先由 LLM 生成：

```text
请用中文 3-6 条要点概括目的地、天数、偏好和已做决策，不要编造。
```

相同旧历史和模型的成功摘要会缓存在进程内 LRU（最多 128 项）。连续 3 次 LLM 摘要
失败后会熔断 10 分钟；LLM 未启用、没有 API key、熔断中、调用失败或没有返回有效
内容时，`_rule_summarize()` 逐条截取旧消息，每条最多保留约 120 个字符。

例如 30 条历史消息、保留最近 10 轮时：

```text
前 10 条消息 -> compressed_summary
后 20 条消息 -> history_for_prompt
```

#### 压缩器如何初始化

`MemoryFramework.build()` 会先创建 `MemoryCompressor`：

```python
compressor = MemoryCompressor(
    llm_settings=settings.llm if llm_enabled else None,
    keep_recent_turns=memory_settings.keep_recent_turns,
)
```

`settings.llm if llm_enabled else None` 是条件表达式：LLM 可用时传入当前配置的模型、
API key、接口地址和超时配置；不可用时传入 `None`，后续改用本地规则摘要。
`keep_recent_turns` 控制压缩后保留多少轮原文，不决定何时开始压缩。默认值 10 会通过
`max_messages = keep_recent_turns * 2` 换算成最多 20 条用户/助手消息。

#### 三种模式如何处理历史

核心分支可以概括为：

```python
if mode == "profile_only":
    max_messages = memory_settings.keep_recent_turns * 2
    hist = history[-max_messages:]
    old_history = history[:-max_messages]
    summary = llm_or_rule_summarize(old_history)
elif mode == "compressed":
    hist, summary = compressor.compress(history)
else:
    hist, summary = history, ""
```

切片 `history[-max_messages:]` 取得最近消息原文，`history[:-max_messages]` 取得需要
摘要的旧消息。`profile_only` 和 `compressed` 都优先使用 LLM 摘要，并在 LLM 不可用
或调用失败时退回规则摘要；`full` 直接保留全部历史且摘要为空。

当前 `profile_only` 还有一个边界：如果历史消息少于 20 条，但少数消息特别长，模式
可能因 token 阈值进入 `profile_only`，此时 `old_history` 仍为空，原文不会被切分。
因此当前阈值是模式切换预算，还不是严格的近期原文 token 硬上限。

#### `_llm_summarize()` 如何调用 OpenAI 兼容模型

LLM 摘要器先把历史转换成带角色标签的文本：

```text
用户: 我想去杭州玩三天
助手: 预算大概是多少？
用户: 人均三千，带父母
```

然后通过当前配置的 OpenAI 兼容接口创建 `ChatOpenAI`：

```python
model = ChatOpenAI(
    model=self.llm_settings.model,
    api_key=self.llm_settings.api_key,
    base_url=self.llm_settings.base_url,
    temperature=0.1,
    timeout=self.llm_settings.timeout_seconds,
)
```

虽然客户端类名是 `ChatOpenAI`，实际 provider 和模型仍由 `model` 与 `base_url`
决定；配置为千问和百炼地址时才调用千问。这里的 `temperature=0.1` 是摘要器中固定的
低随机性参数，不读取主 Agent 的 `[llm].temperature`。摘要需要稳定保留事实，因此
使用 0.1；它会减少随机变化，但不保证每次结果完全一致。

`model.invoke()` 是 LangChain Runnable 的同步调用方法。它把 `SystemMessage` 和
`HumanMessage` 发送给模型，阻塞等待完整结果，然后返回 `AIMessage`：

```python
resp = model.invoke(
    [
        SystemMessage(content="你是旅行助手记忆压缩器……"),
        HumanMessage(content=transcript),
    ]
)
```

一个代表性的返回对象是：

```python
AIMessage(
    content=(
        "- 目的地为杭州，计划游玩三天。\n"
        "- 人均预算约 3000 元。\n"
        "- 与父母同行，行程节奏应轻松。"
    ),
    response_metadata={
        "model_name": "<当前配置模型>",
        "finish_reason": "stop",
    },
    usage_metadata={
        "input_tokens": 82,
        "output_tokens": 38,
        "total_tokens": 120,
    },
)
```

兼容接口和 LangChain 版本不同，metadata 字段可能不同；稳定使用的是
`resp.content`。代码确认它是非空字符串后执行 `strip()` 并返回。网络错误、超时、
权限或返回格式异常目前都会返回空字符串，外层据此调用 `_rule_summarize()`。生产监控
应记录这些异常，而不是只静默降级；`usage_metadata` 可用于校准前面的 token 估算。

同步 `invoke()` 的对应形式包括异步 `await model.ainvoke(...)`、流式
`model.stream(...)` 和批量 `model.batch(...)`。当前摘要路径使用同步调用，保证生成
摘要后再继续构建本轮 Agent 上下文。

### L2：会话工具结果记忆

L2 来自 `ctx.store`，它是当前 session 的 `ArtifactStore`。工具不会把完整 POI 列表、行程等大对象反复塞进 LLM 对话，而是先保存为 artifact：

```python
artifact_id = ctx.store.put(
    "weather",
    {
        "city": "杭州",
        "condition": "多云",
        "temperature_c": 25,
        "source": "amap",
    },
)
```

保存后的完整记录大致是：

```json
{
  "artifact_id": "weather_a1b2c3d4",
  "kind": "weather",
  "session_id": "sess_123",
  "created_at": 1785800000.0,
  "payload": {
    "city": "杭州",
    "condition": "多云",
    "temperature_c": 25,
    "source": "amap"
  }
}
```

`MemoryFramework.build()` 不会把所有完整 artifact 都发给模型，而是调用：

```python
ctx.store.build_prompt_snapshot()
```

把关键状态压缩成 L2 快照，例如：

```text
- 天气：杭州 多云 25°C
- 已检索 POI：杭州 共 18 个
- 已打分候选：18 个
- 已有行程：杭州三日轻松游；critic通过=True
```

当前快照只汇总 `weather`、`candidates`、`ranked` 和 `itinerary` 四类 artifact。完整数据仍保存在 `ArtifactStore` 中，后续工具通过 `latest(kind)` 或 artifact id 读取。

如果启用了持久化，artifact 会写到：

```text
data/artifacts/<session_id>/<artifact_id>.json
```

因此服务重启后，`SessionLifecycleManager` 可以按 `session_id` 恢复本次会话的工具结果。

### L3：跨会话用户画像

L3 由 `UserMemoryService` 管理，并把长期数据分开保存为两类：

```text
稳定偏好
    多值：interests、food_preference、avoid
    单值：budget_level、pace、transport_mode

最近行程
    destination、start_date、days、companions、hotel_area、must_visit
    以及紧凑行程摘要和最多 30 个主要 POI
```

`destination`、`days`、`start_date`、`companions`、`hotel_area` 和 `must_visit`
不是稳定偏好。历史行程可以帮助 Agent 理解用户经历，但不会自动填入本轮画像。

稳定偏好采用事件和 evidence 两层结构。一次明确表达先生成确定性事件 ID，同一会话、
同一轮、同一分类和值的重试不会重复计数；然后聚合出正负次数、最后状态和最后时间。
“不要排太满”“不喜欢博物馆”等否定会停用对应值，之后再次正向表达可以重新激活。
多值分类各保留最多 20 个有效值，单值分类以最近一次明确表达为准。继承的 L3 内容和
Agent 推断值不会重新记为事件，因此不会因为模型反复看到提示词而被错误强化。

行程只有同时满足以下条件才保存：

```text
最新 itinerary artifact 存在
critic.passed is True
本轮回复不是 clarification
```

同一 `session_id` 的后续修改执行 upsert。每用户只保留最近 20 条且两年内的行程；偏好
事件保留最近 200 条且两年内。行程不保存完整路线、天气或工具 payload。

`MemoryFramework.build()` 通过 repository 读取稳定画像和最近 3 条行程：

```python
memory_service = get_user_memory_service(memory_settings)
l3_profile = memory_service.load_stable_profile(user_id)
recent_trips = memory_service.list_recent_trips(user_id, limit=3)
```

然后 `_format_l3()` 生成不超过约 2K 估算 token 的紧凑快照，例如：

```text
用户ID：u_001
长期偏好：food, history
常用节奏：relaxed
常用预算：mid
长期避开：intensive
最近行程：杭州｜2026-07-01｜3天｜西湖、灵隐寺……
```

`MemoryFramework.build()` 只读取 L3 并生成提示词快照，不会在这里直接修改 `ctx.profile`。
项目只在旅行轮次准备或 session 恢复时按既定规则引用适合复用的稳定偏好；历史目的地和
天数永远不会作为本轮已确认值。`run_architecture_turn()` 的统一外层收尾记录本轮明确
偏好，并仅在 critic 通过后保存紧凑行程，V0–V3、离线和追问路径不再各自写画像。

#### L3 存储后端与表结构

本地开发和测试默认使用 `JsonUserMemoryRepository`，继续写入
`data/profiles/<user_id>.json`。生产环境使用 `PostgresUserMemoryRepository`，PostgreSQL
是唯一事实源，包含四张表：

| 表 | 用途 |
| --- | --- |
| `user_memories` | 用户记忆根记录、版本和更新时间 |
| `user_preference_events` | 幂等的正向/否定偏好事件 |
| `user_preference_evidence` | 按用户、分类和值聚合的快速读取画像 |
| `user_recent_trips` | 按用户和 session 唯一的紧凑已完成行程 |

生产配置和迁移命令：

```bash
export TRAVEL_AGENT_MEMORY_BACKEND=postgres
export TRAVEL_AGENT_DATABASE_URL='postgresql+psycopg://user:password@host:5432/travel_agent'
.venv/bin/alembic upgrade head
PYTHONPATH=src .venv/bin/python scripts/migrate_user_profiles.py
```

PostgreSQL URL 缺失或连接失败时应用直接启动失败，不会回退到 JSON。旧版 JSON 在用户
首次访问时自动幂等迁移：稳定字段转换为一次正向 evidence，历史目的地或天数转换为一条
`legacy-*` 行程；原文件不会删除。上线前也可以运行批量命令，并用 `--profile-dir` 指定
来源目录；命令会输出 imported、skipped 和 failed 数量。

存储服务还提供 `forget_preference()`、`clear_preferences()` 和
`delete_user_memory()`，分别用于忘记单项偏好、清空稳定偏好和删除用户全部 L3 数据。

### 记忆如何进入 V0 Agent

`_run_react()`（V0 单 Agent 链路）调用 `MemoryFramework.build()`，随后把三层快照组装
进 V0 的系统提示词。生产 V3 目前不经过这里；它通过共享 runtime 外层应用 L3 稳定
偏好，并让 Subagent 通过显式 artifact id 协作，但尚未把 L1 摘要和 L2 快照统一注入
Orchestrator/Subagent prompt。

随后 `build_system_prompt()` 把记忆组成系统提示词：

```text
System:
你是一个专业的中文旅行规划助手……

当前已知出行画像（多轮累积）：
- 目的地：杭州
- 天数：3
- 偏好：美食、历史

【记忆模式：compressed】

L1 历史摘要：
历史对话摘要：用户计划带父母旅行，要求节奏轻松……

L2 本会话工具快照：
- 天气：杭州 多云 25°C
- 已检索 POI：杭州 共 18 个

L3 跨会话用户画像：
用户ID：u_001
长期偏好：美食, 历史
常用节奏：relaxed

User:
当前这句话
```

在 V0 的三个记忆模式下，只要 `history_for_prompt` 非空，它都会作为独立的 `HumanMessage`、
`AIMessage` 放在系统提示词与当前用户消息之间：

```text
System: 系统提示词 + 当前画像 + L1 摘要 + L2 + L3
User:   保留下来的历史用户消息
AI:     保留下来的历史助手回复
User:   当前用户消息
```

在 `profile_only` 模式下，`history_for_prompt` 最多保留最近
`keep_recent_turns * 2` 条消息；更早的消息进入 L1 摘要。若 LLM 摘要不可用，系统
退回规则摘要。当前画像、L2 和 L3 始终继续注入。

### `ctx.profile`、L2 和 L3 的区别

这三个概念容易混在一起：

```text
ctx.profile
    当前 session 正在使用的旅行需求和偏好，是工具执行时直接读取的工作状态。

L2 / ctx.store
    当前 session 已经执行过的工具结果，例如天气、候选 POI、排序和行程。

L3 / UserMemoryService
    按 user_id 保存稳定偏好和近期已完成行程，用于跨 session 提供紧凑参考。
```

例如用户本轮说“去杭州三天，带父母，节奏轻松”：

```text
ctx.profile.destination = "杭州"
ctx.profile.days = 3
ctx.profile.companions = "parents"
ctx.profile.pace = "relaxed"
```

调用天气和 POI 工具后：

```text
ctx.store 中新增 weather、candidates 等 L2 artifacts
```

本轮结束后：

```text
本轮明确偏好以幂等事件写入 L3；critic 通过的紧凑行程按 session upsert
session_state.json 保存当前 session 的 profile、history 和 pois_by_id
```

### 一句话总结

```text
MemoryFramework = 从 L1 对话、L2 工具结果、L3 用户画像中读取信息，
根据上下文长度做压缩，并把整理后的记忆注入 V0 ReAct Agent。
```

它是当前 V0 的提示词记忆接入框架；真正保存数据的是 `history`、`ArtifactStore`、
`UserMemoryRepository` 和 `SessionLifecycleManager`。生产 V3 已共享这些底层状态与 L3
生命周期，但 L1/L2 的统一 prompt 注入仍是明确的待接通边界。

## 意图分流与早退路由（turn_lifecycle / intent）

从 WebSocket 入口到 `run_turn_lifecycle()` 后，消息先经过 `prepare_turn()` 的共享预检，核心是“规则优先 + 可回退 LLM”。

- `analyze_travel_turn()`：
  - `classify_message_rule_based` 先给出 `MessageKind`（`greeting / travel / ambiguous / out_of_scope`）
  - `classify_task_type_rule_based` 给出 `TaskType`（`unknown / full_trip_plan / route_query / poi_advice / itinerary_revision / day_advice`）
  - 每轮先调用一次 `extract_profile_rule_based(user_message)`，得到共享的规则画像提取结果；
    `classify_message_rule_based`、`build_rule_patches` 和同轮偏好信号判断复用该对象，不再
    分别重复解析同一条消息。上述公开函数仍接受可选的提取结果，单独调用时可以自行提取，
    保持原有调用方式兼容。
  - `unknown` 表示尚未安全映射到已支持工作流，不再使用 `full_trip_plan` 充当未知类型占位；
    LLM 可以把它补充为一种已支持类型，仍为 `unknown` 时会在 `prepare_turn()` 追问，不进入 Engine。
  - 明确的目的地/天数 Patch 可以把 `unknown` 提升为完整规划；已有行程下的约束或偏好补充
    可以提升为行程修改。能力未覆盖且没有旅行状态依据的请求保持 `unknown`。
  - 旅行偏弱信号先走 `AMBIGUOUS`，必要时交给 LLM 复判；LLM 异常时回退规则结果。

- `contextual_followup`：
  - 条件是有历史 `itinerary`（`ctx.store.latest_id("itinerary")`）且
  - 用户表达了“随便/沿用”或有偏好信号（`user_skips_preference_prompt` 或 `turn_expresses_preferences`）。
  - 该标志为 `True` 时，某些非旅行信号不会直接被当作一般 out_of_scope 处理。

- 早退分支（不进 Engine）：
  - 触发条件：
    - `analysis.kind != TRAVEL`
    - `analysis.task_type in {UNKNOWN, FULL_TRIP_PLAN}`
    - `not contextual_followup`
  - 直接返回 `PreparedTurn(..., early_reply=AgentReply(...))`
  - 其中 `AgentReply` 的文本由 `conversation_reply_text()` 生成：
    - `GREETING`：返回问候+能力边界提示
    - `AMBIGUOUS`：先确认是否要旅行/行程意图，并索取目的地、天数、偏好
    - `OUT_OF_SCOPE`：回显用户片段并说明不在主要服务边界
  - 状态返回：`AMBIGUOUS` 设为 `STATUS_CLARIFICATION_REQUIRED`，其余一般为 `STATUS_COMPLETED`。
  - 即使 `kind == TRAVEL`，若最终 `task_type == UNKNOWN`，也会返回带
    `failure_reason="unknown_task_type"` 的澄清回复，避免误启动完整规划和 Planner。

### `UNKNOWN` 与“规划缺槽位”不是一回事

`TaskType.UNKNOWN` 表示系统还不知道应该执行哪一种工作流；`FULL_TRIP_PLAN` 缺少
`destination` 或 `days` 则表示工作流已经确定，只是规划所需信息尚未补齐。两者触发的
追问目的不同：

```text
UNKNOWN
    追问“你具体想做完整规划、路线查询、地点建议、单日建议，还是修改行程？”

FULL_TRIP_PLAN + 缺槽位
    通过 request_travel_info 追问“目的地和天数是什么？”
```

当前有三条典型转换路径。

第一种：已经确定为完整规划，但缺少必填槽位：

```text
第一轮：“我想去旅行”
    → task_type = FULL_TRIP_PLAN
    → destination / days 缺失
    → request_travel_info 追问

第二轮：“厦门，玩四天”
    → Patch 写入 ctx.profile.destination="厦门"、days=4
    → 仍是 FULL_TRIP_PLAN
    → 必填槽位齐全，进入规划
```

第二种：任务目标确实未知，用户下一轮明确工作流：

```text
第一轮：“帮我看看”
    → task_type = UNKNOWN
    → 追问具体旅行任务，不进入 Engine

第二轮：“帮我规划厦门四天”
    → task_type = FULL_TRIP_PLAN
    → 目的地和天数齐全，进入规划
```

第三种：没有“规划”关键词，但本轮已经提取到核心行程字段：

```text
本轮：“厦门四天”
    → 初步 task_type = UNKNOWN
    → build_rule_patches 提取 destination="厦门"、days=4
    → has_trip_slot_patch = True
    → 同一轮提升为 FULL_TRIP_PLAN，无需先追问一轮
```

如果当前 `ctx.profile` 已经保存目的地或天数，本轮又补充天数、兴趣、节奏、预算等旅行
信息，也会继续提升或保持为 `FULL_TRIP_PLAN`。如果 `ctx.store` 已经存在 itinerary，
同类约束或偏好补充则优先解释为 `ITINERARY_REVISION`，因为此时目标是修改现有计划，
不是重新建立一份计划。

能力未覆盖且没有可证明的旅行规划上下文时保持 `UNKNOWN`。例如“推荐旅行保险”不会仅
因为包含“旅行”二字就回退成完整规划。

### `_analyze_with_llm()`：复杂轮次的 LLM 增强

`prepare_turn()` 不会把每条消息都交给模型。`analyze_travel_turn()` 会先生成完整的
`rule_result`；只有消息属于 `TRAVEL`/`AMBIGUOUS`、LLM 已启用，并且消息为
`AMBIGUOUS` 或命中复杂信号时，才调用一次 `_analyze_with_llm()`。非旅行消息、简单明确
消息和 LLM 未启用场景直接使用规则结果。

```text
规则生成 rule_result
        |
        +-- 非旅行 / 简单明确 / LLM 未启用 --> 直接返回 rule_result
        |
        +-- 模糊或复杂表达 --> _analyze_with_llm()
                                  |
                                  +-- 失败/超时/非法 JSON --> 回退 rule_result
                                  |
                                  +-- 成功 --> _merge_llm_payload()
                                                  |
                                                  v
                                      经确定性校验的 TurnAnalysis
```

函数输入是：

```python
_analyze_with_llm(
    user_message,       # 当前用户消息
    ctx,                # 当前 SessionContext，只读取画像/约束
    settings,           # 模型、Provider、API 和超时配置
    history,            # 当前会话历史
    evaluation_trace,   # 可选评测追踪容器
)
```

它从 `ctx.profile` 读取当前的 `destination`、`days`、`start_date` 和完整
`constraint_state`，再附加最近 4 条历史消息及当前输入。这里的 4 条是消息数，不一定是
4 轮对话。模型调用属于 `preflight` 阶段，当前预算为：超时 120 秒、最大输出 512
tokens、重试 0 次，并请求 Provider 返回 JSON Object。因为规则结果已经是安全兜底，
模型失败时不重复长时间重试。

#### 发送给 LLM 的 `SystemMessage`

```python
SystemMessage(content=_TURN_ANALYSIS_SYSTEM)
```

`SystemMessage` 是固定的“轮次分析岗位说明书”，不包含某位用户的具体旅行内容。它要求
模型只输出 JSON，并一次给出四类结构化信息：

```json
{
  "kind": "greeting|travel|ambiguous|out_of_scope",
  "task_type": "unknown|full_trip_plan|route_query|poi_advice|itinerary_revision|day_advice",
  "slots": {
    "字段名": {"op": "set", "value": "..."}
  },
  "constraint_state": {
    "约束字段": "JSON 值"
  }
}
```

其中的规则重点是：

```text
kind
    区分纯寒暄、明确旅行、旅行弱信号和非旅行内容。

task_type
    只能选择 UNKNOWN 或 5 种已支持工作流，不能发明新类型。

slots
    只抽取本轮明确表达的画像更新；支持 set/clear，不得补写未提字段。
    destination 使用中文城市名，days 为正整数，枚举字段必须使用约定值。

constraint_state
    输出结合当前请求和最近对话后的有效显式约束；保留中文原义，规范日期和时间，
    本轮修改覆盖旧值，删除地点进入 removed，禁止推测用户未表达的要求。
```

系统 Prompt 还列出标准约束字段，例如住宿区域、酒店预算、人均预算、忌口、步行上限、
返程截止时间、停车偏好、固定活动、室内备选和无障碍要求等。这使模型尽量落到已有
Schema，而不是临时创造不稳定字段。

#### 发送给 LLM 的 `HumanMessage`

```python
HumanMessage(content=prompt)
```

`HumanMessage` 是当前用户与会话相关的动态输入，由三部分拼成：

```text
已知出行画像：目的地=上海；天数=4；出发日期=2026-10-01；
当前有效约束={"lodging_area":"人民广场附近"}

最近对话：
user: 上海四天，酒店住人民广场附近
assistant: 已为你生成行程

当前用户输入：酒店别换，第二天下雨就改成室内活动
```

如果没有画像字段，会写成 `已知出行画像：空`；没有历史时省略“最近对话”部分。
两条消息的分工是：

```text
SystemMessage
    定义模型怎样分析、允许输出什么 Schema、哪些内容不能推测。

HumanMessage
    提供这次要分析的用户原话，以及理解指代和省略所需的当前画像与最近上下文。
```

模型返回的 JSON 仍是不可信候选，不能直接写入 `ctx`。调用方会先解析并执行
`_merge_llm_payload(payload, rule_result)`：验证枚举和 Patch 类型，阻止 LLM 将完整规划
降级为轻量任务、把普通候选强化成 `must_visit`、从步行限制推断整趟步行，或在修改任务
中擅自改变目的地、日期和总天数。约束合并时确定性规则结果优先。最终 `source="llm"`
只表示本轮使用了 LLM 增强，不表示所有字段均来自模型。

- `turn_expresses_preferences()` 的“偏好信号”判定：
  - 命中任一结构化偏好：`interests / budget_level / companions`
  - 或 `pace != standard`、`must_visit`、`avoid`
  - 或包含偏好关键词（如：美食、自然、历史、轻松、预算、情侣、亲子、自驾、步行等）

- 该设计目标：
  - 非明确旅行问题快速回收
  - 避免无效主链路调用
  - 在异常/失败下保持确定性离线行为（不会卡死）
