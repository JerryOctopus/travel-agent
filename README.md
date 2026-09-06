# Personalized Travel Planning Agent

基于 **LangGraph** 的多 Agent 可控旅行规划系统。

- **生产入口固定 Multi-Agent Full（V3）**：Main Orchestrator 动态派工给五个领域
  Subagent（景点 / 餐厅 / 酒店 / 交通 / 规划），Reviewer 语义审查最多一次定向
  修复周期，渲染统一由 Engine 经 Renderer Gate 执行；
- **规划核心是可控、可评估的规划子图** `plan_and_critique`（plan → critic → revise
  闭环），保证最终行程是约束满足、可执行的，而不是 LLM 一次性吐出的不可控文本；
- **V0–V3 消融对照共用同一 Engine**：同一模型 / 工具 / 数据 / Token 预算下显式
  复现四种编排版本，证明多智能体不是「为了复杂而复杂」。

> 一句话卖点：既展示 agent 能力（真实自主 tool calling + 真实 API），又展示
> 「知道何时该用确定性控制」的工程判断力（可控规划子图 + Renderer Gate + 量化评估）。

完整目标与里程碑见 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)（**M0–M8 均已落地**）。

## 架构

```mermaid
flowchart TB
  user["用户"] --> web["FastAPI + Web (WebSocket /chat, 高德JS地图, A2UI卡片)"]
  web --> entry["run_production_turn (固定 Multi-Agent Full = V3)"]
  subgraph engine["MultiAgentEngine"]
    orch["Main Orchestrator (动态派工, 不持有渲染工具)"]
    sub["五个 Subagent: attraction / restaurant / hotel / transport / planner"]
    review["Reviewer (语义审查 + 最多一次定向修复)"]
    gate["Renderer Gate (统一渲染, 绑定 plan_artifact_id)"]
    orch --> sub
    sub --> review
    review --> gate
  end
  entry --> engine
  sub --> tools["工具层 (LangChain @tool, 按 agent 白名单)"]
  gate --> cards["A2UI 卡片 / 高德地图数据"]
  engine -. 同一套 toolkit .-> mcp["FastMCP Server (streamable-http)"]
  engine --> store["ArtifactStore (会话级落盘)"]
```

- 无 LLM key 时，`run_production_turn` 自动降级为**确定性兜底**（复用同一套工具按固定顺序
  跑通），保证项目可离线演示；有 key 时走真正的 ReAct 自主编排。
- in-process agent 与 MCP server 共用 `travel_agent.agent.toolkit` 的同一套工具实现
  （单一事实源），避免「黑盒大工具」。

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .          # 或 pip install -e ".[dev,mcp]"
```

> 国内网络可加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2. 配置（可选，无 key 也能离线跑）

```bash
cp config.toml.example config.toml
```

在 `config.toml` 填写：

- `[llm]`：`provider`（freellmapi/qwen/deepseek/zhipu/google/siliconflow/openai）+ `api_key` → 启用真实 ReAct agent；
- `[amap].web_key`：高德 **Web 服务** key（后端 REST：POI/天气/路线）；
- `[amap].js_key`：高德 **JS API** key（前端地图渲染，与上面是两把不同的 key）。

环境变量优先级高于 `config.toml`（便于 CI / 临时覆盖）。

#### 使用硅基流动 DeepSeek-V4-Flash

从硅基流动控制台创建 API key，然后配置：

```toml
[llm]
provider = "siliconflow"
api_key = "你的-硅基流动-API-Key"
# 以下两项可省略，默认值如下：
base_url = "https://api.siliconflow.cn/v1"
model = "deepseek-ai/DeepSeek-V4-Flash"
timeout_seconds = 60
temperature = 0.2
thinking_enabled = false
```

也可以用 `SILICONFLOW_API_KEY` 环境变量保存密钥。`thinking_enabled = true` 时，
项目会为该模型请求 `reasoning_effort = "high"`；旅行工具编排优先建议先用非思考模式，
确认链路稳定后再开启。

#### 使用 Google Gemini API

从 Google AI Studio 创建 Gemini API key，然后配置：

```toml
[llm]
provider = "google"
api_key = "你的-Gemini-API-Key"
# 以下两项可省略，默认值如下：
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
model = "gemini-2.5-flash"
timeout_seconds = 60
```

项目复用现有 OpenAI-compatible 客户端和 ReAct tool calling 链路；`gemini`、
`google-gemini` 也可作为 provider 别名。配置后运行
`.venv/bin/python scripts/validate_api_keys.py` 验证密钥，再启动 Web。
也可以不在 TOML 保存密钥，改用 `GEMINI_API_KEY` 环境变量；provider 切换时不会
复用其他服务留在 `config.toml` 里的凭据。

#### 用 FreeLLMAPI 聚合免费模型

[FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) 在本机提供统一的
OpenAI-compatible API，并在已配置的免费 provider 之间路由和故障切换。

macOS / Windows 推荐直接安装其
[Desktop Release](https://github.com/tashfeenahmed/freellmapi/releases/latest)，不需要 Docker。
macOS 安装包未签名，首次启动可先在 Finder 中右键应用并选择“打开”；如果系统仍提示
应用“已损坏”，确认 DMG 来自官方 Release 后，按项目官方 issue #373 的 workaround 执行
`sudo xattr -cr /Applications/FreeLLMAPI.app`。Linux 或希望用容器运行时再使用官方安装
脚本（需要 Docker）：

```bash
curl -fsSL https://freellmapi.co/install.sh | bash
```

Desktop 启动后使用 macOS/Windows 菜单栏（托盘）的 FreeLLMAPI 图标进入原生 Dashboard；
它会创建本机内部账户，不需要在外部浏览器注册。Docker/server 安装才打开
`http://localhost:3001`。在 **Keys** 页面添加各 provider 的免费 API key，并复制
Dashboard 生成的 unified key。然后将本项目的 `config.toml` 配成：

```toml
[llm]
provider = "freellmapi"
api_key = "freellmapi-your-unified-key"
# 以下两项可省略；这是 freellmapi provider 的内置默认值。
base_url = "http://localhost:31415/v1"
model = "auto"
timeout_seconds = 120
```

Desktop 默认监听 `31415`；Docker/server 安装可把 `provider` 改为
`freellmapi-docker`（默认 `http://localhost:3001/v1`），也可以显式覆盖 `base_url`。

`model = "auto"` 让路由器自动选择；也可以换成 `auto:fast`、`auto:smart`、Dashboard
里的命名 profile 或具体模型 ID。更换上游免费 API 和调整 fallback chain 都在
FreeLLMAPI Dashboard 完成，本项目配置无需随之改动。首次运行前可验证连接：

```bash
.venv/bin/python scripts/validate_api_keys.py
```

本项目的 Multi-Agent 链路依赖 tool calling；如果固定具体模型，请选择支持 tools 的模型。
免费额度、可用性和限速可能变化，适合个人实验和原型，不应视为生产 SLA。

### 3. 启动 Web

```bash
PYTHONPATH=src .venv/bin/python -m travel_agent.server
# 打开 http://localhost:8000
```

浏览器里完成「多轮对话 → 工具调用 → 带 critic 修正的行程 → 地图点位与路线」。

## 工具集合（拆细，供 LLM 自主编排）

| 工具 | 作用 |
| --- | --- |
| `update_travel_profile` | 累积多轮出行画像 |
| `request_travel_info` | 信息缺失时追问 |
| `search_poi` | 检索候选 POI（高德优先，本地 seed 兜底） |
| `check_weather` | 天气（雨天户外冲突判断） |
| `plan_route` | 两点距离/通勤时长 |
| `build_constraints` | 从画像生成可验证 ConstraintSet |
| `search_restaurant` | 餐厅候选：菜系 / 区域 / 预算 |
| `search_hotel` | 酒店候选：区域 / 预算 / 评分 |
| `estimate_budget` | 住宿 / 餐饮 / 门票 / 市内交通预算估算 |
| `recommend_candidates` | 多目标打分 + 多样性重排 |
| `plan_and_critique` | **可控规划子图**：plan → critic → revise 闭环 |
| `render_itinerary` / `render_map` | A2UI 卡片 / 高德地图数据 |

## 可控规划子图（差异化核心）

`src/travel_agent/planning_subgraph.py` 用 LangGraph `StateGraph` 把现有
planner / critic / reviser 编排成有限闭环：

```text
plan -> critic -> [passed? END : revise -> critic -> ...]   (最多 max_iters 轮)
```

刻意保持**确定性**（不调用 LLM）：什么样的行程算可行、约束是否满足必须可控、
可评估。输出保留「原始稿 vs 修正稿」对比，量化 critic 的价值。

## MCP Server（工具与 agent 解耦）

```bash
# 启动 MCP server（streamable-http, 默认 8765）
PYTHONPATH=src .venv/bin/python -m travel_agent.mcp_server

# 演示：动态拉取工具并调用
.venv/bin/python scripts/mcp_client_demo.py
# -> 从 MCP Server 拉取到 9 个工具：search_poi / plan_and_critique / ...
```

## M4–M6 加分项（已实现）

### M4 三层记忆 + 会话持久化

- **L1** `MemoryCompressor`：历史超阈值时规则/LLM 压缩
- **L2** `ArtifactStore`：工具结果落盘 + 注入 system prompt 快照
- **L3** `UserMemoryService`：稳定偏好 + 最近完成行程；本地 JSON、生产 PostgreSQL
- `SessionLifecycleManager`：按 `session_id` 恢复历史与 artifacts（重启可续）
- `choose_memory_mode`：`full` / `compressed` / `profile_only`

`memory_mode` 按以下顺序选择，token 阈值优先于消息数阈值：

```python
def estimate_text_tokens(content):
    ascii_chars = sum(char.isascii() for char in content)
    non_ascii_chars = len(content) - ascii_chars
    return (ascii_chars + 3) // 4 + non_ascii_chars

est_tokens = sum(estimate_text_tokens(content) for _, content in history)
if est_tokens >= profile_only_token_threshold:
    mode = "profile_only"
elif len(history) >= compress_message_threshold:
    mode = "compressed"
else:
    mode = "full"
```

面向 `qwen3.7-plus` 的生产默认值：

```toml
[memory]
compress_message_threshold = 20
profile_only_token_threshold = 64000
keep_recent_turns = 10
```

| 模式 | 触发条件 | 注入 Agent 的 L1 历史 |
| --- | --- | --- |
| `full` | 少于 20 条消息，且估算历史少于 64K token | 全部原始历史 |
| `compressed` | 至少 20 条消息，且估算历史少于 64K token | 旧消息摘要 + 最近 10 轮原文 |
| `profile_only` | 估算历史达到 64K token | 更早消息摘要 + 最近 10 轮原文 |

`profile_only` 是兼容保留的模式名，现在不会清空近期历史；三种模式都会继续注入
当前旅行画像、L2 工具快照和 L3 长期画像。64K 是 Agent 的历史运行预算，不是
`qwen3.7-plus` 的模型上下文上限。模型虽然支持更大的上下文，但 ReAct 每次工具调用
都可能再次携带历史，因此生产环境主动压缩可以控制延迟和累计 token 成本。

当前估算按 ASCII 约 4 字符/token、非 ASCII 约 1 字符/token 计算，更适合以中文
为主的千问会话。例如“杭州旅游规划”按 6 token 估算。它仍不是千问 tokenizer 的
精确结果，代码、URL 和特殊符号会存在误差；上线监控应同时记录模型 API 返回的实际
输入 token，用于后续校准阈值。环境变量
`TRAVEL_AGENT_MEMORY_COMPRESS_THRESHOLD`、
`TRAVEL_AGENT_MEMORY_PROFILE_ONLY_TOKEN_THRESHOLD` 和
`TRAVEL_AGENT_MEMORY_KEEP_RECENT_TURNS` 可以覆盖默认值。

#### L3 长期画像与 PostgreSQL

L3 不再永久合并整份 `TravelProfile`。它只保存两类跨会话信息：

- **稳定偏好**：`interests`、`food_preference`、`avoid` 以及最近一次明确表达的
  `budget_level`、`pace`、`transport_mode`。偏好按确定性事件 ID 去重，支持否定和重新激活。
- **最近行程**：只有最新 itinerary 存在、critic 通过且回复不是追问时才按
  `(user_id, session_id)` 写入。目的地、日期、天数等只进入这里，不会自动填成本轮需求。

生产部署使用 PostgreSQL：

```bash
export TRAVEL_AGENT_MEMORY_BACKEND=postgres
export TRAVEL_AGENT_DATABASE_URL='postgresql+psycopg://user:password@host:5432/travel_agent'
.venv/bin/alembic upgrade head
PYTHONPATH=src .venv/bin/python scripts/migrate_user_profiles.py
```

PostgreSQL 配置缺失或连接失败会直接启动失败，不会静默退回 JSON。本地和测试默认
`backend = "json"`，旧 `data/profiles/*.json` 会在首次访问或批量迁移时幂等转换，原文件
保留。也可用 `--profile-dir PATH` 指定旧文件目录。

保留策略为每用户最近 20 条且两年内的行程、最近 200 条且两年内的偏好事件；稳定偏好
的每个多值分类最多保留 20 个有效值。L3 prompt 只注入有效稳定偏好和最近 3 条紧凑行程，
整体控制在约 2K 估算 token，不包含完整路线、天气或工具 payload。

### V0–V3 架构消融实验

用同一模型 / 工具 / 数据 / Token 预算对比四种编排架构，证明多智能体不是
「为了复杂而复杂」。四个版本共用同一个 `MultiAgentEngine`，差异只体现在
`EngineCapabilities`：V0 单 Agent 基线 → V1 确定性规则派工 → V2 动态
Orchestrator（无 Reviewer）→ V3 = Production Full（动态 Orchestrator + Reviewer +
最多一次修复周期）。版本选择只能通过 `run_variant_turn("v0".."v3")` 显式指定，
生产入口不读取任何 variant。代码见 `src/travel_agent/orchestration/variants/`，
实验设计与运行方式见 [docs/ABLATION_V0_V3.md](docs/ABLATION_V0_V3.md)。

### M6 声明式 Skills

`.storyline/skills/*/SKILL.md` 启动时加载为 `skill_<id>` 工具，无需改 Python。内置：

- `full_trip_planner` / `rainy_day_alternative` / `structured_planner`

## 评估（M8）

当前评测体系分为三条相互隔离的证据链：

```text
production_v1.1：192 条（dev 34 调优 / core 94 + challenge 34 定版 / shadow 30 仅最终版），
              主指标 strict_task_success（七项合取 + gating 一票否决）
ChinaTravel：Mini-Dev 链路验证 + Human-154 固定沙箱回归
Live Tools Shadow：高德 API live/replay 工具可靠性

三套结果先做失败归因，再决定修 Prompt、工作流、工具、记忆或进入 SFT。
```

production_v1 的 split 纪律、V3 单候选 Core/Challenge/Shadow 发布验收与历史稳定性指标
见 [docs/EVALUATION_PRODUCT.md](docs/EVALUATION_PRODUCT.md)；V0–V3 架构消融见
[docs/ABLATION_V0_V3.md](docs/ABLATION_V0_V3.md)。

命令：

```bash
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite agent-product --env offline --product-split dev

# 非冻结 Dev 可定向运行；Core/Challenge/Shadow 禁止使用 legacy/all/relay 入口
PYTHONPATH=src python scripts/eval_product_multi_model.py --product-split dev --model deepseek:deepseek-v4-flash --write-report --json

# 冻结发布只能运行单候选、单模型、不可 resume 的 manifest 入口
PYTHONPATH=src python scripts/eval_product_multi_model.py --official-frozen \
  --release-manifest data/eval/releases/<candidate-id>/manifest.json \
  --product-split core_frozen --model deepseek:deepseek-v4-flash --run-id <run-id> --write-report --json

# 对已保存 Product run 做质量后处理，不重新调用被测 Agent
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py rules --run-dir data/eval/product/runs/<run_id>
# 独立 Judge 默认使用 Google / gemini-3.6-flash；也可通过 FreeLLMAPI 运行：
# TRAVEL_AGENT_JUDGE_PROVIDER=freellmapi FREELLMAPI_API_KEY=<unified-key>
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py judge --run-dir data/eval/product/runs/<run_id> --resume
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py sample --run-dir data/eval/product/runs/<run_id> --size 30
PYTHONPATH=src .venv/bin/python scripts/eval_plan_quality.py calibrate --run-dir data/eval/product/runs/<run_id> --reviews data/eval/product/runs/<run_id>/human_review.csv

PYTHONPATH=src python scripts/eval_chinatravel.py --suite mini-dev --write-report
PYTHONPATH=src python scripts/eval_chinatravel.py --suite human154 --resume --write-report
PYTHONPATH=src python scripts/eval_chinatravel.py --suite human1000 --resume --write-report

PYTHONPATH=src python scripts/eval_live_tools.py --provider amap --suite shadow-full --mode live --write-report
PYTHONPATH=src python scripts/eval_live_tools.py --provider amap --suite shadow-full --mode replay --write-report
PYTHONPATH=src python scripts/eval_live_tools.py --provider chinatravel --suite shadow-full --mode live --write-report
PYTHONPATH=src python scripts/eval_live_tools.py --provider chinatravel --suite shadow-full --mode replay --write-report
```

Judge 对 schedule、route、constraints、personalization、completeness、diversity、
clarity 七个维度给出总计 100 分，并把逐条结果写入 case 的
`evaluation.independent_judge`，聚合结果写入 `summary.json` 的
`metrics.plan_quality_judge`。Judge 不读取 Agent 内部 critic 或 deterministic
rule 结论；`strict_task_success` 与 Judge 分数分别报告，不合成为单一优化分数。

当前 Shadow Testing `shadow-full` 为 100 条 case。高德 replay 覆盖 300 个
POI / 天气 / 路线操作；ChinaTravel replay 覆盖 600 个 POI / 天气 / 路线 /
餐厅 / 酒店 / 预算操作。最新 ChinaTravel replay 基线：operation success
`100%`、restaurant / hotel / budget success `100% / 100% / 100%`、schema
valid `100%`、duplicate POI `0%`、unexpected route unavailable `0%`、
fallback `8.33%`、primary usable `91.67%`。

前端灰度闭环已补齐：用户可在导出卡里修改 POI 名称和开始时间，确认后导出
高德路书草稿，也可按人工修改后的方案重新生成；全程不自动下单、不自动支付。

```bash
# Agent / 规划 / 闭环 级评估（强制离线，可复现），并生成 docs/EVALUATION.md
.venv/bin/python scripts/eval_agent.py --write-report

# 校验 LLM / 高德 key 是否可用
.venv/bin/python scripts/validate_api_keys.py
```

评估分四组（理解 / Agent、ChinaTravel 式计划质量、闭环、真实多 Agent 的
agent_trace 口径），详见 [docs/EVALUATION.md](docs/EVALUATION.md)。
V2 数据集、指标、故障归因与微调门槛详见 [docs/EVALUATION_V2.md](docs/EVALUATION_V2.md)。
面试讲述稿见 [docs/INTERVIEW_PITCH.md](docs/INTERVIEW_PITCH.md)。

## 测试

```bash
.venv/bin/python -m pytest -q
```

## 旧版规则 workflow（离线兜底 / 对照）

旧的固定流水线 `run_mvp_workflow` 仍保留为无 key 时的离线兜底与评估对照：

```bash
PYTHONPATH=src .venv/bin/python demo.py "帮我规划北京两天，喜欢历史和美食，不要太累"
PYTHONPATH=src .venv/bin/python chat_demo.py
```

## 项目文档

- [开发路线与里程碑](docs/PROJECT_PLAN.md)
- [评估报告](docs/EVALUATION.md)
- [面试讲述稿](docs/INTERVIEW_PITCH.md)
- [架构设计](docs/ARCHITECTURE.md)
- [V0–V3 架构消融实验](docs/ABLATION_V0_V3.md)
- [工具接口](docs/TOOL_INTERFACE.md)
- [LLM 结构化抽取设计](docs/LLM_EXTRACTOR.md)
- [数据结构说明](docs/DATA_STRUCTURE.md)
