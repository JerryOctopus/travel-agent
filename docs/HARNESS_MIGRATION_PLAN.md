# Travel Agent Harness 化改造方案

## 目标

当前项目已经有 Agent 运行入口、工具层、会话状态、评测脚本和报告生成，但这些能力分散在 `runtime.py`、`server.py`、`scripts/eval_*.py` 中。

往 harness 方向改造的目标是：

```text
把“如何跑 Agent、如何给输入、如何接工具、如何管理上下文、如何验证输出、如何记录错误和指标”抽成统一框架。
```

改造后，`run_turn()` 仍然是 Agent 单轮能力入口；harness 负责批量驱动、环境隔离、工具策略、上下文注入、运行记录、验证和评估。

## 当前已有的 Harness 雏形

项目里已经存在这些 harness 元素：

```text
Agent 被驱动对象
  src/travel_agent/agent/runtime.py
  run_turn(...)

会话和上下文
  src/travel_agent/agent/session.py
  src/travel_agent/storage/session_manager.py
  ArtifactStore / SessionContext / SessionLifecycleManager

工具来源
  src/travel_agent/agent/tool_source.py
  local tools / MCP tools 切换

长期用户画像
  src/travel_agent/storage/user_profile.py
  UserProfileStore

评测脚本
  scripts/eval_agent.py
  scripts/eval_chinatravel.py
  scripts/eval_live_tools.py

测试 case
  eval/cases.json
```

主要问题：

```text
评测脚本承担了太多 harness 职责
case、运行结果、diagnostics、metrics 没有统一类型
环境配置逻辑散在脚本里
离线、真实 LLM、MCP、replay 等运行模式没有统一抽象
错误处理和运行记录不能被 Web / eval / CI 复用
```

## 推荐新增模块

建议新增一个包：

```text
src/travel_agent/harness/
```

第一阶段只做轻量抽象，不重写 Agent 核心。

推荐文件：

```text
src/travel_agent/harness/cases.py
src/travel_agent/harness/runner.py
src/travel_agent/harness/result.py
src/travel_agent/harness/validators.py
src/travel_agent/harness/reporting.py
src/travel_agent/harness/environments.py
```

### cases.py

定义统一 case 格式。

建议结构：

```python
@dataclass
class HarnessCase:
    case_id: str
    turns: list[str]
    expected_city: str | None = None
    expected_days: int | None = None
    required_interests: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    expect_clarification: bool | None = None
    expect_itinerary: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

单轮 case 可以转成：

```python
HarnessCase(case_id="x", turns=["杭州三天"])
```

多轮 case 可以转成：

```python
HarnessCase(case_id="x", turns=["我想去杭州", "三天", "喜欢美食"])
```

### result.py

定义统一运行结果。

建议结构：

```python
@dataclass
class HarnessTurnResult:
    user_message: str
    reply_text: str
    tool_trace: list[str]
    used_real_agent: bool
    clarification: bool
    profile: dict[str, Any]
    artifacts: dict[str, Any]
    error: str | None = None
    duration_ms: float | None = None


@dataclass
class HarnessCaseResult:
    case_id: str
    turns: list[HarnessTurnResult]
    final_profile: dict[str, Any]
    final_artifacts: dict[str, Any]
    passed: bool | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
```

这样 `eval_agent.py`、`eval_chinatravel.py`、CI 和调试工具都能读同一种结果。

### environments.py

定义运行环境策略。

建议支持：

```text
offline
  LLM provider=rule
  不调用真实高德
  profile_dir 使用临时目录

real_agent
  使用真实 LLM
  可选真实高德

mcp_tools
  Agent 通过 MCP 拉工具

replay
  使用录制好的工具返回
```

第一阶段可以先实现 `offline` 和 `real_agent`。

### runner.py

定义 harness 主执行器。

职责：

```text
创建 SessionContext
准备 Settings
执行 turns
调用 run_turn()
收集 reply、profile、artifacts、tool_trace
捕获异常
统计耗时
返回 HarnessCaseResult
```

建议接口：

```python
class AgentHarness:
    def __init__(self, settings: Settings, environment: HarnessEnvironment) -> None:
        ...

    def run_case(self, case: HarnessCase) -> HarnessCaseResult:
        ...

    def run_cases(self, cases: list[HarnessCase]) -> list[HarnessCaseResult]:
        ...
```

注意：harness 不应该改写 `run_turn()` 的业务逻辑，只负责驱动它。

### validators.py

定义验证器，把现有 eval 里的判断逻辑移出来。

建议拆成：

```text
IntentGateValidator
ProfileValidator
ToolTraceValidator
ItineraryValidator
PlanQualityValidator
```

第一阶段可以只实现一个统一函数：

```python
def validate_case_result(case: HarnessCase, result: HarnessCaseResult) -> dict[str, Any]:
    ...
```

输出：

```text
city_ok
days_ok
clarification_ok
itinerary_produced
tool_coverage_ok
critic_passed
environment_pass
constraint_pass
preference_pass
```

### reporting.py

负责聚合指标和写报告。

从 `scripts/eval_agent.py` 中迁移：

```text
准确率
完成率
平均工具步数
critic 通过率
ChinaTravel-style 指标
失败样例
Markdown 报告
JSON summary
```

脚本只保留 CLI 参数解析和调用 harness。

## 改造后的数据流

推荐数据流：

```text
HarnessCase
  -> AgentHarness.run_case()
  -> build_session()
  -> run_turn()
  -> AgentReply + ctx.store artifacts
  -> HarnessCaseResult
  -> validate_case_result()
  -> aggregate_metrics()
  -> JSON summary / Markdown report
```

和当前项目关系：

```text
run_turn() 不下沉到 harness
toolkit.py 不改业务语义
planning_subgraph.py 不改
eval 脚本逐步瘦身
```

## 第一阶段最小改造

建议先做一个小闭环，不要一次性重构所有评测脚本。

步骤：

```text
1. 新增 travel_agent.harness 包
2. 定义 HarnessCase / HarnessCaseResult
3. 实现 offline AgentHarness.run_case()
4. 把 eval/cases.json 转成 HarnessCase
5. 迁移 scripts/eval_agent.py 的 NL eval 到 harness
6. 保持输出 summary 字段基本兼容
7. 跑 tests 和 eval_agent.py 验证
```

当前已落地第一阶段骨架：

```text
src/travel_agent/harness/cases.py
src/travel_agent/harness/result.py
src/travel_agent/harness/environments.py
src/travel_agent/harness/runner.py
src/travel_agent/harness/validators.py
tests/test_harness.py
```

已具备：

```text
从 eval/cases.json 读取 HarnessCase
offline HarnessEnvironment
AgentHarness.run_case() / run_cases()
多轮 history 管理
每轮 reply、tool_trace、profile、artifact snapshot 记录
基础 city/days/clarification/itinerary/tool/plan quality 验证
```

暂时不动：

```text
ChinaTravel 官方评测适配
live tools replay
WebSocket server
MCP server
Multi-Agent 编排（现 orchestration/multi_agent）内部逻辑
```

这样风险最小，也能体现 harness 化方向。

## 第二阶段增强

第一阶段稳定后，再做：

```text
统一 real_agent / variant（V0–V3）/ mcp_tools 环境策略
把 scripts/eval_chinatravel.py 的运行诊断迁进 harness
把 live/replay tool provider 接入 HarnessEnvironment
统一 diagnostics JSON 结构
加入 case-level timeout / retry / stop policy
加入 tool trace schema 校验
加入 artifact snapshot 导出
```

当前 real-agent 评测已经接入 harness 聚合：

```text
scripts/eval_agent.py --real-multi-agent --real-limit 1 --json
  -> AgentHarness(mode="real_agent")
  -> aggregate_real_multi_agent_results(...)
```

真实大模型验收命令：

```bash
TRAVEL_AGENT_LLM_MODEL=qwen-turbo PYTHONPATH=src .venv/bin/python scripts/eval_agent.py --real-multi-agent --real-limit 1 --json
```

如果返回 `AllocationQuota.FreeTierOnly` / `Free quota exhausted`，说明请求已经到达真实模型服务，但云厂商账户禁止继续付费调用。需要在控制台充值或关闭“仅使用免费额度”限制后重跑。

`scripts/eval_agent.py --real-multi-agent` 会先发送一个极小的 LLM preflight 请求。preflight 失败时不会进入 Multi-Agent case 执行，错误会明确标出 provider、model、base_url、HTTP status 和云厂商返回摘要，避免把账号额度/权限问题误判为 harness 或 Agent 逻辑失败。

## 当前全量 Harness 入口

现已提供统一 harness CLI：

```bash
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite agent-nl --env offline --write-report
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite chinatravel-mini --env offline --chinatravel-root /Users/carrier/run/projects/ChinaTravel --write-report
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite live-tools-shadow --provider chinatravel --env replay --write-report
PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite all --env offline --write-report --json
```

旧脚本仍保持兼容，但入口已委托给 harness runner：

```text
scripts/eval_chinatravel.py -> ChinaTravelHarnessRunner
scripts/eval_live_tools.py -> LiveToolsHarnessRunner
```

上线汇总报告输出到：

```text
docs/EVALUATION_RELEASE.md
```

当前 release gate 会把 `agent-nl`、`chinatravel-mini`、`live-tools-shadow`
聚合为 `pass` / `blocked` / `fail` 口径。真实 LLM 仍通过 `agent-real`
suite 单独验收，当前阻塞为 DashScope `AllocationQuota.FreeTierOnly`。

## 后续产品化

更进一步可以做：

```text
支持失败 case resume
支持对比两个版本的输出
支持 golden trace
支持 CI gate
```

示例命令：

```bash
TRAVEL_AGENT_LLM_MODEL=qwen-turbo PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite agent-real --env real_agent --limit 1 --json
TRAVEL_AGENT_LLM_MODEL=qwen-turbo PYTHONPATH=src .venv/bin/python -m travel_agent.harness.cli --suite agent-real --env real_agent --json
```

## 验收标准

第一阶段完成后，应满足：

```text
eval/cases.json 能通过 AgentHarness 批量运行
每个 case 都有统一 HarnessCaseResult
每轮都记录 user_message、reply_text、tool_trace、profile、artifacts
离线模式不依赖 LLM key 和高德 key
原 eval_agent.py 的主要指标仍能输出
现有 pytest 不回退
```

建议验证命令：

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/eval_agent.py --write-report
```

## 面试叙事

可以这样讲：

```text
我把项目从“一个能跑的 Agent demo”往“可评估 Agent 系统”升级。
核心不是改 Agent 业务逻辑，而是抽出 harness：
它统一管理执行环境、输入 case、上下文、工具策略、循环控制、错误处理、验证和报告。
这样同一个 Agent 可以在 offline、real LLM、MCP tools、replay 等环境里可复现地运行和对比。
```

这比单纯说“我写了几个 eval 脚本”更工程化。
