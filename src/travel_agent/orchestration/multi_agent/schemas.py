"""多 Agent 架构核心 Schema（Step 1 基础框架）。

对应最终架构的结构化契约：

- ``SubagentTask``：一次派工任务（Engine/Orchestrator → Subagent）；
- ``SubagentResult``：Subagent 结构化回流结果——Orchestrator 与 Planner 主要
  消费 ``payload / evidence / constraints_used / warnings / unresolved``，
  ``summary`` 仅供人类阅读与最终回复措辞，禁止只拼接自然语言；
- ``ReviewIssue`` / ``ReviewResult``：Semantic Reviewer 的结构化输出；
- ``EngineCapabilities``：架构能力开关（V0–V3 由同一 Engine 配置降级产生）。

Evidence 与 Artifact 必须携带 request_id / task_id / agent / artifact_id /
data_source，保证并发场景下可精确关联。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

# --- 状态与分级常量 --------------------------------------------------------- #

STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_INCOMPLETE = "incomplete"
STATUS_CLARIFICATION_REQUIRED = "clarification_required"

SEVERITY_CRITICAL = "critical"
SEVERITY_RECOVERABLE = "recoverable"
SEVERITY_NONCRITICAL = "noncritical"
SEVERITIES = (SEVERITY_CRITICAL, SEVERITY_RECOVERABLE, SEVERITY_NONCRITICAL)

VERDICT_PASS = "pass"
VERDICT_REWORK = "rework"
VERDICT_FAILED = "failed"


def new_request_id() -> str:
    """本轮请求唯一 ID，贯穿 artifact / evidence / agent_trace。"""
    return f"req_{uuid.uuid4().hex[:12]}"


def new_task_id(agent: str) -> str:
    """每次派工生成唯一任务 ID，如 ``attraction-a1b2c3d4``。"""
    return f"{agent}-{uuid.uuid4().hex[:8]}"


@dataclass
class SubagentTask:
    """一次派工任务。``depends_on`` 为通用依赖字段（fixed/dynamic 均可携带）。"""

    request_id: str
    task_id: str
    agent: str
    instruction: str
    inputs: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_text(self) -> str:
        """渲染为下发给 Subagent 的任务文本。"""
        lines = [self.instruction]
        if self.inputs:
            lines.append(f"输入上下文（JSON）：{_compact_json(self.inputs)}")
        if self.constraints:
            lines.append(f"必须遵守的约束（JSON）：{_compact_json(self.constraints)}")
        if self.depends_on:
            lines.append(f"依赖的上游任务：{', '.join(self.depends_on)}")
        return "\n".join(lines)


@dataclass
class SubagentResult:
    """Subagent 结构化回流结果。"""

    request_id: str
    task_id: str
    agent: str
    status: str
    attempt: int = 1

    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    constraints_used: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    tool_trace: list[str] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewIssue:
    """Reviewer 发现的单个问题，携带定向修复信息。"""

    issue_type: str
    severity: str  # critical | recoverable | noncritical
    description: str
    evidence: list[str] = field(default_factory=list)
    repair_target: str = ""  # 领域 subagent 类型或 "planner"
    repair_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewResult:
    """Semantic Reviewer 的结构化输出（Engine 单层无工具 LLM 调用产出）。"""

    verdict: str  # pass | rework | failed
    issues: list[ReviewIssue] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def critical_issues(self) -> list[ReviewIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_CRITICAL]


@dataclass(frozen=True)
class EngineCapabilities:
    """架构能力开关：V0–V3 由同一 Engine 按此配置降级产生。

    ``reviewer_enabled`` 是 Reviewer 启停的唯一决策点（架构开关）；
    settings 只保存运行参数（步数/超时/模型），不设架构启停字段。
    """

    mode: str  # "single" | "orchestrated"
    dispatch: str  # "none" | "fixed" | "dynamic"
    reviewer_enabled: bool
    max_rework: int  # 0 | 1


def _compact_json(data: Any) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, default=str)
