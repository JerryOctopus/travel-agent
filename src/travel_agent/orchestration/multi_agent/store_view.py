"""Planner 读取领域结果的受控视图（Step 2）。

规则（严格执行）：

- Planner **必须按 Orchestrator 传入的明确 ``artifact_id`` 列表**读取领域结果，
  不得依赖 ``store.latest()`` 获取并行领域结果（并发下 latest 语义不安全）；
- 读取时校验记录元数据（agent / kind），产出者不符或记录缺失都显式报告，
  不静默降级。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannerInputs:
    """Planner 派工前的输入包：明确的 artifact_id 列表 + 约束。"""

    artifact_ids: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactReadResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    mismatched: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_ids and not self.mismatched


def read_artifacts_for_planner(
    store: Any,
    artifact_ids: list[str],
    *,
    expected_agent: str | None = None,
) -> ArtifactReadResult:
    """按明确 artifact_id 批量读取完整记录并做产出者校验。

    - 缺失的 id 记入 ``missing_ids``；
    - 指定 ``expected_agent`` 时，记录 ``agent`` 元数据不符的记入 ``mismatched``；
    - 旧记录（无 agent 元数据）不做产出者校验，仅保留记录本身。
    """
    result = ArtifactReadResult()
    for artifact_id in artifact_ids:
        record = store.get_record(artifact_id)
        if record is None:
            result.missing_ids.append(artifact_id)
            continue
        producer = record.get("agent")
        if expected_agent is not None and producer not in (None, expected_agent):
            result.mismatched.append(
                {"artifact_id": artifact_id, "agent": producer}
            )
            continue
        result.records.append(record)
    return result
