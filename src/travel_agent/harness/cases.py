from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HarnessCase:
    """A single harness case, either one-turn or multi-turn.

    兼容两种数据格式：项目内旧 JSON（Product-120 遗留字段）与外部
    travel-agent-eval-production-v1 的 JSONL schema（conversation/gold/fixture_id，
    见 data/eval/production_v1/）。
    """

    case_id: str
    turns: list[str]
    expected_city: str | None = None
    expected_days: int | None = None
    required_interests: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    expect_clarification: bool | None = None
    expect_itinerary: bool | None = None
    split: str = "dev"
    category: str = "legacy"
    session_ids: list[str] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    tool_argument_assertions: list[dict[str, Any]] = field(default_factory=list)
    hard_constraints: dict[str, Any] = field(default_factory=dict)
    soft_preferences: dict[str, Any] = field(default_factory=dict)
    expected_memory: list[dict[str, Any]] = field(default_factory=list)
    # Partial state assertions evaluated after specific user turns.
    turn_expectations: list[dict[str, Any]] = field(default_factory=list)
    required_behaviors: list[str] = field(default_factory=list)
    forbidden_behaviors: list[str] = field(default_factory=list)
    failure_injection: dict[str, Any] = field(default_factory=dict)
    high_risk: bool = False
    snapshot_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional artifact-first schema field. Legacy cases omit it and are
    # classified from immutable gold/task text by the evaluator.
    expected_artifact_type: str | None = None
    # production_v1 外部 schema 字段
    gold_outcome: str | None = None
    gold_constraints_tree: dict[str, Any] = field(default_factory=dict)
    must_clarify_before_plan: list[str] = field(default_factory=list)
    fixture_id: str | None = None
    fixture_profile: str | None = None
    evaluators: list[str] = field(default_factory=list)
    architecture_policy: dict[str, Any] = field(default_factory=dict)
    subset: str | None = None
    difficulty: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessCase":
        if "conversation" in data and "gold" in data:
            return cls._from_production_dict(data)
        return cls._from_legacy_dict(data)

    @classmethod
    def _from_production_dict(cls, data: dict[str, Any]) -> "HarnessCase":
        """外部 travel-agent-eval-production-v1 JSONL schema（保真导入，gold 仅存不喂给 agent）。"""
        conversation = data.get("conversation") or []
        turns = [
            str(item.get("content") or "")
            for item in conversation
            if str(item.get("role") or "user") == "user"
        ]
        if not turns and data.get("user_query"):
            turns = [str(data["user_query"])]
        gold = data.get("gold") or {}
        constraints_tree = dict(gold.get("constraints") or {})
        metadata = {
            "dataset_version": data.get("dataset_version"),
            "schema_version": data.get("schema_version"),
            "title": data.get("title"),
            "tags": list(data.get("tags") or []),
            "turn_mode": data.get("turn_mode"),
            "reference_datetime": data.get("reference_datetime"),
            "source_style": data.get("source_style"),
            "notes": data.get("notes"),
        }
        return cls(
            case_id=str(data.get("case_id")),
            turns=turns,
            split=str(data.get("split") or "dev"),
            category=str(data.get("subset") or "production"),
            required_tools=list(data.get("required_tools") or []),
            required_behaviors=list(gold.get("required_behaviors") or []),
            forbidden_behaviors=list(gold.get("forbidden_behaviors") or []),
            hard_constraints=dict(constraints_tree),
            turn_expectations=list(data.get("turn_expectations") or []),
            snapshot_date=str(data.get("reference_datetime") or "") or None,
            metadata={key: value for key, value in metadata.items() if value is not None},
            expected_artifact_type=str(data.get("expected_artifact_type") or "") or None,
            gold_outcome=str(gold.get("expected_outcome") or "") or None,
            gold_constraints_tree=constraints_tree,
            must_clarify_before_plan=list(gold.get("must_clarify_before_plan") or []),
            fixture_id=str(data.get("fixture_id") or "") or None,
            fixture_profile=str(data.get("fixture_profile") or "") or None,
            evaluators=list(data.get("evaluators") or []),
            architecture_policy=dict(data.get("architecture_policy") or {}),
            subset=str(data.get("subset") or "") or None,
            difficulty=str(data.get("difficulty") or "") or None,
        )

    @classmethod
    def _from_legacy_dict(cls, data: dict[str, Any]) -> "HarnessCase":
        turns = data.get("turns")
        if turns is None:
            query = data.get("query")
            turns = [query] if query else []
        expected_tools = list(data.get("expected_tools") or [])
        expected_tool = data.get("expected_tool")
        if expected_tool:
            expected_tools.append(str(expected_tool))
        expect_itinerary = data.get("expect_itinerary")
        if expect_itinerary is None and data.get("expect_no_itinerary") is not None:
            expect_itinerary = not bool(data.get("expect_no_itinerary"))
        known_keys = {
            "id",
            "case_id",
            "query",
            "turns",
            "expected_city",
            "expected_days",
            "required_interests",
            "expected_tools",
            "expected_tool",
            "expected_clarification",
            "expect_clarification",
            "expect_itinerary",
            "expect_no_itinerary",
            "split",
            "category",
            "session_ids",
            "user_ids",
            "required_tools",
            "allowed_tools",
            "forbidden_tools",
            "tool_argument_assertions",
            "hard_constraints",
            "soft_preferences",
            "expected_memory",
            "turn_expectations",
            "required_behaviors",
            "forbidden_behaviors",
            "failure_injection",
            "high_risk",
            "snapshot_date",
            "metadata",
            "expected_artifact_type",
        }
        return cls(
            case_id=str(data.get("case_id") or data.get("id")),
            turns=[str(turn) for turn in turns],
            expected_city=data.get("expected_city"),
            expected_days=data.get("expected_days"),
            required_interests=list(data.get("required_interests") or []),
            expected_tools=expected_tools,
            expect_clarification=data.get(
                "expect_clarification",
                data.get("expected_clarification"),
            ),
            expect_itinerary=expect_itinerary,
            split=str(data.get("split") or "dev"),
            category=str(data.get("category") or "legacy"),
            session_ids=[str(item) for item in data.get("session_ids") or []],
            user_ids=[str(item) for item in data.get("user_ids") or []],
            required_tools=list(data.get("required_tools") or expected_tools),
            allowed_tools=list(data.get("allowed_tools") or []),
            forbidden_tools=list(data.get("forbidden_tools") or []),
            tool_argument_assertions=list(data.get("tool_argument_assertions") or []),
            hard_constraints=dict(data.get("hard_constraints") or {}),
            soft_preferences=dict(data.get("soft_preferences") or {}),
            expected_memory=list(data.get("expected_memory") or []),
            turn_expectations=list(data.get("turn_expectations") or []),
            required_behaviors=list(data.get("required_behaviors") or []),
            forbidden_behaviors=list(data.get("forbidden_behaviors") or []),
            failure_injection=dict(data.get("failure_injection") or {}),
            high_risk=bool(data.get("high_risk", False)),
            snapshot_date=data.get("snapshot_date"),
            metadata={
                **dict(data.get("metadata") or {}),
                **{k: v for k, v in data.items() if k not in known_keys},
            },
            expected_artifact_type=str(data.get("expected_artifact_type") or "") or None,
        )


def load_cases_json(path: Path | str) -> list[HarnessCase]:
    file_path = Path(path)
    if file_path.suffix == ".jsonl":
        cases: list[HarnessCase] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cases.append(HarnessCase.from_dict(json.loads(line)))
        return cases
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Harness case file must contain a JSON list.")
    return [HarnessCase.from_dict(item) for item in payload]
