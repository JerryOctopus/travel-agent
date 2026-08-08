from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from travel_agent.harness.result import HarnessSuiteResult
from travel_agent.settings import get_settings

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_ROOT = ROOT / "data" / "eval" / "live_tools" / "snapshots"
SUMMARY_PATH = ROOT / "data" / "eval" / "live_tools" / "summary.json"
CASE_ROOT = ROOT / "data" / "eval" / "live_tools" / "cases"


@dataclass(frozen=True)
class LiveToolsHarnessRunner:
    snapshot_root: Path = SNAPSHOT_ROOT
    case_root: Path = CASE_ROOT

    def run(
        self,
        *,
        provider: str = "amap",
        mode: str = "replay",
        suite: str = "shadow-full",
        limit: int | None = None,
        agent_real_multi_agent: bool = False,
        continue_on_llm_error: bool = False,
    ) -> HarnessSuiteResult:
        legacy = _legacy()
        settings = (
            legacy._configure_real_multi_agent()
            if agent_real_multi_agent
            else get_settings()
        )
        cases = _load_cases(self.case_root, suite, limit=limit)
        provider_bundle = legacy._build_shadow_provider(provider, settings)
        rows: list[dict[str, Any]] = []
        agent_rows: list[dict[str, Any]] = []
        stopped_early = False
        stop_reason = None

        for case in cases:
            snapshot_path = self.snapshot_root / provider / suite / f"{case['id']}.json"
            if mode == "replay":
                payload = legacy._load_snapshot(snapshot_path)
            else:
                payload = legacy._run_live_case(
                    case,
                    provider_bundle,
                    provider_name=provider,
                )
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                payload = _redact_snapshot(payload)
                snapshot_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            rows.extend(legacy._rows_from_payload(case, payload, replay=mode == "replay"))
            if agent_real_multi_agent:
                agent_row = legacy._run_agent_shadow_case(case, provider_bundle, settings)
                agent_rows.append(agent_row)
                if agent_row.get("external_llm_error") and not continue_on_llm_error:
                    stopped_early = True
                    stop_reason = "external_llm_error"
                    break

        metrics = legacy._metrics(rows)
        artifacts = {
            "provider": provider,
            "mode": mode,
            "agent_real_multi_agent": agent_real_multi_agent,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "architecture": "multi_agent_full_v3",
            "provider_configured": provider_bundle["primary"] is not None,
            "snapshot_root": str(self.snapshot_root / provider / suite),
            "snapshot_version": _snapshot_version(self.snapshot_root / provider / suite),
            "agent_metrics": (
                legacy._agent_metrics(agent_rows)
                if agent_real_multi_agent
                else None
            ),
        }
        return HarnessSuiteResult(
            suite=f"live-tools-{suite}",
            mode=mode,
            environment="real_agent" if agent_real_multi_agent else mode,
            case_count=len(cases),
            attempted_count=len(rows),
            metrics=metrics,
            rows=rows,
            artifacts=artifacts,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )


def build_live_tools_summary(result: HarnessSuiteResult) -> dict[str, Any]:
    return {
        "provider": result.artifacts["provider"],
        "mode": result.artifacts["mode"],
        "suite": result.suite.replace("live-tools-", "", 1),
        "agent_real_multi_agent": result.artifacts["agent_real_multi_agent"],
        "llm_provider": result.artifacts["llm_provider"],
        "llm_model": result.artifacts["llm_model"],
        "architecture": result.artifacts["architecture"],
        "provider_configured": result.artifacts["provider_configured"],
        "snapshot_root": result.artifacts["snapshot_root"],
        "snapshot_version": result.artifacts.get("snapshot_version"),
        "metrics": result.metrics,
        "agent_metrics": result.artifacts.get("agent_metrics"),
        "rows": result.rows,
        "agent_rows": [],
        "stopped_early": result.stopped_early,
        "stop_reason": result.stop_reason,
    }


def build_live_tools_report(result: HarnessSuiteResult) -> str:
    return _legacy().build_report(build_live_tools_summary(result))


def write_live_tools_summary(result: HarnessSuiteResult) -> None:
    summary_path = SUMMARY_PATH.parent / f"summary_{result.suite.replace('live-tools-', '', 1)}.json"
    provider_summary_path = (
        SUMMARY_PATH.parent
        / f"summary_{result.artifacts['provider']}_{result.suite.replace('live-tools-', '', 1)}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json = json.dumps(
        build_live_tools_summary(result),
        ensure_ascii=False,
        indent=2,
    )
    summary_path.write_text(summary_json, encoding="utf-8")
    provider_summary_path.write_text(summary_json, encoding="utf-8")


def _load_cases(case_root: Path, suite: str, *, limit: int | None) -> list[dict[str, Any]]:
    path = case_root / f"{suite.replace('-', '_')}.json"
    if not path.exists():
        raise RuntimeError(f"找不到 shadow case 文件：{path}")
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise RuntimeError(f"shadow case 文件格式错误：{path}")
    if limit is not None:
        cases = cases[: max(1, limit)]
    return cases


def _legacy():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import eval_live_tools as legacy

    return legacy


def _redact_snapshot(value: Any) -> Any:
    sensitive = {"api_key", "key", "token", "authorization", "password"}
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in sensitive else _redact_snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_snapshot(item) for item in value]
    return value


def _snapshot_version(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path.glob("*.json")) if path.exists() else []
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest() if files else "missing"
