from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from travel_agent.evaluation.chinatravel_adapter import (
    SUITE_TO_SPLIT,
    evaluate_predictions,
    load_chinatravel_cases,
    load_query_data_for_cases,
    preflight_chinatravel,
    run_agent_for_case,
    run_agent_for_case_with_diagnostics,
    save_prediction,
)
from travel_agent.harness.result import HarnessSuiteResult
from travel_agent.settings import get_settings, load_settings

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "data" / "eval" / "chinatravel"
DEFAULT_CHINATRAVEL_ROOT = Path(
    os.getenv("TRAVEL_AGENT_CHINATRAVEL_ROOT", "/Users/carrier/run/projects/ChinaTravel")
)


@dataclass(frozen=True)
class ChinaTravelHarnessRunner:
    chinatravel_root: Path = DEFAULT_CHINATRAVEL_ROOT
    output_root: Path = OUTPUT_ROOT
    lang: str = "zh"

    def run(
        self,
        *,
        suite: str = "mini-dev",
        smoke_limit: int | None = None,
        resume: bool = False,
        resume_valid_only: bool = False,
        real_multi_agent: bool = False,
        continue_on_llm_error: bool = False,
    ) -> HarnessSuiteResult:
        legacy = _legacy()
        preflight = preflight_chinatravel(self.chinatravel_root)
        settings = _configure_real_multi_agent() if real_multi_agent else _configure_offline()
        cases = load_chinatravel_cases(
            self.chinatravel_root,
            suite,
            smoke_limit=smoke_limit,
            lang=self.lang,
        )
        suite_dir = self.output_root / suite
        mode_name = "real_multi_agent" if real_multi_agent else "deterministic"
        predictions_dir = suite_dir / (
            "predictions_real_multi_agent" if real_multi_agent else "predictions"
        )
        diagnostics_dir = suite_dir / (
            "diagnostics_real_multi_agent" if real_multi_agent else "diagnostics"
        )
        predictions: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        stopped_early = False
        stop_reason = None

        for case in cases:
            prediction_path = predictions_dir / f"{case.query_id}.json"
            diagnostics_path = diagnostics_dir / f"{case.query_id}.json"
            if resume and prediction_path.exists() and (
                not resume_valid_only
                or not real_multi_agent
                or legacy._diagnostics_is_valid_real_run(diagnostics_path)
            ):
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
                diagnostics = (
                    json.loads(diagnostics_path.read_text(encoding="utf-8"))
                    if diagnostics_path.exists()
                    else {}
                )
                status = "resume"
            else:
                if real_multi_agent:
                    result = run_agent_for_case_with_diagnostics(
                        case, settings, self.chinatravel_root
                    )
                    prediction = result["prediction"]
                    diagnostics = result["diagnostics"]
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    diagnostics_path.write_text(
                        json.dumps(diagnostics, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                else:
                    prediction = run_agent_for_case(
                        case, settings, self.chinatravel_root
                    )
                    diagnostics = {}
                if real_multi_agent and legacy._diagnostics_has_external_llm_error(diagnostics):
                    rows.append(
                        {
                            "query_id": case.query_id,
                            "status": "llm_error",
                            "delivered": False,
                            **legacy._diagnostic_row(diagnostics),
                        }
                    )
                    if not continue_on_llm_error:
                        stopped_early = True
                        stop_reason = "external_llm_error"
                        break
                    continue
                save_prediction(prediction, predictions_dir, case.query_id)
                status = "generated" if prediction else "empty"
            predictions[case.query_id] = prediction
            rows.append(
                {
                    "query_id": case.query_id,
                    "status": status,
                    "delivered": bool(prediction),
                    **legacy._diagnostic_row(diagnostics),
                }
            )

        split = SUITE_TO_SPLIT[suite]
        evaluation_cases = [case for case in cases if case.query_id in predictions]
        query_ids, query_data = load_query_data_for_cases(
            self.chinatravel_root,
            split,
            evaluation_cases,
            lang=self.lang,
        )
        metrics = evaluate_predictions(
            self.chinatravel_root,
            split,
            query_ids,
            query_data,
            predictions,
            lang=self.lang,
        )
        cumulative_rows = (
            legacy._load_diagnostic_rows(diagnostics_dir, [case.query_id for case in cases])
            if real_multi_agent
            else []
        )
        prediction_file_count = (
            len(list(predictions_dir.glob("*.json"))) if predictions_dir.exists() else 0
        )
        artifacts = {
            "suite": suite,
            "split": split,
            "mode": mode_name,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "architecture": "multi_agent_full_v3",
            "provider": "chinatravel_official_database",
            "chinatravel_root": preflight["root"],
            "database_root": preflight["database_root"],
            "database_version": preflight["database_version"],
            "preflight": preflight,
            "predictions_dir": str(predictions_dir),
            "diagnostics_dir": str(diagnostics_dir) if real_multi_agent else None,
            "prediction_file_count": prediction_file_count,
            "multi_agent_metrics": legacy._multi_agent_metrics(rows) if real_multi_agent else None,
            "cumulative_multi_agent_metrics": (
                legacy._multi_agent_metrics(cumulative_rows) if real_multi_agent else None
            ),
        }
        return HarnessSuiteResult(
            suite=f"chinatravel-{suite}",
            mode=mode_name,
            environment="real_agent" if real_multi_agent else "offline",
            case_count=len(cases),
            attempted_count=len(rows),
            metrics=metrics,
            rows=rows,
            artifacts=artifacts,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )


def build_chinatravel_summary(result: HarnessSuiteResult) -> dict[str, Any]:
    return {
        "suite": result.artifacts["suite"],
        "split": result.artifacts["split"],
        "mode": result.artifacts["mode"],
        "llm_provider": result.artifacts["llm_provider"],
        "llm_model": result.artifacts["llm_model"],
        "architecture": result.artifacts["architecture"],
        "provider": result.artifacts.get("provider"),
        "chinatravel_root": result.artifacts.get("chinatravel_root"),
        "database_version": result.artifacts.get("database_version"),
        "preflight": result.artifacts.get("preflight"),
        "case_count": result.case_count,
        "attempted_case_count": result.attempted_count,
        "evaluated_case_count": len([row for row in result.rows if row.get("query_id")]),
        "prediction_file_count": result.artifacts.get("prediction_file_count"),
        "stopped_early": result.stopped_early,
        "stop_reason": result.stop_reason,
        "predictions_dir": result.artifacts["predictions_dir"],
        "diagnostics_dir": result.artifacts.get("diagnostics_dir"),
        "metrics": result.metrics,
        "multi_agent_metrics": result.artifacts.get("multi_agent_metrics"),
        "cumulative_multi_agent_metrics": result.artifacts.get("cumulative_multi_agent_metrics"),
        "rows": result.rows,
    }


def build_chinatravel_report(result: HarnessSuiteResult) -> str:
    return _legacy().build_report(build_chinatravel_summary(result))


def _configure_offline():
    os.environ["TRAVEL_AGENT_LLM_PROVIDER"] = "rule"
    os.environ["TRAVEL_AGENT_LLM_MODEL"] = "rule-deterministic"
    os.environ.pop("TRAVEL_AGENT_LLM_API_KEY", None)
    os.environ["TRAVEL_AGENT_TOOL_PROVIDER"] = "local"
    os.environ["TRAVEL_AGENT_AMAP_WEB_KEY"] = ""
    os.environ["TRAVEL_AGENT_PROFILE_DIR"] = tempfile.mkdtemp(prefix="ct_eval_profile_")
    _configure_hf_cache()
    get_settings.cache_clear()
    return get_settings()


def _configure_real_multi_agent():
    os.environ["TRAVEL_AGENT_SKILLS_ENABLED"] = "false"
    os.environ["TRAVEL_AGENT_PROFILE_DIR"] = tempfile.mkdtemp(prefix="ct_real_ma_profile_")
    _configure_hf_cache()
    get_settings.cache_clear()
    settings = load_settings()
    if not settings.llm.enabled:
        raise RuntimeError("真实 Multi-Agent 评测需要可用 LLM 配置。")
    return settings


def _configure_hf_cache() -> None:
    hf_home = ROOT / "data" / "eval" / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))


def _legacy():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import eval_chinatravel as legacy

    return legacy
