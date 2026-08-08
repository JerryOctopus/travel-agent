from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.evaluation.chinatravel_adapter import (
    SUITE_TO_SPLIT,
    evaluate_predictions,
    load_chinatravel_cases,
    load_query_data_for_cases,
    run_agent_for_case,
    run_agent_for_case_with_diagnostics,
    save_prediction,
)
from travel_agent.harness.chinatravel import (
    ChinaTravelHarnessRunner,
    build_chinatravel_report,
    build_chinatravel_summary,
)
from travel_agent.settings import get_settings, load_settings


DEFAULT_CHINATRAVEL_ROOT = Path(os.getenv("TRAVEL_AGENT_CHINATRAVEL_ROOT", "/Users/carrier/run/projects/ChinaTravel"))
OUTPUT_ROOT = ROOT / "data" / "eval" / "chinatravel"
REPORT_PATH = ROOT / "docs" / "EVALUATION_CHINATRAVEL.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChinaTravel offline benchmark suites.")
    parser.add_argument("--suite", choices=["mini-dev", "human154", "human1000"], default="mini-dev")
    parser.add_argument("--chinatravel-root", type=Path, default=DEFAULT_CHINATRAVEL_ROOT)
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-valid-only",
        action="store_true",
        help="真实 Multi-Agent 模式下只复用已通过五层诊断且未 fallback 的预测；失败/额度错误样本会重跑。",
    )
    parser.add_argument(
        "--continue-on-llm-error",
        action="store_true",
        help="真实 Multi-Agent 模式下遇到 LLM quota/auth 等错误仍继续后续 case；默认会停止，避免污染评测。",
    )
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh")
    parser.add_argument(
        "--real-multi-agent",
        action="store_true",
        help="使用真实 LLM 跑 Multi-Agent Full（V3），而不是离线 deterministic fallback。",
    )
    args = parser.parse_args()

    result = ChinaTravelHarnessRunner(
        chinatravel_root=args.chinatravel_root,
        output_root=OUTPUT_ROOT,
        lang=args.lang,
    ).run(
        suite=args.suite,
        smoke_limit=args.smoke_limit,
        resume=args.resume,
        resume_valid_only=args.resume_valid_only,
        real_multi_agent=args.real_multi_agent,
        continue_on_llm_error=args.continue_on_llm_error,
    )
    summary = build_chinatravel_summary(result)
    suite_dir = OUTPUT_ROOT / args.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.write_report:
        REPORT_PATH.write_text(build_chinatravel_report(result), encoding="utf-8")
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    return

    settings = _configure_real_multi_agent() if args.real_multi_agent else _configure_offline()
    cases = load_chinatravel_cases(
        args.chinatravel_root,
        args.suite,
        smoke_limit=args.smoke_limit,
        lang=args.lang,
    )
    suite_dir = OUTPUT_ROOT / args.suite
    mode_name = "real_multi_agent" if args.real_multi_agent else "deterministic"
    predictions_dir = suite_dir / ("predictions_real_multi_agent" if args.real_multi_agent else "predictions")
    diagnostics_dir = suite_dir / ("diagnostics_real_multi_agent" if args.real_multi_agent else "diagnostics")
    predictions: dict[str, dict[str, Any]] = {}
    rows = []
    stopped_early = False
    stop_reason = None

    for index, case in enumerate(cases, start=1):
        prediction_path = predictions_dir / f"{case.query_id}.json"
        diagnostics_path = diagnostics_dir / f"{case.query_id}.json"
        if args.resume and prediction_path.exists() and (
            not args.resume_valid_only
            or not args.real_multi_agent
            or _diagnostics_is_valid_real_run(diagnostics_path)
        ):
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            diagnostics = (
                json.loads(diagnostics_path.read_text(encoding="utf-8"))
                if diagnostics_path.exists()
                else {}
            )
            status = "resume"
        else:
            if args.real_multi_agent:
                result = run_agent_for_case_with_diagnostics(case, settings)
                prediction = result["prediction"]
                diagnostics = result["diagnostics"]
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(
                    json.dumps(diagnostics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                prediction = run_agent_for_case(case, settings)
                diagnostics = {}
            if args.real_multi_agent and _diagnostics_has_external_llm_error(diagnostics):
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                diagnostics_path.write_text(
                    json.dumps(diagnostics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "query_id": case.query_id,
                        "status": "llm_error",
                        "delivered": False,
                        **_diagnostic_row(diagnostics),
                    }
                )
                print(f"[{index}/{len(cases)}] {case.query_id} llm_error")
                if not args.continue_on_llm_error:
                    stopped_early = True
                    stop_reason = "external_llm_error"
                    print("检测到外部 LLM quota/auth 错误，停止评测；修复额度后可用 --resume --resume-valid-only 继续。")
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
                **_diagnostic_row(diagnostics),
            }
        )
        print(f"[{index}/{len(cases)}] {case.query_id} {status}")

    split = SUITE_TO_SPLIT[args.suite]
    evaluation_cases = [case for case in cases if case.query_id in predictions]
    query_ids, query_data = load_query_data_for_cases(
        args.chinatravel_root,
        split,
        evaluation_cases,
        lang=args.lang,
    )
    metrics = evaluate_predictions(
        args.chinatravel_root,
        split,
        query_ids,
        query_data,
        predictions,
        lang=args.lang,
    )
    cumulative_diagnostic_rows = (
        _load_diagnostic_rows(diagnostics_dir, [case.query_id for case in cases])
        if args.real_multi_agent
        else []
    )
    prediction_file_count = (
        len(list(predictions_dir.glob("*.json"))) if predictions_dir.exists() else 0
    )
    summary = {
        "suite": args.suite,
        "split": split,
        "mode": mode_name,
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "architecture": "multi_agent_full_v3",
        "case_count": len(cases),
        "attempted_case_count": len(rows),
        "evaluated_case_count": len(evaluation_cases),
        "prediction_file_count": prediction_file_count,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "predictions_dir": str(predictions_dir),
        "diagnostics_dir": str(diagnostics_dir) if args.real_multi_agent else None,
        "metrics": metrics,
        "multi_agent_metrics": _multi_agent_metrics(rows) if args.real_multi_agent else None,
        "cumulative_multi_agent_metrics": (
            _multi_agent_metrics(cumulative_diagnostic_rows) if args.real_multi_agent else None
        ),
        "rows": rows,
    }
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.write_report:
        REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_report(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# ChinaTravel 离线评测报告",
        "",
        "> 自动生成：`PYTHONPATH=src python scripts/eval_chinatravel.py --suite ... --write-report`。",
        "",
        "## 评测层级",
        "",
        "- Mini-Dev：5-10 条黄金微型集，用于 Prompt / 多 Agent 协作流调试；",
        "- Human-154：版本级回归集，用于保存重要算法版本；",
        "- Human-1000：最终离线验收集，用于进入 Shadow Testing 前的完整跑分。",
        "",
        "## 本次结果",
        "",
        f"- suite：`{summary['suite']}`",
        f"- split：`{summary['split']}`",
        f"- mode：`{summary.get('mode', 'deterministic')}`",
        f"- llm：`{summary.get('llm_provider')}` / `{summary.get('llm_model')}`",
        f"- architecture：`{summary.get('architecture')}`",
        f"- provider：`{summary.get('provider')}`",
        f"- chinatravel_root：`{summary.get('chinatravel_root')}`",
        f"- database_version：`{summary.get('database_version')}`",
        f"- case_count：{summary['case_count']}",
        f"- attempted_case_count：{summary.get('attempted_case_count', summary['case_count'])}",
        f"- evaluated_case_count：{summary.get('evaluated_case_count', summary['case_count'])}",
        f"- prediction_file_count：{summary.get('prediction_file_count')}",
        f"- stopped_early：{summary.get('stopped_early', False)}",
        f"- stop_reason：{summary.get('stop_reason')}",
        f"- predictions：`{summary['predictions_dir']}`",
        f"- diagnostics：`{summary.get('diagnostics_dir')}`",
        f"- official_eval_available：{metrics.get('official_eval_available')}",
        f"- DR / Delivery Rate：{metrics.get('delivery_rate')}",
        f"- Schema Pass Rate：{metrics.get('schema_pass_rate')}",
        f"- Schema Error Count：{metrics.get('schema_error_count')}",
        f"- EPR micro：{metrics.get('epr_micro')}",
        f"- LPR micro：{metrics.get('lpr_micro')}",
        f"- C-LPR micro：{metrics.get('conditional_lpr_micro')}",
        f"- FPR：{metrics.get('fpr')}",
        f"- Preference Pass Rate：{metrics.get('preference_pass_rate')}",
        f"- All Pass Count：{metrics.get('all_pass_count')}",
        "",
    ]
    ma = summary.get("multi_agent_metrics") or {}
    if ma:
        lines.extend(
            [
                "## 真实 Multi-Agent 诊断（本次命令）",
                "",
                f"- row_count：{ma.get('row_count')}",
                f"- strict_valid_count：{ma.get('strict_valid_count')}",
                f"- used_real_agent_rate：{ma.get('used_real_agent_rate')}",
                f"- agent_trace_rate：{ma.get('agent_trace_rate')}",
                f"- required_agents_rate：{ma.get('required_agents_rate')}",
                f"- itinerary_produced_rate：{ma.get('itinerary_produced_rate')}",
                f"- fallback_used_rate：{ma.get('fallback_used_rate')}",
                f"- external_llm_error_rate：{ma.get('external_llm_error_rate')}",
                "",
            ]
        )
    cumulative_ma = summary.get("cumulative_multi_agent_metrics") or {}
    if cumulative_ma:
        lines.extend(
            [
                "## 真实 Multi-Agent 诊断（累计 diagnostics）",
                "",
                f"- row_count：{cumulative_ma.get('row_count')}",
                f"- strict_valid_count：{cumulative_ma.get('strict_valid_count')}",
                f"- strict_valid_rate：{cumulative_ma.get('strict_valid_rate')}",
                f"- used_real_agent_rate：{cumulative_ma.get('used_real_agent_rate')}",
                f"- agent_trace_rate：{cumulative_ma.get('agent_trace_rate')}",
                f"- required_agents_rate：{cumulative_ma.get('required_agents_rate')}",
                f"- itinerary_produced_rate：{cumulative_ma.get('itinerary_produced_rate')}",
                f"- fallback_used_rate：{cumulative_ma.get('fallback_used_rate')}",
                f"- external_llm_error_rate：{cumulative_ma.get('external_llm_error_rate')}",
                "",
            ]
        )
    breakdown = metrics.get("failure_breakdown") or {}
    if breakdown:
        lines.extend(["## 失败原因 TopN", ""])
        for title, key in [("Commonsense / 环境约束", "commonsense"), ("Logical / 需求约束", "logical")]:
            items = breakdown.get(key) or []
            lines.extend([f"### {title}", "", "| reason | fail_rate |", "| --- | ---: |"])
            if items:
                for item in items:
                    lines.append(f"| {item.get('reason')} | {item.get('fail_rate'):.4f} |")
            else:
                lines.append("| 无 | 0 |")
            lines.append("")
    failed_examples = metrics.get("failed_examples") or []
    if failed_examples:
        lines.extend(["## 未通过样例 Top20", "", "| query_id | commonsense | logical | query |", "| --- | --- | --- | --- |"])
        for item in failed_examples[:20]:
            query = str(item.get("query") or "").replace("\n", " ")[:120]
            lines.append(
                f"| {item.get('query_id')} | {'; '.join(item.get('commonsense') or []) or '-'} | "
                f"{'; '.join(item.get('logical') or []) or '-'} | {query} |"
            )
        lines.append("")
    if metrics.get("error"):
        lines.extend(
            [
                "## 官方评估未完成",
                "",
                metrics["error"],
                "",
                "这通常表示当前数据只适合检查 delivery/schema；若要计算 LPR / C-LPR / FPR，需要带 `hard_logic_py` 的官方评测文件。补齐后可用 `--resume --write-report` 复用预测文件重跑。",
                "",
            ]
        )
    schema_failed = metrics.get("schema_failed_examples") or []
    if schema_failed:
        lines.extend(["## Schema 失败样例", "", "| query_id | errors |", "| --- | --- |"])
        for item in schema_failed:
            lines.append(
                f"| {item.get('query_id')} | {'; '.join(item.get('errors') or [])} |"
            )
        lines.append("")
    failures = [row for row in summary["rows"] if not row["delivered"]][:20]
    lines.extend(["## 未交付样例 Top20", "", "| query_id | status |", "| --- | --- |"])
    for row in failures:
        lines.append(f"| {row['query_id']} | {row['status']} |")
    if not failures:
        lines.append("| 无 | - |")
    lines.append("")
    return "\n".join(lines)


def _configure_offline():
    os.environ["TRAVEL_AGENT_LLM_PROVIDER"] = "rule"
    os.environ.pop("TRAVEL_AGENT_LLM_API_KEY", None)
    os.environ["TRAVEL_AGENT_TOOL_PROVIDER"] = "local"
    os.environ["TRAVEL_AGENT_AMAP_WEB_KEY"] = ""
    os.environ["TRAVEL_AGENT_PROFILE_DIR"] = tempfile.mkdtemp(prefix="ct_eval_profile_")
    hf_home = ROOT / "data" / "eval" / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    get_settings.cache_clear()
    return get_settings()


def _configure_real_multi_agent():
    os.environ["TRAVEL_AGENT_SKILLS_ENABLED"] = "false"
    os.environ["TRAVEL_AGENT_PROFILE_DIR"] = tempfile.mkdtemp(prefix="ct_real_ma_profile_")
    hf_home = ROOT / "data" / "eval" / "hf_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    get_settings.cache_clear()
    settings = load_settings()
    if not settings.llm.enabled:
        raise RuntimeError("真实 Multi-Agent 评测需要可用 LLM 配置。")
    return settings


def _diagnostic_row(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if not diagnostics:
        return {}
    return {
        "used_real_agent": diagnostics.get("used_real_agent"),
        "agent_trace_present": diagnostics.get("agent_trace_present"),
        "required_agents_present": diagnostics.get("required_agents_present"),
        "itinerary_produced": diagnostics.get("itinerary_produced"),
        "fallback_used": diagnostics.get("fallback_used"),
        "tool_steps": len(diagnostics.get("tool_trace") or []),
        "failed_agent_count": len(diagnostics.get("failed_agent_reasons") or []),
        "external_llm_error": _diagnostics_has_external_llm_error(diagnostics),
    }


def _load_diagnostic_rows(diagnostics_dir: Path, query_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        path = diagnostics_dir / f"{query_id}.json"
        if not path.exists():
            continue
        try:
            diagnostics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append({"query_id": query_id, **_diagnostic_row(diagnostics)})
    return rows


def _multi_agent_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_valid_count = sum(1 for row in rows if _diagnostic_row_is_valid_real_run(row))
    return {
        "row_count": len(rows),
        "strict_valid_count": strict_valid_count,
        "strict_valid_rate": round(strict_valid_count / len(rows), 4) if rows else None,
        "used_real_agent_rate": _rate(rows, "used_real_agent"),
        "agent_trace_rate": _rate(rows, "agent_trace_present"),
        "required_agents_rate": _rate(rows, "required_agents_present"),
        "itinerary_produced_rate": _rate(rows, "itinerary_produced"),
        "fallback_used_rate": _rate(rows, "fallback_used"),
        "external_llm_error_rate": _rate(rows, "external_llm_error"),
        "avg_tool_steps": _avg(rows, "tool_steps"),
        "avg_failed_agent_count": _avg(rows, "failed_agent_count"),
    }


def _diagnostic_row_is_valid_real_run(row: dict[str, Any]) -> bool:
    return bool(
        row.get("used_real_agent")
        and row.get("itinerary_produced")
        and row.get("required_agents_present")
        and not row.get("fallback_used")
        and not row.get("external_llm_error")
    )


def _diagnostics_is_valid_real_run(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(
        diagnostics.get("used_real_agent")
        and diagnostics.get("itinerary_produced")
        and diagnostics.get("required_agents_present")
        and not diagnostics.get("fallback_used")
        and not _diagnostics_has_external_llm_error(diagnostics)
    )


def _diagnostics_has_external_llm_error(diagnostics: dict[str, Any]) -> bool:
    payload = json.dumps(diagnostics, ensure_ascii=False)
    markers = (
        "AllocationQuota",
        "FreeTierOnly",
        "free quota",
        "quota",
        "401",
        "403",
        "Unauthorized",
        "authentication",
        "api key",
        "invalid_parameter_error",
        "enable_thinking",
        "only support stream mode",
        "stream mode",
    )
    return any(marker in payload for marker in markers)


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 4)


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


if __name__ == "__main__":
    main()
