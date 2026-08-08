"""V0–V3 架构消融评测：同一模型/工具/数据/测试集/预算边界下对比四种架构。

实验设计见 docs/ABLATION_V0_V3.md；数据集与三阶段执行规程见
docs/EVALUATION_PRODUCT.md（production_v1 180 条）。本脚本：

1. 对每个 variant（默认 v0,v1,v2,v3）用同一模型与同一 production_v1 split
   各跑一遍（复用 harness.product 的评测机器，主指标 strict_task_success）；
2. 每个 variant 落一份独立报告（runs/<variant>/summary.json + cases/）；
3. 汇总跨版本对比表（comparison.json / comparison.md）：strict 成功率、
   gating/grounding/授权/架构策略通过率、token/时延/工具调用数、预算超限数、
   V3 返工触发率，以及成本归一化成功率；
4. 输出逐对结论：V0→V1（拆分值不值）、V1→V2（动态路由增量）、
   V2→V3（Critic 增量）；
5. 阶段三稳定性：--cases-file + --repeat-index 追加运行，
   --stability-merge 合并三轮结果输出五项稳定性指标
   （Pass@1、Pass³、硬约束稳定满足率、工具轨迹稳定性、输出波动）。

用法示例：
    # 开发调优（默认 dev 30 条，可反复跑；冒烟加 --limit 3）
    python scripts/eval_ablation.py --variants v0,v1,v2,v3 --product-split dev
    # 阶段一：四版本正式对比（core_frozen 与 challenge_frozen 各跑一次，合计 480 次）
    python scripts/eval_ablation.py --product-split core_frozen
    python scripts/eval_ablation.py --product-split challenge_frozen
    # 阶段二：最终版本跑影子集（单 variant 才允许，30 次）
    python scripts/eval_ablation.py --variants v3 --product-split shadow_frozen
    # 阶段三：最终版本稳定性补跑两遍（60 条 × 2 = 120 次）
    python scripts/eval_ablation.py --variants v3 --cases-file data/eval/production_v1/stability_60.json --repeat-index 2
    python scripts/eval_ablation.py --variants v3 --cases-file data/eval/production_v1/stability_60.json --repeat-index 3
    # 合并三轮（第一遍来自阶段一的 frozen 运行目录）输出稳定性报告
    python scripts/eval_ablation.py --stability-merge --merge-runs <阶段一目录>,<repeat2目录>,<repeat3目录>
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent.harness.environments import HarnessEnvironment
from travel_agent.harness.product import (
    DEFAULT_PRODUCT_CASES,
    run_product_suite,
    write_product_run,
)
from travel_agent.harness.runner import AgentHarness
from travel_agent.orchestration.variants import VARIANTS
from travel_agent.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "eval" / "ablation"
ALL_VARIANTS = ("v0", "v1", "v2", "v3")
SPLIT_CHOICES = ["dev", "core_frozen", "challenge_frozen", "shadow_frozen", "all"]
# 阶段三稳定性：每轮必须稳定出现的核心工具步骤（不要求顺序一致）。
CORE_TRACE_STEPS = ("search_poi", "plan_route", "estimate_budget", "plan_and_critique")
# 三阶段执行计划（正常环境总执行次数口径）。
EXECUTION_PLAN = {
    "stage1_four_versions": "120 cases (core 90 + challenge 30) x 4 versions = 480",
    "stage2_shadow": "30 shadow cases x 1 final version = 30",
    "stage3_stability": "60 cases x 2 extra repeats = 120 (first pass included in stage 1)",
    "total_normal_runs": 630,
    "final_version_runs": 270,
    "independent_frozen_tasks": 150,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default=",".join(ALL_VARIANTS),
        help="要对比的架构版本，逗号分隔（v0=单Agent基线, v1=规则派工, v2=动态Orchestrator, v3=Production Full+Reviewer）。",
    )
    parser.add_argument("--product-cases", default=str(DEFAULT_PRODUCT_CASES))
    parser.add_argument(
        "--product-split",
        default="dev",
        choices=SPLIT_CHOICES,
        help=(
            "dev=30条调优集（可反复跑）；core/challenge_frozen=定版集（禁止回流调优）；"
            "shadow_frozen=影子集，仅允许单 variant（不参与版本选择）。"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 条（冒烟用）。")
    parser.add_argument(
        "--cases-file",
        default=None,
        help="case_id 清单 JSON（阶段三稳定性 stability_60.json）；指定后覆盖 split 过滤。",
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=1,
        help="重复轮次标记（阶段三补跑用 2/3），写入落盘目录与 row。",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="四版本共用的 Token 硬上限；缺省用 settings.orchestration.variant_token_budget。",
    )
    parser.add_argument(
        "--stability-merge",
        action="store_true",
        help="不跑 case，合并 --merge-runs 给出的三轮运行目录，输出五项稳定性指标。",
    )
    parser.add_argument(
        "--merge-runs",
        default="",
        help="逗号分隔的三个运行目录（同一 variant 的三轮，含 runs/<variant>/summary.json）。",
    )
    return parser


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        raise ValueError(f"未知 variant: {unknown}，可用: {sorted(VARIANTS)}")
    return variants or list(ALL_VARIANTS)


def run_variant(
    variant: str,
    *,
    settings,
    case_path: str,
    split: str,
    limit: int | None,
    token_budget: int,
    case_ids: list[str] | None = None,
) -> Any:
    variant_settings = replace(
        settings,
        orchestration=replace(
            settings.orchestration,
            variant_token_budget=token_budget,
        ),
    )
    harness = AgentHarness(
        settings=variant_settings,
        environment=HarnessEnvironment(
            mode="real_agent",
            persist=False,
            user_id=f"ablation_{variant}",
            variant=variant,
        ),
    )
    return run_product_suite(
        harness,
        case_path=case_path,
        split="all" if case_ids is not None else split,
        limit=limit,
        case_ids=case_ids,
    )


def _variant_case_metrics(result: Any) -> list[dict[str, Any]]:
    """从 case outputs 中抽取每 case 的 variant_metrics（步数/worker/返工等）。"""
    outputs = result.artifacts.get("_case_outputs") or []
    collected: list[dict[str, Any]] = []
    for output in outputs:
        case_id = str((output.get("case") or {}).get("case_id"))
        artifacts = output.get("final_artifacts") or {}
        metrics = artifacts.get("variant_metrics") or {}
        collected.append({"case_id": case_id, **metrics})
    return collected


def _safe_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def build_variant_summary(variant: str, result: Any) -> dict[str, Any]:
    metrics = dict(result.metrics)
    case_metrics = _variant_case_metrics(result)
    tokens = [item.get("tokens_used") for item in case_metrics if item.get("tokens_used")]
    budget_exceeded = sum(1 for item in case_metrics if item.get("budget_exceeded"))
    incomplete = [item.get("incomplete_reason") for item in case_metrics if item.get("incomplete_reason")]
    tool_calls = [row.get("tool_call_count") for row in result.rows if row.get("tool_call_count") is not None]

    summary: dict[str, Any] = {
        "variant": variant,
        "description": VARIANTS[variant].description,
        "case_count": result.case_count,
        # 主指标：strict_task_success（七项合取 + gating 一票否决）
        "strict_success_rate": metrics.get("strict_success_rate"),
        "gating_pass_rate": metrics.get("gating_pass_rate"),
        "grounding_pass_rate": metrics.get("grounding_pass_rate"),
        "authorization_pass_rate": metrics.get("authorization_pass_rate"),
        "architecture_policy_pass_rate": metrics.get("architecture_policy_pass_rate"),
        "outcome_match_rate": metrics.get("outcome_match_rate"),
        # 次要指标（诊断用）
        "passed_rate": _safe_mean([1.0 if row.get("passed") else 0.0 for row in result.rows])
        if result.rows
        else None,
        "task_completion_rate": metrics.get("task_completion_rate"),
        "hard_constraint_rate": metrics.get("hard_constraint_rate"),
        "soft_preference_rate": metrics.get("soft_preference_rate"),
        "critic_pass_rate": metrics.get("critic_pass_rate"),
        "clarification_accuracy": metrics.get("clarification_accuracy"),
        "avg_total_tokens": metrics.get("avg_total_tokens"),
        "latency_p50_ms": metrics.get("latency_p50_ms"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "avg_tool_calls": _safe_mean([float(value) for value in tool_calls]),
        "variant_avg_tokens_used": _safe_mean([float(value) for value in tokens]),
        "budget_exceeded_count": budget_exceeded,
        "incomplete_count": len(incomplete),
        "incomplete_reasons": dict(
            (reason, incomplete.count(reason)) for reason in set(incomplete)
        ),
    }

    if variant == "v3":
        reviews = [item.get("review") for item in case_metrics if item.get("review")]
        reworked = [item for item in reviews if item.get("rework_used")]
        summary["reviewer"] = {
            "reviewer_run_count": len(reviews),
            "rework_triggered_count": len(reworked),
            "rework_trigger_rate": _safe_mean([1.0] * len(reworked) + [0.0] * (len(reviews) - len(reworked)))
            if reviews
            else None,
            "verdict_breakdown": {
                verdict: sum(1 for item in reviews if item.get("verdict") == verdict)
                for verdict in {str(item.get("verdict")) for item in reviews}
            },
        }
    return summary


def _delta(summary_a: dict[str, Any], summary_b: dict[str, Any], key: str) -> float | None:
    value_a = summary_a.get(key)
    value_b = summary_b.get(key)
    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return None
    return round(value_b - value_a, 4)


def build_pairwise_conclusions(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [("v0", "v1", "任务拆分本身是否已经足够（规则派工相对单 Agent）"),
             ("v1", "v2", "动态路由的增量（动态 Orchestrator 相对规则派工）"),
             ("v2", "v3", "Reviewer 的增量（定向返工相对无验收）")]
    conclusions: list[dict[str, Any]] = []
    for base, candidate, question in pairs:
        if base not in summaries or candidate not in summaries:
            continue
        summary_a, summary_b = summaries[base], summaries[candidate]
        strict_delta = _delta(summary_a, summary_b, "strict_success_rate")
        completion_delta = _delta(summary_a, summary_b, "task_completion_rate")
        token_delta = _delta(summary_a, summary_b, "avg_total_tokens")
        conclusions.append(
            {
                "comparison": f"{base} -> {candidate}",
                "question": question,
                "strict_success_delta": strict_delta,
                "task_completion_delta": completion_delta,
                "avg_token_delta": token_delta,
                "verdict": _pair_verdict(strict_delta, token_delta),
            }
        )
    return conclusions


def _pair_verdict(strict_delta: float | None, token_delta: float | None) -> str:
    if strict_delta is None:
        return "证据不足（样本缺失）"
    if strict_delta > 0.02:
        if token_delta is not None and token_delta > 0:
            return "strict 提升，但成本上升：结合成本归一化成功率判断是否值得"
        return "strict 提升且成本未上升：架构复杂度有收益"
    if strict_delta < -0.02:
        return "strict 下降：该层架构复杂度没有收益"
    return "strict 基本持平：若成本上升，则该层复杂度为了复杂而复杂"


def _cost_normalized(summary: dict[str, Any]) -> float | None:
    """每万 token 的 strict 成功率，防止“赢只是因为它花得多”。"""
    rate = summary.get("strict_success_rate")
    tokens = summary.get("avg_total_tokens")
    if not isinstance(rate, (int, float)) or not isinstance(tokens, (int, float)) or tokens <= 0:
        return None
    return round(rate / (tokens / 10000.0), 6)


def build_comparison_markdown(
    *,
    run_id: str,
    split: str,
    model: str,
    token_budget: int,
    summaries: dict[str, dict[str, Any]],
    conclusions: list[dict[str, Any]],
) -> str:
    lines = [
        "# V0–V3 架构消融实验对比报告",
        "",
        f"- run_id: `{run_id}`",
        f"- split: `{split}`",
        f"- model: `{model}`",
        f"- token 硬上限: {token_budget if token_budget > 0 else '未设置'}",
        "- 实验设计: docs/ABLATION_V0_V3.md；数据集与三阶段规程: docs/EVALUATION_PRODUCT.md",
        "",
        "| 指标 | " + " | ".join(summaries) + " |",
        "| --- | " + " | ".join("---" for _ in summaries) + " |",
    ]
    rows = [
        ("说明", lambda s: s.get("description", "")),
        ("case 数", lambda s: s.get("case_count")),
        ("strict 成功率（主指标）", lambda s: s.get("strict_success_rate")),
        ("gating 通过率", lambda s: s.get("gating_pass_rate")),
        ("grounding 通过率", lambda s: s.get("grounding_pass_rate")),
        ("授权通过率", lambda s: s.get("authorization_pass_rate")),
        ("架构策略通过率", lambda s: s.get("architecture_policy_pass_rate")),
        ("outcome 匹配率", lambda s: s.get("outcome_match_rate")),
        ("任务完成率（次要）", lambda s: s.get("task_completion_rate")),
        ("硬约束通过率（次要）", lambda s: s.get("hard_constraint_rate")),
        ("追问准确率", lambda s: s.get("clarification_accuracy")),
        ("平均 token", lambda s: s.get("avg_total_tokens")),
        ("平均步数", lambda s: s.get("variant_avg_steps_used")),
        ("平均工具调用", lambda s: s.get("avg_tool_calls")),
        ("预算超限 case 数", lambda s: s.get("budget_exceeded_count")),
        ("未完成 case 数", lambda s: s.get("incomplete_count")),
        ("时延 p95 (ms)", lambda s: s.get("latency_p95_ms")),
    ]
    for label, getter in rows:
        values = []
        for variant in summaries:
            value = getter(summaries[variant])
            values.append("" if value is None else str(value)[:60])
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    normalized = {variant: _cost_normalized(summary) for variant, summary in summaries.items()}
    values = ["" if normalized.get(variant) is None else str(normalized[variant]) for variant in summaries]
    lines.append("| 成本归一化成功率（strict/万token） | " + " | ".join(values) + " |")

    lines.extend(["", "## 逐对结论", ""])
    for item in conclusions:
        lines.append(
            f"- **{item['comparison']}**（{item['question']}）："
            f"strictΔ={item['strict_success_delta']}，任务完成率Δ={item['task_completion_delta']}，"
            f"tokenΔ={item['avg_token_delta']} → {item['verdict']}"
        )
    v3 = summaries.get("v3", {}).get("critic")
    if v3:
        lines.extend(["", "## V3 Critic 运行统计", "", json.dumps(v3, ensure_ascii=False, indent=2)])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 阶段三：稳定性合并（同一条案例 3 次运行的五项指标）
# ---------------------------------------------------------------------------


def _load_run_rows(run_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    """读取一个运行目录的 rows（自动定位 runs/<variant>/summary.json）。"""
    candidates = sorted(Path(run_dir).glob("runs/*/summary.json"))
    if not candidates:
        direct = Path(run_dir) / "summary.json"
        if not direct.exists():
            raise FileNotFoundError(f"未在 {run_dir} 找到 summary.json")
        candidates = [direct]
    summary = json.loads(candidates[0].read_text(encoding="utf-8"))
    return summary.get("rows") or [], candidates[0].parent


def _poi_sets_from_case_outputs(case_dir: Path) -> dict[tuple[str, int], tuple[set[str], list[list[str]]]]:
    """从 cases/*.json 提取 (case_id, repeat) → (POI 集合, 逐日 POI 序列)。"""
    extracted: dict[tuple[str, int], tuple[set[str], list[list[str]]]] = {}
    if not case_dir.is_dir():
        return extracted
    for path in case_dir.glob("*.json"):
        output = json.loads(path.read_text(encoding="utf-8"))
        case_id = str((output.get("case") or {}).get("case_id"))
        repeat = int((output.get("execution") or {}).get("repeat", 1))
        itinerary = ((output.get("final_itinerary") or {}).get("itinerary")) or {}
        poi_set: set[str] = set()
        day_sequences: list[list[str]] = []
        for day in itinerary.get("days") or []:
            sequence = []
            for stop in day.get("stops") or []:
                poi_id = (stop.get("poi") or {}).get("poi_id")
                if poi_id:
                    poi_set.add(str(poi_id))
                    sequence.append(str(poi_id))
            day_sequences.append(sequence)
        extracted[(case_id, repeat)] = (poi_set, day_sequences)
    return extracted


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def _sequence_overlap(seq_a: list[str], seq_b: list[str]) -> float:
    if not seq_a and not seq_b:
        return 1.0
    length = max(len(seq_a), len(seq_b))
    matches = sum(1 for i in range(min(len(seq_a), len(seq_b))) if seq_a[i] == seq_b[i])
    return matches / length if length else 1.0


def _variance(values: list[float]) -> float | None:
    return round(statistics.pvariance(values), 4) if len(values) >= 2 else None


def aggregate_stability_runs(run_dirs: list[Path]) -> dict[str, Any]:
    """五项稳定性指标：Pass@1、Pass³、硬约束稳定满足率、工具轨迹稳定性、输出波动。"""
    per_run_rows: list[dict[str, Any]] = []
    per_run_pois: list[dict[tuple[str, int], tuple[set[str], list[list[str]]]]] = []
    for run_dir in run_dirs:
        rows, case_root = _load_run_rows(run_dir)
        per_run_rows.append(rows)
        per_run_pois.append(_poi_sets_from_case_outputs(case_root / "cases"))
    case_ids = sorted(
        {row["case_id"] for rows in per_run_rows for row in rows}
    )
    runs_per_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in case_ids}
    for rows in per_run_rows:
        for row in rows:
            runs_per_case.setdefault(row["case_id"], []).append(row)

    total_runs = sum(len(items) for items in runs_per_case.values())
    total_pass = sum(
        1 for items in runs_per_case.values() for row in items if row.get("strict_task_success")
    )
    full_runs = {case_id: items for case_id, items in runs_per_case.items() if len(items) == len(run_dirs)}
    pass_cubed = sum(
        1 for items in full_runs.values() if all(row.get("strict_task_success") for row in items)
    )
    at_least_once = sum(
        1 for items in full_runs.values() if any(row.get("strict_task_success") for row in items)
    )

    constraint_stable = 0
    constraint_fluctuations: dict[str, list[str]] = {}
    trace_stable = 0
    poi_overlaps: list[float] = []
    structure_overlaps: list[float] = []
    call_variances: list[float] = []
    token_variances: list[float] = []
    duration_variances: list[float] = []
    for case_id, items in full_runs.items():
        missing_sets = [
            {entry.get("path") for entry in (row.get("constraint_tree_missing") or [])}
            for row in items
        ]
        if all(not missing for missing in missing_sets):
            constraint_stable += 1
        else:
            unstable = sorted(set().union(*missing_sets))
            constraint_fluctuations[case_id] = unstable
        core_hit = all(
            all(step in (row.get("tool_trace") or []) for step in CORE_TRACE_STEPS)
            for row in items
        )
        trace_stable += core_hit

        poi_data = []
        for index, rows in enumerate(per_run_rows):
            row = next((item for item in rows if item["case_id"] == case_id), None)
            repeat = int(row["repeat"]) if row else index + 1
            data = per_run_pois[index].get((case_id, repeat))
            poi_data.append(data or (set(), []))
        for (poi_a, _), (poi_b, _) in itertools.combinations(poi_data, 2):
            poi_overlaps.append(_jaccard(poi_a, poi_b))
        for (_, days_a), (_, days_b) in itertools.combinations(poi_data, 2):
            if not days_a and not days_b:
                continue
            overlaps = [
                _sequence_overlap(days_a[i], days_b[i])
                for i in range(max(len(days_a), len(days_b)))
            ]
            if overlaps:
                structure_overlaps.append(sum(overlaps) / len(overlaps))
        for key, bucket in (
            ("tool_call_count", call_variances),
            ("total_tokens", token_variances),
            ("duration_ms", duration_variances),
        ):
            values = [float(row[key]) for row in items if isinstance(row.get(key), (int, float))]
            variance = _variance(values)
            if variance is not None:
                bucket.append(variance)

    return {
        "case_count": len(case_ids),
        "runs_per_case_expected": len(run_dirs),
        "fully_repeated_case_count": len(full_runs),
        "pass_at_1": round(total_pass / total_runs, 4) if total_runs else None,
        "at_least_once_rate": round(at_least_once / len(full_runs), 4) if full_runs else None,
        "pass_cubed_rate": round(pass_cubed / len(full_runs), 4) if full_runs else None,
        "hard_constraint_stable_rate": (
            round(constraint_stable / len(full_runs), 4) if full_runs else None
        ),
        "constraint_fluctuations": constraint_fluctuations,
        "core_trace_stable_rate": round(trace_stable / len(full_runs), 4) if full_runs else None,
        "core_trace_steps": list(CORE_TRACE_STEPS),
        "poi_overlap_mean": round(statistics.mean(poi_overlaps), 4) if poi_overlaps else None,
        "structure_overlap_mean": (
            round(statistics.mean(structure_overlaps), 4) if structure_overlaps else None
        ),
        "tool_call_variance_mean": _safe_mean(call_variances),
        "tool_call_variance_max": round(max(call_variances), 4) if call_variances else None,
        "token_variance_mean": _safe_mean(token_variances),
        "token_variance_max": round(max(token_variances), 4) if token_variances else None,
        "duration_variance_mean": _safe_mean(duration_variances),
        "duration_variance_max": round(max(duration_variances), 4) if duration_variances else None,
    }


def build_stability_markdown(report: dict[str, Any], run_dirs: list[Path]) -> str:
    lines = [
        "# 阶段三稳定性报告（同一最终版本 × 3 轮 × 60 条）",
        "",
        "红线：不要求三次行程完全一致，但不能一次完全可行、一次严重超预算、一次漏掉返程"
        "（由 Pass³ 与硬约束稳定满足率把关）。",
        "",
        f"- 合并运行目录: {', '.join(str(item) for item in run_dirs)}",
        f"- 案例数: {report['case_count']}（完整 3 轮: {report['fully_repeated_case_count']}）",
        f"- Pass@1（单次成功率）: {report['pass_at_1']}",
        f"- 至少成功一次: {report['at_least_once_rate']}",
        f"- Pass³（三次全部成功，生产稳定性结论以此为准）: {report['pass_cubed_rate']}",
        f"- 硬约束三轮恒满足率: {report['hard_constraint_stable_rate']}",
        f"- 工具轨迹稳定率（核心步骤: {', '.join(report['core_trace_steps'])}）: "
        f"{report['core_trace_stable_rate']}",
        f"- 推荐地点重合率（两两 Jaccard 均值）: {report['poi_overlap_mean']}",
        f"- 行程结构重合率（逐日序列均值）: {report['structure_overlap_mean']}",
        f"- 工具调用次数方差（均值/最大）: {report['tool_call_variance_mean']} / "
        f"{report['tool_call_variance_max']}",
        f"- Token 消耗方差（均值/最大）: {report['token_variance_mean']} / "
        f"{report['token_variance_max']}",
        f"- 时延方差（均值/最大）: {report['duration_variance_mean']} / "
        f"{report['duration_variance_max']}",
        "",
    ]
    fluctuations = report.get("constraint_fluctuations") or {}
    if fluctuations:
        lines.append("## 波动约束清单（哪一项在哪次破了）")
        lines.append("")
        for case_id, paths in sorted(fluctuations.items()):
            lines.append(f"- `{case_id}`: {', '.join(paths)}")
    else:
        lines.append("## 波动约束清单：无（全部约束三轮恒满足）")
    return "\n".join(lines) + "\n"


def _fair_controls(settings, token_budget: int) -> dict[str, Any]:
    return {
        "same_model_and_temperature": {
            "model": settings.llm.model,
            "provider": settings.llm.provider,
            "temperature": settings.llm.temperature,
        },
        "same_tool_schema_and_snapshots": "live_tools 快照 + harness 工具集，四版本共享",
        "same_plan_and_critique": "共享确定性 plan→critic→revise 闭环与规则集",
        "same_token_budget": token_budget,
        "same_evaluator_version": "production_evaluators (strict_task_success + gating)",
    }


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()

    if args.stability_merge:
        run_dirs = [Path(item.strip()) for item in args.merge_runs.split(",") if item.strip()]
        if len(run_dirs) != 3:
            raise SystemExit("--stability-merge 需要 --merge-runs 恰好给出三个运行目录（三轮）。")
        report = aggregate_stability_runs(run_dirs)
        run_root = Path(args.output_root) / (args.run_id or "stability")
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "stability_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_root / "stability_report.md").write_text(
            build_stability_markdown(report, run_dirs), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n稳定性报告已写入: {run_root / 'stability_report.md'}")
        return

    if not settings.llm.enabled and not args.stability_merge:
        # 影子集纪律先于 LLM 检查，保证离线环境也能拦住误用。
        variants = parse_variants(args.variants)
        if len(variants) > 1 and args.product_split == "shadow_frozen":
            raise SystemExit(
                "影子集（shadow_frozen）不参与版本选择：只允许最终选定版本单独运行。"
            )
        raise RuntimeError("消融评测需要真实 LLM 配置（provider != rule 且配置 api_key）。")

    variants = parse_variants(args.variants)
    if len(variants) > 1 and args.product_split == "shadow_frozen":
        raise SystemExit(
            "影子集（shadow_frozen）不参与版本选择：只允许最终选定版本单独运行。"
        )
    token_budget = args.token_budget if args.token_budget is not None else settings.orchestration.variant_token_budget
    case_ids = None
    if args.cases_file:
        payload = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("case_ids") or []
        case_ids = [str(item) for item in payload]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = Path(args.output_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        print(f"\n=== 运行消融变体 {variant}: {VARIANTS[variant].description}")
        result = run_variant(
            variant,
            settings=settings,
            case_path=args.product_cases,
            split=args.product_split,
            limit=args.limit,
            token_budget=token_budget,
            case_ids=case_ids,
        )
        if args.repeat_index != 1:
            for row in result.rows:
                row["repeat"] = args.repeat_index
        variant_dir = (
            variant if args.repeat_index == 1 else f"{variant}_repeat-{args.repeat_index}"
        )
        write_product_run(result, run_root / variant_dir, run_id=variant_dir)
        summary = build_variant_summary(variant, result)
        summary["repeat_index"] = args.repeat_index
        summaries[variant] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    conclusions = build_pairwise_conclusions(summaries)
    comparison = {
        "schema_version": "ablation-comparison-v2",
        "run_id": run_id,
        "split": args.product_split,
        "cases_file": args.cases_file,
        "repeat_index": args.repeat_index,
        "limit": args.limit,
        "model": settings.llm.model,
        "provider": settings.llm.provider,
        "token_budget": token_budget,
        "cases_path": args.product_cases,
        "fair_controls": _fair_controls(settings, token_budget),
        "execution_plan": EXECUTION_PLAN,
        "variants": summaries,
        "cost_normalized_success": {
            variant: _cost_normalized(summary) for variant, summary in summaries.items()
        },
        "pairwise_conclusions": conclusions,
    }
    (run_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = build_comparison_markdown(
        run_id=run_id,
        split=args.product_split,
        model=settings.llm.model,
        token_budget=token_budget,
        summaries=summaries,
        conclusions=conclusions,
    )
    (run_root / "comparison.md").write_text(markdown, encoding="utf-8")
    print(f"\n对比报告已写入: {run_root / 'comparison.md'}")


if __name__ == "__main__":
    main()
