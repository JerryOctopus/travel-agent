"""Build an auditable per-case report artifact from the frozen Dev34 run."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from statistics import mean
from datetime import datetime, timedelta, timezone

from travel_agent.evaluation.artifact_contract import (
    ITINERARY_TYPES,
    NON_ITINERARY_TYPES,
    actual_artifact_type,
    aggregate_artifact_metrics,
    artifact_type_matches,
    expected_artifact_type,
    itinerary_judge_route,
    task_completion_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an internally consistent Dev34 report from one run directory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--expected-case-count", type=int, default=34)
    parser.add_argument(
        "--plan-start-strict-count",
        type=int,
        help=(
            "Historical strict count recorded when the Dev34 improvement plan "
            "started. It is reported separately from both the original run "
            "metric and this script's current evaluator replay."
        ),
    )
    return parser


def yes_no(value: object) -> str:
    if value is True:
        return "通过"
    if value is False:
        return "失败"
    return "不适用"


def percent(value: object) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "N/A"


def quote_lines(text: str) -> str:
    lines = text.splitlines() or [text]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def user_dialogue(turns: list[str]) -> str:
    if not turns:
        return "> 未保存用户消息"
    if len(turns) == 1:
        return quote_lines(turns[0])
    rendered = []
    for index, turn in enumerate(turns, start=1):
        rendered.append(f"> 第 {index} 轮：{turn}")
    return "\n\n".join(rendered)


CATEGORY_LABELS = {
    "scenic": "景点",
    "museum": "博物馆",
    "food": "餐饮",
    "shopping": "街区/购物",
    "culture": "文化体验",
    "nightlife": "夜间活动",
    "transport": "交通",
}

MODE_LABELS = {
    "public_transport": "公共交通",
    "taxi": "打车",
    "walking": "步行",
    "driving": "驾车",
}

SCORE_LABELS = {
    "schedule": "时间安排",
    "route": "路线",
    "constraints": "约束满足",
    "personalization": "个性化",
    "completeness": "完整性",
    "diversity": "丰富度",
    "clarity": "清晰度",
}


def end_time(start: str, duration: object) -> str | None:
    if not start or not isinstance(duration, (int, float)):
        return None
    try:
        parsed = datetime.strptime(start, "%H:%M")
    except ValueError:
        return None
    return (parsed + timedelta(minutes=float(duration))).strftime("%H:%M")


def itinerary_markdown(payload: dict) -> str:
    itinerary = (payload.get("final_itinerary") or {}).get("itinerary") or {}
    days = itinerary.get("days") or []
    if not days:
        return "未生成结构化逐日行程。"
    lines: list[str] = []
    for day in days:
        day_index = day.get("day_index", "?")
        theme = day.get("theme") or "未命名主题"
        stops = day.get("stops") or []
        lines.append(f"**第 {day_index} 天｜{theme}**")
        if not stops:
            lines.append("- 当天没有生成具体安排。")
            lines.append("")
            continue
        for stop in stops:
            poi = stop.get("poi") or {}
            time = stop.get("start_time") or "时间未定"
            duration = stop.get("duration_min")
            finish = end_time(time, duration)
            time_text = f"{time}–{finish}" if finish else time
            category = CATEGORY_LABELS.get(poi.get("category"), poi.get("category") or "活动")
            lines.append(f"- {time_text}　{poi.get('name') or '未命名地点'}（{category}）")
            route = stop.get("route_from_previous") or {}
            if route:
                mode = MODE_LABELS.get(route.get("mode"), route.get("mode") or "交通方式未定")
                distance = route.get("distance_km")
                route_minutes = route.get("duration_min")
                parts = [f"从上一站乘{mode}"]
                if distance is not None:
                    parts.append(f"约 {distance} 公里")
                if route_minutes is not None:
                    parts.append(f"约 {route_minutes} 分钟")
                lines.append("  - 交通：" + "，".join(parts))
        lines.append("")
    return "\n".join(lines)


def status_note(judge: dict) -> str:
    if judge.get("status") == "not_applicable":
        return "不适用：该任务不要求行程 Judge。"
    if judge.get("status") == "not_run":
        return "未运行：" + judge_reason_label(str(judge.get("reason") or "not_run"))
    if judge.get("status") in {"missing", "error"}:
        return "Judge 未完成：" + judge_reason_label(str(judge.get("reason") or judge.get("status")))
    if judge.get("reasonable") is True:
        return "通过"
    return "未通过"


def reviewed_judge_label(judge: dict) -> str:
    if judge.get("status") == "not_applicable":
        return "N/A"
    if judge.get("status") == "not_run":
        return "未运行"
    if judge.get("status") != "ok":
        return "错误"
    return "通过" if judge.get("reasonable") is True else "未通过"


def judge_problem_lines(judge: dict, limit: int = 3) -> list[str]:
    issues = judge.get("critical_issues") or []
    selected = sorted(issues, key=lambda item: item.get("severity") != "critical")[:limit]
    replacements = {
        "hard_constraints.": "用户要求中的",
        "hard_constraints": "用户要求",
        "final_profile.": "最终记录中的",
        "final_profile": "最终记录",
        "tool_facts.": "工具结果中的",
        "tool_facts": "工具结果",
        "itinerary.days": "每日行程",
        "itinerary": "行程",
        "final_answer": "回答",
        "lodging_plan": "住宿方案",
        "candidate_verification": "候选核验",
    }
    cleaned: list[str] = []
    for item in selected:
        text = str(item.get("evidence") or item.get("code") or "未说明原因")
        for old, new in replacements.items():
            text = text.replace(old, new)
        cleaned.append(text)
    return cleaned


def judge_reason_label(reason: str) -> str:
    return {
        "task_does_not_require_itinerary": "任务本来不需要行程 Judge",
        "missing_expected_artifact": "缺少预期行程产物",
        "system_error": "系统执行失败",
        "partial_rubric_not_run": "Partial Rubric 尚未执行",
        "judge_not_attached": "Judge 未配置",
    }.get(reason, reason or "未提供原因")


def classified_evaluation(payload: dict) -> dict:
    case = payload.get("case") or {}
    metrics = ((payload.get("evaluation") or {}).get("rule_metrics") or {})
    stored_contract = metrics.get("expected_artifact_type") in {
        *ITINERARY_TYPES, *NON_ITINERARY_TYPES,
    }
    derived_expected = expected_artifact_type(case)
    expected = str(metrics.get("expected_artifact_type") or derived_expected)
    derived_actual = actual_artifact_type(payload, expected)
    actual = metrics.get("actual_artifact_type") if stored_contract else derived_actual
    match = (
        bool(metrics.get("artifact_type_match"))
        if stored_contract
        else artifact_type_matches(expected, actual)
    )
    if stored_contract:
        discrepancies = []
        if expected != derived_expected:
            discrepancies.append(
                f"expected artifact stored={expected} derived={derived_expected}"
            )
        if actual != derived_actual:
            discrepancies.append(
                f"actual artifact stored={actual} derived={derived_actual}"
            )
        if match != artifact_type_matches(expected, actual):
            discrepancies.append("artifact_type_match disagrees with stored expected/actual")
        if discrepancies:
            raise ValueError(
                f"{case.get('case_id')}: " + "; ".join(discrepancies)
            )
    constraints_ok = not bool(metrics.get("constraint_tree_missing") or [])
    grounding_ok = metrics.get("grounding_ok") is not False
    status_success = str(((payload.get("turns") or [{}])[-1].get("status") or "")) in {
        "completed", "completed_with_warnings"
    }
    task_judge = task_completion_evaluation(
        expected,
        actual,
        constraints_passed=constraints_ok,
        grounding_passed=grounding_ok,
        status_success=status_success,
        reply_present=bool((payload.get("turns") or [{}])[-1].get("reply_text")),
    )
    original = ((payload.get("evaluation") or {}).get("independent_judge") or {})
    route = itinerary_judge_route(expected, actual, system_error=bool(payload.get("errors")))
    if route["status"] in {"not_applicable", "not_run"}:
        judge = {**original, **route, "expected_artifact_type": expected, "actual_artifact_type": actual}
    elif actual == "partial_itinerary" and original.get("rubric") != "partial_itinerary":
        judge = {
            "status": "not_run",
            "reason": "partial_rubric_not_run",
            "rubric": "partial_itinerary",
            "diagnostic_only": expected == "full_itinerary",
            "expected_artifact_type": expected,
            "actual_artifact_type": actual,
        }
    else:
        judge = {
            **original,
            "rubric": original.get("rubric") or "full_itinerary",
            "expected_artifact_type": expected,
            "actual_artifact_type": actual,
        }
    strict = bool(metrics.get("strict_task_success")) and match
    failure_reason = "missing_expected_artifact" if expected in ITINERARY_TYPES and not match else None
    return {
        "expected_artifact_type": expected,
        "actual_artifact_type": actual,
        "artifact_type_match": match,
        "strict_task_success": strict,
        "failure_reason": failure_reason,
        "task_completion_judge": task_judge,
        "independent_judge": judge,
        "metric_source": "stored" if stored_contract else "legacy_evaluator_correction",
    }


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / "dev34_case_report_artifact.json"
    html_output = run_dir / "dev34_case_report.html"
    markdown_output = (
        args.markdown_out.resolve()
        if args.markdown_out
        else run_dir / "dev34_case_report.md"
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    if (
        args.plan_start_strict_count is not None
        and not 0 <= args.plan_start_strict_count <= args.expected_case_count
    ):
        raise SystemExit("--plan-start-strict-count must be within the expected case count")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows_by_id = {row["case_id"]: row for row in summary["rows"]}
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "cases").glob("*.json"))
    ]
    payloads.sort(key=lambda item: item["case"]["case_id"])
    if len(payloads) != args.expected_case_count:
        raise SystemExit(
            f"expected {args.expected_case_count} case artifacts, got {len(payloads)}"
        )

    classified_by_id = {
        payload["case"]["case_id"]: classified_evaluation(payload) for payload in payloads
    }
    for case_id, classified in classified_by_id.items():
        row = rows_by_id.get(case_id)
        if row is None:
            raise SystemExit(f"summary row missing for {case_id}")
        if classified["metric_source"] == "legacy_evaluator_correction":
            continue
        for key in (
            "expected_artifact_type", "actual_artifact_type",
            "artifact_type_match", "strict_task_success",
        ):
            if key in row and row.get(key) != classified.get(key):
                raise SystemExit(
                    f"summary/case disagreement for {case_id}.{key}: "
                    f"summary={row.get(key)!r} case={classified.get(key)!r}"
                )
    report_outputs = []
    for payload in payloads:
        classified = classified_by_id[payload["case"]["case_id"]]
        copied = {**payload, "evaluation": {**(payload.get("evaluation") or {})}}
        copied["evaluation"]["rule_metrics"] = {
            **((copied["evaluation"].get("rule_metrics") or {})),
            **{key: classified[key] for key in (
                "expected_artifact_type", "actual_artifact_type", "artifact_type_match",
                "strict_task_success", "failure_reason", "task_completion_judge",
            )},
        }
        copied["evaluation"]["task_completion_judge"] = classified["task_completion_judge"]
        copied["evaluation"]["independent_judge"] = classified["independent_judge"]
        report_outputs.append(copied)
    artifact_summary = aggregate_artifact_metrics(report_outputs)

    source = {
        "id": "dev34-run",
        "label": "Dev34 run artifacts",
        "path": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "query": {
            "description": "Frozen harness outputs, execution traces, final itineraries, and independent Judge results.",
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "WITH cases AS (SELECT * FROM read_json_auto("
                f"'{run_dir}/cases/*.json', "
                "union_by_name=true)) "
                "SELECT case.case_id AS case_id, "
                "evaluation.rule_metrics.strict_task_success AS strict_pass, "
                "evaluation.independent_judge.status AS judge_status, "
                "evaluation.independent_judge.total_score AS judge_score, "
                "evaluation.independent_judge.reasonable AS judge_pass "
                "FROM cases ORDER BY case_id"
            ),
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "filters": ["split=dev", f"{len(payloads)} cases", "repeat=1"],
            "tables_used": ["summary.json", "cases/*.json"],
            "metric_definitions": [
                "strict_pass: harness strict_task_success is true",
                "judge_pass: independent Judge reasonable is true",
                "judge_score: independent Judge total score on a 0-100 rubric",
            ],
        },
    }

    overview: list[dict] = []
    judge_scores: list[dict] = []
    strict_count = sum(classified["strict_task_success"] for classified in classified_by_id.values())
    raw_strict_count = sum(bool(row.get("strict_task_success")) for row in rows_by_id.values())
    correction_count = sum(
        item["metric_source"] == "legacy_evaluator_correction"
        for item in classified_by_id.values()
    )
    plan_start_strict_count = args.plan_start_strict_count
    full_count = sum(
        item["expected_artifact_type"] == "full_itinerary" for item in classified_by_id.values()
    )
    non_count = sum(
        item["expected_artifact_type"] in NON_ITINERARY_TYPES for item in classified_by_id.values()
    )
    judge_applicable = artifact_summary["itinerary_judge_applicable_count"]
    judge_completed = artifact_summary["itinerary_judge_completed_count"]
    full_average = artifact_summary["itinerary_judge_average_by_rubric"].get("full_itinerary")
    blocks: list[dict] = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Dev34 用户、行程与评分",
        },
        {
            "id": "executive-summary",
            "type": "markdown",
            "sourceId": "dev34-run",
            "body": (
                "## Executive Summary\n\n"
                f"- **报告共 {len(payloads)} 个 case。** 每个 case 按“用户 → 具体行程 → 评分”排列。\n"
                f"- **Overall strict success：{strict_count}/{len(payloads)}。**\n"
                f"- **原始历史 run strict：{raw_strict_count}/{len(payloads)}。**\n"
                + (
                    f"- **计划启动时 Artifact 合同审计：{plan_start_strict_count}/{len(payloads)}。**\n"
                    if plan_start_strict_count is not None
                    else ""
                )
                + f"- **当前 evaluator 对同一历史 run 的回放审计：{strict_count}/{len(payloads)}；"
                f"合同重解释 case：{correction_count}。** 以上口径变化均为 evaluator correction，"
                "不计产品提升，也不表示被测产品重新运行后退步。\n"
                f"- **Artifact type match：{percent(artifact_summary['artifact_type_match_rate'])}。**\n"
                f"- **Full-itinerary success：{percent(artifact_summary['full_itinerary_success_rate'])}（{full_count} 个适用 case）。**\n"
                f"- **Non-itinerary success：{percent(artifact_summary['non_itinerary_success_rate'])}（{non_count} 个适用 case）。**\n"
                f"- **Expected itinerary missing rate：{percent(artifact_summary['expected_itinerary_missing_rate'])}。**\n"
                f"- **Itinerary Judge：applicable {judge_applicable}，completed {judge_completed}，full-rubric average {full_average if full_average is not None else 'N/A'}。**\n"
                f"- **Task-completion Judge：applicable {artifact_summary['task_completion_judge_applicable_count']}，pass rate {artifact_summary['task_completion_judge_pass_rate'] if artifact_summary['task_completion_judge_pass_rate'] is not None else 'N/A'}。**\n"
                f"- **Judge 状态：{artifact_summary['judge_status_distribution']}。**"
            ),
        },
    ]

    cards = [
        {
            "id": "card-cases",
            "description": "冻结 Dev case 总数",
            "dataset": "headline",
            "sourceId": "dev34-run",
            "metrics": [{"label": "Cases", "field": "cases", "format": "number"}],
        },
        {
            "id": "card-strict",
            "description": "Harness strict_task_success",
            "dataset": "headline",
            "sourceId": "dev34-run",
            "metrics": [{"label": "Strict pass", "field": "strict_pass", "format": "number"}],
        },
        {
            "id": "card-judge",
            "description": "Judge reasonable / applicable",
            "dataset": "headline",
            "sourceId": "dev34-run",
            "metrics": [
                {"label": "Judge pass", "field": "judge_pass", "format": "number"},
                {"label": "Applicable", "field": "judge_applicable", "format": "number"},
            ],
        },
        {
            "id": "card-score",
            "description": "仅 full-itinerary rubric 的 Judge 平均分，不混合 Partial rubric",
            "dataset": "headline",
            "sourceId": "dev34-run",
            "metrics": [{"label": "Judge avg", "field": "judge_avg", "format": "number"}],
        },
    ]
    blocks.append(
        {
            "id": "headline-metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        }
    )
    blocks.append(
        {
            "id": "score-chart-narrative",
            "type": "markdown",
            "body": (
                "## Judge 分数覆盖范围\n\n"
                f"下图只展示 {judge_completed} 个已完成 case 的 Judge 分数；其余 case 的状态应按 not_applicable、not_run、missing 或 error 解读。"
            ),
        }
    )

    for payload in payloads:
        case = payload["case"]
        case_id = case["case_id"]
        row = rows_by_id[case_id]
        metrics = payload["evaluation"]["rule_metrics"]
        classified = classified_by_id[case_id]
        judge = classified["independent_judge"]
        itinerary = (payload.get("final_itinerary") or {}).get("itinerary") or {}
        budget = (payload.get("final_itinerary") or {}).get("budget_plan") or {}
        lodging = (payload.get("final_itinerary") or {}).get("lodging_plan") or {}
        critic = (payload.get("final_itinerary") or {}).get("critic") or {}
        score = judge.get("total_score")
        overview.append(
            {
                "case_id": case_id,
                "category": case.get("category") or "",
                "city": itinerary.get("city") or (payload.get("final_profile") or {}).get("destination") or "",
                "strict": yes_no(classified["strict_task_success"]),
                "expected_artifact_type": classified["expected_artifact_type"],
                "actual_artifact_type": classified["actual_artifact_type"],
                "artifact_type_match": yes_no(classified["artifact_type_match"]),
                "failure_reason": classified["failure_reason"],
                "gating": yes_no(metrics.get("gating_passed")),
                "hard": yes_no(metrics.get("hard_constraints_annotation_ok")),
                "critic": yes_no(metrics.get("critic_passed")),
                "judge_status": judge.get("status") or "missing",
                "judge_score": score,
                "judge_pass": reviewed_judge_label(judge),
                "latency_s": round(row.get("duration_ms", 0) / 1000, 3),
                "tokens": row.get("total_tokens", 0),
            }
        )
        if judge.get("status") == "ok" and judge.get("rubric") == "full_itinerary":
            judge_scores.append(
                {
                    "case_id": case_id,
                    "score": score,
                    "result": "pass" if judge.get("reasonable") else "fail",
                }
            )

        turns = case.get("turns") or []
        reply_text = ((payload.get("turns") or [{}])[-1].get("reply_text") or "未保存 Agent 文本回答。")
        judge_details = f"- 状态：**{reviewed_judge_label(judge)}**\n- 说明：{status_note(judge)}"
        if judge.get("status") == "ok":
            dimensions = judge.get("scores") or {}
            dimension_text = "｜".join(
                f"{SCORE_LABELS.get(name, name)} {value}" for name, value in dimensions.items()
            )
            critical = [item for item in (judge.get("critical_issues") or []) if item.get("severity") in {"critical", "error"}]
            warnings = [item for item in (judge.get("critical_issues") or []) if item.get("severity") == "warning"]
            critical_text = "\n".join(
                f"- {item.get('evidence') or item.get('code')}" for item in critical[:3]
            ) or "- 无"
            warning_text = "\n".join(
                f"- {item.get('evidence') or item.get('code')}" for item in warnings[:3]
            ) or "- 无"
            judge_details = (
                f"- 总分：**{score}/100**\n"
                f"- 结论：**{status_note(judge)}**\n"
                f"- Rubric：`{judge.get('rubric') or 'full_itinerary'}`\n"
                f"- 分项：{dimension_text}\n"
                f"- 主要失败原因（error/critical）：\n{critical_text}\n"
                f"- 可选改进（warning）：\n{warning_text}"
            )
        hotel = lodging.get("hotel") or {}
        lodging_text = (
            (f"{hotel.get('name')}，{hotel.get('address') or hotel.get('area') or '地址未提供'}" if hotel else "未选定")
        )
        lodging_required = bool(lodging.get("required") or lodging.get("explicit_requirement"))
        expected_budget = budget.get("total_expected_cny", budget.get("total_low_cny"))
        budget_limit = budget.get("user_limit_cny")
        budget_text = (
            (f"预计 ¥{expected_budget}" if expected_budget is not None else "未提供")
            + (f"，预算上限 ¥{budget_limit}" if budget_limit is not None else "")
            + ("，预计不超预算" if budget.get("within_user_limit") is True else "")
        )
        issues = critic.get("issues") or []
        critic_text = "；".join(str(item.get("message") or item.get("code")) for item in issues)
        has_itinerary = bool(itinerary.get("days"))
        if has_itinerary:
            answer_parts = [itinerary_markdown(payload)]
            if lodging_required or hotel:
                answer_parts.append(f"**住宿**：{lodging_text}")
            if expected_budget is not None or budget_limit is not None:
                answer_parts.append(f"**预算**：{budget_text}")
            answer_body = "\n\n".join(answer_parts)
        else:
            answer_body = quote_lines(reply_text)
        harness_summary = (
            f"- Harness strict：**{yes_no(classified['strict_task_success'])}**\n"
            f"- Expected artifact：`{classified['expected_artifact_type']}`\n"
            f"- Actual artifact：`{classified['actual_artifact_type'] or 'missing'}`\n"
            f"- Artifact type match：**{yes_no(classified['artifact_type_match'])}**\n"
            f"- 硬约束检查：**{yes_no(metrics.get('hard_constraints_annotation_ok'))}**\n"
            f"- 行程检查：**{yes_no(metrics.get('critic_passed'))}**"
        )
        critical_critic = [item for item in issues if item.get("severity") in {"error", "critical"}]
        warning_critic = [item for item in issues if item.get("severity") == "warning"]
        if critical_critic:
            harness_summary += "\n- 主要失败原因：" + "；".join(
                str(item.get("message") or item.get("code")) for item in critical_critic
            )
        if warning_critic:
            harness_summary += "\n- 可选改进：" + "；".join(
                str(item.get("message") or item.get("code")) for item in warning_critic
            )
        block_body = (
            f"## {case_id} · {case.get('metadata', {}).get('title') or case.get('category') or '未命名'}\n\n"
            f"### 用户\n\n{user_dialogue(turns)}\n\n"
            f"### 回答\n\n{answer_body}\n\n"
            f"### 评分\n\n{harness_summary}\n\n"
            f"**LLM-as-Judge**\n\n{judge_details}"
        )
        blocks.append(
            {
                "id": f"case-{case_id}",
                "type": "markdown",
                "sourceId": "dev34-run",
                "body": block_body,
            }
        )

    judge_avg = mean(row["score"] for row in judge_scores) if judge_scores else None
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": [
                {
                    "cases": len(payloads),
                    "raw_strict_pass": raw_strict_count,
                    "plan_start_strict_pass": plan_start_strict_count,
                    "current_evaluator_replay_strict_pass": strict_count,
                    "evaluator_correction_case_count": correction_count,
                    "strict_pass": strict_count,
                    "artifact_match_rate": artifact_summary["artifact_type_match_rate"],
                    "full_itinerary_success_rate": artifact_summary["full_itinerary_success_rate"],
                    "non_itinerary_success_rate": artifact_summary["non_itinerary_success_rate"],
                    "expected_itinerary_missing_rate": artifact_summary["expected_itinerary_missing_rate"],
                    "task_completion_pass_rate": artifact_summary["task_completion_judge_pass_rate"],
                    "judge_pass": sum(item["judge_pass"] == "通过" for item in overview),
                    "judge_applicable": judge_applicable,
                    "judge_completed": judge_completed,
                    "judge_avg": round(judge_avg, 1) if judge_avg is not None else None,
                    "judge_status_distribution": artifact_summary["judge_status_distribution"],
                    "judge_reason_distribution": artifact_summary["judge_reason_distribution"],
                }
            ],
            "overview": overview,
            "judge_scores": judge_scores,
        },
    }
    chart = {
        "id": "judge-score-chart",
        "title": "Applicable cases 的 Judge 总分",
        "subtitle": f"{judge_completed} 个已完成 case，满分 100；70 分线仅为 Judge rubric 阈值",
        "type": "bar",
        "dataset": "judge_scores",
        "sourceId": "dev34-run",
        "encodings": {
            "x": {"field": "case_id", "type": "nominal", "label": "Case"},
            "y": {"field": "score", "type": "quantitative", "label": "Judge score"},
            "color": {"field": "result", "type": "nominal", "label": "Judge result"},
        },
        "valueFormat": "number",
        "maxRows": 34,
        "layout": "full",
    }
    blocks.insert(5, {"id": "judge-score-block", "type": "chart", "chartId": chart["id"]})
    blocks.insert(
        6,
        {
            "id": "overview-heading",
            "type": "markdown",
            "body": (
                f"## {len(payloads)} 个 case 总览\n\n"
                "下表用于快速定位。正文按“用户、回答、评分”展示每个 case。"
            ),
        },
    )
    table = {
        "id": "case-overview",
        "title": "Dev34 case 评分总览",
        "subtitle": "冻结版本、每 case 单次运行；按 case_id 排序",
        "dataset": "overview",
        "sourceId": "dev34-run",
        "defaultSort": {"field": "case_id", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "case_id", "label": "Case"},
            {"field": "city", "label": "城市"},
            {"field": "strict", "label": "Strict"},
            {"field": "expected_artifact_type", "label": "Expected artifact"},
            {"field": "actual_artifact_type", "label": "Actual artifact"},
            {"field": "artifact_type_match", "label": "Artifact match"},
            {"field": "judge_score", "label": "Judge score", "format": "number"},
            {"field": "judge_pass", "label": "Judge 结论"},
        ],
    }
    blocks.insert(7, {"id": "overview-table-block", "type": "table", "tableId": table["id"]})
    blocks.insert(
        8,
        {
            "id": "detail-heading",
            "type": "markdown",
            "body": (
                "## 每个 case 的用户、回答和评分\n\n"
                "有行程时直接列出每天的时间与地点；没有行程时显示 Agent 实际回复。"
            ),
        },
    )
    block_by_id = {block["id"]: block for block in blocks}
    case_blocks = [block for block in blocks if block["id"].startswith("case-")]
    blocks = [
        block_by_id["title"],
        block_by_id["executive-summary"],
        block_by_id["headline-metrics"],
        block_by_id["score-chart-narrative"],
        block_by_id["judge-score-block"],
        block_by_id["overview-heading"],
        block_by_id["overview-table-block"],
        block_by_id["detail-heading"],
        *case_blocks,
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Dev34 用户、行程与评分",
            "description": f"{len(payloads)} 个 Dev case，按用户、具体行程和评分展示。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": [chart],
            "tables": [table],
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": snapshot,
        "sources": [source],
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_lines = [
        "# Dev34 用户、行程与评分",
        "",
        f"> 单次运行的 {len(payloads)} 个 Dev cases。每个 case 仅保留用户问题、Agent 的具体行程和评测结论。",
        "",
        "## 总览",
        "",
        f"- 当前 evaluator 对同一历史 run 的回放审计：{strict_count}/{len(payloads)}",
        f"- 原始历史 run strict：{raw_strict_count}/{len(payloads)}",
        *(
            [f"- 计划启动时 Artifact 合同审计：{plan_start_strict_count}/{len(payloads)}"]
            if plan_start_strict_count is not None
            else []
        ),
        f"- evaluator correction case 数：{correction_count}（仅表示口径重解释，不计产品提升，也不表示产品重新运行后退步）",
        f"- Artifact type match：{percent(artifact_summary['artifact_type_match_rate'])}",
        f"- Full-itinerary success：{percent(artifact_summary['full_itinerary_success_rate'])}",
        f"- Non-itinerary success：{percent(artifact_summary['non_itinerary_success_rate'])}",
        f"- Expected itinerary missing rate：{percent(artifact_summary['expected_itinerary_missing_rate'])}",
        f"- Itinerary Judge：applicable {judge_applicable}，completed {judge_completed}，full-rubric average {round(judge_avg, 1) if judge_avg is not None else 'N/A'}",
        f"- Task-completion Judge：applicable {artifact_summary['task_completion_judge_applicable_count']}，pass rate {artifact_summary['task_completion_judge_pass_rate']}",
        f"- Judge 状态分布：{artifact_summary['judge_status_distribution']}",
        f"- Judge 原因分布：{artifact_summary['judge_reason_distribution']}",
        "",
        "| Case | 城市 | Expected artifact | Actual artifact | Match | Harness strict | Judge 分数 | Judge 结论 |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in overview:
        judge_score = item["judge_score"] if item["judge_score"] is not None else "—"
        markdown_lines.append(
            f"| `{item['case_id']}` | {item['city'] or '—'} | {item['expected_artifact_type']} | "
            f"{item['actual_artifact_type'] or 'missing'} | {item['artifact_type_match']} | {item['strict']} | "
            f"{judge_score} | {item['judge_pass']} |"
        )
    markdown_lines.extend(["", "## 逐 Case 记录", ""])
    markdown_lines.extend(block["body"] + "\n" for block in case_blocks)
    markdown_lines.extend(
        [
            "## 说明",
            "",
            "- Harness 不是百分制，因此仅展示 strict、硬约束检查和行程检查。",
            "- Judge 分数仅在同一 Rubric 内汇总；Full 与 Partial 百分制不直接混合平均。",
            "- not_applicable 表示任务本来不需要行程 Judge；not_run 表示缺少预期产物或系统失败；missing/error 表示 Judge 自身未完成。",
            "- 行程、路线、住宿和预算来自冻结运行快照，不代表实时可预订状态。",
            "",
        ]
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(markdown_lines), encoding="utf-8")
    markdown_text = "\n".join(markdown_lines)
    try:
        from markdown_it import MarkdownIt

        rendered = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown_text)
    except ImportError:
        rendered = f"<pre>{escape(markdown_text)}</pre>"
    html_document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(artifact['manifest']['title'])}</title>
  <style>
    :root {{ color-scheme: light; --ink: #17202a; --muted: #667085; --line: #d0d5dd; --accent: #175cd3; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f8fafc; color: var(--ink); font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 32px auto; padding: 36px 44px; background: white; border: 1px solid #eaecf0; border-radius: 16px; box-shadow: 0 8px 30px rgb(16 24 40 / 8%); }}
    h1 {{ margin-top: 0; font-size: 32px; }} h2 {{ margin-top: 48px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }}
    h3 {{ margin-top: 28px; }} blockquote {{ margin: 12px 0; padding: 8px 16px; color: #344054; background: #f9fafb; border-left: 4px solid #84adff; }}
    table {{ width: 100%; border-collapse: collapse; display: block; overflow-x: auto; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border: 1px solid var(--line); text-align: left; white-space: nowrap; }} th {{ background: #f2f4f7; }}
    code {{ color: var(--accent); background: #eff4ff; padding: 2px 5px; border-radius: 4px; }}
    li {{ margin: 4px 0; }} strong {{ color: #101828; }}
    @media (max-width: 720px) {{ main {{ width: 100%; margin: 0; padding: 20px; border: 0; border-radius: 0; }} h1 {{ font-size: 26px; }} }}
  </style>
</head>
<body><main>{rendered}</main></body>
</html>
"""
    html_output.write_text(html_document, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "markdown_output": str(markdown_output),
        "html_output": str(html_output),
        "cases": len(payloads),
        "applicable_judges": judge_applicable,
        "completed_judges": judge_completed,
        "bytes": output.stat().st_size,
        "blocks": len(blocks),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
