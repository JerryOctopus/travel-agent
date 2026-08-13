"""Build an auditable per-case report artifact from the frozen Dev34 run."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from datetime import datetime, timedelta


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data/eval/product/runs/dev34_frozen_full_20260813a"
OUT = RUN_DIR / "dev34_case_report_artifact.json"
MARKDOWN_OUT = ROOT / "docs/DEV34_CASE_REPORT.md"


def yes_no(value: object) -> str:
    if value is True:
        return "通过"
    if value is False:
        return "失败"
    return "不适用"


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


def status_note(case_id: str, judge: dict) -> str:
    if case_id == "dev_018":
        return "人工复核为 Judge 误判：110 米路段的交通标签有瑕疵，但不足以判整份行程失败。"
    if judge.get("status") == "not_applicable":
        return "未评分：该 case 没有形成可供行程 Judge 评分的完整计划。"
    if judge.get("reasonable") is True:
        return "通过"
    return "未通过"


def reviewed_judge_label(case_id: str, judge: dict) -> str:
    if case_id == "dev_018":
        return "Judge 误判"
    if judge.get("status") != "ok":
        return "未评分"
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


def main() -> None:
    summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))
    rows_by_id = {row["case_id"]: row for row in summary["rows"]}
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((RUN_DIR / "cases").glob("*.json"))
    ]
    payloads.sort(key=lambda item: item["case"]["case_id"])
    if len(payloads) != 34:
        raise SystemExit(f"expected 34 case artifacts, got {len(payloads)}")

    source = {
        "id": "frozen-dev34",
        "label": "Frozen Dev34 run artifacts",
        "path": "data/eval/product/runs/dev34_frozen_full_20260813a",
        "query": {
            "description": "Frozen harness outputs, execution traces, final itineraries, and independent Judge results.",
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "WITH cases AS (SELECT * FROM read_json_auto("
                "'data/eval/product/runs/dev34_frozen_full_20260813a/cases/*.json', "
                "union_by_name=true)) "
                "SELECT case.case_id AS case_id, "
                "evaluation.rule_metrics.strict_task_success AS strict_pass, "
                "evaluation.independent_judge.status AS judge_status, "
                "evaluation.independent_judge.total_score AS judge_score, "
                "evaluation.independent_judge.reasonable AS judge_pass "
                "FROM cases ORDER BY case_id"
            ),
            "executed_at": "2026-08-13",
            "filters": ["split=dev", "frozen candidate", "34 cases", "repeat=1"],
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
    blocks: list[dict] = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Dev34 用户、行程与评分",
        },
        {
            "id": "executive-summary",
            "type": "markdown",
            "sourceId": "frozen-dev34",
            "body": (
                "## Executive Summary\n\n"
                "- **报告共 34 个 case。** 每个 case 按“用户 → 具体行程 → 评分”排列。\n"
                "- **Harness strict 通过 20/34。**\n"
                "- **19 个 case 获得 Judge 评分。** 平均 65.1 分，其中 6 个通过；另外 15 个未形成可评分的完整行程。\n"
                "- **dev_018 已标为 Judge 误判。**"
            ),
        },
    ]

    cards = [
        {
            "id": "card-cases",
            "description": "冻结 Dev case 总数",
            "dataset": "headline",
            "sourceId": "frozen-dev34",
            "metrics": [{"label": "Cases", "field": "cases", "format": "number"}],
        },
        {
            "id": "card-strict",
            "description": "Harness strict_task_success",
            "dataset": "headline",
            "sourceId": "frozen-dev34",
            "metrics": [{"label": "Strict pass", "field": "strict_pass", "format": "number"}],
        },
        {
            "id": "card-judge",
            "description": "Judge reasonable / applicable",
            "dataset": "headline",
            "sourceId": "frozen-dev34",
            "metrics": [
                {"label": "Judge pass", "field": "judge_pass", "format": "number"},
                {"label": "Applicable", "field": "judge_applicable", "format": "number"},
            ],
        },
        {
            "id": "card-score",
            "description": "19 个可评 case 的 Judge 平均分",
            "dataset": "headline",
            "sourceId": "frozen-dev34",
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
                "下图只展示 19 个 applicable case 的 Judge 分数；没有柱子的 case 是 not_applicable，不能把它们当成 Judge pass。"
            ),
        }
    )

    for payload in payloads:
        case = payload["case"]
        case_id = case["case_id"]
        row = rows_by_id[case_id]
        metrics = payload["evaluation"]["rule_metrics"]
        judge = payload["evaluation"].get("independent_judge") or {}
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
                "strict": yes_no(metrics.get("strict_task_success")),
                "gating": yes_no(metrics.get("gating_passed")),
                "hard": yes_no(metrics.get("hard_constraints_annotation_ok")),
                "critic": yes_no(metrics.get("critic_passed")),
                "judge_status": judge.get("status") or "missing",
                "judge_score": score,
                "judge_pass": reviewed_judge_label(case_id, judge),
                "latency_s": round(row.get("duration_ms", 0) / 1000, 3),
                "tokens": row.get("total_tokens", 0),
            }
        )
        if judge.get("status") == "ok":
            judge_scores.append(
                {
                    "case_id": case_id,
                    "score": score,
                    "result": "pass" if judge.get("reasonable") else "fail",
                }
            )

        turns = case.get("turns") or []
        reply_text = ((payload.get("turns") or [{}])[-1].get("reply_text") or "未保存 Agent 文本回答。")
        judge_details = "**未评分**：没有生成可供行程 Judge 评分的完整计划。"
        if judge.get("status") == "ok":
            dimensions = judge.get("scores") or {}
            dimension_text = "｜".join(
                f"{SCORE_LABELS.get(name, name)} {value}" for name, value in dimensions.items()
            )
            problems = judge_problem_lines(judge)
            problem_text = "\n".join(f"- {problem}" for problem in problems) or "- 无主要问题"
            judge_details = (
                f"- 总分：**{score}/100**\n"
                f"- 结论：**{status_note(case_id, judge)}**\n"
                f"- 分项：{dimension_text}\n"
                f"- 主要问题：\n{problem_text}"
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
            f"- Harness strict：**{yes_no(metrics.get('strict_task_success'))}**\n"
            f"- 硬约束检查：**{yes_no(metrics.get('hard_constraints_annotation_ok'))}**\n"
            f"- 行程检查：**{yes_no(metrics.get('critic_passed'))}**"
        )
        if critic_text:
            harness_summary += f"\n- Harness 主要问题：{critic_text}"
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
                "sourceId": "frozen-dev34",
                "body": block_body,
            }
        )

    judge_avg = mean(row["score"] for row in judge_scores)
    snapshot = {
        "version": 1,
        "generatedAt": "2026-08-13T00:00:00+08:00",
        "status": "ready",
        "datasets": {
            "headline": [
                {
                    "cases": 34,
                    "strict_pass": sum(item["strict"] == "通过" for item in overview),
                    "judge_pass": sum(item["judge_pass"] == "通过" for item in overview),
                    "judge_applicable": len(judge_scores),
                    "judge_avg": round(judge_avg, 1),
                }
            ],
            "overview": overview,
            "judge_scores": judge_scores,
        },
    }
    chart = {
        "id": "judge-score-chart",
        "title": "Applicable cases 的 Judge 总分",
        "subtitle": "19 个可评 case，满分 100；70 分线仅为 Judge rubric 阈值",
        "type": "bar",
        "dataset": "judge_scores",
        "sourceId": "frozen-dev34",
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
                "## 34 个 case 总览\n\n"
                "下表用于快速定位。正文按“用户、回答、评分”展示每个 case。"
            ),
        },
    )
    table = {
        "id": "case-overview",
        "title": "Dev34 case 评分总览",
        "subtitle": "冻结版本、每 case 单次运行；按 case_id 排序",
        "dataset": "overview",
        "sourceId": "frozen-dev34",
        "defaultSort": {"field": "case_id", "direction": "asc"},
        "density": "dense",
        "layout": "full",
        "columns": [
            {"field": "case_id", "label": "Case"},
            {"field": "city", "label": "城市"},
            {"field": "strict", "label": "Strict"},
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
            "description": "34 个冻结 Dev case，按用户、具体行程和评分展示。",
            "generatedAt": "2026-08-13T00:00:00+08:00",
            "cards": cards,
            "charts": [chart],
            "tables": [table],
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": snapshot,
        "sources": [source],
    }
    OUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_lines = [
        "# Dev34 用户、行程与评分",
        "",
        "> 冻结版本的 34 个 Dev cases。每个 case 仅保留用户问题、Agent 的具体行程和评测结论。",
        "",
        "## 总览",
        "",
        "- Harness strict：20/34 通过",
        "- LLM-as-Judge：19 个 case 获得评分，平均 65.1 分，6 个通过",
        "- 15 个 case 未形成可供行程 Judge 评分的完整计划",
        "- `dev_018` 经人工复核标记为 Judge 误判",
        "",
        "| Case | 城市 | Harness strict | Judge 分数 | Judge 结论 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for item in overview:
        judge_score = item["judge_score"] if item["judge_score"] is not None else "—"
        markdown_lines.append(
            f"| `{item['case_id']}` | {item['city'] or '—'} | {item['strict']} | "
            f"{judge_score} | {item['judge_pass']} |"
        )
    markdown_lines.extend(["", "## 逐 Case 记录", ""])
    markdown_lines.extend(block["body"] + "\n" for block in case_blocks)
    markdown_lines.extend(
        [
            "## 说明",
            "",
            "- Harness 不是百分制，因此仅展示 strict、硬约束检查和行程检查。",
            "- Judge 分数满分 100；未评分不等于通过。",
            "- 行程、路线、住宿和预算来自冻结运行快照，不代表实时可预订状态。",
            "",
        ]
    )
    MARKDOWN_OUT.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "markdown_output": str(MARKDOWN_OUT),
        "cases": len(payloads),
        "applicable_judges": len(judge_scores),
        "bytes": OUT.stat().st_size,
        "blocks": len(blocks),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
