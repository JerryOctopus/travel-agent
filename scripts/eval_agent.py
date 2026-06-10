"""Agent 级 + 规划级 + 闭环级评估（对应 PROJECT_PLAN M8 与「评估方案」）。

分层产出指标：

- 理解层：目的地 / 天数抽取准确率、追问触达率；
- Agent 层：任务完成率（是否产出可用行程）、平均工具调用步数、工具成功率；
- 规划层：critic 通过率、约束满足；
- 闭环层（最重要）：plan 原始稿 vs revise 修正稿的违规项下降幅度，量化 critic 价值。

注意（如实标注局限）：NL 用例为人工构造的小样本（见 eval/cases.json），
闭环用例为带约束的合成 profile；指标用于自检与回归，不代表真实分布上的泛化。
评估在离线确定性路径（无 LLM key）下运行，保证可复现。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.agent.runtime import run_turn
from travel_agent.agent.session import build_session
from travel_agent.planning_subgraph import plan_and_critique
from travel_agent.recommendation import score_pois
from travel_agent.schemas import TravelProfile
from travel_agent.providers import build_tool_provider

DEFAULT_POI_PATH = ROOT / "data" / "seed" / "pois.json"


# --------------------------------------------------------------------------- #
# 闭环用例：带约束的合成 profile，用于量化 critic→reviser 的价值
# --------------------------------------------------------------------------- #
CLOSED_LOOP_CASES: list[dict[str, Any]] = [
    {
        "id": "hangzhou_1d_museum_uncovered",
        "profile": {"destination": "杭州", "days": 1, "interests": ["museum"]},
    },
    {
        "id": "hangzhou_1d_multi_interest",
        "profile": {
            "destination": "杭州",
            "days": 1,
            "interests": ["history", "food", "nature", "museum"],
        },
    },
    {
        "id": "shanghai_2d_multi_interest",
        "profile": {
            "destination": "上海",
            "days": 2,
            "interests": ["history", "food", "nature", "museum"],
        },
    },
    {
        "id": "chengdu_1d_food_museum",
        "profile": {"destination": "成都", "days": 1, "interests": ["food", "museum"]},
    },
]


def _round(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


def run_nl_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for case in cases:
        ctx = build_session(persist=False, poi_path=DEFAULT_POI_PATH)
        reply = run_turn(case["query"], ctx=ctx)
        itinerary_artifact = ctx.store.latest("itinerary")
        expected_clarification = bool(case.get("expected_clarification", False))

        produced = itinerary_artifact is not None
        critic = (itinerary_artifact or {}).get("critic", {})
        rows.append(
            {
                "id": case["id"],
                "city_ok": _opt_eq(ctx.profile.destination, case.get("expected_city")),
                "days_ok": _opt_eq(ctx.profile.days, case.get("expected_days")),
                "clarification_ok": (
                    reply.clarification if expected_clarification else not reply.clarification
                ),
                "itinerary_produced": (None if expected_clarification else produced),
                "critic_passed": (critic.get("passed") if produced else None),
                "tool_steps": len(reply.tool_trace),
            }
        )

    completable = [r for r in rows if r["itinerary_produced"] is not None]
    return {
        "sample_size": len(rows),
        "city_accuracy": _rate(rows, "city_ok"),
        "days_accuracy": _rate(rows, "days_ok"),
        "clarification_accuracy": _rate(rows, "clarification_ok"),
        "task_completion_rate": _rate(completable, "itinerary_produced"),
        "critic_pass_rate": _rate(rows, "critic_passed"),
        "avg_tool_steps": _round(
            sum(r["tool_steps"] for r in rows) / len(rows) if rows else None, 2
        ),
        "cases": rows,
    }


def run_closed_loop_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    provider = build_tool_provider(DEFAULT_POI_PATH)
    rows = []
    total_before = 0
    total_after = 0
    for case in cases:
        profile = TravelProfile(**case["profile"])
        candidates = provider.search_pois(city=profile.destination or "", max_results=50)
        ranked = score_pois(candidates, profile)
        result = plan_and_critique(ranked, profile, route_estimator=provider)
        total_before += result.original_issue_count
        total_after += result.final_issue_count
        rows.append(
            {
                "id": case["id"],
                "original_issues": result.original_issue_count,
                "final_issues": result.final_issue_count,
                "iterations": result.iterations,
                "passed": result.critic_result.passed,
                "revision_notes": result.revision_notes,
            }
        )

    reduction = None
    if total_before > 0:
        reduction = round((total_before - total_after) / total_before, 4)
    return {
        "sample_size": len(rows),
        "total_issues_before": total_before,
        "total_issues_after": total_after,
        "issue_reduction_rate": reduction,
        "final_pass_rate": _rate(rows, "passed"),
        "cases": rows,
    }


def _opt_eq(actual: Any, expected: Any) -> bool | None:
    if expected is None:
        return None
    return actual == expected


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def build_report(nl: dict[str, Any], closed: dict[str, Any]) -> str:
    lines = [
        "# 评估报告（EVALUATION）",
        "",
        "> 自动生成：`python scripts/eval_agent.py --write-report`。",
        "> 评估在离线确定性路径下运行（无 LLM key），保证可复现。",
        "",
        "## 样本与局限",
        "",
        f"- NL 用例：{nl['sample_size']} 条人工构造小样本（`eval/cases.json`）；",
        f"- 闭环用例：{closed['sample_size']} 条带约束的合成 profile；",
        "- 指标用于自检与回归，**不代表真实分布上的泛化能力**，后续需用真实 query 扩充。",
        "",
        "## 理解层 / Agent 层",
        "",
        f"- 目的地抽取准确率：{nl['city_accuracy']}",
        f"- 天数抽取准确率：{nl['days_accuracy']}",
        f"- 追问触达准确率：{nl['clarification_accuracy']}",
        f"- 任务完成率（产出可用行程）：{nl['task_completion_rate']}",
        f"- 平均工具调用步数：{nl['avg_tool_steps']}",
        "",
        "## 规划层",
        "",
        f"- critic 通过率：{nl['critic_pass_rate']}",
        "",
        "## 闭环层（critic→reviser 价值，最重要）",
        "",
        f"- 修正前违规项合计：{closed['total_issues_before']}",
        f"- 修正后违规项合计：{closed['total_issues_after']}",
        f"- 违规项下降比例：{closed['issue_reduction_rate']}",
        f"- 修正后通过率：{closed['final_pass_rate']}",
        "",
        "### 闭环明细",
        "",
        "| 用例 | 修正前 | 修正后 | 闭环轮数 | 是否通过 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in closed["cases"]:
        lines.append(
            f"| {row['id']} | {row['original_issues']} | {row['final_issues']} | "
            f"{row['iterations']} | {'是' if row['passed'] else '否'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent / 规划 / 闭环 级评估")
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    parser.add_argument("--write-report", action="store_true", help="写入 docs/EVALUATION.md")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    nl = run_nl_eval(cases)
    closed = run_closed_loop_eval(CLOSED_LOOP_CASES)

    if args.json:
        print(json.dumps({"nl": nl, "closed_loop": closed}, ensure_ascii=False, indent=2))
    else:
        print("== 理解层 / Agent 层 ==")
        for k in [
            "sample_size",
            "city_accuracy",
            "days_accuracy",
            "clarification_accuracy",
            "task_completion_rate",
            "critic_pass_rate",
            "avg_tool_steps",
        ]:
            print(f"{k}: {nl[k]}")
        print("\n== 闭环层 ==")
        for k in [
            "sample_size",
            "total_issues_before",
            "total_issues_after",
            "issue_reduction_rate",
            "final_pass_rate",
        ]:
            print(f"{k}: {closed[k]}")

    if args.write_report:
        report = build_report(nl, closed)
        out = ROOT / "docs" / "EVALUATION.md"
        out.write_text(report, encoding="utf-8")
        print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
