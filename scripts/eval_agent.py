"""Agent 级 + 规划级 + 闭环级评估（对应 PROJECT_PLAN M8 与「评估方案」）。

分层产出指标：

- 理解层：目的地 / 天数抽取准确率、追问触达率；
- Agent 层：任务完成率（是否产出可用行程）、平均 tool 步数、意图门控准确率；
- 规划层：critic 通过率、约束满足；
- 闭环层（最重要）：plan 原始稿 vs revise 修正稿的违规项下降幅度，量化 critic 价值。

注意：NL 用例为人工构造小样本（见 eval/cases.json），闭环用例为带约束的合成 profile；
指标用于自检与回归，不代表真实分布上的泛化。默认评估强制走离线确定性路径，
保证可复现；显式传入 ``--real-multi-agent`` 时，才读取真实 LLM 配置并跑
Multi-Agent Full（V3）评测，可观测性以 ``agent_trace`` 为准。
"""

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

from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.data_loader import load_seed_pois
from travel_agent.evaluation.plan_eval import evaluate_plan_artifact
from travel_agent.harness import AgentHarness, HarnessCase, HarnessEnvironment
from travel_agent.harness.reporting import (
    aggregate_case_results,
    aggregate_real_multi_agent_results,
)
from travel_agent.harness.preflight import preflight_llm
from travel_agent.orchestration.multi_agent import SUBAGENT_REGISTRY
from travel_agent.planning_subgraph import plan_and_critique
from travel_agent.providers import LocalToolProvider
from travel_agent.recommendation import score_pois
from travel_agent.schemas import TravelProfile
from travel_agent.settings import get_settings
from travel_agent.settings import load_settings

DEFAULT_POI_PATH = ROOT / "data" / "seed" / "pois.json"

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

MULTI_TURN_CASES: list[dict[str, Any]] = [
    {
        "id": "food_then_nature_food",
        "turns": [
            "帮我规划杭州三天，推荐美食",
            "自然风景呢，结合吃饭",
        ],
        "required_interests": ["nature", "food"],
    }
]

_EVAL_ENV_KEYS = [
    "TRAVEL_AGENT_LLM_PROVIDER",
    "TRAVEL_AGENT_LLM_API_KEY",
    "TRAVEL_AGENT_AMAP_WEB_KEY",
    "TRAVEL_AGENT_PROFILE_DIR",
    "TRAVEL_AGENT_SKILLS_ENABLED",
]


def _capture_eval_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _EVAL_ENV_KEYS}


def _restore_eval_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


def _configure_offline_eval() -> Any:
    """强制离线评估，避免误读本地 config.toml 中的真实 key 导致挂起或不可复现。"""
    os.environ["TRAVEL_AGENT_LLM_PROVIDER"] = "rule"
    os.environ.pop("TRAVEL_AGENT_LLM_API_KEY", None)
    os.environ["TRAVEL_AGENT_AMAP_WEB_KEY"] = ""
    tmp = tempfile.mkdtemp(prefix="travel_eval_")
    os.environ["TRAVEL_AGENT_PROFILE_DIR"] = tmp
    get_settings.cache_clear()
    return get_settings()


def _configure_real_multi_agent_eval() -> Any:
    """读取真实配置，运行生产 Multi-Agent Full（V3）评测。"""
    os.environ["TRAVEL_AGENT_SKILLS_ENABLED"] = "false"
    tmp = tempfile.mkdtemp(prefix="travel_real_ma_eval_")
    os.environ["TRAVEL_AGENT_PROFILE_DIR"] = tmp
    get_settings.cache_clear()
    settings = load_settings()
    if not settings.llm.enabled:
        raise RuntimeError(
            "真实 Multi-Agent 评测需要配置 LLM：请设置 config.toml [llm] "
            "provider/api_key，或 TRAVEL_AGENT_LLM_PROVIDER / TRAVEL_AGENT_LLM_API_KEY。"
        )
    return settings


def _preflight_llm(settings: Any) -> dict[str, Any]:
    """Backward-compatible wrapper around harness LLM preflight."""
    return preflight_llm(settings)


def _round(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


def run_nl_eval(cases: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    harness_cases = [HarnessCase.from_dict(case) for case in cases]
    harness = AgentHarness(
        settings=settings,
        environment=HarnessEnvironment(
            mode="offline",
            poi_path=DEFAULT_POI_PATH,
            persist=False,
            user_id="eval_user",
        ),
    )
    results = harness.run_cases(harness_cases)
    return aggregate_case_results(results, harness_cases)


def run_multi_turn_eval(cases: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    rows = []
    local_provider = LocalToolProvider(load_seed_pois(DEFAULT_POI_PATH))
    for case in cases:
        ctx = build_session(persist=False, poi_path=DEFAULT_POI_PATH)
        ctx.provider = local_provider
        last_reply = None
        for turn in case["turns"]:
            last_reply = run_production_turn(
                turn, ctx=ctx, settings=settings, user_id="eval_user"
            )
        itinerary_artifact = ctx.store.latest("itinerary")
        plan_eval = evaluate_plan_artifact(
            itinerary_artifact,
            profile=toolkit_profile_dict(ctx.profile),
            required_interests=list(case.get("required_interests") or []),
        )
        success = bool(
            itinerary_artifact
            and plan_eval
            and plan_eval.preference_pass
            and plan_eval.constraint_pass
        )
        rows.append(
            {
                "id": case["id"],
                "turn_count": len(case["turns"]),
                "tool_trace": list(last_reply.tool_trace if last_reply else []),
                "multi_turn_edit_success": success,
                "preference_pass": plan_eval.preference_pass if plan_eval else None,
                "constraint_pass": plan_eval.constraint_pass if plan_eval else None,
                "final_pass": plan_eval.final_pass if plan_eval else None,
                "issues": list(plan_eval.issues) if plan_eval else ["no_itinerary"],
            }
        )
    return {
        "sample_size": len(rows),
        "multi_turn_edit_success_rate": _rate(rows, "multi_turn_edit_success"),
        "final_pass_rate": _rate(rows, "final_pass"),
        "cases": rows,
    }


def run_closed_loop_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    provider = LocalToolProvider(load_seed_pois(DEFAULT_POI_PATH))
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


def toolkit_profile_dict(profile: TravelProfile) -> dict[str, Any]:
    return {
        "destination": profile.destination,
        "days": profile.days,
        "interests": list(profile.interests),
        "budget_level": profile.budget_level,
        "pace": profile.pace,
        "companions": profile.companions,
        "transport_mode": profile.transport_mode,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _is_llm_quota_or_auth_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "allocationquota",
            "free quota",
            "free tier",
            "quota",
            "403",
            "unauthorized",
            "invalid api key",
        )
    )


def run_real_multi_agent_eval(cases: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    """真实 LLM Multi-Agent Full（V3）评测。

    与默认离线评估不同，这里必须实际调用 LLM 的 Orchestrator / Subagent，并校验：
    - `reply.used_real_agent=True`
    - 产生 itinerary
    - agent_trace 覆盖 attraction/hotel/restaurant/transport/planner 五个执行型 Subagent
    - research/planning Subagent 触发关键工具，且工具调用不出白名单
    """
    allowed_by_agent = {
        name: set(definition.tool_names)
        for name, definition in SUBAGENT_REGISTRY.items()
    }
    runnable_cases = [
        case
        for case in cases
        if not case.get("expected_clarification") and not case.get("expect_no_itinerary")
    ]
    harness_cases = [HarnessCase.from_dict(case) for case in runnable_cases]
    harness = AgentHarness(
        settings=settings,
        environment=HarnessEnvironment(
            mode="real_agent",
            poi_path=DEFAULT_POI_PATH,
            persist=False,
            user_id="real_ma_eval",
        ),
    )
    results = harness.run_cases(harness_cases)
    return aggregate_real_multi_agent_results(
        results,
        harness_cases,
        allowed_by_agent=allowed_by_agent,
    )


def build_report(
    nl: dict[str, Any],
    closed: dict[str, Any],
    multi_turn: dict[str, Any] | None = None,
    real_multi_agent: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# 评估报告（EVALUATION）",
        "",
        "> 自动生成：`python scripts/eval_agent.py --write-report`。",
        "> 评估在**离线确定性路径**下运行（强制 `provider=rule`），保证可复现。",
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
        f"- 追问/澄清触达准确率：{nl['clarification_accuracy']}",
        f"- 意图门控准确率（寒暄/缺信息不误规划）：{nl.get('intent_gate_accuracy')}",
        f"- 任务完成率（产出可用行程）：{nl['task_completion_rate']}",
        f"- 平均工具调用步数：{nl['avg_tool_steps']}",
        "",
        "## 规划层",
        "",
        f"- critic 通过率：{nl['critic_pass_rate']}",
        "",
        "## ChinaTravel-style 计划质量层",
        "",
        "这组指标按 ChinaTravel 的 planning eval 口径做轻量实现：",
        "先看是否交付计划，再看环境可行性、逻辑约束与最终通过率。",
        "",
        f"- DR / Delivery Rate（成功交付行程）：{nl.get('delivery_rate')}",
        f"- EPR / Environmental Pass Rate（环境可行性）：{nl.get('epr')}",
        f"- LPR / Logical Pass Rate（硬约束满足）：{nl.get('lpr')}",
        f"- FPR / Final Pass Rate（环境 + 约束 + 偏好整体通过）：{nl.get('fpr')}",
        f"- C-LPR / Conditional Logical Pass Rate（环境通过后的逻辑通过）：{nl.get('conditional_logical_pass_rate')}",
        f"- Preference Pass Rate（软偏好覆盖）：{nl.get('preference_pass_rate')}",
        f"- 餐厅饭点有效率：{nl.get('meal_time_validity')}",
        f"- 路线可行率：{nl.get('route_feasibility_rate')}",
        f"- 单日负载有效率：{nl.get('daily_load_validity')}",
        f"- POI 城市一致率：{nl.get('poi_city_validity')}",
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

    if real_multi_agent:
        lines.extend(
            [
                "",
                "## 真实 LLM Multi-Agent 层（agent_trace 口径）",
                "",
                f"- 用例数：{real_multi_agent['sample_size']}",
                f"- used_real_agent rate：{real_multi_agent['used_real_agent_rate']}",
                f"- agent trace rate：{real_multi_agent['agent_trace_rate']}",
                f"- required agent coverage：{real_multi_agent['agent_completed_rate']}",
                f"- research agent tool rate：{real_multi_agent['research_agent_tool_rate']}",
                f"- planning agent tool rate：{real_multi_agent['planning_agent_tool_rate']}",
                f"- agent tool whitelist rate：{real_multi_agent['agent_tool_whitelist_rate']}",
                f"- external LLM error rate：{real_multi_agent['external_llm_error_rate']}",
                f"- required tool coverage：{real_multi_agent['tool_coverage_rate']}",
                f"- final pass rate：{real_multi_agent['final_pass_rate']}",
                f"- full multi-agent pass rate：{real_multi_agent['full_multi_agent_pass_rate']}",
                "",
                "| 用例 | 真 LLM | agent trace | Agent 覆盖 | 白名单 | research工具 | planning工具 | 必要工具 | 最终通过 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in real_multi_agent["cases"]:
            lines.append(
                f"| {row['id']} | {'是' if row['used_real_agent'] else '否'} | "
                f"{'是' if row['agent_trace_present'] else '否'} | "
                f"{'是' if row['required_agents_present'] else '否'} | "
                f"{'是' if row['agent_tool_whitelist_ok'] else '否'} | "
                f"{'是' if row['research_tools_ok'] else '否'} | "
                f"{'是' if row['planning_tools_ok'] else '否'} | "
                f"{'是' if row['required_tools_ok'] else '否'} | "
                f"{'是' if row['final_pass'] else '否'} |"
            )
        lines.append("")

    if multi_turn:
        lines.extend(
            [
                "",
                "## 多轮修改层",
                "",
                f"- 多轮用例数：{multi_turn['sample_size']}",
                f"- 多轮修改成功率：{multi_turn['multi_turn_edit_success_rate']}",
                f"- 多轮最终通过率：{multi_turn['final_pass_rate']}",
                "",
                "| 用例 | 轮数 | 修改成功 | 最终通过 | 问题 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in multi_turn["cases"]:
            lines.append(
                f"| {row['id']} | {row['turn_count']} | "
                f"{'是' if row['multi_turn_edit_success'] else '否'} | "
                f"{'是' if row['final_pass'] else '否'} | "
                f"{', '.join(row['issues']) or '无'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 三句话结论（作品集 / 面试）",
            "",
            _pitch_sentences(nl, closed),
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _pitch_sentences(nl: dict[str, Any], closed: dict[str, Any]) -> str:
    reduction = closed.get("issue_reduction_rate")
    reduction_pct = int((reduction or 0) * 100)
    return "\n".join(
        [
            f"1. 可控规划子图通过 critic→reviser 闭环，把合成用例违规项从 "
            f"{closed['total_issues_before']} 降到 {closed['total_issues_after']}（下降 {reduction_pct}%）。",
            f"2. 离线 Agent 路径任务完成率 {nl['task_completion_rate']}，"
            f"澄清/追问触达率 {nl['clarification_accuracy']}，平均 {nl['avg_tool_steps']} 步工具调用。",
            "3. 生产入口固定 Multi-Agent Full（V3）：Orchestrator 派工五个执行型 Subagent，"
            "Reviewer 最多一次修复；无 key 时用 deterministic fallback 复现指标。",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent / 规划 / 闭环 级评估")
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    parser.add_argument("--write-report", action="store_true", help="写入 docs/EVALUATION.md")
    parser.add_argument(
        "--real-multi-agent",
        action="store_true",
        help="使用真实 LLM 配置运行 Multi-Agent Full（V3）评测",
    )
    parser.add_argument(
        "--real-limit",
        type=int,
        default=None,
        help="限制真实 LLM Multi-Agent 评测用例数，便于控制成本",
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    original_env = _capture_eval_env()
    settings = _configure_offline_eval()
    nl = run_nl_eval(cases, settings)
    real_multi_agent = None
    if args.real_multi_agent:
        _restore_eval_env(original_env)
        real_cases = list(cases)
        if args.real_limit is not None:
            real_cases = real_cases[: max(1, args.real_limit)]
        real_settings = _configure_real_multi_agent_eval()
        preflight = _preflight_llm(real_settings)
        if not preflight.get("ok"):
            print(
                "\n真实 LLM preflight 失败，未进入 Multi-Agent 评测："
                f"provider={preflight.get('provider')} "
                f"model={preflight.get('model')} "
                f"base_url={preflight.get('base_url')} "
                f"status={preflight.get('status')} "
                f"detail={preflight.get('detail')}",
                file=sys.stderr,
            )
            sys.exit(1)
        real_multi_agent = run_real_multi_agent_eval(real_cases, real_settings)
    else:
        _restore_eval_env(original_env)
    multi_turn = run_multi_turn_eval(MULTI_TURN_CASES, settings)
    closed = run_closed_loop_eval(CLOSED_LOOP_CASES)

    if args.json:
        print(
            json.dumps(
                {
                    "nl": nl,
                    "real_multi_agent": real_multi_agent,
                    "multi_turn": multi_turn,
                    "closed_loop": closed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("== 理解层 / Agent 层 ==")
        for k in [
            "sample_size",
            "city_accuracy",
            "days_accuracy",
            "clarification_accuracy",
            "intent_gate_accuracy",
            "task_completion_rate",
            "delivery_rate",
            "critic_pass_rate",
            "environment_pass_rate",
            "epr",
            "constraint_pass_rate",
            "lpr",
            "preference_pass_rate",
            "china_travel_style_final_pass_rate",
            "fpr",
            "conditional_logical_pass_rate",
            "meal_time_validity",
            "tool_coverage_rate",
            "avg_tool_steps",
        ]:
            print(f"{k}: {nl[k]}")
        if real_multi_agent is not None:
            print("\n== 真实 LLM Multi-Agent 层（agent_trace 口径） ==")
            for k in [
                "sample_size",
                "used_real_agent_rate",
                "agent_trace_rate",
                "agent_completed_rate",
                "research_agent_tool_rate",
                "planning_agent_tool_rate",
                "agent_tool_whitelist_rate",
                "external_llm_error_rate",
                "tool_coverage_rate",
                "final_pass_rate",
                "full_multi_agent_pass_rate",
            ]:
                print(f"{k}: {real_multi_agent[k]}")
        print("\n== 多轮修改层 ==")
        for k in [
            "sample_size",
            "multi_turn_edit_success_rate",
            "final_pass_rate",
        ]:
            print(f"{k}: {multi_turn[k]}")
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
        report = build_report(nl, closed, multi_turn, real_multi_agent)
        out = ROOT / "docs" / "EVALUATION.md"
        out.write_text(report, encoding="utf-8")
        print(f"\n报告已写入 {out}")

    if (
        args.real_multi_agent
        and real_multi_agent is not None
        and real_multi_agent.get("full_multi_agent_pass_rate") != 1.0
    ):
        if real_multi_agent.get("external_llm_error_rate"):
            print(
                "\n真实 LLM Multi-Agent 评测遇到外部模型错误："
                f"external_llm_error_rate={real_multi_agent.get('external_llm_error_rate')}。"
                "请检查 API key、额度、或关闭云厂商控制台的“仅使用免费额度”限制。",
                file=sys.stderr,
            )
        print(
            "\n真实 LLM Multi-Agent 评测未完全通过："
            f"full_multi_agent_pass_rate={real_multi_agent.get('full_multi_agent_pass_rate')}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
