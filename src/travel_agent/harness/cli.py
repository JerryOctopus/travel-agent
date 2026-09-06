from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from travel_agent.harness.chinatravel import (
    ChinaTravelHarnessRunner,
    build_chinatravel_report,
)
from travel_agent.harness.live_tools import (
    LiveToolsHarnessRunner,
    build_live_tools_report,
    write_live_tools_summary,
)
from travel_agent.harness.long_horizon import (
    DEFAULT_LONG_HORIZON_CASES,
    run_long_horizon_suite,
)
from travel_agent.harness.reporting import (
    build_release_report,
    evaluate_release_gates,
)
from travel_agent.harness import AgentHarness, HarnessEnvironment
from travel_agent.harness.product import (
    DEFAULT_PRODUCT_DEV_CASES,
    build_product_report,
    run_product_suite,
    write_product_run,
)
from travel_agent.settings import load_settings

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHINATRAVEL_ROOT = Path("/Users/carrier/run/projects/ChinaTravel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Travel Agent evaluation harness.")
    parser.add_argument(
        "--suite",
        choices=[
            "agent-nl",
            "agent-real",
            "agent-product",
            "agent-long-horizon",
            "chinatravel-mini",
            "chinatravel-human154",
            "chinatravel-human1000",
            "live-tools-shadow",
            "all",
        ],
        default="agent-nl",
    )
    parser.add_argument(
        "--env",
        choices=["offline", "real_agent", "replay", "live"],
        default="offline",
    )
    parser.add_argument("--provider", choices=["amap", "chinatravel", "local"], default="chinatravel")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--continue-on-llm-error", action="store_true")
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    parser.add_argument("--product-cases", default=str(DEFAULT_PRODUCT_DEV_CASES))
    parser.add_argument("--long-horizon-cases", default=str(DEFAULT_LONG_HORIZON_CASES))
    parser.add_argument(
        "--long-horizon-split",
        choices=["dev", "frozen", "all"],
        default="dev",
    )
    parser.add_argument(
        "--product-split",
        choices=["dev", "core_frozen", "challenge_frozen", "shadow_frozen", "all"],
        default="dev",
    )
    parser.add_argument("--remediation-complete", action="store_true")
    parser.add_argument("--chinatravel-root", type=Path, default=DEFAULT_CHINATRAVEL_ROOT)
    parser.add_argument("--shadow-suite", choices=["shadow-dev", "shadow-full"], default="shadow-full")
    args = parser.parse_args()

    results = run_harness_cli(args)
    release = evaluate_release_gates(results)
    if args.write_report:
        (ROOT / "docs" / "EVALUATION_RELEASE.md").write_text(
            build_release_report(release),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(release, ensure_ascii=False, indent=2))
    else:
        print(build_release_report(release))

    sys.exit(0 if release["status"] == "pass" else 1)


def run_harness_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.suite == "all":
        suites = ["agent-nl", "chinatravel-mini", "live-tools-shadow"]
        if args.env == "real_agent":
            suites.insert(1, "agent-real")
            suites.insert(2, "agent-product")
    else:
        suites = [args.suite]

    results: dict[str, Any] = {}
    for suite in suites:
        if suite == "agent-nl":
            results[suite] = _run_agent_nl(args)
        elif suite == "agent-real":
            results[suite] = _run_agent_real(args)
        elif suite == "agent-product":
            results[suite] = _run_agent_product(args)
        elif suite == "agent-long-horizon":
            results[suite] = _run_agent_long_horizon(args)
        elif suite.startswith("chinatravel-"):
            results[suite] = _run_chinatravel(args, suite)
        elif suite == "live-tools-shadow":
            results[suite] = _run_live_tools(args)
        else:  # pragma: no cover - argparse prevents this
            raise ValueError(f"Unsupported suite: {suite}")
    return results


def _run_agent_product(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    if args.env == "real_agent" and not settings.llm.enabled:
        raise RuntimeError(
            "production_v1 real baseline requires a configured fixed LLM provider and API key."
        )
    settings = replace(
        settings,
        memory=replace(
            settings.memory,
            backend="json",
            database_url=None,
            profile_dir=Path(tempfile.mkdtemp(prefix="product_eval_profile_")),
        ),
    )
    environment = HarnessEnvironment(
        mode="real_agent" if args.env == "real_agent" else "offline",
        persist=False,
        user_id="product_eval",
    )
    result = run_product_suite(
        AgentHarness(settings=settings, environment=environment),
        case_path=args.product_cases,
        split=args.product_split,
        limit=args.limit,
        remediation_complete=args.remediation_complete,
    )
    summary = {
        "metrics": result.metrics,
        "artifacts": {
            key: value for key, value in result.artifacts.items() if key != "_case_outputs"
        },
        "rows": result.rows,
    }
    if args.write_report:
        output_root = ROOT / "data" / "eval" / "product"
        summary = write_product_run(result, output_root)
        # 运行报告落独立文件；docs/EVALUATION_PRODUCT.md 是 production_v1 静态文档，不被覆盖。
        (ROOT / "docs" / "EVALUATION_PRODUCT_RUN.md").write_text(
            build_product_report(result)
            + f"\n## Saved outputs\n\n- run: `{summary['artifacts']['run_dir']}`\n"
            + f"- per-case outputs: `{summary['artifacts']['case_output_dir']}`\n",
            encoding="utf-8",
        )
    return summary


def _run_agent_long_horizon(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    if args.env == "real_agent" and not settings.llm.enabled:
        raise RuntimeError("long_horizon_v1 requires a configured LLM in real_agent mode.")
    settings = replace(
        settings,
        memory=replace(
            settings.memory,
            backend="json",
            database_url=None,
            profile_dir=Path(tempfile.mkdtemp(prefix="long_horizon_eval_profile_")),
        ),
    )
    environment = HarnessEnvironment(
        mode="real_agent" if args.env == "real_agent" else "offline",
        persist=False,
        user_id="long_horizon_eval",
    )
    result = run_long_horizon_suite(
        AgentHarness(settings=settings, environment=environment),
        case_path=args.long_horizon_cases,
        split=args.long_horizon_split,
        limit=args.limit,
    )
    return {
        "metrics": result.metrics,
        "artifacts": result.artifacts,
        "rows": result.rows,
    }


def _run_agent_nl(args: argparse.Namespace) -> dict[str, Any]:
    from scripts import eval_agent as legacy

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    settings = legacy._configure_offline_eval()
    summary = legacy.run_nl_eval(cases, settings)
    if args.write_report:
        report = legacy.build_report(
            summary,
            legacy.run_closed_loop_eval(legacy.CLOSED_LOOP_CASES),
            legacy.run_multi_turn_eval(legacy.MULTI_TURN_CASES, settings),
            None,
        )
        (ROOT / "docs" / "EVALUATION.md").write_text(report, encoding="utf-8")
    return summary


def _run_agent_real(args: argparse.Namespace) -> dict[str, Any]:
    from scripts import eval_agent as legacy

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.limit is not None:
        cases = cases[: max(1, args.limit)]
    settings = legacy._configure_real_multi_agent_eval()
    preflight = legacy._preflight_llm(settings)
    if not preflight.get("ok"):
        return {
            "sample_size": len(cases),
            "external_llm_error_rate": 1.0,
            "used_real_agent_rate": 0.0,
            "agent_trace_rate": 0.0,
            "full_multi_agent_pass_rate": 0.0,
            "preflight": preflight,
        }
    return legacy.run_real_multi_agent_eval(cases, settings)


def _run_chinatravel(args: argparse.Namespace, suite: str) -> dict[str, Any]:
    suite_name = {
        "chinatravel-mini": "mini-dev",
        "chinatravel-human154": "human154",
        "chinatravel-human1000": "human1000",
    }[suite]
    result = ChinaTravelHarnessRunner(
        chinatravel_root=args.chinatravel_root,
    ).run(
        suite=suite_name,
        smoke_limit=args.limit,
        resume=args.resume,
        real_multi_agent=args.env == "real_agent",
        continue_on_llm_error=args.continue_on_llm_error,
    )
    if args.write_report:
        (ROOT / "docs" / "EVALUATION_CHINATRAVEL.md").write_text(
            build_chinatravel_report(result),
            encoding="utf-8",
        )
    return {"metrics": result.metrics, "artifacts": result.artifacts}


def _run_live_tools(args: argparse.Namespace) -> dict[str, Any]:
    provider = "chinatravel" if args.provider == "local" else args.provider
    mode = "live" if args.env == "live" else "replay"
    result = LiveToolsHarnessRunner().run(
        provider=provider,
        mode=mode,
        suite=args.shadow_suite,
        limit=args.limit,
        agent_real_multi_agent=args.env == "real_agent",
        continue_on_llm_error=args.continue_on_llm_error,
    )
    write_live_tools_summary(result)
    if args.write_report:
        (ROOT / "docs" / "EVALUATION_LIVE_TOOLS.md").write_text(
            build_live_tools_report(result),
            encoding="utf-8",
        )
    return result.metrics


if __name__ == "__main__":
    main()
