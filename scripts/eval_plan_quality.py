"""Post-process a saved Product run with rules, Judge and human calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.evaluation.plan_quality_pipeline import (  # noqa: E402
    apply_human_calibration,
    apply_independent_judge,
    apply_quality_rules,
    export_human_review_sample,
)
from travel_agent.evaluation.plan_quality_judge import preflight_judge  # noqa: E402
from travel_agent.evaluation.frozen_release_acceptance import (  # noqa: E402
    JUDGE_MODEL,
    STAGE_CONTRACTS,
    evaluate_frozen_stage,
    evaluate_release_readiness_dev34,
    load_release_manifest,
)
from travel_agent.settings import get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved Product itineraries")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("rules", "judge", "sample", "calibrate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-dir", type=Path, required=True)
        if command == "judge":
            subparser.add_argument("--resume", action="store_true")
            subparser.add_argument("--official-release", action="store_true")
            subparser.add_argument("--release-manifest", type=Path)
        elif command == "sample":
            subparser.add_argument("--size", type=int, default=30)
            subparser.add_argument("--output", type=Path)
        elif command == "calibrate":
            subparser.add_argument("--reviews", type=Path, required=True)
    subparsers.add_parser("preflight")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "rules":
        result = apply_quality_rules(args.run_dir)
    elif args.command == "judge":
        settings = get_settings().evaluation.judge
        if args.official_release:
            _require_formal_judge_configuration(settings, resume=args.resume)
            _require_deterministic_release_gate(
                args.run_dir, args.release_manifest
            )
        result = apply_independent_judge(
            args.run_dir,
            settings,
            resume=args.resume,
        )
    elif args.command == "sample":
        if args.size <= 0:
            raise ValueError("--size must be positive")
        result = export_human_review_sample(
            args.run_dir,
            size=args.size,
            output=args.output,
        )
    elif args.command == "calibrate":
        result = apply_human_calibration(args.run_dir, args.reviews)
    else:
        result = preflight_judge(get_settings().evaluation.judge)
        if not result.get("ok"):
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _require_formal_judge_configuration(settings, *, resume: bool) -> None:
    failures = []
    if resume:
        failures.append("--resume is forbidden for a formal release Judge run")
    if settings.provider != JUDGE_MODEL["provider"]:
        failures.append("formal Judge provider must be siliconflow")
    if settings.model != JUDGE_MODEL["model"]:
        failures.append(f"formal Judge model must be {JUDGE_MODEL['model']}")
    if float(settings.temperature) != JUDGE_MODEL["temperature"]:
        failures.append("formal Judge temperature must be 0")
    if settings.thinking_enabled is not False:
        failures.append("formal Judge thinking must be disabled")
    if failures:
        raise RuntimeError("formal Judge rejected before API preflight: " + "; ".join(failures))


def _require_deterministic_release_gate(
    run_dir: Path, release_manifest: Path | None
) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts") or {}
    split = str(artifacts.get("frozen_split") or "dev")
    if split in STAGE_CONTRACTS:
        if release_manifest is None:
            raise RuntimeError(
                "formal frozen Judge requires --release-manifest before API preflight"
            )
        gate = evaluate_frozen_stage(
            run_dir,
            load_release_manifest(release_manifest),
            split,
            require_judge=False,
        )
    else:
        if release_manifest is not None:
            raise RuntimeError("Dev34 formal Judge must not use a frozen release manifest")
        gate = evaluate_release_readiness_dev34(run_dir, require_judge=False)
    if not gate.get("passed"):
        raise RuntimeError(
            "deterministic release gate failed before Judge API preflight: "
            + "; ".join(gate.get("failures") or ["unknown deterministic failure"])
        )


if __name__ == "__main__":
    main()
