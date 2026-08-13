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
from travel_agent.settings import get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved Product itineraries")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("rules", "judge", "sample", "calibrate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-dir", type=Path, required=True)
        if command == "judge":
            subparser.add_argument("--resume", action="store_true")
        elif command == "sample":
            subparser.add_argument("--size", type=int, default=30)
            subparser.add_argument("--output", type=Path)
        elif command == "calibrate":
            subparser.add_argument("--reviews", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "rules":
        result = apply_quality_rules(args.run_dir)
    elif args.command == "judge":
        result = apply_independent_judge(
            args.run_dir,
            get_settings().evaluation.judge,
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
    else:
        result = apply_human_calibration(args.run_dir, args.reviews)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
