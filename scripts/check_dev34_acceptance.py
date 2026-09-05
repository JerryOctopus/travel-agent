"""Check one Dev34 run or the required pair of consecutive frozen runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.evaluation.dev34_acceptance import (  # noqa: E402
    evaluate_consecutive_dev34_runs,
    evaluate_dev34_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Dev34 acceptance contract")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--second-run-dir", type=Path)
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Check the pre-Judge gate for a single run.",
    )
    args = parser.parse_args()
    if args.second_run_dir and args.deterministic_only:
        parser.error("--deterministic-only cannot be combined with --second-run-dir")
    result = (
        evaluate_consecutive_dev34_runs(args.run_dir, args.second_run_dir)
        if args.second_run_dir
        else evaluate_dev34_run(
            args.run_dir, require_judge=not args.deterministic_only
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
