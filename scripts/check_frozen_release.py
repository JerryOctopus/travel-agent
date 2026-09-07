"""Create and enforce the single-candidate frozen release contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.evaluation.frozen_release_acceptance import (  # noqa: E402
    build_release_manifest,
    evaluate_consecutive_release_dev34,
    evaluate_frozen_release,
    evaluate_frozen_stage,
    load_release_manifest,
    runtime_release_fingerprints,
    validate_release_manifest,
)
from travel_agent.settings import load_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dev = commands.add_parser("check-dev")
    dev.add_argument("--first-run", type=Path, required=True)
    dev.add_argument("--second-run", type=Path, required=True)
    dev.add_argument("--output", type=Path)

    create = commands.add_parser("create")
    create.add_argument("--candidate-id", required=True)
    create.add_argument("--tag", required=True)
    create.add_argument("--first-dev-run", type=Path, required=True)
    create.add_argument("--second-dev-run", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--core-file", type=Path, default=ROOT / "data/eval/production_v1/core_frozen.jsonl")
    create.add_argument("--challenge-file", type=Path, default=ROOT / "data/eval/production_v1/challenge_frozen.jsonl")
    create.add_argument("--shadow-file", type=Path, default=ROOT / "data/eval/production_v1/shadow_frozen.jsonl")

    stage = commands.add_parser("check-stage")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--split", choices=("core_frozen", "challenge_frozen", "shadow_frozen"), required=True)
    stage.add_argument("--run-dir", type=Path, required=True)
    stage.add_argument("--without-judge", action="store_true")
    stage.add_argument("--output", type=Path)

    release = commands.add_parser("check-release")
    release.add_argument("--manifest", type=Path, required=True)
    release.add_argument("--core-run", type=Path, required=True)
    release.add_argument("--challenge-run", type=Path, required=True)
    release.add_argument("--shadow-run", type=Path, required=True)
    release.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check-dev":
        result = evaluate_consecutive_release_dev34(args.first_run, args.second_run)
        _emit(result, args.output)
        raise SystemExit(0 if result["passed"] else 1)
    if args.command == "create":
        readiness = evaluate_consecutive_release_dev34(args.first_dev_run, args.second_dev_run)
        if not readiness["passed"]:
            _emit(readiness, args.output.with_name("dev_readiness_failed.json"))
            raise SystemExit("Dev34 release-readiness gate did not pass")
        commit_sha = _git("rev-parse", "HEAD")
        branch = _git("branch", "--show-current")
        if _git("rev-parse", args.tag) != commit_sha:
            raise SystemExit("release tag must resolve to the current candidate commit")
        if _git("status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("tracked worktree must be clean before creating a release manifest")
        fingerprints = runtime_release_fingerprints(load_settings(), "configured")
        recorded = readiness["second"].get("fingerprints") or {}
        mismatches = [
            field for field, value in fingerprints.items()
            if recorded.get(field) != value
        ]
        if mismatches:
            raise SystemExit("current runtime differs from Dev34 candidate: " + ", ".join(mismatches))
        manifest = build_release_manifest(
            candidate_id=args.candidate_id,
            commit_sha=commit_sha,
            branch=branch,
            tag=args.tag,
            split_files={
                "core_frozen": args.core_file,
                "challenge_frozen": args.challenge_file,
                "shadow_frozen": args.shadow_file,
            },
            fingerprints=fingerprints,
            provider_reported_models=list(
                (readiness["second"].get("fingerprints") or {}).get(
                    "provider_reported_models"
                )
                or []
            ),
        )
        failures = validate_release_manifest(manifest)
        if failures:
            raise SystemExit("generated invalid manifest: " + "; ".join(failures))
        _emit(manifest, args.output)
        return
    if args.command == "check-stage":
        manifest = load_release_manifest(args.manifest)
        result = evaluate_frozen_stage(
            args.run_dir,
            manifest,
            args.split,
            require_judge=not args.without_judge,
        )
        _emit(result, args.output)
        raise SystemExit(0 if result["passed"] else 1)
    manifest = load_release_manifest(args.manifest)
    result = evaluate_frozen_release(
        manifest,
        args.core_run,
        args.challenge_run,
        args.shadow_run,
    )
    _emit(result, args.output)
    raise SystemExit(0 if result["passed"] else 1)


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(text)


def _markdown(payload: dict[str, Any]) -> str:
    lines = ["# Frozen Release Acceptance", "", f"- passed: `{payload.get('passed')}`"]
    if payload.get("candidate_id"):
        lines.append(f"- candidate: `{payload['candidate_id']}`")
    if payload.get("split"):
        lines.append(f"- split: `{payload['split']}`")
    if payload.get("counts"):
        lines.extend([
            "",
            "## Counts",
            "",
            "```json",
            json.dumps(payload["counts"], ensure_ascii=False, indent=2),
            "```",
        ])
    if payload.get("judge"):
        lines.extend([
            "",
            "## Judge",
            "",
            "```json",
            json.dumps(payload["judge"], ensure_ascii=False, indent=2),
            "```",
        ])
    if payload.get("operations"):
        lines.extend([
            "",
            "## Cost, tokens and latency",
            "",
            "```json",
            json.dumps(payload["operations"], ensure_ascii=False, indent=2),
            "```",
        ])
    if payload.get("stages"):
        lines.extend(["", "## Stages", ""])
        for name, stage in payload["stages"].items():
            lines.append(
                f"- {name}: passed=`{stage.get('passed')}`, "
                f"strict=`{(stage.get('counts') or {}).get('strict')}`, "
                f"hard=`{(stage.get('counts') or {}).get('hard')}`"
            )
    lines.extend(["", "## Failures", ""])
    failures = payload.get("failures") or []
    lines.extend(f"- {item}" for item in failures)
    if not failures:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
