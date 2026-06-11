"""从落盘 artifact 聚合分层编排指标（PROJECT_PLAN M5/M8）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "artifacts"


def aggregate(artifact_dir: Path) -> dict:
    records: list[dict] = []
    for path in artifact_dir.glob("*/*.json"):
        if path.name == "session_state.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("kind") != "layer_metrics":
            continue
        records.append(data.get("payload", {}))

    if not records:
        return {"count": 0, "layer_hit_rate": None, "rollback_rate": None}

    hit_rates = [r["layer_hit_rate"] for r in records if r.get("layer_hit_rate") is not None]
    rollback_rates = [r["rollback_rate"] for r in records if r.get("rollback_rate") is not None]
    return {
        "count": len(records),
        "layer_hit_rate": round(sum(hit_rates) / len(hit_rates), 4) if hit_rates else None,
        "rollback_rate": round(sum(rollback_rates) / len(rollback_rates), 4) if rollback_rates else None,
        "completed_rate": round(
            sum(1 for r in records if r.get("completed")) / len(records), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate layered orchestration metrics.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = aggregate(Path(args.artifact_dir))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("Layer Metrics Summary")
        print("====================")
        for k, v in summary.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
