"""Retired historical selector; frozen data is sealed in the current release cycle.

规则（对齐外部 version_matrix 的 stability 段）：
- core_frozen 按 ``subset`` 字段分层抽取 30 条（层内固定 seed 随机，
  层间按配额比例分配，余数按层大小顺序补齐）；
- challenge_frozen 中原 production_v1 基线的 30 条（不含后续并入的长程集）；
- 固定 seed=42，重跑结果一致，产物 ``stability_60.json`` 提交入库。

用法：
    python scripts/make_stability_selection.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "eval" / "production_v1"
SEED = 42
CORE_QUOTA = 30


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified_sample(cases: list[dict], quota: int, rng: random.Random) -> list[str]:
    """按 subset 分层抽样：层配额 = quota × 层占比（向下取整），余数按层大小补齐。"""
    strata: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        strata[str(case.get("subset") or "unknown")].append(str(case["case_id"]))
    total = sum(len(ids) for ids in strata.values())
    quotas: dict[str, int] = {
        subset: len(ids) * quota // total for subset, ids in strata.items()
    }
    remainder = quota - sum(quotas.values())
    for subset in sorted(strata, key=lambda item: (-len(strata[item]), item)):
        if remainder <= 0:
            break
        if quotas[subset] < len(strata[subset]):
            quotas[subset] += 1
            remainder -= 1
    selected: list[str] = []
    for subset in sorted(strata):
        ids = sorted(strata[subset])
        rng.shuffle(ids)
        selected.extend(ids[: quotas[subset]])
    return sorted(selected)


def build_selection(data_dir: Path = DATA_DIR, seed: int = SEED) -> dict:
    raise RuntimeError(
        "current single-candidate release forbids legacy stability selection from "
        "opening Core/Challenge; use the official frozen manifest runner"
    )
    # Historical implementation is intentionally unreachable during this
    # release cycle.  It remains below solely to document the committed sample.
    core = [
        case
        for case in load_jsonl(data_dir / "core_frozen.jsonl")
        if case.get("subset") != "long_horizon_state"
    ]
    challenge = [
        case
        for case in load_jsonl(data_dir / "challenge_frozen.jsonl")
        if case.get("subset") != "long_horizon_state"
    ]
    rng = random.Random(seed)
    core_selected = stratified_sample(core, CORE_QUOTA, rng)
    challenge_selected = sorted(str(case["case_id"]) for case in challenge)
    selection = core_selected + challenge_selected
    return {
        "schema_version": "stability-selection-v1",
        "seed": seed,
        "core_quota": CORE_QUOTA,
        "case_ids": selection,
        "core_case_ids": core_selected,
        "challenge_case_ids": challenge_selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output", default=str(DATA_DIR / "stability_60.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    payload = build_selection(Path(args.data_dir), seed=args.seed)
    if len(payload["case_ids"]) != 60:
        raise SystemExit(f"抽样结果应为 60 条，实际 {len(payload['case_ids'])} 条")
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写入 {args.output}：core 分层 {len(payload['core_case_ids'])} 条 + "
          f"challenge 全部 {len(payload['challenge_case_ids'])} 条 = {len(payload['case_ids'])} 条")


if __name__ == "__main__":
    main()
