"""校验 config.toml / 环境变量中的 API key 是否可用（PROJECT_PLAN M1/M8）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.settings import get_settings
from travel_agent.harness.preflight import preflight_amap, preflight_llm
from travel_agent.evaluation.plan_quality_judge import preflight_judge


def _check_llm(settings) -> dict:
    llm = settings.llm
    if not llm.enabled:
        return {"ok": True, "skipped": True, "detail": "provider=rule 或无 api_key，走离线兜底"}
    return preflight_llm(settings)


def _check_amap(settings) -> dict:
    return preflight_amap(settings)


def _check_judge(settings) -> dict:
    return preflight_judge(settings.evaluation.judge)


def validate_all(settings) -> dict:
    """Preflight in spend-safe order and short-circuit every downstream API."""
    amap = _check_amap(settings)
    if not amap.get("ok") or amap.get("skipped"):
        blocked = {
            "ok": False,
            "skipped": True,
            "detail": "blocked because configured AMap preflight failed",
        }
        return {"amap_rest": amap, "llm": blocked, "judge": blocked}

    llm = _check_llm(settings)
    if not llm.get("ok") or llm.get("skipped"):
        return {
            "amap_rest": amap,
            "llm": llm,
            "judge": {
                "ok": False,
                "skipped": True,
                "detail": "blocked because tested-model preflight failed",
            },
        }

    return {
        "amap_rest": amap,
        "llm": llm,
        "judge": _check_judge(settings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM and Amap API keys")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    report = validate_all(settings)
    ok = all(v.get("ok") for v in report.values())

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for name, result in report.items():
            status = "OK" if result.get("ok") else "FAIL"
            if result.get("skipped"):
                status = "SKIP"
            print(f"[{status}] {name}: {result}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
