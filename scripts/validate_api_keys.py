"""校验 config.toml / 环境变量中的 API key 是否可用（PROJECT_PLAN M1/M8）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from travel_agent.settings import get_settings
from travel_agent.harness.preflight import preflight_llm


def _check_llm(settings) -> dict:
    llm = settings.llm
    if not llm.enabled:
        return {"ok": True, "skipped": True, "detail": "provider=rule 或无 api_key，走离线兜底"}
    return preflight_llm(settings)


def _check_amap(settings) -> dict:
    amap = settings.amap
    if not amap.rest_enabled:
        return {"ok": True, "skipped": True, "detail": "未配置 web_key，使用本地 seed 兜底"}
    checks: dict[str, dict] = {}
    requests = {
        "weather": (
            "/v3/weather/weatherInfo",
            {"key": amap.web_key, "city": "杭州", "extensions": "base"},
        ),
        "place_search": (
            "/v5/place/text",
            {
                "key": amap.web_key,
                "keywords": "景点",
                "region": "杭州",
                "city_limit": "true",
                "page_size": "1",
                "page_num": "1",
                "types": "110000",
                "show_fields": "business",
            },
        ),
    }
    for name, (path, arguments) in requests.items():
        url = f"{amap.base_url.rstrip('/')}{path}?{urlencode(arguments)}"
        try:
            with urlopen(url, timeout=amap.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode())
            if payload.get("status") == "1":
                checks[name] = {"ok": True}
            else:
                checks[name] = {
                    "ok": False,
                    "detail": payload.get("info", "unknown"),
                    "infocode": str(payload.get("infocode") or ""),
                }
        except URLError as exc:
            checks[name] = {"ok": False, "detail": str(exc.reason)}
    return {"ok": all(item["ok"] for item in checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM and Amap API keys")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    report = {
        "llm": _check_llm(settings),
        "amap_rest": _check_amap(settings),
    }
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
