from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def preflight_amap(settings: Any) -> dict[str, Any]:
    """Validate the configured AMap key and the quotas used by production tools."""
    amap = settings.amap
    if not amap.rest_enabled:
        return {
            "ok": True,
            "skipped": True,
            "detail": "未配置 web_key，使用本地 seed 兜底",
        }

    checks: dict[str, dict[str, Any]] = {}
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
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            checks[name] = {
                "ok": False,
                "status": exc.code,
                "detail": detail,
            }
        except json.JSONDecodeError as exc:
            checks[name] = {
                "ok": False,
                "detail": f"invalid JSON response: {exc}",
            }
        except URLError as exc:
            checks[name] = {"ok": False, "detail": str(exc.reason)}

    return {"ok": all(item["ok"] for item in checks.values()), "checks": checks}


def preflight_llm(settings: Any) -> dict[str, Any]:
    """Send a tiny OpenAI-compatible chat request to validate endpoint/key/quota."""
    llm = settings.llm
    url = f"{llm.base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": llm.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "temperature": llm.temperature,
    }
    if (
        str(llm.provider).lower() == "deepseek"
        and str(llm.model).lower() in {"deepseek-v4-flash", "deepseek-v4-pro"}
    ):
        payload["thinking"] = {
            "type": "enabled" if llm.thinking_enabled else "disabled"
        }
    body = json.dumps(payload).encode()
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {llm.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=llm.timeout_seconds) as resp:
            response = json.loads(resp.read().decode("utf-8"))
        return {
            "ok": True,
            "provider": llm.provider,
            "model": llm.model,
            "provider_reported_model": response.get("model"),
            "base_url": llm.base_url,
            "temperature": llm.temperature,
            "thinking_enabled": llm.thinking_enabled,
        }
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        return {
            "ok": False,
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
            "status": exc.code,
            "detail": detail,
        }
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
            "detail": f"invalid JSON response: {exc}",
        }
    except URLError as exc:
        return {
            "ok": False,
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
            "detail": str(exc.reason),
        }
