from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def preflight_llm(settings: Any) -> dict[str, Any]:
    """Send a tiny OpenAI-compatible chat request to validate endpoint/key/quota."""
    llm = settings.llm
    url = f"{llm.base_url.rstrip('/')}/chat/completions"
    body = json.dumps(
        {
            "model": llm.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4,
            "temperature": 0,
        }
    ).encode()
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
            resp.read()
        return {
            "ok": True,
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
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
    except URLError as exc:
        return {
            "ok": False,
            "provider": llm.provider,
            "model": llm.model,
            "base_url": llm.base_url,
            "detail": str(exc.reason),
        }
