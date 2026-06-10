"""FastAPI 服务：WebSocket ``/chat`` + 静态 Web 前端（对应 PROJECT_PLAN M7）。

- ``/`` 返回单页前端（聊天 + 高德 JS 地图 + A2UI 卡片）；
- ``/config`` 暴露前端需要的高德 JS API key 与是否启用真实 agent；
- ``/chat`` WebSocket：按 session 维持多轮上下文，逐事件推送
  状态/卡片/地图/文本（A2UI 风格），并带超时保护。

run_turn 是同步的（可能调用 LLM），这里放到线程池执行，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from travel_agent.agent.runtime import run_turn
from travel_agent.agent.session import SessionContext, build_session
from travel_agent.settings import get_settings

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Personalized Travel Planning Agent")

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


class SessionLifecycleManager:
    """简单的会话隔离管理：session_id -> (SessionContext, history)。"""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def get_or_create(self, session_id: str | None) -> tuple[str, SessionContext, list]:
        if session_id and session_id in self._sessions:
            entry = self._sessions[session_id]
            return session_id, entry["ctx"], entry["history"]
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        ctx = build_session(session_id=sid)
        self._sessions[sid] = {"ctx": ctx, "history": []}
        return sid, ctx, self._sessions[sid]["history"]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


MANAGER = SessionLifecycleManager()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/config")
async def config() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(
        {
            "amap_js_key": settings.amap.js_key or "",
            "real_agent_enabled": settings.llm.enabled,
            "amap_rest_enabled": settings.amap.rest_enabled,
        }
    )


@app.websocket("/chat")
async def chat(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    try:
        while True:
            payload = await websocket.receive_json()
            message = (payload.get("message") or "").strip()
            session_id = payload.get("session_id")
            if not message:
                await websocket.send_json({"type": "error", "message": "空消息"})
                continue

            sid, ctx, history = MANAGER.get_or_create(session_id)
            await websocket.send_json({"type": "session", "session_id": sid})
            await websocket.send_json({"type": "status", "message": "正在思考与调用工具…"})

            try:
                reply = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_turn, message, ctx, list(history), settings
                    ),
                    timeout=settings.agent.request_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "message": "请求超时，请重试或简化需求。"})
                continue
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json({"type": "error", "message": f"处理出错：{exc}"})
                continue

            history.append(("user", message))
            history.append(("assistant", reply.text))

            await websocket.send_json(
                {
                    "type": "trace",
                    "tool_trace": reply.tool_trace,
                    "used_real_agent": reply.used_real_agent,
                    "clarification": reply.clarification,
                    "profile": reply.profile,
                }
            )
            for card in reply.cards:
                await websocket.send_json({"type": "a2ui", "card": card})
            if reply.map_payload:
                await websocket.send_json({"type": "map", "map": reply.map_payload})
            await websocket.send_json({"type": "text", "message": reply.text})
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        return


def main() -> None:
    import uvicorn

    uvicorn.run("travel_agent.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
