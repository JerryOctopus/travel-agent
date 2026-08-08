"""FastAPI 服务：WebSocket ``/chat`` + 静态 Web 前端（对应 PROJECT_PLAN M7）。

- ``/`` 返回单页前端（聊天 + 高德 JS 地图 + A2UI 卡片）；
- ``/config`` 暴露前端需要的高德 JS API key 与是否启用真实 agent；
- ``/chat`` WebSocket：按 session 维持多轮上下文，逐事件推送
  状态/卡片/地图/文本（A2UI 风格），并带超时保护。

run_production_turn 是同步的（可能调用 LLM），这里放到线程池执行，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from travel_agent.agent.runtime import run_production_turn
from travel_agent.settings import get_settings
from travel_agent.storage.session_manager import SessionLifecycleManager

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Personalized Travel Planning Agent")

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

def _build_manager() -> SessionLifecycleManager:
    settings = get_settings()
    return SessionLifecycleManager(profile_dir=settings.memory.profile_dir)


MANAGER = _build_manager()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/config")
async def config() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(
        {
            "amap_js_key": settings.amap.js_key or "",
            "amap_js_security_key": settings.amap.js_security_key or "",
            "real_agent_enabled": settings.llm.enabled,
            "amap_rest_enabled": settings.amap.rest_enabled,
            "skills_enabled": settings.skills.enabled,
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
            user_id = (payload.get("user_id") or "default").strip() or "default"
            if not message:
                await websocket.send_json({"type": "error", "message": "空消息"})
                continue

            sid, ctx, history = MANAGER.get_or_create(session_id, user_id=user_id)
            await websocket.send_json({"type": "session", "session_id": sid})
            await websocket.send_json({"type": "status", "message": "正在思考与调用工具…"})

            try:
                reply = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_production_turn,
                        message,
                        ctx,
                        list(history),
                        settings,
                        user_id,
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
            MANAGER.persist_turn(sid, ctx, history, user_id=user_id)

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
