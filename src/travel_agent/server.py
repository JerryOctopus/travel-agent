"""FastAPI 服务：WebSocket ``/chat`` + 静态 Web 前端（对应 PROJECT_PLAN M7）。

- ``/`` 返回单页前端（聊天 + 高德 JS 地图 + A2UI 卡片）；
- ``/config`` 暴露前端需要的高德 JS API key 与是否启用真实 agent；
- ``/chat`` WebSocket：按 session 维持多轮上下文，逐事件推送
  状态/卡片/地图/文本（A2UI 风格），并带超时保护。

run_production_turn 是同步的（可能调用 LLM），这里放到线程池执行，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import RequestControl, SessionContext
from travel_agent.orchestration.multi_agent.schemas import new_request_id
from travel_agent.settings import get_settings
from travel_agent.storage.session_manager import SessionLifecycleManager
from travel_agent.storage.user_memory import close_user_memory_services

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
DEV34_REPORT_FILE = (
    Path(__file__).resolve().parents[2]
    / "data/eval/product/runs/dev34_frozen_full_20260813a/dev34_case_report.html"
)
_SRC = Path(__file__).resolve().parents[1]

_mcp_process: subprocess.Popen[bytes] | None = None


def _wait_mcp_port(host: str, port: int, timeout: float = 15.0) -> bool:
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _mcp_process
    settings = get_settings()
    if settings.mcp.auto_start:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        env["TRAVEL_AGENT_MCP_HOST"] = settings.mcp.host
        env["TRAVEL_AGENT_MCP_PORT"] = str(settings.mcp.port)
        _mcp_process = subprocess.Popen(
            [sys.executable, "-m", "travel_agent.mcp_server"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_mcp_port(settings.mcp.host, settings.mcp.port)
    try:
        yield
    finally:
        close_user_memory_services()
        if _mcp_process is not None:
            _mcp_process.terminate()
            try:
                _mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _mcp_process.kill()
            _mcp_process = None


app = FastAPI(title="Personalized Travel Planning Agent", lifespan=_lifespan)

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

def _build_manager() -> SessionLifecycleManager:
    settings = get_settings()
    return SessionLifecycleManager(memory_settings=settings.memory)


MANAGER = _build_manager()


async def _run_isolated_request(
    message: str,
    ctx: SessionContext,
    history: list[tuple[str, str]],
    settings: Any,
    user_id: str,
):
    """Run sync agent work in a private snapshot and publish only on success."""
    control = RequestControl(new_request_id())
    snapshot = ctx.clone_isolated(control)
    task = asyncio.create_task(
        asyncio.to_thread(
            run_production_turn,
            message,
            snapshot,
            list(history),
            settings,
            user_id,
        )
    )
    try:
        reply = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=settings.agent.request_timeout_seconds,
        )
        control.check_active()
        ctx.commit_from(snapshot)
        return reply
    except (asyncio.TimeoutError, asyncio.CancelledError):
        control.cancel()
        task.cancel()
        raise


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/dev34-report")
async def dev34_report() -> FileResponse:
    return FileResponse(str(DEV34_REPORT_FILE), media_type="text/html")


@app.get("/config")
async def config() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(
        {
            "amap_js_key": settings.amap.js_key or "",
            "amap_js_security_key": settings.amap.js_security_key or "",
            "real_agent_enabled": settings.llm.enabled,
            "amap_rest_enabled": settings.amap.rest_enabled,
            "mcp_use_tools": settings.mcp.use_mcp_tools,
            "mcp_auto_start": settings.mcp.auto_start,
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
                reply = await _run_isolated_request(
                    message, ctx, list(history), settings, user_id
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
