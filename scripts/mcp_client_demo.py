"""演示「agent 从 MCP Server 动态拉取工具」（对应 PROJECT_PLAN M3 的 DoD）。

流程：
1. 以子进程拉起 ``travel_agent.mcp_server``（streamable-http）；
2. 用 ``MultiServerMCPClient.get_tools()`` 动态拉取工具列表，打印数量与名称；
3. 直接调用一次 ``search_poi`` 验证端到端可用；
4. 关闭子进程。

运行：``python scripts/mcp_client_demo.py``
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

HOST = "127.0.0.1"
PORT = int(os.getenv("TRAVEL_AGENT_MCP_PORT", "8765"))
URL = f"http://{HOST}:{PORT}/mcp/"


def _wait_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


async def _pull_tools() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"travel": {"url": URL, "transport": "streamable_http"}}
    )
    tools = await client.get_tools()
    print(f"从 MCP Server 拉取到 {len(tools)} 个工具：")
    for tool in tools:
        print(f"  - {tool.name}")

    by_name = {tool.name: tool for tool in tools}
    if "search_poi" in by_name:
        result = await by_name["search_poi"].ainvoke(
            {"session_id": "demo", "city": "杭州"}
        )
        print("\n调用 search_poi(city=杭州) 结果片段：")
        print(str(result)[:200])


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["TRAVEL_AGENT_MCP_HOST"] = HOST
    env["TRAVEL_AGENT_MCP_PORT"] = str(PORT)

    server = subprocess.Popen(
        [sys.executable, "-m", "travel_agent.mcp_server"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_port(HOST, PORT):
            print("MCP Server 启动超时", file=sys.stderr)
            return
        asyncio.run(_pull_tools())
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
