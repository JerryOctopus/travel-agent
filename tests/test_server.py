from __future__ import annotations

from fastapi.testclient import TestClient

from travel_agent.server import app


def _drain(ws):
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "done":
            break
    return events


def test_config_endpoint():
    client = TestClient(app)
    data = client.get("/config").json()
    assert "amap_js_key" in data
    assert "real_agent_enabled" in data


def test_websocket_chat_flow():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"message": "帮我规划杭州三天，喜欢自然和美食，轻松一点"})
        events = _drain(ws)

    types = [e["type"] for e in events]
    assert "session" in types
    assert "map" in types
    assert "text" in types
    assert any(e["type"] == "a2ui" for e in events)


def test_websocket_clarification_flow():
    client = TestClient(app)
    with client.websocket_connect("/chat") as ws:
        ws.send_json({"message": "我想去旅行"})
        events = _drain(ws)

    trace = next(e for e in events if e["type"] == "trace")
    assert trace["clarification"] is True
