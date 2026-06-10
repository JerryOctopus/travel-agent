from __future__ import annotations

from pathlib import Path

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session

POI_PATH = Path(__file__).resolve().parents[1] / "data" / "seed" / "pois.json"


def _session():
    return build_session(persist=False, poi_path=POI_PATH)


def test_full_tool_pipeline():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2, interests=["nature", "food"])

    search = toolkit.search_poi(ctx)
    assert search["isError"] is False
    assert search["count"] > 0

    weather = toolkit.check_weather(ctx)
    assert weather["isError"] is False

    rec = toolkit.recommend_candidates(ctx)
    assert rec["isError"] is False
    assert rec["count"] > 0

    plan = toolkit.plan_and_critique(ctx)
    assert plan["isError"] is False
    assert plan["final_issue_count"] <= plan["original_issue_count"]

    cards = toolkit.render_itinerary(ctx)
    assert any(c["type"] == "day" for c in cards["cards"])

    map_payload = toolkit.render_map(ctx)
    assert len(map_payload["markers"]) > 0


def test_request_travel_info_when_missing():
    ctx = _session()
    toolkit.update_travel_profile(ctx, interests=["food"])
    info = toolkit.request_travel_info(ctx)
    assert "destination" in info["missing_fields"]
    assert info["question"]


def test_plan_requires_recommend_first():
    ctx = _session()
    toolkit.update_travel_profile(ctx, destination="杭州", days=2)
    result = toolkit.plan_and_critique(ctx)
    assert result["isError"] is True


def test_search_poi_requires_city():
    ctx = _session()
    result = toolkit.search_poi(ctx)
    assert result["isError"] is True
