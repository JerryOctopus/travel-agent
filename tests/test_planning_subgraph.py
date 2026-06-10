from __future__ import annotations

from pathlib import Path

from travel_agent.planning_subgraph import plan_and_critique
from travel_agent.providers import build_tool_provider
from travel_agent.recommendation import score_pois
from travel_agent.schemas import TravelProfile

POI_PATH = Path(__file__).resolve().parents[1] / "data" / "seed" / "pois.json"


def _ranked(profile: TravelProfile):
    provider = build_tool_provider(POI_PATH)
    candidates = provider.search_pois(city=profile.destination or "", max_results=50)
    return score_pois(candidates, profile), provider


def test_plan_and_critique_outputs_itinerary():
    profile = TravelProfile(destination="杭州", days=2, interests=["nature", "food"])
    ranked, provider = _ranked(profile)
    result = plan_and_critique(ranked, profile, route_estimator=provider)

    assert result.itinerary.city == "杭州"
    assert len(result.itinerary.days) == 2
    assert result.final_issue_count <= result.original_issue_count


def test_plan_and_critique_closes_the_loop():
    # 单日 + 偏好「博物馆」会让初稿出现 interest_not_covered，reviser 应修正。
    profile = TravelProfile(destination="杭州", days=1, interests=["museum"])
    ranked, provider = _ranked(profile)
    result = plan_and_critique(ranked, profile, route_estimator=provider)

    assert result.original_issue_count >= 1
    assert result.final_issue_count < result.original_issue_count
    assert result.revised is True
    assert result.critic_result.passed is True
