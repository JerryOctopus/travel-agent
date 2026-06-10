from travel_agent.critic import critique_itinerary
from travel_agent.data_loader import load_seed_pois
from travel_agent.planning import build_simple_itinerary
from travel_agent.recommendation import score_pois
from travel_agent.reviser import revise_itinerary
from travel_agent.schemas import TravelProfile
from travel_agent.tools import search_poi
from travel_agent.workflow import DEFAULT_POI_PATH


def test_reviser_adds_missing_must_visit_when_candidate_exists() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["food"],
        must_visit=["故宫"],
        pace="relaxed",
    )
    pois = load_seed_pois(DEFAULT_POI_PATH)
    candidates = search_poi(pois, city="北京", query_tags=profile.interests)
    ranked = score_pois(candidates, profile)
    itinerary = build_simple_itinerary(ranked, profile)
    critic_result = critique_itinerary(itinerary, profile)

    revised, revised_result, notes = revise_itinerary(
        itinerary=itinerary,
        ranked_pois=score_pois(search_poi(pois, city="北京"), profile),
        profile=profile,
        critic_result=critic_result,
    )

    names = [
        stop.poi.name
        for day in revised.days
        for stop in day.stops
    ]
    assert any("故宫" in name for name in names)
    assert notes
    assert "must_visit_missing" not in {issue.code for issue in revised_result.issues}
