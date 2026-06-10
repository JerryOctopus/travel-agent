from travel_agent.recommendation import score_pois
from travel_agent.schemas import POI, TravelProfile


def test_interest_match_ranks_relevant_poi_first() -> None:
    profile = TravelProfile(
        destination="Hangzhou",
        days=2,
        interests=["nature"],
        pace="relaxed",
    )
    pois = [
        POI(
            poi_id="museum",
            name="Museum",
            city="Hangzhou",
            category="museum",
            lat=0,
            lng=0,
            rating=4.8,
            popularity=0.8,
            tags=["culture"],
            estimated_duration_min=90,
            price_level="free",
        ),
        POI(
            poi_id="lake",
            name="Lake",
            city="Hangzhou",
            category="scenic",
            lat=0,
            lng=0,
            rating=4.7,
            popularity=0.9,
            tags=["nature"],
            estimated_duration_min=120,
            price_level="free",
        ),
    ]

    ranked = score_pois(pois, profile)

    assert ranked[0].poi.poi_id == "lake"
