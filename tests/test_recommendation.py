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


def test_constraint_boost_requires_canonical_entity_identity() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=2,
        must_visit=["西湖", "浙江省博物馆"],
    )
    scenic = POI(
        "lake", "杭州西湖风景名胜区", "杭州市", "scenic", 30.2, 120.1,
        4.8, 0.9, ["nature"], 120, "free", source="amap",
        canonical_name="杭州西湖风景名胜区", entity_type="attraction",
    )
    named_museum = POI(
        "lake-museum", "西湖博物馆", "杭州市", "museum", 30.2, 120.1,
        4.8, 0.9, ["history"], 90, "free", source="amap",
        canonical_name="西湖博物馆", entity_type="museum",
    )
    specialty = POI(
        "geology", "浙江省地质博物馆", "杭州市", "museum", 30.2, 120.1,
        4.8, 0.9, ["history"], 90, "free", source="amap",
        canonical_name="浙江省地质博物馆", entity_type="museum",
    )

    ranked = {
        item.poi.poi_id: item
        for item in score_pois([named_museum, specialty, scenic], profile)
    }

    assert "满足强约束" in ranked["lake"].reasons
    assert "满足强约束" not in ranked["lake-museum"].reasons
    assert "满足强约束" not in ranked["geology"].reasons
