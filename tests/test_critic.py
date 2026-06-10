from travel_agent.critic import critique_itinerary
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    RouteInfo,
    TravelProfile,
)


def test_critic_passes_when_itinerary_covers_interests() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["history", "food"],
        pace="relaxed",
    )
    itinerary = Itinerary(
        city="北京",
        summary="北京1天轻松行程",
        days=[
            ItineraryDay(
                day_index=1,
                theme="历史与美食",
                stops=[
                    ItineraryStop(
                        poi=_poi("故宫博物院", "culture", ["history", "culture"]),
                        start_time="09:30",
                        duration_min=150,
                        note="历史文化核心景点",
                    ),
                    ItineraryStop(
                        poi=_poi("南锣鼓巷", "food", ["food", "local"]),
                        start_time="13:30",
                        duration_min=90,
                        note="本地美食体验",
                    ),
                ],
            )
        ],
    )

    result = critique_itinerary(itinerary, profile)

    assert result.passed is True
    assert result.issues == []


def test_critic_reports_missing_must_visit_and_uncovered_interest() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["food"],
        must_visit=["故宫"],
        pace="relaxed",
    )
    itinerary = Itinerary(
        city="北京",
        summary="北京1天轻松行程",
        days=[
            ItineraryDay(
                day_index=1,
                theme="自然体验",
                stops=[
                    ItineraryStop(
                        poi=_poi("景山公园", "scenic", ["nature"]),
                        start_time="09:30",
                        duration_min=90,
                        note="轻松游览",
                    )
                ],
            )
        ],
    )

    result = critique_itinerary(itinerary, profile)
    codes = {issue.code for issue in result.issues}

    assert result.passed is False
    assert "interest_not_covered" in codes
    assert "must_visit_missing" in codes


def test_critic_reports_route_too_long() -> None:
    profile = TravelProfile(
        destination="北京",
        days=1,
        interests=["history"],
        pace="relaxed",
    )
    itinerary = Itinerary(
        city="北京",
        summary="北京1天轻松行程",
        days=[
            ItineraryDay(
                day_index=1,
                theme="历史文化",
                stops=[
                    ItineraryStop(
                        poi=_poi("故宫博物院", "culture", ["history"]),
                        start_time="09:30",
                        duration_min=120,
                        note="历史文化核心景点",
                    ),
                    ItineraryStop(
                        poi=_poi("八达岭长城", "culture", ["history"]),
                        start_time="13:30",
                        duration_min=150,
                        note="历史文化核心景点",
                        route_from_previous=RouteInfo(
                            origin_poi_id="故宫博物院",
                            destination_poi_id="八达岭长城",
                            distance_km=65.0,
                            duration_min=95,
                            mode="public_transport",
                            source="test",
                        ),
                    ),
                ],
            )
        ],
    )

    result = critique_itinerary(itinerary, profile)
    codes = {issue.code for issue in result.issues}

    assert result.passed is False
    assert "route_too_long" in codes
    assert "daily_route_too_long" in codes


def _poi(name: str, category: str, tags: list[str]) -> POI:
    return POI(
        poi_id=name,
        name=name,
        city="北京",
        category=category,
        lat=39.9,
        lng=116.4,
        rating=4.5,
        popularity=0.8,
        tags=tags,
        estimated_duration_min=90,
        price_level="mid",
    )
