from travel_agent.constraints import ConstraintSet
from travel_agent.schemas import TravelProfile


def test_constraint_set_from_profile() -> None:
    profile = TravelProfile(
        destination="杭州",
        days=3,
        budget_level="mid",
        budget_limit=5000,
        start_date="2026-10-11",
        companions="一家三口",
        party_size=3,
        interests=["nature", "food"],
        food_preference=["本帮菜"],
        hotel_area="西湖",
        transport_mode="taxi",
        pace="relaxed",
    )

    constraints = ConstraintSet.from_profile(profile)

    assert constraints.destination == "杭州"
    assert constraints.days == 3
    assert constraints.budget_limit == 5000
    assert constraints.start_date == "2026-10-11"
    assert constraints.party_size == 3
    assert constraints.required_interests == ["nature", "food"]
    assert constraints.cuisine_preferences == ["本帮菜"]
    assert constraints.hotel_area == "西湖"
    assert constraints.transport_mode == "taxi"
