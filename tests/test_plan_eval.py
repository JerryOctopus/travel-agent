from travel_agent.evaluation.plan_eval import evaluate_plan_artifact


def _artifact(*, food_time: str = "11:30", include_scenic: bool = True):
    stops = [
        {
            "start_time": food_time,
            "duration_min": 60,
            "poi": {
                "poi_id": "food",
                "name": "本地小吃",
                "city": "杭州",
                "category": "food",
                "lat": 30.25,
                "lng": 120.15,
                "tags": ["food"],
            },
        }
    ]
    if include_scenic:
        stops.insert(
            0,
            {
                "start_time": "09:30",
                "duration_min": 90,
                "poi": {
                    "poi_id": "scenic",
                    "name": "西湖",
                    "city": "杭州",
                    "category": "scenic",
                    "lat": 30.26,
                    "lng": 120.14,
                    "tags": ["nature"],
                },
            },
        )
    return {
        "itinerary": {
            "city": "杭州",
            "days": [
                {
                    "day_index": 1,
                    "stops": stops,
                }
            ],
        }
    }


def test_plan_eval_passes_reasonable_food_and_nature_plan() -> None:
    result = evaluate_plan_artifact(
        _artifact(),
        profile={"pace": "standard", "interests": ["food", "nature"]},
        required_interests=["food", "nature"],
    )

    assert result is not None
    assert result.meal_time_valid is True
    assert result.preference_pass is True
    assert result.final_pass is True


def test_plan_eval_rejects_food_outside_meal_time() -> None:
    result = evaluate_plan_artifact(
        _artifact(food_time="09:30"),
        profile={"pace": "standard", "interests": ["food", "nature"]},
        required_interests=["food", "nature"],
    )

    assert result is not None
    assert result.meal_time_valid is False
    assert "meal_time_invalid:本地小吃:09:30" in result.issues


def test_plan_eval_rejects_missing_nature_after_followup() -> None:
    result = evaluate_plan_artifact(
        _artifact(include_scenic=False),
        profile={"pace": "standard", "interests": ["food", "nature"]},
        required_interests=["food", "nature"],
    )

    assert result is not None
    assert result.preference_pass is False
    assert "interest_not_covered:nature" in result.issues


def test_plan_eval_counts_restaurant_artifact_for_food_preference() -> None:
    artifact = _artifact(include_scenic=True)
    artifact["itinerary"]["days"][0]["stops"] = [
        stop
        for stop in artifact["itinerary"]["days"][0]["stops"]
        if stop["poi"]["category"] != "food"
    ]

    result = evaluate_plan_artifact(
        artifact,
        profile={"pace": "standard", "interests": ["food", "nature"]},
        required_interests=["food", "nature"],
        supporting_artifacts={
            "restaurants": {
                "restaurants": [
                    {
                        "name": "本地小吃",
                        "category": "food",
                        "tags": ["food", "local"],
                    }
                ]
            }
        },
    )

    assert result is not None
    assert result.preference_pass is True
    assert "interest_not_covered:food" not in result.issues


def test_plan_eval_allows_explicit_cross_city_fixed_venue() -> None:
    artifact = _artifact()
    artifact["itinerary"]["days"][0]["stops"][0]["poi"].update(
        {"name": "三星堆博物馆", "city": "广汉"}
    )
    result = evaluate_plan_artifact(
        artifact,
        profile={
            "must_visit": ["三星堆博物馆"],
            "constraint_state": {
                "fixed_events": [{"location": "三星堆博物馆"}],
            },
        },
    )

    assert result is not None
    assert result.poi_city_valid is True


def test_plan_eval_treats_city_suffix_as_same_city() -> None:
    artifact = _artifact()
    for stop in artifact["itinerary"]["days"][0]["stops"]:
        stop["poi"]["city"] = "杭州市"

    result = evaluate_plan_artifact(artifact)

    assert result is not None
    assert result.poi_city_valid is True


def test_plan_eval_rejects_known_walking_distance_over_cap() -> None:
    artifact = _artifact()
    artifact["itinerary"]["days"][0]["stops"][1]["route_from_previous"] = {
        "duration_min": 20,
        "distance_km": 8.0,
        "mode": "public_transport",
        "walking_distance_km": 6.5,
    }
    result = evaluate_plan_artifact(
        artifact,
        profile={"constraint_state": {"max_walking_km_per_day": 6}},
    )

    assert result is not None
    assert result.route_feasible is False
    assert any(issue.startswith("walking_distance_exceeded") for issue in result.issues)


def test_default_pace_route_limit_is_advisory_not_a_hard_constraint() -> None:
    artifact = _artifact()
    artifact["itinerary"]["days"][0]["stops"][1]["route_from_previous"] = {
        "duration_min": 64,
        "distance_km": 12.0,
        "mode": "public_transport",
    }

    result = evaluate_plan_artifact(
        artifact,
        profile={
            "pace": "standard",
            "constraint_state": {"public_transport_required": True},
        },
    )

    assert result is not None
    assert result.route_feasible is True
    assert "route_too_long:day1:64>60" in result.issues


def test_explicit_pace_route_limit_remains_a_constraint() -> None:
    artifact = _artifact()
    artifact["itinerary"]["days"][0]["stops"][1]["route_from_previous"] = {
        "duration_min": 64,
        "distance_km": 12.0,
        "mode": "public_transport",
    }

    result = evaluate_plan_artifact(
        artifact,
        profile={
            "pace": "standard",
            "constraint_state": {"pace": "standard"},
        },
    )

    assert result is not None
    assert result.route_feasible is False
    assert "route_too_long:day1:64>60" in result.issues
