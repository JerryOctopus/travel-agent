from travel_agent.evaluation.chinatravel_adapter import (
    convert_itinerary_to_chinatravel_output,
    extract_query_text,
)
from travel_agent.workflow_rules import extract_profile_rule_based


def test_convert_itinerary_to_chinatravel_output_schema_shape() -> None:
    artifact = {
        "city": "杭州",
        "days": [
            {
                "day_index": 1,
                "stops": [
                    {
                        "poi": {
                            "name": "西湖",
                            "city": "杭州",
                            "category": "scenic",
                            "price_level": "mid",
                        },
                        "start_time": "09:30",
                        "duration_min": 90,
                        "route_from_previous": None,
                    },
                    {
                        "poi": {
                            "name": "西湖餐厅",
                            "city": "杭州",
                            "category": "food",
                            "price_level": "mid",
                        },
                        "start_time": "11:30",
                        "duration_min": 90,
                        "route_from_previous": {
                            "origin_poi_id": "西湖",
                            "destination_poi_id": "西湖餐厅",
                            "distance_km": 2.0,
                            "duration_min": 20,
                            "mode": "public_transport",
                        },
                    },
                ],
            }
        ],
    }

    result = convert_itinerary_to_chinatravel_output(
        artifact,
        query_data={"people_number": 2, "start_city": "上海", "target_city": "杭州"},
        tool_trace=["search_poi", "plan_and_critique"],
    )

    assert result["people_number"] == 2
    assert result["start_city"] == "上海"
    assert result["target_city"] == "杭州"
    assert result["itinerary"][0]["activities"][0]["type"] == "attraction"
    assert result["itinerary"][0]["activities"][1]["type"] == "lunch"
    assert result["itinerary"][0]["activities"][-1]["type"] in {"train", "airplane", "lunch"}


def test_extract_query_text_fallback() -> None:
    assert extract_query_text({"query": "帮我规划杭州三天"}) == "帮我规划杭州三天"
    assert "杭州" in extract_query_text({"target_city": "杭州", "days": 2})


def test_chinatravel_metadata_prefix_is_extracted() -> None:
    profile = extract_profile_rule_based(
        "[当前位置南京,目标位置重庆,旅行人数4,旅行天数4] 我想去重庆吃当地美食"
    )

    assert profile.destination == "重庆"
    assert profile.days == 4
    assert profile.interests == ["food"]
