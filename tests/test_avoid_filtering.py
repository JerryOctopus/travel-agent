from __future__ import annotations

from travel_agent.agent import toolkit
from travel_agent.agent.session import build_session
from travel_agent.critic import critique_itinerary
from travel_agent.poi_evidence import poi_avoid_match
from travel_agent.reviser import revise_itinerary
from travel_agent.schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryStop,
    POI,
    ScoredPOI,
    TravelProfile,
)


def _poi(
    poi_id: str,
    name: str,
    *,
    category: str = "scenic",
    tags: list[str] | None = None,
    canonical_name: str | None = None,
    aliases: list[str] | None = None,
    source_poi_id: str | None = None,
) -> POI:
    return POI(
        poi_id=poi_id,
        name=name,
        city="杭州",
        category=category,
        lat=30.25,
        lng=120.16,
        rating=4.7,
        popularity=0.9,
        tags=list(tags or []),
        estimated_duration_min=90,
        price_level="mid",
        canonical_name=canonical_name or name,
        aliases=list(aliases or []),
        source_poi_id=source_poi_id or poi_id,
        verification_status="verified",
    )


def _poi_dict(poi: POI) -> dict:
    from travel_agent.agent.serde import poi_to_dict

    return poi_to_dict(poi)


def _rank(profile: TravelProfile, pois: list[POI]):
    ctx = build_session(session_id=f"avoid-{id(profile)}", persist=False)
    ctx.profile = profile
    artifact_id = ctx.store.put("candidates", {"pois": [_poi_dict(poi) for poi in pois]})
    result = toolkit.recommend_candidates(ctx, artifact_ids=[artifact_id])
    ranked = ctx.store.get(result["artifact_id"]) if not result["isError"] else None
    return result, ranked


def test_specific_venue_matches_canonical_variant_alias_and_provider_id() -> None:
    west_lake = _poi(
        "west-lake",
        "西湖景区",
        canonical_name="杭州西湖风景名胜区",
        aliases=["西湖"],
        source_poi_id="amap-west-lake",
    )
    for term in ("西湖", "杭州西湖风景名胜区", "amap-west-lake"):
        match = poi_avoid_match(west_lake, TravelProfile(destination="杭州", avoid=[term]))
        assert match is not None, term
        assert match.term == term


def test_removed_matches_only_specific_provider_identity() -> None:
    west_lake = _poi(
        "west-lake",
        "西湖景区",
        canonical_name="杭州西湖风景名胜区",
        aliases=["西湖"],
        source_poi_id="amap-west-lake",
    )
    for term in ("西湖", "杭州西湖风景名胜区", "amap-west-lake"):
        profile = TravelProfile(
            destination="杭州",
            constraint_state={"removed": [term]},
        )
        assert poi_avoid_match(west_lake, profile) is not None, term


def test_removed_is_entity_only_but_trip_avoid_can_filter_category() -> None:
    museum = _poi("museum", "浙江省博物馆", category="museum", tags=["history"])
    removed_profile = TravelProfile(
        destination="杭州",
        constraint_state={"removed": ["博物馆"]},
    )
    assert poi_avoid_match(museum, removed_profile) is None

    avoid_profile = TravelProfile(
        destination="杭州",
        constraint_state={"avoid": ["博物馆"]},
    )
    match = poi_avoid_match(museum, avoid_profile)
    assert match is not None
    assert match.method == "interest_category"


def test_same_term_in_removed_and_avoid_keeps_the_stronger_semantic_rule() -> None:
    museum = _poi("museum", "浙江省博物馆", category="museum", tags=["history"])
    profile = TravelProfile(
        destination="杭州",
        constraint_state={"removed": ["博物馆"], "avoid": ["博物馆"]},
    )

    match = poi_avoid_match(museum, profile)

    assert match is not None
    assert match.source == "avoid"
    assert match.method == "interest_category"


def test_coffee_avoid_does_not_remove_all_food() -> None:
    cafe = _poi("cafe", "湖畔咖啡店", category="food", tags=["coffee", "cafe"])
    restaurant = _poi("restaurant", "杭帮菜馆", category="food", tags=["local"])
    profile = TravelProfile(destination="杭州", days=1, avoid=["咖啡店"])

    result, ranked = _rank(profile, [cafe, restaurant])

    assert result["isError"] is False
    assert [item["poi"]["poi_id"] for item in ranked["pois"]] == ["restaurant"]
    assert ranked["excluded_by_avoid"][0]["poi_id"] == "cafe"
    assert ranked["excluded_by_avoid"][0]["method"] == "specific_alias"


def test_seaside_and_garden_aliases_do_not_expand_to_all_scenic_pois() -> None:
    beach = _poi("beach", "金沙滩", tags=["beach", "nature"])
    garden = _poi("garden", "拙政园", tags=["园林", "nature"])
    mountain = _poi("mountain", "天目山", tags=["nature"])

    seaside_profile = TravelProfile(destination="杭州", avoid=["海边"])
    assert poi_avoid_match(beach, seaside_profile) is not None
    assert poi_avoid_match(mountain, seaside_profile) is None

    garden_profile = TravelProfile(destination="杭州", avoid=["园林"])
    assert poi_avoid_match(garden, garden_profile) is not None
    assert poi_avoid_match(mountain, garden_profile) is None


def test_ambiguous_lifestyle_preferences_do_not_hard_filter_without_evidence() -> None:
    park = _poi("park", "湖滨公园", tags=["nature"])
    profile = TravelProfile(
        destination="杭州",
        avoid=["早起", "长时间排队的网红点", "连续爬坡", "长楼梯"],
    )
    assert poi_avoid_match(park, profile) is None


def test_mobility_avoid_filters_explicit_mountain_trail_not_flat_walkway() -> None:
    mountain_trail = _poi("mountain-trail", "山地观景步道", tags=["sightseeing"])
    flat_walkway = _poi("waterfront", "滨水步道", tags=["nature"])
    profile = TravelProfile(
        destination="测试城",
        avoid=["连续爬坡", "长楼梯"],
        constraint_state={"avoid": ["连续爬坡", "长楼梯"]},
    )

    match = poi_avoid_match(mountain_trail, profile)

    assert match is not None
    assert match.term == "连续爬坡"
    assert poi_avoid_match(flat_walkway, profile) is None


def test_current_must_visit_overrides_profile_only_avoid() -> None:
    west_lake = _poi("west-lake", "西湖景区", aliases=["西湖"])
    profile = TravelProfile(
        destination="杭州",
        must_visit=["西湖"],
        avoid=["西湖"],
        constraint_state={"must_visit": ["西湖"]},
    )
    assert poi_avoid_match(west_lake, profile) is None


def test_trip_scoped_avoid_overrides_must_visit() -> None:
    west_lake = _poi("west-lake", "西湖景区", aliases=["西湖"])
    profile = TravelProfile(
        destination="杭州",
        must_visit=["西湖"],
        constraint_state={"must_visit": ["西湖"], "avoid": ["西湖"]},
    )
    match = poi_avoid_match(west_lake, profile)
    assert match is not None
    assert match.source == "avoid"


def test_ranked_artifact_excludes_avoided_poi_and_keeps_audit() -> None:
    avoided = _poi("avoided", "西湖景区", aliases=["西湖"])
    kept = _poi("kept", "灵隐寺")
    profile = TravelProfile(destination="杭州", days=1, avoid=["西湖"])

    result, ranked = _rank(profile, [avoided, kept])

    assert result["isError"] is False
    assert [item["poi"]["poi_id"] for item in ranked["pois"]] == ["kept"]
    assert ranked["excluded_by_avoid"] == [
        {
            "poi_id": "avoided",
            "name": "西湖景区",
            "term": "西湖",
            "source": "profile",
            "method": "canonical_identity",
            "matched_value": "西湖",
        }
    ]


def test_all_candidates_filtered_returns_not_found_with_audit() -> None:
    profile = TravelProfile(destination="杭州", days=1, avoid=["购物"])
    result, ranked = _rank(
        profile,
        [_poi("mall", "湖滨银泰商场", category="shopping", tags=["shopping"])],
    )

    assert ranked is None
    assert result["isError"] is True
    assert result["error_code"] == "NOT_FOUND"
    assert result["excluded_by_avoid"][0]["method"] == "interest_category"


def test_critic_uses_same_canonical_avoid_matcher() -> None:
    west_lake = _poi("west-lake", "杭州西湖风景名胜区", aliases=["西湖"])
    itinerary = Itinerary(
        city="杭州",
        days=[
            ItineraryDay(
                day_index=1,
                theme="测试",
                stops=[ItineraryStop(west_lake, "09:00", 90, "测试")],
            )
        ],
        summary="测试",
    )
    profile = TravelProfile(destination="杭州", days=1, avoid=["西湖"])

    result = critique_itinerary(itinerary, profile)

    assert any(issue.code == "avoid_term_included" for issue in result.issues)


def test_reviser_does_not_reintroduce_avoided_ranked_candidate() -> None:
    current = _poi("current", "西湖景区", aliases=["西湖"])
    avoided_replacement = _poi("other-west-lake", "杭州西湖风景名胜区", aliases=["西湖"])
    safe_replacement = _poi("safe", "灵隐寺")
    itinerary = Itinerary(
        city="杭州",
        days=[
            ItineraryDay(
                day_index=1,
                theme="测试",
                stops=[ItineraryStop(current, "09:00", 90, "测试")],
            )
        ],
        summary="测试",
    )
    profile = TravelProfile(destination="杭州", days=1, avoid=["西湖"])

    revised, _result, _notes = revise_itinerary(
        itinerary,
        [
            ScoredPOI(avoided_replacement, 1.0, []),
            ScoredPOI(safe_replacement, 0.8, []),
        ],
        profile,
        critique_itinerary(itinerary, profile),
    )

    revised_names = [stop.poi.name for day in revised.days for stop in day.stops]
    assert "杭州西湖风景名胜区" not in revised_names
    assert "灵隐寺" in revised_names
