from __future__ import annotations

from pathlib import Path

from travel_agent.critic import critique_itinerary
from travel_agent.llm_extractor import RuleBasedTravelProfileExtractor, TravelProfileExtractor
from travel_agent.planning import apply_structured_schedule_constraints, build_simple_itinerary
from travel_agent.providers import TravelToolProvider, build_tool_provider
from travel_agent.rag import retrieve_destination_knowledge
from travel_agent.recommendation import score_pois
from travel_agent.reviser import revise_itinerary
from travel_agent.schemas import TravelProfile, WorkflowResult
from travel_agent.workflow_rules import extract_profile_rule_based, normalize_interests


DEFAULT_POI_PATH = Path(__file__).resolve().parents[2] / "data" / "seed" / "pois.json"
DEFAULT_EXTRACTOR = RuleBasedTravelProfileExtractor()


def run_mvp_workflow(
    user_message: str,
    existing_profile: TravelProfile | None = None,
    extractor: TravelProfileExtractor = DEFAULT_EXTRACTOR,
    poi_path: Path | str = DEFAULT_POI_PATH,
    tool_provider: TravelToolProvider | None = None,
) -> WorkflowResult:
    extracted_profile = extractor.extract(user_message)
    profile = (
        merge_profile(existing_profile, extracted_profile)
        if existing_profile
        else extracted_profile
    )
    missing = profile.missing_required_fields()
    if missing:
        return WorkflowResult(
            profile=profile,
            ranked_pois=[],
            itinerary=None,
            clarification_question=_make_clarification_question(missing),
        )

    provider = tool_provider or build_tool_provider(poi_path)
    knowledge_chunks = retrieve_destination_knowledge(
        city=profile.destination or "",
        query=user_message or " ".join(profile.interests),
    )
    weather = provider.get_weather(profile.destination or "")
    preference_candidates = provider.search_pois(
        city=profile.destination or "",
        query_tags=profile.interests,
        max_results=50,
    )
    city_candidates = provider.search_pois(
        city=profile.destination or "",
        max_results=50,
    )
    candidates = _merge_pois(preference_candidates, city_candidates)
    ranked = score_pois(candidates, profile)
    itinerary = build_simple_itinerary(ranked, profile, route_estimator=provider)
    critic_result = critique_itinerary(itinerary, profile)
    revised = False
    revision_notes: list[str] = []
    if not critic_result.passed:
        itinerary, critic_result, revision_notes = revise_itinerary(
            itinerary=itinerary,
            ranked_pois=ranked,
            profile=profile,
            critic_result=critic_result,
        )
        itinerary = itinerary.__class__(
            city=itinerary.city,
            days=apply_structured_schedule_constraints(
                list(itinerary.days), ranked, profile, provider
            ),
            summary=itinerary.summary,
        )
        critic_result = critique_itinerary(itinerary, profile)
        revised = bool(revision_notes)
    return WorkflowResult(
        profile=profile,
        ranked_pois=ranked,
        itinerary=itinerary,
        critic_result=critic_result,
        knowledge_chunks=knowledge_chunks,
        weather=weather,
        revised=revised,
        revision_notes=revision_notes,
    )


def merge_l3_preferences(base: TravelProfile, l3: TravelProfile) -> TravelProfile:
    """合并 L3 跨会话画像中的长期偏好，不注入目的地/天数等单次行程槽位。"""
    if not any(
        (
            l3.budget_level,
            l3.pace != "standard",
            l3.companions,
            l3.food_preference,
            l3.must_visit,
            l3.avoid,
            l3.hotel_area,
            l3.transport_mode != "public_transport",
        )
    ):
        return base
    return TravelProfile(
        destination=base.destination,
        days=base.days,
        start_date=base.start_date,
        budget_level=l3.budget_level or base.budget_level,
        budget_limit=base.budget_limit,
        interests=list(base.interests),
        companions=l3.companions or base.companions,
        party_size=base.party_size,
        pace=l3.pace if l3.pace != "standard" else base.pace,
        hotel_area=l3.hotel_area or base.hotel_area,
        food_preference=_merge_unique(base.food_preference, l3.food_preference),
        must_visit=_merge_unique(base.must_visit, l3.must_visit),
        avoid=_merge_unique(base.avoid, l3.avoid),
        transport_mode=(
            l3.transport_mode
            if l3.transport_mode != "public_transport"
            else base.transport_mode
        ),
        constraint_state=dict(base.constraint_state),
    )


def merge_profile(base: TravelProfile, update: TravelProfile) -> TravelProfile:
    """合并多轮对话中的出行画像，新一轮信息优先。"""
    explicit_public_transport = (
        update.constraint_state.get("self_driving_allowed") is False
        or update.constraint_state.get("public_transport_required") is True
    )
    return TravelProfile(
        destination=update.destination or base.destination,
        days=update.days or base.days,
        start_date=update.start_date or base.start_date,
        budget_level=update.budget_level or base.budget_level,
        budget_limit=update.budget_limit or base.budget_limit,
        interests=normalize_interests(_merge_unique(base.interests, update.interests)),
        companions=update.companions or base.companions,
        party_size=update.party_size or base.party_size,
        pace=update.pace if update.pace != "standard" else base.pace,
        hotel_area=update.hotel_area or base.hotel_area,
        food_preference=_merge_unique(base.food_preference, update.food_preference),
        must_visit=_merge_unique(base.must_visit, update.must_visit),
        avoid=_merge_unique(base.avoid, update.avoid),
        transport_mode=(
            "public_transport"
            if explicit_public_transport
            else (
                update.transport_mode
                if update.transport_mode != "public_transport"
                else base.transport_mode
            )
        ),
        constraint_state={**base.constraint_state, **update.constraint_state},
    )


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _merge_pois(left, right):
    merged = list(left)
    seen = {poi.poi_id for poi in merged}
    for poi in right:
        if poi.poi_id not in seen:
            merged.append(poi)
            seen.add(poi.poi_id)
    return merged


def _make_clarification_question(missing: list[str]) -> str:
    if "destination" in missing and "days" in missing:
        return "你想去哪个城市，计划玩几天？"
    if "destination" in missing:
        return "你想去哪个城市旅行？"
    if "days" in missing:
        return "你计划玩几天？"
    return "我还需要补充一些旅行信息。"
