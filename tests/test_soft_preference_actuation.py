from __future__ import annotations

from dataclasses import dataclass, replace

from travel_agent.agent import toolkit
from travel_agent.agent.serde import poi_to_dict
from travel_agent.agent.session import build_session, reset_task_meta, set_current_task_meta
from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.hybrid_planning.preference_resolver import PlanningPolicy, PolicyPriority
from travel_agent.hybrid_planning.soft_preference_actuation import (
    apply_preference_score,
    build_candidate_evidence_matrix,
    build_retrieval_plan,
    preference_candidates_from_matrix,
)
from travel_agent.schemas import POI, ScoredPOI, TravelProfile
from travel_agent.settings import HybridPlanningSettings


@dataclass
class FakeClient:
    payload: dict
    calls: int = 0

    def complete_json(self, **_kwargs) -> StructuredLLMResponse:
        self.calls += 1
        return StructuredLLMResponse(self.payload, model="fixture", latency_ms=1)


def _settings(offline_settings, **flags):
    return replace(
        offline_settings,
        hybrid_planning=HybridPlanningSettings(**flags),
    )


def _poi(
    poi_id: str,
    *,
    tag: str,
    rating: float = 5.0,
    popularity: float = 1.0,
    lat: float = 30.0,
    lng: float = 120.0,
    average_cost: float | None = None,
) -> POI:
    return POI(
        poi_id=poi_id,
        name=f"候选{poi_id}",
        city="测试城",
        category="attraction",
        lat=lat,
        lng=lng,
        rating=rating,
        popularity=popularity,
        tags=[tag],
        estimated_duration_min=120,
        price_level="low",
        average_cost=average_cost,
        source="seed",
        source_poi_id=poi_id,
        verification_status="verified",
    )


def _normalized_context() -> dict:
    return {
        "normalized_interests": [
            {
                "label": "industrial_heritage",
                "source_text": "旧厂房改造成公共空间",
                "polarity": "prefer",
                "confidence": 0.91,
                "source_turn": "turn_3",
                "basis": "llm",
            },
            {
                "label": "commercialized",
                "source_text": "不要太商业化",
                "polarity": "avoid",
                "confidence": 0.88,
                "source_turn": "turn_3",
                "basis": "llm",
            },
        ]
    }


def test_retrieval_plan_preserves_raw_phrase_bounds_queries_and_never_recalls_avoid() -> None:
    plan = build_retrieval_plan(
        _normalized_context(),
        destination="测试城",
        request_id="req-1",
        turn_id="turn-3",
        constraint_revision=2,
        constraint_hash="hash-2",
    )

    queries = [*plan["raw_queries"], *plan["taxonomy_queries"]]
    assert queries[0]["query"] == "旧厂房改造成公共空间"
    assert any("工业遗产" in item["query"] for item in plan["taxonomy_queries"])
    assert len(queries) <= plan["query_budget"] == 4
    assert all(item["taxonomy_label"] != "commercialized" for item in queries)
    assert plan["avoid_semantics"] == [{
        "raw_interest": "不要太商业化",
        "taxonomy_label": "commercialized",
        "retrieval_target": False,
    }]
    assert len({item["query_fingerprint"] for item in queries}) == len(queries)


def test_evidence_matrix_and_score_delta_are_deterministic_bounded_and_fail_closed() -> None:
    profile = TravelProfile(
        destination="测试城",
        days=1,
        interests=["industrial_heritage"],
        constraint_state={"normalized_interest_context": _normalized_context()},
    )
    ranked = [
        ScoredPOI(_poi("match", tag="industrial_heritage"), 0.60, []),
        ScoredPOI(_poi("other", tag="nature"), 0.64, []),
        ScoredPOI(_poi("unknown", tag="uncontrolled-provider-tag"), 0.65, []),
    ]
    source_id = "candidates_fixture"
    records = [{
        "artifact_id": source_id,
        "kind": "candidates",
        "payload": {"pois": [poi_to_dict(item.poi) for item in ranked]},
    }]
    matrix = build_candidate_evidence_matrix(
        ranked,
        records,
        profile,
        request_id="req-2",
        turn_id="turn-4",
        constraint_revision=1,
        constraint_hash="hash-1",
    )
    assert matrix["candidates"]["match"]["dimensions"]["user_interest_match"]["value"] == 1.0
    assert matrix["candidates"]["other"]["dimensions"]["user_interest_match"]["value"] == 0.0
    assert matrix["candidates"]["unknown"]["dimensions"]["user_interest_match"]["evidence_status"] == "unknown"

    policy = PlanningPolicy(
        priorities=[PolicyPriority(
            "user_interest_match", "maximize", 1.0, "turn_4", "兴趣优先"
        )],
        candidate_order=["match", "other", "unknown"],
        source="llm",
    )
    first, first_trace = apply_preference_score(ranked, matrix, policy)
    second, second_trace = apply_preference_score(ranked, matrix, policy)

    assert [item.poi.poi_id for item in first] == [item.poi.poi_id for item in second]
    assert first_trace == second_trace
    assert first_trace["match"]["total_soft_delta"] == 0.12
    assert first_trace["unknown"]["total_soft_delta"] == 0.0
    assert max(item["total_soft_delta"] for item in first_trace.values()) <= 0.25


def test_route_dimensions_use_one_verified_anchor_and_normalize_walking_to_minutes() -> None:
    profile = TravelProfile(destination="测试城", days=1, interests=["nature"])
    ranked = [
        ScoredPOI(_poi("near", tag="nature"), 0.60, []),
        ScoredPOI(_poi("far", tag="culture"), 0.62, []),
    ]
    records = [{
        "artifact_id": "candidates-route",
        "kind": "candidates",
        "payload": {"pois": [poi_to_dict(item.poi) for item in ranked]},
    }]
    for candidate_id, duration, walking in (("near", 10, 0.5), ("far", 30, 1.5)):
        records.append({
            "artifact_id": f"route-{candidate_id}",
            "kind": "routes",
            "payload": {
                "origin_poi_id": "hotel-anchor",
                "destination_poi_id": candidate_id,
                "duration_min": duration,
                "walking_distance_km": walking,
                "transfer_count": 0 if candidate_id == "near" else 2,
                "evidence_status": "provider_verified",
            },
        })

    matrix = build_candidate_evidence_matrix(
        ranked,
        records,
        profile,
        request_id="req-route",
        turn_id="turn-route",
        constraint_revision=1,
        constraint_hash="route-hash",
    )

    near = matrix["candidates"]["near"]["dimensions"]
    assert near["commute_time"]["value"] == 10
    assert near["commute_time"]["evidence_status"] == "provider_verified"
    assert near["walking_load"]["value"] == 6.0
    assert near["walking_load"]["unit"] == "minutes"
    assert near["transfer_count"]["evidence_ids"] == ["route-near"]


def test_intent_retrieval_bridge_executes_same_plan_once_and_records_provenance(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        interests=["industrial_heritage"],
        constraint_state={"normalized_interest_context": _normalized_context()},
    )
    ctx.active_task_type = "full_itinerary"
    ctx.hybrid_request_scope = "req-retrieval"
    ctx.runtime_settings = _settings(
        offline_settings, enable_llm_intent_normalizer=True
    )
    delegate = ctx.provider

    class SpyProvider:
        def __init__(self):
            self.calls: list[tuple[str, ...]] = []

        def search_pois(self, city, query_tags=None, category=None, max_results=20):
            key = tuple(query_tags or ())
            self.calls.append(key)
            if key and (
                "旧厂房改造成公共空间" in key[0]
                or "工业遗产" in key[0]
            ):
                return [_poi("industrial-space", tag="industrial_heritage")]
            return delegate.search_pois(city, query_tags, category, max_results)

        def __getattr__(self, name):
            return getattr(delegate, name)

    spy = SpyProvider()
    ctx.provider = spy
    token = set_current_task_meta({
        "request_id": "req-retrieval",
        "turn_id": "turn-3",
        "task_id": "attraction-1",
        "agent": "attraction",
    })
    try:
        first = toolkit.search_poi(ctx, city="测试城", max_results=8)
        second = toolkit.search_poi(ctx, city="测试城", max_results=8)
    finally:
        reset_task_meta(token)

    supplemental_calls = [
        call for call in spy.calls
        if call and ("旧厂房" in call[0] or "工业遗产" in call[0])
    ]
    assert len(supplemental_calls) == 2
    first_payload = ctx.store.get(first["artifact_id"])
    second_payload = ctx.store.get(second["artifact_id"])
    assert first_payload["retrieval_plan"]["execution_status"] == "executed"
    assert second_payload["retrieval_plan"]["execution_status"] == "cache_hit"
    provenance = first_payload["retrieval_provenance"]["industrial-space"]
    assert {item["taxonomy_label"] for item in provenance} == {"industrial_heritage"}
    assert all(item["tool_evidence_id"] for item in provenance)


def test_recommend_candidates_consumes_bound_route_evidence_for_commute_policy(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        constraint_state={"preference": "通勤优先"},
    )
    ctx.active_task_type = "full_itinerary"
    ctx.active_delivery_intent = "rebuild_now"
    ctx.hybrid_request_scope = "req-commute"
    ctx.runtime_settings = _settings(
        offline_settings, enable_llm_preference_resolver=True
    )
    pois = [
        _poi("near", tag="nature", rating=4.0, popularity=0.7),
        _poi("far", tag="culture", rating=5.0, popularity=1.0),
    ]
    candidate_id = ctx.store.put(
        "candidates", {"city": "测试城", "pois": [poi_to_dict(poi) for poi in pois]}
    )
    route_ids = []
    for candidate, duration in (("near", 8), ("far", 35)):
        route_ids.append(ctx.store.put("routes", {
            "origin_poi_id": "hotel-anchor",
            "destination_poi_id": candidate,
            "duration_min": duration,
            "walking_distance_km": 0.5,
            "evidence_status": "provider_verified",
        }))
    ctx.hybrid_llm_client = FakeClient({
        "priorities": [{
            "dimension": "commute_time",
            "direction": "minimize",
            "importance": 1.0,
            "source": "通勤优先",
            "reason": "减少通勤",
        }],
        "pace": "normal",
        "acceptable_tradeoffs": {},
        "candidate_order": ["near", "far"],
        "unresolved": [],
        "confidence": 0.95,
    })

    ranked = toolkit.recommend_candidates(
        ctx, artifact_ids=[candidate_id, *route_ids]
    )
    payload = ctx.store.get(ranked["artifact_id"])
    contract = payload["soft_preference_actuation"]

    assert contract["final_ranking"][0] == "near"
    commute = contract["candidate_evidence"]["near"]["dimensions"]["commute_time"]
    assert commute["value"] == 8
    assert commute["evidence_status"] == "provider_verified"
    assert commute["evidence_ids"] == [route_ids[0]]

    planned = toolkit.plan_and_critique(
        ctx, artifact_ids=[candidate_id, ranked["artifact_id"], *route_ids]
    )
    plan_payload = ctx.store.get(planned["artifact_id"])
    plan_contract = plan_payload["soft_preference_actuation"]
    assert plan_contract["selected_candidate_ids"][0] == "near"
    assert plan_contract["fingerprints"]["planner_input_fingerprint"]
    assert plan_contract["fingerprints"]["final_artifact_fingerprint"]
    assert plan_contract["funnel"]["selected_candidates"] >= 1


def test_production_planner_consumes_adjusted_ranking_and_persists_selection_provenance(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        pace="relaxed",
        interests=["industrial_heritage"],
        avoid=["候选popular-c"],
        constraint_state={"normalized_interest_context": _normalized_context()},
    )
    ctx.active_task_type = "full_itinerary"
    ctx.active_delivery_intent = "rebuild_now"
    ctx.hybrid_request_scope = "req-actuation"
    ctx.runtime_settings = _settings(
        offline_settings, enable_llm_preference_resolver=True
    )
    pois = [
        _poi("interest", tag="industrial_heritage", rating=0.0, popularity=0.0),
        _poi("popular-a", tag="nature"),
        _poi("popular-b", tag="culture", rating=4.9, popularity=0.98),
        _poi("popular-c", tag="history", rating=4.8, popularity=0.96),
    ]
    candidate_id = ctx.store.put("candidates", {"city": "测试城", "pois": [poi_to_dict(poi) for poi in pois]})
    ctx.remember_pois(pois)
    ctx.hybrid_llm_client = FakeClient({
        "priorities": [{
            "dimension": "user_interest_match",
            "direction": "maximize",
            "importance": 1.0,
            "source": "旧厂房改造成公共空间",
            "reason": "兴趣匹配优先",
        }],
        "pace": "relaxed",
        "acceptable_tradeoffs": {},
        "candidate_order": [poi.poi_id for poi in pois[:3]],
        "unresolved": [],
        "confidence": 0.95,
    })

    ranked_result = toolkit.recommend_candidates(ctx, artifact_ids=[candidate_id])
    ranked_payload = ctx.store.get(ranked_result["artifact_id"])
    contract = ranked_payload["soft_preference_actuation"]
    assert contract["base_ranking"][0] != "interest"
    assert contract["final_ranking"][0] == "interest"
    assert contract["causal_diagnostic"]["ranking_changed"] is True
    assert "popular-c" not in contract["candidate_evidence"]
    assert set(contract["fingerprints"]) == {
        "intent_input_fingerprint",
        "normalized_interest_fingerprint",
        "retrieval_plan_fingerprint",
        "tool_input_fingerprints",
        "retrieved_candidate_set_fingerprint",
        "hard_filtered_candidate_set_fingerprint",
        "evidence_matrix_fingerprint",
        "preference_policy_fingerprint",
        "score_trace_fingerprint",
        "adjusted_ranking_fingerprint",
        "planner_input_fingerprint",
        "final_artifact_fingerprint",
    }
    assert contract["funnel"] == {
        "retrieval_queries": 0,
        "retrieved_candidates": 3,
        "hard_legal_candidates": 3,
        "evidence_complete_candidates": 3,
        "preference_eligible_candidates": 3,
        "planner_candidates": 3,
        "selected_candidates": 0,
    }

    planned = toolkit.plan_and_critique(
        ctx, artifact_ids=[candidate_id, ranked_result["artifact_id"]]
    )
    plan_payload = ctx.store.get(planned["artifact_id"])
    plan_contract = plan_payload["soft_preference_actuation"]
    assert plan_contract["selected_candidate_ids"][0] == "interest"
    assert plan_contract["final_selection_provenance"]["all_selected_from_legal_ranking"] is True
    assert plan_contract["request_id"] == "req-actuation"
    assert plan_contract["constraint_hash"] == plan_payload["state_version"]["constraint_hash"]

    reworked = toolkit.plan_and_critique(ctx, artifact_ids=[planned["artifact_id"]])
    reworked_payload = ctx.store.get(reworked["artifact_id"])
    assert reworked_payload["soft_preference_actuation"]["final_ranking"][0] == "interest"


def test_known_over_budget_candidate_never_reaches_resolver_ranking_or_planner(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="测试城",
        days=1,
        party_size=2,
        budget_limit=500,
        constraint_state={"preference": "价格优先"},
    )
    ctx.active_task_type = "full_itinerary"
    ctx.active_delivery_intent = "rebuild_now"
    ctx.hybrid_request_scope = "req-budget-gate"
    ctx.runtime_settings = _settings(
        offline_settings, enable_llm_preference_resolver=True
    )
    pois = [
        _poi("cheap-a", tag="nature", average_cost=100),
        _poi("cheap-b", tag="culture", average_cost=120, rating=4.9),
        _poi("over-budget", tag="history", average_cost=400),
    ]
    candidate_id = ctx.store.put(
        "candidates", {"city": "测试城", "pois": [poi_to_dict(poi) for poi in pois]}
    )
    ctx.remember_pois(pois)
    ctx.hybrid_llm_client = FakeClient({
        "priorities": [{
            "dimension": "price",
            "direction": "minimize",
            "importance": 1.0,
            "source": "价格优先",
            "reason": "控制预算",
        }],
        "pace": "normal",
        "acceptable_tradeoffs": {},
        "candidate_order": ["cheap-a", "cheap-b"],
        "unresolved": [],
        "confidence": 0.95,
    })

    ranked_result = toolkit.recommend_candidates(ctx, artifact_ids=[candidate_id])
    ranked_payload = ctx.store.get(ranked_result["artifact_id"])
    contract = ranked_payload["soft_preference_actuation"]

    assert contract["hard_filter"]["over-budget"]["status"] == "failed"
    assert contract["hard_filter"]["over-budget"]["reason_codes"] == [
        "candidate_cost_exceeds_total_budget"
    ]
    assert "over-budget" not in contract["candidate_evidence"]
    assert "over-budget" not in contract["final_ranking"]
    assert "over-budget" not in [
        item["candidate_id"]
        for item in preference_candidates_from_matrix({
            "candidates": contract["candidate_evidence"]
        })
    ]
    assert "over-budget" not in [
        item["poi"]["poi_id"] for item in ranked_payload["pois"]
    ]
    assert contract["funnel"]["retrieved_candidates"] == 3
    assert contract["funnel"]["hard_legal_candidates"] == 2

    planned = toolkit.plan_and_critique(
        ctx, artifact_ids=[candidate_id, ranked_result["artifact_id"]]
    )
    plan_payload = ctx.store.get(planned["artifact_id"])
    plan_contract = plan_payload["soft_preference_actuation"]
    assert "over-budget" not in plan_contract["selected_candidate_ids"]
    assert plan_contract["final_selection_provenance"][
        "all_selected_from_legal_ranking"
    ] is True
