from __future__ import annotations

from dataclasses import dataclass, replace

from travel_agent.agent import toolkit
from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.agent.session import reset_task_meta, set_current_task_meta
from travel_agent.agent.turn_lifecycle import prepare_turn
from travel_agent.hybrid_planning.llm_types import StructuredLLMResponse
from travel_agent.orchestration.multi_agent.trace import (
    AgentTraceLog,
    reset_current_trace,
    set_current_trace,
)
from travel_agent.schemas import TravelProfile
from travel_agent.settings import HybridPlanningSettings


@dataclass
class FakeClient:
    payload: dict
    calls: int = 0

    def complete_json(self, **_kwargs) -> StructuredLLMResponse:
        self.calls += 1
        return StructuredLLMResponse(self.payload, model="fake", latency_ms=1)


def enabled_settings(offline_settings, **flags):
    return replace(
        offline_settings,
        hybrid_planning=HybridPlanningSettings(**flags),
    )


def test_intent_normalizer_enriches_only_soft_interests_in_preflight(offline_settings) -> None:
    message = "规划杭州两天行程，预算1000元，西湖必须去，也想在老城里像居民一样逛"
    client = FakeClient(
        {
            "normalized_interests": [
                {
                    "label": "local_life",
                    "source_text": message,
                    "confidence": 0.91,
                    "reason": "偏好本地生活氛围",
                    "polarity": "prefer",
                }
            ],
            "unresolved": [],
        }
    )
    ctx = build_session(persist=False)
    ctx.hybrid_llm_client = client
    settings = enabled_settings(
        offline_settings,
        enable_llm_intent_normalizer=True,
    )
    trace = AgentTraceLog("req-intent")
    token = set_current_trace(trace)
    try:
        prepare_turn(message, ctx, settings, [])
    finally:
        reset_current_trace(token)

    assert "local_life" in ctx.profile.interests
    assert ctx.profile.budget_limit == 1000
    assert "西湖" in ctx.profile.must_visit
    assert client.calls == 1
    assert trace.snapshot()[0]["request_id"] == "req-intent"


def test_preference_policy_reorders_only_filtered_candidate_ids(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="杭州", days=1, pace="relaxed", interests=["nature"]
    )
    ctx.active_task_type = "full_itinerary"
    settings = enabled_settings(
        offline_settings,
        enable_llm_preference_resolver=True,
    )
    ctx.runtime_settings = settings
    searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
    raw = ctx.store.get(searched["artifact_id"])["pois"]
    candidate_ids = [item["poi_id"] for item in raw]
    desired = list(reversed(candidate_ids))
    ctx.hybrid_llm_client = FakeClient(
        {
            "priorities": [
                {
                    "dimension": "user_interest_match",
                    "direction": "maximize",
                    "importance": 0.8,
                    "source": "turn_1",
                    "reason": "自然体验优先",
                }
            ],
            "pace": "relaxed",
            "acceptable_tradeoffs": {},
            "candidate_order": desired,
            "unresolved": [],
            "confidence": 0.9,
        }
    )
    ranked_result = toolkit.recommend_candidates(
        ctx, artifact_ids=[searched["artifact_id"]]
    )
    payload = ctx.store.get(ranked_result["artifact_id"])
    actual = [item["poi"]["poi_id"] for item in payload["pois"]]

    # The model may only express policy; Python owns the deterministic score
    # delta and therefore does not blindly consume model candidate_order.
    assert actual[0] == candidate_ids[0]
    assert set(actual) == set(candidate_ids)
    assert payload["planning_policy"]["source"] == "llm"
    assert payload["soft_preference_actuation"]["final_ranking"] == actual


def test_structured_duration_estimator_is_consumed_before_shared_validation(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(
        destination="杭州", days=1, pace="relaxed", companions="elderly"
    )
    ctx.active_task_type = "full_itinerary"
    ctx.runtime_settings = enabled_settings(
        offline_settings,
        enable_structured_duration_estimator=True,
    )
    search_token = set_current_task_meta({
        "request_id": "req-duration",
        "turn_id": "turn-7",
        "task_id": "attraction-1",
        "agent": "attraction",
    })
    try:
        searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
    finally:
        reset_task_meta(search_token)
    rank_token = set_current_task_meta({
        "request_id": "req-duration",
        "turn_id": "turn-7",
        "task_id": "ranker-1",
        "agent": "planner",
        "artifact_ids": [searched["artifact_id"]],
    })
    try:
        ranked = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])
        assert not ranked.get("isError"), ranked
    finally:
        reset_task_meta(rank_token)
    token = set_current_task_meta({
        "request_id": "req-duration",
        "turn_id": "turn-7",
        "task_id": "planner-1",
        "agent": "planner",
        "artifact_ids": [searched["artifact_id"], ranked["artifact_id"]],
    })
    try:
        planned = toolkit.plan_and_critique(
            ctx, artifact_ids=[searched["artifact_id"], ranked["artifact_id"]]
        )
    finally:
        reset_task_meta(token)
    payload = ctx.store.get(planned["artifact_id"])
    diagnostic = ctx.store.latest("duration_diagnostic")

    assert "duration_estimates" not in payload
    assert diagnostic["candidate_artifact_id"] == planned["artifact_id"]
    assert diagnostic["request_id"] == "req-duration"
    assert diagnostic["turn_id"] == "turn-7"
    assert diagnostic["active_constraint_revision"] == payload["state_version"]["constraint_revision"]
    assert diagnostic["active_constraint_hash"] == payload["state_version"]["constraint_hash"]
    assert diagnostic["duration_estimator_version"] == "activity-duration-v1"
    estimates = {item["candidate_id"]: item for item in diagnostic["estimates"]}
    assert payload["validation_result"]["validated_constraint_hash"] == payload["state_version"]["constraint_hash"]
    for day in payload["itinerary"]["days"]:
        for stop in day["stops"]:
            estimate = estimates[stop["poi"]["poi_id"]]
            assert stop["duration_min"] == estimate["estimated_minutes"]
            assert estimate["range_minutes"][0] <= stop["duration_min"] <= estimate["range_minutes"][1]
            assert estimate["tool_evidence_reference"]["poi_id"] == stop["poi"]["poi_id"]
    assert ctx.store.promote_itinerary(planned["artifact_id"]) is True
    assert ctx.store.latest_current_id("itinerary") == planned["artifact_id"]
    assert ctx.store.latest("duration_diagnostic")["candidate_artifact_id"] == planned["artifact_id"]
    assert ctx.store.latest_current("duration_diagnostic") is None


def test_duration_diagnostic_survives_reviewer_rejection_and_promotion_failure(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1, pace="relaxed")
    ctx.active_task_type = "full_itinerary"
    ctx.runtime_settings = enabled_settings(
        offline_settings, enable_structured_duration_estimator=True
    )
    search_token = set_current_task_meta({
        "request_id": "req-rejected",
        "turn_id": "turn-rejected",
        "task_id": "attraction-rejected",
        "agent": "attraction",
    })
    try:
        searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
    finally:
        reset_task_meta(search_token)
    rank_token = set_current_task_meta({
        "request_id": "req-rejected",
        "turn_id": "turn-rejected",
        "task_id": "ranker-rejected",
        "agent": "planner",
        "artifact_ids": [searched["artifact_id"]],
    })
    try:
        ranked = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])
        assert not ranked.get("isError"), ranked
    finally:
        reset_task_meta(rank_token)
    token = set_current_task_meta({
        "request_id": "req-rejected",
        "turn_id": "turn-rejected",
        "task_id": "planner-rejected",
        "agent": "planner",
        "artifact_ids": [searched["artifact_id"], ranked["artifact_id"]],
    })
    try:
        planned = toolkit.plan_and_critique(
            ctx, artifact_ids=[searched["artifact_id"], ranked["artifact_id"]]
        )
    finally:
        reset_task_meta(token)

    ctx.store.reject_itinerary_candidate(planned["artifact_id"], reason="reviewer_rejected")
    assert ctx.store.promote_itinerary(planned["artifact_id"]) is False
    diagnostic = ctx.store.latest("duration_diagnostic")
    assert diagnostic["candidate_artifact_id"] == planned["artifact_id"]
    assert diagnostic["request_id"] == "req-rejected"
    assert diagnostic["estimates"]


def test_duration_diagnostic_is_fresh_per_candidate_and_not_written_when_disabled(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1)
    ctx.active_task_type = "full_itinerary"
    ctx.runtime_settings = enabled_settings(
        offline_settings, enable_structured_duration_estimator=True
    )
    candidate_ids = []
    for request_id in ("req-one", "req-two"):
        search_token = set_current_task_meta({
            "request_id": request_id,
            "turn_id": f"turn:{request_id}",
            "task_id": f"attraction:{request_id}",
            "agent": "attraction",
        })
        try:
            searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
        finally:
            reset_task_meta(search_token)
        rank_token = set_current_task_meta({
            "request_id": request_id,
            "turn_id": f"turn:{request_id}",
            "task_id": f"ranker:{request_id}",
            "agent": "planner",
            "artifact_ids": [searched["artifact_id"]],
        })
        try:
            ranked = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])
            assert not ranked.get("isError"), ranked
        finally:
            reset_task_meta(rank_token)
        token = set_current_task_meta({
            "request_id": request_id,
            "turn_id": f"turn:{request_id}",
            "task_id": f"planner:{request_id}",
            "agent": "planner",
            "artifact_ids": [searched["artifact_id"], ranked["artifact_id"]],
        })
        try:
            planned = toolkit.plan_and_critique(
                ctx, artifact_ids=[searched["artifact_id"], ranked["artifact_id"]]
            )
        finally:
            reset_task_meta(token)
        candidate_ids.append(planned["artifact_id"])

    diagnostic_ids = ctx.store.artifact_ids_for_request("req-two", kind="duration_diagnostic")
    assert len(diagnostic_ids) == 1
    assert ctx.store.get(diagnostic_ids[0])["candidate_artifact_id"] == candidate_ids[1]
    assert ctx.store.get(diagnostic_ids[0])["candidate_artifact_id"] != candidate_ids[0]

    disabled_ctx = build_session(persist=False)
    disabled_ctx.profile = TravelProfile(destination="杭州", days=1)
    disabled_ctx.active_task_type = "full_itinerary"
    disabled_ctx.runtime_settings = offline_settings
    searched = toolkit.search_poi(disabled_ctx, city="杭州", max_results=8)
    ranked = toolkit.recommend_candidates(disabled_ctx, artifact_ids=[searched["artifact_id"]])
    toolkit.plan_and_critique(
        disabled_ctx, artifact_ids=[searched["artifact_id"], ranked["artifact_id"]]
    )
    assert disabled_ctx.store.latest("duration_diagnostic") is None


def test_duration_diagnostic_survives_validation_failure(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1)
    ctx.active_task_type = "full_itinerary"
    ctx.runtime_settings = enabled_settings(
        offline_settings, enable_structured_duration_estimator=True
    )
    searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
    ranked = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])
    ctx.profile.must_visit = ["不存在于候选集的地点"]

    planned = toolkit.plan_and_critique(
        ctx, artifact_ids=[searched["artifact_id"], ranked["artifact_id"]]
    )

    payload = ctx.store.get(planned["artifact_id"])
    diagnostic = ctx.store.latest("duration_diagnostic")
    assert payload["artifact_status"] == "validation_failure"
    assert diagnostic["candidate_artifact_id"] == planned["artifact_id"]
    assert diagnostic["estimates"]
    assert ctx.store.latest_current_id("itinerary") is None


def test_all_flags_off_preserve_ranked_and_itinerary_payload_shapes(offline_settings) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1, interests=["nature"])
    ctx.active_task_type = "full_itinerary"
    ctx.runtime_settings = offline_settings
    searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
    ranked = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])
    ranked_payload = ctx.store.get(ranked["artifact_id"])
    planned = toolkit.plan_and_critique(
        ctx, artifact_ids=[searched["artifact_id"], ranked["artifact_id"]]
    )
    plan_payload = ctx.store.get(planned["artifact_id"])

    assert "planning_policy" not in ranked_payload
    assert "duration_estimates" not in plan_payload


def test_route_only_request_does_not_invoke_hybrid_llm_modules(offline_settings) -> None:
    client = FakeClient({})
    ctx = build_session(persist=False)
    ctx.hybrid_llm_client = client
    settings = enabled_settings(
        offline_settings,
        enable_llm_intent_normalizer=True,
        enable_llm_preference_resolver=True,
    )

    prepare_turn("从西湖到灵隐寺怎么走", ctx, settings, [])

    assert client.calls == 0


def test_intent_unresolved_does_not_turn_full_plan_into_safe_decline(offline_settings) -> None:
    message = "规划杭州两天完整行程，喜欢一种难以归类的未来城市气质"
    client = FakeClient({
        "normalized_interests": [{
            "label": "citywalk",
            "source_text": "未来城市气质",
            "confidence": 0.2,
            "polarity": "prefer",
        }],
        "unresolved": ["未来城市气质"],
    })
    ctx = build_session(persist=False)
    ctx.hybrid_llm_client = client
    settings = enabled_settings(
        offline_settings, enable_llm_intent_normalizer=True
    )

    prepared = prepare_turn(message, ctx, settings, [])

    assert client.calls == 1
    assert prepared.analysis.task_type.value == "full_itinerary"
    assert prepared.analysis.task_type.value != "safe_decline"
    assert ctx.profile.interests == []


def test_preference_same_candidate_set_is_deduped_and_never_publishes_itinerary(
    offline_settings,
) -> None:
    ctx = build_session(persist=False)
    ctx.profile = TravelProfile(destination="杭州", days=1, interests=["nature"])
    ctx.active_task_type = "full_itinerary"
    ctx.active_delivery_intent = "rebuild_now"
    ctx.runtime_settings = enabled_settings(
        offline_settings, enable_llm_preference_resolver=True
    )
    searched = toolkit.search_poi(ctx, city="杭州", max_results=8)
    raw = ctx.store.get(searched["artifact_id"])["pois"]
    candidate_ids = [item["poi_id"] for item in raw]
    client = FakeClient({
        "priorities": [{
            "dimension": "user_interest_match",
            "direction": "maximize",
            "importance": 0.9,
            "source": "nature",
            "reason": "兴趣匹配",
        }],
        "pace": "normal",
        "acceptable_tradeoffs": {},
        "candidate_order": candidate_ids,
        "unresolved": [],
        "confidence": 0.9,
    })
    ctx.hybrid_llm_client = client

    first = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])
    second = toolkit.recommend_candidates(ctx, artifact_ids=[searched["artifact_id"]])

    assert client.calls == 1
    assert not first.get("isError") and not second.get("isError")
    assert ctx.store.latest_current_id("itinerary") is None


def test_intent_enabled_but_ineligible_has_zero_business_side_effect(offline_settings) -> None:
    base = build_session(persist=False)
    disabled = base.clone_isolated()
    enabled = base.clone_isolated()
    client = FakeClient({})
    enabled.hybrid_llm_client = client
    enabled_settings_value = enabled_settings(
        offline_settings, enable_llm_intent_normalizer=True
    )
    message = "规划杭州两天完整行程，预算1000元，西湖必须去"

    disabled_prepared = prepare_turn(message, disabled, offline_settings, [])
    enabled_prepared = prepare_turn(message, enabled, enabled_settings_value, [])

    assert client.calls == 0
    assert disabled_prepared.analysis.constraint_state == enabled_prepared.analysis.constraint_state
    assert disabled_prepared.turn_inputs == enabled_prepared.turn_inputs
    assert disabled.profile == enabled.profile


def test_preference_enabled_but_ineligible_preserves_planner_input_and_fingerprint(offline_settings) -> None:
    base = build_session(persist=False)
    base.profile = TravelProfile(destination="杭州", days=1)
    base.active_task_type = "full_itinerary"
    searched = toolkit.search_poi(base, city="杭州", max_results=8)
    disabled = base.clone_isolated()
    enabled = base.clone_isolated()
    disabled.runtime_settings = offline_settings
    client = FakeClient({})
    enabled.hybrid_llm_client = client
    enabled.runtime_settings = enabled_settings(
        offline_settings, enable_llm_preference_resolver=True
    )

    off_result = toolkit.recommend_candidates(
        disabled, artifact_ids=[searched["artifact_id"]]
    )
    on_result = toolkit.recommend_candidates(
        enabled, artifact_ids=[searched["artifact_id"]]
    )
    off_payload = disabled.store.get(off_result["artifact_id"])
    on_payload = enabled.store.get(on_result["artifact_id"])
    from travel_agent.artifact_policy import constraint_fingerprint

    assert client.calls == 0
    assert off_payload == on_payload
    assert "planning_policy" not in on_payload
    assert constraint_fingerprint("ranked", disabled.profile.constraint_state, off_payload) == constraint_fingerprint(
        "ranked", enabled.profile.constraint_state, on_payload
    )


def test_both_llm_flags_ineligible_do_not_change_full_plan_business_inputs(offline_settings) -> None:
    base = build_session(persist=False)
    disabled = base.clone_isolated()
    enabled = base.clone_isolated()
    client = FakeClient({})
    enabled.hybrid_llm_client = client
    flags = enabled_settings(
        offline_settings,
        enable_llm_intent_normalizer=True,
        enable_llm_preference_resolver=True,
    )
    message = "请生成杭州一日完整行程"

    off_prepared = prepare_turn(message, disabled, offline_settings, [])
    on_prepared = prepare_turn(message, enabled, flags, [])

    assert client.calls == 0
    assert off_prepared.turn_inputs == on_prepared.turn_inputs
    assert disabled.profile == enabled.profile


def test_each_production_turn_flushes_a_fresh_request_trace(offline_settings) -> None:
    ctx = build_session(persist=False)
    first = run_production_turn("你好", ctx=ctx, settings=offline_settings)
    second = run_production_turn("谢谢", ctx=ctx, settings=offline_settings)

    trace_ids = ctx.store.artifact_ids_for_request(first.request_id, kind="agent_trace")
    second_ids = ctx.store.artifact_ids_for_request(second.request_id, kind="agent_trace")
    assert first.request_id != second.request_id
    assert len(trace_ids) == 1
    assert len(second_ids) == 1
    assert ctx.store.get(trace_ids[0])["request_id"] == first.request_id
    assert ctx.store.get(second_ids[0])["request_id"] == second.request_id
