from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from travel_agent.agent import toolkit
from travel_agent.agent.serde import poi_to_dict
from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session, current_task_meta
from travel_agent.agent.turn_analysis import DeliveryIntent
from travel_agent.agent.turn_lifecycle import _mark_failed_rebuild_pending
from travel_agent.artifact_policy import constraint_fingerprint, constraint_version
from travel_agent.poi_evidence import canonical_entity_match_evidence
from travel_agent.plan_invariants import validate_plan_artifact
from travel_agent.orchestration.multi_agent.orchestrator_agent import (
    RoutingDecision,
    RoutingTask,
)
from travel_agent.providers import LocalToolProvider
from travel_agent.schemas import POI


def test_failed_rebuild_without_current_plan_enters_pending_state() -> None:
    ctx = build_session(persist=False)
    analysis = SimpleNamespace(delivery_intent=DeliveryIntent.REBUILD_NOW)
    reply = SimpleNamespace(plan_artifact_id=None, status="incomplete")

    changed = _mark_failed_rebuild_pending(ctx, analysis, reply)

    assert changed is True
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"


def _install_deterministic_planner(monkeypatch) -> None:
    def fake_plan_and_critique(ctx, artifact_ids=None, **_kwargs):
        del artifact_ids
        state = ctx.profile.constraint_state or {}
        version = constraint_version(ctx.profile)
        parent = current_task_meta().get("parent_plan_artifact_id")
        parent_payload = ctx.store.get(parent) if parent else None
        lineage = list((parent_payload or {}).get("revision_lineage") or [])
        if parent and parent not in lineage:
            lineage.append(parent)
        removed = [str(item) for item in state.get("removed") or []]
        required = [
            str(item)
            for item in state.get("must_visit") or ctx.profile.must_visit or []
            if not any(term in str(item) or str(item) in term for term in removed)
        ]
        names = required or ["通用活动"]
        days = []
        for day_index in range(1, int(ctx.profile.days or 1) + 1):
            day_names = names if day_index == 1 else [f"通用活动{day_index}"]
            days.append({
                "day_index": day_index,
                "stops": [{
                    "poi": {
                        "poi_id": f"poi-{day_index}-{offset}",
                        "name": name,
                        "canonical_name": name,
                        "source": "deterministic_test",
                        "verification_status": "verified",
                        "lng": 120.15 + day_index * 0.01 + offset * 0.001,
                        "lat": 30.25 + day_index * 0.01 + offset * 0.001,
                    },
                    "start": "09:00",
                    "end": "10:00",
                } for offset, name in enumerate(day_names, 1)],
            })
        payload = {
            "parent_plan_artifact_id": parent,
            "revision_lineage": lineage,
            "state_version": {
                "constraint_revision": version["revision"],
                "constraint_hash": version["constraint_hash"],
                "constraint_snapshot": version["constraint_snapshot"],
            },
            "itinerary": {"city": ctx.profile.destination, "days": days},
            "critic": {"passed": True, "issues": []},
            "budget_plan": {
                "constraint_revision": version["revision"],
                "constraint_hash": version["constraint_hash"],
                "people": int(
                    state.get("traveler_count") or ctx.profile.party_size or 1
                ),
                "days": int(state.get("duration_days") or ctx.profile.days or 1),
                "user_limit_cny": ctx.profile.budget_limit,
                "breakdown_cny": {
                    "lodging": 0,
                    "transport": 300,
                    "tickets": 200,
                    "meals": 300,
                    "fixed_event_cost": 0,
                    "contingency": 100,
                },
                "expected_total": 900,
                "high_total": 1100,
                "unknown_items": [],
            },
        }
        fixed_events = [
            dict(item)
            for item in state.get("fixed_events") or []
            if isinstance(item, dict)
        ]
        if fixed_events:
            payload["fixed_event_plan"] = {
                "status": "fixed_appointment_buffers",
                "events": [
                    {
                        **item,
                        "route_evidence_status": "unavailable",
                        "required_buffer_min": 15,
                    }
                    for item in fixed_events
                ],
            }
        if state.get("return_location") and state.get("return_deadline"):
            final_stop_id = days[-1]["stops"][-1]["poi"]["poi_id"]
            route = {
                "kind": "last_stop_to_return_location",
                "required_name": state["return_location"],
                "origin_poi_id": final_stop_id,
                "destination_poi_id": "return-endpoint",
                "mode": "public_transport",
                "duration_min": 30,
                "distance_km": 8.0,
                "source": "deterministic_test",
                "evidence_status": "deterministic_estimate",
                "return_deadline": state["return_deadline"],
                "recommended_latest_departure": "16:00",
                "required_buffer_min": 30,
            }
            payload["required_route_anchors"] = {
                "status": "bound_required_endpoints",
                "legs": [route],
                "last_stop_to_return_location": route,
            }
        payload["validation_result"] = validate_plan_artifact(payload, ctx.profile)
        payload["artifact_status"] = (
            "candidate" if payload["validation_result"]["passed"] else "validation_failure"
        )
        artifact_id = ctx.store.put("itinerary", payload)
        if payload["artifact_status"] == "candidate":
            state["_plan_status"] = "candidate_ready"
        return {"artifact_id": artifact_id, "summary": "deterministic planner completed"}

    monkeypatch.setattr(toolkit, "plan_and_critique", fake_plan_and_critique)


def _domain_executor(_settings, model=None):
    del model

    def executor(definition, task, ctx):
        trace: list[str] = []
        if definition.name == "attraction":
            toolkit.search_poi(ctx, city=ctx.profile.destination, max_results=8)
            trace.append("search_poi")
        elif definition.name == "transport":
            pois = list(ctx.pois_by_id.values())
            if len(pois) >= 2:
                toolkit.plan_route(ctx, pois[0].poi_id, pois[1].poi_id)
                trace.append("plan_route")
            if (ctx.profile.constraint_state or {}).get("budget_max_cny") is not None:
                toolkit.estimate_budget(ctx)
                trace.append("estimate_budget")
        elif definition.name == "restaurant":
            # Deliberately return no restaurant artifact.  The production gate
            # must ignore this domain for dietary-only requests but fail closed
            # when concrete restaurants were requested.
            return {"summary": "restaurant provider returned empty", "tool_trace": []}
        elif definition.name == "planner":
            result = toolkit.plan_and_critique(
                ctx, artifact_ids=list(task.inputs.get("artifact_ids") or [])
            )
            trace.append("plan_and_critique")
            return {"summary": result["summary"], "tool_trace": trace}
        return {"summary": f"{definition.name} complete", "tool_trace": trace}

    return executor


def _production_context():
    ctx = build_session(persist=False)
    ctx.profile.destination = "杭州"
    ctx.profile.days = 1
    ctx.provider = LocalToolProvider([
        POI(
            poi_id=f"poi-{index}", name=name, city="杭州", category="scenic",
            lat=30.2 + index * 0.01, lng=120.1 + index * 0.01,
            rating=4.6, popularity=0.8, tags=["文化"],
            estimated_duration_min=90, price_level="mid", source="provider",
            canonical_name=name, entity_type="attraction", source_poi_id=f"canonical-{index}",
            verification_status="verified",
        )
        for index, name in enumerate(("文化公园", "城市展馆", "滨水步道"), 1)
    ])
    return ctx


def _orchestrated_settings(offline_settings):
    return replace(
        offline_settings,
        llm=replace(
            offline_settings.llm,
            provider="openai",
            api_key="test-key",
            model="test-model",
        ),
    )


def _agents(reply) -> list[str]:
    return [
        str(item.get("agent") or "")
        for item in reply.agent_trace
        if item.get("kind") == "subagent"
    ]


def _install_rule_router(monkeypatch) -> None:
    def fake_route(*args, **kwargs):
        missing = list(kwargs.get("missing_evidence") or [])
        tasks = []
        for label, agent in (
            ("景点候选", "attraction"),
            ("路线可行性", "transport"),
            ("预算证据", "transport"),
            ("具体餐厅证据", "restaurant"),
        ):
            if label in missing:
                tasks.append(RoutingTask(
                    agent=agent,
                    instruction=f"补充{label}",
                    objective=f"补充{label}",
                ))
        return RoutingDecision(tasks=tuple(tasks), ready=not tasks)

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        fake_route,
    )


def _install_pass_reviewer(monkeypatch) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.review.build_review_callable",
        lambda *args, **kwargs: (
            lambda review_ctx: {"verdict": "pass", "issues": []}
        ),
    )


def test_production_entry_budget_change_creates_new_current_itinerary(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    ctx = build_session(persist=False)

    first = run_production_turn(
        "请规划杭州两天完整行程。", ctx=ctx, settings=offline_settings
    )
    old_id = first.plan_artifact_id
    assert old_id == ctx.store.latest_current_id("itinerary")

    second = run_production_turn(
        "整体预算改成5000元，请调整现有方案。",
        ctx=ctx,
        settings=offline_settings,
    )
    new_id = second.plan_artifact_id

    assert new_id and new_id != old_id
    assert ctx.store.get_record(old_id)["artifact_status"] == "stale"
    assert ctx.store.latest_current_id("itinerary") == new_id
    payload = ctx.store.get(new_id)
    assert payload["parent_plan_artifact_id"] == old_id
    assert old_id in payload["revision_lineage"]
    assert payload["state_version"]["constraint_hash"] == payload[
        "validation_result"
    ]["validated_constraint_hash"]
    assert payload["state_version"]["constraint_revision"] == payload[
        "validation_result"
    ]["validated_constraint_revision"]
    assert payload["state_version"]["constraint_snapshot"]["budget_max_cny"] == 5000.0


def test_production_structured_existing_budget_replacement_rebuilds_immediately(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    ctx = build_session(persist=False)
    initial = run_production_turn(
        "请规划杭州两天完整行程，总预算8000元。",
        ctx=ctx,
        settings=offline_settings,
    )

    replacement = run_production_turn(
        "总预算改为6200元。", ctx=ctx, settings=offline_settings
    )

    assert replacement.planner_status == "planned"
    assert replacement.plan_artifact_id != initial.plan_artifact_id
    payload = ctx.store.get(replacement.plan_artifact_id)
    assert payload["budget_plan"]["user_limit_cny"] == 6200
    assert payload["budget_plan"]["constraint_hash"] == payload["state_version"]["constraint_hash"]


def test_production_first_lodging_declaration_enters_pending_without_planner(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    ctx = build_session(persist=False)
    initial = run_production_turn(
        "请规划杭州两天完整行程。", ctx=ctx, settings=offline_settings
    )

    lodging = run_production_turn(
        "住宿安排在湖滨区。", ctx=ctx, settings=offline_settings
    )

    assert lodging.planner_status == "deferred_state_update"
    assert lodging.plan_artifact_id is None
    assert ctx.profile.constraint_state["lodging_area"] == "湖滨区"
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"
    assert ctx.store.get_record(initial.plan_artifact_id)["artifact_status"] == "stale"


def test_pending_multi_constraint_updates_wait_for_one_final_planner_run(
    monkeypatch,
    offline_settings,
) -> None:
    calls = 0
    _install_deterministic_planner(monkeypatch)
    deterministic = toolkit.plan_and_critique

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return deterministic(*args, **kwargs)

    monkeypatch.setattr(toolkit, "plan_and_critique", counted)
    ctx = build_session(persist=False)
    first = run_production_turn(
        "请规划杭州三天完整行程，2026年10月1日出发，预算8000元，必去西湖。",
        ctx=ctx,
        settings=offline_settings,
    )
    assert first.planner_status == "planned"

    updates = [
        "住宿安排在湖滨区，先记住。",
        "10月2日10:00到12:00已经预约城市展馆，继续记住。",
        "西湖不去了，换成灵隐寺。",
        "总预算改为6200元。",
        "最后一天17点前回到杭州东站。",
    ]
    replies = [
        run_production_turn(message, ctx=ctx, settings=offline_settings)
        for message in updates
    ]

    assert all(reply.planner_status == "deferred_state_update" for reply in replies)
    assert calls == 1
    final = run_production_turn(
        "按全部条件生成最终版完整行程。",
        ctx=ctx,
        settings=offline_settings,
    )
    # The rebuild runs exactly once, but the newly added timed appointment has
    # no route evidence in the offline fixture, so the finalizer must fail
    # closed instead of promoting an apparently complete plan.
    assert final.planner_status == "planner_failed"
    assert calls == 2
    assert ctx.store.latest_current_id("itinerary") is None
    assert final.plan_artifact_id is None


def test_pending_clothing_interjection_answers_with_current_turn_trace(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    ctx = _production_context()
    run_production_turn(
        "请规划杭州一天完整行程。", ctx=ctx, settings=offline_settings
    )
    pending = run_production_turn(
        "住宿安排在湖滨区。", ctx=ctx, settings=offline_settings
    )
    pending_hash = ctx.profile.constraint_state["_constraint_hash"]

    clothing = run_production_turn(
        "明天该怎么穿？", ctx=ctx, settings=offline_settings
    )

    assert clothing.status == "completed"
    assert clothing.planner_status == "not_required"
    assert "建议" in clothing.text and "穿" in clothing.text
    assert clothing.tool_trace == ["check_weather"]
    assert clothing.request_id != pending.request_id
    assert clothing.agent_trace
    assert {
        item["request_id"] for item in clothing.agent_trace
    } == {clothing.request_id}
    contract = next(
        item for item in clothing.agent_trace if item["kind"] == "turn_contract"
    )
    assert contract["detail"]["delivery_intent"] == "lightweight_advice"
    assert contract["detail"]["planner_required"] is False
    assert not any(
        item.get("kind") == "admission" and item.get("agent") == "planner"
        for item in clothing.agent_trace
    )
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"
    assert ctx.profile.constraint_state["_constraint_hash"] == pending_hash


def test_production_entry_accumulates_constraints_then_rebuilds_final_version(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    ctx = build_session(persist=False)
    initial = run_production_turn(
        "请规划杭州两天完整行程。", ctx=ctx, settings=offline_settings
    )
    old_id = initial.plan_artifact_id
    assert initial.planner_status == "planned"

    budget = run_production_turn(
        "整体预算改成5000元，先记住，之后再出最终版。",
        ctx=ctx,
        settings=offline_settings,
    )
    assert budget.planner_status == "deferred_state_update"
    dietary = run_production_turn(
        "再补充：全程饮食清淡，先记住，等我说最终版。",
        ctx=ctx,
        settings=offline_settings,
    )
    assert dietary.planner_status == "deferred_state_update"
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"
    assert ctx.profile.constraint_state["_revisable_parent_plan_artifact_id"] == old_id

    final = run_production_turn(
        "按全部条件生成最终版完整行程。",
        ctx=ctx,
        settings=offline_settings,
    )
    new_id = final.plan_artifact_id
    assert final.planner_status == "planned"
    payload = ctx.store.get(new_id)

    assert new_id and new_id != old_id
    assert payload["parent_plan_artifact_id"] == old_id
    snapshot = payload["state_version"]["constraint_snapshot"]
    assert snapshot["budget_max_cny"] == 5000.0
    assert snapshot["dietary"] == ["清淡"]
    assert payload["validation_result"]["passed"] is True
    assert ctx.store.latest_current_id("itinerary") == new_id


def test_rebuild_pending_weather_interjection_preserves_state_and_skips_planner(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    ctx = _production_context()
    initial = run_production_turn(
        "请规划杭州一天完整行程。", ctx=ctx, settings=offline_settings
    )
    parent_id = initial.plan_artifact_id

    update = run_production_turn(
        "整体预算改成5000元。", ctx=ctx, settings=offline_settings
    )
    assert update.planner_status == "deferred_state_update"
    pending_hash = ctx.profile.constraint_state["_constraint_hash"]

    weather = run_production_turn(
        "明天天气怎么样？", ctx=ctx, settings=offline_settings
    )

    assert weather.status == "completed"
    assert weather.planner_status == "not_required"
    assert weather.tool_trace == ["check_weather"]
    assert "plan_and_critique" not in weather.tool_trace
    assert ctx.profile.constraint_state["_plan_status"] == "rebuild_pending"
    assert ctx.profile.constraint_state["_constraint_hash"] == pending_hash
    assert ctx.profile.constraint_state["_revisable_parent_plan_artifact_id"] == parent_id
    assert ctx.profile.constraint_state["budget_max_cny"] == 5000.0


def test_production_dietary_only_constraint_does_not_require_restaurant_agent(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    _install_rule_router(monkeypatch)
    _install_pass_reviewer(monkeypatch)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _domain_executor,
    )
    ctx = _production_context()

    reply = run_production_turn(
        "请规划杭州一天完整行程，全程饮食清淡，不吃海鲜。",
        ctx=ctx,
        settings=_orchestrated_settings(offline_settings),
    )

    assert "restaurant" not in _agents(reply)
    assert "planner" in _agents(reply)
    assert ctx.store.latest("restaurants") is None
    assert reply.plan_artifact_id is not None


def test_production_specific_restaurant_request_requires_restaurant_evidence(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    _install_rule_router(monkeypatch)
    _install_pass_reviewer(monkeypatch)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _domain_executor,
    )
    ctx = _production_context()

    reply = run_production_turn(
        "请规划杭州一天完整行程，并推荐具体餐厅。",
        ctx=ctx,
        settings=_orchestrated_settings(offline_settings),
    )

    assert "restaurant" in _agents(reply)
    assert "planner" not in _agents(reply)
    assert reply.status == "incomplete"
    assert "具体餐厅证据" in reply.text


def test_production_budget_change_reuses_verified_poi_after_refresh_failure(
    monkeypatch,
    offline_settings,
) -> None:
    _install_deterministic_planner(monkeypatch)
    _install_pass_reviewer(monkeypatch)
    ctx = _production_context()
    first = run_production_turn(
        "请规划杭州一天完整行程。", ctx=ctx, settings=offline_settings
    )
    assert first.plan_artifact_id

    state = dict(ctx.profile.constraint_state)
    poi_payload = {"pois": [poi_to_dict(ctx.provider.pois[0])]}
    poi_artifact_id = ctx.store.put("candidates", poi_payload)
    ctx.store._items[poi_artifact_id]["constraint_basis"] = {}
    ctx.store._items[poi_artifact_id]["constraint_fingerprint"] = constraint_fingerprint(
        "candidates", state, poi_payload
    )

    class FailingRefreshProvider(LocalToolProvider):
        def search_pois(self, *args, **kwargs):
            raise RuntimeError("controlled refresh failure")

    ctx.provider = FailingRefreshProvider(list(ctx.provider.pois))

    def refresh_then_budget(*args, **kwargs):
        wave = int(kwargs.get("wave") or 1)
        if wave == 1:
            return RoutingDecision(tasks=(
                RoutingTask("attraction", "尝试刷新 POI", "刷新 POI"),
                RoutingTask("transport", "补充预算", "补充预算证据"),
            ))
        return RoutingDecision(ready=True)

    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.orchestrator_agent.route_wave",
        refresh_then_budget,
    )
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _domain_executor,
    )

    reply = run_production_turn(
        "整体预算改成5000元，请调整现有方案。",
        ctx=ctx,
        settings=_orchestrated_settings(offline_settings),
    )

    contract = next(item for item in reply.agent_trace if item.get("kind") == "turn_contract")
    reused = next(
        item
        for item in contract["detail"]["artifact_reuse_audit"]
        if item.get("artifact_id") == poi_artifact_id
    )
    assert reused["reason"] == "artifact_reuse"
    assert reused["reuse_reason"] == "kind_specific_fingerprint_match"
    assert reused["source_lineage"] == [poi_artifact_id]
    assert "planner" in _agents(reply)
    assert reply.plan_artifact_id and reply.plan_artifact_id != first.plan_artifact_id


def test_production_must_visit_canonical_variant_records_match_evidence(
    monkeypatch,
    offline_settings,
) -> None:
    canonical_poi = POI(
        poi_id="provider-poi-1", name="杭州市自然博物馆东馆", city="杭州",
        category="museum", lat=30.25, lng=120.15, rating=4.7, popularity=0.9,
        tags=["自然", "博物馆"], estimated_duration_min=120, price_level="free",
        source="provider", canonical_name="杭州市自然博物馆东馆",
        entity_type="museum", source_poi_id="canonical-museum-1",
        verification_status="verified",
    )

    def canonical_planner(ctx, artifact_ids=None, **_kwargs):
        del artifact_ids
        required = str((ctx.profile.constraint_state.get("must_visit") or [""])[0])
        evidence = canonical_entity_match_evidence(canonical_poi, required)
        version = constraint_version(ctx.profile)
        payload = {
            "artifact_status": "candidate",
            "state_version": {
                "constraint_revision": version["revision"],
                "constraint_hash": version["constraint_hash"],
                "constraint_snapshot": version["constraint_snapshot"],
            },
            "itinerary": {"city": "杭州", "days": [{
                "day_index": 1,
                "stops": [{"poi": poi_to_dict(canonical_poi), "start": "09:00", "end": "11:00"}],
            }]},
            "critic": {"passed": True, "issues": []},
            "candidate_verification": {"status": "verified_named_candidates", "results": [{
                "requested_name": required,
                "matched_name": canonical_poi.name,
                "original_name": canonical_poi.name,
                "canonical_name": canonical_poi.canonical_name,
                "canonical_match_evidence": evidence,
                "status": "suitable",
            }]},
        }
        payload["validation_result"] = validate_plan_artifact(payload, ctx.profile)
        payload["artifact_status"] = "candidate" if payload["validation_result"]["passed"] else "validation_failure"
        artifact_id = ctx.store.put("itinerary", payload)
        return {"artifact_id": artifact_id, "summary": "canonical planner complete"}

    monkeypatch.setattr(toolkit, "plan_and_critique", canonical_planner)
    _install_rule_router(monkeypatch)
    _install_pass_reviewer(monkeypatch)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _domain_executor,
    )
    ctx = build_session(persist=False)
    ctx.profile.destination = "杭州"
    ctx.profile.days = 1
    ctx.provider = LocalToolProvider([canonical_poi])

    reply = run_production_turn(
        "请规划杭州一天完整行程，必去杭州自然博物馆。",
        ctx=ctx,
        settings=_orchestrated_settings(offline_settings),
    )

    payload = ctx.store.get(reply.plan_artifact_id)
    match = payload["candidate_verification"]["results"][0]["canonical_match_evidence"]
    assert payload["validation_result"]["passed"] is True, payload["validation_result"]
    assert reply.status == "completed", reply.text
    assert match["requested_name"] == "杭州自然博物馆"
    assert match["original_name"] == "杭州市自然博物馆东馆"
    assert match["canonical_id"] == "canonical-museum-1"
