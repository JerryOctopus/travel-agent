from __future__ import annotations

from dataclasses import replace

from travel_agent.agent import toolkit
from travel_agent.agent.runtime import run_production_turn
from travel_agent.agent.session import build_session
from travel_agent.candidate_comparison import comparison_candidates, entity_matches
from travel_agent.evaluation.artifact_contract import artifact_content_valid
from travel_agent.providers import LocalToolProvider
from travel_agent.schemas import POI
from travel_agent.settings import Settings


CITY = "测试城"


def _settings(offline_settings: Settings) -> Settings:
    return replace(
        offline_settings,
        llm=replace(
            offline_settings.llm,
            provider="openai",
            api_key="test-key",
            model="test-model",
        ),
    )


def _poi(poi_id: str, name: str, tags: list[str], offset: float) -> POI:
    return POI(
        poi_id=poi_id,
        name=name,
        city=CITY,
        category="scenic",
        lat=30.0 + offset,
        lng=120.0 + offset,
        rating=4.8 - offset * 10,
        popularity=0.9 - offset,
        tags=tags,
        estimated_duration_min=60,
        price_level="mid",
        address=f"{name}公共交通生活圈",
        source="test-provider",
        entity_type="district",
    )


def _context() -> object:
    ctx = build_session(persist=False)
    ctx.profile.destination = CITY
    ctx.provider = LocalToolProvider(
        [
            _poi("area-a", "甲区生活圈", ["甲区", "适老", "无障碍", "交通", "安静"], 0.01),
            _poi("area-b", "乙区生活圈", ["乙区", "适老", "无障碍", "地铁", "热闹"], 0.03),
            _poi("area-c", "丙区生活圈", ["丙区", "适老", "无障碍", "公交", "安静"], 0.05),
            _poi("anchor", "中央车站", ["中央车站", "交通"], 0.08),
        ]
    )
    return ctx


def _context_without_transit_nodes() -> object:
    ctx = build_session(persist=False)
    ctx.profile.destination = CITY
    ctx.provider = LocalToolProvider(
        [
            _poi("area-a", "甲区生活圈", ["甲区", "住宅"], 0.01),
            _poi("area-b", "乙区生活圈", ["乙区", "住宅"], 0.03),
            _poi("area-c", "丙区生活圈", ["丙区", "住宅"], 0.05),
            _poi("unrelated", "无关公园", ["公园"], 0.07),
        ]
    )
    return ctx


def _context_with_partial_accessibility_needs() -> object:
    ctx = build_session(persist=False)
    ctx.profile.destination = CITY
    ctx.provider = LocalToolProvider(
        [
            _poi("area-a", "甲区生活圈", ["甲区", "交通", "无障碍"], 0.01),
            _poi("area-b", "乙区生活圈", ["乙区", "地铁", "适老"], 0.03),
            _poi("area-c", "丙区生活圈", ["丙区", "公交"], 0.05),
        ]
    )
    return ctx


def _tool_executor(_settings, model=None):
    del model
    def executor(definition, task, ctx):
        state = ctx.profile.constraint_state or {}
        scoped = str(task.inputs.get("comparison_candidate") or "").strip()
        candidates = [scoped] if scoped else comparison_candidates(state)
        target = str(state.get("target_anchor") or "").strip()
        trace: list[str] = []
        if definition.name in {"attraction", "transport"}:
            for name in dict.fromkeys([*candidates, *([target] if target else [])]):
                toolkit.search_poi(ctx, city=CITY, interests=[name], max_results=8)
                trace.append("search_poi")
        if definition.name == "transport" and target:
            target_poi = next(
                poi for poi in ctx.pois_by_id.values() if entity_matches(target, poi.name)
            )
            for candidate in candidates:
                candidate_poi = next(
                    poi
                    for poi in ctx.pois_by_id.values()
                    if entity_matches(candidate, poi.name)
                )
                toolkit.plan_route(
                    ctx,
                    candidate_poi.poi_id,
                    target_poi.poi_id,
                    mode="public_transport",
                )
                trace.append("plan_route")
        return {"summary": f"{definition.name} evidence complete", "tool_trace": trace}

    return executor


def _empty_executor(_settings, model=None):
    del model
    def executor(definition, task, ctx):
        del definition, task, ctx
        return {"summary": "worker returned no tool evidence", "tool_trace": []}

    return executor


def _production_postcondition_executor(_settings, model=None):
    """Exercise the real bounded comparison/route repair through the turn entrypoint."""
    del model
    from travel_agent.orchestration.multi_agent.executor import (
        _enforce_comparison_search_postcondition,
        _enforce_transport_route_postcondition,
    )

    def executor(definition, task, ctx):
        result = {"summary": "model omitted tools", "tool_trace": []}
        result = _enforce_comparison_search_postcondition(task, ctx, result)
        if definition.name == "transport":
            candidates = comparison_candidates(ctx.profile.constraint_state or {})
            candidate = str(task.inputs.get("comparison_candidate") or "").strip()
            if candidates and candidate == candidates[-1]:
                first_id = next(
                    poi.poi_id
                    for poi in ctx.pois_by_id.values()
                    if entity_matches(candidates[0], poi.name)
                )
                second_id = next(
                    poi.poi_id
                    for poi in ctx.pois_by_id.values()
                    if entity_matches(candidates[1], poi.name)
                )
                # Seed the reverse direction.  The real repair must recognize
                # it as the same pair/mode and only add the other matrix cells.
                toolkit.plan_route(
                    ctx,
                    second_id,
                    first_id,
                    mode="public_transport",
                )
                result["tool_trace"].append("plan_route")
            result = _enforce_transport_route_postcondition(task, ctx, result)
        return result

    return executor


def _unrelated_route_executor(_settings, model=None):
    del model

    def executor(definition, task, ctx):
        candidate = str(task.inputs.get("comparison_candidate") or "").strip()
        trace: list[str] = []
        if definition.name != "transport":
            return {"summary": "not transport", "tool_trace": trace}
        toolkit.search_poi(ctx, city=CITY, interests=[candidate], max_results=8)
        toolkit.search_poi(ctx, city=CITY, interests=["无关公园"], max_results=8)
        candidate_id = next(
            poi.poi_id for poi in ctx.pois_by_id.values() if entity_matches(candidate, poi.name)
        )
        unrelated_id = next(
            poi.poi_id for poi in ctx.pois_by_id.values() if entity_matches("无关公园", poi.name)
        )
        toolkit.plan_route(ctx, candidate_id, unrelated_id, mode="public_transport")
        trace.extend(["search_poi", "search_poi", "plan_route"])
        return {"summary": "unrelated route must be rejected", "tool_trace": trace}

    return executor


class _PartialProvider(LocalToolProvider):
    """Controlled provider: a combined query returns only its first candidate."""

    def __init__(self, pois):
        super().__init__(pois)
        self.named_queries: list[tuple[str, ...]] = []

    def __deepcopy__(self, memo):
        del memo
        return self

    def search_pois(self, city, query_tags=None, category=None, max_results=20):
        if not query_tags:
            return []
        query = tuple(str(item) for item in query_tags)
        self.named_queries.append(query)
        return super().search_pois(city, list(query[:1]), category, max_results)


def _scoped_executor(*, missing_route_candidate: str | None = None, missing_area_candidate: str | None = None):
    def factory(_settings, model=None):
        del model

        def executor(definition, task, ctx):
            candidate = str(task.inputs.get("comparison_candidate") or "").strip()
            dimensions = list(task.inputs.get("comparison_dimensions") or [])
            state = ctx.profile.constraint_state or {}
            candidates = [candidate] if candidate else comparison_candidates(state)
            target = str(state.get("target_anchor") or "").strip()
            trace: list[str] = []
            if definition.name == "attraction":
                for name in candidates:
                    if name == missing_area_candidate:
                        continue
                    toolkit.search_poi(ctx, city=CITY, interests=[name], max_results=8)
                    trace.append("search_poi")
            if definition.name == "transport":
                for name in candidates:
                    if name == missing_route_candidate:
                        continue
                    toolkit.search_poi(ctx, city=CITY, interests=[name], max_results=8)
                    toolkit.search_poi(ctx, city=CITY, interests=[target], max_results=8)
                    candidate_poi = next(
                        poi for poi in ctx.pois_by_id.values() if entity_matches(name, poi.name)
                    )
                    target_poi = next(
                        poi for poi in ctx.pois_by_id.values() if entity_matches(target, poi.name)
                    )
                    toolkit.plan_route(
                        ctx, candidate_poi.poi_id, target_poi.poi_id, mode="public_transport"
                    )
                    trace.extend(["search_poi", "search_poi", "plan_route"])
            return {
                "summary": f"{definition.name} scoped={candidate} dimensions={dimensions}",
                "tool_trace": trace,
            }

        return executor

    return factory


def _agents(reply) -> list[str]:
    return [
        str(item.get("agent") or "")
        for item in reply.agent_trace
        if item.get("kind") == "subagent"
    ]


def test_production_three_lodging_areas_without_inventory_dispatches_evidence_workers(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _tool_executor,
    )
    ctx = _context()

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个住宿区域，适合父母少走路并兼顾安静氛围，不要具体酒店库存。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed"
    assert set(_agents(reply)) == {"attraction", "transport"}
    assert "planner" not in _agents(reply) and "hotel" not in _agents(reply)
    assert ctx.store.latest("hotels") is None and ctx.store.latest("itinerary") is None
    assert set(reply.tool_trace) == {"search_poi"}
    assert artifact_content_valid("candidate_comparison", artifact)
    assert artifact["coverage"] == {"required": 6, "covered": 6, "complete": True}


def test_production_unanchored_accessibility_uses_same_transit_fact_for_all_candidates(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _tool_executor,
    )
    ctx = _context()

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个区域，哪个交通更方便。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed"
    assert _agents(reply) == ["transport", "transport", "transport"]
    assert "planner" not in _agents(reply)
    assert "plan_route" not in reply.tool_trace
    assert artifact["subject"]["accessibility_mode"] == "unanchored"
    assert artifact["accessibility_contract"]["evidence_strategy"] == "candidate_transit_nodes"
    assert artifact["coverage_matrix"] == {
        name: {"accessibility": True} for name in ("甲区", "乙区", "丙区")
    }
    assert artifact["recommendation"]["dimensions_used"] == ["accessibility"]
    assert artifact["recommendation"]["comparable_metrics"] == ["transit_node_count"]
    assert artifact_content_valid("candidate_comparison", artifact)


def test_production_unanchored_complete_route_matrix_uses_average_metrics_once_per_pair(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _production_postcondition_executor,
    )
    ctx = _context_without_transit_nodes()

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个区域，哪个交通更方便。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    route_contract = artifact["accessibility_contract"]["route_matrix"]
    assert reply.status == "completed"
    assert "planner" not in _agents(reply)
    assert reply.tool_trace.count("search_poi") == 3
    assert reply.tool_trace.count("plan_route") == 3
    assert route_contract["scope"] == "complete"
    assert len(route_contract["pairs"]) == 3
    assert artifact["accessibility_contract"]["evidence_strategy"] == "candidate_route_matrix"
    assert artifact["recommendation"]["comparable_metrics"] == ["average_duration_min"]
    assert len(artifact["recommendation"]["source_artifact_ids"]) == 3
    assert all(
        sum(item["candidate"] == name for item in artifact["evidence"]) == 2
        for name in ("甲区", "乙区", "丙区")
    )
    assert artifact_content_valid("candidate_comparison", artifact)


def test_production_complete_accessibility_with_partial_needs_delivers_limited_result(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _tool_executor,
    )
    ctx = _context_with_partial_accessibility_needs()

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个住宿区域，交通便利度为核心，并参考是否适合父母少走路，不要具体酒店库存。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed"
    assert artifact["core_dimensions"] == ["accessibility"]
    assert artifact["coverage_matrix"]["丙区"] == {
        "accessibility": True,
        "accessibility_needs": False,
    }
    assert artifact["recommendation"]["dimensions_used"] == ["accessibility"]
    assert "accessibility_needs" not in artifact["recommendation"]["dimensions_used"]
    assert any("不据此声称适老或无障碍优势" in item for item in artifact["limitations"])
    assert artifact_content_valid("candidate_comparison", artifact)


def test_production_two_areas_to_fixed_anchor_binds_one_route_per_candidate(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _tool_executor,
    )
    ctx = _context()

    reply = run_production_turn(
        "比较甲区和乙区到中央车站的通勤时间",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed"
    assert _agents(reply) == ["transport", "transport"]
    assert "planner" not in _agents(reply)
    assert reply.tool_trace.count("plan_route") == 2
    assert {item["candidate"] for item in artifact["evidence"]} == {"甲区", "乙区"}
    assert all(item["dimension"] == "accessibility" for item in artifact["evidence"])
    assert artifact["subject"]["accessibility_mode"] == "anchored"
    assert artifact["accessibility_contract"] == {
        "mode": "anchored",
        "target_anchor": "中央车站",
        "evidence_strategy": "candidate_to_anchor",
    }
    assert artifact["recommendation"]["comparable_metrics"] == ["duration_min"]
    assert artifact["recommendation"]["candidate"] == "乙区"
    assert artifact_content_valid("candidate_comparison", artifact)


def test_production_non_transport_comparison_does_not_dispatch_transport(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _tool_executor,
    )
    ctx = _context()

    reply = run_production_turn(
        "比较甲区和乙区哪个氛围更安静",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed"
    assert _agents(reply) == ["attraction", "attraction"]
    assert "transport" not in _agents(reply) and "planner" not in _agents(reply)
    assert artifact_content_valid("candidate_comparison", artifact)


def test_production_empty_worker_evidence_fails_closed(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _empty_executor,
    )
    ctx = _context()

    reply = run_production_turn(
        "比较甲区和乙区到中央车站的通勤时间",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "incomplete"
    assert _agents(reply) == ["transport", "transport"] and "planner" not in _agents(reply)
    assert artifact["evidence"] == []
    assert artifact["recommendation"] is None
    assert artifact["coverage"]["complete"] is False
    assert not artifact_content_valid("candidate_comparison", artifact)


def test_production_partial_provider_is_queried_once_per_missing_candidate(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _scoped_executor(),
    )
    ctx = _context()
    ctx.provider = _PartialProvider(list(ctx.provider.pois))

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个住宿区域，哪个氛围更安静，不要具体酒店库存。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    assert reply.status == "completed"
    artifact = ctx.store.latest("candidate_comparison")
    assert artifact_content_valid("candidate_comparison", artifact)
    candidate_queries = [query for query in ctx.provider.named_queries if query[0] in {"甲区", "乙区", "丙区"}]
    assert candidate_queries == [("甲区",), ("乙区",), ("丙区",)]


def test_production_partial_secondary_dimension_still_delivers_limited_recommendation(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _scoped_executor(missing_area_candidate="丙区"),
    )
    ctx = _context()
    ctx.provider = _PartialProvider(list(ctx.provider.pois))

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个住宿区域，到中央车站的通勤时间为核心，并参考哪个氛围更安静，不要具体酒店库存。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed"
    assert artifact["core_dimensions"] == ["accessibility"]
    assert artifact["recommendation"]["dimensions_used"] == ["accessibility"]
    assert any("丙区" in item and "ambience" in item for item in artifact["limitations"])
    assert artifact["coverage_matrix"]["丙区"] == {
        "accessibility": True,
        "ambience": False,
    }
    assert artifact["coverage"]["complete"] is False
    assert artifact_content_valid("candidate_comparison", artifact)


def test_production_missing_core_candidate_fails_closed_even_if_secondary_is_complete(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _scoped_executor(missing_route_candidate="丙区"),
    )
    ctx = _context()
    ctx.provider = _PartialProvider(list(ctx.provider.pois))

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个住宿区域，到中央车站的通勤时间为核心，并参考哪个氛围更安静，不要具体酒店库存。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "incomplete"
    assert artifact["recommendation"] is None
    assert any("核心维度" in item and "丙区" in item for item in artifact["limitations"])
    assert not artifact_content_valid("candidate_comparison", artifact)


def test_production_unanchored_routes_with_unrelated_endpoints_do_not_count(
    monkeypatch,
    offline_settings,
) -> None:
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _unrelated_route_executor,
    )
    ctx = _context_without_transit_nodes()

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个区域，哪个交通更方便。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "incomplete"
    assert "planner" not in _agents(reply)
    assert reply.tool_trace.count("plan_route") == 3
    assert artifact["evidence"] == []
    assert artifact["coverage"] == {"required": 3, "covered": 0, "complete": False}
    assert artifact["recommendation"] is None
    assert not artifact_content_valid("candidate_comparison", artifact)


def test_production_collector_and_artifact_validator_share_accessibility_contract(
    monkeypatch,
    offline_settings,
) -> None:
    import travel_agent.candidate_comparison as comparison_module

    observed = []
    real_collector = comparison_module.collect_comparison_evidence

    def recording_collector(*args, **kwargs):
        collected = real_collector(*args, **kwargs)
        observed.append(collected)
        return collected

    monkeypatch.setattr(comparison_module, "collect_comparison_evidence", recording_collector)
    monkeypatch.setattr(
        "travel_agent.orchestration.multi_agent.executor.build_subagent_executor",
        _production_postcondition_executor,
    )
    ctx = _context_without_transit_nodes()

    reply = run_production_turn(
        "只比较甲区、乙区和丙区三个区域，哪个交通更方便。",
        ctx=ctx,
        settings=_settings(offline_settings),
    )

    artifact = ctx.store.latest("candidate_comparison")
    assert reply.status == "completed" and observed
    final_evidence = observed[-1]
    assert final_evidence.complete
    assert final_evidence.dimension_strategies == {
        "accessibility": "candidate_route_matrix"
    }
    assert artifact["coverage"]["covered"] == len(final_evidence.covered_cells())
    assert artifact_content_valid("candidate_comparison", artifact)
