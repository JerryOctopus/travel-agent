from __future__ import annotations

import json
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from travel_agent.evaluation.plan_quality import (
    aggregate_rule_quality,
    evaluate_plan_quality,
)
from travel_agent.evaluation.plan_quality_human import (
    calibrate_judge,
    load_human_reviews,
    select_human_review_sample,
)
from travel_agent.evaluation.plan_quality_judge import (
    DIMENSIONS,
    JudgeOutputError,
    PlanQualityJudge,
    _judge_payload,
    aggregate_judge_results,
    parse_judge_output,
    preflight_judge,
)
from travel_agent.evaluation.plan_quality_pipeline import apply_quality_rules
from travel_agent.evaluation.plan_quality_pipeline import apply_independent_judge
from travel_agent.settings import JudgeSettings


def test_judge_preflight_uses_independent_configuration() -> None:
    captured: list[list[dict[str, str]]] = []
    settings = JudgeSettings(
        provider="google",
        api_key="judge-only-key",
        model="gemini-3.6-flash",
        temperature=0.0,
        thinking_enabled=False,
    )

    result = preflight_judge(
        settings,
        invoke=lambda messages: (
            captured.append(messages)
            or SimpleNamespace(content="pong", response_metadata={}, usage_metadata={})
        ),
    )

    assert result == {
        "ok": True,
        "provider": "google",
        "model": "gemini-3.6-flash",
        "base_url": settings.base_url,
        "temperature": 0.0,
        "thinking_enabled": False,
    }
    assert captured and captured[0][-1]["content"] == "Reply with pong."


def test_judge_preflight_fails_closed_without_credential() -> None:
    result = preflight_judge(JudgeSettings(api_key=None))

    assert result["ok"] is False
    assert "not configured" in result["detail"]


def test_siliconflow_qwen_judge_explicitly_disables_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    settings = JudgeSettings(
        provider="siliconflow",
        api_key="judge-key",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3.5-397B-A17B",
        temperature=0.0,
        thinking_enabled=False,
    )

    PlanQualityJudge(settings, Path("."))

    assert captured["temperature"] == 0.0
    assert captured["extra_body"] == {"enable_thinking": False}


def test_deterministic_plan_quality_accepts_grounded_feasible_plan() -> None:
    result = evaluate_plan_quality(
        {
            "expected_city": "杭州",
            "expected_days": 1,
            "hard_constraints": {
                "destination": "杭州",
                "days": 1,
                "budget_level": "low",
                "must_visit": ["西湖"],
                "avoid": ["酒吧"],
                "transport_mode": "public_transport",
            },
            "soft_preferences": {"pace": "standard", "interests": ["history"]},
        },
        {"destination": "杭州", "days": 1, "pace": "standard"},
        _artifacts(),
    )

    assert result is not None
    assert result.hard_feasibility_pass is True
    assert result.rule_quality_pass is True
    assert result.grounded_poi_rate == 1.0
    assert result.route_coverage_rate == 1.0
    assert result.schedule_conflict_free is True
    assert result.transfer_feasible is True
    assert result.must_visit_coverage_rate == 1.0
    assert result.avoid_compliance is True
    assert result.budget_consistency is True
    assert result.opening_hours_coverage == 1.0
    assert result.opening_hours_valid_rate == 1.0


def test_missing_opening_hours_is_unknown_not_failure() -> None:
    artifacts = _artifacts()
    for poi in artifacts["candidates"]["pois"]:
        poi["opening_hours"] = None
    for stop in artifacts["itinerary"]["itinerary"]["days"][0]["stops"]:
        stop["poi"]["opening_hours"] = None

    result = evaluate_plan_quality(
        {"expected_city": "杭州", "expected_days": 1},
        {"destination": "杭州", "days": 1},
        artifacts,
    )

    assert result is not None
    assert result.hard_feasibility_pass is True
    assert result.opening_hours_valid_rate is None
    assert "opening_hours" in result.unknown_checks


def test_transfer_conflict_and_ungrounded_poi_fail_hard_rules() -> None:
    artifacts = _artifacts()
    second = artifacts["itinerary"]["itinerary"]["days"][0]["stops"][1]
    second["start_time"] = "10:20"
    artifacts["candidates"]["pois"] = artifacts["candidates"]["pois"][:1]
    artifacts["ranked"]["pois"] = artifacts["ranked"]["pois"][:1]

    result = evaluate_plan_quality(
        {"expected_city": "杭州", "expected_days": 1},
        {"destination": "杭州", "days": 1},
        artifacts,
    )

    assert result is not None
    assert result.transfer_feasible is False
    assert result.grounded_poi_rate == 0.5
    assert result.hard_feasibility_pass is False
    assert {issue["code"] for issue in result.issues} >= {
        "transfer_time_conflict",
        "ungrounded_poi",
    }


def test_known_opening_hours_and_itinerary_city_are_hard_checks() -> None:
    artifacts = _artifacts()
    artifacts["itinerary"]["itinerary"]["city"] = "上海"
    artifacts["itinerary"]["itinerary"]["days"][0]["stops"][0]["start_time"] = "19:00"

    result = evaluate_plan_quality(
        {"expected_city": "杭州", "expected_days": 1},
        {"destination": "杭州", "days": 1},
        artifacts,
    )

    assert result is not None
    assert result.opening_hours_valid_rate == 0.5
    assert result.city_and_coordinates_valid is False
    assert result.hard_feasibility_pass is False


def test_city_suffix_is_normalized_for_poi_and_itinerary_checks() -> None:
    artifacts = _artifacts()
    artifacts["itinerary"]["itinerary"]["city"] = "杭州市"
    for stop in artifacts["itinerary"]["itinerary"]["days"][0]["stops"]:
        stop["poi"]["city"] = "杭州市"

    result = evaluate_plan_quality(
        {"hard_constraints": {"destination": "杭州", "days": 1}},
        {"destination": "杭州", "days": 1},
        artifacts,
    )

    assert result is not None
    assert result.city_and_coordinates_valid is True
    assert "poi_city_mismatch" not in {issue["code"] for issue in result.issues}


def test_fixed_event_venue_may_be_in_a_neighboring_city() -> None:
    artifacts = _artifacts()
    fixed_stop = artifacts["itinerary"]["itinerary"]["days"][0]["stops"][1]
    fixed_stop["poi"]["name"] = "已预约场馆"
    fixed_stop["poi"]["city"] = "邻市"

    result = evaluate_plan_quality(
        {
            "expected_city": "测试城",
            "expected_days": 1,
            "hard_constraints": {
                "destination": "测试城",
                "fixed_events": [{"location": "已预约场馆", "start": "11:30", "end": "13:30"}],
            },
        },
        {"destination": "测试城", "days": 1},
        artifacts,
    )

    assert result is not None
    assert "poi_city_mismatch" not in {issue["code"] for issue in result.issues}


def test_judge_aggregate_excludes_missing_artifacts_from_applicable_count() -> None:
    summary = aggregate_judge_results([
        {"status": "ok", "rubric": "full_itinerary", "total_score": 85, "reasonable": True, "critical_issues": []},
        {"status": "not_run", "reason": "missing_expected_artifact"},
        {"status": "not_applicable", "reason": "task_does_not_require_itinerary"},
    ])

    assert summary["applicable_count"] == 1
    assert summary["completed_count"] == 1
    assert summary["completion_rate"] == 1.0


def test_route_pace_warning_is_quality_not_hard_feasibility() -> None:
    artifacts = _artifacts()
    route = artifacts["itinerary"]["itinerary"]["days"][0]["stops"][1]["route_from_previous"]
    route["duration_min"] = 65
    artifacts["itinerary"]["itinerary"]["days"][0]["stops"][1]["start_time"] = "12:00"

    result = evaluate_plan_quality(
        {"expected_city": "杭州", "expected_days": 1},
        {"destination": "杭州", "days": 1, "pace": "standard"},
        artifacts,
    )

    assert result is not None
    assert result.transfer_feasible is True
    assert result.hard_feasibility_pass is True
    assert result.rule_quality_pass is False
    assert "route_duration_exceeded" in {issue["code"] for issue in result.issues}


def test_must_visit_quality_uses_provider_alias_identity() -> None:
    artifacts = _artifacts()
    first = artifacts["itinerary"]["itinerary"]["days"][0]["stops"][0]["poi"]
    first.update({
        "source": "provider",
        "canonical_name": "西湖风景名胜区",
        "source_poi_id": "p1",
        "verification_status": "verified",
        "entity_type": "attraction",
        "aliases": ["西湖"],
    })

    result = evaluate_plan_quality(
        {
            "expected_city": "杭州",
            "expected_days": 1,
            "hard_constraints": {"must_visit": ["西湖"]},
        },
        {"destination": "杭州", "days": 1},
        artifacts,
    )

    assert result is not None
    assert result.must_visit_coverage_rate == 1.0
    assert result.hard_feasibility_pass is True


def test_rule_quality_aggregation_uses_evidence_denominators() -> None:
    first = evaluate_plan_quality(
        {"expected_city": "杭州", "expected_days": 1},
        {"destination": "杭州", "days": 1},
        _artifacts(),
    )
    assert first is not None
    summary = aggregate_rule_quality([first.to_dict()])
    assert summary["grounded_poi_rate"] == 1.0
    assert summary["route_coverage_rate"] == 1.0
    assert summary["rule_quality_pass_rate"] == 1.0


def test_judge_parser_validates_exact_rubric_and_ranges() -> None:
    valid = _judge_json()
    parsed = parse_judge_output(f"```json\n{valid}\n```")
    assert parsed["scores"]["schedule"] == 18

    payload = json.loads(valid)
    payload["scores"]["clarity"] = 6
    with pytest.raises(JudgeOutputError, match="clarity"):
        parse_judge_output(json.dumps(payload))


def test_freellmapi_judge_client_bypasses_environment_proxy(tmp_path, monkeypatch) -> None:
    import httpx
    import langchain_openai

    captured: dict[str, object] = {}
    sentinel_client = object()

    def fake_http_client(**kwargs):
        captured["httpx"] = kwargs
        return sentinel_client

    monkeypatch.setattr(httpx, "Client", fake_http_client)

    class FakeChatOpenAI:
        def __init__(self, **kwargs) -> None:
            captured["chat"] = kwargs

        def invoke(self, messages):
            return messages

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)

    judge = PlanQualityJudge(
        JudgeSettings(
            provider="freellmapi",
            api_key="test",
            base_url="http://localhost:31415/v1",
            model="auto",
        ),
        tmp_path,
    )

    assert judge.invoke([{"role": "user", "content": "ping"}])
    assert captured["httpx"] == {"trust_env": False}
    assert captured["chat"]["http_client"] is sentinel_client


def test_judge_retries_computes_total_and_caches(tmp_path) -> None:
    responses = iter(
        [
            "not json",
            SimpleNamespace(
                content=_judge_json(),
                usage_metadata={"input_tokens": 100, "output_tokens": 30},
            ),
        ]
    )
    settings = JudgeSettings(api_key="test", model="glm-4.7-flash", requests_per_second=0)
    judge = PlanQualityJudge(
        settings,
        tmp_path,
        invoke=lambda messages: next(responses),
        sleeper=lambda seconds: None,
    )
    case = _case_output()

    result = judge.evaluate(case, tested_model="deepseek-v4-flash")

    assert result["status"] == "ok"
    assert result["attempt_count"] == 2
    assert result["total_score"] == 86
    assert result["reasonable"] is True
    assert result["usage"]["input_tokens"] == 100

    cached_judge = PlanQualityJudge(
        settings,
        tmp_path,
        invoke=lambda messages: (_ for _ in ()).throw(AssertionError("must use cache")),
        sleeper=lambda seconds: None,
    )
    cached = cached_judge.evaluate(case, resume=True)
    assert cached["cache_hit"] is True
    assert cached["total_score"] == 86


def test_judge_honors_gemini_retry_after_on_429(tmp_path) -> None:
    calls = 0
    sleeps: list[float] = []

    def invoke(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED: quota exceeded; Please retry in 44.914s."
            )
        return _judge_json()

    judge = PlanQualityJudge(
        JudgeSettings(api_key="test", requests_per_second=0),
        tmp_path,
        invoke=invoke,
        sleeper=sleeps.append,
    )

    result = judge.evaluate(_case_output())

    assert result["status"] == "ok"
    assert result["attempt_count"] == 2
    assert sleeps == [44.914]


def test_judge_invocation_failure_is_error_not_not_run_or_missing(tmp_path) -> None:
    judge = PlanQualityJudge(
        JudgeSettings(api_key="test", requests_per_second=0),
        tmp_path,
        invoke=lambda messages: (_ for _ in ()).throw(RuntimeError("provider unavailable")),
        sleeper=lambda seconds: None,
    )

    result = judge.evaluate(_case_output())

    assert result["status"] == "error"
    assert result["attempt_count"] == 2
    assert len(result["errors"]) == 2


def test_judge_without_expected_itinerary_is_not_run(tmp_path) -> None:
    judge = PlanQualityJudge(
        JudgeSettings(api_key="test", requests_per_second=0),
        tmp_path,
        invoke=lambda messages: "{}",
    )
    result = judge.evaluate({"case": {"turns": ["规划测试城两日游"]}, "final_itinerary": None})
    assert result["status"] == "not_run"
    assert result["reason"] == "missing_expected_artifact"


def test_independent_judge_payload_excludes_internal_critic() -> None:
    payload = _judge_payload(_case_output())
    assert payload["itinerary"]["city"] == "杭州"
    assert "critic" not in payload["itinerary"]
    assert "rule_quality" not in payload
    assert "case_id" not in payload
    assert "variant" not in json.dumps(payload).lower()
    assert "tested_model" not in payload
    assert payload["final_answer"]
    assert payload["execution_summary"]["tool_trace"]
    assert payload["automated_evaluator"]
    assert "lat" not in json.dumps(payload["itinerary"])
    assert len(json.dumps(payload, ensure_ascii=False)) < 10_000


def test_independent_judge_payload_includes_bound_transport_evidence() -> None:
    case = _case_output()
    case["final_itinerary"]["domain_inputs"] = {
        "transport": [
            {
                "artifact_id": "routes_return",
                "payload": {
                    "origin_name": "杭州",
                    "destination_name": "上海虹桥",
                    "duration_min": 75,
                    "mode": "public_transport",
                },
            }
        ]
    }

    payload = _judge_payload(case)

    assert payload["tool_facts"]["routes"][0]["destination_name"] == "上海虹桥"


def test_independent_judge_payload_keeps_every_final_grounded_poi() -> None:
    case = _case_output()
    pois = [
        {
            "poi_id": f"poi-{index}",
            "name": f"景点{index}",
            "city": "杭州",
            "category": "scenic",
            "source": "amap",
        }
        for index in range(9)
    ]
    case["final_itinerary"]["itinerary"] = {
        "city": "杭州",
        "days": [{
            "day_index": 1,
            "stops": [
                {"poi": poi, "start_time": "09:00", "duration_min": 30}
                for poi in pois
            ],
        }],
    }
    case["final_artifacts"]["candidates"] = {"city": "杭州", "pois": pois}

    payload = _judge_payload(case)

    evidence_ids = {
        item["poi_id"] for item in payload["tool_facts"]["candidates"]["pois"]
    }
    assert evidence_ids == {poi["poi_id"] for poi in pois}


def test_independent_judge_payload_merges_bound_domain_attractions_and_routes() -> None:
    case = _case_output()
    final_pois = [
        {
            "poi_id": "poi-a",
            "name": "甲景点",
            "city": "杭州",
            "category": "scenic",
            "source": "amap",
        },
        {
            "poi_id": "poi-b",
            "name": "乙景点",
            "city": "杭州",
            "category": "scenic",
            "source": "amap",
        },
    ]
    case["final_itinerary"]["itinerary"] = {
        "city": "杭州",
        "days": [{
            "day_index": 1,
            "stops": [
                {"poi": final_pois[0], "start_time": "09:00", "duration_min": 60},
                {
                    "poi": final_pois[1],
                    "start_time": "11:00",
                    "duration_min": 60,
                    "route_from_previous": {
                        "origin_poi_id": "poi-a",
                        "destination_poi_id": "poi-b",
                        "duration_min": 30,
                        "distance_km": 2.5,
                        "mode": "public_transport",
                        "source": "amap",
                    },
                },
            ],
        }],
    }
    case["final_itinerary"]["domain_inputs"] = {
        "attractions": [{"payload": {"city": "杭州", "pois": final_pois}}],
        "transport": [{"payload": {
            "origin_poi_id": "poi-a",
            "destination_poi_id": "poi-b",
            "duration_min": 30,
            "distance_km": 2.5,
            "mode": "public_transport",
            "source": "amap",
            "evidence_status": "provider_verified",
        }}],
    }
    case["final_artifacts"]["candidates"] = {
        "city": "杭州",
        "pois": [{"poi_id": "unrelated", "name": "旧候选", "source": "amap"}],
    }

    payload = _judge_payload(case)

    evidence_ids = {
        item["poi_id"] for item in payload["tool_facts"]["candidates"]["pois"]
    }
    assert evidence_ids == {"poi-a", "poi-b"}
    assert payload["tool_facts"]["routes"] == [
        case["final_itinerary"]["domain_inputs"]["transport"][0]["payload"]
    ]


def test_independent_judge_payload_includes_structured_return_plan() -> None:
    case = _case_output()
    case["final_itinerary"]["return_plan"] = {
        "required": True,
        "from_city": "杭州",
        "to_location": "上海",
        "arrival_deadline": "21:30",
        "activity_cutoff": "18:30",
        "intercity_segment": {"status": "requires_live_verification"},
    }

    payload = _judge_payload(case)

    assert payload["itinerary"]["return_plan"]["arrival_deadline"] == "21:30"
    assert payload["itinerary"]["return_plan"]["intercity_segment"]["status"] == "requires_live_verification"


def test_independent_judge_payload_includes_lodging_and_budget_plans() -> None:
    case = _case_output()
    case["final_itinerary"]["lodging_plan"] = {
        "status": "recommended_not_booked",
        "hotel": {"name": "王府井大饭店", "source": "amap"},
    }
    case["final_itinerary"]["budget_plan"] = {
        "total_high_cny": 2254.5,
        "user_limit_cny": 4500,
        "within_user_limit": True,
    }

    payload = _judge_payload(case)

    assert payload["itinerary"]["lodging_plan"]["hotel"]["name"] == "王府井大饭店"
    assert payload["itinerary"]["budget_plan"]["within_user_limit"] is True


def test_independent_judge_payload_includes_explicit_free_time_plan() -> None:
    case = _case_output()
    case["final_itinerary"]["free_time_plan"] = {
        "status": "explicitly_reserved",
        "windows": [{
            "day_index": 1,
            "start_time": "13:00",
            "end_time": "16:00",
            "purpose": "午休或低强度自由活动",
        }],
    }

    payload = _judge_payload(case)

    assert payload["itinerary"]["free_time_plan"]["windows"][0]["end_time"] == "16:00"


def test_human_sample_has_20_representative_10_diagnostic_and_10_rescores() -> None:
    cases = []
    for index in range(35):
        case = _case_output(case_id=f"case-{index}")
        case["case"]["category"] = f"category-{index % 5}"
        case["evaluation"]["independent_judge"] = {
            "status": "ok",
            "total_score": 65 + index % 15,
            "reasonable": index % 2 == 0,
            "critical_issues": [],
        }
        cases.append((Path(f"case-{index}.json"), case))

    rows = select_human_review_sample(cases, size=30)

    assert len({row["case_id"] for row in rows}) == 30
    assert sum(row["sample_group"] == "representative" for row in rows) == 20
    assert sum(row["sample_group"] == "diagnostic" for row in rows) == 10
    assert sum(row["sample_group"] == "blind_rescore" for row in rows) == 10


def test_human_calibration_promotes_only_trusted_judge() -> None:
    reviews = []
    cases = {}
    for index in range(20):
        reasonable = index % 2 == 0
        critical = index < 6 and not reasonable
        case_id = f"case-{index}"
        cases[case_id] = _case_output(case_id=case_id)
        cases[case_id]["evaluation"]["independent_judge"] = {
            "status": "ok",
            "total_score": 85 if reasonable else 55,
            "reasonable": reasonable,
            "hard_feasibility_pass": True,
            "critical_issues": (
                [{"severity": "critical", "code": "bad", "evidence": "x"}]
                if critical
                else []
            ),
        }
        review = _human_review(case_id, reasonable, critical, round_index=1)
        reviews.append(review)
        if index < 10:
            reviews.append(_human_review(case_id, reasonable, critical, round_index=2))

    calibration = calibrate_judge(reviews, cases)

    assert calibration["trusted"] is True
    assert calibration["accuracy"] == 1.0
    assert calibration["cohen_kappa"] == 1.0
    assert calibration["critical_issue_recall"] == 1.0
    assert calibration["intra_rater"]["pair_count"] == 10


def test_human_csv_rejects_duplicate_and_out_of_range_reviews(tmp_path) -> None:
    path = tmp_path / "reviews.csv"
    fields = [
        "case_id",
        "case_file",
        "sample_group",
        "reviewer_id",
        "review_round",
        *[f"{name}_score" for name in DIMENSIONS],
        "critical_issue",
        "human_reasonable",
        "notes",
    ]
    row = {
        "case_id": "case-1",
        "case_file": "case-1.json",
        "sample_group": "representative",
        "reviewer_id": "reviewer-1",
        "review_round": "1",
        **{f"{name}_score": str(maximum) for name, maximum in DIMENSIONS.items()},
        "critical_issue": "false",
        "human_reasonable": "true",
        "notes": "",
    }
    row["clarity_score"] = "99"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError, match="invalid human review CSV"):
        load_human_reviews(path)


def test_judge_postprocessing_updates_cases_and_summary(tmp_path) -> None:
    run_dir = _write_run(tmp_path)

    class FakeJudge:
        def evaluate(self, case_output, *, tested_model=None, resume=False):
            return {
                "status": "ok",
                "total_score": 82,
                "reasonable": True,
                "hard_feasibility_pass": True,
                "critical_issues": [],
            }

    metrics = apply_independent_judge(
        run_dir,
        JudgeSettings(api_key="test", model="glm-4.7-flash"),
        judge=FakeJudge(),  # type: ignore[arg-type]
    )

    saved = json.loads(
        (run_dir / "cases" / "case-1__repeat-1.json").read_text(encoding="utf-8")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["evaluation"]["independent_judge"]["total_score"] == 82
    assert metrics["judge_average_score"] == 82
    assert summary["metrics"]["judge_completion_rate"] == 1.0


def test_rule_postprocessing_updates_old_run_without_overwriting_itinerary(tmp_path) -> None:
    run_dir = _write_run(tmp_path)
    cases_dir = run_dir / "cases"
    case = json.loads((cases_dir / "case-1__repeat-1.json").read_text(encoding="utf-8"))
    original = json.loads(json.dumps(case["final_itinerary"]))

    metrics = apply_quality_rules(run_dir)
    saved = json.loads((cases_dir / "case-1__repeat-1.json").read_text(encoding="utf-8"))

    assert metrics["evaluated_itinerary_count"] == 1
    assert saved["final_itinerary"] == original
    assert saved["evaluation"]["rule_quality"]["hard_feasibility_pass"] is True
    assert (run_dir / "quality_report.md").exists()


def _write_run(tmp_path) -> Path:
    run_dir = tmp_path / "runs" / "run-1"
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True)
    case = _case_output()
    (cases_dir / "case-1__repeat-1.json").write_text(
        json.dumps(case, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"metrics": {}, "artifacts": {"model": "deepseek"}, "rows": []}),
        encoding="utf-8",
    )
    return run_dir


def _artifacts() -> dict:
    first = {
        "poi_id": "p1",
        "name": "西湖历史文化景区",
        "city": "杭州",
        "category": "culture",
        "lat": 30.25,
        "lng": 120.15,
        "tags": ["history"],
        "indoor": False,
        "opening_hours": "08:00-18:00",
    }
    second = {
        "poi_id": "p2",
        "name": "浙江省博物馆",
        "city": "杭州",
        "category": "museum",
        "lat": 30.26,
        "lng": 120.16,
        "tags": ["history"],
        "indoor": True,
        "opening_hours": "09:00-17:00",
    }
    return {
        "candidates": {"pois": [first, second]},
        "ranked": {"pois": [{"poi": first}, {"poi": second}]},
        "budget": {"budget_level": "low", "total_low": 500, "total_high": 700},
        "itinerary": {
            "itinerary": {
                "city": "杭州",
                "days": [
                    {
                        "day_index": 1,
                        "stops": [
                            {
                                "poi": first,
                                "start_time": "09:00",
                                "duration_min": 60,
                                "route_from_previous": None,
                            },
                            {
                                "poi": second,
                                "start_time": "11:00",
                                "duration_min": 60,
                                "route_from_previous": {
                                    "duration_min": 30,
                                    "mode": "public_transport",
                                },
                            },
                        ],
                    }
                ],
            },
            "critic": {"passed": True, "issues": []},
        },
    }


def _case_output(case_id: str = "case-1") -> dict:
    artifacts = _artifacts()
    return {
        "schema_version": "product-case-output-v1",
        "case": {
            "case_id": case_id,
            "category": "basic_exploration",
            "turns": ["杭州一日游"],
            "expected_city": "杭州",
            "expected_days": 1,
            "expect_itinerary": True,
            "hard_constraints": {"destination": "杭州", "days": 1},
            "soft_preferences": {},
        },
        "execution": {"repeat": 1, "passed": True, "errors": []},
        "turns": [
            {
                "reply_text": "已为你整理杭州一日行程。",
                "status": "completed",
                "tool_trace": ["search_poi", "plan_route", "plan_and_critique"],
                "duration_ms": 1200,
            }
        ],
        "final_profile": {"destination": "杭州", "days": 1},
        "final_artifacts": artifacts,
        "final_itinerary": artifacts["itinerary"],
        "evaluation": {
            "rule_metrics": {
                "strict_task_success": True,
                "gating_passed": True,
                "grounding_ok": True,
            },
            "rule_quality": {"hard_feasibility_pass": True},
            "independent_judge": None,
            "human_review": None,
        },
    }


def _judge_json() -> str:
    return json.dumps(
        {
            "scores": {
                "schedule": 18,
                "route": 18,
                "constraints": 18,
                "personalization": 12,
                "completeness": 8,
                "diversity": 7,
                "clarity": 5,
            },
            "critical_issues": [],
            "insufficient_evidence": ["opening_hours"],
            "reason": "整体可执行。",
        },
        ensure_ascii=False,
    )


def _human_review(
    case_id: str, reasonable: bool, critical: bool, *, round_index: int
) -> dict:
    total = 85 if reasonable else 55
    scores = {name: 0 for name in DIMENSIONS}
    remaining = total
    for name, maximum in DIMENSIONS.items():
        scores[name] = min(maximum, remaining)
        remaining -= scores[name]
    return {
        "case_id": case_id,
        "case_file": f"{case_id}.json",
        "sample_group": "representative" if round_index == 1 else "blind_rescore",
        "reviewer_id": "reviewer-1",
        "review_round": round_index,
        "scores": scores,
        "total_score": total,
        "critical_issue": critical,
        "human_reasonable": reasonable,
        "notes": "",
    }
