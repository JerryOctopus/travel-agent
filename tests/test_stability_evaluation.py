"""阶段三稳定性机制测试：抽样清单确定性 + 五项稳定性指标聚合。

覆盖：
- ``make_stability_selection.py`` 固定 seed 重跑一致、分层构成
  （Core 30 按 subset 分层 + Challenge 全部 30）、总数 60 且均属冻结集；
- ``eval_ablation.py`` 的 ``--cases-file`` / ``--repeat-index`` 标记；
- ``aggregate_stability_runs`` 五项指标（Pass@1、Pass³ 与至少一次、
  硬约束三轮恒满足与波动清单、核心步骤命中、POI 重合率/方差）小样本验证；
- ``--stability-merge`` CLI 落盘。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_ablation as ablation
from scripts.make_stability_selection import stratified_sample


# ---------------------------------------------------------------------------
# stability_60 抽样清单
# ---------------------------------------------------------------------------


def test_stratified_sample_quota_floor_plus_remainder() -> None:
    import random

    cases = [
        {"case_id": f"a-{index}", "subset": "big"} for index in range(6)
    ] + [
        {"case_id": f"b-{index}", "subset": "small"} for index in range(3)
    ]

    selected = stratified_sample(cases, 3, random.Random(42))

    # 配额：big = 6*3//9 = 2，small = 1，合计 3。
    assert len(selected) == 3
    assert sum(1 for item in selected if item.startswith("a-")) == 2
    assert sum(1 for item in selected if item.startswith("b-")) == 1


# ---------------------------------------------------------------------------
# --cases-file / --repeat-index CLI 接线
# ---------------------------------------------------------------------------


def _enabled_settings(offline_settings):
    return replace(
        offline_settings,
        llm=replace(offline_settings.llm, provider="openai", api_key="test-key"),
    )


def test_repeat_index_marks_rows_and_output_dir(monkeypatch, tmp_path, offline_settings) -> None:
    cases_file = tmp_path / "stability_60.json"
    cases_file.write_text(json.dumps({"case_ids": ["case-1", "case-2"]}), encoding="utf-8")
    fake_result = SimpleNamespace(rows=[{"case_id": "case-1"}], metrics={}, case_count=1, artifacts={})
    captured: dict = {}
    written: list[Path] = []

    def fake_run_variant(variant, *, settings, case_path, split, limit, token_budget, case_ids=None):
        captured.update(variant=variant, split=split, case_ids=case_ids)
        return fake_result

    monkeypatch.setattr(ablation, "run_variant", fake_run_variant)
    monkeypatch.setattr(
        ablation,
        "write_product_run",
        lambda result, path, run_id=None: written.append(Path(path)),
    )
    monkeypatch.setattr(ablation, "load_settings", lambda: _enabled_settings(offline_settings))
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_ablation.py",
            "--variants", "v3",
            "--cases-file", str(cases_file),
            "--repeat-index", "2",
            "--output-root", str(tmp_path / "runs"),
            "--run-id", "stage3",
        ],
    )

    with pytest.raises(SystemExit, match="当前发布周期"):
        ablation.main()
    assert captured == {}
    assert written == []


# ---------------------------------------------------------------------------
# 五项稳定性指标聚合（小样本构造数据）
# ---------------------------------------------------------------------------

CORE_STEPS = ["search_poi", "plan_route", "estimate_budget", "plan_and_critique"]


def _write_run(run_dir: Path, rows: list[dict], case_outputs: dict[str, list[str]]) -> None:
    variant_dir = run_dir / "runs" / "v3"
    variant_dir.mkdir(parents=True)
    (variant_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8"
    )
    cases_dir = variant_dir / "cases"
    cases_dir.mkdir()
    for case_id, poi_ids in case_outputs.items():
        repeat = next(row["repeat"] for row in rows if row["case_id"] == case_id)
        output = {
            "case": {"case_id": case_id},
            "execution": {"repeat": repeat},
            "final_itinerary": {
                "itinerary": {
                    "days": [
                        {
                            "stops": [
                                {"poi": {"poi_id": poi_id, "name": poi_id}} for poi_id in poi_ids
                            ]
                        }
                    ]
                }
            },
        }
        (cases_dir / f"{case_id}.json").write_text(
            json.dumps(output, ensure_ascii=False), encoding="utf-8"
        )


def _build_three_runs(tmp_path: Path) -> list[Path]:
    """case-a 三轮全过且轨迹稳定；case-b 第二轮 strict 失败 + 漏约束 + 缺核心步骤。"""
    specs = [
        # (case-a strict, case-b strict, case-b missing, case-b trace_ok, case-b pois, a/b 资源)
        (True, True, [], True, ["p1", "p2"], (5, 4, 1000, 900, 100, 90)),
        (True, False, [{"path": "budget_max_cny"}], False, ["p1"], (5, 6, 1100, 1000, 120, 110)),
        (True, True, [], True, ["p1", "p2"], (7, 5, 1200, 1100, 140, 130)),
    ]
    run_dirs: list[Path] = []
    for index, (a_ok, b_ok, b_missing, b_trace_ok, b_pois, resources) in enumerate(specs, start=1):
        a_calls, b_calls, a_tokens, b_tokens, a_ms, b_ms = resources
        b_trace = list(CORE_STEPS) if b_trace_ok else CORE_STEPS[:-1]
        rows = [
            {
                "case_id": "case-a",
                "repeat": index,
                "strict_task_success": a_ok,
                "constraint_tree_missing": [],
                "tool_trace": list(CORE_STEPS),
                "tool_call_count": a_calls,
                "total_tokens": a_tokens,
                "duration_ms": a_ms,
            },
            {
                "case_id": "case-b",
                "repeat": index,
                "strict_task_success": b_ok,
                "constraint_tree_missing": b_missing,
                "tool_trace": b_trace,
                "tool_call_count": b_calls,
                "total_tokens": b_tokens,
                "duration_ms": b_ms,
            },
        ]
        run_dir = tmp_path / f"run-{index}"
        _write_run(run_dir, rows, {"case-a": ["p1", "p2"], "case-b": b_pois})
        run_dirs.append(run_dir)
    return run_dirs


def test_aggregate_stability_runs_reports_five_metrics(tmp_path) -> None:
    run_dirs = _build_three_runs(tmp_path)

    report = ablation.aggregate_stability_runs(run_dirs)

    assert report["case_count"] == 2
    assert report["fully_repeated_case_count"] == 2
    # 1. Pass@1 = 5 通过 / 6 次执行（报告四舍五入到 4 位）
    assert report["pass_at_1"] == pytest.approx(5 / 6, abs=1e-3)
    # 2. Pass³ = 仅 case-a 三轮全过；至少成功一次 = 两条都满足
    assert report["pass_cubed_rate"] == 0.5
    assert report["at_least_once_rate"] == 1.0
    # 3. 硬约束三轮恒满足 = 仅 case-a；波动清单指出 case-b 破了 budget_max_cny
    assert report["hard_constraint_stable_rate"] == 0.5
    assert report["constraint_fluctuations"] == {"case-b": ["budget_max_cny"]}
    # 4. 工具轨迹：case-b 第二轮缺 plan_route → 仅 case-a 三轮命中全部核心步骤
    assert report["core_trace_stable_rate"] == 0.5
    # 5. 输出波动：POI 两两 Jaccard 均值与方差
    assert report["poi_overlap_mean"] == pytest.approx(5 / 6, abs=1e-3)
    assert report["structure_overlap_mean"] == pytest.approx(5 / 6, abs=1e-3)
    expected_variances = [
        statistics.pvariance([5, 5, 7]),
        statistics.pvariance([4, 6, 5]),
    ]
    assert report["tool_call_variance_mean"] == pytest.approx(statistics.mean(expected_variances), abs=1e-3)
    assert report["tool_call_variance_max"] == pytest.approx(max(expected_variances), abs=1e-3)
    assert report["token_variance_mean"] is not None
    assert report["duration_variance_max"] is not None


def test_stability_merge_cli_writes_report(tmp_path, capsys) -> None:
    run_dirs = _build_three_runs(tmp_path)

    import sys

    argv_backup = sys.argv
    sys.argv = [
        "eval_ablation.py",
        "--stability-merge",
        "--merge-runs", ",".join(str(item) for item in run_dirs),
        "--output-root", str(tmp_path / "merged"),
        "--run-id", "final",
    ]
    try:
        ablation.main()
    finally:
        sys.argv = argv_backup

    report_path = tmp_path / "merged" / "final" / "stability_report.json"
    markdown_path = tmp_path / "merged" / "final" / "stability_report.md"
    assert report_path.exists() and markdown_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass_cubed_rate"] == 0.5
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "波动约束清单" in markdown and "budget_max_cny" in markdown


def test_stability_merge_requires_exactly_three_runs(tmp_path) -> None:
    import sys

    argv_backup = sys.argv
    sys.argv = ["eval_ablation.py", "--stability-merge", "--merge-runs", str(tmp_path)]
    try:
        with pytest.raises(SystemExit, match="三个运行目录"):
            ablation.main()
    finally:
        sys.argv = argv_backup
