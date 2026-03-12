from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import aggregate_runs_csv


def _write_summary(
    run_dir: Path,
    *,
    total_duration_s: float,
    avg: float,
    min_value: float,
    max_value: float,
) -> None:
    output_dir = run_dir / "post-processed" / "global"
    output_dir.mkdir(parents=True)
    (output_dir / "trial-timing-summary.json").write_text(
        json.dumps(
            {
                "total_duration_s": total_duration_s,
                "trial_duration_stats_s": {
                    "avg": avg,
                    "min": min_value,
                    "max": max_value,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_vllm_stats(run_dir: Path, metrics: dict[str, dict[str, object]]) -> None:
    output_dir = run_dir / "post-processed" / "vllm-log"
    output_dir.mkdir(parents=True)
    (output_dir / "gauge-counter-timeseries.stats.json").write_text(
        json.dumps({"metrics": metrics}),
        encoding="utf-8",
    )


def _write_gateway_usage(
    run_dir: Path,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    cached_prompt_tokens: int,
    prefill_prompt_tokens: int,
    avg_worker_max_request_length: float | None = None,
) -> None:
    output_dir = run_dir / "post-processed" / "gateway" / "usage"
    output_dir.mkdir(parents=True)
    (output_dir / "usage-summary.json").write_text(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "cached_prompt_tokens": cached_prompt_tokens,
                    "prefill_prompt_tokens": prefill_prompt_tokens,
                    "avg_worker_max_request_length": avg_worker_max_request_length,
                }
            }
        ),
        encoding="utf-8",
    )


def test_discover_global_summary_paths_filters_layout(tmp_path: Path) -> None:
    good = tmp_path / "a" / "post-processed" / "global" / "trial-timing-summary.json"
    bad = tmp_path / "b" / "global" / "trial-timing-summary.json"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_text("{}", encoding="utf-8")
    bad.write_text("{}", encoding="utf-8")

    discovered = aggregate_runs_csv.discover_global_summary_paths(tmp_path)

    assert discovered == [good.resolve()]


def test_build_rows_sorts_by_relative_run_path(tmp_path: Path) -> None:
    run_b = tmp_path / "z-run"
    run_a = tmp_path / "a-run"
    _write_summary(run_b, total_duration_s=20.0, avg=2.0, min_value=1.0, max_value=3.0)
    _write_summary(run_a, total_duration_s=10.0, avg=1.0, min_value=0.5, max_value=1.5)

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert [row["run_path"] for row in rows] == ["a-run", "z-run"]
    assert rows[0]["total_duration_s"] == 10.0
    assert rows[0]["trial_duration_avg_s"] == 1.0
    assert rows[0]["trial_duration_min_s"] == 0.5
    assert rows[0]["trial_duration_max_s"] == 1.5


def test_build_rows_sorts_numeric_path_levels_by_value(tmp_path: Path) -> None:
    for value in ["10", "1", "2"]:
        _write_summary(
            tmp_path / "root" / value / "leaf",
            total_duration_s=float(value),
            avg=float(value),
            min_value=float(value),
            max_value=float(value),
        )

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert [row["run_path"] for row in rows] == [
        "root/1/leaf",
        "root/2/leaf",
        "root/10/leaf",
    ]


def test_build_rows_adds_vllm_metric_avg_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    _write_summary(
        run_dir,
        total_duration_s=10.0,
        avg=1.0,
        min_value=0.5,
        max_value=1.5,
    )
    _write_vllm_stats(
        run_dir,
        {
            "vllm:kv_cache_usage_perc|engine=0": {
                "name": "vllm:kv_cache_usage_perc",
                "avg": 0.25,
                "min": 0.1,
                "max": 0.4,
                "sample_count": 4,
            },
            "vllm:num_requests_running|engine=0": {
                "name": "vllm:num_requests_running",
                "avg": 1.0,
                "min": 0.0,
                "max": 2.0,
                "sample_count": 1,
            },
            "vllm:num_requests_running|engine=1": {
                "name": "vllm:num_requests_running",
                "avg": 3.0,
                "min": 1.0,
                "max": 5.0,
                "sample_count": 3,
            },
            "vllm:num_requests_waiting|engine=0": {
                "name": "vllm:num_requests_waiting",
                "avg": 0.5,
                "min": 0.0,
                "max": 1.0,
                "sample_count": 2,
            },
        },
    )

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert rows[0]["vllm:kv_cache_usage_perc:avg"] == 0.25
    assert rows[0]["vllm:kv_cache_usage_perc:min"] == 0.1
    assert rows[0]["vllm:kv_cache_usage_perc:max"] == 0.4
    assert rows[0]["vllm:num_requests_running:avg"] == 2.5
    assert rows[0]["vllm:num_requests_running:min"] == 0.0
    assert rows[0]["vllm:num_requests_running:max"] == 5.0
    assert rows[0]["vllm:num_requests_waiting:avg"] == 0.5
    assert rows[0]["vllm:num_requests_waiting:min"] == 0.0
    assert rows[0]["vllm:num_requests_waiting:max"] == 1.0


def test_build_rows_adds_gateway_usage_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    _write_summary(
        run_dir,
        total_duration_s=10.0,
        avg=1.0,
        min_value=0.5,
        max_value=1.5,
    )
    _write_gateway_usage(
        run_dir,
        prompt_tokens=120,
        generation_tokens=30,
        cached_prompt_tokens=70,
        prefill_prompt_tokens=50,
        avg_worker_max_request_length=75.5,
    )

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert rows[0]["prompt_tokens"] == 120
    assert rows[0]["generation_tokens"] == 30
    assert rows[0]["cached_prompt_tokens"] == 70
    assert rows[0]["prefill_prompt_tokens"] == 50
    assert rows[0]["avg_worker_max_request_length"] == 75.5


def test_build_rows_gateway_usage_fallbacks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    _write_summary(
        run_dir,
        total_duration_s=10.0,
        avg=1.0,
        min_value=0.5,
        max_value=1.5,
    )
    output_dir = run_dir / "post-processed" / "gateway" / "usage"
    output_dir.mkdir(parents=True)
    (output_dir / "usage-summary.json").write_text(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 33,
                    "cached_prompt_tokens": 70,
                }
            }
        ),
        encoding="utf-8",
    )

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert rows[0]["prompt_tokens"] == 120
    assert rows[0]["generation_tokens"] == 33
    assert rows[0]["cached_prompt_tokens"] == 70
    assert rows[0]["prefill_prompt_tokens"] == 50


def test_main_writes_default_csv(tmp_path: Path) -> None:
    _write_summary(
        tmp_path / "x" / "y",
        total_duration_s=123.0,
        avg=10.0,
        min_value=7.0,
        max_value=12.0,
    )

    exit_code = aggregate_runs_csv.main(["--root-dir", str(tmp_path)])

    assert exit_code == 0
    output_path = tmp_path / "trial-timing-summary.csv"
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["run_path"] == "x/y"
    assert rows[0]["total_duration_s"] == "123.0"
    assert rows[0]["trial_duration_avg_s"] == "10.0"
    assert rows[0]["trial_duration_min_s"] == "7.0"
    assert rows[0]["trial_duration_max_s"] == "12.0"
    assert rows[0]["prompt_tokens"] == ""
    assert rows[0]["generation_tokens"] == ""
    assert rows[0]["cached_prompt_tokens"] == ""
    assert rows[0]["prefill_prompt_tokens"] == ""
    assert rows[0]["avg_worker_max_request_length"] == ""
    assert rows[0]["vllm:kv_cache_usage_perc:avg"] == ""
    assert rows[0]["vllm:kv_cache_usage_perc:min"] == ""
    assert rows[0]["vllm:kv_cache_usage_perc:max"] == ""
    assert rows[0]["vllm:num_requests_running:avg"] == ""
    assert rows[0]["vllm:num_requests_running:min"] == ""
    assert rows[0]["vllm:num_requests_running:max"] == ""
    assert rows[0]["vllm:num_requests_waiting:avg"] == ""
    assert rows[0]["vllm:num_requests_waiting:min"] == ""
    assert rows[0]["vllm:num_requests_waiting:max"] == ""
