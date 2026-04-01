from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import aggregate_runs_csv


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_run_layout(run_dir: Path, *, percent: float | None = None) -> None:
    processed_subdir_name = aggregate_runs_csv._processed_subdir_name(percent=percent)
    (run_dir / processed_subdir_name).mkdir(parents=True, exist_ok=True)


def _write_key_stats(run_dir: Path, payload: object, *, percent: float | None = None) -> None:
    _write_json(
        run_dir / aggregate_runs_csv._summary_rel_path(
            aggregate_runs_csv.KEY_STATS_SUBPATH,
            percent=percent,
        ),
        payload,
    )


def test_discover_frequency_run_dirs_filters_by_slug_and_post_processed(tmp_path: Path) -> None:
    good = tmp_path / "batch-a" / "core-345-1005"
    bad_name = tmp_path / "batch-a" / "run-345-1005"
    bad_layout = tmp_path / "batch-a" / "core-345-810"

    _write_minimal_run_layout(good)
    bad_name.mkdir(parents=True)
    bad_layout.mkdir(parents=True)

    discovered = aggregate_runs_csv.discover_frequency_run_dirs(tmp_path)

    assert discovered == [good.resolve()]


def test_discover_non_freq_run_dirs_uses_post_processed_layout(tmp_path: Path) -> None:
    good = tmp_path / "qps0_4" / "20260319T052451Z"
    bad = tmp_path / "qps0_5" / "20260319T052452Z"
    _write_minimal_run_layout(good)
    _write_minimal_run_layout(bad)
    _write_json(
        good / aggregate_runs_csv.POWER_SUMMARY_REL_PATH,
        {"power_stats_w": {"avg": 321.0}},
    )

    discovered = aggregate_runs_csv.discover_non_freq_run_dirs(tmp_path)

    assert discovered == [good.resolve()]


def test_discover_frequency_run_dirs_uses_percent_selected_layout(tmp_path: Path) -> None:
    good = tmp_path / "batch-a" / "core-345-1005"
    bad_default = tmp_path / "batch-a" / "core-345-810"

    _write_minimal_run_layout(good, percent=50)
    _write_minimal_run_layout(bad_default)

    discovered = aggregate_runs_csv.discover_frequency_run_dirs(tmp_path, percent=50)

    assert discovered == [good.resolve()]


def test_discover_non_freq_run_dirs_uses_percent_selected_layout(tmp_path: Path) -> None:
    good = tmp_path / "qps0_4" / "20260319T052451Z"
    bad = tmp_path / "qps0_5" / "20260319T052452Z"
    _write_minimal_run_layout(good, percent=12.5)
    _write_minimal_run_layout(bad)
    _write_json(
        good / aggregate_runs_csv._summary_rel_path(
            aggregate_runs_csv.POWER_SUMMARY_SUBPATH,
            percent=12.5,
        ),
        {"power_stats_w": {"avg": 321.0}},
    )

    discovered = aggregate_runs_csv.discover_non_freq_run_dirs(tmp_path, percent=12.5)

    assert discovered == [good.resolve()]


def test_build_rows_extracts_requested_metrics(tmp_path: Path) -> None:
    run_1005 = tmp_path / "20260322T034746Z" / "core-345-1005"
    run_810 = tmp_path / "20260322T034746Z" / "core-345-810"
    _write_minimal_run_layout(run_1005)
    _write_minimal_run_layout(run_810)

    _write_json(
        run_1005 / aggregate_runs_csv.POWER_SUMMARY_REL_PATH,
        {
            "power_stats_w": {"avg": 260.5},
            "power_points": [
                {"power_w": 250.0},
                {"power_w": 260.0},
                {"power_w": 270.0},
            ],
        },
    )
    _write_json(
        run_1005 / aggregate_runs_csv.JOB_THROUGHPUT_REL_PATH,
        {
            "throughput_points": [
                {"throughput_jobs_per_s": 0.2},
                {"throughput_jobs_per_s": 0.4},
            ]
        },
    )
    _write_json(
        run_1005 / aggregate_runs_csv.GLOBAL_SUMMARY_REL_PATH,
        {
            "agent_time_breakdown_s": {
                "agent_total_time_stats_s": {"avg": 200.0, "std": 20.0},
                "llm_time_stats_s": {"avg": 150.0, "std": 15.0},
                "non_llm_time_stats_s": {"avg": 50.0, "std": 5.0},
            }
        },
    )
    _write_json(
        run_1005 / aggregate_runs_csv.LLM_REQUEST_STATS_REL_PATH,
        {
            "average_stage_speed_tokens_per_s": {
                "prefill": {"avg_tokens_per_s": 12345.0, "std_tokens_per_s": 123.0},
                "decode": {"avg_tokens_per_s": 30.0, "std_tokens_per_s": 3.0},
            },
            "metrics": {
                "completion_tokens": {"avg": 111.0, "std": 11.0},
                "cached_tokens": {"avg": 222.0, "std": 22.0},
                "duration_ms": {"avg": 333.0, "std": 33.0},
                "prompt_tokens": {"avg": 444.0, "std": 44.0},
            },
        },
    )
    _write_key_stats(
        run_1005,
        {
            "job_concurrency": {
                "sample_count": 2,
                "avg": 1.5,
                "min": 1.0,
                "max": 2.0,
                "std": 0.5,
            },
            "gateway": {
                "stack": {
                    "prompt_tokens": {
                        "sample_count": 3,
                        "avg": 10.0,
                        "min": 5.0,
                        "max": 15.0,
                        "std": 4.0,
                    }
                },
                "llm_requests": {
                    "average_stage_speed_tokens_per_s": {
                        "prefill": {
                            "eligible_request_count": 3,
                            "excluded_request_count": 1,
                            "avg": 12345.0,
                            "min": 12000.0,
                            "max": 13000.0,
                        }
                    },
                    "metrics": {
                        "prompt_tokens": {
                            "count": 7,
                            "avg": 444.0,
                            "min": 400.0,
                            "max": 500.0,
                        }
                    },
                },
            },
        },
    )

    # Include only one metric in the lower frequency run to validate blanks + sorting.
    _write_json(
        run_810 / aggregate_runs_csv.POWER_SUMMARY_REL_PATH,
        {"power_stats_w": {"avg": 100.0}},
    )

    rows = aggregate_runs_csv.build_rows(
        tmp_path,
        expected_core_min_mhz=345,
        allow_duplicate_frequency=False,
    )

    assert [row["frequency_mhz"] for row in rows] == [810, 1005]
    row_810 = rows[0]
    row_1005 = rows[1]

    assert row_810["avg_power_w"] == 100.0
    assert row_810["std_power_w"] is None
    assert row_810["avg_job_throughput_jobs_per_s"] is None
    assert row_810["std_job_throughput_jobs_per_s"] is None

    assert row_1005["avg_power_w"] == 260.5
    assert row_1005["std_power_w"] == pytest.approx(8.1649658093)
    assert row_1005["avg_job_throughput_jobs_per_s"] == 0.30000000000000004
    assert row_1005["std_job_throughput_jobs_per_s"] == pytest.approx(0.1)
    assert row_1005["agent_total_time_avg_s"] == 200.0
    assert row_1005["agent_total_time_std_s"] == 20.0
    assert row_1005["llm_time_avg_s"] == 150.0
    assert row_1005["llm_time_std_s"] == 15.0
    assert row_1005["non_llm_time_avg_s"] == 50.0
    assert row_1005["non_llm_time_std_s"] == 5.0
    assert row_1005["prefill_avg_tokens_per_s"] == 12345.0
    assert row_1005["prefill_std_tokens_per_s"] == 123.0
    assert row_1005["decode_avg_tokens_per_s"] == 30.0
    assert row_1005["decode_std_tokens_per_s"] == 3.0
    assert row_1005["completion_tokens_avg"] == 111.0
    assert row_1005["completion_tokens_std"] == 11.0
    assert row_1005["cached_tokens_avg"] == 222.0
    assert row_1005["cached_tokens_std"] == 22.0
    assert row_1005["duration_ms_avg"] == 333.0
    assert row_1005["duration_ms_std"] == 33.0
    assert row_1005["prompt_tokens_avg"] == 444.0
    assert row_1005["prompt_tokens_std"] == 44.0
    assert row_1005["key_stats_job_concurrency_avg"] == 1.5
    assert row_1005["key_stats_job_concurrency_min"] == 1.0
    assert row_1005["key_stats_job_concurrency_max"] == 2.0
    assert row_1005["key_stats_job_concurrency_std"] == 0.5
    assert row_1005["key_stats_gateway_stack_prompt_tokens_avg"] == 10.0
    assert (
        row_1005["key_stats_gateway_llm_requests_average_stage_speed_tokens_per_s_prefill_avg"]
        == 12345.0
    )
    assert row_1005["key_stats_gateway_llm_requests_metrics_prompt_tokens_count"] == 7.0
    assert row_810.get("key_stats_job_concurrency_avg") is None


def test_build_rows_non_freq_mode_accepts_non_core_directory_names(tmp_path: Path) -> None:
    run_dir = tmp_path / "qps0_4" / "20260319T052451Z"
    _write_minimal_run_layout(run_dir)
    _write_json(
        run_dir / aggregate_runs_csv.POWER_SUMMARY_REL_PATH,
        {"power_stats_w": {"avg": 123.4}},
    )

    rows = aggregate_runs_csv.build_rows(
        tmp_path,
        mode="non-freq",
        expected_core_min_mhz=345,
        allow_duplicate_frequency=False,
    )

    assert len(rows) == 1
    assert rows[0]["run_path"] == "qps0_4/20260319T052451Z"
    assert rows[0]["frequency_mhz"] is None
    assert rows[0]["avg_power_w"] == 123.4


def test_build_rows_reads_selected_percent_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "qps0_4" / "20260319T052451Z"
    _write_minimal_run_layout(run_dir, percent=50)
    _write_json(
        run_dir / aggregate_runs_csv._summary_rel_path(
            aggregate_runs_csv.POWER_SUMMARY_SUBPATH,
            percent=50,
        ),
        {"power_stats_w": {"avg": 555.5}},
    )

    rows = aggregate_runs_csv.build_rows(
        tmp_path,
        mode="non-freq",
        percent=50,
        expected_core_min_mhz=345,
        allow_duplicate_frequency=False,
    )

    assert len(rows) == 1
    assert rows[0]["avg_power_w"] == 555.5


def test_build_rows_rejects_duplicate_frequency_by_default(tmp_path: Path) -> None:
    run_a = tmp_path / "20260322T000000Z" / "core-345-810"
    run_b = tmp_path / "20260322T010000Z" / "core-345-810"
    _write_minimal_run_layout(run_a)
    _write_minimal_run_layout(run_b)

    with pytest.raises(ValueError, match="Duplicate frequency_mhz values"):
        aggregate_runs_csv.build_rows(
            tmp_path,
            expected_core_min_mhz=345,
            allow_duplicate_frequency=False,
        )


def test_main_writes_default_output_under_root_subdir(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch" / "core-345-1005"
    _write_minimal_run_layout(run_dir)
    _write_json(
        run_dir / aggregate_runs_csv.POWER_SUMMARY_REL_PATH,
        {"power_stats_w": {"avg": 123.4}},
    )
    _write_key_stats(
        run_dir,
        {
            "job_concurrency": {
                "sample_count": 2,
                "avg": 1.5,
                "min": 1.0,
                "max": 2.0,
                "std": 0.5,
            }
        },
    )

    exit_code = aggregate_runs_csv.main(["--root-dir", str(tmp_path)])

    assert exit_code == 0
    output_path = (
        tmp_path
        / aggregate_runs_csv.DEFAULT_OUTPUT_SUBDIR
        / aggregate_runs_csv.DEFAULT_FREQ_OUTPUT_NAME
    )
    avg_only_output_path = output_path.with_name(
        f"{output_path.stem}{aggregate_runs_csv.AVG_ONLY_SUFFIX}{output_path.suffix}"
    )
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    with avg_only_output_path.open("r", encoding="utf-8", newline="") as handle:
        avg_only_reader = csv.DictReader(handle)
        avg_only_rows = list(avg_only_reader)
        avg_only_fieldnames = avg_only_reader.fieldnames or []

    assert len(rows) == 1
    assert len(avg_only_rows) == 1
    assert rows[0]["frequency_mhz"] == "1005"
    assert rows[0]["avg_power_w"] == "123.4"
    assert "std_power_w" in fieldnames
    assert "key_stats_job_concurrency_min" in fieldnames
    assert "key_stats_job_concurrency_std" in fieldnames
    assert "std_power_w" not in avg_only_fieldnames
    assert "key_stats_job_concurrency_min" not in avg_only_fieldnames
    assert "key_stats_job_concurrency_max" not in avg_only_fieldnames
    assert "key_stats_job_concurrency_std" not in avg_only_fieldnames
    assert "key_stats_job_concurrency_avg" in avg_only_fieldnames
    assert "key_stats_job_concurrency_sample_count" in avg_only_fieldnames
    assert avg_only_rows[0]["avg_power_w"] == "123.4"
    assert avg_only_rows[0]["key_stats_job_concurrency_avg"] == "1.5"


def test_main_non_freq_writes_mode_specific_default_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "qps0_4" / "20260319T052451Z"
    _write_minimal_run_layout(run_dir)
    _write_json(
        run_dir / aggregate_runs_csv.POWER_SUMMARY_REL_PATH,
        {"power_stats_w": {"avg": 222.2}},
    )

    exit_code = aggregate_runs_csv.main(
        ["--mode", "non-freq", "--root-dir", str(tmp_path)]
    )

    assert exit_code == 0
    output_path = (
        tmp_path
        / aggregate_runs_csv.DEFAULT_OUTPUT_SUBDIR
        / aggregate_runs_csv.DEFAULT_NON_FREQ_OUTPUT_NAME
    )
    avg_only_output_path = output_path.with_name(
        f"{output_path.stem}{aggregate_runs_csv.AVG_ONLY_SUFFIX}{output_path.suffix}"
    )
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with avg_only_output_path.open("r", encoding="utf-8", newline="") as handle:
        avg_only_rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert len(avg_only_rows) == 1
    assert rows[0]["run_path"] == "qps0_4/20260319T052451Z"
    assert rows[0]["frequency_mhz"] == ""
    assert rows[0]["avg_power_w"] == "222.2"
    assert avg_only_rows[0]["avg_power_w"] == "222.2"


def test_main_percent_writes_mode_specific_percent_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "qps0_4" / "20260319T052451Z"
    _write_minimal_run_layout(run_dir, percent=50)
    _write_json(
        run_dir / aggregate_runs_csv._summary_rel_path(
            aggregate_runs_csv.POWER_SUMMARY_SUBPATH,
            percent=50,
        ),
        {"power_stats_w": {"avg": 222.2}},
    )

    exit_code = aggregate_runs_csv.main(
        ["--mode", "non-freq", "--root-dir", str(tmp_path), "--percent", "50"]
    )

    assert exit_code == 0
    output_path = (
        tmp_path
        / "power-stats-50"
        / aggregate_runs_csv.DEFAULT_NON_FREQ_OUTPUT_NAME
    )
    avg_only_output_path = output_path.with_name(
        f"{output_path.stem}{aggregate_runs_csv.AVG_ONLY_SUFFIX}{output_path.suffix}"
    )
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with avg_only_output_path.open("r", encoding="utf-8", newline="") as handle:
        avg_only_rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["avg_power_w"] == "222.2"
    assert len(avg_only_rows) == 1
    assert avg_only_rows[0]["avg_power_w"] == "222.2"
