from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_minimal_inputs(post_processed_dir: Path) -> None:
    _write_json(
        post_processed_dir / "global" / "trial-timing-summary.json",
        {"source_run_dir": str(post_processed_dir.parent)},
    )
    _write_json(
        post_processed_dir / "vllm-log" / "gauge-counter-timeseries.json",
        {
            "metrics": {
                "vllm:kv_cache_usage_perc|engine=0": {
                    "name": "vllm:kv_cache_usage_perc",
                    "labels": {"engine": "0"},
                    "value": [0.1, 0.3],
                },
                "vllm:kv_cache_usage_perc|engine=1": {
                    "name": "vllm:kv_cache_usage_perc",
                    "labels": {"engine": "1"},
                    "value": [0.5, 0.7],
                },
            }
        },
    )
    _write_json(
        post_processed_dir / "job-concurrency" / "job-concurrency-timeseries.json",
        {
            "service_failure_detected": True,
            "service_failure_cutoff_time_utc": "2026-03-29T00:00:00Z",
            "concurrency_points": [
                {"second": 0, "concurrency": 1},
                {"second": 1, "concurrency": 2},
                {"second": 2, "concurrency": 1},
            ],
        },
    )
    _write_json(
        post_processed_dir / "job-throughput" / "job-throughput-timeseries.json",
        {
            "throughput_points": [
                {"time_s": 0.0, "throughput_jobs_per_s": 0.2},
                {"time_s": 1.0, "throughput_jobs_per_s": 0.5},
                {"time_s": 2.0, "throughput_jobs_per_s": 0.8},
            ]
        },
    )
    _write_json(
        post_processed_dir / "gateway" / "stack-kv" / "kv-usage-stacked-histogram.json",
        {
            "points": [
                {"second": 0, "accumulated_value": 10.0},
                {"second": 1, "accumulated_value": 20.0},
                {"second": 2, "accumulated_value": 30.0},
            ]
        },
    )
    _write_json(
        post_processed_dir / "gateway" / "stack-context" / "context-usage-stacked-histogram.json",
        {
            "points": [
                {"second": 0, "accumulated_value": 7.0},
                {"second": 1, "accumulated_value": 9.0},
                {"second": 2, "accumulated_value": 11.0},
            ]
        },
    )
    for metric, relative_path in extract_run.STACK_INPUT_REL_PATHS.items():
        base_value = {
            "prompt_tokens": 1.0,
            "cached_tokens": 2.0,
            "compute_prompt_tokens": 3.0,
            "completion_tokens": 4.0,
            "compute_prompt_plus_completion_tokens": 5.0,
        }[metric]
        _write_json(
            post_processed_dir / relative_path,
            {
                "metric": metric,
                "points": [
                    {"second": 0, "accumulated_value": base_value},
                    {"second": 1, "accumulated_value": base_value + 1.0},
                    {"second": 2, "accumulated_value": base_value + 2.0},
                ],
            },
        )
    _write_json(
        post_processed_dir / "gateway" / "llm-requests" / "llm-request-stats.json",
        {
            "request_count": 5,
            "metric_count": 2,
            "metrics": {
                "prompt_tokens": {"count": 5, "avg": 120.0, "min": 80.0, "max": 200.0},
                "completion_tokens": {"count": 5, "avg": 60.0, "min": 20.0, "max": 100.0},
            },
            "average_stage_speed_tokens_per_s": {
                "request_status_code": 200,
                "request_count_200": 4,
                "prefill": {
                    "eligible_request_count": 3,
                    "excluded_request_count": 1,
                    "avg_tokens_per_s": 1000.0,
                    "min_tokens_per_s": 800.0,
                    "max_tokens_per_s": 1200.0,
                },
                "decode": {
                    "eligible_request_count": 4,
                    "excluded_request_count": 0,
                    "avg_tokens_per_s": 50.0,
                    "min_tokens_per_s": 40.0,
                    "max_tokens_per_s": 60.0,
                },
            },
        },
    )


def test_build_key_stats_payload_summarizes_requested_metrics(tmp_path: Path) -> None:
    post_processed_dir = tmp_path / "job" / "post-processed"
    _write_minimal_inputs(post_processed_dir)

    payload = extract_run.build_key_stats_payload(post_processed_dir)

    assert payload["source_run_dir"] == str((tmp_path / "job").resolve())
    assert payload["source_post_processed_dir"] == str(post_processed_dir.resolve())
    assert payload["service_failure_detected"] is True
    assert payload["service_failure_cutoff_time_utc"] == "2026-03-29T00:00:00Z"

    vllm_summary = payload["vllm_metrics"]["kv_cache_usage_perc"]
    assert vllm_summary["metric_name"] == "vllm:kv_cache_usage_perc"
    assert vllm_summary["series_count"] == 2
    assert vllm_summary["sample_count"] == 4
    assert vllm_summary["avg"] == pytest.approx(0.4)
    assert vllm_summary["min"] == pytest.approx(0.1)
    assert vllm_summary["max"] == pytest.approx(0.7)
    assert vllm_summary["std"] == pytest.approx(math.sqrt(0.05))

    job_concurrency = payload["job_concurrency"]
    assert job_concurrency["sample_count"] == 3
    assert job_concurrency["avg"] == pytest.approx(4.0 / 3.0)
    assert job_concurrency["min"] == pytest.approx(1.0)
    assert job_concurrency["max"] == pytest.approx(2.0)

    stack_kv = payload["gateway"]["stack_kv"]["kv_usage_tokens"]
    assert stack_kv["sample_count"] == 3
    assert stack_kv["avg"] == pytest.approx(20.0)
    assert stack_kv["std"] == pytest.approx(math.sqrt(200.0 / 3.0))

    stack_prompt = payload["gateway"]["stack"]["prompt_tokens"]
    assert stack_prompt["avg"] == pytest.approx(2.0)
    assert stack_prompt["min"] == pytest.approx(1.0)
    assert stack_prompt["max"] == pytest.approx(3.0)

    stage_speed = payload["gateway"]["llm_requests"]["average_stage_speed_tokens_per_s"]
    assert stage_speed["request_status_code"] == 200
    assert stage_speed["prefill"] == {
        "eligible_request_count": 3,
        "excluded_request_count": 1,
        "avg": 1000.0,
        "min": 800.0,
        "max": 1200.0,
    }
    assert stage_speed["decode"]["avg"] == pytest.approx(50.0)

    request_metrics = payload["gateway"]["llm_requests"]["metrics"]
    assert request_metrics["completion_tokens"] == {
        "count": 5,
        "avg": 60.0,
        "min": 20.0,
        "max": 100.0,
    }


def test_main_post_processed_dir_writes_default_output(tmp_path: Path) -> None:
    post_processed_dir = tmp_path / "job" / "post-processed-50"
    _write_minimal_inputs(post_processed_dir)

    exit_code = extract_run.main(["--post-processed-dir", str(post_processed_dir)])

    assert exit_code == 0
    output_path = post_processed_dir / "key-stats" / extract_run.DEFAULT_OUTPUT_NAME
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_post_processed_dir"] == str(post_processed_dir.resolve())
    assert payload["gateway"]["llm_requests"]["metric_count"] == 2


def test_discover_run_dirs_with_post_processed_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"

    for run_dir in (run_a, run_b):
        _write_json(
            run_dir / "post-processed" / extract_run.DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH,
            {},
        )

    discovered = extract_run.discover_run_dirs_with_post_processed(root_dir)
    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    for run_dir in (run_a, run_b):
        _write_json(
            run_dir / "post-processed" / extract_run.DEFAULT_REQUIRED_DISCOVERY_RELATIVE_PATH,
            {},
        )

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
    ) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "post-processed" / "key-stats" / extract_run.DEFAULT_OUTPUT_NAME

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]


def test_extract_run_rejects_invalid_option_combinations(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)
    post_processed_dir = run_dir / "post-processed"
    post_processed_dir.mkdir(parents=True)
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")

    try:
        extract_run.main(["--post-processed-dir", str(post_processed_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when --dry-run is used with --post-processed-dir"
        )

    try:
        extract_run.main(["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")])
    except ValueError as exc:
        assert "--output can only be used with --run-dir or --post-processed-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")
