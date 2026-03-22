from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_llm_requests(run_dir: Path, payload: dict[str, object]) -> Path:
    llm_requests_dir = run_dir / "post-processed" / "gateway" / "llm-requests"
    llm_requests_dir.mkdir(parents=True)
    llm_requests_path = llm_requests_dir / "llm-requests.json"
    llm_requests_path.write_text(json.dumps(payload), encoding="utf-8")
    return llm_requests_path


def test_extract_prefill_concurrency_from_llm_requests_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_llm_requests(
        run_dir,
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "request_count": 3,
            "requests": [
                {
                    "request_id": "req-1",
                    "trace_id": "trace-1",
                    "gateway_run_id": "run-1",
                    "gateway_profile_id": None,
                    "status_code": 200,
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 0.04,
                    "request_end_to_run_end_s": 0.01,
                    "gen_ai.latency.time_in_queue": 0.01,
                    "gen_ai.latency.time_in_model_prefill": 0.02,
                },
                {
                    "request_id": "req-2",
                    "trace_id": "trace-2",
                    "gateway_run_id": "run-1",
                    "gateway_profile_id": None,
                    "status_code": 200,
                    "request_start_offset_s": 0.015,
                    "request_end_offset_s": 0.03,
                    "request_end_to_run_end_s": 0.02,
                    "gen_ai.latency.time_in_model_prefill": 0.02,
                },
                {
                    "request_id": "req-3",
                    "trace_id": "trace-3",
                    "gateway_run_id": "run-1",
                    "gateway_profile_id": None,
                    "status_code": 200,
                    "request_start_offset_s": 0.02,
                    "request_end_offset_s": 0.04,
                    "request_end_to_run_end_s": 0.01,
                    "gen_ai.latency.time_in_queue": 0.0,
                    "gen_ai.latency.time_in_model_prefill": None,
                },
            ],
        },
    )

    activities_payload, timeseries_payload, stats_payload = (
        extract_run.extract_prefill_concurrency_from_run_dir(run_dir, tick_ms=10)
    )

    assert activities_payload["prefill_activity_count"] == 2
    assert activities_payload["total_duration_s"] == 0.05
    assert activities_payload["activities"] == [
        {
            "gateway_run_id": "run-1",
            "gateway_profile_id": None,
            "trace_id": "trace-1",
            "request_id": "req-1",
            "status_code": 200,
            "request_start_offset_s": 0.0,
            "request_end_offset_s": 0.04,
            "time_in_queue_s": 0.01,
            "prefill_duration_s": 0.02,
            "prefill_start_offset_s": 0.01,
            "prefill_end_offset_s": 0.03,
        },
        {
            "gateway_run_id": "run-1",
            "gateway_profile_id": None,
            "trace_id": "trace-2",
            "request_id": "req-2",
            "status_code": 200,
            "request_start_offset_s": 0.015,
            "request_end_offset_s": 0.03,
            "time_in_queue_s": 0.0,
            "prefill_duration_s": 0.02,
            "prefill_start_offset_s": 0.015,
            "prefill_end_offset_s": 0.035,
        },
    ]

    assert timeseries_payload["sample_count"] == 5
    assert timeseries_payload["concurrency_points"] == [
        {"tick_index": 0, "time_offset_s": 0.0, "concurrency": 0},
        {"tick_index": 1, "time_offset_s": 0.01, "concurrency": 2},
        {"tick_index": 2, "time_offset_s": 0.02, "concurrency": 2},
        {"tick_index": 3, "time_offset_s": 0.03, "concurrency": 1},
        {"tick_index": 4, "time_offset_s": 0.04, "concurrency": 0},
    ]

    assert stats_payload["sample_count"] == 5
    assert stats_payload["min_concurrency"] == 0
    assert stats_payload["max_concurrency"] == 2
    assert stats_payload["avg_concurrency"] == 1.0
    assert stats_payload["concurrency_interval_length_stats"] == {
        "0": {
            "concurrency": 0,
            "interval_count": 2,
            "avg_interval_length_ticks": 1.0,
            "min_interval_length_ticks": 1,
            "max_interval_length_ticks": 1,
            "std_interval_length_ticks": 0.0,
            "avg_interval_length_s": 0.01,
            "min_interval_length_s": 0.01,
            "max_interval_length_s": 0.01,
            "std_interval_length_s": 0.0,
        },
        "1": {
            "concurrency": 1,
            "interval_count": 1,
            "avg_interval_length_ticks": 1.0,
            "min_interval_length_ticks": 1,
            "max_interval_length_ticks": 1,
            "std_interval_length_ticks": 0.0,
            "avg_interval_length_s": 0.01,
            "min_interval_length_s": 0.01,
            "max_interval_length_s": 0.01,
            "std_interval_length_s": 0.0,
        },
        "2": {
            "concurrency": 2,
            "interval_count": 1,
            "avg_interval_length_ticks": 2.0,
            "min_interval_length_ticks": 2,
            "max_interval_length_ticks": 2,
            "std_interval_length_ticks": 0.0,
            "avg_interval_length_s": 0.02,
            "min_interval_length_s": 0.02,
            "max_interval_length_s": 0.02,
            "std_interval_length_s": 0.0,
        },
    }


def test_extract_prefill_concurrency_handles_no_valid_prefill_activities(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_llm_requests(
        run_dir,
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": True,
            "service_failure_cutoff_time_utc": "2026-03-08T00:00:03Z",
            "request_count": 1,
            "requests": [
                {
                    "request_id": "req-1",
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 0.03,
                    "request_end_to_run_end_s": 0.0,
                    "gen_ai.latency.time_in_model_prefill": 0.0,
                }
            ],
        },
    )

    activities_payload, timeseries_payload, stats_payload = (
        extract_run.extract_prefill_concurrency_from_run_dir(run_dir, tick_ms=10)
    )

    assert activities_payload["service_failure_detected"] is True
    assert activities_payload["prefill_activity_count"] == 0
    assert timeseries_payload["sample_count"] == 3
    assert all(point["concurrency"] == 0 for point in timeseries_payload["concurrency_points"])
    assert stats_payload["min_concurrency"] == 0
    assert stats_payload["max_concurrency"] == 0
    assert stats_payload["avg_concurrency"] == 0.0
    assert stats_payload["concurrency_interval_length_stats"] == {
        "0": {
            "concurrency": 0,
            "interval_count": 1,
            "avg_interval_length_ticks": 3.0,
            "min_interval_length_ticks": 3,
            "max_interval_length_ticks": 3,
            "std_interval_length_ticks": 0.0,
            "avg_interval_length_s": 0.03,
            "min_interval_length_s": 0.03,
            "max_interval_length_s": 0.03,
            "std_interval_length_s": 0.0,
        }
    }


def test_build_concurrency_interval_length_stats_computes_std() -> None:
    points = [
        {"concurrency": 0},
        {"concurrency": 0},
        {"concurrency": 1},
        {"concurrency": 1},
        {"concurrency": 1},
        {"concurrency": 0},
        {"concurrency": 2},
        {"concurrency": 2},
        {"concurrency": 0},
        {"concurrency": 0},
        {"concurrency": 0},
    ]

    stats = extract_run._build_concurrency_interval_length_stats(points, tick_s=0.1)

    assert stats["0"] == {
        "concurrency": 0,
        "interval_count": 3,
        "avg_interval_length_ticks": 2.0,
        "min_interval_length_ticks": 1,
        "max_interval_length_ticks": 3,
        "std_interval_length_ticks": 0.816497,
        "avg_interval_length_s": 0.2,
        "min_interval_length_s": 0.1,
        "max_interval_length_s": 0.3,
        "std_interval_length_s": 0.08165,
    }


def test_discover_run_dirs_with_llm_requests_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    _write_llm_requests(run_a, {"requests": []})
    _write_llm_requests(run_b, {"requests": []})

    discovered = extract_run.discover_run_dirs_with_llm_requests(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_llm_requests(run_a, {"requests": []})
    _write_llm_requests(run_b, {"requests": []})

    processed: list[tuple[Path, Path | None, Path | None, int]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        llm_requests_path: Path | None = None,
        tick_ms: int = 10,
    ) -> list[Path]:
        processed.append((run_dir, output_dir, llm_requests_path, tick_ms))
        return [run_dir / "post-processed" / "prefill-concurrency" / "prefill-activities.json"]

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(
        ["--root-dir", str(root_dir), "--max-procs", "1", "--tick-ms", "20"]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, None, 20),
        (run_b.resolve(), None, None, 20),
    ]


def test_extract_run_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_extract_run_rejects_llm_requests_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(
            [
                "--root-dir",
                str(root_dir),
                "--llm-requests",
                str(tmp_path / "llm-requests.json"),
            ]
        )
    except ValueError as exc:
        assert "--llm-requests can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --llm-requests is used with --root-dir")


def test_extract_run_rejects_non_positive_tick_ms(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_llm_requests(run_dir, {"requests": []})

    try:
        extract_run.main(["--run-dir", str(run_dir), "--tick-ms", "0"])
    except ValueError as exc:
        assert "--tick-ms must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --tick-ms <= 0")
