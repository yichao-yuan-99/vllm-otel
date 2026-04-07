from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_llm_requests(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_extract_request_throughput_from_llm_requests_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    llm_requests_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / "llm-requests.json"
    )
    _write_llm_requests(
        llm_requests_path,
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "request_count": 5,
            "requests": [
                {
                    "request_id": "req-1",
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 1.0,
                    "status_code": 200,
                },
                {
                    "request_id": "req-2",
                    "request_start_offset_s": 1.5,
                    "request_end_offset_s": 3.0,
                    "status_code": 200,
                },
                {
                    "request_id": "req-3",
                    "request_start_offset_s": 2.5,
                    "request_end_offset_s": 4.0,
                    "status_code": 499,
                },
                {
                    "request_id": "req-4",
                    "request_start_offset_s": 4.5,
                    "request_end_offset_s": 5.0,
                    "status_code": 200,
                },
                {
                    "request_id": "req-5",
                    "request_start_offset_s": 5.5,
                    "request_end_offset_s": None,
                    "status_code": 200,
                },
            ],
        },
    )

    result = extract_run.extract_request_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=1.0,
    )

    assert result["request_count"] == 5
    assert result["finished_request_count"] == 4
    assert result["finished_request_count_status_200"] == 3
    assert result["non_200_finished_request_count"] == 1
    assert result["first_request_start_s"] == 0.0
    assert result["last_request_end_s"] == 5.0
    assert result["total_duration_s"] == 6.0
    assert result["sample_count"] == 6
    assert result["throughput_points"] == [
        {"time_s": 0.0, "throughput_requests_per_s": 1.0},
        {"time_s": 1.0, "throughput_requests_per_s": 0.5},
        {"time_s": 2.0, "throughput_requests_per_s": 1.0},
        {"time_s": 3.0, "throughput_requests_per_s": 1.0},
        {"time_s": 4.0, "throughput_requests_per_s": 1.5},
        {"time_s": 5.0, "throughput_requests_per_s": 1.0},
    ]
    assert result["throughput_points_status_200"] == [
        {"time_s": 0.0, "throughput_requests_per_s": 1.0},
        {"time_s": 1.0, "throughput_requests_per_s": 0.5},
        {"time_s": 2.0, "throughput_requests_per_s": 1.0},
        {"time_s": 3.0, "throughput_requests_per_s": 0.5},
        {"time_s": 4.0, "throughput_requests_per_s": 1.0},
        {"time_s": 5.0, "throughput_requests_per_s": 0.5},
    ]


def test_extract_request_throughput_uses_request_starts_when_no_requests_finish(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    llm_requests_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / "llm-requests.json"
    )
    _write_llm_requests(
        llm_requests_path,
        {
            "service_failure_detected": True,
            "service_failure_cutoff_time_utc": "2026-03-08T00:00:03Z",
            "request_count": 2,
            "requests": [
                {
                    "request_id": "req-1",
                    "request_start_offset_s": 0.2,
                    "request_end_offset_s": None,
                    "status_code": 499,
                },
                {
                    "request_id": "req-2",
                    "request_start_offset_s": 2.3,
                    "request_end_offset_s": None,
                    "status_code": 500,
                },
            ],
        },
    )

    result = extract_run.extract_request_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=2.0,
        window_size_s=0.5,
    )

    assert result["service_failure_detected"] is True
    assert result["service_failure_cutoff_time_utc"] == "2026-03-08T00:00:03Z"
    assert result["finished_request_count"] == 0
    assert result["total_duration_s"] == 2.5
    assert result["sample_count"] == 5
    assert all(point["throughput_requests_per_s"] == 0.0 for point in result["throughput_points"])


def test_extract_request_throughput_adds_profile_series(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    llm_requests_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / "llm-requests.json"
    )
    _write_llm_requests(
        llm_requests_path,
        {
            "port_profile_ids": [2, 13],
            "requests": [
                {
                    "request_id": "req-1",
                    "gateway_profile_id": 2,
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 1.0,
                    "status_code": 200,
                },
                {
                    "request_id": "req-2",
                    "gateway_profile_id": 13,
                    "request_start_offset_s": 1.0,
                    "request_end_offset_s": 2.0,
                    "status_code": 499,
                },
            ],
        },
    )

    result = extract_run.extract_request_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=1.0,
    )

    assert result["multi_profile"] is True
    assert result["port_profile_ids"] == [2, 13]
    assert result["series_keys"] == ["profile-2", "profile-13"]
    assert result["series_by_profile"]["profile-2"]["request_count"] == 1
    assert result["series_by_profile"]["profile-2"]["finished_request_count"] == 1
    assert result["series_by_profile"]["profile-2"]["finished_request_count_status_200"] == 1
    assert result["series_by_profile"]["profile-13"]["request_count"] == 1
    assert result["series_by_profile"]["profile-13"]["finished_request_count"] == 1
    assert result["series_by_profile"]["profile-13"]["finished_request_count_status_200"] == 0


def test_discover_run_dirs_with_llm_requests_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    _write_llm_requests(
        run_a / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
        {"requests": []},
    )
    _write_llm_requests(
        run_b / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
        {"requests": []},
    )

    discovered = extract_run.discover_run_dirs_with_llm_requests(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_llm_requests(
        run_a / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
        {"requests": []},
    )
    _write_llm_requests(
        run_b / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
        {"requests": []},
    )

    processed: list[tuple[Path, Path | None, Path | None, float, float]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        llm_requests_path: Path | None = None,
        timepoint_freq_hz: float = 1.0,
        window_size_s: float = 600.0,
    ) -> Path:
        processed.append(
            (run_dir, output_path, llm_requests_path, timepoint_freq_hz, window_size_s)
        )
        return (
            run_dir
            / "post-processed"
            / "request-throughput"
            / extract_run.DEFAULT_OUTPUT_NAME
        )

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(
        [
            "--root-dir",
            str(root_dir),
            "--max-procs",
            "1",
            "--timepoint-freq-hz",
            "2",
            "--window-size-s",
            "30",
        ]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, None, 2.0, 30.0),
        (run_b.resolve(), None, None, 2.0, 30.0),
    ]


def test_extract_run_rejects_invalid_option_combinations(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")

    try:
        extract_run.main(["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")])
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")

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

    try:
        extract_run.main(["--run-dir", str(run_dir), "--timepoint-freq-hz", "0"])
    except ValueError as exc:
        assert "--timepoint-freq-hz must be a positive number" in str(exc)
    else:
        raise AssertionError("Expected ValueError when timepoint frequency is non-positive")

    try:
        extract_run.main(["--run-dir", str(run_dir), "--window-size-s", "0"])
    except ValueError as exc:
        assert "--window-size-s must be a positive number" in str(exc)
    else:
        raise AssertionError("Expected ValueError when window size is non-positive")

    try:
        extract_run.main(["--root-dir", str(root_dir), "--max-procs", "0"])
    except ValueError as exc:
        assert "--max-procs must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --max-procs <= 0")
