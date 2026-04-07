from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def test_extract_job_throughput_from_replay_run_counts_finished_jobs(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:06.000Z",
                "worker_results": {
                    "trial-0001": {
                        "worker_id": "trial-0001",
                        "status": "completed",
                        "finished_at": "2026-03-08T00:00:01.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "time_bound_finished",
                        "finished_at": "2026-03-08T00:00:03.000Z",
                    },
                    "trial-0003": {
                        "worker_id": "trial-0003",
                        "status": "cancelled",
                        "finished_at": "2026-03-08T00:00:04.000Z",
                    },
                    "trial-0004": {
                        "worker_id": "trial-0004",
                        "status": "completed",
                        "finished_at": "2026-03-08T00:00:05.000Z",
                    },
                    "trial-0005": {
                        "worker_id": "trial-0005",
                        "status": "timed_out",
                        "finished_at": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_job_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=1.0,
    )

    assert result["source_type"] == "replay"
    assert result["replay_count"] == 5
    assert result["finished_replay_count"] == 4
    assert result["finished_replay_count_excluding_cancelled"] == 3
    assert result["cancelled_finished_replay_count"] == 1
    assert result["total_duration_s"] == 6.0
    assert result["sample_count"] == 6
    assert result["throughput_points"] == [
        {"time_s": 0.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 1.0, "throughput_jobs_per_s": 0.5},
        {"time_s": 2.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 3.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 4.0, "throughput_jobs_per_s": 1.5},
        {"time_s": 5.0, "throughput_jobs_per_s": 1.0},
    ]
    assert result["throughput_points_excluding_cancelled"] == [
        {"time_s": 0.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 1.0, "throughput_jobs_per_s": 0.5},
        {"time_s": 2.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 3.0, "throughput_jobs_per_s": 0.5},
        {"time_s": 4.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 5.0, "throughput_jobs_per_s": 0.5},
    ]


def test_extract_job_throughput_from_con_driver_run_with_missing_finished_at(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job-con-driver"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    (meta_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00+00:00",
                "finished_at": "2026-03-08T00:00:04+00:00",
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {"trial_id": "1", "finished_at": "2026-03-08T00:00:01+00:00"},
                {"trial_id": "2", "finished_at": "2026-03-08T00:00:03+00:00"},
                {"trial_id": "3", "finished_at": None},
            ]
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_job_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=2.0,
    )

    assert result["source_type"] == "con-driver"
    assert result["replay_count"] == 3
    assert result["finished_replay_count"] == 2
    assert result["finished_replay_count_excluding_cancelled"] == 2
    assert result["cancelled_finished_replay_count"] == 0
    assert result["sample_count"] == 4
    assert result["throughput_points"] == [
        {"time_s": 0.0, "throughput_jobs_per_s": 0.5},
        {"time_s": 1.0, "throughput_jobs_per_s": 0.666667},
        {"time_s": 2.0, "throughput_jobs_per_s": 0.5},
        {"time_s": 3.0, "throughput_jobs_per_s": 0.666667},
    ]
    assert result["throughput_points_excluding_cancelled"] == result["throughput_points"]


def test_extract_job_throughput_hard_cuts_off_at_time_constraint(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:10.000Z",
                "time_constraint_s": 4.0,
                "worker_results": {
                    "trial-0001": {
                        "worker_id": "trial-0001",
                        "status": "completed",
                        "finished_at": "2026-03-08T00:00:01.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "completed",
                        "finished_at": "2026-03-08T00:00:03.000Z",
                    },
                    "trial-0003": {
                        "worker_id": "trial-0003",
                        "status": "time_bound_finished",
                        "finished_at": "2026-03-08T00:00:05.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_job_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=1.0,
    )

    assert result["time_constraint_s"] == 4.0
    assert result["finished_replay_count"] == 2
    assert result["finished_replay_count_excluding_cancelled"] == 2
    assert result["total_duration_s"] == 4.0
    assert result["sample_count"] == 4
    assert result["throughput_points"] == [
        {"time_s": 0.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 1.0, "throughput_jobs_per_s": 0.5},
        {"time_s": 2.0, "throughput_jobs_per_s": 1.0},
        {"time_s": 3.0, "throughput_jobs_per_s": 0.5},
    ]
    assert result["throughput_points_excluding_cancelled"] == result["throughput_points"]


def test_extract_job_throughput_applies_service_failure_cutoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    sbatch_logs_dir = run_dir / "sbatch-logs"
    replay_dir.mkdir(parents=True)
    sbatch_logs_dir.mkdir(parents=True)

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:10.000Z",
                "worker_results": {
                    "trial-0001": {
                        "worker_id": "trial-0001",
                        "status": "completed",
                        "finished_at": "2026-03-08T00:00:01.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "completed",
                        "finished_at": "2026-03-08T00:00:05.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (sbatch_logs_dir / "vllm.1.log").write_text(
        "2026-03-08T00:00:03Z AsyncLLM output_handler failed.\n",
        encoding="utf-8",
    )

    result = extract_run.extract_job_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=1.0,
    )

    assert result["service_failure_detected"] is True
    assert result["service_failure_cutoff_time_utc"] == "2026-03-08T00:00:03Z"
    assert result["finished_replay_count"] == 1
    assert result["total_duration_s"] == 3.0
    assert result["sample_count"] == 3


def test_extract_job_throughput_adds_profile_series_from_gateway_manifests(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)

    token_a = "token-a"
    token_b = "token-b"
    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:04.000Z",
                "port_profile_id_list": [2, 13],
                "worker_results": {
                    "trial-0001": {
                        "worker_id": "trial-0001",
                        "status": "completed",
                        "api_token": token_a,
                        "finished_at": "2026-03-08T00:00:01.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "completed",
                        "api_token": token_b,
                        "finished_at": "2026-03-08T00:00:02.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    gateway_output_dir = run_dir / "gateway-output"
    (gateway_output_dir / "run_a").mkdir(parents=True)
    (gateway_output_dir / "run_b").mkdir(parents=True)
    (gateway_output_dir / "run_a" / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hashlib.sha256(token_a.encode("utf-8")).hexdigest(),
                "backend_port_profile_id": 2,
            }
        ),
        encoding="utf-8",
    )
    (gateway_output_dir / "run_b" / "manifest.json").write_text(
        json.dumps(
            {
                "api_token_hash": hashlib.sha256(token_b.encode("utf-8")).hexdigest(),
                "backend_port_profile_id": 13,
            }
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_job_throughput_from_run_dir(
        run_dir,
        timepoint_freq_hz=1.0,
        window_size_s=1.0,
    )

    assert result["multi_profile"] is True
    assert result["port_profile_ids"] == [2, 13]
    assert result["series_keys"] == ["profile-2", "profile-13"]
    assert result["series_by_profile"]["profile-2"]["replay_count"] == 1
    assert result["series_by_profile"]["profile-2"]["finished_replay_count"] == 1
    assert result["series_by_profile"]["profile-13"]["replay_count"] == 1
    assert result["series_by_profile"]["profile-13"]["finished_replay_count"] == 1


def test_discover_run_dirs_with_job_throughput_sources_scans_recursively(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    replay_run = root_dir / "replay-job"
    con_driver_run = root_dir / "con-driver-job"
    (replay_run / "replay").mkdir(parents=True)
    (replay_run / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (con_driver_run / "meta").mkdir(parents=True)
    (con_driver_run / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (con_driver_run / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")

    discovered = extract_run.discover_run_dirs_with_job_throughput_sources(root_dir)

    assert discovered == [con_driver_run.resolve(), replay_run.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    (run_a / "replay").mkdir(parents=True)
    (run_a / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (run_b / "meta").mkdir(parents=True)
    (run_b / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (run_b / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")

    processed: list[tuple[Path, Path | None, float, float]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        timepoint_freq_hz: float = 1.0,
        window_size_s: float = 600.0,
    ) -> Path:
        processed.append((run_dir, output_path, timepoint_freq_hz, window_size_s))
        return run_dir / "post-processed" / "job-throughput" / "job-throughput-timeseries.json"

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
        (run_a.resolve(), None, 2.0, 30.0),
        (run_b.resolve(), None, 2.0, 30.0),
    ]


def test_extract_run_rejects_non_positive_timepoint_frequency(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--timepoint-freq-hz", "0"])
    except ValueError as exc:
        assert "--timepoint-freq-hz must be a positive number" in str(exc)
    else:
        raise AssertionError("Expected ValueError when timepoint frequency is non-positive")


def test_extract_run_rejects_non_positive_window_size(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--window-size-s", "0"])
    except ValueError as exc:
        assert "--window-size-s must be a positive number" in str(exc)
    else:
        raise AssertionError("Expected ValueError when window size is non-positive")


def test_extract_run_rejects_file_path_for_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)
    output_path = run_dir / "job-throughput-timeseries.json"
    output_path.write_text("{}", encoding="utf-8")

    try:
        extract_run.main(["--run-dir", str(output_path)])
    except ValueError as exc:
        assert "--run-dir must point to a directory" in str(exc)
        assert str(output_path.parent) in str(exc)
    else:
        raise AssertionError("Expected ValueError when --run-dir points to a file")
