from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def test_extract_job_concurrency_from_replay_run(tmp_path: Path) -> None:
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
                        "started_at": "2026-03-08T00:00:00.000Z",
                        "finished_at": "2026-03-08T00:00:03.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:01.000Z",
                        "finished_at": "2026-03-08T00:00:04.000Z",
                    },
                    "trial-0003": {
                        "worker_id": "trial-0003",
                        "status": "time_bound_finished",
                        "started_at": "2026-03-08T00:00:02.000Z",
                        "finished_at": "2026-03-08T00:00:02.400Z",
                    },
                    "trial-0004": {
                        "worker_id": "trial-0004",
                        "status": "timed_out",
                        "started_at": "2026-03-08T00:00:04.000Z",
                        "finished_at": None,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_job_concurrency_from_run_dir(run_dir)

    assert result["source_type"] == "replay"
    assert result["replay_count"] == 4
    assert result["jobs_with_valid_range_count"] == 3
    assert result["total_duration_s"] == 6.0
    assert result["sample_count"] == 6
    assert result["max_concurrency"] == 3
    assert result["concurrency_points"] == [
        {"second": 0, "concurrency": 1},
        {"second": 1, "concurrency": 2},
        {"second": 2, "concurrency": 3},
        {"second": 3, "concurrency": 1},
        {"second": 4, "concurrency": 0},
        {"second": 5, "concurrency": 0},
    ]


def test_extract_job_concurrency_from_con_driver_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-con-driver"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    (meta_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00+00:00",
                "finished_at": "2026-03-08T00:00:05+00:00",
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "trial_id": "trial-1",
                    "status": "succeeded",
                    "started_at": "2026-03-08T00:00:00.200+00:00",
                    "finished_at": "2026-03-08T00:00:02.200+00:00",
                },
                {
                    "trial_id": "trial-2",
                    "status": "succeeded",
                    "started_at": "2026-03-08T00:00:01.700+00:00",
                    "finished_at": "2026-03-08T00:00:04.000+00:00",
                },
                {
                    "trial_id": "trial-3",
                    "status": "failed",
                    "started_at": "2026-03-08T00:00:03.000+00:00",
                    "finished_at": "2026-03-08T00:00:03.000+00:00",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_job_concurrency_from_run_dir(run_dir)

    assert result["source_type"] == "con-driver"
    assert result["replay_count"] == 3
    assert result["jobs_with_valid_range_count"] == 2
    assert result["total_duration_s"] == 5.0
    assert result["sample_count"] == 5
    assert result["concurrency_points"] == [
        {"second": 0, "concurrency": 1},
        {"second": 1, "concurrency": 2},
        {"second": 2, "concurrency": 2},
        {"second": 3, "concurrency": 1},
        {"second": 4, "concurrency": 0},
    ]


def test_extract_job_concurrency_applies_service_failure_cutoff(tmp_path: Path) -> None:
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
                        "started_at": "2026-03-08T00:00:00.000Z",
                        "finished_at": "2026-03-08T00:00:08.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:02.000Z",
                        "finished_at": "2026-03-08T00:00:09.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (sbatch_logs_dir / "vllm.1.log").write_text(
        "2026-03-08T00:00:04Z AsyncLLM output_handler failed.\n",
        encoding="utf-8",
    )

    result = extract_run.extract_job_concurrency_from_run_dir(run_dir)

    assert result["service_failure_detected"] is True
    assert result["service_failure_cutoff_time_utc"] == "2026-03-08T00:00:04Z"
    assert result["time_constraint_s"] == 4.0
    assert result["total_duration_s"] == 4.0
    assert result["sample_count"] == 4
    assert result["concurrency_points"] == [
        {"second": 0, "concurrency": 1},
        {"second": 1, "concurrency": 1},
        {"second": 2, "concurrency": 2},
        {"second": 3, "concurrency": 2},
    ]


def test_discover_run_dirs_with_job_concurrency_sources_scans_recursively(
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

    discovered = extract_run.discover_run_dirs_with_job_concurrency_sources(root_dir)

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

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "post-processed" / "job-concurrency" / "job-concurrency-timeseries.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
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
