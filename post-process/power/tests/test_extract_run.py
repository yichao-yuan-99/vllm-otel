from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_power_log(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_extract_power_summary_truncates_points_before_run_start(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    replay_dir = run_dir / "replay"
    power_dir = run_dir / "power"
    replay_dir.mkdir(parents=True)
    power_dir.mkdir(parents=True)

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:10.000Z",
                "finished_at": "2026-03-08T00:00:14.000Z",
                "worker_results": {},
            }
        ),
        encoding="utf-8",
    )
    _write_power_log(
        power_dir / "power-log.jsonl",
        [
            {
                "timestamp": "2026-03-08T00:00:09.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 50.0}}},
            },
            {
                "timestamp": "2026-03-08T00:00:10.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 200.0}}},
            },
            {
                "timestamp": "2026-03-08T00:00:11.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 300.0}}},
            },
            {
                "timestamp": "2026-03-08T00:00:13.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 100.0}}},
            },
        ],
    )

    result = extract_run.extract_power_summary_from_run_dir(run_dir)

    assert result["power_log_found"] is True
    assert result["power_sample_count"] == 3
    assert result["power_stats_w"] == {"avg": 200.0, "min": 100.0, "max": 300.0}
    assert result["total_energy_j"] == 650.0
    assert result["power_points"] == [
        {"time_offset_s": 0.0, "power_w": 200.0},
        {"time_offset_s": 1.0, "power_w": 300.0},
        {"time_offset_s": 3.0, "power_w": 100.0},
    ]


def test_extract_power_summary_reports_missing_power_log_without_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:05.000Z",
                "worker_results": {},
            }
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_power_summary_from_run_dir(run_dir)

    assert result["power_log_found"] is False
    assert result["power_sample_count"] == 0
    assert result["power_stats_w"] == {"avg": None, "min": None, "max": None}
    assert result["total_energy_j"] == 0.0
    assert result["total_energy_kwh"] == 0.0
    assert result["power_points"] == []


def test_extract_power_summary_applies_service_failure_cutoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    replay_dir = run_dir / "replay"
    power_dir = run_dir / "power"
    sbatch_logs_dir = run_dir / "sbatch-logs"
    replay_dir.mkdir(parents=True)
    power_dir.mkdir(parents=True)
    sbatch_logs_dir.mkdir(parents=True)

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:10.000Z",
                "worker_results": {},
            }
        ),
        encoding="utf-8",
    )
    _write_power_log(
        power_dir / "power-log.jsonl",
        [
            {
                "timestamp": "2026-03-08T00:00:01.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 100.0}}},
            },
            {
                "timestamp": "2026-03-08T00:00:02.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 200.0}}},
            },
            {
                "timestamp": "2026-03-08T00:00:04.000Z",
                "payload": {"/var/run/zeusd.sock": {"gpu_power_w": {"0": 400.0}}},
            },
        ],
    )
    (sbatch_logs_dir / "vllm.1.log").write_text(
        "2026-03-08T00:00:03Z AsyncLLM output_handler failed.\n",
        encoding="utf-8",
    )

    result = extract_run.extract_power_summary_from_run_dir(run_dir)

    assert result["service_failure_detected"] is True
    assert result["service_failure_cutoff_time_utc"] == "2026-03-08T00:00:03Z"
    assert result["power_sample_count"] == 2
    assert result["power_stats_w"] == {"avg": 150.0, "min": 100.0, "max": 200.0}
    assert result["total_energy_j"] == 150.0


def test_discover_run_dirs_with_power_sources_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    replay_run = root_dir / "replay-job"
    con_driver_run = root_dir / "con-driver-job"
    unrelated = root_dir / "orphan-power"
    (replay_run / "replay").mkdir(parents=True)
    (replay_run / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (con_driver_run / "meta").mkdir(parents=True)
    (con_driver_run / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (con_driver_run / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")
    (unrelated / "power").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_power_sources(root_dir)

    assert discovered == [con_driver_run.resolve(), replay_run.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
        return run_dir / "post-processed" / "power" / "power-summary.json"

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
