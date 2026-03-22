from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_power_summary(run_dir: Path, *, power_points: list[dict[str, float]]) -> Path:
    power_summary_path = run_dir / "post-processed" / "power" / "power-summary.json"
    _write_json(
        power_summary_path,
        {
            "source_run_dir": str(run_dir),
            "source_type": "replay",
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "power_log_found": bool(power_points),
            "power_points": power_points,
        },
    )
    return power_summary_path


def _write_prefill_timeseries(
    run_dir: Path,
    *,
    concurrency_points: list[dict[str, int | float]],
    tick_ms: int,
) -> Path:
    prefill_timeseries_path = (
        run_dir
        / "post-processed"
        / "prefill-concurrency"
        / "prefill-concurrency-timeseries.json"
    )
    _write_json(
        prefill_timeseries_path,
        {
            "source_run_dir": str(run_dir),
            "request_count": 5,
            "prefill_activity_count": 4,
            "total_duration_s": 2.0,
            "tick_ms": tick_ms,
            "tick_s": round(tick_ms / 1000.0, 6),
            "concurrency_points": concurrency_points,
        },
    )
    return prefill_timeseries_path


def test_extract_power_sampling_summary_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_power_summary(
        run_dir,
        power_points=[
            {"time_offset_s": 0.0, "power_w": 100.0},
            {"time_offset_s": 1.0, "power_w": 200.0},
            {"time_offset_s": 2.0, "power_w": 300.0},
        ],
    )
    _write_prefill_timeseries(
        run_dir,
        tick_ms=500,
        concurrency_points=[
            {"tick_index": 0, "time_offset_s": 0.0, "concurrency": 0},
            {"tick_index": 1, "time_offset_s": 0.5, "concurrency": 1},
            {"tick_index": 2, "time_offset_s": 1.0, "concurrency": 1},
            {"tick_index": 3, "time_offset_s": 1.5, "concurrency": 2},
            {"tick_index": 4, "time_offset_s": 2.0, "concurrency": 0},
        ],
    )

    result = extract_run.extract_power_sampling_summary_from_run_dir(run_dir)

    assert result["power_point_count"] == 3
    assert result["prefill_tick_count"] == 5
    assert result["sampled_tick_count"] == 5
    assert result["all_power_stats_w"] == {
        "sample_count": 5,
        "avg_power_w": 200.0,
        "min_power_w": 100.0,
        "max_power_w": 300.0,
        "std_power_w": 70.710678,
    }
    assert result["non_zero_power_stats_w"] == {
        "sample_count": 3,
        "avg_power_w": 200.0,
        "min_power_w": 150.0,
        "max_power_w": 250.0,
        "std_power_w": 40.824829,
    }
    assert result["concurrency_power_stats_w"] == {
        "0": {
            "concurrency": 0,
            "sample_count": 2,
            "avg_power_w": 200.0,
            "min_power_w": 100.0,
            "max_power_w": 300.0,
            "std_power_w": 100.0,
        },
        "1": {
            "concurrency": 1,
            "sample_count": 2,
            "avg_power_w": 175.0,
            "min_power_w": 150.0,
            "max_power_w": 200.0,
            "std_power_w": 25.0,
        },
        "2": {
            "concurrency": 2,
            "sample_count": 1,
            "avg_power_w": 250.0,
            "min_power_w": 250.0,
            "max_power_w": 250.0,
            "std_power_w": 0.0,
        },
    }


def test_extract_power_sampling_summary_handles_missing_power_points(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_power_summary(run_dir, power_points=[])
    _write_prefill_timeseries(
        run_dir,
        tick_ms=10,
        concurrency_points=[
            {"tick_index": 0, "time_offset_s": 0.0, "concurrency": 0},
            {"tick_index": 1, "time_offset_s": 0.01, "concurrency": 2},
            {"tick_index": 2, "time_offset_s": 0.02, "concurrency": 0},
        ],
    )

    result = extract_run.extract_power_sampling_summary_from_run_dir(run_dir)

    assert result["power_log_found"] is False
    assert result["power_point_count"] == 0
    assert result["prefill_tick_count"] == 3
    assert result["sampled_tick_count"] == 0
    assert result["all_power_stats_w"] == {
        "sample_count": 0,
        "avg_power_w": None,
        "min_power_w": None,
        "max_power_w": None,
        "std_power_w": None,
    }
    assert result["non_zero_power_stats_w"] == {
        "sample_count": 0,
        "avg_power_w": None,
        "min_power_w": None,
        "max_power_w": None,
        "std_power_w": None,
    }
    assert result["concurrency_power_stats_w"] == {}


def test_discover_run_dirs_with_power_sampling_inputs_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_good = root_dir / "a" / "job-good"
    run_missing_power = root_dir / "b" / "job-missing-power"
    run_missing_prefill = root_dir / "c" / "job-missing-prefill"

    _write_power_summary(run_good, power_points=[])
    _write_prefill_timeseries(run_good, tick_ms=10, concurrency_points=[])

    _write_prefill_timeseries(run_missing_power, tick_ms=10, concurrency_points=[])
    _write_power_summary(run_missing_prefill, power_points=[])

    discovered = extract_run.discover_run_dirs_with_power_sampling_inputs(root_dir)

    assert discovered == [run_good.resolve()]


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_power_summary(run_a, power_points=[])
    _write_prefill_timeseries(run_a, tick_ms=10, concurrency_points=[])
    _write_power_summary(run_b, power_points=[])
    _write_prefill_timeseries(run_b, tick_ms=10, concurrency_points=[])

    processed: list[tuple[Path, Path | None, Path | None, Path | None]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        power_summary_path: Path | None = None,
        prefill_timeseries_path: Path | None = None,
    ) -> Path:
        processed.append((run_dir, output_path, power_summary_path, prefill_timeseries_path))
        return run_dir / "post-processed" / "power-sampling" / "power-sampling-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, None, None),
        (run_b.resolve(), None, None, None),
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


def test_extract_run_rejects_output_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")])
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")


def test_extract_run_rejects_power_summary_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(
            [
                "--root-dir",
                str(root_dir),
                "--power-summary",
                str(tmp_path / "power-summary.json"),
            ]
        )
    except ValueError as exc:
        assert "--power-summary can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when --power-summary is used with --root-dir"
        )


def test_extract_run_rejects_prefill_timeseries_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(
            [
                "--root-dir",
                str(root_dir),
                "--prefill-timeseries",
                str(tmp_path / "prefill-concurrency-timeseries.json"),
            ]
        )
    except ValueError as exc:
        assert "--prefill-timeseries can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when --prefill-timeseries is used with --root-dir"
        )


def test_extract_run_rejects_non_positive_max_procs(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--root-dir", str(root_dir), "--max-procs", "0"])
    except ValueError as exc:
        assert "--max-procs must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --max-procs <= 0")
