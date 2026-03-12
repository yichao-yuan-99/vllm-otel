from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def test_extract_global_progress_from_replay_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)

    worker_results: dict[str, dict[str, object]] = {}
    for index in range(1, 121):
        worker_results[f"trial-{index:04d}"] = {
            "worker_id": f"trial-{index:04d}",
            "status": "completed",
            "started_at": "2026-03-08T00:00:00.000Z",
            "finished_at": f"2026-03-08T00:02:{index:02d}.000Z" if index <= 59 else None,
        }
    # Keep deterministic finish offsets: 1..120 seconds.
    for index in range(1, 121):
        minute = (index // 60)
        second = (index % 60)
        worker_results[f"trial-{index:04d}"]["finished_at"] = (
            f"2026-03-08T00:{minute:02d}:{second:02d}.000Z"
        )

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:03:00.000Z",
                "worker_results": worker_results,
            }
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_global_progress_from_run_dir(run_dir)

    assert result["source_type"] == "replay"
    assert result["replay_count"] == 120
    assert result["finished_replay_count"] == 120
    assert result["milestones"] == [
        {"replay_count": 50, "finish_time_s": 50.0},
        {"replay_count": 100, "finish_time_s": 100.0},
        {"replay_count": 120, "finish_time_s": 120.0},
    ]


def test_extract_global_progress_from_con_driver_run_with_missing_finished_at(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job-con-driver"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)

    (meta_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00+00:00",
                "finished_at": "2026-03-08T00:00:12+00:00",
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {"trial_id": "1", "finished_at": "2026-03-08T00:00:01+00:00"},
                {"trial_id": "2", "finished_at": "2026-03-08T00:00:04+00:00"},
                {"trial_id": "3", "finished_at": "2026-03-08T00:00:06+00:00"},
                {"trial_id": "4", "finished_at": "2026-03-08T00:00:08+00:00"},
                {"trial_id": "5", "finished_at": None},
            ]
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_global_progress_from_run_dir(run_dir, milestone_step=2)

    assert result["source_type"] == "con-driver"
    assert result["replay_count"] == 5
    assert result["finished_replay_count"] == 4
    assert result["milestones"] == [
        {"replay_count": 2, "finish_time_s": 4.0},
        {"replay_count": 4, "finish_time_s": 8.0},
        {"replay_count": 5, "finish_time_s": None},
    ]


def test_discover_run_dirs_with_global_sources_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    replay_run = root_dir / "replay-job"
    con_driver_run = root_dir / "con-driver-job"
    (replay_run / "replay").mkdir(parents=True)
    (replay_run / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (con_driver_run / "meta").mkdir(parents=True)
    (con_driver_run / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (con_driver_run / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")

    discovered = extract_run.discover_run_dirs_with_global_sources(root_dir)

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

    processed: list[tuple[Path, Path | None, int]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        milestone_step: int = 50,
    ) -> Path:
        processed.append((run_dir, output_path, milestone_step))
        return run_dir / "post-processed" / "global-progress" / "replay-progress-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(
        ["--root-dir", str(root_dir), "--max-procs", "1", "--milestone-step", "10"]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, 10),
        (run_b.resolve(), None, 10),
    ]


def test_extract_run_rejects_non_positive_milestone_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--milestone-step", "0"])
    except ValueError as exc:
        assert "--milestone-step must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when milestone step is non-positive")


def test_extract_run_rejects_file_path_for_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "trial-timing-summary.csv"
    csv_path.write_text("run_path,total_duration_s\n", encoding="utf-8")

    try:
        extract_run.main(["--run-dir", str(csv_path)])
    except ValueError as exc:
        assert "--run-dir must point to a directory" in str(exc)
        assert str(csv_path.parent) in str(exc)
    else:
        raise AssertionError("Expected ValueError when --run-dir points to a file")
