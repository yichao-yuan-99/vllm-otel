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


def _write_replay_summary(
    run_dir: Path,
    *,
    replay_count: int,
) -> None:
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    worker_results: dict[str, dict[str, object]] = {}
    for index in range(1, replay_count + 1):
        minute = index // 60
        second = index % 60
        worker_results[f"trial-{index:04d}"] = {
            "worker_id": f"trial-{index:04d}",
            "status": "completed",
            "started_at": "2026-03-08T00:00:00.000Z",
            "finished_at": f"2026-03-08T00:{minute:02d}:{second:02d}.000Z",
        }
    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T01:00:00.000Z",
                "worker_results": worker_results,
            }
        ),
        encoding="utf-8",
    )


def _write_progress_summary(
    run_dir: Path,
    *,
    milestone_step: int,
    milestones: list[tuple[int, float | None]],
) -> None:
    output_dir = run_dir / "post-processed" / "global-progress"
    output_dir.mkdir(parents=True)
    (output_dir / "replay-progress-summary.json").write_text(
        json.dumps(
            {
                "source_type": "replay",
                "replay_count": milestones[-1][0] if milestones else 0,
                "finished_replay_count": len([value for _count, value in milestones if value is not None]),
                "milestone_step": milestone_step,
                "milestones": [
                    {"replay_count": count, "finish_time_s": value}
                    for count, value in milestones
                ],
            }
        ),
        encoding="utf-8",
    )


def test_main_extracts_and_writes_csv(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a-run"
    run_b = root_dir / "b-run"
    _write_replay_summary(run_a, replay_count=120)
    _write_replay_summary(run_b, replay_count=75)

    exit_code = aggregate_runs_csv.main(
        ["--root-dir", str(root_dir), "--max-procs", "1", "--milestone-step", "50"]
    )

    assert exit_code == 0
    output_path = root_dir / "replay-progress-summary.csv"
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["run_path"] for row in rows] == ["a-run", "b-run"]
    assert rows[0]["milestone_step"] == "50"
    assert rows[0]["finish_time_s_at_50"] == "50.0"
    assert rows[0]["finish_time_s_at_100"] == "100.0"
    assert rows[0]["finish_time_s_at_120"] == "120.0"
    assert rows[1]["finish_time_s_at_50"] == "50.0"
    assert rows[1]["finish_time_s_at_75"] == "75.0"
    assert rows[1]["finish_time_s_at_100"] == ""
    assert (run_a / "post-processed" / "global-progress" / "replay-progress-summary.json").is_file()
    assert (run_b / "post-processed" / "global-progress" / "replay-progress-summary.json").is_file()


def test_build_rows_uses_existing_progress_summaries(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "x" / "10"
    run_b = root_dir / "x" / "2"
    _write_progress_summary(
        run_a,
        milestone_step=50,
        milestones=[(50, 10.0), (100, 20.0)],
    )
    _write_progress_summary(
        run_b,
        milestone_step=50,
        milestones=[(50, 5.0), (70, 9.0)],
    )

    rows, milestone_columns = aggregate_runs_csv.build_rows(root_dir)

    assert [row["run_path"] for row in rows] == ["x/2", "x/10"]
    assert milestone_columns == [
        "finish_time_s_at_50",
        "finish_time_s_at_70",
        "finish_time_s_at_100",
    ]


def test_main_dry_run_lists_discovered_runs(tmp_path: Path, capsys) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    _write_replay_summary(run_a, replay_count=10)

    exit_code = aggregate_runs_csv.main(["--root-dir", str(root_dir), "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(run_a.resolve()) in captured.out
    assert not (root_dir / "replay-progress-summary.csv").exists()
