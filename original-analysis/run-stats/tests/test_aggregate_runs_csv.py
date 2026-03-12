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


def _write_run_summary(
    run_dir: Path,
    *,
    agent_type: str,
    dataset: str,
    score: float,
    job_count: int,
    avg_job_max_request_length: float,
    max_job_max_request_length: float,
    avg_turns_per_run: float,
    max_turns_per_run: float,
    avg_run_prompt_tokens_per_request: float,
    avg_run_generation_tokens_per_request: float,
) -> None:
    output_dir = run_dir / "run-stats"
    output_dir.mkdir(parents=True)
    (output_dir / "run-stats-summary.json").write_text(
        json.dumps(
            {
                "agent_type": agent_type,
                "dataset": dataset,
                "score": score,
                "job_count": job_count,
                "avg_job_max_request_length": avg_job_max_request_length,
                "max_job_max_request_length": max_job_max_request_length,
                "avg_turns_per_run": avg_turns_per_run,
                "max_turns_per_run": max_turns_per_run,
                "avg_run_prompt_tokens_per_request": avg_run_prompt_tokens_per_request,
                "avg_run_generation_tokens_per_request": avg_run_generation_tokens_per_request,
            }
        ),
        encoding="utf-8",
    )


def test_discover_run_stats_summary_paths(tmp_path: Path) -> None:
    good = tmp_path / "a" / "run-stats" / "run-stats-summary.json"
    bad = tmp_path / "b" / "other" / "run-stats-summary.json"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_text("{}", encoding="utf-8")
    bad.write_text("{}", encoding="utf-8")

    discovered = aggregate_runs_csv.discover_run_stats_summary_paths(tmp_path)

    assert discovered == [good.resolve()]


def test_build_rows_sorts_by_run_path(tmp_path: Path) -> None:
    run_b = tmp_path / "z" / "10"
    run_a = tmp_path / "z" / "2"
    _write_run_summary(
        run_b,
        agent_type="agent-b",
        dataset="d10",
        score=0.2,
        job_count=10,
        avg_job_max_request_length=100.0,
        max_job_max_request_length=120.0,
        avg_turns_per_run=8.0,
        max_turns_per_run=10.0,
        avg_run_prompt_tokens_per_request=30.0,
        avg_run_generation_tokens_per_request=4.0,
    )
    _write_run_summary(
        run_a,
        agent_type="agent-a",
        dataset="d2",
        score=0.1,
        job_count=2,
        avg_job_max_request_length=20.0,
        max_job_max_request_length=21.0,
        avg_turns_per_run=5.0,
        max_turns_per_run=7.0,
        avg_run_prompt_tokens_per_request=10.0,
        avg_run_generation_tokens_per_request=2.0,
    )

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert [row["run_path"] for row in rows] == ["z/2", "z/10"]
    assert rows[0]["agent_type"] == "agent-a"
    assert rows[0]["dataset"] == "d2"
    assert rows[1]["dataset"] == "d10"


def test_main_writes_default_csv(tmp_path: Path) -> None:
    _write_run_summary(
        tmp_path / "x" / "y",
        agent_type="agent-x",
        dataset="demo",
        score=0.5,
        job_count=3,
        avg_job_max_request_length=42.0,
        max_job_max_request_length=99.0,
        avg_turns_per_run=7.5,
        max_turns_per_run=11.0,
        avg_run_prompt_tokens_per_request=12.0,
        avg_run_generation_tokens_per_request=3.5,
    )

    exit_code = aggregate_runs_csv.main(["--root-dir", str(tmp_path)])

    assert exit_code == 0
    output_path = tmp_path / "run-stats-summary.csv"
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["run_path"] == "x/y"
    assert rows[0]["agent_type"] == "agent-x"
    assert rows[0]["dataset"] == "demo"
    assert rows[0]["score"] == "0.5"
    assert rows[0]["job_count"] == "3"
    assert rows[0]["avg_job_max_request_length"] == "42.0"
    assert rows[0]["max_job_max_request_length"] == "99.0"
    assert rows[0]["avg_turns_per_run"] == "7.5"
    assert rows[0]["max_turns_per_run"] == "11.0"
    assert rows[0]["avg_run_prompt_tokens_per_request"] == "12.0"
    assert rows[0]["avg_run_generation_tokens_per_request"] == "3.5"


def test_build_rows_backfills_new_fields_from_older_summary_shape(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo-dataset" / "agent-a" / "run-1"
    output_dir = run_dir / "run-stats"
    output_dir.mkdir(parents=True)
    (output_dir / "run-stats-summary.json").write_text(
        json.dumps(
            {
                "dataset": "demo-dataset",
                "score": 0.3,
                "job_count": 2,
                "job_max_request_lengths": [10, 30],
                "jobs": [
                    {"request_count": 4},
                    {"request_count": 8},
                ],
                "avg_job_max_request_length": 20.0,
                "avg_run_prompt_tokens_per_request": 11.0,
                "avg_run_generation_tokens_per_request": 2.0,
            }
        ),
        encoding="utf-8",
    )

    rows = aggregate_runs_csv.build_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["agent_type"] == "agent-a"
    assert rows[0]["max_job_max_request_length"] == 30.0
    assert rows[0]["avg_turns_per_run"] == 6.0
    assert rows[0]["max_turns_per_run"] == 8.0
