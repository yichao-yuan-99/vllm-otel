from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import run_all


def _script_tail(command: list[str]) -> str:
    return Path(command[1]).as_posix().split("post-process/")[-1]


def test_main_run_dir_invokes_scripts_in_order(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    assert [_script_tail(command) for command in commands] == [
        "service-failure/extract_run.py",
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "job-concurrency/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/stack/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
        "visualization/job-throughput/generate_all_figures.py",
        "visualization/job-concurrency/generate_all_figures.py",
        "visualization/gateway-stack/generate_all_figures.py",
        "visualization/vllm-metrics/generate_all_figures.py",
    ]
    for command in commands:
        assert command[2:] == ["--run-dir", str(run_dir.resolve())]


def test_main_root_dir_invokes_scripts_in_order_and_aggregate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--root-dir", str(root_dir), "--max-procs", "4"])

    assert exit_code == 0
    assert [_script_tail(command) for command in commands] == [
        "service-failure/extract_run.py",
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "job-concurrency/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/stack/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
        "visualization/job-throughput/generate_all_figures.py",
        "visualization/job-concurrency/generate_all_figures.py",
        "visualization/gateway-stack/generate_all_figures.py",
        "visualization/vllm-metrics/generate_all_figures.py",
        "global/aggregate_runs_csv.py",
    ]
    for command in commands[:-1]:
        assert command[2:] == [
            "--root-dir",
            str(root_dir.resolve()),
            "--max-procs",
            "4",
        ]
    assert commands[-1][2:] == [
        "--root-dir",
        str(root_dir.resolve()),
    ]


def test_main_root_dir_dry_run_skips_aggregate(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--root-dir", str(root_dir), "--dry-run", "--max-procs", "2"])

    assert exit_code == 0
    assert [_script_tail(command) for command in commands] == [
        "service-failure/extract_run.py",
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "job-concurrency/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/stack/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
        "visualization/job-throughput/generate_all_figures.py",
        "visualization/job-concurrency/generate_all_figures.py",
        "visualization/gateway-stack/generate_all_figures.py",
        "visualization/vllm-metrics/generate_all_figures.py",
    ]
    for command in commands:
        assert command[-1] == "--dry-run"


def test_main_reports_failures_but_continues(monkeypatch, tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        if _script_tail(command) == "gateway/usage/extract_run.py":
            return 3
        if _script_tail(command) == "visualization/gateway-stack/generate_all_figures.py":
            return 7
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--run-dir", str(run_dir)])
    captured = capsys.readouterr()

    assert exit_code == 3
    assert "[error] post-process pipeline completed with 2 failing step(s):" in captured.out
    assert "gateway/usage/extract_run.py" in captured.out
    assert "visualization/gateway-stack/generate_all_figures.py" in captured.out
    assert [_script_tail(command) for command in commands] == [
        "service-failure/extract_run.py",
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "job-concurrency/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/stack/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
        "visualization/job-throughput/generate_all_figures.py",
        "visualization/job-concurrency/generate_all_figures.py",
        "visualization/gateway-stack/generate_all_figures.py",
        "visualization/vllm-metrics/generate_all_figures.py",
    ]


def test_main_rejects_dry_run_with_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        run_all.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_main_run_dir_with_nested_runs_falls_back_to_root_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "orchestrator"
    nested_run = run_dir / "nested-job"
    (nested_run / "replay").mkdir(parents=True)
    (nested_run / "replay" / "summary.json").write_text("{}", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run_command(command: list[str]) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(run_all, "_run_command", fake_run_command)

    exit_code = run_all.main(["--run-dir", str(run_dir), "--max-procs", "3"])

    assert exit_code == 0
    assert [_script_tail(command) for command in commands] == [
        "service-failure/extract_run.py",
        "global/extract_run.py",
        "global-progress/extract_run.py",
        "job-throughput/extract_run.py",
        "job-concurrency/extract_run.py",
        "gateway/llm-requests/extract_run.py",
        "gateway/stack/extract_run.py",
        "gateway/usage/extract_run.py",
        "split/duration/extract_run.py",
        "vllm-metrics/extract_run.py",
        "vllm-metrics/summarize_timeseries.py",
        "visualization/job-throughput/generate_all_figures.py",
        "visualization/job-concurrency/generate_all_figures.py",
        "visualization/gateway-stack/generate_all_figures.py",
        "visualization/vllm-metrics/generate_all_figures.py",
        "global/aggregate_runs_csv.py",
    ]
    for command in commands[:-1]:
        assert command[2:] == [
            "--root-dir",
            str(run_dir.resolve()),
            "--max-procs",
            "3",
        ]
    assert commands[-1][2:] == [
        "--root-dir",
        str(run_dir.resolve()),
    ]
