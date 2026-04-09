from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_ranges_input(run_dir: Path) -> Path:
    stack_context_dir = run_dir / "post-processed" / "gateway" / "stack-context"
    stack_context_dir.mkdir(parents=True)
    input_path = stack_context_dir / generate_all_figures.DEFAULT_INPUT_NAME
    input_path.write_text(
        json.dumps(
            {
                "metric": "context_usage_tokens",
                "phase": "context",
                "entries": [
                    {
                        "agent_key": "agent-a",
                        "segment_type": "active",
                        "range_start_s": 0.0,
                        "range_end_s": 60.0,
                        "avg_value_per_s": 10.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return input_path


def _command_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_discover_run_dirs_with_stacked_per_agent_input_scans_recursively(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-input"

    _write_ranges_input(good_run)
    (bad_run / "post-processed" / "gateway" / "stack-context").mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_stacked_per_agent_input(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figures_for_run_dir_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    _write_ranges_input(run_dir)

    def fake_run_command(command: list[str]) -> int:
        output_path = Path(_command_value(command, "--output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if Path(command[1]).name == "materialize_stacked_per_agent.py":
            start_s = float(_command_value(command, "--start-s"))
            end_s = float(_command_value(command, "--end-s"))
            window_size_s = float(_command_value(command, "--window-size-s"))
            payload = {
                "window_size_s": window_size_s,
                "analysis_window_start_s": start_s,
                "analysis_window_end_s": end_s,
                "analysis_window_duration_s": end_s - start_s,
                "agent_order": _command_value(command, "--agent-order"),
                "agent_count": 2,
                "window_count": 6,
                "metric": "context_usage_tokens",
                "phase": "context",
            }
            output_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            return 0

        output_path.write_text("fake-image", encoding="utf-8")
        return 0

    monkeypatch.setattr(generate_all_figures, "_run_command", fake_run_command)

    manifest_path = generate_all_figures.generate_figures_for_run_dir(
        run_dir,
        window_size_s=60.0,
        analysis_start_s=120.0,
        analysis_end_s=480.0,
        value_mode="integral",
        legend="show",
        image_format="svg",
        dpi=150,
    )

    assert manifest_path == (
        run_dir
        / "post-processed"
        / "visualization"
        / "stacked-per-agent"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["materialized_file_name"] == "stacked-per-agent.window-60s.start-120.end-480.json"
    assert manifest["figure_file_name"] == "stacked-per-agent.window-60s.start-120.end-480.integral.svg"
    assert manifest["figure_generated"] is True
    assert manifest["image_format"] == "svg"
    assert manifest["dpi"] == 150
    assert manifest["window_size_s"] == 60.0
    assert manifest["analysis_window_start_s"] == 120.0
    assert manifest["analysis_window_end_s"] == 480.0
    assert manifest["analysis_window_duration_s"] == 360.0
    assert manifest["agent_order"] == "first-active"
    assert manifest["agent_count"] == 2
    assert manifest["window_count"] == 6
    assert manifest["value_mode"] == "integral"
    assert manifest["legend"] == "show"
    assert manifest["title"] == generate_all_figures.DEFAULT_TITLE
    assert Path(manifest["materialized_data_path"]).is_file()
    assert Path(manifest["figure_path"]).is_file()


def test_generate_figures_for_run_dir_rejects_missing_input_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figures_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing context-usage ranges file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when ranges input is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_ranges_input(run_a)
    _write_ranges_input(run_b)

    processed: list[tuple[Path, float, str, int]] = []

    def fake_generate_figures_for_run_dir(
        run_dir: Path,
        *,
        ranges_input_path: Path | None = None,
        output_dir: Path | None = None,
        window_size_s: float = 120.0,
        analysis_start_s: float = 0.0,
        analysis_end_s: float | None = None,
        agent_order: str = "first-active",
        value_mode: str = "average",
        legend: str = "auto",
        legend_max_agents: int = 24,
        title: str = generate_all_figures.DEFAULT_TITLE,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del (
            ranges_input_path,
            output_dir,
            analysis_start_s,
            analysis_end_s,
            agent_order,
            value_mode,
            legend,
            legend_max_agents,
            title,
        )
        processed.append((run_dir, window_size_s, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "stacked-per-agent"
            / "figures-manifest.json"
        )

    monkeypatch.setattr(
        generate_all_figures,
        "generate_figures_for_run_dir",
        fake_generate_figures_for_run_dir,
    )

    exit_code = generate_all_figures.main(
        [
            "--root-dir",
            str(root_dir),
            "--max-procs",
            "1",
            "--window-size-s",
            "90",
            "--format",
            "pdf",
            "--dpi",
            "144",
        ]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), 90.0, "pdf", 144),
        (run_b.resolve(), 90.0, "pdf", 144),
    ]


def test_main_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_main_rejects_output_dir_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(
            ["--root-dir", str(root_dir), "--output-dir", str(tmp_path / "figs")]
        )
    except ValueError as exc:
        assert "--output-dir can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output-dir is used with --root-dir")
