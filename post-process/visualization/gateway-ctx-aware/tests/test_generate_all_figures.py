from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_ctx_aware_timeseries(run_dir: Path) -> Path:
    processed_dir = run_dir / "post-processed" / "gateway" / "ctx-aware-log"
    processed_dir.mkdir(parents=True)
    timeseries_path = processed_dir / generate_all_figures.DEFAULT_INPUT_NAME
    timeseries_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_ctx_aware_log_path": str(
                    run_dir / "gateway-output" / "job" / "ctx_aware_20260404T153801Z.jsonl"
                ),
                "sample_count": 3,
                "duration_s": 0.4,
                "avg_sample_interval_s": 0.2,
                "started_at": "2026-04-04T15:38:01.552Z",
                "ended_at": "2026-04-04T15:38:01.952Z",
                "samples": [
                    {
                        "second": 0.0,
                        "timestamp": "2026-04-04T15:38:01.552Z",
                        "ongoing_agent_count": 0,
                        "pending_agent_count": 0,
                        "ongoing_effective_context_tokens": 0,
                        "pending_effective_context_tokens": 0,
                        "agents_turned_pending_due_to_context_threshold": 0,
                        "agents_turned_ongoing": 0,
                        "new_agents_added_as_pending": 0,
                        "new_agents_added_as_ongoing": 0,
                    },
                    {
                        "second": 0.2,
                        "timestamp": "2026-04-04T15:38:01.752Z",
                        "ongoing_agent_count": 1,
                        "pending_agent_count": 0,
                        "ongoing_effective_context_tokens": 3000,
                        "pending_effective_context_tokens": 0,
                        "agents_turned_pending_due_to_context_threshold": 0,
                        "agents_turned_ongoing": 0,
                        "new_agents_added_as_pending": 0,
                        "new_agents_added_as_ongoing": 1,
                    },
                    {
                        "second": 0.4,
                        "timestamp": "2026-04-04T15:38:01.952Z",
                        "ongoing_agent_count": 1,
                        "pending_agent_count": 1,
                        "ongoing_effective_context_tokens": 2800,
                        "pending_effective_context_tokens": 3000,
                        "agents_turned_pending_due_to_context_threshold": 1,
                        "agents_turned_ongoing": 0,
                        "new_agents_added_as_pending": 1,
                        "new_agents_added_as_ongoing": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return timeseries_path


def test_discover_run_dirs_with_gateway_ctx_aware_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-timeseries"

    _write_ctx_aware_timeseries(good_run)
    bad_processed_dir = bad_run / "post-processed" / "gateway" / "ctx-aware-log"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_gateway_ctx_aware(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figure_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_ctx_aware_timeseries(run_dir)

    def fake_render_ctx_aware_figure(
        *,
        timeseries_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del timeseries_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_ctx_aware_figure",
        fake_render_ctx_aware_figure,
    )

    manifest_path = generate_all_figures.generate_figure_for_run_dir(
        run_dir,
        image_format="png",
        dpi=150,
    )

    assert manifest_path == (
        run_dir
        / "post-processed"
        / "visualization"
        / "gateway-ctx-aware"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 1
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "ctx-aware-over-time.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["sample_count"] == 3
    assert manifest["duration_s"] == 0.4
    assert (
        manifest["series_summaries"]["agents_turned_pending_due_to_context_threshold"]["total"]
        == 1.0
    )
    assert manifest["series_summaries"]["ongoing_agent_count"]["max"] == 1.0
    assert Path(manifest["figure_path"]).is_file()


def test_generate_figure_for_run_dir_rejects_missing_timeseries_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figure_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing timeseries file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when timeseries file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_ctx_aware_timeseries(run_a)
    _write_ctx_aware_timeseries(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figure_for_run_dir(
        run_dir: Path,
        *,
        timeseries_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del timeseries_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "gateway-ctx-aware"
            / "figures-manifest.json"
        )

    monkeypatch.setattr(
        generate_all_figures,
        "generate_figure_for_run_dir",
        fake_generate_figure_for_run_dir,
    )

    exit_code = generate_all_figures.main(
        ["--root-dir", str(root_dir), "--max-procs", "1", "--format", "svg", "--dpi", "144"]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), "svg", 144),
        (run_b.resolve(), "svg", 144),
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


def test_main_rejects_timeseries_input_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(
            ["--root-dir", str(root_dir), "--timeseries-input", str(tmp_path / "timeseries.json")]
        )
    except ValueError as exc:
        assert "--timeseries-input can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --timeseries-input is used with --root-dir")
