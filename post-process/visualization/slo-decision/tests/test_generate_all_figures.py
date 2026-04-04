from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_slo_decision_summary(run_dir: Path) -> Path:
    processed_dir = run_dir / "post-processed" / "slo-decision"
    processed_dir.mkdir(parents=True)
    summary_path = processed_dir / "slo-decision-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_type": "replay",
                "slo_decision_log_found": True,
                "slo_decision_point_count": 2,
                "slo_decision_change_count": 1,
                "target_output_throughput_tokens_per_s": 8.0,
                "min_window_min_output_tokens_per_s": 4.0,
                "max_window_min_output_tokens_per_s": 4.5,
                "min_frequency_mhz": 1200,
                "max_frequency_mhz": 1500,
                "analysis_window_start_utc": "2026-04-02T00:00:00Z",
                "decision_points": [
                    {
                        "time_offset_s": 5.0,
                        "window_min_output_tokens_per_s": 4.5,
                        "current_frequency_mhz": 1200,
                        "target_frequency_mhz": 1500,
                        "changed": True,
                    },
                    {
                        "time_offset_s": 10.0,
                        "window_min_output_tokens_per_s": 4.0,
                        "current_frequency_mhz": 1500,
                        "target_frequency_mhz": 1500,
                        "changed": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_discover_run_dirs_with_slo_decision_summary_scans_recursively(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-summary"

    _write_slo_decision_summary(good_run)
    bad_processed_dir = bad_run / "post-processed" / "slo-decision"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_slo_decision_summary(
        root_dir
    )

    assert discovered == [good_run.resolve()]


def test_generate_figure_for_run_dir_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    summary_path = _write_slo_decision_summary(run_dir)

    def fake_render_slo_decision_figure(
        *,
        slo_decision_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del slo_decision_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_slo_decision_figure",
        fake_render_slo_decision_figure,
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
        / "slo-decision"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_slo_decision_summary_path"] == str(summary_path.resolve())
    assert manifest["figure_count"] == 1
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "slo-decision-timeline.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["slo_decision_point_count"] == 2
    assert manifest["slo_decision_change_count"] == 1
    assert manifest["target_output_throughput_tokens_per_s"] == 8.0
    assert manifest["min_window_min_output_tokens_per_s"] == 4.0
    assert manifest["max_window_min_output_tokens_per_s"] == 4.5
    assert manifest["min_frequency_mhz"] == 1200
    assert manifest["max_frequency_mhz"] == 1500
    assert Path(manifest["figure_path"]).is_file()


def test_generate_figure_for_run_dir_rejects_missing_slo_decision_summary_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figure_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing SLO-decision summary file" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when SLO-decision summary file is missing"
        )


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_slo_decision_summary(run_a)
    _write_slo_decision_summary(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figure_for_run_dir(
        run_dir: Path,
        *,
        slo_decision_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del slo_decision_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "slo-decision"
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
