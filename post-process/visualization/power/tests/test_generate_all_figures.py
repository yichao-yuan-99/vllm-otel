from __future__ import annotations

import json
from typing import Any
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_power_summary(
    run_dir: Path,
    *,
    per_gpu_power: list[dict[str, Any]] | None = None,
) -> Path:
    processed_dir = run_dir / "post-processed" / "power"
    processed_dir.mkdir(parents=True)
    power_summary_path = processed_dir / "power-summary.json"
    power_summary_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_type": "replay",
                "power_log_found": True,
                "power_sample_count": 3,
                "total_energy_j": 1400.0,
                "total_energy_kwh": 0.000388888889,
                "analysis_window_start_utc": "2026-03-08T00:00:00Z",
                "power_points": [
                    {"time_offset_s": 0.0, "power_w": 100.0},
                    {"time_offset_s": 1.0, "power_w": 300.0},
                    {"time_offset_s": 2.0, "power_w": 200.0},
                ],
                "per_gpu_power": per_gpu_power or [],
            }
        ),
        encoding="utf-8",
    )
    return power_summary_path


def test_discover_run_dirs_with_power_summary_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-summary"

    _write_power_summary(good_run)
    bad_processed_dir = bad_run / "post-processed" / "power"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_power_summary(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figure_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_power_summary(run_dir)

    def fake_render_power_figure(
        *,
        power_summary_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del power_summary_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_power_figure",
        fake_render_power_figure,
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
        / "power"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 1
    assert manifest["per_gpu_figure_count"] == 0
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "gpu-power-over-time.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["sample_count"] == 3
    assert manifest["avg_power_w"] == 200.0
    assert manifest["max_power_w"] == 300.0
    assert manifest["peak_time_s"] == 1.0
    assert manifest["total_energy_j"] == 1400.0
    assert Path(manifest["figure_path"]).is_file()
    assert manifest["figures"] == [
        {
            "figure_kind": "aggregate",
            "figure_generated": True,
            "figure_file_name": "gpu-power-over-time.png",
            "figure_path": str(
                (
                    run_dir
                    / "post-processed"
                    / "visualization"
                    / "power"
                    / "gpu-power-over-time.png"
                ).resolve()
            ),
            "display_label": "All GPUs",
            "gpu_key": None,
            "gpu_id": None,
            "source_endpoint": None,
            "sample_count": 3,
            "avg_power_w": 200.0,
            "min_power_w": 100.0,
            "max_power_w": 300.0,
            "peak_time_s": 1.0,
            "total_energy_j": 1400.0,
            "total_energy_kwh": 0.000388888889,
            "skip_reason": None,
        }
    ]


def test_generate_figure_for_run_dir_writes_per_gpu_figures_for_multi_gpu_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    _write_power_summary(
        run_dir,
        per_gpu_power=[
            {
                "gpu_key": "0",
                "gpu_id": "0",
                "display_label": "GPU 0",
                "source_endpoint": "/var/run/zeusd.sock",
                "power_sample_count": 3,
                "power_stats_w": {"avg": 90.0, "min": 80.0, "max": 100.0},
                "total_energy_j": 180.0,
                "total_energy_kwh": 0.00005,
                "power_points": [
                    {"time_offset_s": 0.0, "power_w": 80.0},
                    {"time_offset_s": 1.0, "power_w": 90.0},
                    {"time_offset_s": 2.0, "power_w": 100.0},
                ],
            },
            {
                "gpu_key": "1",
                "gpu_id": "1",
                "display_label": "GPU 1",
                "source_endpoint": "/var/run/zeusd.sock",
                "power_sample_count": 3,
                "power_stats_w": {"avg": 110.0, "min": 90.0, "max": 130.0},
                "total_energy_j": 220.0,
                "total_energy_kwh": 0.000061111111,
                "power_points": [
                    {"time_offset_s": 0.0, "power_w": 90.0},
                    {"time_offset_s": 1.0, "power_w": 110.0},
                    {"time_offset_s": 2.0, "power_w": 130.0},
                ],
            },
        ],
    )

    rendered_labels: list[tuple[str | None, str]] = []

    def fake_render_power_figure(
        *,
        power_summary_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del image_format, dpi
        rendered_labels.append(
            (
                power_summary_payload.get("_figure_variant_label"),
                output_path.name,
            )
        )
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_power_figure",
        fake_render_power_figure,
    )

    manifest_path = generate_all_figures.generate_figure_for_run_dir(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert rendered_labels == [
        (None, "gpu-power-over-time.png"),
        ("GPU 0", "gpu-power-over-time-0.png"),
        ("GPU 1", "gpu-power-over-time-1.png"),
    ]
    assert manifest["figure_count"] == 3
    assert manifest["per_gpu_figure_count"] == 2
    assert [figure["figure_kind"] for figure in manifest["figures"]] == [
        "aggregate",
        "per_gpu",
        "per_gpu",
    ]
    assert [figure["figure_file_name"] for figure in manifest["figures"]] == [
        "gpu-power-over-time.png",
        "gpu-power-over-time-0.png",
        "gpu-power-over-time-1.png",
    ]
    assert [figure["display_label"] for figure in manifest["figures"]] == [
        "All GPUs",
        "GPU 0",
        "GPU 1",
    ]


def test_generate_figure_for_run_dir_rejects_missing_power_summary_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figure_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing power summary file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when power summary file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_power_summary(run_a)
    _write_power_summary(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figure_for_run_dir(
        run_dir: Path,
        *,
        power_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del power_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return run_dir / "post-processed" / "visualization" / "power" / "figures-manifest.json"

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


def test_main_rejects_power_input_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(
            ["--root-dir", str(root_dir), "--power-input", str(tmp_path / "power-summary.json")]
        )
    except ValueError as exc:
        assert "--power-input can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --power-input is used with --root-dir")
