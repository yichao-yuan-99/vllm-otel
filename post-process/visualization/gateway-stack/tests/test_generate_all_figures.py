from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_stack_histograms(run_dir: Path) -> Path:
    stack_dir = run_dir / "post-processed" / "gateway" / "stack"
    stack_dir.mkdir(parents=True)

    aggregate_points = [
        {"second": 0, "accumulated_value": 1.0},
        {"second": 1, "accumulated_value": 3.0},
        {"second": 2, "accumulated_value": 2.0},
    ]
    profile_2_points = [
        {"second": 0, "accumulated_value": 0.5},
        {"second": 1, "accumulated_value": 2.0},
        {"second": 2, "accumulated_value": 1.5},
    ]
    profile_13_points = [
        {"second": 0, "accumulated_value": 0.5},
        {"second": 1, "accumulated_value": 1.0},
        {"second": 2, "accumulated_value": 0.5},
    ]
    for metric_spec in generate_all_figures.METRIC_SPECS:
        metric_name = metric_spec["metric"]
        phase = "prefill"
        if metric_name == "completion_tokens":
            phase = "decode"
        if metric_name == "compute_prompt_plus_completion_tokens":
            phase = "mixed"
        payload = {
            "metric": metric_name,
            "phase": phase,
            "multi_profile": True,
            "port_profile_ids": [2, 13],
            "series_keys": ["profile-2", "profile-13"],
            "bucket_width_s": 1,
            "points": aggregate_points,
            "series_by_profile": {
                "profile-2": {
                    "metric": metric_name,
                    "phase": phase,
                    "gateway_profile_id": 2,
                    "bucket_width_s": 1,
                    "points": profile_2_points,
                },
                "profile-13": {
                    "metric": metric_name,
                    "phase": phase,
                    "gateway_profile_id": 13,
                    "bucket_width_s": 1,
                    "points": profile_13_points,
                },
            },
        }
        (stack_dir / metric_spec["input_name"]).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    return stack_dir


def test_discover_run_dirs_with_gateway_stack_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-input"

    _write_stack_histograms(good_run)
    bad_stack_dir = bad_run / "post-processed" / "gateway" / "stack"
    bad_stack_dir.mkdir(parents=True)
    (bad_stack_dir / generate_all_figures.METRIC_SPECS[0]["input_name"]).write_text(
        "{}",
        encoding="utf-8",
    )

    discovered = generate_all_figures.discover_run_dirs_with_gateway_stack(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figures_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_stack_histograms(run_dir)

    def fake_render_metric_figure(
        *,
        metric_spec: dict[str, str],
        histogram_payload: dict[str, object],
        window_size_s: int | None,
        title_suffix: str,
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> tuple[bool, dict[str, object]]:
        del metric_spec, histogram_payload, window_size_s, title_suffix, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return (
            True,
            {
                "sample_count": 3,
                "avg": 2.0,
                "min": 1.0,
                "max": 3.0,
                "peak_second": 1.0,
                "total_accumulated_value": 6.0,
            },
        )

    monkeypatch.setattr(
        generate_all_figures,
        "_render_metric_figure",
        fake_render_metric_figure,
    )

    manifest_path = generate_all_figures.generate_figures_for_run_dir(
        run_dir,
        image_format="png",
        dpi=150,
    )

    assert manifest_path == (
        run_dir
        / "post-processed"
        / "visualization"
        / "gateway-stack"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["multi_profile"] is True
    assert manifest["port_profile_ids"] == [2, 13]
    assert manifest["series_count"] == 3
    assert manifest["metric_count"] == 5
    assert manifest["variant_count"] == 5
    assert manifest["figure_count"] == 75
    assert manifest["skipped_metric_variant_count"] == 0
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert len(manifest["figures"]) == 75
    assert any(
        entry["figure_file_name"] == "prompt-tokens-stacked-histogram.png"
        for entry in manifest["figures"]
    )
    assert any(
        entry["figure_file_name"] == "prompt-tokens-stacked-histogram-smoothed-10s.png"
        for entry in manifest["figures"]
    )
    assert any(
        entry["figure_file_name"]
        == "compute-prompt-plus-completion-tokens-stacked-histogram-smoothed-120s.png"
        for entry in manifest["figures"]
    )
    assert any(
        entry["figure_file_name"] == "prompt-tokens-stacked-histogram.png"
        and entry["relative_output_subdir"] == "profile-2"
        for entry in manifest["figures"]
    )
    assert any(
        entry["figure_file_name"]
        == "prompt-tokens-stacked-histogram-smoothed-10s.png"
        and entry["relative_output_subdir"] == "profile-13"
        for entry in manifest["figures"]
    )
    for entry in manifest["figures"]:
        assert entry["figure_generated"] is True
        assert Path(entry["figure_path"]).is_file()


def test_smooth_values_with_centered_window() -> None:
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [0.0, 1.0, 2.0, 3.0, 4.0]

    smoothed = generate_all_figures._smooth_values_with_centered_window(
        x_values,
        y_values,
        window_size_s=2.0,
    )

    assert smoothed == pytest.approx([0.5, 1.0, 2.0, 3.0, 3.5])


def test_generate_figures_for_run_dir_rejects_missing_input_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    stack_dir = run_dir / "post-processed" / "gateway" / "stack"
    stack_dir.mkdir(parents=True)

    for metric_spec in generate_all_figures.METRIC_SPECS[:-1]:
        (stack_dir / metric_spec["input_name"]).write_text("{}", encoding="utf-8")

    try:
        generate_all_figures.generate_figures_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing stack histogram file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when stack histogram file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_stack_histograms(run_a)
    _write_stack_histograms(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figures_for_run_dir(
        run_dir: Path,
        *,
        stack_input_dir: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del stack_input_dir, output_dir
        processed.append((run_dir, image_format, dpi))
        return run_dir / "post-processed" / "visualization" / "gateway-stack" / "figures-manifest.json"

    monkeypatch.setattr(
        generate_all_figures,
        "generate_figures_for_run_dir",
        fake_generate_figures_for_run_dir,
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


def test_main_rejects_stack_input_dir_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        generate_all_figures.main(
            ["--root-dir", str(root_dir), "--stack-input-dir", str(tmp_path / "stack")]
        )
    except ValueError as exc:
        assert "--stack-input-dir can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --stack-input-dir is used with --root-dir")
