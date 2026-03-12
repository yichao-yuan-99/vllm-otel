from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_timeseries_and_stats(run_dir: Path) -> tuple[Path, Path]:
    processed_dir = run_dir / "post-processed" / "vllm-log"
    processed_dir.mkdir(parents=True)
    timeseries_path = processed_dir / "gauge-counter-timeseries.json"
    stats_path = processed_dir / "gauge-counter-timeseries.stats.json"

    timeseries_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "metrics": {
                    "vllm:num_requests_running|engine=0": {
                        "name": "vllm:num_requests_running",
                        "labels": {"engine": "0"},
                        "time_from_start_s": [0.0, 1.0, 2.0],
                        "value": [0.0, 1.0, 2.0],
                    },
                    "vllm:kv_cache_usage_perc|engine=0": {
                        "name": "vllm:kv_cache_usage_perc",
                        "labels": {"engine": "0"},
                        "time_from_start_s": [0.0, 1.0, 2.0],
                        "value": [0.2, 0.3, 0.4],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "vllm:num_requests_running|engine=0": {
                        "sample_count": 3,
                        "min": 0.0,
                        "max": 2.0,
                        "avg": 1.0,
                    },
                    "vllm:kv_cache_usage_perc|engine=0": {
                        "sample_count": 3,
                        "min": 0.2,
                        "max": 0.4,
                        "avg": 0.3,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return timeseries_path, stats_path


def test_discover_run_dirs_with_metric_stats_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-timeseries"

    _write_timeseries_and_stats(good_run)
    bad_processed_dir = bad_run / "post-processed" / "vllm-log"
    bad_processed_dir.mkdir(parents=True)
    (bad_processed_dir / "gauge-counter-timeseries.stats.json").write_text(
        "{}",
        encoding="utf-8",
    )

    discovered = generate_all_figures.discover_run_dirs_with_metric_stats(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figures_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_timeseries_and_stats(run_dir)

    def fake_render_metric_figure(
        *,
        series_key: str,
        metric_payload: dict[str, object],
        stats_payload: dict[str, object] | None,
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del series_key, metric_payload, stats_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

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
        / "vllm-metrics"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metric_count"] == 2
    assert manifest["figure_count"] == 2
    assert manifest["skipped_metric_count"] == 0
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    for figure in manifest["figures"]:
        assert Path(figure["figure_path"]).is_file()


def test_generate_figures_for_run_dir_rejects_missing_stats_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    processed_dir = run_dir / "post-processed" / "vllm-log"
    processed_dir.mkdir(parents=True)
    (processed_dir / "gauge-counter-timeseries.json").write_text(
        json.dumps({"metrics": {}}),
        encoding="utf-8",
    )

    try:
        generate_all_figures.generate_figures_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing stats file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when stats file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_timeseries_and_stats(run_a)
    _write_timeseries_and_stats(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figures_for_run_dir(
        run_dir: Path,
        *,
        timeseries_input_path: Path | None = None,
        stats_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del timeseries_input_path, stats_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return run_dir / "post-processed" / "visualization" / "vllm-metrics" / "figures-manifest.json"

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
