from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_throughput_timeseries(run_dir: Path) -> Path:
    processed_dir = run_dir / "post-processed" / "request-throughput"
    processed_dir.mkdir(parents=True)
    timeseries_path = processed_dir / "request-throughput-timeseries.json"
    timeseries_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "request_count": 4,
                "finished_request_count": 4,
                "finished_request_count_status_200": 3,
                "non_200_finished_request_count": 1,
                "total_duration_s": 3.0,
                "timepoint_frequency_hz": 1.0,
                "window_size_s": 1.0,
                "throughput_points": [
                    {"time_s": 0.0, "throughput_requests_per_s": 0.5},
                    {"time_s": 1.0, "throughput_requests_per_s": 1.0},
                    {"time_s": 2.0, "throughput_requests_per_s": 0.75},
                ],
                "throughput_points_status_200": [
                    {"time_s": 0.0, "throughput_requests_per_s": 0.5},
                    {"time_s": 1.0, "throughput_requests_per_s": 0.75},
                    {"time_s": 2.0, "throughput_requests_per_s": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    return timeseries_path


def test_discover_run_dirs_with_request_throughput_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-timeseries"

    _write_throughput_timeseries(good_run)
    bad_processed_dir = bad_run / "post-processed" / "request-throughput"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_request_throughput(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figure_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_throughput_timeseries(run_dir)

    def fake_render_throughput_figure(
        *,
        series_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del series_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_throughput_figure",
        fake_render_throughput_figure,
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
        / "request-throughput"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 2
    assert manifest["variant_count"] == 2
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "request-throughput.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["sample_count"] == 3
    assert manifest["avg_throughput_requests_per_s"] == 0.75
    assert manifest["max_throughput_requests_per_s"] == 1.0
    assert manifest["peak_time_s"] == 1.0
    assert Path(manifest["figure_path"]).is_file()
    assert [figure["variant_id"] for figure in manifest["figures"]] == [
        "all-finished",
        "status-200-only",
    ]
    assert [figure["figure_file_name"] for figure in manifest["figures"]] == [
        "request-throughput.png",
        "request-throughput-status-200.png",
    ]
    for figure in manifest["figures"]:
        assert Path(figure["figure_path"]).is_file()


def test_generate_figure_for_run_dir_writes_profile_specific_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    processed_dir = run_dir / "post-processed" / "request-throughput"
    processed_dir.mkdir(parents=True)
    (processed_dir / "request-throughput-timeseries.json").write_text(
        json.dumps(
            {
                "request_count": 2,
                "finished_request_count": 2,
                "finished_request_count_status_200": 1,
                "total_duration_s": 2.0,
                "timepoint_frequency_hz": 1.0,
                "window_size_s": 1.0,
                "throughput_points": [{"time_s": 0.0, "throughput_requests_per_s": 1.0}],
                "throughput_points_status_200": [{"time_s": 0.0, "throughput_requests_per_s": 0.5}],
                "multi_profile": True,
                "port_profile_ids": [2, 13],
                "series_keys": ["profile-2", "profile-13"],
                "series_by_profile": {
                    "profile-2": {
                        "gateway_profile_id": 2,
                        "request_count": 1,
                        "finished_request_count": 1,
                        "finished_request_count_status_200": 1,
                        "total_duration_s": 2.0,
                        "timepoint_frequency_hz": 1.0,
                        "window_size_s": 1.0,
                        "throughput_points": [{"time_s": 0.0, "throughput_requests_per_s": 1.0}],
                        "throughput_points_status_200": [{"time_s": 0.0, "throughput_requests_per_s": 1.0}],
                    },
                    "profile-13": {
                        "gateway_profile_id": 13,
                        "request_count": 1,
                        "finished_request_count": 1,
                        "finished_request_count_status_200": 0,
                        "total_duration_s": 2.0,
                        "timepoint_frequency_hz": 1.0,
                        "window_size_s": 1.0,
                        "throughput_points": [{"time_s": 0.0, "throughput_requests_per_s": 0.5}],
                        "throughput_points_status_200": [{"time_s": 0.0, "throughput_requests_per_s": 0.0}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_render_throughput_figure(
        *,
        series_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del series_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_throughput_figure",
        fake_render_throughput_figure,
    )

    manifest_path = generate_all_figures.generate_figure_for_run_dir(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["series_count"] == 3
    assert manifest["variant_count"] == 2
    assert manifest["figure_count"] == 6
    assert any(
        figure["figure_file_name"] == "request-throughput.png"
        and figure["relative_output_subdir"] == "profile-2"
        for figure in manifest["figures"]
    )
    assert any(
        figure["figure_file_name"] == "request-throughput-status-200.png"
        and figure["relative_output_subdir"] == "profile-13"
        for figure in manifest["figures"]
    )


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
    _write_throughput_timeseries(run_a)
    _write_throughput_timeseries(run_b)

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
            / "request-throughput"
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
