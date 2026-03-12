from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import plot_job_context_cdf


def _write_run_summary(run_dir: Path, payload: dict[str, object]) -> Path:
    summary_path = run_dir / "run-stats" / "run-stats-summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    return summary_path


def test_discover_run_dirs_with_run_stats_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_path = root_dir / "b" / "other"

    _write_run_summary(good_run, payload={"job_max_request_lengths": [1, 2, 3]})
    bad_path.mkdir(parents=True)
    (bad_path / "run-stats-summary.json").write_text("{}", encoding="utf-8")

    discovered = plot_job_context_cdf.discover_run_dirs_with_run_stats(root_dir)
    assert discovered == [good_run.resolve()]


def test_generate_cdf_for_run_dir_writes_default_output(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_run_summary(
        run_dir,
        payload={
            "dataset": "demo",
            "agent_type": "mini-swe-agent",
            "job_max_request_lengths": [30, 10, 20],
        },
    )

    captured: dict[str, object] = {}

    def fake_render_cdf_figure(
        *,
        x_values: list[int],
        y_values: list[float],
        title: str,
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> None:
        captured["x_values"] = x_values
        captured["y_values"] = y_values
        captured["title"] = title
        captured["image_format"] = image_format
        captured["dpi"] = dpi
        output_path.write_text("fake-image", encoding="utf-8")

    monkeypatch.setattr(
        plot_job_context_cdf,
        "_render_cdf_figure",
        fake_render_cdf_figure,
    )

    output_path = plot_job_context_cdf.generate_cdf_for_run_dir(
        run_dir,
        image_format="png",
        dpi=150,
    )

    assert output_path == (
        run_dir / "run-stats" / "job-max-request-length-cdf.png"
    ).resolve()
    assert output_path.is_file()
    assert captured["x_values"] == [10, 20, 30]
    assert captured["y_values"] == [1 / 3, 2 / 3, 1.0]
    assert captured["title"] == "Job Context Usage CDF (demo, mini-swe-agent)"
    assert captured["image_format"] == "png"
    assert captured["dpi"] == 150


def test_generate_cdf_for_run_dir_backfills_from_jobs(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_run_summary(
        run_dir,
        payload={
            "dataset": "demo",
            "agent_type": "agent-a",
            "jobs": [
                {"max_request_length": 8},
                {"max_request_length": 3},
                {"max_request_length": None},
            ],
        },
    )

    captured: dict[str, object] = {}

    def fake_render_cdf_figure(
        *,
        x_values: list[int],
        y_values: list[float],
        title: str,
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> None:
        del y_values, title, image_format, dpi
        captured["x_values"] = x_values
        output_path.write_text("fake-image", encoding="utf-8")

    monkeypatch.setattr(
        plot_job_context_cdf,
        "_render_cdf_figure",
        fake_render_cdf_figure,
    )

    output_path = plot_job_context_cdf.generate_cdf_for_run_dir(run_dir)

    assert output_path.is_file()
    assert captured["x_values"] == [3, 8]


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_run_summary(run_a, payload={"job_max_request_lengths": [1]})
    _write_run_summary(run_b, payload={"job_max_request_lengths": [2]})

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_cdf_for_run_dir(
        run_dir: Path,
        *,
        summary_input_path: Path | None = None,
        output_path: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del summary_input_path, output_path
        processed.append((run_dir, image_format, dpi))
        return run_dir / "run-stats" / "job-max-request-length-cdf.svg"

    monkeypatch.setattr(
        plot_job_context_cdf,
        "generate_cdf_for_run_dir",
        fake_generate_cdf_for_run_dir,
    )

    exit_code = plot_job_context_cdf.main(
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
        plot_job_context_cdf.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_main_rejects_output_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        plot_job_context_cdf.main(
            ["--root-dir", str(root_dir), "--output", str(tmp_path / "figure.png")]
        )
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")
