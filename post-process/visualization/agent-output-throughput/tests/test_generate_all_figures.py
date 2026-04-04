from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_agent_output_summary(run_dir: Path) -> Path:
    processed_dir = run_dir / "post-processed" / "agent-output-throughput"
    processed_dir.mkdir(parents=True)
    input_path = processed_dir / "agent-output-throughput.json"
    input_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_gateway_output_dir": str(run_dir / "gateway-output"),
                "service_failure_detected": False,
                "service_failure_cutoff_time_utc": None,
                "agent_count": 2,
                "request_count": 3,
                "output_tokens": 35,
                "llm_request_duration_s": 3.0,
                "output_throughput_tokens_per_s": 11.666667,
                "agent_output_throughput_tokens_per_s_summary": {
                    "sample_count": 2,
                    "avg": 13.75,
                    "min": 7.5,
                    "max": 20.0,
                    "std": 6.25,
                },
                "agent_output_throughput_tokens_per_s_histogram": {
                    "metric": "output_throughput_tokens_per_s",
                    "bin_size": 1.0,
                    "sample_count": 2,
                    "bin_count": 14,
                    "min": 7.5,
                    "max": 20.0,
                    "bins": [
                        {"bin_start": 7.0, "bin_end": 8.0, "count": 1},
                        {"bin_start": 8.0, "bin_end": 9.0, "count": 0},
                        {"bin_start": 9.0, "bin_end": 10.0, "count": 0},
                        {"bin_start": 10.0, "bin_end": 11.0, "count": 0},
                        {"bin_start": 11.0, "bin_end": 12.0, "count": 0},
                        {"bin_start": 12.0, "bin_end": 13.0, "count": 0},
                        {"bin_start": 13.0, "bin_end": 14.0, "count": 0},
                        {"bin_start": 14.0, "bin_end": 15.0, "count": 0},
                        {"bin_start": 15.0, "bin_end": 16.0, "count": 0},
                        {"bin_start": 16.0, "bin_end": 17.0, "count": 0},
                        {"bin_start": 17.0, "bin_end": 18.0, "count": 0},
                        {"bin_start": 18.0, "bin_end": 19.0, "count": 0},
                        {"bin_start": 19.0, "bin_end": 20.0, "count": 0},
                        {"bin_start": 20.0, "bin_end": 21.0, "count": 1},
                    ],
                },
                "agents": [
                    {
                        "gateway_run_id": "run-b",
                        "gateway_profile_id": None,
                        "api_token_hash": "hash-b",
                        "request_count": 1,
                        "output_tokens": 20,
                        "llm_request_duration_s": 1.0,
                        "output_throughput_tokens_per_s": 20.0,
                    },
                    {
                        "gateway_run_id": "run-c",
                        "gateway_profile_id": None,
                        "api_token_hash": "hash-c",
                        "request_count": 2,
                        "output_tokens": 15,
                        "llm_request_duration_s": 2.0,
                        "output_throughput_tokens_per_s": 7.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return input_path


def test_discover_run_dirs_with_agent_output_throughput_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-input"

    _write_agent_output_summary(good_run)
    bad_processed_dir = bad_run / "post-processed" / "agent-output-throughput"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_agent_output_throughput(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figures_for_run_dir_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "job"
    _write_agent_output_summary(run_dir)

    def fake_render_histogram_figure(
        *,
        agent_output_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del agent_output_payload, image_format, dpi
        output_path.write_text("fake-histogram", encoding="utf-8")
        return True

    def fake_render_scatter_figure(
        *,
        agent_output_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del agent_output_payload, image_format, dpi
        output_path.write_text("fake-scatter", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_histogram_figure",
        fake_render_histogram_figure,
    )
    monkeypatch.setattr(
        generate_all_figures,
        "_render_scatter_figure",
        fake_render_scatter_figure,
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
        / "agent-output-throughput"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 2
    assert manifest["requested_figure_count"] == 2
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "agent-output-throughput-histogram.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["agent_count"] == 2
    assert manifest["run_output_throughput_tokens_per_s"] == 11.666667
    assert Path(manifest["figure_path"]).is_file()
    assert [figure["figure_id"] for figure in manifest["figures"]] == [
        "histogram",
        "scatter",
    ]
    assert [figure["figure_file_name"] for figure in manifest["figures"]] == [
        "agent-output-throughput-histogram.png",
        "agent-output-throughput-vs-output-tokens.png",
    ]
    for figure in manifest["figures"]:
        assert Path(figure["figure_path"]).is_file()


def test_generate_figures_for_run_dir_rejects_missing_input_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figures_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing agent-output-throughput file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when input file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_agent_output_summary(run_a)
    _write_agent_output_summary(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figures_for_run_dir(
        run_dir: Path,
        *,
        agent_output_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del agent_output_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "agent-output-throughput"
            / "figures-manifest.json"
        )

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
