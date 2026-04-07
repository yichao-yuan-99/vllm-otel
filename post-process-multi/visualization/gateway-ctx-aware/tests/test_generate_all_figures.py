from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_all_figures.py"
MODULE_NAME = "post_process_multi_gateway_ctx_aware_generate_all_figures"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec for {MODULE_PATH}")
generate_all_figures = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = generate_all_figures
SPEC.loader.exec_module(generate_all_figures)


def _write_ctx_aware_timeseries(run_dir: Path) -> Path:
    processed_dir = run_dir / "post-processed" / "gateway" / "ctx-aware-log"
    processed_dir.mkdir(parents=True)
    timeseries_path = processed_dir / generate_all_figures.DEFAULT_INPUT_NAME
    timeseries_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "multi_profile": True,
                "port_profile_ids": [2, 13],
                "sample_count": 4,
                "duration_s": 0.3,
                "avg_sample_interval_s": 0.1,
                "started_at": "2026-04-04T15:38:01.000Z",
                "ended_at": "2026-04-04T15:38:01.300Z",
                "samples": [
                    {
                        "second": 0.0,
                        "timestamp": "2026-04-04T15:38:01.000Z",
                        "ongoing_agent_count": 1,
                        "pending_agent_count": 0,
                        "ongoing_effective_context_tokens": 100,
                        "pending_effective_context_tokens": 0,
                        "agents_turned_pending_due_to_context_threshold": 0,
                        "agents_turned_ongoing": 0,
                        "new_agents_added_as_pending": 0,
                        "new_agents_added_as_ongoing": 1,
                    },
                    {
                        "second": 0.1,
                        "timestamp": "2026-04-04T15:38:01.100Z",
                        "ongoing_agent_count": 3,
                        "pending_agent_count": 0,
                        "ongoing_effective_context_tokens": 300,
                        "pending_effective_context_tokens": 0,
                        "agents_turned_pending_due_to_context_threshold": 0,
                        "agents_turned_ongoing": 0,
                        "new_agents_added_as_pending": 0,
                        "new_agents_added_as_ongoing": 2,
                    },
                ],
                "ctx_aware_logs": [
                    {
                        "series_key": "profile-2",
                        "display_label": "Profile 2",
                        "port_profile_id": 2,
                        "source_ctx_aware_log_path": str(
                            run_dir / "gateway-output" / "job" / "ctx_aware_20260404T153801Z_profile-2.jsonl"
                        ),
                        "sample_count": 2,
                        "duration_s": 0.2,
                        "avg_sample_interval_s": 0.2,
                        "started_at": "2026-04-04T15:38:01.000Z",
                        "ended_at": "2026-04-04T15:38:01.200Z",
                        "samples": [
                            {
                                "second": 0.0,
                                "timestamp": "2026-04-04T15:38:01.000Z",
                                "ongoing_agent_count": 1,
                                "pending_agent_count": 0,
                                "ongoing_effective_context_tokens": 100,
                                "pending_effective_context_tokens": 0,
                                "agents_turned_pending_due_to_context_threshold": 0,
                                "agents_turned_ongoing": 0,
                                "new_agents_added_as_pending": 0,
                                "new_agents_added_as_ongoing": 1,
                            }
                        ],
                    },
                    {
                        "series_key": "profile-13",
                        "display_label": "Profile 13",
                        "port_profile_id": 13,
                        "source_ctx_aware_log_path": str(
                            run_dir / "gateway-output" / "job" / "ctx_aware_20260404T153801Z_profile-13.jsonl"
                        ),
                        "sample_count": 2,
                        "duration_s": 0.2,
                        "avg_sample_interval_s": 0.2,
                        "started_at": "2026-04-04T15:38:01.100Z",
                        "ended_at": "2026-04-04T15:38:01.300Z",
                        "samples": [
                            {
                                "second": 0.0,
                                "timestamp": "2026-04-04T15:38:01.100Z",
                                "ongoing_agent_count": 2,
                                "pending_agent_count": 0,
                                "ongoing_effective_context_tokens": 200,
                                "pending_effective_context_tokens": 0,
                                "agents_turned_pending_due_to_context_threshold": 0,
                                "agents_turned_ongoing": 0,
                                "new_agents_added_as_pending": 0,
                                "new_agents_added_as_ongoing": 2,
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return timeseries_path


def test_generate_figure_for_run_dir_writes_manifest_for_multiple_profiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    _write_ctx_aware_timeseries(run_dir)

    def fake_render_ctx_aware_figure(
        *,
        timeseries_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
        figure_title: str,
        subtitle_prefix: str | None = None,
    ) -> bool:
        del timeseries_payload, image_format, dpi, figure_title, subtitle_prefix
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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["multi_profile"] is True
    assert manifest["port_profile_ids"] == [2, 13]
    assert manifest["figure_count"] == 3
    assert manifest["figure_file_name"] == "ctx-aware-over-time.png"
    assert [figure["series_key"] for figure in manifest["figures"]] == [
        "aggregate",
        "profile-2",
        "profile-13",
    ]
    assert [figure["figure_file_name"] for figure in manifest["figures"]] == [
        "ctx-aware-over-time.png",
        "ctx-aware-over-time.png",
        "ctx-aware-over-time.png",
    ]
    assert [figure["relative_output_subdir"] for figure in manifest["figures"]] == [
        "",
        "profile-2",
        "profile-13",
    ]
    assert Path(manifest["figures"][0]["figure_path"]).is_file()
    assert Path(manifest["figures"][1]["figure_path"]).is_file()
    assert Path(manifest["figures"][2]["figure_path"]).is_file()


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
