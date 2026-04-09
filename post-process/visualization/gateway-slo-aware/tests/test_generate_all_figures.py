from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_slo_aware_summary(run_dir: Path) -> Path:
    processed_dir = run_dir / "post-processed" / "gateway" / "slo-aware-log"
    processed_dir.mkdir(parents=True)
    summary_path = processed_dir / generate_all_figures.DEFAULT_INPUT_NAME
    summary_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_type": "replay",
                "source_slo_aware_log_paths": [
                    str(
                        run_dir
                        / "gateway-output"
                        / "job"
                        / "slo_aware_decisions_20260408T195807Z.jsonl"
                    )
                ],
                "slo_aware_log_found": True,
                "slo_aware_event_count": 2,
                "unique_agent_count": 1,
                "target_output_throughput_tokens_per_s": 25.0,
                "event_type_counts": {
                    "agent_entered_ralexation": 1,
                    "agent_left_ralexation": 1,
                },
                "wake_reason_counts": {"slo_recovered": 1},
                "resume_disposition_counts": {"ctx_aware_admitted": 1},
                "min_output_tokens_per_s_at_events": 41.773814,
                "max_output_tokens_per_s_at_events": 41.773814,
                "min_slo_slack_s": 173.320279,
                "max_slo_slack_s": 173.320279,
                "min_ralexation_duration_s": 86.66014,
                "max_ralexation_duration_s": 86.66014,
                "analysis_window_start_utc": "2026-04-08T19:58:00Z",
                "events": [
                    {
                        "time_offset_s": 7.014,
                        "event_type": "agent_entered_ralexation",
                        "output_tokens_per_s": 41.773814,
                        "slo_slack_s": 173.320279,
                        "slo_target_tokens_per_s": 25.0,
                        "min_output_tokens_per_s": 19.088245,
                        "avg_output_tokens_per_s": 38.009882,
                        "to_schedule_state": "ralexation",
                        "ralexation_duration_s": 86.66014,
                    },
                    {
                        "time_offset_s": 8.296,
                        "event_type": "agent_left_ralexation",
                        "output_tokens_per_s": 41.773814,
                        "slo_slack_s": 173.320279,
                        "slo_target_tokens_per_s": 25.0,
                        "min_output_tokens_per_s": 27.474132,
                        "avg_output_tokens_per_s": 38.841427,
                        "to_schedule_state": "ongoing",
                        "wake_reason": "slo_recovered",
                        "resume_disposition": "ctx_aware_admitted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_discover_run_dirs_with_gateway_slo_aware_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "job-missing-summary"

    _write_slo_aware_summary(good_run)
    bad_processed_dir = bad_run / "post-processed" / "gateway" / "slo-aware-log"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_gateway_slo_aware(root_dir)

    assert discovered == [good_run.resolve()]


def test_generate_figure_for_run_dir_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    summary_path = _write_slo_aware_summary(run_dir)

    def fake_render_gateway_slo_aware_figure(
        *,
        slo_aware_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del slo_aware_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_gateway_slo_aware_figure",
        fake_render_gateway_slo_aware_figure,
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
        / "gateway-slo-aware"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_slo_aware_summary_path"] == str(summary_path.resolve())
    assert manifest["figure_count"] == 1
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "slo-aware-events-timeline.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["slo_aware_event_count"] == 2
    assert manifest["unique_agent_count"] == 1
    assert manifest["target_output_throughput_tokens_per_s"] == 25.0
    assert manifest["wake_reason_counts"] == {"slo_recovered": 1}
    assert Path(manifest["figure_path"]).is_file()


def test_generate_figure_for_run_dir_rejects_missing_summary_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figure_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing gateway SLO-aware summary file" in str(exc)
    else:
        raise AssertionError("Expected ValueError when summary file is missing")


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_slo_aware_summary(run_a)
    _write_slo_aware_summary(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figure_for_run_dir(
        run_dir: Path,
        *,
        slo_aware_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del slo_aware_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "gateway-slo-aware"
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
