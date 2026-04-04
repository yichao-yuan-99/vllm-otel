from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_all_figures


def _write_freq_control_summary(
    run_dir: Path,
    *,
    summary_dir_name: str = "freq-control",
    segmented_policy: bool = False,
) -> Path:
    processed_dir = run_dir / "post-processed" / summary_dir_name
    processed_dir.mkdir(parents=True)
    summary_path = processed_dir / "freq-control-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_type": "replay",
                "freq_control_log_found": True,
                "query_log_found": True,
                "decision_log_found": True,
                "query_point_count": 3,
                "pending_query_point_count": 1,
                "active_query_point_count": 2,
                "query_error_count": 2,
                "control_error_log_found": True,
                "control_error_point_count": 2,
                "decision_point_count": 2,
                "decision_change_count": 1,
                "first_job_active_time_offset_s": 0.5,
                "lower_bound": 100.0,
                "upper_bound": 200.0,
                "segmented_policy_detected": segmented_policy,
                "low_freq_threshold": 75.0 if segmented_policy else None,
                "low_freq_cap_mhz": 900 if segmented_policy else None,
                "max_context_usage": 180.0,
                "max_window_context_usage": 170.0,
                "min_frequency_mhz": 900,
                "max_frequency_mhz": 1005,
                "analysis_window_start_utc": "2026-04-02T00:00:00Z",
                "query_points": [
                    {
                        "time_offset_s": -0.5,
                        "context_usage": 0.0,
                        "phase": "pending",
                        "job_active": False,
                        "agent_count": 0,
                    },
                    {
                        "time_offset_s": 0.5,
                        "context_usage": 120.0,
                        "phase": "active",
                        "job_active": True,
                        "agent_count": 1,
                        "error": "Connection reset by peer",
                    },
                    {
                        "time_offset_s": 1.5,
                        "context_usage": 180.0,
                        "phase": "active",
                        "job_active": True,
                        "agent_count": 1,
                        "error": "Connection reset by peer",
                    },
                ],
                "decision_points": [
                    {
                        "time_offset_s": 5.0,
                        "window_context_usage": 140.0,
                        "current_frequency_mhz": 900,
                        "target_frequency_mhz": 1005,
                        "changed": True,
                        "low_freq_threshold": 75.0 if segmented_policy else None,
                        "low_freq_cap_mhz": 900 if segmented_policy else None,
                    },
                    {
                        "time_offset_s": 10.0,
                        "window_context_usage": 170.0,
                        "current_frequency_mhz": 1005,
                        "target_frequency_mhz": 1005,
                        "changed": False,
                        "low_freq_threshold": 75.0 if segmented_policy else None,
                        "low_freq_cap_mhz": 900 if segmented_policy else None,
                    },
                ],
                "control_error_points": [
                    {
                        "time_offset_s": 5.5,
                        "reason": "control_decision",
                        "action": "increase",
                        "error": "[Errno 104] Connection reset by peer",
                    },
                    {
                        "time_offset_s": 7.5,
                        "reason": "control_decision",
                        "action": "increase",
                        "error": "[Errno 104] Connection reset by peer",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_discover_run_dirs_with_freq_control_summary_scans_recursively(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    good_run_seg = root_dir / "b" / "job-ok-seg"
    bad_run = root_dir / "b" / "job-missing-summary"

    _write_freq_control_summary(good_run)
    _write_freq_control_summary(
        good_run_seg,
        summary_dir_name="freq-control-seg",
        segmented_policy=True,
    )
    bad_processed_dir = bad_run / "post-processed" / "freq-control"
    bad_processed_dir.mkdir(parents=True)

    discovered = generate_all_figures.discover_run_dirs_with_freq_control_summary(
        root_dir
    )

    assert discovered == [good_run.resolve(), good_run_seg.resolve()]


def test_generate_figure_for_run_dir_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    _write_freq_control_summary(run_dir)

    def fake_render_freq_control_figure(
        *,
        freq_control_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del freq_control_payload, image_format, dpi
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_freq_control_figure",
        fake_render_freq_control_figure,
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
        / "freq-control"
        / "figures-manifest.json"
    ).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 1
    assert manifest["figure_generated"] is True
    assert manifest["figure_file_name"] == "freq-control-timeline.png"
    assert manifest["image_format"] == "png"
    assert manifest["dpi"] == 150
    assert manifest["query_point_count"] == 3
    assert manifest["query_error_count"] == 2
    assert manifest["control_error_log_found"] is True
    assert manifest["control_error_point_count"] == 2
    assert manifest["decision_point_count"] == 2
    assert manifest["decision_change_count"] == 1
    assert manifest["max_context_usage"] == 180.0
    assert manifest["max_window_context_usage"] == 170.0
    assert manifest["min_frequency_mhz"] == 900
    assert manifest["max_frequency_mhz"] == 1005
    assert Path(manifest["figure_path"]).is_file()


def test_generate_figure_for_run_dir_uses_segmented_summary_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    summary_path = _write_freq_control_summary(
        run_dir,
        summary_dir_name="freq-control-seg",
        segmented_policy=True,
    )
    captured_payloads: list[dict[str, object]] = []

    def fake_render_freq_control_figure(
        *,
        freq_control_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del image_format, dpi
        captured_payloads.append(freq_control_payload)
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_freq_control_figure",
        fake_render_freq_control_figure,
    )

    manifest_path = generate_all_figures.generate_figure_for_run_dir(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == (
        run_dir
        / "post-processed"
        / "visualization"
        / "freq-control-seg"
        / "figures-manifest.json"
    ).resolve()
    assert manifest["source_freq_control_summary_path"] == str(summary_path.resolve())
    assert manifest["output_dir"] == str(
        (
            run_dir
            / "post-processed"
            / "visualization"
            / "freq-control-seg"
        ).resolve()
    )
    assert manifest["segmented_policy_detected"] is True
    assert manifest["low_freq_threshold"] == 75.0
    assert manifest["low_freq_cap_mhz"] == 900
    assert captured_payloads[0]["low_freq_threshold"] == 75.0
    assert captured_payloads[0]["low_freq_cap_mhz"] == 900


def test_generate_figure_for_run_dir_uses_linespace_summary_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "job"
    summary_path = _write_freq_control_summary(
        run_dir,
        summary_dir_name="freq-control-linespace",
    )
    captured_payloads: list[dict[str, object]] = []

    def fake_render_freq_control_figure(
        *,
        freq_control_payload: dict[str, object],
        output_path: Path,
        image_format: str,
        dpi: int,
    ) -> bool:
        del image_format, dpi
        captured_payloads.append(freq_control_payload)
        output_path.write_text("fake-image", encoding="utf-8")
        return True

    monkeypatch.setattr(
        generate_all_figures,
        "_render_freq_control_figure",
        fake_render_freq_control_figure,
    )

    manifest_path = generate_all_figures.generate_figure_for_run_dir(run_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == (
        run_dir
        / "post-processed"
        / "visualization"
        / "freq-control-linespace"
        / "figures-manifest.json"
    ).resolve()
    assert manifest["source_freq_control_summary_path"] == str(summary_path.resolve())
    assert manifest["output_dir"] == str(
        (
            run_dir
            / "post-processed"
            / "visualization"
            / "freq-control-linespace"
        ).resolve()
    )
    assert captured_payloads[0]["source_run_dir"] == str(run_dir)


def test_generate_figure_for_run_dir_rejects_missing_freq_control_summary_file(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        generate_all_figures.generate_figure_for_run_dir(run_dir)
    except ValueError as exc:
        assert "Missing freq-control summary file" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when freq-control summary file is missing"
        )


def test_extract_query_error_series_groups_and_positions_events() -> None:
    payload = {
        "query_points": [
            {
                "time_offset_s": -1.0,
                "phase": "pending",
                "error": "socket closed unexpectedly after partial response",
            },
            {
                "time_offset_s": 2.0,
                "phase": "active",
                "error": "Connection reset by peer",
            },
            {
                "time_offset_s": 4.0,
                "phase": "active",
                "error": "Connection reset by peer",
            },
        ]
    }

    labels, pending_x, pending_y, active_x, active_y = (
        generate_all_figures._extract_query_error_series(payload)
    )

    assert labels == [
        "socket closed unexpectedly after partial response",
        "Connection reset by peer (2)",
    ]
    assert pending_x == [-1.0]
    assert pending_y == [0]
    assert active_x == [2.0, 4.0]
    assert active_y == [1, 1]


def test_extract_control_error_series_groups_and_positions_events() -> None:
    payload = {
        "control_error_points": [
            {
                "time_offset_s": 3.0,
                "reason": "control_decision",
                "action": "increase",
                "error": "[Errno 104] Connection reset by peer",
            },
            {
                "time_offset_s": 5.0,
                "reason": "control_decision",
                "action": "increase",
                "error": "[Errno 104] Connection reset by peer",
            },
        ]
    }

    labels, control_x, control_y = generate_all_figures._extract_control_error_series(
        payload
    )

    assert len(labels) == 1
    assert labels[0].startswith("control_decision increase: [Errno 104] Connection res")
    assert labels[0].endswith("(2)")
    assert control_x == [3.0, 5.0]
    assert control_y == [0, 0]


def test_extract_combined_failure_series_groups_query_and_control_events() -> None:
    payload = {
        "query_points": [
            {
                "time_offset_s": -1.0,
                "phase": "pending",
                "error": "socket closed unexpectedly after partial response",
            },
            {
                "time_offset_s": 2.0,
                "phase": "active",
                "error": "Connection reset by peer",
            },
        ],
        "control_error_points": [
            {
                "time_offset_s": 3.5,
                "reason": "control_decision",
                "action": "decrease",
                "error": "[Errno 104] Connection reset by peer",
            }
        ],
    }

    (
        labels,
        pending_x,
        pending_y,
        active_x,
        active_y,
        control_x,
        control_y,
    ) = generate_all_figures._extract_combined_failure_series(payload)

    assert labels[:2] == [
        "socket closed unexpectedly after partial response",
        "Connection reset by peer",
    ]
    assert labels[2].startswith("control_decision decrease: [Errno 104] Connection res")
    assert pending_x == [-1.0]
    assert pending_y == [0]
    assert active_x == [2.0]
    assert active_y == [1]
    assert control_x == [3.5]
    assert control_y == [2]


def test_extract_segmented_policy_markers_uses_decision_points_when_needed() -> None:
    low_freq_threshold, low_freq_cap_mhz, segmented_policy_detected = (
        generate_all_figures._extract_segmented_policy_markers(
            {
                "decision_points": [
                    {
                        "time_offset_s": 1.0,
                        "low_freq_threshold": 88.0,
                        "low_freq_cap_mhz": 810,
                    }
                ]
            }
        )
    )

    assert low_freq_threshold == 88.0
    assert low_freq_cap_mhz == 810
    assert segmented_policy_detected is True


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_freq_control_summary(run_a)
    _write_freq_control_summary(run_b)

    processed: list[tuple[Path, str, int]] = []

    def fake_generate_figure_for_run_dir(
        run_dir: Path,
        *,
        freq_control_input_path: Path | None = None,
        output_dir: Path | None = None,
        image_format: str = "png",
        dpi: int = 220,
    ) -> Path:
        del freq_control_input_path, output_dir
        processed.append((run_dir, image_format, dpi))
        return (
            run_dir
            / "post-processed"
            / "visualization"
            / "freq-control"
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
