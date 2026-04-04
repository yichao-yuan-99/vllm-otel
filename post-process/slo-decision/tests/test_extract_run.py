from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_replay_summary(
    run_dir: Path,
    *,
    started_at: str = "2026-04-02T00:00:00Z",
    finished_at: str = "2026-04-02T00:01:00Z",
    time_constraint_s: float = 120.0,
) -> None:
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "time_constraint_s": time_constraint_s,
                "worker_results": {},
            }
        ),
        encoding="utf-8",
    )


def _write_slo_decision_log(
    run_dir: Path,
    *,
    log_dir_name: str = "freq-control-linespace",
) -> Path:
    log_dir = run_dir / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "freq-controller-ls.slo-decision.20260402T000000Z.jsonl"
    records = [
        {
            "timestamp": "2026-04-02T00:00:05.000Z",
            "action": "increase_for_slo",
            "changed": True,
            "decision_policy": "throughput_slo_precedence",
            "slo_override_applied": True,
            "current_frequency_mhz": 1200,
            "target_frequency_mhz": 1500,
            "target_frequency_index": 3,
            "context_target_frequency_index": 0,
            "window_context_usage": 50.0,
            "window_min_output_tokens_per_s": 4.5,
            "sample_count": 5,
            "throughput_sample_count": 5,
            "target_context_usage_threshold": 100.0,
            "target_output_throughput_tokens_per_s": 8.0,
        },
        {
            "timestamp": "2026-04-02T00:00:10.000Z",
            "action": "increase_for_slo",
            "changed": False,
            "decision_policy": "throughput_slo_precedence",
            "slo_override_applied": True,
            "current_frequency_mhz": 1500,
            "target_frequency_mhz": 1500,
            "target_frequency_index": 3,
            "context_target_frequency_index": 0,
            "window_context_usage": 55.0,
            "window_min_output_tokens_per_s": 4.0,
            "sample_count": 5,
            "throughput_sample_count": 5,
            "target_context_usage_threshold": 100.0,
            "target_output_throughput_tokens_per_s": 8.0,
        },
        {
            "timestamp": "2026-04-02T00:01:10.000Z",
            "action": "increase_for_slo",
            "changed": True,
            "decision_policy": "throughput_slo_precedence",
            "slo_override_applied": True,
            "current_frequency_mhz": 900,
            "target_frequency_mhz": 1200,
            "window_min_output_tokens_per_s": 3.5,
            "target_output_throughput_tokens_per_s": 8.0,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_extract_run_dir_writes_slo_decision_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    log_path = _write_slo_decision_log(run_dir)

    output_path = extract_run.extract_run_dir(run_dir)

    assert output_path == (
        run_dir / "post-processed" / "slo-decision" / "slo-decision-summary.json"
    ).resolve()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_run_dir"] == str(run_dir.resolve())
    assert payload["source_type"] == "replay"
    assert payload["source_slo_decision_log_dir_name"] == "freq-control-linespace"
    assert payload["source_slo_decision_log_paths"] == [str(log_path.resolve())]
    assert payload["slo_decision_log_found"] is True
    assert payload["slo_decision_point_count"] == 2
    assert payload["slo_decision_change_count"] == 1
    assert payload["first_slo_decision_time_offset_s"] == 5.0
    assert payload["target_output_throughput_tokens_per_s"] == 8.0
    assert payload["min_window_min_output_tokens_per_s"] == 4.0
    assert payload["max_window_min_output_tokens_per_s"] == 4.5
    assert payload["min_frequency_mhz"] == 1200
    assert payload["max_frequency_mhz"] == 1500

    assert payload["decision_points"][0]["time_offset_s"] == 5.0
    assert payload["decision_points"][0]["action"] == "increase_for_slo"
    assert payload["decision_points"][0]["changed"] is True
    assert payload["decision_points"][0]["window_min_output_tokens_per_s"] == 4.5
    assert payload["decision_points"][1]["time_offset_s"] == 10.0
    assert payload["decision_points"][1]["changed"] is False


def test_extract_run_dir_succeeds_when_slo_decision_logs_are_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir / "post-processed" / "slo-decision" / "slo-decision-summary.json"
    ).resolve()
    assert payload["slo_decision_log_found"] is False
    assert payload["slo_decision_point_count"] == 0
    assert payload["slo_decision_change_count"] == 0
    assert payload["decision_points"] == []


def test_extract_run_dir_accepts_legacy_root_level_slo_decision_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    log_path = run_dir / "freq-controller-ls.slo-decision.20260402T000000Z.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:05.000Z",
                "action": "increase_for_slo",
                "changed": True,
                "decision_policy": "throughput_slo_precedence",
                "slo_override_applied": True,
                "current_frequency_mhz": 900,
                "target_frequency_mhz": 1005,
                "window_min_output_tokens_per_s": 4.0,
                "target_output_throughput_tokens_per_s": 8.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert payload["slo_decision_log_found"] is True
    assert payload["source_slo_decision_log_paths"] == [str(log_path.resolve())]
    assert payload["source_slo_decision_log_dir_name"] == "freq-control-linespace"


def test_discover_run_dirs_with_slo_decision_sources_scans_recursively(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "not-a-run"

    _write_replay_summary(good_run)
    (bad_run / "some-dir").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_slo_decision_sources(root_dir)

    assert discovered == [good_run.resolve()]
