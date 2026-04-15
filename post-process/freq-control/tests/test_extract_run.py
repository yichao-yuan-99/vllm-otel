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


def _write_query_log(
    run_dir: Path,
    *,
    log_dir_name: str = "freq-control",
    prefix: str = "freq-controller",
) -> Path:
    log_dir = run_dir / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}.query.20260402T000000Z.jsonl"
    records = [
        {
            "timestamp": "2026-04-01T23:59:59.500Z",
            "phase": "pending",
            "context_usage": 0.0,
            "job_active": False,
            "agent_count": 0,
        },
        {
            "timestamp": "2026-04-02T00:00:00.500Z",
            "phase": "active",
            "context_usage": 120.0,
            "job_active": True,
            "agent_count": 1,
            "sample_count_window": 1,
        },
        {
            "timestamp": "2026-04-02T00:00:01.500Z",
            "phase": "active",
            "error": "gateway timeout",
            "job_active": True,
            "agent_count": 1,
        },
        {
            "timestamp": "2026-04-02T00:01:30.000Z",
            "phase": "active",
            "context_usage": 999.0,
            "job_active": True,
            "agent_count": 1,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _write_decision_log(
    run_dir: Path,
    *,
    log_dir_name: str = "freq-control",
    prefix: str = "freq-controller",
    segmented_policy: bool = False,
    linespace_policy: bool = False,
) -> Path:
    log_dir = run_dir / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}.decision.20260402T000000Z.jsonl"
    records = [
        {
            "timestamp": "2026-04-02T00:00:05.000Z",
            "action": "increase",
            "changed": True,
            "current_frequency_mhz": 900,
            "target_frequency_mhz": 1005,
            "window_context_usage": 140.0,
            "sample_count": 5,
            **(
                {
                    "target_context_usage_threshold": 300.0,
                    "segment_count": 3,
                    "segment_width_context_usage": 100.0,
                    "target_frequency_index": 1,
                }
                if linespace_policy
                else {
                    "lower_bound": 100.0,
                    "upper_bound": 200.0,
                }
            ),
            **(
                {
                    "low_freq_threshold": 75.0,
                    "low_freq_cap_mhz": 900,
                    "effective_min_frequency_mhz": 900,
                }
                if segmented_policy
                else {}
            ),
        },
        {
            "timestamp": "2026-04-02T00:00:10.000Z",
            "action": "hold",
            "changed": False,
            "current_frequency_mhz": 1005,
            "target_frequency_mhz": 1005,
            "window_context_usage": 160.0,
            "sample_count": 5,
            **(
                {
                    "target_context_usage_threshold": 300.0,
                    "segment_count": 3,
                    "segment_width_context_usage": 100.0,
                    "target_frequency_index": 1,
                }
                if linespace_policy
                else {
                    "lower_bound": 100.0,
                    "upper_bound": 200.0,
                }
            ),
            **(
                {
                    "low_freq_threshold": 75.0,
                    "low_freq_cap_mhz": 900,
                    "effective_min_frequency_mhz": 900,
                }
                if segmented_policy
                else {}
            ),
        },
        {
            "timestamp": "2026-04-02T00:01:10.000Z",
            "action": "decrease",
            "changed": True,
            "current_frequency_mhz": 1005,
            "target_frequency_mhz": 900,
            "window_context_usage": 80.0,
            "sample_count": 5,
            **(
                {
                    "target_context_usage_threshold": 300.0,
                    "segment_count": 3,
                    "segment_width_context_usage": 100.0,
                    "target_frequency_index": 0,
                }
                if linespace_policy
                else {
                    "lower_bound": 100.0,
                    "upper_bound": 200.0,
                }
            ),
            **(
                {
                    "low_freq_threshold": 75.0,
                    "low_freq_cap_mhz": 900,
                    "effective_min_frequency_mhz": 900,
                }
                if segmented_policy
                else {}
            ),
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _write_control_error_log(
    run_dir: Path,
    *,
    log_dir_name: str = "freq-control",
    prefix: str = "freq-controller",
) -> Path:
    log_dir = run_dir / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}.control-error.20260402T000000Z.jsonl"
    records = [
        {
            "timestamp": "2026-04-02T00:00:06.000Z",
            "reason": "control_decision",
            "action": "increase",
            "error": "[Errno 104] Connection reset by peer",
            "attempted_frequency_index": 2,
            "attempted_frequency_mhz": 1005,
            "current_frequency_index": 1,
            "current_frequency_mhz": 900,
            "moving_average_context_usage": 141.0,
            "sample_count": 5,
        },
        {
            "timestamp": "2026-04-02T00:01:10.000Z",
            "reason": "control_loop",
            "action": "error",
            "error": "unexpected failure while computing next step",
            "attempted_frequency_index": 1,
            "attempted_frequency_mhz": 900,
            "current_frequency_index": 1,
            "current_frequency_mhz": 900,
            "moving_average_context_usage": 99.0,
            "sample_count": 3,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_extract_run_dir_writes_freq_control_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(run_dir)
    decision_log_path = _write_decision_log(run_dir)
    control_error_log_path = _write_control_error_log(run_dir)

    output_path = extract_run.extract_run_dir(run_dir)

    assert output_path == (
        run_dir / "post-processed" / "freq-control" / "freq-control-summary.json"
    ).resolve()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_run_dir"] == str(run_dir.resolve())
    assert payload["source_type"] == "replay"
    assert payload["source_freq_control_log_dir_name"] == "freq-control"
    assert payload["query_log_found"] is True
    assert payload["decision_log_found"] is True
    assert payload["control_error_log_found"] is True
    assert payload["freq_control_log_found"] is True
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["source_control_error_log_paths"] == [
        str(control_error_log_path.resolve())
    ]
    assert payload["query_point_count"] == 3
    assert payload["pending_query_point_count"] == 1
    assert payload["active_query_point_count"] == 2
    assert payload["query_error_count"] == 1
    assert payload["control_error_point_count"] == 1
    assert payload["decision_point_count"] == 2
    assert payload["decision_change_count"] == 1
    assert payload["first_job_active_time_offset_s"] == 0.5
    assert payload["first_control_error_time_offset_s"] == 6.0
    assert payload["lower_bound"] == 100.0
    assert payload["upper_bound"] == 200.0
    assert payload["max_context_usage"] == 120.0
    assert payload["max_window_context_usage"] == 160.0
    assert payload["min_frequency_mhz"] == 900
    assert payload["max_frequency_mhz"] == 1005
    assert payload["segmented_policy_detected"] is False
    assert payload["low_freq_threshold"] is None
    assert payload["low_freq_cap_mhz"] is None

    assert payload["query_points"][0]["time_offset_s"] == -0.5
    assert payload["query_points"][0]["phase"] == "pending"
    assert payload["query_points"][1]["time_offset_s"] == 0.5
    assert payload["query_points"][1]["context_usage"] == 120.0
    assert payload["query_points"][2]["error"] == "gateway timeout"

    assert payload["decision_points"][0]["time_offset_s"] == 5.0
    assert payload["decision_points"][0]["target_frequency_mhz"] == 1005
    assert payload["decision_points"][1]["time_offset_s"] == 10.0
    assert payload["decision_points"][1]["changed"] is False
    assert payload["decision_points"][1]["low_freq_threshold"] is None
    assert payload["control_error_points"][0]["time_offset_s"] == 6.0
    assert payload["control_error_points"][0]["reason"] == "control_decision"
    assert payload["control_error_points"][0]["action"] == "increase"
    assert (
        payload["control_error_points"][0]["error"]
        == "[Errno 104] Connection reset by peer"
    )


def test_extract_run_dir_succeeds_when_freq_control_logs_are_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir / "post-processed" / "freq-control" / "freq-control-summary.json"
    ).resolve()
    assert payload["freq_control_log_found"] is False
    assert payload["query_log_found"] is False
    assert payload["decision_log_found"] is False
    assert payload["control_error_log_found"] is False
    assert payload["query_point_count"] == 0
    assert payload["control_error_point_count"] == 0
    assert payload["decision_point_count"] == 0
    assert payload["query_points"] == []
    assert payload["decision_points"] == []
    assert payload["control_error_points"] == []


def test_extract_run_dir_uses_segmented_layout_and_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(run_dir, log_dir_name="freq-control-seg")
    decision_log_path = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-seg",
        segmented_policy=True,
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir / "post-processed" / "freq-control-seg" / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-seg"
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["segmented_policy_detected"] is True
    assert payload["low_freq_threshold"] == 75.0
    assert payload["low_freq_cap_mhz"] == 900
    assert payload["min_effective_min_frequency_mhz"] == 900
    assert payload["max_effective_min_frequency_mhz"] == 900
    assert payload["decision_points"][0]["low_freq_threshold"] == 75.0
    assert payload["decision_points"][0]["low_freq_cap_mhz"] == 900
    assert payload["decision_points"][0]["effective_min_frequency_mhz"] == 900


def test_extract_run_dir_accepts_linespace_log_prefix_and_fields(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(run_dir, prefix="freq-controller-ls")
    decision_log_path = _write_decision_log(
        run_dir,
        prefix="freq-controller-ls",
        linespace_policy=True,
    )
    control_error_log_path = _write_control_error_log(
        run_dir,
        prefix="freq-controller-ls",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace"
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["source_control_error_log_paths"] == [
        str(control_error_log_path.resolve())
    ]
    assert payload["linespace_policy_detected"] is True
    assert payload["segmented_policy_detected"] is False
    assert payload["lower_bound"] is None
    assert payload["upper_bound"] is None
    assert payload["target_context_usage_threshold"] == 300.0
    assert payload["segment_count"] == 3
    assert payload["segment_width_context_usage"] == 100.0
    assert payload["decision_points"][0]["target_context_usage_threshold"] == 300.0
    assert payload["decision_points"][0]["segment_count"] == 3
    assert payload["decision_points"][0]["segment_width_context_usage"] == 100.0
    assert payload["decision_points"][0]["target_frequency_index"] == 1


def test_extract_run_dir_accepts_instance_linespace_layout_and_prefix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(
        run_dir,
        log_dir_name="freq-control-linespace-instance",
        prefix="freq-controller-ls-instance",
    )
    decision_log_path = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-linespace-instance",
        prefix="freq-controller-ls-instance",
        linespace_policy=True,
    )
    control_error_log_path = _write_control_error_log(
        run_dir,
        log_dir_name="freq-control-linespace-instance",
        prefix="freq-controller-ls-instance",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-instance"
        / "freq-control-summary.json"
    ).resolve()
    assert (
        payload["source_freq_control_log_dir_name"]
        == "freq-control-linespace-instance"
    )
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["source_control_error_log_paths"] == [
        str(control_error_log_path.resolve())
    ]
    assert payload["linespace_policy_detected"] is True
    assert payload["segmented_policy_detected"] is False
    assert payload["target_context_usage_threshold"] == 300.0
    assert payload["segment_count"] == 3
    assert payload["segment_width_context_usage"] == 100.0
    assert payload["decision_points"][0]["target_frequency_index"] == 1


def test_extract_run_dir_accepts_instance_slo_linespace_layout_and_prefix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(
        run_dir,
        log_dir_name="freq-control-linespace-instance-slo",
        prefix="freq-controller-ls-instance-slo",
    )
    decision_log_path = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-linespace-instance-slo",
        prefix="freq-controller-ls-instance-slo",
        linespace_policy=True,
    )
    control_error_log_path = _write_control_error_log(
        run_dir,
        log_dir_name="freq-control-linespace-instance-slo",
        prefix="freq-controller-ls-instance-slo",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-instance-slo"
        / "freq-control-summary.json"
    ).resolve()
    assert (
        payload["source_freq_control_log_dir_name"]
        == "freq-control-linespace-instance-slo"
    )
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["source_control_error_log_paths"] == [
        str(control_error_log_path.resolve())
    ]
    assert payload["linespace_policy_detected"] is True
    assert payload["segmented_policy_detected"] is False
    assert payload["target_context_usage_threshold"] == 300.0
    assert payload["segment_count"] == 3
    assert payload["segment_width_context_usage"] == 100.0
    assert payload["decision_points"][0]["target_frequency_index"] == 1


def test_extract_run_dir_accepts_multi_linespace_layout_and_prefix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(
        run_dir,
        log_dir_name="freq-control-linespace-multi",
        prefix="freq-controller-ls-multi",
    )
    decision_log_path = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-linespace-multi",
        prefix="freq-controller-ls-multi",
        linespace_policy=True,
    )
    control_error_log_path = _write_control_error_log(
        run_dir,
        log_dir_name="freq-control-linespace-multi",
        prefix="freq-controller-ls-multi",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-multi"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace-multi"
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["source_control_error_log_paths"] == [
        str(control_error_log_path.resolve())
    ]
    assert payload["linespace_policy_detected"] is True
    assert payload["segmented_policy_detected"] is False
    assert payload["target_context_usage_threshold"] == 300.0
    assert payload["segment_count"] == 3
    assert payload["segment_width_context_usage"] == 100.0
    assert payload["decision_points"][0]["target_frequency_index"] == 1


def test_extract_run_dir_accepts_amd_linespace_layout_and_prefix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path = _write_query_log(
        run_dir,
        log_dir_name="freq-control-linespace-amd",
        prefix="freq-controller-ls-amd",
    )
    decision_log_path = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-linespace-amd",
        prefix="freq-controller-ls-amd",
        linespace_policy=True,
    )
    control_error_log_path = _write_control_error_log(
        run_dir,
        log_dir_name="freq-control-linespace-amd",
        prefix="freq-controller-ls-amd",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-amd"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace-amd"
    assert payload["source_query_log_paths"] == [str(query_log_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(decision_log_path.resolve())]
    assert payload["source_control_error_log_paths"] == [
        str(control_error_log_path.resolve())
    ]
    assert payload["linespace_policy_detected"] is True
    assert payload["segmented_policy_detected"] is False
    assert payload["target_context_usage_threshold"] == 300.0
    assert payload["segment_count"] == 3
    assert payload["segment_width_context_usage"] == 100.0
    assert payload["decision_points"][0]["target_frequency_index"] == 1


def test_extract_run_dir_accepts_nested_profile_linespace_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    query_log_path_a = _write_query_log(
        run_dir,
        log_dir_name="freq-control-linespace/profile-2",
        prefix="freq-controller-ls",
    )
    query_log_path_b = _write_query_log(
        run_dir,
        log_dir_name="freq-control-linespace/profile-13",
        prefix="freq-controller-ls",
    )
    decision_log_path_a = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-linespace/profile-2",
        prefix="freq-controller-ls",
        linespace_policy=True,
    )
    decision_log_path_b = _write_decision_log(
        run_dir,
        log_dir_name="freq-control-linespace/profile-13",
        prefix="freq-controller-ls",
        linespace_policy=True,
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace"
    assert payload["multi_profile"] is True
    assert payload["port_profile_ids"] == [2, 13]
    assert payload["series_keys"] == ["profile-2", "profile-13"]
    assert set(payload["source_query_log_paths"]) == {
        str(query_log_path_a.resolve()),
        str(query_log_path_b.resolve()),
    }
    assert set(payload["source_decision_log_paths"]) == {
        str(decision_log_path_a.resolve()),
        str(decision_log_path_b.resolve()),
    }
    assert payload["query_log_found"] is True
    assert payload["decision_log_found"] is True
    assert payload["freq_control_log_found"] is True
    assert payload["query_point_count"] == 6
    assert payload["pending_query_point_count"] == 2
    assert payload["active_query_point_count"] == 4
    assert payload["decision_point_count"] == 4
    assert payload["decision_change_count"] == 2
    assert payload["linespace_policy_detected"] is True
    assert payload["target_context_usage_threshold"] == 300.0
    assert payload["segment_count"] == 3
    assert payload["segment_width_context_usage"] == 100.0
    assert {point["port_profile_id"] for point in payload["query_points"]} == {2, 13}
    assert {point["port_profile_id"] for point in payload["decision_points"]} == {2, 13}


def test_extract_run_dir_uses_segmented_output_when_segmented_log_dir_exists_but_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    (run_dir / "freq-control-seg").mkdir(parents=True)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir / "post-processed" / "freq-control-seg" / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-seg"
    assert payload["freq_control_log_found"] is False
    assert payload["segmented_policy_detected"] is True


def test_extract_run_dir_uses_linespace_output_when_linespace_log_dir_exists_but_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    (run_dir / "freq-control-linespace").mkdir(parents=True)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace"
    assert payload["freq_control_log_found"] is False
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_uses_instance_linespace_output_when_instance_log_dir_exists_but_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    (run_dir / "freq-control-linespace-instance").mkdir(parents=True)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-instance"
        / "freq-control-summary.json"
    ).resolve()
    assert (
        payload["source_freq_control_log_dir_name"]
        == "freq-control-linespace-instance"
    )
    assert payload["freq_control_log_found"] is False
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_uses_instance_slo_linespace_output_when_instance_slo_log_dir_exists_but_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    (run_dir / "freq-control-linespace-instance-slo").mkdir(parents=True)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-instance-slo"
        / "freq-control-summary.json"
    ).resolve()
    assert (
        payload["source_freq_control_log_dir_name"]
        == "freq-control-linespace-instance-slo"
    )
    assert payload["freq_control_log_found"] is False
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_uses_multi_linespace_output_when_multi_log_dir_exists_but_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    (run_dir / "freq-control-linespace-multi").mkdir(parents=True)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-multi"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace-multi"
    assert payload["freq_control_log_found"] is False
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_uses_amd_linespace_output_when_amd_log_dir_exists_but_is_empty(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    (run_dir / "freq-control-linespace-amd").mkdir(parents=True)

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == (
        run_dir
        / "post-processed"
        / "freq-control-linespace-amd"
        / "freq-control-summary.json"
    ).resolve()
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace-amd"
    assert payload["freq_control_log_found"] is False
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_accepts_legacy_root_level_freq_control_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    legacy_query_path = run_dir / "freq-controller.query.20260402T000000Z.jsonl"
    legacy_decision_path = run_dir / "freq-controller.decision.20260402T000000Z.jsonl"
    legacy_query_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:00.500Z",
                "phase": "active",
                "context_usage": 10.0,
                "job_active": True,
                "agent_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_decision_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:05.000Z",
                "action": "hold",
                "changed": False,
                "current_frequency_mhz": 900,
                "target_frequency_mhz": 900,
                "window_context_usage": 10.0,
                "sample_count": 1,
                "lower_bound": 100.0,
                "upper_bound": 200.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["query_log_found"] is True
    assert payload["decision_log_found"] is True
    assert payload["source_freq_control_log_dir_name"] == "freq-control"
    assert payload["source_query_log_paths"] == [str(legacy_query_path.resolve())]
    assert payload["source_decision_log_paths"] == [str(legacy_decision_path.resolve())]


def test_extract_run_dir_accepts_legacy_root_level_control_error_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    legacy_control_error_path = (
        run_dir / "freq-controller-ls.control-error.20260402T000000Z.jsonl"
    )
    legacy_control_error_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:05.000Z",
                "reason": "control_decision",
                "action": "decrease",
                "error": "[Errno 104] Connection reset by peer",
                "attempted_frequency_index": 0,
                "attempted_frequency_mhz": 600,
                "current_frequency_index": 2,
                "current_frequency_mhz": 1200,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["control_error_log_found"] is True
    assert payload["control_error_point_count"] == 1
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace"
    assert payload["source_control_error_log_paths"] == [
        str(legacy_control_error_path.resolve())
    ]
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_accepts_legacy_root_level_multi_linespace_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    legacy_query_path = run_dir / "freq-controller-ls-multi.query.20260402T000000Z.jsonl"
    legacy_query_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:00.500Z",
                "phase": "active",
                "context_usage": 10.0,
                "job_active": True,
                "agent_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["query_log_found"] is True
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace-multi"
    assert payload["source_query_log_paths"] == [str(legacy_query_path.resolve())]
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_accepts_legacy_root_level_instance_linespace_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    legacy_query_path = (
        run_dir / "freq-controller-ls-instance.query.20260402T000000Z.jsonl"
    )
    legacy_query_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:00.500Z",
                "phase": "active",
                "context_usage": 10.0,
                "job_active": True,
                "agent_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["query_log_found"] is True
    assert (
        payload["source_freq_control_log_dir_name"]
        == "freq-control-linespace-instance"
    )
    assert payload["source_query_log_paths"] == [str(legacy_query_path.resolve())]
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_accepts_legacy_root_level_instance_slo_linespace_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    legacy_query_path = (
        run_dir / "freq-controller-ls-instance-slo.query.20260402T000000Z.jsonl"
    )
    legacy_query_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:00.500Z",
                "phase": "active",
                "context_usage": 10.0,
                "job_active": True,
                "agent_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["query_log_found"] is True
    assert (
        payload["source_freq_control_log_dir_name"]
        == "freq-control-linespace-instance-slo"
    )
    assert payload["source_query_log_paths"] == [str(legacy_query_path.resolve())]
    assert payload["linespace_policy_detected"] is True


def test_extract_run_dir_accepts_legacy_root_level_amd_linespace_logs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    legacy_query_path = run_dir / "freq-controller-ls-amd.query.20260402T000000Z.jsonl"
    legacy_query_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-02T00:00:00.500Z",
                "phase": "active",
                "context_usage": 10.0,
                "job_active": True,
                "agent_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = extract_run.extract_run_dir(run_dir)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["query_log_found"] is True
    assert payload["source_freq_control_log_dir_name"] == "freq-control-linespace-amd"
    assert payload["source_query_log_paths"] == [str(legacy_query_path.resolve())]
    assert payload["linespace_policy_detected"] is True


def test_discover_run_dirs_with_freq_control_sources_scans_recursively(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    good_run = root_dir / "a" / "job-ok"
    bad_run = root_dir / "b" / "not-a-run"

    _write_replay_summary(good_run)
    (bad_run / "some-dir").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_freq_control_sources(root_dir)

    assert discovered == [good_run.resolve()]
