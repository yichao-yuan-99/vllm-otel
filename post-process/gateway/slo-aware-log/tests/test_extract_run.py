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
    started_at: str = "2026-04-08T19:58:00Z",
    finished_at: str = "2026-04-08T20:00:00Z",
    time_constraint_s: float = 300.0,
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


def _write_slo_aware_log(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_extract_run_generates_slo_aware_event_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)
    log_path = (
        run_dir
        / "gateway-output"
        / "job"
        / "slo_aware_decisions_20260408T195807Z.jsonl"
    )
    _write_slo_aware_log(
        log_path,
        [
            {
                "timestamp": "2026-04-08T19:58:07.014Z",
                "event_type": "agent_entered_ralexation",
                "api_token_hash": "agent-a",
                "trace_id": "trace-a",
                "schedule_state": "ralexation",
                "context_tokens": 23004,
                "effective_context_tokens": 23004,
                "output_tokens_per_s": 41.773814,
                "slo_slack_s": 173.320279,
                "slo_target_tokens_per_s": 25.0,
                "min_output_tokens_per_s": 19.088245,
                "avg_output_tokens_per_s": 38.009882,
                "from_schedule_state": "ongoing",
                "to_schedule_state": "ralexation",
                "policy_mode": "push-back-half-slack",
                "ralexation_duration_s": 86.66014,
                "ralexation_until": "2026-04-08T19:59:33.674Z",
            },
            {
                "timestamp": "2026-04-08T19:58:08.296Z",
                "event_type": "agent_left_ralexation",
                "api_token_hash": "agent-a",
                "trace_id": "trace-a",
                "schedule_state": "ongoing",
                "context_tokens": 23004,
                "effective_context_tokens": 23004,
                "output_tokens_per_s": 41.773814,
                "slo_slack_s": 173.320279,
                "slo_target_tokens_per_s": 25.0,
                "min_output_tokens_per_s": 27.474132,
                "avg_output_tokens_per_s": 38.841427,
                "wake_reason": "slo_recovered",
                "from_schedule_state": "ralexation",
                "to_schedule_state": "ongoing",
                "resume_disposition": "ctx_aware_admitted",
                "ralexation_until": None,
            },
            {
                "timestamp": "2026-04-08T20:00:10.000Z",
                "event_type": "agent_entered_ralexation",
                "api_token_hash": "agent-b",
                "trace_id": "trace-b",
                "output_tokens_per_s": 40.0,
            },
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "slo-aware-log"
        / extract_run.DEFAULT_OUTPUT_NAME
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["source_slo_aware_log_paths"] == [str(log_path.resolve())]
    assert payload["slo_aware_log_found"] is True
    assert payload["slo_aware_event_count"] == 2
    assert payload["unique_agent_count"] == 1
    assert payload["first_slo_aware_event_time_offset_s"] == 7.014
    assert payload["target_output_throughput_tokens_per_s"] == 25.0
    assert payload["event_type_counts"] == {
        "agent_entered_ralexation": 1,
        "agent_left_ralexation": 1,
    }
    assert payload["wake_reason_counts"] == {"slo_recovered": 1}
    assert payload["resume_disposition_counts"] == {"ctx_aware_admitted": 1}
    assert payload["min_output_tokens_per_s_at_events"] == 41.773814
    assert payload["max_output_tokens_per_s_at_events"] == 41.773814
    assert payload["min_slo_slack_s"] == 173.320279
    assert payload["max_ralexation_duration_s"] == 86.66014
    assert payload["events"][0]["policy_mode"] == "push-back-half-slack"
    assert payload["events"][0]["ralexation_until_utc"] == "2026-04-08T19:59:33.674000Z"
    assert payload["events"][1]["wake_reason"] == "slo_recovered"
    assert payload["events"][1]["resume_disposition"] == "ctx_aware_admitted"


def test_extract_run_succeeds_when_slo_aware_logs_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_replay_summary(run_dir)

    payload = extract_run.extract_slo_aware_log_from_run_dir(run_dir)

    assert payload["slo_aware_log_found"] is False
    assert payload["slo_aware_event_count"] == 0
    assert payload["unique_agent_count"] == 0
    assert payload["events"] == []


def test_discover_run_dirs_with_slo_aware_log_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    _write_slo_aware_log(
        run_a / "gateway-output" / "job" / "slo_aware_decisions_20260408T195807Z.jsonl",
        [],
    )
    _write_slo_aware_log(
        run_b / "gateway-output" / "job" / "slo_aware_decisions_20260408T195907Z.jsonl",
        [],
    )

    discovered = extract_run.discover_run_dirs_with_slo_aware_log(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_slo_aware_log(
        run_a / "gateway-output" / "job" / "slo_aware_decisions_20260408T195807Z.jsonl",
        [],
    )
    _write_slo_aware_log(
        run_b / "gateway-output" / "job" / "slo_aware_decisions_20260408T195907Z.jsonl",
        [],
    )

    processed: list[tuple[Path, Path | None, Path | None]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        slo_aware_log_path: Path | None = None,
    ) -> Path:
        processed.append((run_dir, output_path, slo_aware_log_path))
        return (
            run_dir
            / "post-processed"
            / "gateway"
            / "slo-aware-log"
            / extract_run.DEFAULT_OUTPUT_NAME
        )

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, None),
        (run_b.resolve(), None, None),
    ]


def test_extract_run_rejects_invalid_option_combinations(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")

    try:
        extract_run.main(["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")])
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")

    try:
        extract_run.main(
            [
                "--root-dir",
                str(root_dir),
                "--slo-aware-log",
                str(tmp_path / "slo_aware_decisions.jsonl"),
            ]
        )
    except ValueError as exc:
        assert "--slo-aware-log can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when --slo-aware-log is used with --root-dir"
        )

    try:
        extract_run.main(["--root-dir", str(root_dir), "--max-procs", "0"])
    except ValueError as exc:
        assert "--max-procs must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --max-procs <= 0")
