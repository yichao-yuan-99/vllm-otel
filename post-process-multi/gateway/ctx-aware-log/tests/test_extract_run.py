from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "extract_run.py"
MODULE_NAME = "post_process_multi_gateway_ctx_aware_extract_run"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec for {MODULE_PATH}")
extract_run = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = extract_run
SPEC.loader.exec_module(extract_run)


def _write_ctx_aware_log(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_extract_run_generates_single_series_ctx_aware_timeseries(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    log_path = (
        run_dir
        / "gateway-output"
        / "job"
        / "ctx_aware_20260404T153801Z.jsonl"
    )
    _write_ctx_aware_log(
        log_path,
        [
            {
                "timestamp": "2026-04-04T15:38:01.552Z",
                "ongoing_agent_count": 0,
                "pending_agent_count": 0,
                "ongoing_effective_context_tokens": 0,
                "pending_effective_context_tokens": 0,
                "agents_turned_pending_due_to_context_threshold": 0,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 0,
                "new_agents_added_as_ongoing": 0,
            },
            {
                "timestamp": "2026-04-04T15:38:01.752Z",
                "ongoing_agent_count": 1,
                "pending_agent_count": 0,
                "ongoing_effective_context_tokens": 3000,
                "pending_effective_context_tokens": 0,
                "agents_turned_pending_due_to_context_threshold": 0,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 0,
                "new_agents_added_as_ongoing": 1,
            },
            {
                "timestamp": "2026-04-04T15:38:01.952Z",
                "ongoing_agent_count": 1,
                "pending_agent_count": 1,
                "ongoing_effective_context_tokens": 2800,
                "pending_effective_context_tokens": 3000,
                "agents_turned_pending_due_to_context_threshold": 1,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 1,
                "new_agents_added_as_ongoing": 0,
            },
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "ctx-aware-log"
        / extract_run.DEFAULT_OUTPUT_NAME
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["multi_profile"] is False
    assert payload["source_ctx_aware_log_path"] == str(log_path.resolve())
    assert payload["selected_ctx_aware_log_file_name"] == log_path.name
    assert payload["ctx_aware_log_count"] == 1
    assert payload["sample_count"] == 3
    assert payload["duration_s"] == 0.4
    assert payload["avg_sample_interval_s"] == 0.2
    assert payload["started_at"] == "2026-04-04T15:38:01.552Z"
    assert payload["ended_at"] == "2026-04-04T15:38:01.952Z"
    assert [point["second"] for point in payload["samples"]] == [0.0, 0.2, 0.4]
    assert payload["ctx_aware_logs"][0]["source_ctx_aware_log_path"] == str(log_path.resolve())


def test_extract_run_aggregates_multiple_profile_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    log_profile_2 = (
        run_dir
        / "gateway-output"
        / "job"
        / "ctx_aware_20260404T153801Z_profile-2.jsonl"
    )
    log_profile_13 = (
        run_dir
        / "gateway-output"
        / "job"
        / "ctx_aware_20260404T153801Z_profile-13.jsonl"
    )

    _write_ctx_aware_log(
        log_profile_2,
        [
            {
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
                "timestamp": "2026-04-04T15:38:01.200Z",
                "ongoing_agent_count": 0,
                "pending_agent_count": 1,
                "ongoing_effective_context_tokens": 0,
                "pending_effective_context_tokens": 80,
                "agents_turned_pending_due_to_context_threshold": 1,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 0,
                "new_agents_added_as_ongoing": 0,
            },
        ],
    )
    _write_ctx_aware_log(
        log_profile_13,
        [
            {
                "timestamp": "2026-04-04T15:38:01.100Z",
                "ongoing_agent_count": 2,
                "pending_agent_count": 0,
                "ongoing_effective_context_tokens": 200,
                "pending_effective_context_tokens": 0,
                "agents_turned_pending_due_to_context_threshold": 0,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 0,
                "new_agents_added_as_ongoing": 2,
            },
            {
                "timestamp": "2026-04-04T15:38:01.300Z",
                "ongoing_agent_count": 1,
                "pending_agent_count": 1,
                "ongoing_effective_context_tokens": 120,
                "pending_effective_context_tokens": 70,
                "agents_turned_pending_due_to_context_threshold": 1,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 0,
                "new_agents_added_as_ongoing": 0,
            },
        ],
    )

    payload = extract_run.extract_ctx_aware_log_from_run_dir(run_dir)

    assert payload["multi_profile"] is True
    assert payload["source_ctx_aware_log_path"] is None
    assert payload["selected_ctx_aware_log_file_name"] is None
    assert payload["ctx_aware_log_count"] == 2
    assert payload["port_profile_ids"] == [2, 13]
    assert len(payload["ctx_aware_logs"]) == 2
    assert [entry["series_key"] for entry in payload["ctx_aware_logs"]] == ["profile-2", "profile-13"]

    assert payload["sample_count"] == 4
    assert [point["second"] for point in payload["samples"]] == [0.0, 0.1, 0.2, 0.3]
    assert [point["ongoing_agent_count"] for point in payload["samples"]] == [1, 3, 2, 1]
    assert [point["pending_agent_count"] for point in payload["samples"]] == [0, 0, 1, 2]
    assert [
        point["agents_turned_pending_due_to_context_threshold"]
        for point in payload["samples"]
    ] == [0, 0, 1, 1]
    assert [point["new_agents_added_as_ongoing"] for point in payload["samples"]] == [1, 2, 0, 0]


def test_main_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_ctx_aware_log(
        run_a / "gateway-output" / "job" / "ctx_aware_20260404T153801Z_profile-2.jsonl",
        [],
    )
    _write_ctx_aware_log(
        run_b / "gateway-output" / "job" / "ctx_aware_20260404T153901Z_profile-13.jsonl",
        [],
    )

    processed: list[tuple[Path, Path | None, Path | None]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        ctx_aware_log_path: Path | None = None,
    ) -> Path:
        processed.append((run_dir, output_path, ctx_aware_log_path))
        return (
            run_dir
            / "post-processed"
            / "gateway"
            / "ctx-aware-log"
            / extract_run.DEFAULT_OUTPUT_NAME
        )

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, None),
        (run_b.resolve(), None, None),
    ]
