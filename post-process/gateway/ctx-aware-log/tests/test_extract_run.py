from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_ctx_aware_log(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_extract_run_generates_ctx_aware_timeseries(tmp_path: Path) -> None:
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

    assert payload["source_ctx_aware_log_path"] == str(log_path.resolve())
    assert payload["selected_ctx_aware_log_file_name"] == log_path.name
    assert payload["sample_count"] == 3
    assert payload["duration_s"] == 0.4
    assert payload["avg_sample_interval_s"] == 0.2
    assert payload["started_at"] == "2026-04-04T15:38:01.552Z"
    assert payload["ended_at"] == "2026-04-04T15:38:01.952Z"
    assert [point["second"] for point in payload["samples"]] == [0.0, 0.2, 0.4]
    assert [point["ongoing_agent_count"] for point in payload["samples"]] == [0, 1, 1]
    assert [point["pending_agent_count"] for point in payload["samples"]] == [0, 0, 1]
    assert payload["metric_summaries"]["ongoing_agent_count"]["max"] == 1.0
    assert (
        payload["metric_summaries"]["agents_turned_pending_due_to_context_threshold"]["total"]
        == 1.0
    )
    assert payload["metric_summaries"]["new_agents_added_as_ongoing"]["total"] == 1.0


def test_extract_run_chooses_latest_ctx_aware_log_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    older_log = run_dir / "gateway-output" / "job" / "ctx_aware_20260404T153801Z.jsonl"
    newer_log = run_dir / "gateway-output" / "job" / "ctx_aware_20260404T153901Z.jsonl"

    _write_ctx_aware_log(
        older_log,
        [
            {
                "timestamp": "2026-04-04T15:38:01.552Z",
                "ongoing_agent_count": 1,
                "pending_agent_count": 0,
                "ongoing_effective_context_tokens": 3000,
                "pending_effective_context_tokens": 0,
                "agents_turned_pending_due_to_context_threshold": 0,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 0,
                "new_agents_added_as_ongoing": 1,
            }
        ],
    )
    _write_ctx_aware_log(
        newer_log,
        [
            {
                "timestamp": "2026-04-04T15:39:01.552Z",
                "ongoing_agent_count": 2,
                "pending_agent_count": 1,
                "ongoing_effective_context_tokens": 6000,
                "pending_effective_context_tokens": 3000,
                "agents_turned_pending_due_to_context_threshold": 1,
                "agents_turned_ongoing": 0,
                "new_agents_added_as_pending": 1,
                "new_agents_added_as_ongoing": 0,
            }
        ],
    )

    payload = extract_run.extract_ctx_aware_log_from_run_dir(run_dir)

    assert payload["source_ctx_aware_log_path"] == str(newer_log.resolve())
    assert payload["ctx_aware_log_candidate_count"] == 2
    assert payload["selected_ctx_aware_log_file_name"] == newer_log.name
    assert payload["samples"][0]["ongoing_agent_count"] == 2


def test_discover_run_dirs_with_ctx_aware_log_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    _write_ctx_aware_log(
        run_a / "gateway-output" / "job" / "ctx_aware_20260404T153801Z.jsonl",
        [],
    )
    _write_ctx_aware_log(
        run_b / "gateway-output" / "job" / "ctx_aware_20260404T153901Z.jsonl",
        [],
    )

    discovered = extract_run.discover_run_dirs_with_ctx_aware_log(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_ctx_aware_log(
        run_a / "gateway-output" / "job" / "ctx_aware_20260404T153801Z.jsonl",
        [],
    )
    _write_ctx_aware_log(
        run_b / "gateway-output" / "job" / "ctx_aware_20260404T153901Z.jsonl",
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
                "--ctx-aware-log",
                str(tmp_path / "ctx_aware.jsonl"),
            ]
        )
    except ValueError as exc:
        assert "--ctx-aware-log can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --ctx-aware-log is used with --root-dir")

    try:
        extract_run.main(["--root-dir", str(root_dir), "--max-procs", "0"])
    except ValueError as exc:
        assert "--max-procs must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --max-procs <= 0")
