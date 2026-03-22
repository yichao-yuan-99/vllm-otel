from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def test_extract_global_summary_from_replay_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:10.000Z",
                "worker_results": {
                    "trial-0001": {
                        "worker_id": "trial-0001",
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:01.000Z",
                        "finished_at": "2026-03-08T00:00:04.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:02.000Z",
                        "finished_at": "2026-03-08T00:00:07.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = run_dir / "post-processed" / "global" / "trial-timing-summary.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_type"] == "replay"
    assert payload["trial_count"] == 2
    assert payload["trail_count"] == 2
    assert payload["total_duration_s"] == 10.0
    assert payload["trial_duration_stats_s"]["min"] == 3.0
    assert payload["trial_duration_stats_s"]["max"] == 5.0
    assert payload["trial_duration_stats_s"]["avg"] == 4.0
    assert payload["agent_time_breakdown_s"]["agent_count"] == 0
    assert payload["agent_time_breakdown_s"]["llm_time_sum_s"] == 0.0
    assert payload["agent_time_breakdown_s"]["non_llm_time_sum_s"] == 0.0
    assert payload["trials"][0]["trial_id"] == "trial-0001"
    assert payload["trials"][0]["start_offset_s"] == 1.0
    assert payload["trials"][0]["end_offset_s"] == 4.0
    assert payload["trials"][0]["duration_s"] == 3.0


def test_extract_global_summary_from_con_driver_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-con-driver"
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00+00:00",
                "finished_at": "2026-03-08T00:00:08+00:00",
                "run_duration_s": 8.0,
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "trial_id": "trial-0001",
                    "status": "ok",
                    "started_at": "2026-03-08T00:00:01+00:00",
                    "finished_at": "2026-03-08T00:00:03+00:00",
                    "duration_s": 2.0,
                },
                {
                    "trial_id": "trial-0002",
                    "status": "ok",
                    "started_at": "2026-03-08T00:00:02+00:00",
                    "finished_at": "2026-03-08T00:00:06+00:00",
                    "duration_s": 4.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = extract_run.extract_global_trial_summary_from_run_dir(run_dir)
    assert result["source_type"] == "con-driver"
    assert result["trial_count"] == 2
    assert result["total_duration_s"] == 8.0
    assert result["trial_duration_stats_s"]["min"] == 2.0
    assert result["trial_duration_stats_s"]["max"] == 4.0
    assert result["trial_duration_stats_s"]["avg"] == 3.0
    assert result["agent_time_breakdown_s"]["agent_count"] == 0
    assert result["trials"][0]["trial_id"] == "trial-0001"
    assert result["trials"][0]["start_offset_s"] == 1.0
    assert result["trials"][0]["end_offset_s"] == 3.0


def test_extract_global_summary_includes_per_agent_llm_and_non_llm_time(tmp_path: Path) -> None:
    run_dir = tmp_path / "job-replay"
    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True)

    trial1_token = "token-trial-1"
    trial2_token = "token-trial-2"
    trial1_hash = hashlib.sha256(trial1_token.encode("utf-8")).hexdigest()
    trial2_hash = hashlib.sha256(trial2_token.encode("utf-8")).hexdigest()

    (replay_dir / "summary.json").write_text(
        json.dumps(
            {
                "started_at": "2026-03-08T00:00:00.000Z",
                "finished_at": "2026-03-08T00:00:30.000Z",
                "worker_results": {
                    "trial-0001": {
                        "worker_id": "trial-0001",
                        "api_token": trial1_token,
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:01.000Z",
                        "finished_at": "2026-03-08T00:00:11.000Z",
                    },
                    "trial-0002": {
                        "worker_id": "trial-0002",
                        "api_token": trial2_token,
                        "status": "completed",
                        "started_at": "2026-03-08T00:00:02.000Z",
                        "finished_at": "2026-03-08T00:00:22.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    run1_dir = run_dir / "gateway-output" / "run_01"
    run2_dir = run_dir / "gateway-output" / "run_02"
    (run1_dir / "events").mkdir(parents=True)
    (run1_dir / "requests").mkdir(parents=True)
    (run2_dir / "events").mkdir(parents=True)
    (run2_dir / "requests").mkdir(parents=True)

    (run1_dir / "events" / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "job_start",
                        "timestamp": "2026-03-08T00:00:00.000Z",
                        "trace_id": "trace-1",
                        "api_token_hash": trial1_hash,
                    }
                ),
                json.dumps(
                    {
                        "event_type": "agent_start",
                        "timestamp": "2026-03-08T00:00:01.000Z",
                        "trace_id": "trace-1",
                        "api_token_hash": trial1_hash,
                    }
                ),
                json.dumps(
                    {
                        "event_type": "agent_end",
                        "timestamp": "2026-03-08T00:00:11.000Z",
                        "trace_id": "trace-1",
                        "api_token_hash": trial1_hash,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run1_dir / "requests" / "model_inference.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_start_time": "2026-03-08T00:00:02.000Z",
                        "request_end_time": "2026-03-08T00:00:04.000Z",
                        "request_duration_ms": 2000,
                    }
                ),
                json.dumps(
                    {
                        "request_start_time": "2026-03-08T00:00:05.000Z",
                        "request_end_time": "2026-03-08T00:00:08.000Z",
                        "request_duration_ms": 3000,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (run2_dir / "events" / "lifecycle.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "job_start",
                        "timestamp": "2026-03-08T00:00:00.000Z",
                        "trace_id": "trace-2",
                        "api_token_hash": trial2_hash,
                    }
                ),
                json.dumps(
                    {
                        "event_type": "agent_start",
                        "timestamp": "2026-03-08T00:00:02.000Z",
                        "trace_id": "trace-2",
                        "api_token_hash": trial2_hash,
                    }
                ),
                json.dumps(
                    {
                        "event_type": "agent_end",
                        "timestamp": "2026-03-08T00:00:22.000Z",
                        "trace_id": "trace-2",
                        "api_token_hash": trial2_hash,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run2_dir / "requests" / "model_inference.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_start_time": "2026-03-08T00:00:03.000Z",
                        "request_end_time": "2026-03-08T00:00:07.000Z",
                        "request_duration_ms": 4000,
                    }
                ),
                json.dumps(
                    {
                        "request_start_time": "2026-03-08T00:00:08.000Z",
                        "request_end_time": "2026-03-08T00:00:12.000Z",
                        "request_duration_ms": 4000,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = extract_run.extract_global_trial_summary_from_run_dir(run_dir)

    breakdown = result["agent_time_breakdown_s"]
    assert breakdown["agent_count"] == 2
    assert breakdown["mapped_trial_count"] == 2
    assert breakdown["llm_time_sum_s"] == 13.0
    assert breakdown["non_llm_time_sum_s"] == 17.0
    assert breakdown["agent_total_time_sum_s"] == 30.0

    agent_entries = breakdown["agents"]
    assert [entry["trial_id"] for entry in agent_entries] == ["trial-0001", "trial-0002"]
    assert agent_entries[0]["llm_time_s"] == 5.0
    assert agent_entries[0]["non_llm_time_s"] == 5.0
    assert agent_entries[0]["agent_total_time_s"] == 10.0
    assert agent_entries[1]["llm_time_s"] == 8.0
    assert agent_entries[1]["non_llm_time_s"] == 12.0
    assert agent_entries[1]["agent_total_time_s"] == 20.0


def test_extract_global_summary_rejects_unknown_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "unknown"
    run_dir.mkdir(parents=True)

    try:
        extract_run.extract_global_trial_summary_from_run_dir(run_dir)
    except ValueError as exc:
        assert "Unrecognized run layout" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown run directory layout")


def test_discover_run_dirs_with_global_sources_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    replay_run = root_dir / "replay-job"
    con_driver_run = root_dir / "con-driver-job"
    (replay_run / "replay").mkdir(parents=True)
    (replay_run / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (con_driver_run / "meta").mkdir(parents=True)
    (con_driver_run / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (con_driver_run / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")

    discovered = extract_run.discover_run_dirs_with_global_sources(root_dir)

    assert discovered == [con_driver_run.resolve(), replay_run.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    (run_a / "replay").mkdir(parents=True)
    (run_a / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (run_b / "meta").mkdir(parents=True)
    (run_b / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (run_b / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "post-processed" / "global" / "trial-timing-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]


def test_extract_run_root_dir_continues_after_failure(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    bad_run = root_dir / "bad"
    good_run = root_dir / "good"
    (bad_run / "replay").mkdir(parents=True)
    (bad_run / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (good_run / "meta").mkdir(parents=True)
    (good_run / "meta" / "results.json").write_text("[]", encoding="utf-8")
    (good_run / "meta" / "run_manifest.json").write_text("{}", encoding="utf-8")

    processed: list[Path] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append(run_dir)
        if run_dir == bad_run.resolve():
            raise ValueError("broken run")
        return run_dir / "post-processed" / "global" / "trial-timing-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 1
    assert processed == [bad_run.resolve(), good_run.resolve()]


def test_extract_run_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_extract_run_rejects_output_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")])
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")


def test_extract_run_root_dir_uses_process_pool(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    (run_a / "replay").mkdir(parents=True)
    (run_a / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (run_b / "replay").mkdir(parents=True)
    (run_b / "replay" / "summary.json").write_text("{}", encoding="utf-8")

    captured_max_workers: list[int] = []
    captured_inputs: list[list[str]] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            captured_max_workers.append(max_workers)

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def map(self, fn, iterable):
            inputs = list(iterable)
            captured_inputs.append(inputs)
            return [fn(item) for item in inputs]

    monkeypatch.setattr(extract_run, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        extract_run,
        "_extract_run_dir_worker",
        lambda run_dir_text: (
            run_dir_text,
            str(Path(run_dir_text) / "post-processed" / "global" / "trial-timing-summary.json"),
            None,
        ),
    )

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "2"])

    assert exit_code == 0
    assert captured_max_workers == [2]
    assert captured_inputs == [[str(run_a.resolve()), str(run_b.resolve())]]


def test_extract_run_root_dir_falls_back_to_sequential_when_pool_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    (run_a / "replay").mkdir(parents=True)
    (run_a / "replay" / "summary.json").write_text("{}", encoding="utf-8")
    (run_b / "replay").mkdir(parents=True)
    (run_b / "replay" / "summary.json").write_text("{}", encoding="utf-8")

    parallel_attempts: list[tuple[list[Path], int]] = []
    sequential_attempts: list[list[Path]] = []

    def fake_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
        parallel_attempts.append((run_dirs, max_procs))
        raise PermissionError("semaphore blocked")

    def fake_sequential(run_dirs: list[Path]) -> int:
        sequential_attempts.append(run_dirs)
        return 0

    monkeypatch.setattr(extract_run, "_run_root_dir_parallel", fake_parallel)
    monkeypatch.setattr(extract_run, "_run_root_dir_sequential", fake_sequential)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "2"])

    assert exit_code == 0
    assert parallel_attempts == [([run_a.resolve(), run_b.resolve()], 2)]
    assert sequential_attempts == [[run_a.resolve(), run_b.resolve()]]
