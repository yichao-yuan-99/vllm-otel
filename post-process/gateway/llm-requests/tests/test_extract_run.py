from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_extract_run_generates_request_list_and_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    gateway_run_dir = run_dir / "gateway-output" / "profile-0" / "run_001"
    requests_dir = gateway_run_dir / "requests"
    events_dir = gateway_run_dir / "events"
    trace_dir = gateway_run_dir / "trace"
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    _write_jsonl(
        events_dir / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "timestamp": "2026-03-06T06:23:00.000Z"},
            {"event_type": "job_end", "timestamp": "2026-03-06T06:23:10.000Z"},
        ],
    )

    _write_jsonl(
        requests_dir / "model_inference.jsonl",
        [
            {
                "request_id": "req-2",
                "trace_id": "trace-1",
                "model_inference_span_id": "span-2",
                "model_inference_parent_span_id": "agent-span",
                "request_start_time": "2026-03-06T06:23:02.000Z",
                "request_end_time": "2026-03-06T06:23:04.000Z",
                "request_duration_ms": 2000.0,
                "duration_ms": 2000.0,
                "status_code": 499,
                "http_method": "POST",
                "http_path": "v1/chat/completions",
                "model": "Test-Model",
                "response": {
                    "usage": {
                        "prompt_tokens": 20,
                        "total_tokens": 25,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    }
                },
            },
            {
                "request_id": "req-1",
                "trace_id": "trace-1",
                "model_inference_span_id": "span-1",
                "model_inference_parent_span_id": "agent-span",
                "request_start_time": "2026-03-06T06:23:01.000Z",
                "request_end_time": "2026-03-06T06:23:02.000Z",
                "request_duration_ms": 1000.0,
                "duration_ms": 1000.0,
                "status_code": 200,
                "http_method": "POST",
                "http_path": "v1/chat/completions",
                "model": "Test-Model",
                "response": {
                    "usage": {
                        "prompt_tokens": 10,
                        "total_tokens": 12,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 1},
                    }
                },
            },
        ],
    )

    (trace_dir / "jaeger_trace.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "spans": [
                            {
                                "traceID": "trace-1",
                                "spanID": "llm-1",
                                "operationName": "llm_request",
                                "references": [
                                    {
                                        "refType": "CHILD_OF",
                                        "traceID": "trace-1",
                                        "spanID": "span-1",
                                    }
                                ],
                                "tags": [
                                    {
                                        "key": "gen_ai.latency.time_to_first_token",
                                        "type": "float64",
                                        "value": 0.1,
                                    },
                                    {
                                        "key": "gen_ai.latency.time_in_model_prefill",
                                        "type": "float64",
                                        "value": 0.25,
                                    },
                                    {
                                        "key": "gen_ai.latency.time_in_model_decode",
                                        "type": "float64",
                                        "value": 0.5,
                                    },
                                    {
                                        "key": "gen_ai.request.id",
                                        "type": "string",
                                        "value": "chatcmpl-1",
                                    },
                                ],
                            },
                            {
                                "traceID": "trace-1",
                                "spanID": "llm-2",
                                "operationName": "llm_request",
                                "references": [
                                    {
                                        "refType": "CHILD_OF",
                                        "traceID": "trace-1",
                                        "spanID": "span-2",
                                    }
                                ],
                                "tags": [
                                    {
                                        "key": "gen_ai.latency.time_to_first_token",
                                        "type": "float64",
                                        "value": 0.2,
                                    },
                                    {
                                        "key": "gen_ai.latency.time_in_model_prefill",
                                        "type": "float64",
                                        "value": 1.0,
                                    },
                                    {
                                        "key": "gen_ai.latency.time_in_model_decode",
                                        "type": "float64",
                                        "value": 2.0,
                                    },
                                    {
                                        "key": "gen_ai.request.id",
                                        "type": "string",
                                        "value": "chatcmpl-2",
                                    },
                                ],
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_dir = run_dir / "post-processed" / "gateway" / "llm-requests"
    requests_payload = json.loads((output_dir / "llm-requests.json").read_text(encoding="utf-8"))
    stats_payload = json.loads((output_dir / "llm-request-stats.json").read_text(encoding="utf-8"))
    speed_stats_payload = json.loads(
        (output_dir / "llm-request-speed-stats.json").read_text(encoding="utf-8")
    )
    longest_payload = json.loads((output_dir / "llm-requests-longest-10.json").read_text(encoding="utf-8"))
    shortest_payload = json.loads((output_dir / "llm-requests-shortest-10.json").read_text(encoding="utf-8"))
    stats_200_payload = json.loads((output_dir / "llm-requests-stats.200.json").read_text(encoding="utf-8"))
    stats_499_payload = json.loads((output_dir / "llm-requests-stats.499.json").read_text(encoding="utf-8"))

    assert requests_payload["request_count"] == 2
    requests = requests_payload["requests"]
    assert [record["request_id"] for record in requests] == ["req-1", "req-2"]
    assert requests[0]["gateway_profile_id"] == 0
    assert requests[0]["request_start_offset_s"] == 1.0
    assert requests[0]["request_end_offset_s"] == 2.0
    assert requests[0]["request_end_to_run_end_s"] == 8.0
    assert requests[0]["prompt_tokens"] == 10
    assert requests[0]["completion_tokens"] == 2
    assert requests[0]["cached_tokens"] == 1
    assert requests[0]["gen_ai.latency.time_to_first_token"] == 0.1
    assert requests[0]["gen_ai.request.id"] == "chatcmpl-1"

    assert stats_payload["request_count"] == 2
    duration_metric = stats_payload["metrics"]["request_duration_ms"]
    assert duration_metric["min"] == 1000.0
    assert duration_metric["max"] == 2000.0
    assert duration_metric["avg"] == 1500.0
    ttft_metric = stats_payload["metrics"]["gen_ai.latency.time_to_first_token"]
    assert ttft_metric["min"] == 0.1
    assert ttft_metric["max"] == 0.2
    assert ttft_metric["avg"] == 0.15000000000000002
    assert stats_payload["average_stage_speed_tokens_per_s"] == {
        "request_status_code": 200,
        "request_count_200": 1,
        "prefill": {
            "eligible_request_count": 1,
            "excluded_request_count": 0,
            "avg_tokens_per_s": 40.0,
            "min_tokens_per_s": 40.0,
            "max_tokens_per_s": 40.0,
        },
        "decode": {
            "eligible_request_count": 1,
            "excluded_request_count": 0,
            "avg_tokens_per_s": 4.0,
            "min_tokens_per_s": 4.0,
            "max_tokens_per_s": 4.0,
        },
    }
    assert speed_stats_payload["average_stage_speed_tokens_per_s"] == (
        stats_payload["average_stage_speed_tokens_per_s"]
    )

    assert longest_payload["selection"] == "longest"
    assert shortest_payload["selection"] == "shortest"
    assert [record["request_id"] for record in longest_payload["requests"]] == ["req-2", "req-1"]
    assert [record["request_id"] for record in shortest_payload["requests"]] == ["req-1", "req-2"]
    assert stats_200_payload["status_code"] == 200
    assert stats_200_payload["request_count"] == 1
    assert stats_200_payload["metrics"]["request_duration_ms"]["avg"] == 1000.0
    assert stats_499_payload["status_code"] == 499
    assert stats_499_payload["request_count"] == 1
    assert stats_499_payload["metrics"]["request_duration_ms"]["avg"] == 2000.0


def test_extract_run_supports_non_cluster_gateway_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    gateway_run_dir = run_dir / "gateway-output" / "run_001"
    requests_dir = gateway_run_dir / "requests"
    events_dir = gateway_run_dir / "events"
    trace_dir = gateway_run_dir / "trace"
    requests_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)

    _write_jsonl(
        events_dir / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "timestamp": "2026-03-06T06:23:00.000Z"},
        ],
    )
    _write_jsonl(
        requests_dir / "model_inference.jsonl",
        [
            {
                "request_id": "req-1",
                "model_inference_span_id": "span-1",
                "request_start_time": "2026-03-06T06:23:00.100Z",
                "request_end_time": "2026-03-06T06:23:00.200Z",
                "request_duration_ms": 100.0,
                "response": {"usage": {"prompt_tokens": 1, "total_tokens": 2, "completion_tokens": 1}},
            }
        ],
    )
    (trace_dir / "jaeger_trace.json").write_text(json.dumps({"data": [{"spans": []}]}), encoding="utf-8")

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    requests_payload = json.loads(
        (run_dir / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json").read_text(
            encoding="utf-8"
        )
    )
    assert requests_payload["requests"][0]["gateway_profile_id"] is None


def test_extract_run_updates_progress_bar(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    class FakeProgress:
        def __init__(self) -> None:
            self.added: list[dict[str, object]] = []
            self.updates: list[dict[str, object]] = []

        def __enter__(self) -> "FakeProgress":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def add_task(self, description: str, *, total: int, **fields: object) -> int:
            self.added.append(
                {
                    "description": description,
                    "total": total,
                    "fields": fields,
                }
            )
            return 1

        def update(self, task_id: int, *, advance: int = 0, **fields: object) -> None:
            self.updates.append(
                {
                    "task_id": task_id,
                    "advance": advance,
                    "fields": fields,
                }
            )

    fake_progress = FakeProgress()

    def fake_collect(run_dir_arg: Path, *, cutoff_time_utc=None, on_request_loaded=None):
        assert run_dir_arg == run_dir.resolve()
        assert cutoff_time_utc is None
        if on_request_loaded is not None:
            on_request_loaded(1, 2)
            on_request_loaded(2, 2)
        return (
            [
                {"request_id": "req-1", "request_duration_ms": 100.0},
                {"request_id": "req-2", "request_duration_ms": 200.0},
            ],
            2,
        )

    monkeypatch.setattr(extract_run, "create_extract_progress", lambda: fake_progress)
    monkeypatch.setattr(extract_run, "collect_llm_request_records", fake_collect)

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0
    assert fake_progress.added == [
        {
            "description": "extracting gateway llm requests",
            "total": 1,
            "fields": {},
        }
    ]
    assert fake_progress.updates == [
        {
            "task_id": 1,
            "advance": 0,
            "fields": {"completed": 1, "total": 2},
        },
        {
            "task_id": 1,
            "advance": 0,
            "fields": {"completed": 2, "total": 2},
        },
    ]


def test_select_extreme_duration_requests_limits_to_ten() -> None:
    records = []
    for index in range(12):
        records.append(
            {
                "request_id": f"req-{index:02d}",
                "request_start_time": f"2026-03-06T06:23:{index:02d}.000Z",
                "request_duration_ms": float(index + 1),
            }
        )

    longest, shortest = extract_run.select_extreme_duration_requests(records, limit=10)
    assert len(longest) == 10
    assert len(shortest) == 10
    assert longest[0]["request_id"] == "req-11"
    assert longest[-1]["request_id"] == "req-02"
    assert shortest[0]["request_id"] == "req-00"
    assert shortest[-1]["request_id"] == "req-09"


def test_build_stats_by_status_code() -> None:
    records = [
        {"request_id": "req-1", "status_code": 200, "request_duration_ms": 100.0},
        {"request_id": "req-2", "status_code": 200, "request_duration_ms": 200.0},
        {"request_id": "req-3", "status_code": 499, "request_duration_ms": 300.0},
        {"request_id": "req-4", "status_code": None, "request_duration_ms": 400.0},
    ]

    payloads = extract_run.build_stats_by_status_code(records)
    assert sorted(payloads.keys()) == ["200", "499"]
    assert payloads["200"]["request_count"] == 2
    assert payloads["200"]["metrics"]["request_duration_ms"]["avg"] == 150.0
    assert payloads["499"]["request_count"] == 1
    assert payloads["499"]["metrics"]["request_duration_ms"]["avg"] == 300.0


def test_build_average_stage_speed_tokens_per_s_uses_only_200_requests() -> None:
    records = [
        {
            "request_id": "req-1",
            "status_code": 200,
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "gen_ai.latency.time_in_model_prefill": 2.0,
            "gen_ai.latency.time_in_model_decode": 4.0,
        },
        {
            "request_id": "req-2",
            "status_code": 200,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "gen_ai.latency.time_in_model_prefill": 1.0,
            "gen_ai.latency.time_in_model_decode": 2.0,
        },
        {
            "request_id": "req-3",
            "status_code": 499,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "gen_ai.latency.time_in_model_prefill": 0.5,
            "gen_ai.latency.time_in_model_decode": 0.5,
        },
        {
            "request_id": "req-4",
            "status_code": 200,
            "prompt_tokens": 30,
            "completion_tokens": 10,
            "gen_ai.latency.time_in_model_prefill": 0.0,
            "gen_ai.latency.time_in_model_decode": None,
        },
    ]

    payload = extract_run.build_average_stage_speed_tokens_per_s(records)

    assert payload["request_status_code"] == 200
    assert payload["request_count_200"] == 3
    assert payload["prefill"]["eligible_request_count"] == 2
    assert payload["prefill"]["excluded_request_count"] == 1
    assert payload["prefill"]["avg_tokens_per_s"] == 50.0
    assert payload["decode"]["eligible_request_count"] == 2
    assert payload["decode"]["excluded_request_count"] == 1
    assert payload["decode"]["avg_tokens_per_s"] == 10.0


def test_discover_run_dirs_with_gateway_output_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_gateway_output(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    processed: list[tuple[Path, Path | None, bool]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        show_progress: bool = True,
    ) -> list[Path]:
        processed.append((run_dir, output_dir, show_progress))
        output_dir_resolved = (
            output_dir
            if output_dir is not None
            else run_dir / "post-processed" / "gateway" / "llm-requests"
        )
        return [output_dir_resolved / "llm-requests.json"]

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, True),
        (run_b.resolve(), None, True),
    ]


def test_extract_run_root_dir_continues_after_failure(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    bad_run = root_dir / "job-a"
    good_run = root_dir / "job-b"
    (bad_run / "gateway-output").mkdir(parents=True)
    (good_run / "gateway-output").mkdir(parents=True)

    processed: list[Path] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        show_progress: bool = True,
    ) -> list[Path]:
        processed.append(run_dir)
        if run_dir == bad_run.resolve():
            raise ValueError("broken run")
        return [run_dir / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json"]

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


def test_extract_run_rejects_output_dir_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        extract_run.main(["--root-dir", str(root_dir), "--output-dir", str(tmp_path / "out")])
    except ValueError as exc:
        assert "--output-dir can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output-dir is used with --root-dir")


def test_extract_run_root_dir_uses_process_pool(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

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
            [str(Path(run_dir_text) / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json")],
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
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

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
