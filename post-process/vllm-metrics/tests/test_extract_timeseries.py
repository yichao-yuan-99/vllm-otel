from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from common.extract_timeseries import (
    ParsedMetricRecord,
    build_gauge_counter_timeseries,
    count_metric_blocks_in_run_dir,
    extract_gauge_counter_timeseries_from_run_dir,
)
import extract_run


def test_build_gauge_counter_timeseries_computes_counter_deltas() -> None:
    records = [
        ParsedMetricRecord(
            captured_at="2026-03-01T00:00:00+00:00",
            families={
                "vllm:num_requests_running": {
                    "type": "gauge",
                    "help": "running",
                    "samples": [
                        {
                            "name": "vllm:num_requests_running",
                            "labels": {"engine": "0"},
                            "value": 2.0,
                        }
                    ],
                },
                "vllm:prompt_tokens": {
                    "type": "counter",
                    "help": "prompt",
                    "samples": [
                        {
                            "name": "vllm:prompt_tokens_total",
                            "labels": {"engine": "0"},
                            "value": 5.0,
                        }
                    ],
                },
            },
        ),
        ParsedMetricRecord(
            captured_at="2026-03-01T00:00:02+00:00",
            families={
                "vllm:num_requests_running": {
                    "type": "gauge",
                    "help": "running",
                    "samples": [
                        {
                            "name": "vllm:num_requests_running",
                            "labels": {"engine": "0"},
                            "value": 3.0,
                        }
                    ],
                },
                "vllm:prompt_tokens": {
                    "type": "counter",
                    "help": "prompt",
                    "samples": [
                        {
                            "name": "vllm:prompt_tokens_total",
                            "labels": {"engine": "0"},
                            "value": 10.0,
                        }
                    ],
                },
            },
        ),
    ]

    result = build_gauge_counter_timeseries(records)

    gauge = result["metrics"]["vllm:num_requests_running|engine=0"]
    assert gauge["type"] == "gauge"
    assert gauge["value"] == [2.0, 3.0]
    assert gauge["time_from_start_s"] == [0.0, 2.0]

    counter = result["metrics"]["vllm:prompt_tokens|engine=0"]
    assert counter["type"] == "counter"
    assert counter["value"] == [0.0, 5.0]
    assert counter["time_from_start_s"] == [0.0, 2.0]


def test_extract_run_script_reads_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    vllm_log_dir = run_dir / "vllm-log"
    vllm_log_dir.mkdir(parents=True)

    records = [
        {
            "timestamp": 1,
            "captured_at": "2026-03-01T00:00:00+00:00",
            "content": (
                "# HELP vllm:num_requests_running Number of requests in model execution batches.\n"
                "# TYPE vllm:num_requests_running gauge\n"
                'vllm:num_requests_running{engine="0"} 1\n'
                "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n"
                "# TYPE vllm:prompt_tokens_total counter\n"
                'vllm:prompt_tokens_total{engine="0"} 5\n'
            ),
        },
        {
            "timestamp": 2,
            "captured_at": "2026-03-01T00:00:01+00:00",
            "content": (
                "# HELP vllm:num_requests_running Number of requests in model execution batches.\n"
                "# TYPE vllm:num_requests_running gauge\n"
                'vllm:num_requests_running{engine="0"} 4\n'
                "# HELP vllm:prompt_tokens_total Number of prefill tokens processed.\n"
                "# TYPE vllm:prompt_tokens_total counter\n"
                'vllm:prompt_tokens_total{engine="0"} 8\n'
            ),
        },
    ]

    jsonl_path = tmp_path / "block-000000.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    tar_path = vllm_log_dir / "block-000000.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as archive:
        archive.add(jsonl_path, arcname="block-000000.jsonl")

    (vllm_log_dir / "blocks.index.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "file": "block-000000.tar.gz",
                        "member": "block-000000.jsonl",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metric_count"] == 2
    assert payload["metrics"]["vllm:num_requests_running|engine=0"]["value"] == [1.0, 4.0]
    assert payload["metrics"]["vllm:prompt_tokens|engine=0"]["value"] == [0.0, 3.0]


def test_count_metric_blocks_in_run_dir_reads_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    vllm_log_dir = run_dir / "vllm-log"
    vllm_log_dir.mkdir(parents=True)
    (vllm_log_dir / "blocks.index.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {"file": "block-000000.tar.gz", "member": "block-000000.jsonl"},
                    {"file": "block-000001.tar.gz", "member": "block-000001.jsonl"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert count_metric_blocks_in_run_dir(run_dir) == 2


def test_extract_cluster_mode_series_keeps_profiles_separate(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    vllm_log_dir = run_dir / "vllm-log"
    profile_0_dir = vllm_log_dir / "profile-0"
    profile_1_dir = vllm_log_dir / "profile-1"
    profile_0_dir.mkdir(parents=True)
    profile_1_dir.mkdir(parents=True)

    profile_0_records = [
        {
            "captured_at": "2026-03-01T00:00:00+00:00",
            "families": {
                "vllm:prompt_tokens": {
                    "type": "counter",
                    "help": "prompt",
                    "samples": [
                        {
                            "name": "vllm:prompt_tokens_total",
                            "labels": {"engine": "0"},
                            "value": 5.0,
                        }
                    ],
                }
            },
        },
        {
            "captured_at": "2026-03-01T00:00:01+00:00",
            "families": {
                "vllm:prompt_tokens": {
                    "type": "counter",
                    "help": "prompt",
                    "samples": [
                        {
                            "name": "vllm:prompt_tokens_total",
                            "labels": {"engine": "0"},
                            "value": 8.0,
                        }
                    ],
                }
            },
        },
    ]
    profile_1_records = [
        {
            "captured_at": "2026-03-01T00:00:00+00:00",
            "families": {
                "vllm:prompt_tokens": {
                    "type": "counter",
                    "help": "prompt",
                    "samples": [
                        {
                            "name": "vllm:prompt_tokens_total",
                            "labels": {"engine": "0"},
                            "value": 10.0,
                        }
                    ],
                }
            },
        },
        {
            "captured_at": "2026-03-01T00:00:01+00:00",
            "families": {
                "vllm:prompt_tokens": {
                    "type": "counter",
                    "help": "prompt",
                    "samples": [
                        {
                            "name": "vllm:prompt_tokens_total",
                            "labels": {"engine": "0"},
                            "value": 13.0,
                        }
                    ],
                }
            },
        },
    ]

    jsonl_0 = tmp_path / "profile-0-block-000000.jsonl"
    jsonl_1 = tmp_path / "profile-1-block-000000.jsonl"
    jsonl_0.write_text(
        "\n".join(json.dumps(record) for record in profile_0_records) + "\n",
        encoding="utf-8",
    )
    jsonl_1.write_text(
        "\n".join(json.dumps(record) for record in profile_1_records) + "\n",
        encoding="utf-8",
    )

    with tarfile.open(profile_0_dir / "block-000000.tar.gz", mode="w:gz") as archive:
        archive.add(jsonl_0, arcname="block-000000.jsonl")
    with tarfile.open(profile_1_dir / "block-000000.tar.gz", mode="w:gz") as archive:
        archive.add(jsonl_1, arcname="block-000000.jsonl")

    assert count_metric_blocks_in_run_dir(run_dir) == 2

    payload = extract_gauge_counter_timeseries_from_run_dir(run_dir)
    assert payload["cluster_mode"] is True
    assert payload["port_profile_ids"] == [0, 1]

    metric_0 = payload["metrics"]["vllm:prompt_tokens|engine=0|port_profile_id=0"]
    metric_1 = payload["metrics"]["vllm:prompt_tokens|engine=0|port_profile_id=1"]
    assert metric_0["value"] == [0.0, 3.0]
    assert metric_1["value"] == [0.0, 3.0]


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

    def fake_extract(run_dir_arg: Path, *, on_block_loaded=None):
        assert run_dir_arg == run_dir.resolve()
        if on_block_loaded is not None:
            on_block_loaded(1, 2)
            on_block_loaded(2, 2)
        return {
            "source_run_dir": str(run_dir_arg),
            "source_vllm_log_dir": str(run_dir_arg / "vllm-log"),
            "first_captured_at": None,
            "metric_count": 0,
            "metrics": {},
        }

    monkeypatch.setattr(extract_run, "count_metric_blocks_in_run_dir", lambda _: 2)
    monkeypatch.setattr(extract_run, "extract_gauge_counter_timeseries_from_run_dir", fake_extract)
    monkeypatch.setattr(extract_run, "create_extract_progress", lambda: fake_progress)

    exit_code = extract_run.main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    assert fake_progress.added == [
        {
            "description": "extracting vllm metrics",
            "total": 2,
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


def test_discover_run_dirs_with_vllm_log_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    run_a_vllm = run_a / "vllm-log"
    run_b_vllm = run_b / "vllm-log"
    run_a_vllm.mkdir(parents=True)
    run_b_vllm.mkdir(parents=True)
    # This path is output from post-process and should not be treated as a run root.
    (run_a / "post-processed" / "vllm-log").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_vllm_log(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "vllm-log").mkdir(parents=True)
    (run_b / "vllm-log").mkdir(parents=True)

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return (
            run_dir
            / "post-processed"
            / "vllm-log"
            / "gauge-counter-timeseries.json"
        )

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]


def test_extract_run_root_dir_continues_after_failure(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    bad_run = root_dir / "job-a"
    good_run = root_dir / "job-b"
    (bad_run / "vllm-log").mkdir(parents=True)
    (good_run / "vllm-log").mkdir(parents=True)

    processed: list[Path] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append(run_dir)
        if run_dir == bad_run.resolve():
            raise ValueError("broken run")
        return (
            run_dir
            / "post-processed"
            / "vllm-log"
            / "gauge-counter-timeseries.json"
        )

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


def test_extract_run_root_dir_uses_process_pool(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "vllm-log").mkdir(parents=True)
    (run_b / "vllm-log").mkdir(parents=True)

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
            str(Path(run_dir_text) / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json"),
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
    (run_a / "vllm-log").mkdir(parents=True)
    (run_b / "vllm-log").mkdir(parents=True)

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
