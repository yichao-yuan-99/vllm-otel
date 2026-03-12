from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import summarize_timeseries


def test_summarize_timeseries_payload_computes_min_max_avg(tmp_path: Path) -> None:
    source_path = tmp_path / "gauge-counter-timeseries.json"
    payload = {
        "source_run_dir": "/tmp/job",
        "source_vllm_log_dir": "/tmp/job/vllm-log",
        "cluster_mode": True,
        "port_profile_ids": [0, 1],
        "metrics": {
            "vllm:num_requests_running|engine=0": {
                "name": "vllm:num_requests_running",
                "type": "gauge",
                "labels": {"engine": "0"},
                "value": [1.0, 3.0, 2.0],
            },
            "vllm:prompt_tokens|engine=0|port_profile_id=1": {
                "name": "vllm:prompt_tokens",
                "type": "counter",
                "labels": {"engine": "0", "port_profile_id": "1"},
                "value": [0.0, 10.0, 4.0],
            },
        },
    }

    result = summarize_timeseries.summarize_timeseries_payload(
        payload,
        source_timeseries_path=source_path,
    )

    assert result["metric_count"] == 2
    assert result["cluster_mode"] is True
    assert result["port_profile_ids"] == [0, 1]
    assert result["source_timeseries_path"] == str(source_path.resolve())

    gauge = result["metrics"]["vllm:num_requests_running|engine=0"]
    assert gauge["sample_count"] == 3
    assert gauge["min"] == 1.0
    assert gauge["max"] == 3.0
    assert gauge["avg"] == 2.0

    counter = result["metrics"]["vllm:prompt_tokens|engine=0|port_profile_id=1"]
    assert counter["sample_count"] == 3
    assert counter["min"] == 0.0
    assert counter["max"] == 10.0
    assert counter["avg"] == (14.0 / 3.0)


def test_summarize_timeseries_script_writes_default_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    processed_dir = run_dir / "post-processed" / "vllm-log"
    processed_dir.mkdir(parents=True)
    input_path = processed_dir / "gauge-counter-timeseries.json"
    input_path.write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir),
                "source_vllm_log_dir": str(run_dir / "vllm-log"),
                "metrics": {
                    "vllm:num_requests_running|engine=0": {
                        "name": "vllm:num_requests_running",
                        "type": "gauge",
                        "labels": {"engine": "0"},
                        "value": [2.0, 6.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = summarize_timeseries.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = processed_dir / "gauge-counter-timeseries.stats.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    metric = payload["metrics"]["vllm:num_requests_running|engine=0"]
    assert payload["metric_count"] == 1
    assert payload["source_timeseries_path"] == str(input_path.resolve())
    assert metric["min"] == 2.0
    assert metric["max"] == 6.0
    assert metric["avg"] == 4.0


def test_discover_run_dirs_with_extracted_timeseries_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    for run_dir in [run_a, run_b]:
        processed_dir = run_dir / "post-processed" / "vllm-log"
        processed_dir.mkdir(parents=True)
        (processed_dir / "gauge-counter-timeseries.json").write_text(
            json.dumps({"metrics": {}}),
            encoding="utf-8",
        )

    discovered = summarize_timeseries.discover_run_dirs_with_extracted_timeseries(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_summarize_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    for run_dir in [run_a, run_b]:
        processed_dir = run_dir / "post-processed" / "vllm-log"
        processed_dir.mkdir(parents=True)
        (processed_dir / "gauge-counter-timeseries.json").write_text(
            json.dumps({"metrics": {}}),
            encoding="utf-8",
        )

    processed: list[tuple[Path, Path | None, Path | None]] = []

    def fake_summarize_run_dir(
        run_dir: Path,
        *,
        input_path: Path | None = None,
        output_path: Path | None = None,
    ) -> Path:
        processed.append((run_dir, input_path, output_path))
        return run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.stats.json"

    monkeypatch.setattr(summarize_timeseries, "summarize_run_dir", fake_summarize_run_dir)

    exit_code = summarize_timeseries.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, None),
        (run_b.resolve(), None, None),
    ]


def test_summarize_root_dir_continues_after_failure(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    bad_run = root_dir / "job-a"
    good_run = root_dir / "job-b"
    for run_dir in [bad_run, good_run]:
        processed_dir = run_dir / "post-processed" / "vllm-log"
        processed_dir.mkdir(parents=True)
        (processed_dir / "gauge-counter-timeseries.json").write_text(
            json.dumps({"metrics": {}}),
            encoding="utf-8",
        )

    processed: list[Path] = []

    def fake_summarize_run_dir(
        run_dir: Path,
        *,
        input_path: Path | None = None,
        output_path: Path | None = None,
    ) -> Path:
        processed.append(run_dir)
        if run_dir == bad_run.resolve():
            raise ValueError("broken run")
        return run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.stats.json"

    monkeypatch.setattr(summarize_timeseries, "summarize_run_dir", fake_summarize_run_dir)

    exit_code = summarize_timeseries.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 1
    assert processed == [bad_run.resolve(), good_run.resolve()]


def test_summarize_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    try:
        summarize_timeseries.main(["--run-dir", str(run_dir), "--dry-run"])
    except ValueError as exc:
        assert "--dry-run can only be used with --root-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --dry-run is used with --run-dir")


def test_summarize_rejects_input_output_for_root_dir(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    root_dir.mkdir(parents=True)

    try:
        summarize_timeseries.main(["--root-dir", str(root_dir), "--input", "x"])
    except ValueError as exc:
        assert "--input can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --input is used with --root-dir")

    try:
        summarize_timeseries.main(["--root-dir", str(root_dir), "--output", "x"])
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")


def test_summarize_root_dir_uses_process_pool(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    for run_dir in [run_a, run_b]:
        processed_dir = run_dir / "post-processed" / "vllm-log"
        processed_dir.mkdir(parents=True)
        (processed_dir / "gauge-counter-timeseries.json").write_text(
            json.dumps({"metrics": {}}),
            encoding="utf-8",
        )

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

    monkeypatch.setattr(summarize_timeseries, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        summarize_timeseries,
        "_summarize_run_dir_worker",
        lambda run_dir_text: (
            run_dir_text,
            str(Path(run_dir_text) / "post-processed" / "vllm-log" / "gauge-counter-timeseries.stats.json"),
            None,
        ),
    )

    exit_code = summarize_timeseries.main(["--root-dir", str(root_dir), "--max-procs", "2"])

    assert exit_code == 0
    assert captured_max_workers == [2]
    assert captured_inputs == [[str(run_a.resolve()), str(run_b.resolve())]]


def test_summarize_root_dir_falls_back_to_sequential_when_pool_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    for run_dir in [run_a, run_b]:
        processed_dir = run_dir / "post-processed" / "vllm-log"
        processed_dir.mkdir(parents=True)
        (processed_dir / "gauge-counter-timeseries.json").write_text(
            json.dumps({"metrics": {}}),
            encoding="utf-8",
        )

    parallel_attempts: list[tuple[list[Path], int]] = []
    sequential_attempts: list[list[Path]] = []

    def fake_parallel(run_dirs: list[Path], *, max_procs: int) -> int:
        parallel_attempts.append((run_dirs, max_procs))
        raise PermissionError("semaphore blocked")

    def fake_sequential(run_dirs: list[Path]) -> int:
        sequential_attempts.append(run_dirs)
        return 0

    monkeypatch.setattr(summarize_timeseries, "_run_root_dir_parallel", fake_parallel)
    monkeypatch.setattr(summarize_timeseries, "_run_root_dir_sequential", fake_sequential)

    exit_code = summarize_timeseries.main(["--root-dir", str(root_dir), "--max-procs", "2"])

    assert exit_code == 0
    assert parallel_attempts == [([run_a.resolve(), run_b.resolve()], 2)]
    assert sequential_attempts == [[run_a.resolve(), run_b.resolve()]]
