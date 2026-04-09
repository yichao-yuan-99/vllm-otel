from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _expected_percentile(sorted_values: list[float], percentile: int) -> float:
    quantile = percentile / 100.0
    if quantile <= 0.0:
        return float(sorted_values[0])
    if quantile >= 1.0:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value * (1.0 - fraction) + upper_value * fraction


def test_extract_run_generates_run_and_agent_output_throughput_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    token_a_hash = hashlib.sha256(b"token-a").hexdigest()
    token_b_hash = hashlib.sha256(b"token-b").hexdigest()

    run_a_dir = run_dir / "gateway-output" / "profile-0" / "run_001"
    run_b_dir = run_dir / "gateway-output" / "profile-1" / "run_002"

    _write_jsonl(
        run_a_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "api_token_hash": token_a_hash},
            {"event_type": "job_end", "api_token_hash": token_a_hash},
        ],
    )
    _write_jsonl(
        run_b_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "api_token_hash": token_b_hash},
        ],
    )
    (run_dir / "replay").mkdir(parents=True)
    (run_dir / "replay" / "summary.json").write_text(
        json.dumps(
            {
                "worker_results": {
                    "worker-a": {
                        "worker_id": "worker-a",
                        "api_token": "token-a",
                        "status": "completed",
                    },
                    "worker-b": {
                        "worker_id": "worker-b",
                        "api_token": "token-b",
                        "status": "timed_out",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    _write_jsonl(
        run_a_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": 1000,
                "response": {"usage": {"completion_tokens": 10}},
            },
            {
                "duration_ms": 3000,
                "response": {"usage": {"completion_tokens": 20}},
            },
        ],
    )
    _write_jsonl(
        run_b_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": 2000,
                "response": {"usage": {"completion_tokens": 8}},
            },
            {
                "request_start_time": "2026-03-08T00:00:05.000Z",
                "request_end_time": "2026-03-08T00:00:09.000Z",
                "response": {"usage": {"completion_tokens": 12}},
            },
            {
                "request_duration_ms": 500,
                "response": {},
            },
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = (
        run_dir
        / "post-processed"
        / "agent-output-throughput"
        / extract_run.DEFAULT_OUTPUT_NAME
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["agent_count"] == 2
    assert payload["request_count"] == 5
    assert payload["requests_with_output_tokens"] == 4
    assert payload["requests_with_llm_request_duration"] == 5
    assert payload["requests_with_output_tokens_and_llm_request_duration"] == 4
    assert payload["output_tokens"] == 50
    assert payload["completion_tokens"] == 50
    assert payload["llm_request_duration_s"] == 10.5
    assert payload["output_throughput_tokens_per_s"] == pytest.approx(50.0 / 10.5)

    throughput_summary = payload["agent_output_throughput_tokens_per_s_summary"]
    assert throughput_summary["sample_count"] == 2
    assert throughput_summary["avg"] == pytest.approx((7.5 + 20.0 / 6.5) / 2.0)
    assert throughput_summary["min"] == pytest.approx(20.0 / 6.5)
    assert throughput_summary["max"] == pytest.approx(7.5)
    expected_std = math.sqrt(
        (((7.5 - throughput_summary["avg"]) ** 2) + (((20.0 / 6.5) - throughput_summary["avg"]) ** 2))
        / 2.0
    )
    assert throughput_summary["std"] == pytest.approx(expected_std)
    sorted_throughputs = sorted([7.5, 20.0 / 6.5])
    assert set(throughput_summary["percentiles"]) == {
        str(percentile) for percentile in extract_run.DEFAULT_SUMMARY_PERCENTILES
    }
    for percentile in extract_run.DEFAULT_SUMMARY_PERCENTILES:
        assert throughput_summary["percentiles"][str(percentile)] == pytest.approx(
            _expected_percentile(sorted_throughputs, percentile)
        )

    histogram = payload["agent_output_throughput_tokens_per_s_histogram"]
    assert histogram["metric"] == "output_throughput_tokens_per_s"
    assert histogram["bin_size"] == 1.0
    assert histogram["sample_count"] == 2
    assert histogram["bin_count"] == 5
    assert histogram["min"] == pytest.approx(20.0 / 6.5)
    assert histogram["max"] == pytest.approx(7.5)
    assert histogram["bins"] == [
        {"bin_start": 3.0, "bin_end": 4.0, "count": 1},
        {"bin_start": 4.0, "bin_end": 5.0, "count": 0},
        {"bin_start": 5.0, "bin_end": 6.0, "count": 0},
        {"bin_start": 6.0, "bin_end": 7.0, "count": 0},
        {"bin_start": 7.0, "bin_end": 8.0, "count": 1},
    ]

    agents = payload["agents"]
    assert [agent["gateway_run_id"] for agent in agents] == ["run_001", "run_002"]
    assert [agent["gateway_profile_id"] for agent in agents] == [0, 1]
    assert [agent["api_token_hash"] for agent in agents] == [token_a_hash, token_b_hash]
    assert [agent["replay_worker_status"] for agent in agents] == ["completed", "timed_out"]
    assert [agent["replay_completed"] for agent in agents] == [True, False]
    assert payload["multi_profile"] is True
    assert payload["port_profile_ids"] == [0, 1]
    assert payload["series_keys"] == ["profile-0", "profile-1"]

    profile_0 = payload["series_by_profile"]["profile-0"]
    assert profile_0["gateway_profile_id"] == 0
    assert profile_0["agent_count"] == 1
    assert profile_0["request_count"] == 2
    assert profile_0["output_tokens"] == 30
    assert profile_0["llm_request_duration_s"] == 4.0
    assert profile_0["output_throughput_tokens_per_s"] == pytest.approx(7.5)
    assert [agent["gateway_run_id"] for agent in profile_0["agents"]] == ["run_001"]

    profile_1 = payload["series_by_profile"]["profile-1"]
    assert profile_1["gateway_profile_id"] == 1
    assert profile_1["agent_count"] == 1
    assert profile_1["request_count"] == 3
    assert profile_1["output_tokens"] == 20
    assert profile_1["llm_request_duration_s"] == 6.5
    assert profile_1["output_throughput_tokens_per_s"] == pytest.approx(20.0 / 6.5)
    assert [agent["gateway_run_id"] for agent in profile_1["agents"]] == ["run_002"]

    assert agents[0]["request_count"] == 2
    assert agents[0]["output_tokens"] == 30
    assert agents[0]["llm_request_duration_s"] == 4.0
    assert agents[0]["output_throughput_tokens_per_s"] == pytest.approx(7.5)

    assert agents[1]["request_count"] == 3
    assert agents[1]["requests_with_output_tokens"] == 2
    assert agents[1]["requests_with_llm_request_duration"] == 3
    assert agents[1]["requests_with_output_tokens_and_llm_request_duration"] == 2
    assert agents[1]["output_tokens"] == 20
    assert agents[1]["llm_request_duration_s"] == 6.5
    assert agents[1]["output_throughput_tokens_per_s"] == pytest.approx(20.0 / 6.5)


def test_extract_run_supports_non_cluster_gateway_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    gateway_run_dir = run_dir / "gateway-output" / "run_001"

    _write_jsonl(
        gateway_run_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": "2000",
                "response": {"usage": {"completion_tokens": 5.0}},
            }
        ],
    )

    payload = extract_run.extract_agent_output_throughput_from_run_dir(run_dir)

    assert payload["agent_count"] == 1
    assert payload["request_count"] == 1
    assert payload["output_tokens"] == 5
    assert payload["llm_request_duration_s"] == 2.0
    assert payload["output_throughput_tokens_per_s"] == pytest.approx(2.5)
    assert payload["multi_profile"] is False
    assert payload["port_profile_ids"] == []
    assert payload["series_keys"] == []
    assert payload["series_by_profile"] == {}
    assert payload["agents"][0]["gateway_profile_id"] is None
    assert payload["agents"][0]["api_token_hash"] is None
    assert payload["agents"][0]["replay_worker_status"] is None
    assert payload["agents"][0]["replay_completed"] is None


def test_extract_run_reads_profile_id_from_gateway_run_manifest_in_flat_layout(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    run_a_dir = run_dir / "gateway-output" / "run_001"
    run_b_dir = run_dir / "gateway-output" / "run_002"

    (run_a_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run_a_dir / "manifest.json").write_text(
        json.dumps(
            {
                "backend_port_profile_id": "2",
                "api_token_hash": "hash-a",
            }
        ),
        encoding="utf-8",
    )
    (run_b_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run_b_dir / "manifest.json").write_text(
        json.dumps(
            {
                "backend_port_profile_id": 13,
                "api_token_hash": "hash-b",
            }
        ),
        encoding="utf-8",
    )

    _write_jsonl(
        run_a_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": 1000,
                "response": {"usage": {"completion_tokens": 10}},
            }
        ],
    )
    _write_jsonl(
        run_b_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": 2000,
                "response": {"usage": {"completion_tokens": 8}},
            }
        ],
    )

    payload = extract_run.extract_agent_output_throughput_from_run_dir(run_dir)

    assert payload["multi_profile"] is True
    assert payload["port_profile_ids"] == [2, 13]
    assert payload["series_keys"] == ["profile-2", "profile-13"]
    assert [agent["gateway_profile_id"] for agent in payload["agents"]] == [2, 13]
    assert payload["series_by_profile"]["profile-2"]["gateway_profile_id"] == 2
    assert payload["series_by_profile"]["profile-13"]["gateway_profile_id"] == 13


def test_summarize_values_includes_every_5_percentile() -> None:
    values = [1.0, 2.0, 4.0, 8.0]

    summary = extract_run._summarize_values(values)

    assert summary["sample_count"] == 4
    assert summary["avg"] == pytest.approx(3.75)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(8.0)
    assert set(summary["percentiles"]) == {
        str(percentile) for percentile in extract_run.DEFAULT_SUMMARY_PERCENTILES
    }
    sorted_values = sorted(values)
    for percentile in extract_run.DEFAULT_SUMMARY_PERCENTILES:
        assert summary["percentiles"][str(percentile)] == pytest.approx(
            _expected_percentile(sorted_values, percentile)
        )


def test_extract_run_dir_writes_separate_per_profile_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_a_dir = run_dir / "gateway-output" / "run_001"
    run_b_dir = run_dir / "gateway-output" / "run_002"

    (run_a_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run_a_dir / "manifest.json").write_text(
        json.dumps({"backend_port_profile_id": 2}),
        encoding="utf-8",
    )
    (run_b_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run_b_dir / "manifest.json").write_text(
        json.dumps({"backend_port_profile_id": 13}),
        encoding="utf-8",
    )

    _write_jsonl(
        run_a_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": 1000,
                "response": {"usage": {"completion_tokens": 10}},
            }
        ],
    )
    _write_jsonl(
        run_b_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_duration_ms": 2000,
                "response": {"usage": {"completion_tokens": 8}},
            }
        ],
    )

    output_path = extract_run.extract_run_dir(run_dir)
    aggregate_payload = json.loads(output_path.read_text(encoding="utf-8"))

    profile_2_path = output_path.parent / "profile-2" / output_path.name
    profile_13_path = output_path.parent / "profile-13" / output_path.name

    assert profile_2_path.is_file()
    assert profile_13_path.is_file()

    profile_2_payload = json.loads(profile_2_path.read_text(encoding="utf-8"))
    profile_13_payload = json.loads(profile_13_path.read_text(encoding="utf-8"))

    assert profile_2_payload == aggregate_payload["series_by_profile"]["profile-2"]
    assert profile_13_payload == aggregate_payload["series_by_profile"]["profile-13"]
    assert profile_2_payload["gateway_profile_id"] == 2
    assert profile_13_payload["gateway_profile_id"] == 13


def test_extract_run_applies_service_failure_cutoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    gateway_run_dir = run_dir / "gateway-output" / "run_001"
    sbatch_logs_dir = run_dir / "sbatch-logs"

    _write_jsonl(
        gateway_run_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_start_time": "2026-03-08T00:00:01.000Z",
                "request_end_time": "2026-03-08T00:00:02.000Z",
                "response": {"usage": {"completion_tokens": 4}},
            },
            {
                "request_start_time": "2026-03-08T00:00:05.000Z",
                "request_end_time": "2026-03-08T00:00:06.000Z",
                "response": {"usage": {"completion_tokens": 9}},
            },
        ],
    )
    sbatch_logs_dir.mkdir(parents=True)
    (sbatch_logs_dir / "vllm.1.log").write_text(
        "2026-03-08T00:00:03Z AsyncLLM output_handler failed.\n",
        encoding="utf-8",
    )

    payload = extract_run.extract_agent_output_throughput_from_run_dir(run_dir)

    assert payload["service_failure_detected"] is True
    assert payload["service_failure_cutoff_time_utc"] == "2026-03-08T00:00:03Z"
    assert payload["request_count"] == 1
    assert payload["output_tokens"] == 4
    assert payload["llm_request_duration_s"] == 1.0
    assert payload["output_throughput_tokens_per_s"] == pytest.approx(4.0)


def test_discover_run_dirs_with_gateway_output_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_gateway_output(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return (
            run_dir
            / "post-processed"
            / "agent-output-throughput"
            / extract_run.DEFAULT_OUTPUT_NAME
        )

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]


def test_extract_run_rejects_dry_run_for_single_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="--dry-run can only be used with --root-dir"):
        extract_run.main(["--run-dir", str(run_dir), "--dry-run"])


def test_extract_run_rejects_file_path_for_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)
    file_path = run_dir / "not-a-dir.json"
    file_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="--run-dir must point to a directory"):
        extract_run.main(["--run-dir", str(file_path)])
