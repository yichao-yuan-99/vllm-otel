from __future__ import annotations

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


def test_extract_run_generates_run_and_agent_output_throughput_summary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"

    run_a_dir = run_dir / "gateway-output" / "profile-0" / "run_001"
    run_b_dir = run_dir / "gateway-output" / "profile-1" / "run_002"

    _write_jsonl(
        run_a_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "api_token_hash": "token-a"},
            {"event_type": "job_end", "api_token_hash": "token-a"},
        ],
    )
    _write_jsonl(
        run_b_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "api_token_hash": "token-b"},
        ],
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
    assert [agent["api_token_hash"] for agent in agents] == ["token-a", "token-b"]

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
    assert payload["agents"][0]["gateway_profile_id"] is None
    assert payload["agents"][0]["api_token_hash"] is None


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
