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


def test_extract_run_generates_run_and_agent_usage_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"

    run_a_dir = run_dir / "gateway-output" / "profile-0" / "run_001"
    run_b_dir = run_dir / "gateway-output" / "profile-1" / "run_002"
    for gateway_run_dir in [run_a_dir, run_b_dir]:
        (gateway_run_dir / "requests").mkdir(parents=True)
        (gateway_run_dir / "events").mkdir(parents=True)

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
                "response": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    }
                }
            },
            {
                "response": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    }
                }
            },
        ],
    )
    _write_jsonl(
        run_b_dir / "requests" / "model_inference.jsonl",
        [
            {
                "response": {
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "prompt_tokens_details": {"cached_tokens": 1},
                    }
                }
            },
            {
                "response": {
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                    }
                }
            },
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = run_dir / "post-processed" / "gateway" / "usage" / "usage-summary.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["agent_count"] == 2
    assert payload["request_count"] == 4
    assert payload["usage"] == {
        "prompt_tokens": 40,
        "generation_tokens": 11,
        "completion_tokens": 11,
        "cached_prompt_tokens": 7,
        "prefill_prompt_tokens": 33,
        "max_request_length": 25,
        "avg_worker_max_request_length": 17.0,
        "requests_with_prompt_tokens": 4,
        "requests_with_generation_tokens": 4,
        "requests_with_completion_tokens": 4,
        "requests_with_cached_prompt_tokens": 3,
        "requests_with_request_length": 4,
    }

    agents = payload["agents"]
    assert [agent["gateway_run_id"] for agent in agents] == ["run_001", "run_002"]
    assert [agent["gateway_profile_id"] for agent in agents] == [0, 1]
    assert [agent["api_token_hash"] for agent in agents] == ["token-a", "token-b"]

    assert agents[0]["request_count"] == 2
    assert agents[0]["usage"]["prompt_tokens"] == 30
    assert agents[0]["usage"]["generation_tokens"] == 8
    assert agents[0]["usage"]["cached_prompt_tokens"] == 6
    assert agents[0]["usage"]["prefill_prompt_tokens"] == 24
    assert agents[0]["usage"]["max_request_length"] == 25
    assert agents[0]["usage"]["requests_with_request_length"] == 2

    assert agents[1]["request_count"] == 2
    assert agents[1]["usage"]["prompt_tokens"] == 10
    assert agents[1]["usage"]["generation_tokens"] == 3
    assert agents[1]["usage"]["cached_prompt_tokens"] == 1
    assert agents[1]["usage"]["prefill_prompt_tokens"] == 9
    assert agents[1]["usage"]["max_request_length"] == 9
    assert agents[1]["usage"]["requests_with_request_length"] == 2


def test_extract_run_supports_non_cluster_gateway_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    gateway_run_dir = run_dir / "gateway-output" / "run_001"
    (gateway_run_dir / "requests").mkdir(parents=True)

    _write_jsonl(
        gateway_run_dir / "requests" / "model_inference.jsonl",
        [
            {
                "response": {
                    "usage": {
                        "prompt_tokens": "5",
                        "completion_tokens": 1.0,
                        "prompt_tokens_details": {"cached_tokens": "2"},
                    }
                }
            }
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    payload = json.loads(
        (run_dir / "post-processed" / "gateway" / "usage" / "usage-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["agent_count"] == 1
    assert payload["request_count"] == 1
    assert payload["agents"][0]["gateway_profile_id"] is None
    assert payload["agents"][0]["api_token_hash"] is None
    assert payload["usage"]["prompt_tokens"] == 5
    assert payload["usage"]["generation_tokens"] == 1
    assert payload["usage"]["cached_prompt_tokens"] == 2
    assert payload["usage"]["prefill_prompt_tokens"] == 3
    assert payload["usage"]["max_request_length"] == 6
    assert payload["usage"]["avg_worker_max_request_length"] == 6.0


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

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "post-processed" / "gateway" / "usage" / "usage-summary.json"

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
