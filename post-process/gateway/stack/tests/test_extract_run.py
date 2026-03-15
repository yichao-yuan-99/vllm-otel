from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def _write_llm_requests(path: Path, requests: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "request_count": len(requests),
        "requests": requests,
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_extract_run_generates_ranges_and_histograms(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    llm_requests_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME
    )

    _write_llm_requests(
        llm_requests_path,
        [
            {
                "gateway_run_id": "run_001",
                "gateway_profile_id": 0,
                "request_id": "req-1",
                "trace_id": "trace-1",
                "request_start_offset_s": 0.0,
                "prompt_tokens": 10,
                "cached_tokens": 4,
                "completion_tokens": 9,
                "gen_ai.latency.time_in_queue": 0.2,
                "gen_ai.latency.time_in_model_prefill": 1.0,
                "gen_ai.latency.time_to_first_token": 0.5,
                "gen_ai.latency.time_in_model_decode": 1.5,
            },
            {
                "gateway_run_id": "run_001",
                "gateway_profile_id": 0,
                "request_id": "req-2",
                "trace_id": "trace-2",
                "request_start_offset_s": 0.6,
                "prompt_tokens": 5,
                "cached_tokens": 1,
                "completion_tokens": 2,
                "gen_ai.latency.time_in_queue": 0.2,
                "gen_ai.latency.time_in_model_prefill": 1.0,
                "gen_ai.latency.time_to_first_token": 0.6,
                "gen_ai.latency.time_in_model_decode": 0.5,
            },
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_dir = run_dir / "post-processed" / "gateway" / "stack"
    prompt_ranges = json.loads(
        (output_dir / extract_run.RANGE_OUTPUT_NAMES["prompt_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    cached_ranges = json.loads(
        (output_dir / extract_run.RANGE_OUTPUT_NAMES["cached_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    compute_ranges = json.loads(
        (output_dir / extract_run.RANGE_OUTPUT_NAMES["compute_prompt_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    combined_ranges = json.loads(
        (
            output_dir
            / extract_run.RANGE_OUTPUT_NAMES["compute_prompt_plus_completion_tokens"]
        ).read_text(encoding="utf-8")
    )
    completion_histogram = json.loads(
        (output_dir / extract_run.HISTOGRAM_OUTPUT_NAMES["completion_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    prompt_histogram = json.loads(
        (output_dir / extract_run.HISTOGRAM_OUTPUT_NAMES["prompt_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    cached_histogram = json.loads(
        (output_dir / extract_run.HISTOGRAM_OUTPUT_NAMES["cached_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    compute_histogram = json.loads(
        (output_dir / extract_run.HISTOGRAM_OUTPUT_NAMES["compute_prompt_tokens"]).read_text(
            encoding="utf-8"
        )
    )
    combined_histogram = json.loads(
        (
            output_dir
            / extract_run.HISTOGRAM_OUTPUT_NAMES["compute_prompt_plus_completion_tokens"]
        ).read_text(encoding="utf-8")
    )

    assert prompt_ranges["entry_count"] == 2
    assert cached_ranges["entry_count"] == 2
    assert compute_ranges["entry_count"] == 2
    assert combined_ranges["entry_count"] == 4

    assert prompt_ranges["entries"][0]["request_id"] == "req-1"
    assert prompt_ranges["entries"][0]["range_start_s"] == pytest.approx(0.2)
    assert prompt_ranges["entries"][0]["range_end_s"] == pytest.approx(1.2)
    assert prompt_ranges["entries"][0]["avg_value_per_s"] == pytest.approx(10.0)

    assert compute_ranges["entries"][0]["total_value"] == pytest.approx(6.0)
    assert compute_ranges["entries"][1]["total_value"] == pytest.approx(4.0)
    assert [entry["total_value"] for entry in combined_ranges["entries"]] == pytest.approx(
        [6.0, 9.0, 4.0, 2.0]
    )

    completion_points = completion_histogram["points"]
    assert len(completion_points) == 2
    assert completion_points[0]["accumulated_value"] == pytest.approx(3.0)
    assert completion_points[1]["accumulated_value"] == pytest.approx(8.0)

    prompt_points = prompt_histogram["points"]
    assert len(prompt_points) == 2
    assert prompt_points[0]["accumulated_value"] == pytest.approx(9.0)
    assert prompt_points[1]["accumulated_value"] == pytest.approx(6.0)

    cached_points = cached_histogram["points"]
    assert len(cached_points) == 2
    assert cached_points[0]["accumulated_value"] == pytest.approx(3.4)
    assert cached_points[1]["accumulated_value"] == pytest.approx(1.6)

    compute_points = compute_histogram["points"]
    assert len(compute_points) == 2
    assert compute_points[0]["accumulated_value"] == pytest.approx(5.6)
    assert compute_points[1]["accumulated_value"] == pytest.approx(4.4)

    combined_points = combined_histogram["points"]
    assert len(combined_points) == 2
    assert combined_points[0]["accumulated_value"] == pytest.approx(8.6)
    assert combined_points[1]["accumulated_value"] == pytest.approx(12.4)


def test_missing_cached_tokens_defaults_to_zero_for_prefill_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    llm_requests_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME
    )

    _write_llm_requests(
        llm_requests_path,
        [
            {
                "request_id": "req-1",
                "request_start_offset_s": 1.0,
                "prompt_tokens": 8,
                "completion_tokens": 0,
                "gen_ai.latency.time_in_queue": 0.0,
                "gen_ai.latency.time_in_model_prefill": 2.0,
            }
        ],
    )

    range_payloads, _histogram_payloads = extract_run.extract_gateway_stack_from_run_dir(run_dir)

    cached_entries = range_payloads["cached_tokens"]["entries"]
    assert len(cached_entries) == 1
    assert cached_entries[0]["total_value"] == pytest.approx(0.0)
    assert cached_entries[0]["avg_value_per_s"] == pytest.approx(0.0)

    compute_entries = range_payloads["compute_prompt_tokens"]["entries"]
    assert len(compute_entries) == 1
    assert compute_entries[0]["total_value"] == pytest.approx(8.0)
    assert compute_entries[0]["avg_value_per_s"] == pytest.approx(4.0)

    combined_entries = range_payloads["compute_prompt_plus_completion_tokens"]["entries"]
    assert len(combined_entries) == 1
    assert combined_entries[0]["total_value"] == pytest.approx(8.0)
    assert combined_entries[0]["avg_value_per_s"] == pytest.approx(4.0)


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "job-b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    processed: list[tuple[Path, Path | None, Path | None]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        llm_requests_path: Path | None = None,
    ) -> list[Path]:
        processed.append((run_dir, output_dir, llm_requests_path))
        return [run_dir / "post-processed" / "gateway" / "stack" / "prompt-tokens-ranges.json"]

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
        extract_run.main(["--root-dir", str(root_dir), "--output-dir", str(tmp_path / "out")])
    except ValueError as exc:
        assert "--output-dir can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output-dir is used with --root-dir")

    try:
        extract_run.main([
            "--root-dir",
            str(root_dir),
            "--llm-requests",
            str(tmp_path / "llm-requests.json"),
        ])
    except ValueError as exc:
        assert "--llm-requests can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --llm-requests is used with --root-dir")
