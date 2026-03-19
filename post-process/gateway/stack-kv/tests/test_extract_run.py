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


def _write_llm_requests(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_extract_run_generates_kv_ranges_and_histogram(tmp_path: Path) -> None:
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
        {
            "request_count": 4,
            "service_failure_detected": False,
            "service_failure_cutoff_time_utc": None,
            "requests": [
                {
                    "gateway_run_id": "run_a",
                    "gateway_profile_id": 0,
                    "request_id": "req-1",
                    "trace_id": "trace-1",
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 2.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                },
                {
                    "gateway_run_id": "run_b",
                    "gateway_profile_id": 1,
                    "request_id": "req-2",
                    "trace_id": "trace-2",
                    "request_start_offset_s": 1.0,
                    "request_end_offset_s": 3.0,
                    "prompt_tokens": 50,
                    "completion_tokens": 5,
                },
                {
                    "gateway_run_id": "run_c",
                    "gateway_profile_id": 2,
                    "request_id": "req-3",
                    "trace_id": "trace-3",
                    "request_start_offset_s": 2.5,
                    "request_end_offset_s": 4.0,
                    "prompt_tokens": 30,
                },
                {
                    "gateway_run_id": "run_d",
                    "gateway_profile_id": 3,
                    "request_id": "req-4",
                    "trace_id": "trace-4",
                    "request_start_offset_s": 5.0,
                    "request_end_offset_s": 5.0,
                    "prompt_tokens": 999,
                    "completion_tokens": 1,
                },
            ],
        },
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_dir = run_dir / "post-processed" / "gateway" / "stack-kv"
    ranges_payload = json.loads((output_dir / extract_run.RANGES_OUTPUT_NAME).read_text("utf-8"))
    histogram_payload = json.loads(
        (output_dir / extract_run.HISTOGRAM_OUTPUT_NAME).read_text("utf-8")
    )

    assert ranges_payload["entry_count"] == 3
    assert histogram_payload["point_count"] == 4
    assert histogram_payload["metric"] == "kv_usage_tokens"

    by_request = {
        entry["request_id"]: entry
        for entry in ranges_payload["entries"]
        if isinstance(entry.get("request_id"), str)
    }
    assert by_request["req-1"]["range_start_s"] == pytest.approx(0.0)
    assert by_request["req-1"]["range_end_s"] == pytest.approx(2.0)
    assert by_request["req-1"]["kv_usage_tokens"] == pytest.approx(110.0)
    assert by_request["req-1"]["total_value"] == pytest.approx(220.0)

    assert by_request["req-2"]["range_start_s"] == pytest.approx(1.0)
    assert by_request["req-2"]["range_end_s"] == pytest.approx(3.0)
    assert by_request["req-2"]["kv_usage_tokens"] == pytest.approx(55.0)

    assert by_request["req-3"]["range_start_s"] == pytest.approx(2.5)
    assert by_request["req-3"]["range_end_s"] == pytest.approx(4.0)
    assert by_request["req-3"]["kv_usage_tokens"] == pytest.approx(30.0)

    points = histogram_payload["points"]
    assert [point["second"] for point in points] == [0, 1, 2, 3]
    assert [point["accumulated_value"] for point in points] == pytest.approx(
        [110.0, 165.0, 70.0, 30.0]
    )


def test_extract_gateway_stack_kv_from_run_dir_uses_source_gateway_output_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "job"
    source_gateway_output_dir = run_dir / "custom-gateway-output"
    source_gateway_output_dir.mkdir(parents=True, exist_ok=True)

    llm_requests_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME
    )
    _write_llm_requests(
        llm_requests_path,
        {
            "request_count": 1,
            "source_gateway_output_dir": str(source_gateway_output_dir),
            "service_failure_detected": True,
            "service_failure_cutoff_time_utc": "2026-03-18T00:00:00Z",
            "requests": [
                {
                    "request_id": "req-1",
                    "request_start_offset_s": 1.0,
                    "request_end_offset_s": 2.0,
                    "prompt_tokens": 8,
                }
            ],
        },
    )

    ranges_payload, histogram_payload = extract_run.extract_gateway_stack_kv_from_run_dir(run_dir)

    assert ranges_payload["source_gateway_output_dir"] == str(source_gateway_output_dir.resolve())
    assert histogram_payload["source_gateway_output_dir"] == str(
        source_gateway_output_dir.resolve()
    )
    assert ranges_payload["service_failure_detected"] is True
    assert histogram_payload["service_failure_cutoff_time_utc"] == "2026-03-18T00:00:00Z"
    assert ranges_payload["entry_count"] == 1
    assert ranges_payload["entries"][0]["kv_usage_tokens"] == pytest.approx(8.0)
    assert [point["accumulated_value"] for point in histogram_payload["points"]] == pytest.approx(
        [0.0, 8.0]
    )


def test_discover_run_dirs_with_llm_requests_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "job-a"
    run_b = root_dir / "nested" / "job-b"
    _write_llm_requests(
        run_a
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME,
        {"requests": []},
    )
    _write_llm_requests(
        run_b
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME,
        {"requests": []},
    )

    discovered = extract_run.discover_run_dirs_with_llm_requests(root_dir)
    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_llm_requests(
        run_a
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME,
        {"requests": []},
    )
    _write_llm_requests(
        run_b
        / "post-processed"
        / "gateway"
        / "llm-requests"
        / extract_run.DEFAULT_LLM_REQUESTS_OUTPUT_NAME,
        {"requests": []},
    )

    processed: list[tuple[Path, Path | None, Path | None]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        llm_requests_path: Path | None = None,
    ) -> list[Path]:
        processed.append((run_dir, output_dir, llm_requests_path))
        return [
            run_dir
            / "post-processed"
            / "gateway"
            / "stack-kv"
            / extract_run.RANGES_OUTPUT_NAME
        ]

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
        extract_run.main(
            [
                "--root-dir",
                str(root_dir),
                "--llm-requests",
                str(tmp_path / "llm-requests.json"),
            ]
        )
    except ValueError as exc:
        assert "--llm-requests can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --llm-requests is used with --root-dir")

    try:
        extract_run.main(["--root-dir", str(root_dir), "--max-procs", "0"])
    except ValueError as exc:
        assert "--max-procs must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --max-procs <= 0")
