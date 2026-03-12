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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_gateway_job(
    run_dir: Path,
    *,
    profile_id: int | None,
    gateway_run_id: str,
    prompt_tokens: list[int],
    decode_tokens: list[int],
    cached_tokens: list[int],
    lifecycle_start: str,
    lifecycle_end: str,
) -> None:
    if profile_id is None:
        gateway_run_dir = run_dir / "gateway-output" / gateway_run_id
    else:
        gateway_run_dir = (
            run_dir
            / "gateway-output"
            / f"profile-{profile_id}"
            / gateway_run_id
        )

    request_records: list[dict[str, object]] = []
    for prompt, decode, cached in zip(prompt_tokens, decode_tokens, cached_tokens):
        request_records.append(
            {
                "request_start_time": lifecycle_start,
                "request_end_time": lifecycle_end,
                "response": {
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": decode,
                        "prompt_tokens_details": {
                            "cached_tokens": cached,
                        },
                    }
                },
            }
        )

    _write_jsonl(
        gateway_run_dir / "requests" / "model_inference.jsonl",
        request_records,
    )
    _write_jsonl(
        gateway_run_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "timestamp": lifecycle_start},
            {"event_type": "job_end", "timestamp": lifecycle_end},
        ],
    )


def test_extract_run_generates_expected_split_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=0,
        gateway_run_id="run_low",
        prompt_tokens=[5, 3],
        decode_tokens=[2, 1],
        cached_tokens=[1, 0],
        lifecycle_start="2026-03-08T00:00:00.000Z",
        lifecycle_end="2026-03-08T00:00:10.000Z",
    )
    _write_gateway_job(
        run_dir,
        profile_id=1,
        gateway_run_id="run_high",
        prompt_tokens=[20],
        decode_tokens=[10],
        cached_tokens=[5],
        lifecycle_start="2026-03-08T00:00:00.000Z",
        lifecycle_end="2026-03-08T00:00:20.000Z",
    )

    exit_code = extract_run.main(
        ["--run-dir", str(run_dir), "--split-count", "2"]
    )
    assert exit_code == 0

    output_path = (
        run_dir
        / "post-processed"
        / "split"
        / "duration"
        / "duration-split-summary.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["split_count"] == 2
    assert payload["bin_labels"] == ["0-50%", "50-100%"]
    assert payload["job_count"] == 2

    duration_table = payload["tables"]["duration_s"]
    assert duration_table["0-50%"]["avg"] == 10.0
    assert duration_table["50-100%"]["avg"] == 20.0

    turns_table = payload["tables"]["turn_count"]
    assert turns_table["0-50%"]["avg"] == 2.0
    assert turns_table["50-100%"]["avg"] == 1.0

    prompt_table = payload["tables"]["prompt_tokens"]
    assert prompt_table["0-50%"]["avg"] == 8.0
    assert prompt_table["50-100%"]["avg"] == 20.0

    decode_table = payload["tables"]["decode_tokens"]
    assert decode_table["0-50%"]["avg"] == 3.0
    assert decode_table["50-100%"]["avg"] == 10.0

    cached_table = payload["tables"]["cached_prompt_tokens"]
    assert cached_table["0-50%"]["avg"] == 1.0
    assert cached_table["50-100%"]["avg"] == 5.0


def test_discover_run_dirs_with_gateway_output_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "nested" / "b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    discovered = extract_run.discover_run_dirs_with_gateway_output(root_dir)
    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    (run_a / "gateway-output").mkdir(parents=True)
    (run_b / "gateway-output").mkdir(parents=True)

    processed: list[tuple[Path, Path | None, int]] = []

    def fake_extract_run_dir(
        run_dir: Path,
        *,
        output_path: Path | None = None,
        split_count: int = extract_run.DEFAULT_SPLIT_COUNT,
    ) -> Path:
        processed.append((run_dir, output_path, split_count))
        return run_dir / "post-processed" / "split" / "duration" / "duration-split-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(
        ["--root-dir", str(root_dir), "--max-procs", "1", "--split-count", "3"]
    )

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None, 3),
        (run_b.resolve(), None, 3),
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
        extract_run.main(
            ["--root-dir", str(root_dir), "--output", str(tmp_path / "out.json")]
        )
    except ValueError as exc:
        assert "--output can only be used with --run-dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError when --output is used with --root-dir")


def test_extract_uses_agent_window_when_present(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    gateway_run_dir = run_dir / "gateway-output" / "run_001"
    _write_jsonl(
        gateway_run_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_start_time": "2026-03-08T00:00:02.000Z",
                "request_end_time": "2026-03-08T00:00:07.000Z",
                "response": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 1},
                    }
                },
            }
        ],
    )
    _write_jsonl(
        gateway_run_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "timestamp": "2026-03-08T00:00:00.000Z"},
            {"event_type": "agent_start", "timestamp": "2026-03-08T00:00:01.000Z"},
            {"event_type": "agent_end", "timestamp": "2026-03-08T00:00:04.500Z"},
            {"event_type": "job_end", "timestamp": "2026-03-08T00:00:10.000Z"},
        ],
    )

    payload = extract_run.extract_split_duration_from_run_dir(run_dir, split_count=1)

    assert payload["job_count"] == 1
    # Should use agent_start->agent_end, not job_start->job_end.
    assert payload["jobs"][0]["duration_s"] == 3.5


def test_extract_excludes_jobs_with_no_token_usage(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_gateway_job(
        run_dir,
        profile_id=None,
        gateway_run_id="run_valid",
        prompt_tokens=[10],
        decode_tokens=[5],
        cached_tokens=[1],
        lifecycle_start="2026-03-08T00:00:00.000Z",
        lifecycle_end="2026-03-08T00:00:10.000Z",
    )

    run_invalid_dir = run_dir / "gateway-output" / "run_invalid"
    _write_jsonl(
        run_invalid_dir / "requests" / "model_inference.jsonl",
        [
            {
                "request_start_time": "2026-03-08T00:00:00.000Z",
                "request_end_time": "2026-03-08T00:00:30.000Z",
                "response": {"usage": {}},
            },
            {
                "request_start_time": "2026-03-08T00:01:00.000Z",
                "request_end_time": "2026-03-08T00:01:30.000Z",
                "response": {"usage": {}},
            },
        ],
    )
    _write_jsonl(
        run_invalid_dir / "events" / "lifecycle.jsonl",
        [
            {"event_type": "job_start", "timestamp": "2026-03-08T00:00:00.000Z"},
            {"event_type": "job_end", "timestamp": "2026-03-08T00:02:00.000Z"},
        ],
    )

    payload = extract_run.extract_split_duration_from_run_dir(run_dir, split_count=1)

    assert payload["job_count_total"] == 2
    assert payload["job_count"] == 1
    assert payload["job_count_excluded_no_token_usage"] == 1
    assert [job["gateway_run_id"] for job in payload["jobs"]] == ["run_valid"]
    assert [job["gateway_run_id"] for job in payload["excluded_jobs_no_token_usage"]] == [
        "run_invalid"
    ]

    duration_table = payload["tables"]["duration_s"]
    assert duration_table["0-100%"]["count"] == 1
    assert duration_table["0-100%"]["avg"] == 10.0
