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


def _write_minimal_meta(
    run_dir: Path,
    *,
    dataset: str,
    score: float,
    result_count: int,
    agent_type: str = "mini-swe-agent",
) -> None:
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_mode_dataset": dataset,
                "reward_avg": score,
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "results.json").write_text(
        json.dumps(
            [
                {
                    "dataset": dataset,
                    "trial_id": f"trial-{idx}",
                    "command": ["harbor", "trials", "start", "--agent-name", agent_type],
                }
                for idx in range(result_count)
            ]
        ),
        encoding="utf-8",
    )


def test_extract_run_collects_expected_stats(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    _write_minimal_meta(run_dir, dataset="demo-dataset", score=0.42, result_count=2)

    run_a_requests_path = (
        run_dir
        / "gateway-output"
        / "profile-0"
        / "run_001"
        / "requests"
        / "model_inference.jsonl"
    )
    run_b_requests_path = (
        run_dir
        / "gateway-output"
        / "profile-1"
        / "run_002"
        / "requests"
        / "model_inference.jsonl"
    )
    _write_jsonl(
        run_a_requests_path,
        [
            {"response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}},
            {"response": {"usage": {"prompt_tokens": 20, "completion_tokens": 3}}},
        ],
    )
    _write_jsonl(
        run_b_requests_path,
        [
            {"response": {"usage": {"prompt_tokens": 7, "completion_tokens": 2}}},
            {"response": {"usage": {"prompt_tokens": 3, "completion_tokens": 1}}},
        ],
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = run_dir / "run-stats" / "run-stats-summary.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "demo-dataset"
    assert payload["score"] == 0.42
    assert payload["agent_type"] == "mini-swe-agent"
    assert payload["job_count"] == 2
    assert payload["results_count"] == 2
    assert payload["job_max_request_lengths"] == [23, 9]
    assert payload["avg_job_max_request_length"] == 16.0
    assert payload["max_job_max_request_length"] == 23
    assert payload["avg_turns_per_run"] == 2.0
    assert payload["max_turns_per_run"] == 2
    assert payload["job_avg_prompt_tokens_per_request"] == [15.0, 5.0]
    assert payload["job_avg_generation_tokens_per_request"] == [4.0, 1.5]
    assert payload["avg_run_prompt_tokens_per_request"] == 10.0
    assert payload["avg_run_generation_tokens_per_request"] == 2.75
    assert [job["gateway_run_id"] for job in payload["jobs"]] == ["run_001", "run_002"]


def test_discover_con_driver_run_dirs_scans_recursively(tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "nested" / "b"
    _write_minimal_meta(run_a, dataset="d1", score=0.1, result_count=1)
    _write_minimal_meta(run_b, dataset="d2", score=0.2, result_count=1)

    discovered = extract_run.discover_con_driver_run_dirs(root_dir)

    assert discovered == [run_a.resolve(), run_b.resolve()]


def test_extract_run_root_dir_processes_discovered_runs(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "results"
    run_a = root_dir / "a"
    run_b = root_dir / "b"
    _write_minimal_meta(run_a, dataset="d1", score=0.1, result_count=1)
    _write_minimal_meta(run_b, dataset="d2", score=0.2, result_count=1)

    processed: list[tuple[Path, Path | None]] = []

    def fake_extract_run_dir(run_dir: Path, *, output_path: Path | None = None) -> Path:
        processed.append((run_dir, output_path))
        return run_dir / "run-stats" / "run-stats-summary.json"

    monkeypatch.setattr(extract_run, "extract_run_dir", fake_extract_run_dir)

    exit_code = extract_run.main(["--root-dir", str(root_dir), "--max-procs", "1"])

    assert exit_code == 0
    assert processed == [
        (run_a.resolve(), None),
        (run_b.resolve(), None),
    ]


def test_extract_run_agent_type_falls_back_to_parent_dir_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "agent-x" / "job"
    _write_minimal_meta(run_dir, dataset="demo", score=0.1, result_count=1)

    # Overwrite results command with no agent flag to trigger fallback.
    (run_dir / "meta" / "results.json").write_text(
        json.dumps([{"dataset": "demo", "trial_id": "t0", "command": ["harbor", "trials", "start"]}]),
        encoding="utf-8",
    )
    _write_jsonl(
        run_dir
        / "gateway-output"
        / "run_001"
        / "requests"
        / "model_inference.jsonl",
        [{"response": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}}],
    )

    payload = extract_run.extract_run_stats_from_run_dir(run_dir)
    assert payload["agent_type"] == "agent-x"
