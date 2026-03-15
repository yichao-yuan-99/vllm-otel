from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MODULE_ROOT = THIS_DIR.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import extract_run


def test_extract_run_without_sbatch_logs_reports_no_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir(parents=True)

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = run_dir / "post-processed" / "service-failure" / "service-failure.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sbatch_logs_exists"] is False
    assert payload["service_failure_detected"] is False
    assert payload["cutoff_time_utc"] is None
    assert payload["warning_count"] == 0
    assert payload["warnings_preview"] == []


def test_extract_run_detects_failure_and_cutoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    sbatch_logs_dir = run_dir / "sbatch-logs"
    sbatch_logs_dir.mkdir(parents=True)
    (sbatch_logs_dir / "vllm.1.log").write_text(
        "\n".join(
            [
                "(APIServer pid=1) INFO 03-14 03:18:16 [x.py:1] healthy",
                "(APIServer pid=1) ERROR 03-14 03:18:17 [async_llm.py:708] AsyncLLM output_handler failed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = run_dir / "post-processed" / "service-failure" / "service-failure.json"
    resolved = extract_run.extract_run_dir(run_dir)
    assert resolved == output_path.resolve()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sbatch_logs_exists"] is True
    assert payload["service_failure_detected"] is True
    assert payload["cutoff_time_utc"] is not None
    assert payload["matched_rule"] == "async_llm_output_handler_failed"
    assert payload["matched_log_path"].endswith("vllm.1.log")
    assert payload["event_count"] >= 1
    assert payload["warning_count"] == 0
    assert payload["warnings_preview"] == []


def test_extract_run_counter_negative_increment_is_warning_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    sbatch_logs_dir = run_dir / "sbatch-logs"
    sbatch_logs_dir.mkdir(parents=True)
    (sbatch_logs_dir / "vllm.1.log").write_text(
        "\n".join(
            [
                "(APIServer pid=1) INFO 03-14 03:18:16 [x.py:1] healthy",
                "(APIServer pid=1) Counters can only be incremented by non-negative amounts. value -96",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = extract_run.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = run_dir / "post-processed" / "service-failure" / "service-failure.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sbatch_logs_exists"] is True
    assert payload["service_failure_detected"] is False
    assert payload["cutoff_time_utc"] is None
    assert payload["matched_rule"] is None
    assert payload["event_count"] == 0
    assert payload["warning_count"] == 1
    assert payload["warnings_preview"][0]["rule"] == "counter_negative_increment"
