from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "figures"
    / "freq-vs-average-energy-integral"
    / "materialize_integral_metrics.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _make_run(
    root_dir: Path,
    *,
    batch_name: str,
    run_name: str,
    total_power_w: float,
    running_request_count: float,
    completed_trace_id: str,
    completed_trial_id: str,
    incomplete_trace_id: str,
    incomplete_trial_id: str,
) -> None:
    run_dir = root_dir / batch_name / run_name

    _write_json(
        run_dir / "post-processed" / "power" / "power-summary.json",
        {
            "analysis_window_start_utc": "2026-03-22T00:00:00Z",
            "analysis_window_end_utc": "2026-03-22T00:00:10Z",
            "power_points": [
                {"time_offset_s": 0.0, "power_w": total_power_w},
                {"time_offset_s": 5.0, "power_w": total_power_w},
                {"time_offset_s": 10.0, "power_w": total_power_w},
            ],
        },
    )

    _write_json(
        run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json",
        {
            "metrics": {
                "vllm:num_requests_running|engine=0": {
                    "name": "vllm:num_requests_running",
                    "time_from_start_s": [0.0, 5.0, 10.0],
                    "value": [running_request_count, running_request_count, running_request_count],
                }
            }
        },
    )

    _write_json(
        run_dir / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
        {
            "requests": [
                {
                    "trace_id": completed_trace_id,
                    "request_id": f"{completed_trace_id}-req-1",
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 10.0,
                },
                {
                    "trace_id": incomplete_trace_id,
                    "request_id": f"{incomplete_trace_id}-req-1",
                    "request_start_offset_s": 0.0,
                    "request_end_offset_s": 10.0,
                },
            ]
        },
    )

    _write_json(
        run_dir / "post-processed" / "global" / "trial-timing-summary.json",
        {
            "agent_time_breakdown": {
                "agents": [
                    {"trial_id": completed_trial_id, "trace_id": completed_trace_id},
                    {"trial_id": incomplete_trial_id, "trace_id": incomplete_trace_id},
                ]
            }
        },
    )

    _write_json(
        run_dir / "replay" / "summary.json",
        {
            "started_at": "2026-03-22T00:00:00Z",
            "worker_results": {
                completed_trial_id: {
                    "finished_at": "2026-03-22T00:00:10Z",
                    "status": "completed",
                },
                incomplete_trial_id: {
                    "finished_at": "2026-03-22T00:00:12Z",
                    "status": "completed",
                },
            },
        },
    )


class MaterializeIntegralMetricsTest(unittest.TestCase):
    def test_materialize_integral_metrics_writes_sorted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"

            _make_run(
                root_dir,
                batch_name="20260322T000000Z",
                run_name="core-345-1005",
                total_power_w=100.0,
                running_request_count=2.0,
                completed_trace_id="trace-1005-done",
                completed_trial_id="trial-1005-done",
                incomplete_trace_id="trace-1005-other",
                incomplete_trial_id="trial-1005-other",
            )
            _make_run(
                root_dir,
                batch_name="20260322T000000Z",
                run_name="core-345-810",
                total_power_w=80.0,
                running_request_count=1.0,
                completed_trace_id="trace-810-done",
                completed_trial_id="trial-810-done",
                incomplete_trace_id="trace-810-other",
                incomplete_trial_id="trial-810-other",
            )

            output_path = tmp_path / "integral.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--root-dir",
                    str(root_dir),
                    "--start-s",
                    "0",
                    "--end-s",
                    "10",
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual([row["frequency_mhz"] for row in rows], ["810", "1005"])

            row_810 = rows[0]
            self.assertEqual(row_810["selected_finished_replay_count_in_window"], "1")
            self.assertEqual(row_810["mapped_finished_replay_count_in_window"], "1")
            self.assertEqual(
                row_810["total_request_integral_energy_across_finished_replays_j"],
                "800.0",
            )
            self.assertEqual(
                row_810["average_request_integral_energy_per_finished_replay_j"],
                "800.0",
            )
            self.assertEqual(row_810["average_throughput_jobs_per_s"], "0.1")

            row_1005 = rows[1]
            self.assertEqual(row_1005["selected_finished_replay_count_in_window"], "1")
            self.assertEqual(row_1005["mapped_finished_replay_count_in_window"], "1")
            self.assertEqual(
                row_1005["total_request_integral_energy_across_finished_replays_j"],
                "500.0",
            )
            self.assertEqual(
                row_1005["average_request_integral_energy_per_finished_replay_j"],
                "500.0",
            )
            self.assertEqual(row_1005["average_throughput_jobs_per_s"], "0.1")
