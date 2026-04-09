from __future__ import annotations

import importlib.util
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
    / "freq-vs-average-energy-duration-throughtpu"
    / "materialize_windowed_metrics.py"
)
PLOT_SCRIPT_PATH = (
    REPO_ROOT
    / "figures"
    / "freq-vs-average-energy-duration-throughtpu"
    / "plot_freq_vs_average_energy.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _make_run(
    root_dir: Path,
    *,
    batch_name: str,
    run_name: str,
    power_points: list[dict[str, float]],
    worker_finished_times_s: list[float],
    llm_requests: list[dict[str, object]],
) -> None:
    run_dir = root_dir / batch_name / run_name

    _write_json(
        run_dir / "post-processed" / "power" / "power-summary.json",
        {
            "analysis_window_start_utc": "2026-03-22T00:00:00Z",
            "analysis_window_end_utc": "2026-03-22T00:00:10Z",
            "power_points": power_points,
        },
    )

    worker_results = {}
    for index, finish_time_s in enumerate(worker_finished_times_s):
        worker_results[f"trial-{index:04d}"] = {
            "finished_at": f"2026-03-22T00:00:{finish_time_s:06.3f}Z",
            "status": "completed",
        }

    _write_json(
        run_dir / "replay" / "summary.json",
        {
            "started_at": "2026-03-22T00:00:00Z",
            "finished_at": "2026-03-22T00:00:10Z",
            "time_constraint_s": 10.0,
            "worker_results": worker_results,
        },
    )

    _write_json(
        run_dir / "post-processed" / "gateway" / "llm-requests" / "llm-requests.json",
        {
            "request_count": len(llm_requests),
            "requests": llm_requests,
        },
    )


class MaterializeWindowedMetricsTest(unittest.TestCase):
    def test_materialize_windowed_metrics_writes_sorted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"
            _make_run(
                root_dir,
                batch_name="20260322T000000Z",
                run_name="core-345-1005",
                power_points=[
                    {"time_offset_s": 0.0, "power_w": 100.0},
                    {"time_offset_s": 5.0, "power_w": 200.0},
                    {"time_offset_s": 10.0, "power_w": 300.0},
                ],
                worker_finished_times_s=[2.0, 7.0],
                llm_requests=[
                    {
                        "request_start_offset_s": 0.5,
                        "request_end_offset_s": 2.0,
                        "gen_ai.latency.time_in_model_inference": 1.25,
                    },
                    {
                        "request_start_offset_s": 3.0,
                        "request_end_offset_s": 4.0,
                        "gen_ai.latency.time_in_model_inference": 1.75,
                    },
                    {
                        "request_start_offset_s": 6.0,
                        "request_end_offset_s": 7.0,
                        "gen_ai.latency.time_in_model_inference": 9.0,
                    },
                ],
            )
            _make_run(
                root_dir,
                batch_name="20260322T000000Z",
                run_name="core-345-810",
                power_points=[
                    {"time_offset_s": 0.0, "power_w": 50.0},
                    {"time_offset_s": 5.0, "power_w": 50.0},
                    {"time_offset_s": 10.0, "power_w": 50.0},
                ],
                worker_finished_times_s=[5.0],
                llm_requests=[
                    {
                        "request_start_offset_s": 1.0,
                        "request_end_offset_s": 3.0,
                        "gen_ai.latency.time_in_model_inference": 0.5,
                    },
                    {
                        "request_start_offset_s": 4.0,
                        "request_end_offset_s": 4.5,
                    },
                ],
            )

            output_path = tmp_path / "windowed.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--root-dir",
                    str(root_dir),
                    "--start-s",
                    "0",
                    "--end-s",
                    "5",
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
            self.assertEqual(row_810["window_avg_power_w"], "50.0")
            self.assertEqual(row_810["window_energy_estimate_j"], "250.0")
            self.assertEqual(row_810["finished_replay_count_in_window"], "1")
            self.assertEqual(row_810["llm_request_count_in_window"], "2")
            self.assertEqual(
                row_810["llm_request_with_model_inference_time_count_in_window"],
                "1",
            )
            self.assertEqual(row_810["average_request_time_in_llm_s"], "0.5")
            self.assertEqual(row_810["average_throughput_jobs_per_s"], "0.2")
            self.assertEqual(row_810["average_energy_per_finished_replay_j"], "250.0")

            row_1005 = rows[1]
            self.assertEqual(row_1005["window_avg_power_w"], "150.0")
            self.assertEqual(row_1005["window_energy_estimate_j"], "750.0")
            self.assertEqual(row_1005["window_energy_integral_j"], "750.0")
            self.assertEqual(row_1005["finished_replay_count_in_window"], "1")
            self.assertEqual(row_1005["llm_request_count_in_window"], "2")
            self.assertEqual(
                row_1005["llm_request_with_model_inference_time_count_in_window"],
                "2",
            )
            self.assertEqual(row_1005["average_request_time_in_llm_s"], "1.5")
            self.assertEqual(row_1005["average_throughput_jobs_per_s"], "0.2")
            self.assertEqual(row_1005["average_energy_per_finished_replay_j"], "750.0")

    def test_plot_freq_vs_average_energy_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"
            _make_run(
                root_dir,
                batch_name="20260322T000000Z",
                run_name="core-345-1005",
                power_points=[
                    {"time_offset_s": 0.0, "power_w": 100.0},
                    {"time_offset_s": 5.0, "power_w": 200.0},
                    {"time_offset_s": 10.0, "power_w": 300.0},
                ],
                worker_finished_times_s=[2.0, 7.0],
                llm_requests=[
                    {
                        "request_start_offset_s": 0.5,
                        "request_end_offset_s": 2.0,
                        "gen_ai.latency.time_in_model_inference": 1.25,
                    }
                ],
            )
            _make_run(
                root_dir,
                batch_name="20260322T000000Z",
                run_name="core-345-810",
                power_points=[
                    {"time_offset_s": 0.0, "power_w": 50.0},
                    {"time_offset_s": 5.0, "power_w": 50.0},
                    {"time_offset_s": 10.0, "power_w": 50.0},
                ],
                worker_finished_times_s=[5.0],
                llm_requests=[
                    {
                        "request_start_offset_s": 1.0,
                        "request_end_offset_s": 3.0,
                        "gen_ai.latency.time_in_model_inference": 0.5,
                    }
                ],
            )

            input_path = tmp_path / "windowed.csv"
            output_path = tmp_path / "figure.png"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--root-dir",
                    str(root_dir),
                    "--start-s",
                    "0",
                    "--end-s",
                    "5",
                    "--output",
                    str(input_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT_PATH),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
