from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = (
    REPO_ROOT
    / "figures"
    / "energy-context-latency"
    / "materialize_energy_context_latency.py"
)
PLOT_SCRIPT = (
    REPO_ROOT / "figures" / "energy-context-latency" / "plot_energy_context_latency.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _make_run(
    root_dir: Path,
    *,
    dataset_slug: str,
    agent_slug: str,
    qps_slug: str,
    run_dir_name: str,
    nested_run_dir_name: str | None = None,
    power_avg_w: float | None = None,
    duration_s: float | None = None,
    finished_agent_count: int = 0,
    incomplete_agent_count: int = 0,
    workers_completed: int | None = None,
    p5_throughput_tokens_per_s: float | None = None,
    context_values: list[float] | None = None,
    job_throughput_values: list[float] | None = None,
    include_power: bool = True,
    include_context: bool = True,
    include_throughput: bool = True,
    include_job_throughput: bool = True,
    include_replay_summary: bool = True,
) -> None:
    timestamp_dir = (
        root_dir
        / dataset_slug
        / agent_slug
        / "split"
        / "exclude-unranked"
        / qps_slug
        / run_dir_name
    )
    run_dir = (
        timestamp_dir / nested_run_dir_name if nested_run_dir_name is not None else timestamp_dir
    )
    if include_power:
        end_second = 0 if duration_s is None else int(duration_s)
        _write_json(
            run_dir / "post-processed" / "power" / "power-summary.json",
            {
                "analysis_window_start_utc": "2026-04-01T00:00:00Z",
                "analysis_window_end_utc": f"2026-04-01T00:00:{end_second:02d}Z",
                "power_stats_w": {
                    "avg": power_avg_w,
                    "min": power_avg_w,
                    "max": power_avg_w,
                },
            },
        )
    if include_context:
        _write_json(
            run_dir
            / "post-processed"
            / "gateway"
            / "stack-context"
            / "context-usage-stacked-histogram.json",
            {
                "points": [
                    {
                        "second": index,
                        "accumulated_value": value,
                    }
                    for index, value in enumerate(context_values or [])
                ]
            },
        )
    if include_throughput:
        agents = [
            {
                "gateway_run_id": f"completed-{index}",
                "replay_completed": True,
                "output_throughput_tokens_per_s": 60.0 + index,
            }
            for index in range(finished_agent_count)
        ]
        agents.extend(
            {
                "gateway_run_id": f"incomplete-{index}",
                "replay_completed": False,
                "output_throughput_tokens_per_s": 10.0 + index,
            }
            for index in range(incomplete_agent_count)
        )
        _write_json(
            run_dir
            / "post-processed"
            / "agent-output-throughput"
            / "agent-output-throughput.json",
            {
                "agent_output_throughput_tokens_per_s_summary": {
                    "sample_count": len(agents),
                    "percentiles": (
                        {"5": p5_throughput_tokens_per_s}
                        if p5_throughput_tokens_per_s is not None
                        else {}
                    ),
                },
                "agents": agents,
            },
        )
    if include_job_throughput:
        _write_json(
            run_dir
            / "post-processed"
            / "job-throughput"
            / "job-throughput-timeseries.json",
            {
                "throughput_points": [
                    {
                        "time_s": float(index),
                        "throughput_jobs_per_s": value,
                    }
                    for index, value in enumerate(job_throughput_values or [])
                ]
            },
        )
    if include_replay_summary:
        _write_json(
            run_dir / "replay" / "summary.json",
            {
                "workers_completed": (
                    finished_agent_count if workers_completed is None else workers_completed
                ),
                "workers_failed": incomplete_agent_count,
                "worker_results": {},
            },
        )


class EnergyContextLatencyFigureTest(unittest.TestCase):
    def test_materialize_energy_context_latency_selects_latest_runs_and_zero_fills_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            uncontrolled_root = tmp_path / "uncontrolled"
            fixed_freq_root = tmp_path / "fixed-freq"
            steer_root = tmp_path / "steer"

            _make_run(
                uncontrolled_root,
                dataset_slug="swebench-verified",
                agent_slug="mini-swe-agent",
                qps_slug="qps0_04",
                run_dir_name="20260401T000000Z",
                power_avg_w=100.0,
                duration_s=10.0,
                finished_agent_count=2,
                incomplete_agent_count=1,
                p5_throughput_tokens_per_s=40.0,
                context_values=[100.0, 200.0],
                job_throughput_values=[0.1, 0.2],
            )
            _make_run(
                uncontrolled_root,
                dataset_slug="swebench-verified",
                agent_slug="mini-swe-agent",
                qps_slug="qps0_04",
                run_dir_name="20260402T000000Z",
                power_avg_w=120.0,
                duration_s=20.0,
                finished_agent_count=3,
                incomplete_agent_count=1,
                workers_completed=4,
                p5_throughput_tokens_per_s=55.0,
                context_values=[1000.0, 2000.0],
                job_throughput_values=[0.3, 0.5],
            )
            (
                uncontrolled_root
                / "swebench-verified"
                / "mini-swe-agent"
                / "split"
                / "exclude-unranked"
                / "qps0_04"
                / "power-stats"
            ).mkdir(parents=True, exist_ok=True)
            _make_run(
                uncontrolled_root,
                dataset_slug="dabstep",
                agent_slug="mini-swe-agent",
                qps_slug="qps0_03",
                run_dir_name="20260410T142848Z",
                power_avg_w=300.0,
                duration_s=30.0,
                finished_agent_count=2,
                incomplete_agent_count=1,
                p5_throughput_tokens_per_s=44.0,
                context_values=[5000.0, 7000.0, 9000.0],
                job_throughput_values=[0.2, 0.3, 0.4],
            )
            _make_run(
                fixed_freq_root,
                dataset_slug="swebench-verified",
                agent_slug="mini-swe-agent",
                qps_slug="qps0_04",
                run_dir_name="20260411T140633Z",
                nested_run_dir_name="core-345-810",
                finished_agent_count=2,
                incomplete_agent_count=0,
                p5_throughput_tokens_per_s=33.0,
                context_values=[2500.0, 3500.0],
                job_throughput_values=[0.11, 0.13],
                include_power=False,
            )
            _make_run(
                fixed_freq_root,
                dataset_slug="terminal-bench-2.0",
                agent_slug="terminus-2",
                qps_slug="qps0_02",
                run_dir_name="20260411T042857Z",
                nested_run_dir_name="core-345-810",
                power_avg_w=90.0,
                duration_s=10.0,
                finished_agent_count=1,
                incomplete_agent_count=0,
                workers_completed=2,
                p5_throughput_tokens_per_s=20.0,
                context_values=[100.0, 300.0],
                job_throughput_values=[0.05, 0.15],
            )
            _make_run(
                steer_root,
                dataset_slug="terminal-bench-2.0",
                agent_slug="terminus-2",
                qps_slug="qps0_015",
                run_dir_name="20260412T212001Z",
                power_avg_w=150.0,
                duration_s=15.0,
                finished_agent_count=5,
                incomplete_agent_count=0,
                p5_throughput_tokens_per_s=25.0,
                context_values=[1000.0, 2000.0, 3000.0],
                job_throughput_values=[0.07, 0.09, 0.11],
            )

            output_path = tmp_path / "energy-context-latency.json"
            missing_log_path = tmp_path / "energy-context-latency.missing.log"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--uncontrolled-root",
                    str(uncontrolled_root),
                    "--fixed-freq-root",
                    str(fixed_freq_root),
                    "--steer-root",
                    str(steer_root),
                    "--output",
                    str(output_path),
                    "--missing-log",
                    str(missing_log_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "energy-context-latency")
            self.assertEqual(payload["experiment_count"], 3)
            self.assertEqual(payload["implementation_count"], 3)
            self.assertEqual(payload["metric_count"], 5)

            experiment_a = payload["experiments"][0]
            qps_004 = experiment_a["qps"][0]
            uncontrolled_entry = qps_004["implementations"][0]
            self.assertEqual(uncontrolled_entry["run_dir_name"], "20260402T000000Z")
            self.assertEqual(uncontrolled_entry["candidate_run_count"], 2)
            self.assertAlmostEqual(
                uncontrolled_entry["metric_values"]["average_energy_per_finished_agent_kj"],
                0.6,
            )
            self.assertEqual(
                uncontrolled_entry["metrics"]["average_energy_per_finished_agent_kj"]["workers_completed"],
                4,
            )
            self.assertAlmostEqual(
                uncontrolled_entry["metric_values"]["average_power_w"],
                120.0,
            )
            self.assertEqual(
                uncontrolled_entry["metrics"]["average_power_w"]["source"],
                "power_stats_w.avg",
            )
            self.assertAlmostEqual(
                uncontrolled_entry["metrics"]["average_context_usage_pct"]["value"],
                ((1500.0 / 527664.0) * 100.0),
                places=6,
            )
            self.assertAlmostEqual(
                uncontrolled_entry["metric_values"]["p5_output_throughput_tokens_per_s"],
                55.0,
            )
            self.assertAlmostEqual(
                uncontrolled_entry["metric_values"]["average_job_throughput_jobs_per_s"],
                0.4,
            )

            fixed_freq_entry = qps_004["implementations"][1]
            self.assertAlmostEqual(
                fixed_freq_entry["metric_values"]["average_energy_per_finished_agent_kj"],
                0.0,
            )
            self.assertAlmostEqual(
                fixed_freq_entry["metric_values"]["average_power_w"],
                0.0,
            )
            self.assertAlmostEqual(
                fixed_freq_entry["metric_values"]["average_context_usage_pct"],
                ((3000.0 / 527664.0) * 100.0),
                places=6,
            )
            self.assertAlmostEqual(
                fixed_freq_entry["metric_values"]["average_job_throughput_jobs_per_s"],
                0.12,
            )

            experiment_c = payload["experiments"][2]
            qps_002 = experiment_c["qps"][1]
            self.assertAlmostEqual(
                qps_002["implementations"][1]["metric_values"]["average_energy_per_finished_agent_kj"],
                0.45,
            )
            self.assertAlmostEqual(
                qps_002["implementations"][1]["metric_values"]["average_power_w"],
                90.0,
            )
            qps_025 = experiment_c["qps"][2]
            self.assertEqual(
                qps_025["implementations"][2]["metric_values"]["p5_output_throughput_tokens_per_s"],
                0.0,
            )

            missing_log = missing_log_path.read_text(encoding="utf-8")
            self.assertIn("reason=Required file does not exist", missing_log)
            self.assertIn("metric=average_energy_per_finished_agent_kj", missing_log)
            self.assertIn("metric=average_power_w", missing_log)
            self.assertIn("experiment=C", missing_log)
            self.assertIn("qps=qps0_025", missing_log)
            self.assertGreater(payload["missing_entry_count"], 0)

    def test_plot_energy_context_latency_writes_five_outputs(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            uncontrolled_root = tmp_path / "uncontrolled"
            fixed_freq_root = tmp_path / "fixed-freq"
            steer_root = tmp_path / "steer"

            _make_run(
                uncontrolled_root,
                dataset_slug="swebench-verified",
                agent_slug="mini-swe-agent",
                qps_slug="qps0_04",
                run_dir_name="20260402T000000Z",
                power_avg_w=120.0,
                duration_s=20.0,
                finished_agent_count=3,
                p5_throughput_tokens_per_s=55.0,
                context_values=[1000.0, 2000.0],
                job_throughput_values=[0.3, 0.5],
            )
            _make_run(
                fixed_freq_root,
                dataset_slug="dabstep",
                agent_slug="mini-swe-agent",
                qps_slug="qps0_03",
                run_dir_name="20260411T000000Z",
                nested_run_dir_name="core-345-810",
                power_avg_w=200.0,
                duration_s=20.0,
                finished_agent_count=2,
                p5_throughput_tokens_per_s=40.0,
                context_values=[2000.0, 3000.0],
                job_throughput_values=[0.2, 0.4],
            )
            _make_run(
                steer_root,
                dataset_slug="terminal-bench-2.0",
                agent_slug="terminus-2",
                qps_slug="qps0_015",
                run_dir_name="20260412T000000Z",
                power_avg_w=150.0,
                duration_s=15.0,
                finished_agent_count=5,
                p5_throughput_tokens_per_s=25.0,
                context_values=[1000.0, 2000.0, 3000.0],
                job_throughput_values=[0.07, 0.09, 0.11],
            )

            input_path = tmp_path / "energy-context-latency.json"
            output_dir = tmp_path / "figures"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--uncontrolled-root",
                    str(uncontrolled_root),
                    "--fixed-freq-root",
                    str(fixed_freq_root),
                    "--steer-root",
                    str(steer_root),
                    "--output",
                    str(input_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "png",
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            output_paths = sorted(output_dir.glob("*.png"))
            self.assertEqual(len(output_paths), 5)
            self.assertTrue(
                any(path.name.endswith(".average_power_w.png") for path in output_paths)
            )
            for output_path in output_paths:
                self.assertTrue(output_path.is_file())
                self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
