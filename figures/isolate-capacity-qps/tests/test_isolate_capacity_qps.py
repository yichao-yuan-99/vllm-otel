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
    / "isolate-capacity-qps"
    / "materialize_isolate_capacity_qps.py"
)
PLOT_SCRIPT = (
    REPO_ROOT / "figures" / "isolate-capacity-qps" / "plot_isolate_capacity_qps.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _make_run(
    root_dir: Path,
    *,
    case_slug: str,
    profile_dir_name: str,
    qps_slug: str,
    run_dir_name: str,
    throughput_values: list[float],
    completed_agent_llm_times_s: list[float],
    incomplete_agent_llm_times_s: list[float],
    completion_values: list[float],
) -> Path:
    run_dir = root_dir / case_slug / profile_dir_name / qps_slug / run_dir_name

    _write_json(
        run_dir / "post-processed" / "job-throughput" / "job-throughput-timeseries.json",
        {
            "window_size_s": 600.0,
            "throughput_points": [
                {
                    "time_s": float(index),
                    "throughput_jobs_per_s": value,
                }
                for index, value in enumerate(throughput_values)
            ],
            "throughput_points_excluding_cancelled": [
                {
                    "time_s": float(index),
                    "throughput_jobs_per_s": value,
                }
                for index, value in enumerate(throughput_values)
            ],
        },
    )
    _write_json(
        run_dir
        / "post-processed"
        / "agent-output-throughput"
        / "agent-output-throughput.json",
        {
            "agent_count": len(completed_agent_llm_times_s) + len(incomplete_agent_llm_times_s),
            "agents": [
                {
                    "gateway_run_id": f"completed-{index}",
                    "replay_completed": True,
                    "output_tokens": 4144,
                    "llm_request_duration_s": value,
                    "output_throughput_tokens_per_s": (
                        4144.0 / value if value > 0.0 else None
                    ),
                }
                for index, value in enumerate(completed_agent_llm_times_s)
            ]
            + [
                {
                    "gateway_run_id": f"incomplete-{index}",
                    "replay_completed": False,
                    "output_tokens": 4144,
                    "llm_request_duration_s": value,
                    "output_throughput_tokens_per_s": (
                        4144.0 / value if value > 0.0 else None
                    ),
                }
                for index, value in enumerate(incomplete_agent_llm_times_s)
            ],
        },
    )
    _write_json(
        run_dir / "post-processed" / "gateway" / "stack" / "completion-tokens-stacked-histogram.json",
        {
            "metric": "completion_tokens",
            "bucket_width_s": 1.0,
            "points": [
                {
                    "second": index,
                    "accumulated_value": value,
                }
                for index, value in enumerate(completion_values)
            ],
        },
    )
    return run_dir


class IsolateCapacityQpsFigureTest(unittest.TestCase):
    def test_materialize_isolate_capacity_qps_resolves_qps_dir_and_exact_run_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"

            _make_run(
                root_dir,
                case_slug="trail",
                profile_dir_name="profile-a",
                qps_slug="qps0_5",
                run_dir_name="20260331T010101Z",
                throughput_values=[0.1, 0.2],
                completed_agent_llm_times_s=[10.0, 20.0],
                incomplete_agent_llm_times_s=[1000.0],
                completion_values=[1.0, 2.0],
            )
            _make_run(
                root_dir,
                case_slug="trail",
                profile_dir_name="profile-a",
                qps_slug="qps0_5",
                run_dir_name="20260331T020202Z",
                throughput_values=[0.2, 0.4],
                completed_agent_llm_times_s=[100.0, 200.0, 300.0],
                incomplete_agent_llm_times_s=[999.0],
                completion_values=[5.0, 15.0],
            )
            trail_qps06_run = _make_run(
                root_dir,
                case_slug="trail",
                profile_dir_name="profile-a",
                qps_slug="qps0_6",
                run_dir_name="20260331T195832Z",
                throughput_values=[0.1, 0.3, 0.5],
                completed_agent_llm_times_s=[200.0, 400.0],
                incomplete_agent_llm_times_s=[800.0],
                completion_values=[10.0, 20.0, 30.0],
            )
            trail_lmcache_qps06_run = _make_run(
                root_dir,
                case_slug="trail-lmcache",
                profile_dir_name="profile-a",
                qps_slug="qps0_6",
                run_dir_name="20260331T212051Z",
                throughput_values=[0.25, 0.35],
                completed_agent_llm_times_s=[300.0, 500.0, 700.0],
                incomplete_agent_llm_times_s=[900.0],
                completion_values=[7.0, 9.0],
            )

            output_path = tmp_path / "isolate-capacity-qps.json"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--source-dirs",
                    str(root_dir / "trail" / "profile-a" / "qps0_5"),
                    str(trail_qps06_run),
                    str(trail_lmcache_qps06_run),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "isolate-capacity-qps")
            self.assertEqual(payload["source_count"], 3)
            self.assertEqual(payload["source_qps_slugs"], ["qps0_5", "qps0_6"])
            self.assertEqual(payload["metric_count"], 3)
            self.assertEqual(payload["case_count"], 3)
            self.assertEqual(
                [metric["metric_key"] for metric in payload["metrics"]],
                [
                    "average_throughput_jobs_per_s",
                    "average_completed_agent_llm_time_s",
                    "average_completion_tokens",
                ],
            )

            trail_qps05_case = payload["cases"][0]
            self.assertEqual(trail_qps05_case["case_label"], "TRAIL 0.5")
            self.assertEqual(trail_qps05_case["source_input_kind"], "qps_dir")
            self.assertEqual(trail_qps05_case["run_dir_name"], "20260331T020202Z")
            self.assertEqual(trail_qps05_case["candidate_run_count"], 2)
            self.assertAlmostEqual(
                trail_qps05_case["metrics"]["average_throughput_jobs_per_s"]["value"],
                0.3,
            )
            self.assertAlmostEqual(
                trail_qps05_case["metrics"]["average_completed_agent_llm_time_s"]["value"],
                200.0,
            )
            self.assertEqual(
                trail_qps05_case["metrics"]["average_completed_agent_llm_time_s"][
                    "sample_count"
                ],
                3,
            )
            self.assertAlmostEqual(
                trail_qps05_case["metrics"]["average_completion_tokens"]["value"],
                10.0,
            )

            trail_qps06_case = payload["cases"][1]
            self.assertEqual(trail_qps06_case["case_label"], "TRAIL 0.6")
            self.assertEqual(trail_qps06_case["source_input_kind"], "run_dir")
            self.assertAlmostEqual(
                trail_qps06_case["metric_values"]["average_throughput_jobs_per_s"],
                0.3,
            )
            self.assertAlmostEqual(
                trail_qps06_case["metrics"]["average_completed_agent_llm_time_s"]["value"],
                300.0,
            )

            trail_lmcache_qps06_case = payload["cases"][2]
            self.assertEqual(
                trail_lmcache_qps06_case["case_label"],
                "TRAIL + LMCache 0.6",
            )
            self.assertAlmostEqual(
                trail_lmcache_qps06_case["metrics"]["average_completion_tokens"]["value"],
                8.0,
            )

    def test_plot_isolate_capacity_qps_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"

            trail_qps05_run = _make_run(
                root_dir,
                case_slug="trail",
                profile_dir_name="profile-a",
                qps_slug="qps0_5",
                run_dir_name="20260331T185346Z",
                throughput_values=[0.2, 0.4],
                completed_agent_llm_times_s=[100.0, 200.0, 300.0],
                incomplete_agent_llm_times_s=[999.0],
                completion_values=[5.0, 15.0],
            )
            trail_qps06_run = _make_run(
                root_dir,
                case_slug="trail",
                profile_dir_name="profile-a",
                qps_slug="qps0_6",
                run_dir_name="20260331T195832Z",
                throughput_values=[0.1, 0.3, 0.5],
                completed_agent_llm_times_s=[200.0, 400.0],
                incomplete_agent_llm_times_s=[800.0],
                completion_values=[10.0, 20.0, 30.0],
            )
            trail_lmcache_qps06_run = _make_run(
                root_dir,
                case_slug="trail-lmcache",
                profile_dir_name="profile-a",
                qps_slug="qps0_6",
                run_dir_name="20260331T212051Z",
                throughput_values=[0.25, 0.35],
                completed_agent_llm_times_s=[300.0, 500.0, 700.0],
                incomplete_agent_llm_times_s=[900.0],
                completion_values=[7.0, 9.0],
            )

            input_path = tmp_path / "isolate-capacity-qps.json"
            output_path = tmp_path / "isolate-capacity-qps.png"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--source-dirs",
                    str(trail_qps05_run),
                    str(trail_qps06_run),
                    str(trail_lmcache_qps06_run),
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
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
