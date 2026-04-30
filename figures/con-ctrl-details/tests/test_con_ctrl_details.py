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
    REPO_ROOT / "figures" / "con-ctrl-details" / "materialize_con_ctrl_details.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "con-ctrl-details" / "plot_con_ctrl_details.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_replay_summary(run_dir: Path) -> None:
    _write_json(
        run_dir / "replay" / "summary.json",
        {
            "started_at": "2026-04-13T12:00:00Z",
            "finished_at": "2026-04-13T15:00:00Z",
        },
    )


def _make_job_throughput(run_dir: Path, values: list[float]) -> None:
    _write_json(
        run_dir / "post-processed" / "job-throughput" / "job-throughput-timeseries.json",
        {
            "throughput_points": [
                {"time_s": float(index), "throughput_jobs_per_s": value}
                for index, value in enumerate(values)
            ]
        },
    )


def _make_context_histogram(run_dir: Path, values: list[float]) -> None:
    _write_json(
        run_dir
        / "post-processed"
        / "gateway"
        / "stack-context"
        / "context-usage-stacked-histogram.json",
        {
            "points": [
                {"second": float(index), "accumulated_value": value}
                for index, value in enumerate(values)
            ]
        },
    )


def _make_ctx_aware_timeseries(run_dir: Path, pending_values: list[tuple[float, float]]) -> None:
    _write_json(
        run_dir
        / "post-processed"
        / "gateway"
        / "ctx-aware-log"
        / "ctx-aware-timeseries.json",
        {
            "duration_s": 4.0,
            "metric_summaries": {
                "pending_agent_count": {
                    "max": max((value for _, value in pending_values), default=0.0)
                }
            },
            "samples": [
                {
                    "second": second,
                    "pending_agent_count": value,
                    "ongoing_agent_count": 0,
                    "ongoing_effective_context_tokens": 0,
                    "pending_effective_context_tokens": 0,
                    "agents_turned_pending_due_to_context_threshold": 0,
                    "agents_turned_ongoing": 0,
                    "new_agents_added_as_pending": 0,
                    "new_agents_added_as_ongoing": 0,
                }
                for second, value in pending_values
            ],
        },
    )


class ConCtrlDetailsFigureTest(unittest.TestCase):
    def test_materialize_con_ctrl_details_supports_direct_and_timestamped_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            no_thrash_root = tmp_path / "no-thrash"
            ctx_aware_run_dir = tmp_path / "ctx-aware"

            older_no_thrash = no_thrash_root / "20260413T000000Z"
            newer_no_thrash = no_thrash_root / "20260414T000000Z"
            _make_replay_summary(older_no_thrash)
            _make_job_throughput(older_no_thrash, [0.010, 0.011])
            _make_context_histogram(older_no_thrash, [0.0, 100.0])
            _make_replay_summary(newer_no_thrash)
            _make_job_throughput(newer_no_thrash, [0.050, 0.051, 0.052])
            _make_context_histogram(newer_no_thrash, [0.0, 1200.0, 3400.0, 7000.0])

            _make_replay_summary(ctx_aware_run_dir)
            _make_job_throughput(ctx_aware_run_dir, [0.049, 0.050, 0.051])
            _make_context_histogram(ctx_aware_run_dir, [0.0, 1000.0, 3000.0, 6000.0])
            _make_ctx_aware_timeseries(
                ctx_aware_run_dir,
                [
                    (0.0, 0.0),
                    (0.4, 1.0),
                    (0.8, 2.0),
                    (1.1, 0.0),
                    (1.9, 3.0),
                    (2.2, 1.0),
                ],
            )

            output_path = tmp_path / "con-ctrl-details.json"
            missing_log_path = tmp_path / "con-ctrl-details.missing.log"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--no-thrash-run-dir",
                    str(no_thrash_root),
                    "--ctx-aware-run-dir",
                    str(ctx_aware_run_dir),
                    "--context-smooth-window-s",
                    "0",
                    "--output",
                    str(output_path),
                    "--missing-log",
                    str(missing_log_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_key"], "con-ctrl-details")
            self.assertEqual(
                payload["source_run_dirs"]["kairos_no_thrashing_avoidance"],
                str(newer_no_thrash.resolve()),
            )
            self.assertEqual(
                payload["source_run_dirs"]["kairos_with_thrashing_avoidance"],
                str(ctx_aware_run_dir.resolve()),
            )

            no_thrash_series = payload["series"]["job_throughput_no_thrashing_avoidance"]
            self.assertEqual(
                [point["throughput_jobs_per_s"] for point in no_thrash_series],
                [0.05, 0.051, 0.052],
            )

            no_thrash_context_series = payload["series"]["context_usage_no_thrashing_avoidance"]
            self.assertEqual(
                [point["context_usage_tokens"] for point in no_thrash_context_series],
                [0.0, 1200.0, 3400.0, 7000.0],
            )

            context_series = payload["series"]["context_usage_with_thrashing_avoidance"]
            self.assertEqual(
                [point["context_usage_tokens"] for point in context_series],
                [0.0, 1000.0, 3000.0, 6000.0],
            )

            pending_series = payload["series"]["pending_agent_count_with_thrashing_avoidance"]
            self.assertEqual(
                pending_series,
                [
                    {"time_offset_s": 0.0, "pending_agent_count": 2.0},
                    {"time_offset_s": 1.0, "pending_agent_count": 3.0},
                    {"time_offset_s": 2.0, "pending_agent_count": 1.0},
                ],
            )

            missing_log = missing_log_path.read_text(encoding="utf-8")
            self.assertIn("[selected-latest-run]", missing_log)

    def test_plot_con_ctrl_details_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "con-ctrl-details.json"
            output_path = tmp_path / "con-ctrl-details.png"
            _write_json(
                input_path,
                {
                    "figure_key": "con-ctrl-details",
                    "series": {
                        "job_throughput_no_thrashing_avoidance": [
                            {"time_offset_s": 0.0, "throughput_jobs_per_s": 0.050},
                            {"time_offset_s": 60.0, "throughput_jobs_per_s": 0.048},
                        ],
                        "context_usage_no_thrashing_avoidance": [
                            {"time_offset_s": 0.0, "context_usage_tokens": 0.0},
                            {"time_offset_s": 60.0, "context_usage_tokens": 4200.0},
                        ],
                        "job_throughput_with_thrashing_avoidance": [
                            {"time_offset_s": 0.0, "throughput_jobs_per_s": 0.049},
                            {"time_offset_s": 60.0, "throughput_jobs_per_s": 0.051},
                        ],
                        "context_usage_with_thrashing_avoidance": [
                            {"time_offset_s": 0.0, "context_usage_tokens": 0.0},
                            {"time_offset_s": 60.0, "context_usage_tokens": 5000.0},
                        ],
                        "pending_agent_count_with_thrashing_avoidance": [
                            {"time_offset_s": 0.0, "pending_agent_count": 0.0},
                            {"time_offset_s": 60.0, "pending_agent_count": 2.0},
                        ],
                    },
                },
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
