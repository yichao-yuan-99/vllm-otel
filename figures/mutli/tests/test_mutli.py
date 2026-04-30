from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = REPO_ROOT / "figures" / "mutli" / "materialize_mutli.py"
PLOT_SCRIPT = REPO_ROOT / "figures" / "mutli" / "plot_mutli.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_run(
    root_dir: Path,
    *,
    case_slug: str,
    run_dir_name: str,
    total_power_points: list[float],
    per_gpu_power_points: list[list[float]],
    total_energy_j: float,
) -> Path:
    run_dir = root_dir / case_slug / "qps0_1" / run_dir_name
    _write_json(run_dir / "replay" / "summary.json", {"run_id": run_dir_name})
    _write_json(
        run_dir / "post-processed" / "power" / "power-summary.json",
        {
            "source_run_dir": str(run_dir),
            "power_stats_w": {
                "avg": sum(total_power_points) / len(total_power_points),
                "min": min(total_power_points),
                "max": max(total_power_points),
            },
            "total_energy_j": total_energy_j,
            "power_points": [
                {"time_offset_s": float(index), "power_w": value}
                for index, value in enumerate(total_power_points)
            ],
            "per_gpu_power": [
                {
                    "display_label": f"GPU {gpu_index}",
                    "gpu_id": str(gpu_index),
                    "power_points": [
                        {"time_offset_s": float(index), "power_w": value}
                        for index, value in enumerate(values)
                    ],
                }
                for gpu_index, values in enumerate(per_gpu_power_points)
            ],
        },
    )
    return run_dir


class MutliFigureTest(unittest.TestCase):
    def test_materialize_mutli_extracts_average_and_per_gpu_power(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"

            baseline_parent = root_dir / "baseline" / "qps0_1"
            _make_run(
                root_dir,
                case_slug="baseline",
                run_dir_name="20260410T000000Z",
                total_power_points=[900.0, 1100.0],
                per_gpu_power_points=[
                    [200.0, 300.0],
                    [300.0, 300.0],
                    [200.0, 200.0],
                    [200.0, 300.0],
                ],
                total_energy_j=12345.0,
            )
            _make_run(
                root_dir,
                case_slug="baseline",
                run_dir_name="20260411T000000Z",
                total_power_points=[1000.0, 1200.0],
                per_gpu_power_points=[
                    [250.0, 350.0],
                    [300.0, 400.0],
                    [200.0, 200.0],
                    [250.0, 250.0],
                ],
                total_energy_j=20000.0,
            )
            round_robin_run = _make_run(
                root_dir,
                case_slug="round-robin",
                run_dir_name="20260412T000000Z",
                total_power_points=[600.0, 800.0],
                per_gpu_power_points=[
                    [300.0, 300.0],
                    [200.0, 200.0],
                    [100.0, 100.0],
                    [0.0, 100.0],
                ],
                total_energy_j=21000.0,
            )
            kairos_run = _make_run(
                root_dir,
                case_slug="kairos",
                run_dir_name="20260413T000000Z",
                total_power_points=[500.0, 700.0],
                per_gpu_power_points=[
                    [250.0, 250.0],
                    [150.0, 150.0],
                    [100.0, 150.0],
                    [0.0, 50.0],
                ],
                total_energy_j=22000.0,
            )

            output_path = tmp_path / "mutli.json"
            missing_log_path = tmp_path / "mutli.missing.log"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--no-freq-control-path",
                    str(baseline_parent),
                    "--round-robin-path",
                    str(round_robin_run),
                    "--kairos-path",
                    str(kairos_run),
                    "--output",
                    str(output_path),
                    "--missing-log",
                    str(missing_log_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "mutli")
            self.assertEqual(payload["variant_count"], 3)
            self.assertEqual(payload["bar_metric_key"], "average_power_w")
            self.assertEqual(payload["pie_metric_key"], "per_gpu_average_power")

            baseline = payload["variants"][0]
            self.assertEqual(baseline["label"], "No Freq Control")
            self.assertEqual(Path(baseline["resolved_run_dir"]).name, "20260411T000000Z")
            self.assertAlmostEqual(baseline["metrics"]["average_power_w"], 1100.0)
            self.assertAlmostEqual(baseline["metrics"]["total_energy_j"], 20000.0)
            self.assertEqual(
                [gpu["average_power_w"] for gpu in baseline["per_gpu_average_power"]],
                [300.0, 350.0, 200.0, 250.0],
            )

            round_robin = payload["variants"][1]
            self.assertEqual(round_robin["label"], "Round Robin")
            self.assertEqual(
                [gpu["share_pct"] for gpu in round_robin["per_gpu_average_power"]],
                [46.153846, 30.769231, 15.384615, 7.692308],
            )

            self.assertTrue(missing_log_path.is_file())
            self.assertEqual(missing_log_path.read_text(encoding="utf-8"), "")

    def test_plot_mutli_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "mutli.json"
            output_path = tmp_path / "mutli.pdf"
            _write_json(
                input_path,
                {
                    "figure_name": "mutli",
                    "variants": [
                        {
                            "label": "No Freq Control",
                            "metrics": {"average_power_w": 1200.0},
                            "per_gpu_average_power": [
                                {"display_label": "GPU 0", "average_power_w": 300.0},
                                {"display_label": "GPU 1", "average_power_w": 300.0},
                                {"display_label": "GPU 2", "average_power_w": 300.0},
                                {"display_label": "GPU 3", "average_power_w": 300.0},
                            ],
                        },
                        {
                            "label": "Round Robin",
                            "metrics": {"average_power_w": 700.0},
                            "per_gpu_average_power": [
                                {"display_label": "GPU 0", "average_power_w": 300.0},
                                {"display_label": "GPU 1", "average_power_w": 200.0},
                                {"display_label": "GPU 2", "average_power_w": 100.0},
                                {"display_label": "GPU 3", "average_power_w": 100.0},
                            ],
                        },
                        {
                            "label": "KAIROS",
                            "metrics": {"average_power_w": 600.0},
                            "per_gpu_average_power": [
                                {"display_label": "GPU 0", "average_power_w": 250.0},
                                {"display_label": "GPU 1", "average_power_w": 150.0},
                                {"display_label": "GPU 2", "average_power_w": 125.0},
                                {"display_label": "GPU 3", "average_power_w": 75.0},
                            ],
                        },
                    ],
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
