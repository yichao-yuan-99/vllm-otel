from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = REPO_ROOT / "figures" / "kv-usage-time" / "materialize_kv_usage_time.py"
PLOT_SCRIPT = REPO_ROOT / "figures" / "kv-usage-time" / "plot_kv_usage_time.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_timeseries(
    root_dir: Path,
    *,
    batch_name: str,
    run_slug: str,
    times: list[float],
    values: list[float],
) -> None:
    run_dir = root_dir / batch_name / run_slug
    _write_json(
        run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json",
        {
            "source_run_dir": str(run_dir),
            "metrics": {
                "vllm:kv_cache_usage_perc|engine=0": {
                    "name": "vllm:kv_cache_usage_perc",
                    "labels": {"engine": "0", "model_name": "toy"},
                    "time_from_start_s": times,
                    "value": values,
                },
                "vllm:num_requests_running|engine=0": {
                    "name": "vllm:num_requests_running",
                    "labels": {"engine": "0", "model_name": "toy"},
                    "time_from_start_s": times,
                    "value": [0.0 for _ in times],
                },
            },
        },
    )


class KvUsageTimeFigureTest(unittest.TestCase):
    def test_materialize_kv_usage_time_builds_three_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"
            _write_timeseries(
                root_dir,
                batch_name="20260322T035359Z",
                run_slug="core-345-1680",
                times=[0.0, 2.0, 4.0, 8.0],
                values=[0.0, 0.2, 0.4, 0.8],
            )
            _write_timeseries(
                root_dir,
                batch_name="20260322T035359Z",
                run_slug="core-345-1185",
                times=[0.0, 1.0, 3.0, 7.0],
                values=[0.0, 0.1, 0.3, 0.7],
            )
            _write_timeseries(
                root_dir,
                batch_name="20260322T034746Z",
                run_slug="core-345-660",
                times=[0.0, 2.0, 5.0, 6.0],
                values=[0.0, 0.05, 0.25, 0.3],
            )

            output_path = tmp_path / "kv-usage-time.json"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--root-dir",
                    str(root_dir),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["series_count"], 3)
            self.assertEqual(payload["common_available_end_s"], 6.0)
            self.assertEqual(payload["analysis_window_end_s"], 6.0)
            self.assertEqual(payload["smooth_window_s"], 120.0)
            self.assertEqual(
                [item["run_slug"] for item in payload["series"]],
                ["core-345-1680", "core-345-1185", "core-345-660"],
            )

            series_1680 = payload["series"][0]
            self.assertEqual(series_1680["series_label"], "345-1680")
            self.assertEqual(series_1680["frequency_mhz"], 1680)
            self.assertEqual(series_1680["points"][-1]["time_from_start_s"], 6.0)
            self.assertAlmostEqual(series_1680["points"][-1]["raw_value"], 0.6)
            self.assertAlmostEqual(series_1680["points"][0]["value"], 0.3)
            self.assertAlmostEqual(series_1680["points"][-1]["value"], 0.3)
            self.assertEqual(series_1680["stats"]["sample_count"], 4)
            self.assertAlmostEqual(series_1680["stats"]["avg"], 0.3)
            self.assertAlmostEqual(series_1680["stats"]["max"], 0.3)

            series_1185 = payload["series"][1]
            self.assertEqual(series_1185["points"][-1]["time_from_start_s"], 6.0)
            self.assertAlmostEqual(series_1185["points"][-1]["raw_value"], 0.6)
            self.assertAlmostEqual(series_1185["points"][0]["value"], 0.25)
            self.assertAlmostEqual(series_1185["points"][-1]["value"], 0.25)

            series_660 = payload["series"][2]
            self.assertEqual(series_660["series_label"], "345-660")
            self.assertEqual(series_660["frequency_mhz"], 660)
            self.assertEqual(series_660["points"][-1]["time_from_start_s"], 6.0)
            self.assertAlmostEqual(series_660["points"][-1]["raw_value"], 0.3)
            self.assertAlmostEqual(series_660["points"][0]["value"], 0.15)
            self.assertAlmostEqual(series_660["points"][-1]["value"], 0.15)

    def test_plot_kv_usage_time_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results"
            _write_timeseries(
                root_dir,
                batch_name="20260322T035359Z",
                run_slug="core-345-1680",
                times=[0.0, 1.0, 2.0],
                values=[0.0, 0.2, 0.3],
            )
            _write_timeseries(
                root_dir,
                batch_name="20260322T035359Z",
                run_slug="core-345-1185",
                times=[0.0, 1.0, 2.0],
                values=[0.0, 0.1, 0.2],
            )
            _write_timeseries(
                root_dir,
                batch_name="20260322T034746Z",
                run_slug="core-345-660",
                times=[0.0, 1.0, 2.0],
                values=[0.0, 0.05, 0.1],
            )

            input_path = tmp_path / "kv-usage-time.json"
            output_path = tmp_path / "kv-usage-time.png"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--root-dir",
                    str(root_dir),
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
