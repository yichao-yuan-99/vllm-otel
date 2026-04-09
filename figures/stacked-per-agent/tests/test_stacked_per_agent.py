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
    REPO_ROOT / "figures" / "stacked-per-agent" / "materialize_stacked_per_agent.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "stacked-per-agent" / "plot_stacked_per_agent.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _materialize_fixture(run_dir: Path, *, output_path: Path) -> dict[str, object]:
    ranges_path = (
        run_dir
        / "post-processed"
        / "gateway"
        / "stack-context"
        / "context-usage-ranges.json"
    )
    _write_json(
        ranges_path,
        {
            "source_run_dir": str(run_dir),
            "source_gateway_output_dir": str(run_dir / "gateway-output"),
            "metric": "context_usage_tokens",
            "phase": "context",
            "entry_count": 5,
            "entries": [
                {
                    "agent_key": "agent-a",
                    "gateway_run_id": "run-a",
                    "gateway_profile_id": 0,
                    "segment_type": "active",
                    "range_start_s": 0.0,
                    "range_end_s": 120.0,
                    "avg_value_per_s": 10.0,
                },
                {
                    "agent_key": "agent-a",
                    "gateway_run_id": "run-a",
                    "gateway_profile_id": 0,
                    "segment_type": "active",
                    "range_start_s": 120.0,
                    "range_end_s": 240.0,
                    "avg_value_per_s": 5.0,
                },
                {
                    "agent_key": "agent-b",
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "segment_type": "idle",
                    "range_start_s": 0.0,
                    "range_end_s": 60.0,
                    "avg_value_per_s": 0.0,
                },
                {
                    "agent_key": "agent-b",
                    "gateway_run_id": "run-b",
                    "gateway_profile_id": 0,
                    "segment_type": "active",
                    "range_start_s": 60.0,
                    "range_end_s": 180.0,
                    "avg_value_per_s": 20.0,
                },
                {
                    "agent_key": "agent-c",
                    "gateway_run_id": "run-c",
                    "gateway_profile_id": 1,
                    "segment_type": "active",
                    "range_start_s": 180.0,
                    "range_end_s": 270.0,
                    "avg_value_per_s": 30.0,
                },
            ],
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(MATERIALIZE_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--window-size-s",
            "120",
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    return json.loads(output_path.read_text(encoding="utf-8"))


class StackedPerAgentFigureTest(unittest.TestCase):
    def test_materialize_stacked_per_agent_builds_sparse_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            run_dir = tmp_path / "run"
            output_path = tmp_path / "stacked-per-agent.json"

            payload = _materialize_fixture(run_dir, output_path=output_path)

            self.assertEqual(payload["agent_count"], 3)
            self.assertEqual(payload["window_count"], 3)
            self.assertEqual(payload["agent_order"], "first-active")
            self.assertEqual(
                [agent["agent_key"] for agent in payload["agents"]],
                ["agent-a", "agent-b", "agent-c"],
            )
            self.assertEqual(
                [agent["agent_label"] for agent in payload["agents"]],
                ["A001", "A002", "A003"],
            )

            window_0 = payload["windows"][0]
            self.assertEqual(window_0["window_start_s"], 0.0)
            self.assertEqual(window_0["window_end_s"], 120.0)
            self.assertAlmostEqual(window_0["total_average_value"], 20.0)
            self.assertEqual(
                [entry["agent_key"] for entry in window_0["contributions"]],
                ["agent-a", "agent-b"],
            )
            self.assertAlmostEqual(window_0["contributions"][0]["average_value"], 10.0)
            self.assertAlmostEqual(window_0["contributions"][1]["average_value"], 10.0)

            window_1 = payload["windows"][1]
            self.assertEqual(window_1["window_start_s"], 120.0)
            self.assertEqual(window_1["window_end_s"], 240.0)
            self.assertAlmostEqual(window_1["total_average_value"], 30.0)
            self.assertEqual(
                [entry["agent_key"] for entry in window_1["contributions"]],
                ["agent-a", "agent-b", "agent-c"],
            )
            self.assertAlmostEqual(window_1["contributions"][0]["average_value"], 5.0)
            self.assertAlmostEqual(window_1["contributions"][1]["average_value"], 10.0)
            self.assertAlmostEqual(window_1["contributions"][2]["average_value"], 15.0)

            window_2 = payload["windows"][2]
            self.assertEqual(window_2["window_duration_s"], 30.0)
            self.assertAlmostEqual(window_2["total_average_value"], 30.0)
            self.assertEqual(
                [entry["agent_key"] for entry in window_2["contributions"]],
                ["agent-c"],
            )
            self.assertAlmostEqual(window_2["contributions"][0]["integral_value"], 900.0)

    def test_plot_stacked_per_agent_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            run_dir = tmp_path / "run"
            input_path = tmp_path / "stacked-per-agent.json"
            output_path = tmp_path / "stacked-per-agent.png"
            _materialize_fixture(run_dir, output_path=input_path)

            subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--legend",
                    "show",
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
