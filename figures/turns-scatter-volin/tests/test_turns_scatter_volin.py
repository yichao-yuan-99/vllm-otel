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
    REPO_ROOT / "figures" / "turns-scatter-volin" / "materialize_turns_scatter_volin.py"
)
PLOT_SCRIPT = (
    REPO_ROOT / "figures" / "turns-scatter-volin" / "plot_turns_scatter_volin.py"
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_run_summary(
    root_dir: Path,
    *,
    benchmark: str,
    agent_type: str,
    run_dir_name: str,
    turns: list[int],
    score: float,
    avg_turns_per_run: float | None = None,
) -> None:
    run_dir = root_dir / benchmark / agent_type / run_dir_name
    _write_json(
        run_dir / "run-stats" / "run-stats-summary.json",
        {
            "dataset": benchmark,
            "agent_type": agent_type,
            "score": score,
            "job_count": len(turns),
            "avg_turns_per_run": avg_turns_per_run,
            "jobs": [
                {
                    "gateway_run_id": f"{benchmark}-{agent_type}-{index}",
                    "gateway_profile_id": index,
                    "request_count": turn,
                }
                for index, turn in enumerate(turns)
            ],
        },
    )


class TurnsScatterVolinFigureTest(unittest.TestCase):
    def test_materialize_turns_scatter_volin_builds_expected_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="mini-swe-agent",
                run_dir_name="dabstep-20260306T180000Z",
                turns=[9, 11],
                score=0.10,
            )
            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="mini-swe-agent",
                run_dir_name="dabstep-20260306T194929Z",
                turns=[2, 4, 8],
                score=0.20,
                avg_turns_per_run=4.0,
            )
            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="terminus-2",
                run_dir_name="dabstep-20260306T215045Z",
                turns=[3, 6, 12],
                score=0.30,
            )
            _write_run_summary(
                root_dir,
                benchmark="swebench-verified",
                agent_type="mini-swe-agent",
                run_dir_name="swebench-verified-20260306T062226Z",
                turns=[5, 10, 15, 20],
                score=0.40,
            )
            _write_run_summary(
                root_dir,
                benchmark="swebench-verified",
                agent_type="terminus-2",
                run_dir_name="swebench-verified-20260306T082357Z",
                turns=[7, 14, 21],
                score=0.50,
            )
            _write_run_summary(
                root_dir,
                benchmark="terminal-bench-2.0",
                agent_type="mini-swe-agent",
                run_dir_name="terminal-bench@2.0-20260306T163324Z",
                turns=[1, 1, 9],
                score=0.60,
            )
            _write_run_summary(
                root_dir,
                benchmark="terminal-bench-2.0",
                agent_type="terminus-2",
                run_dir_name="terminal-bench@2.0-20260306T174037Z",
                turns=[9, 18, 36, 72],
                score=0.70,
            )

            output_path = tmp_path / "turns-scatter-volin.json"
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
            self.assertEqual(payload["figure_name"], "turns-scatter-volin")
            self.assertEqual(payload["metric_name"], "turns")
            self.assertEqual(
                payload["benchmark_order"],
                ["dabstep", "swebench-verified", "terminal-bench-2.0"],
            )
            self.assertEqual(payload["agent_order"], ["mini-swe-agent", "terminus-2"])
            self.assertEqual(payload["panel_count"], 6)
            self.assertEqual(payload["total_job_count"], 20)

            first_panel = payload["panels"][0]
            self.assertEqual(first_panel["benchmark"], "dabstep")
            self.assertEqual(first_panel["agent_type"], "mini-swe-agent")
            self.assertEqual(first_panel["run_dir_name"], "dabstep-20260306T194929Z")
            self.assertEqual(first_panel["candidate_run_count"], 2)
            self.assertEqual(first_panel["turns"], [2, 4, 8])
            self.assertAlmostEqual(first_panel["avg_turns_per_run_reported"], 4.0)
            self.assertAlmostEqual(first_panel["avg_turns_per_run_computed"], 14.0 / 3.0)
            self.assertEqual(first_panel["stats"]["sample_count"], 3)
            self.assertEqual(first_panel["stats"]["min"], 2)
            self.assertEqual(first_panel["stats"]["max"], 8)
            self.assertAlmostEqual(first_panel["stats"]["median"], 4.0)

            last_panel = payload["panels"][-1]
            self.assertEqual(last_panel["benchmark"], "terminal-bench-2.0")
            self.assertEqual(last_panel["agent_label"], "Terminus")
            self.assertEqual(last_panel["turns"], [9, 18, 36, 72])
            self.assertAlmostEqual(last_panel["stats"]["mean"], 33.75)
            self.assertAlmostEqual(last_panel["stats"]["q1"], 15.75)
            self.assertAlmostEqual(last_panel["stats"]["q3"], 45.0)

    def test_plot_turns_scatter_volin_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            for benchmark, mini_turns, terminus_turns in [
                ("dabstep", [2, 4, 8], [3, 6, 12]),
                ("swebench-verified", [5, 10, 15], [7, 14, 21]),
                ("terminal-bench-2.0", [1, 9, 27], [9, 18, 36]),
            ]:
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="mini-swe-agent",
                    run_dir_name=f"{benchmark}-20260306T100000Z",
                    turns=mini_turns,
                    score=0.2,
                )
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="terminus-2",
                    run_dir_name=f"{benchmark}-20260306T110000Z",
                    turns=terminus_turns,
                    score=0.3,
                )

            input_path = tmp_path / "turns-scatter-volin.json"
            output_path = tmp_path / "turns-scatter-volin.png"
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
