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
    REPO_ROOT / "figures" / "turns-context-violin" / "materialize_turns_context_violin.py"
)
PLOT_SCRIPT = (
    REPO_ROOT / "figures" / "turns-context-violin" / "plot_turns_context_violin.py"
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
    max_request_lengths: list[int],
    score: float,
    avg_job_max_request_length: float | None = None,
) -> None:
    run_dir = root_dir / benchmark / agent_type / run_dir_name
    _write_json(
        run_dir / "run-stats" / "run-stats-summary.json",
        {
            "dataset": benchmark,
            "agent_type": agent_type,
            "score": score,
            "job_count": len(max_request_lengths),
            "job_max_request_lengths": max_request_lengths,
            "avg_job_max_request_length": avg_job_max_request_length,
            "max_job_max_request_length": max(max_request_lengths),
            "jobs": [
                {
                    "gateway_run_id": f"{benchmark}-{agent_type}-{index}",
                    "gateway_profile_id": index,
                    "max_request_length": max_request_length,
                }
                for index, max_request_length in enumerate(max_request_lengths)
            ],
        },
    )


class TurnsContextViolinFigureTest(unittest.TestCase):
    def test_materialize_turns_context_violin_builds_expected_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="mini-swe-agent",
                run_dir_name="dabstep-20260306T180000Z",
                max_request_lengths=[900, 1100],
                score=0.10,
            )
            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="mini-swe-agent",
                run_dir_name="dabstep-20260306T194929Z",
                max_request_lengths=[2000, 4000, 8000],
                score=0.20,
                avg_job_max_request_length=4000.0,
            )
            _write_run_summary(
                root_dir,
                benchmark="dabstep",
                agent_type="terminus-2",
                run_dir_name="dabstep-20260306T215045Z",
                max_request_lengths=[3000, 6000, 12000],
                score=0.30,
            )
            _write_run_summary(
                root_dir,
                benchmark="swebench-verified",
                agent_type="mini-swe-agent",
                run_dir_name="swebench-verified-20260306T062226Z",
                max_request_lengths=[5000, 10000, 15000, 20000],
                score=0.40,
            )
            _write_run_summary(
                root_dir,
                benchmark="swebench-verified",
                agent_type="terminus-2",
                run_dir_name="swebench-verified-20260306T082357Z",
                max_request_lengths=[7000, 14000, 21000],
                score=0.50,
            )
            _write_run_summary(
                root_dir,
                benchmark="terminal-bench-2.0",
                agent_type="mini-swe-agent",
                run_dir_name="terminal-bench@2.0-20260306T163324Z",
                max_request_lengths=[1000, 1000, 9000],
                score=0.60,
            )
            _write_run_summary(
                root_dir,
                benchmark="terminal-bench-2.0",
                agent_type="terminus-2",
                run_dir_name="terminal-bench@2.0-20260306T174037Z",
                max_request_lengths=[9000, 18000, 36000, 72000],
                score=0.70,
            )

            output_path = tmp_path / "turns-context-violin.json"
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
            self.assertEqual(payload["figure_name"], "turns-context-violin")
            self.assertEqual(payload["metric_name"], "max_context_usage")
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
            self.assertEqual(first_panel["max_context_usages"], [2000, 4000, 8000])
            self.assertAlmostEqual(first_panel["avg_job_max_request_length_reported"], 4000.0)
            self.assertAlmostEqual(first_panel["avg_job_max_request_length_computed"], 14000.0 / 3.0)
            self.assertAlmostEqual(first_panel["max_job_max_request_length_reported"], 8000.0)
            self.assertAlmostEqual(first_panel["max_job_max_request_length_computed"], 8000.0)
            self.assertEqual(first_panel["stats"]["sample_count"], 3)
            self.assertEqual(first_panel["stats"]["min"], 2000)
            self.assertEqual(first_panel["stats"]["max"], 8000)
            self.assertAlmostEqual(first_panel["stats"]["median"], 4000.0)

            last_panel = payload["panels"][-1]
            self.assertEqual(last_panel["benchmark"], "terminal-bench-2.0")
            self.assertEqual(last_panel["agent_label"], "Terminus")
            self.assertEqual(last_panel["max_context_usages"], [9000, 18000, 36000, 72000])
            self.assertAlmostEqual(last_panel["stats"]["mean"], 33750.0)
            self.assertAlmostEqual(last_panel["stats"]["q1"], 15750.0)
            self.assertAlmostEqual(last_panel["stats"]["q3"], 45000.0)

    def test_plot_turns_context_violin_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "qwen3-coder-30b"

            for benchmark, mini_contexts, terminus_contexts in [
                ("dabstep", [2000, 4000, 8000], [3000, 6000, 12000]),
                ("swebench-verified", [5000, 10000, 15000], [7000, 14000, 21000]),
                ("terminal-bench-2.0", [1000, 9000, 27000], [9000, 18000, 36000]),
            ]:
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="mini-swe-agent",
                    run_dir_name=f"{benchmark}-20260306T100000Z",
                    max_request_lengths=mini_contexts,
                    score=0.2,
                )
                _write_run_summary(
                    root_dir,
                    benchmark=benchmark,
                    agent_type="terminus-2",
                    run_dir_name=f"{benchmark}-20260306T110000Z",
                    max_request_lengths=terminus_contexts,
                    score=0.3,
                )

            input_path = tmp_path / "turns-context-violin.json"
            output_path = tmp_path / "turns-context-violin.png"
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
