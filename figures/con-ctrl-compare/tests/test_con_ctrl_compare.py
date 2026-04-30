from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = (
    REPO_ROOT / "figures" / "con-ctrl-compare" / "materialize_con_ctrl_compare.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "con-ctrl-compare" / "plot_con_ctrl_compare.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_run(
    run_dir: Path,
    *,
    p5_throughput: float | None = None,
    agent_throughputs: list[float] | None = None,
    job_throughput_points: list[float] | None = None,
    include_throughput: bool = True,
    include_job_throughput: bool = True,
) -> None:
    _write_json(
        run_dir / "replay" / "summary.json",
        {
            "workers_completed": 3,
            "workers_failed": 0,
        },
    )
    if include_throughput:
        throughput_payload: dict[str, object] = {}
        if p5_throughput is not None:
            throughput_payload["agent_output_throughput_tokens_per_s_summary"] = {
                "percentiles": {"5": p5_throughput}
            }
        if agent_throughputs is not None:
            throughput_payload["agents"] = [
                {"output_throughput_tokens_per_s": value}
                for value in agent_throughputs
            ]
        _write_json(
            run_dir
            / "post-processed"
            / "agent-output-throughput"
            / "agent-output-throughput.json",
            throughput_payload,
        )
    if include_job_throughput:
        _write_json(
            run_dir
            / "post-processed"
            / "job-throughput"
            / "job-throughput-timeseries.json",
            {
                "throughput_points": [
                    {"time_s": float(index), "throughput_jobs_per_s": value}
                    for index, value in enumerate(job_throughput_points or [])
                ]
            },
        )


class ConCtrlCompareFigureTest(unittest.TestCase):
    def test_materialize_con_ctrl_compare_supports_direct_and_timestamped_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            uncontrolled_root = tmp_path / "no-freq"
            kairos_no_thrash_dir = tmp_path / "kairos-no-thrash"
            kairos_thrash_aware_dir = tmp_path / "kairos-thrash-aware"

            _make_run(
                uncontrolled_root / "20260412T000000Z",
                p5_throughput=10.0,
                job_throughput_points=[0.010, 0.012],
            )
            _make_run(
                uncontrolled_root / "20260413T000000Z",
                p5_throughput=28.0,
                job_throughput_points=[0.051, 0.053, 0.055],
            )
            _make_run(
                kairos_no_thrash_dir,
                agent_throughputs=[1.0, 3.0, 4.0, 6.0, 8.0],
                job_throughput_points=[0.045, 0.050, 0.055],
            )
            _make_run(
                kairos_thrash_aware_dir,
                p5_throughput=19.5,
                include_job_throughput=False,
            )

            output_path = tmp_path / "con-ctrl-compare.json"
            missing_log_path = tmp_path / "con-ctrl-compare.missing.log"
            module_spec = importlib.util.spec_from_file_location(
                "materialize_con_ctrl_compare",
                MATERIALIZE_SCRIPT,
            )
            assert module_spec is not None
            module = importlib.util.module_from_spec(module_spec)
            assert module_spec.loader is not None
            module_spec.loader.exec_module(module)
            original_variants = module.VARIANTS
            module.VARIANTS = (
                {
                    "variant_key": "no_freq_control",
                    "label": "No Freq Control",
                    "base_path": str(uncontrolled_root),
                },
                {
                    "variant_key": "kairos_no_thrashing_avoidance",
                    "label": "KAIROS\nNo Thrash Avoid.",
                    "base_path": str(kairos_no_thrash_dir),
                },
                {
                    "variant_key": "kairos_with_thrashing_avoidance",
                    "label": "KAIROS\nThrash Avoid.",
                    "base_path": str(kairos_thrash_aware_dir),
                },
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import importlib.util, sys; "
                            f"spec=importlib.util.spec_from_file_location('m', {str(MATERIALIZE_SCRIPT)!r}); "
                            "m=importlib.util.module_from_spec(spec); "
                            "spec.loader.exec_module(m); "
                            f"m.VARIANTS={module.VARIANTS!r}; "
                            f"sys.argv=['prog','--output',{str(output_path)!r},'--missing-log',{str(missing_log_path)!r}]; "
                            "raise SystemExit(m.main())"
                        ),
                    ],
                    check=True,
                    cwd=str(REPO_ROOT),
                    env={
                        **os.environ,
                        "PYTHONPATH": str(REPO_ROOT),
                    },
                )
            finally:
                module.VARIANTS = original_variants

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "con-ctrl-compare")
            self.assertEqual(payload["variant_count"], 3)

            no_freq = payload["variants"][0]
            self.assertTrue(no_freq["selected_run_dir"].endswith("20260413T000000Z"))
            self.assertAlmostEqual(
                no_freq["metrics"]["p5_output_throughput_tokens_per_s"],
                28.0,
            )
            self.assertAlmostEqual(
                no_freq["metrics"]["average_job_throughput_jobs_per_s"],
                0.053,
            )

            kairos_no_thrash = payload["variants"][1]
            self.assertEqual(
                kairos_no_thrash["selected_run_dir"],
                str(kairos_no_thrash_dir.resolve()),
            )
            self.assertAlmostEqual(
                kairos_no_thrash["metrics"]["p5_output_throughput_tokens_per_s"],
                1.4,
            )
            self.assertAlmostEqual(
                kairos_no_thrash["metrics"]["average_job_throughput_jobs_per_s"],
                0.05,
            )

            kairos_thrash_aware = payload["variants"][2]
            self.assertAlmostEqual(
                kairos_thrash_aware["metrics"]["p5_output_throughput_tokens_per_s"],
                19.5,
            )
            self.assertEqual(
                kairos_thrash_aware["metrics"]["average_job_throughput_jobs_per_s"],
                0.0,
            )

            missing_log = missing_log_path.read_text(encoding="utf-8")
            self.assertIn("[missing-file]", missing_log)
            self.assertIn("job-throughput-timeseries.json", missing_log)

    def test_plot_con_ctrl_compare_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "con-ctrl-compare.json"
            output_path = tmp_path / "con-ctrl-compare.png"
            _write_json(
                input_path,
                {
                    "figure_name": "con-ctrl-compare",
                    "dataset": "dabstep",
                    "agent": "mini-swe-agent",
                    "qps": 0.05,
                    "variants": [
                        {
                            "label": "No Freq Control",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 28.6,
                                "average_job_throughput_jobs_per_s": 0.051,
                            },
                        },
                        {
                            "label": "KAIROS\nNo Thrash Avoid.",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 2.3,
                                "average_job_throughput_jobs_per_s": 0.046,
                            },
                        },
                        {
                            "label": "KAIROS\nThrash Avoid.",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 19.9,
                                "average_job_throughput_jobs_per_s": 0.050,
                            },
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
