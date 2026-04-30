from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = REPO_ROOT / "figures" / "overlap-3" / "materialize_overlap_3.py"
PLOT_SCRIPT = REPO_ROOT / "figures" / "overlap-3" / "plot_overlap_3.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_run(
    root_dir: Path,
    *,
    case_slug: str | None = None,
    run_parent_dir: Path | None = None,
    run_dir_name: str,
    throughputs: list[float],
    bin_size: float = 1.0,
) -> Path:
    if run_parent_dir is not None:
        run_dir = run_parent_dir / run_dir_name
    elif case_slug is not None:
        run_dir = (
            root_dir
            / case_slug
            / "dabstep"
            / "mini-swe-agent"
            / "split"
            / "exclude-unranked"
            / "qps0_03"
            / run_dir_name
        )
    else:
        raise ValueError("Either case_slug or run_parent_dir must be provided")
    _write_json(
        run_dir / "replay" / "summary.json",
        {"run_id": run_dir_name},
    )
    _write_json(
        run_dir / "post-processed" / "agent-output-throughput" / "agent-output-throughput.json",
        {
            "source_run_dir": str(run_dir),
            "agent_output_throughput_tokens_per_s_histogram": {
                "metric": "output_throughput_tokens_per_s",
                "bin_size": bin_size,
                "bins": [],
            },
            "agents": [
                {
                    "gateway_run_id": f"agent-{index}",
                    "output_tokens": int(round(throughput * 10.0)),
                    "llm_request_duration_s": 10.0,
                    "output_throughput_tokens_per_s": throughput,
                }
                for index, throughput in enumerate(throughputs)
            ],
        },
    )
    return run_dir


class Overlap3FigureTest(unittest.TestCase):
    def test_materialize_builds_shared_histograms_for_three_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "replay"

            baseline_qps_dir = root_dir / "sweep-qps-docker-power-clean" / "dabstep" / "mini-swe-agent" / "split" / "exclude-unranked" / "qps0_03"
            _make_run(
                root_dir,
                case_slug="sweep-qps-docker-power-clean",
                run_dir_name="20260410T010101Z",
                throughputs=[1.1, 1.9, 3.2],
            )
            variant_a_run = _make_run(
                root_dir,
                case_slug="sweep-qps-docker-power-clean-freq-ctrl-linespace-instance",
                run_dir_name="20260412T014805Z",
                throughputs=[2.1, 2.3, 2.9],
            )
            variant_b_parent = root_dir / "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo" / "dabstep" / "mini-swe-agent" / "split" / "exclude-unranked" / "qps0_03" / "35"
            _make_run(
                root_dir,
                run_parent_dir=variant_b_parent,
                run_dir_name="20260413T010101Z",
                throughputs=[1.5, 2.5, 3.5],
            )
            variant_c_parent = root_dir / "sweep-qps-docker-power-clean-freq-ctrl-linespace-instance-slo" / "dabstep" / "mini-swe-agent" / "split" / "exclude-unranked" / "qps0_03" / "45"
            _make_run(
                root_dir,
                run_parent_dir=variant_c_parent,
                run_dir_name="20260414T010101Z",
                throughputs=[4.1, 4.9],
            )

            output_path = tmp_path / "overlap-3.json"
            missing_log_path = tmp_path / "overlap-3.missing.log"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--baseline-path",
                    str(baseline_qps_dir),
                    "--slo-20-path",
                    str(variant_a_run),
                    "--slo-35-path",
                    str(variant_b_parent),
                    "--slo-45-path",
                    str(variant_c_parent),
                    "--output",
                    str(output_path),
                    "--missing-log",
                    str(missing_log_path),
                    "--bin-size",
                    "1.0",
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "overlap-3")
            self.assertEqual(payload["variant_count"], 4)
            self.assertEqual(payload["comparison_count"], 3)
            self.assertEqual(payload["bin_size"], 1.0)
            self.assertEqual(
                [comparison["label"] for comparison in payload["comparisons"]],
                ["KAIROS SLO 20", "KAIROS SLO 35", "KAIROS SLO 45"],
            )
            baseline_bins = payload["comparisons"][0]["baseline_histogram"]["bins"]
            variant_bins = payload["comparisons"][0]["histogram"]["bins"]
            self.assertEqual(
                [(item["bin_start"], item["count"]) for item in baseline_bins],
                [(1.0, 2), (2.0, 0), (3.0, 1), (4.0, 0)],
            )
            self.assertEqual(
                [(item["bin_start"], item["count"]) for item in variant_bins],
                [(1.0, 0), (2.0, 3), (3.0, 0), (4.0, 0)],
            )
            self.assertTrue(missing_log_path.is_file())

    def test_plot_overlap_3_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "overlap-3.json"
            output_path = tmp_path / "overlap-3.pdf"
            _write_json(
                input_path,
                {
                    "figure_name": "overlap-3",
                    "comparisons": [
                        {
                            "baseline_label": "No Freq Control",
                            "label": "KAIROS SLO 20",
                            "baseline_color": "#111111",
                            "color": "#1D4F91",
                            "baseline_histogram": {
                                "bins": [
                                    {"bin_start": 1.0, "bin_end": 2.0, "count": 2},
                                    {"bin_start": 2.0, "bin_end": 3.0, "count": 0},
                                    {"bin_start": 3.0, "bin_end": 4.0, "count": 1},
                                ]
                            },
                            "histogram": {
                                "bins": [
                                    {"bin_start": 1.0, "bin_end": 2.0, "count": 0},
                                    {"bin_start": 2.0, "bin_end": 3.0, "count": 3},
                                    {"bin_start": 3.0, "bin_end": 4.0, "count": 0},
                                ]
                            },
                        },
                        {
                            "baseline_label": "No Freq Control",
                            "label": "KAIROS SLO 35",
                            "baseline_color": "#111111",
                            "color": "#0F766E",
                            "baseline_histogram": {
                                "bins": [
                                    {"bin_start": 1.0, "bin_end": 2.0, "count": 2},
                                    {"bin_start": 2.0, "bin_end": 3.0, "count": 0},
                                    {"bin_start": 3.0, "bin_end": 4.0, "count": 1},
                                ]
                            },
                            "histogram": {
                                "bins": [
                                    {"bin_start": 1.0, "bin_end": 2.0, "count": 1},
                                    {"bin_start": 2.0, "bin_end": 3.0, "count": 1},
                                    {"bin_start": 3.0, "bin_end": 4.0, "count": 1},
                                ]
                            },
                        },
                        {
                            "baseline_label": "No Freq Control",
                            "label": "KAIROS SLO 45",
                            "baseline_color": "#111111",
                            "color": "#B45309",
                            "baseline_histogram": {
                                "bins": [
                                    {"bin_start": 1.0, "bin_end": 2.0, "count": 2},
                                    {"bin_start": 2.0, "bin_end": 3.0, "count": 0},
                                    {"bin_start": 3.0, "bin_end": 4.0, "count": 1},
                                ]
                            },
                            "histogram": {
                                "bins": [
                                    {"bin_start": 1.0, "bin_end": 2.0, "count": 0},
                                    {"bin_start": 2.0, "bin_end": 3.0, "count": 0},
                                    {"bin_start": 3.0, "bin_end": 4.0, "count": 2},
                                ]
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
