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
    REPO_ROOT / "figures" / "main-fig-1" / "materialize_main_fig_1.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "main-fig-1" / "plot_main_fig_1.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _sample_source_payload() -> dict[str, object]:
    implementations = [
        {
            "implementation_key": "uncontrolled",
            "implementation_label": "Uncontrolled",
            "source_root": "/tmp/uncontrolled",
        },
        {
            "implementation_key": "fixed_freq",
            "implementation_label": "Fixed Freq",
            "source_root": "/tmp/fixed-freq",
        },
        {
            "implementation_key": "steer",
            "implementation_label": "STEER",
            "source_root": "/tmp/steer",
        },
    ]
    metrics = [
        {
            "metric_key": "average_energy_per_finished_agent_kj",
            "metric_label": "Average Energy per Finished Agent",
            "panel_title": "Average Energy per Finished Agent",
            "metric_unit": "kJ",
            "y_axis_label": "Energy per Finished Agent (kJ)",
            "formula": "energy formula",
        },
        {
            "metric_key": "average_power_w",
            "metric_label": "Average Power",
            "panel_title": "Average Power",
            "metric_unit": "W",
            "y_axis_label": "Average Power (W)",
            "formula": "power formula",
        },
        {
            "metric_key": "average_context_usage_pct",
            "metric_label": "Average Context Usage",
            "panel_title": "Average Context Usage",
            "metric_unit": "%",
            "y_axis_label": "Average Context Usage (%)",
            "formula": "context formula",
        },
        {
            "metric_key": "p5_output_throughput_tokens_per_s",
            "metric_label": "P5 Output Throughput",
            "panel_title": "5th Percentile Output Throughput",
            "metric_unit": "tokens/s",
            "y_axis_label": "P5 Output Throughput (tokens/s)",
            "formula": "p5 formula",
        },
        {
            "metric_key": "pct_agents_above_20_output_throughput_tokens_per_s",
            "metric_label": "Agents Above 20 Tokens/s",
            "panel_title": "Agents Above 20 Tokens/s",
            "metric_unit": "%",
            "y_axis_label": "Agents Above 20 Tokens/s (%)",
            "formula": "pct formula",
        },
    ]
    experiment_specs = (
        ("A", "swebench-verified", "mini-swe-agent", ("0.04", "0.06", "0.08")),
        ("B", "dabstep", "mini-swe-agent", ("0.03", "0.04", "0.05")),
        ("C", "terminal-bench-2.0", "terminus-2", ("0.015", "0.02", "0.025")),
    )

    experiments = []
    for experiment_index, (experiment_id, dataset_slug, agent_slug, qps_labels) in enumerate(
        experiment_specs
    ):
        qps_entries = []
        for qps_index, qps_label in enumerate(qps_labels):
            implementation_entries = []
            for implementation_index, implementation in enumerate(implementations):
                base = ((experiment_index + 1) * 10.0) + (qps_index * 3.0) + implementation_index
                metric_values = {
                    "average_energy_per_finished_agent_kj": 4.0 + (base * 0.2),
                    "average_power_w": 120.0 + (base * 4.0),
                    "average_context_usage_pct": 10.0 + base,
                    "p5_output_throughput_tokens_per_s": 18.0 + base,
                    "pct_agents_above_20_output_throughput_tokens_per_s": 72.0 + base,
                }
                implementation_entries.append(
                    {
                        "implementation_key": implementation["implementation_key"],
                        "implementation_label": implementation["implementation_label"],
                        "source_root": implementation["source_root"],
                        "run_dir": (
                            f"/tmp/{dataset_slug}/{agent_slug}/qps{qps_label}/"
                            f"{implementation['implementation_key']}"
                        ),
                        "run_dir_name": "20260415T000000Z",
                        "candidate_run_count": 1,
                        "metric_values": metric_values,
                    }
                )
            qps_entries.append(
                {
                    "qps_slug": f"qps{qps_label.replace('.', '_')}",
                    "qps_value": float(qps_label),
                    "qps_label": qps_label,
                    "implementations": implementation_entries,
                }
            )
        experiments.append(
            {
                "experiment_id": experiment_id,
                "dataset_slug": dataset_slug,
                "dataset_label": dataset_slug,
                "agent_slug": agent_slug,
                "agent_label": agent_slug,
                "subplot_title": f"{experiment_id}. {dataset_slug} + {agent_slug}",
                "qps": qps_entries,
            }
        )

    return {
        "figure_name": "energy-context-latency",
        "missing_entry_count": 3,
        "implementations": implementations,
        "metrics": metrics,
        "experiments": experiments,
    }


class MainFigure1Test(unittest.TestCase):
    def test_materialize_main_fig_1_filters_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "energy-context-latency.json"
            input_missing_log = input_path.with_suffix(".missing.log")
            output_path = tmp_path / "main-fig-1.json"
            _write_json(input_path, _sample_source_payload())
            input_missing_log.write_text("missing upstream entries\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "main-fig-1")
            self.assertEqual(payload["source_figure_name"], "energy-context-latency")
            self.assertEqual(payload["panel_count"], 2)
            self.assertEqual(payload["metric_count"], 4)
            self.assertEqual(payload["source_missing_entry_count"], 3)
            self.assertEqual(payload["source_missing_log"], str(input_missing_log))
            self.assertEqual(
                payload["implementations"][1]["implementation_label"],
                "Fixed Freq (810Mhz)",
            )

            metric_keys = [metric["metric_key"] for metric in payload["metrics"]]
            self.assertEqual(
                metric_keys,
                [
                    "p5_output_throughput_tokens_per_s",
                    "pct_agents_above_20_output_throughput_tokens_per_s",
                    "average_power_w",
                    "average_energy_per_finished_agent_kj",
                ],
            )
            self.assertEqual(
                payload["panels"][0]["secondary_metric_key"],
                "pct_agents_above_20_output_throughput_tokens_per_s",
            )
            first_entry = payload["experiments"][0]["qps"][0]["implementations"][0]
            self.assertEqual(
                set(first_entry["metric_values"].keys()),
                {
                    "p5_output_throughput_tokens_per_s",
                    "pct_agents_above_20_output_throughput_tokens_per_s",
                    "average_power_w",
                    "average_energy_per_finished_agent_kj",
                },
            )
            self.assertNotIn("average_context_usage_pct", first_entry["metric_values"])
            self.assertEqual(
                payload["experiments"][0]["qps"][0]["implementations"][1][
                    "implementation_label"
                ],
                "Fixed Freq (810Mhz)",
            )

    def test_plot_main_fig_1_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "energy-context-latency.json"
            materialized_path = tmp_path / "main-fig-1.json"
            output_path = tmp_path / "main-fig-1.png"
            _write_json(input_path, _sample_source_payload())

            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(materialized_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PLOT_SCRIPT),
                    "--input",
                    str(materialized_path),
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
