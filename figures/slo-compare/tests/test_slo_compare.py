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
    REPO_ROOT / "figures" / "slo-compare" / "materialize_slo_compare.py"
)
PLOT_SCRIPT = REPO_ROOT / "figures" / "slo-compare" / "plot_slo_compare.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_run(
    run_dir: Path,
    *,
    avg_power_w: float | None = None,
    power_points: list[float] | None = None,
    p5_throughput: float | None = None,
    agent_throughputs: list[float] | None = None,
    include_power: bool = True,
    include_throughput: bool = True,
) -> None:
    _write_json(
        run_dir / "replay" / "summary.json",
        {
            "workers_completed": 3,
            "workers_failed": 0,
        },
    )
    if include_power:
        power_payload: dict[str, object] = {
            "analysis_window_start_utc": "2026-04-01T00:00:00Z",
            "analysis_window_end_utc": "2026-04-01T00:01:00Z",
        }
        if avg_power_w is not None:
            power_payload["power_stats_w"] = {"avg": avg_power_w}
        if power_points is not None:
            power_payload["power_points"] = [
                {"time_offset_s": float(index), "power_w": value}
                for index, value in enumerate(power_points)
            ]
        _write_json(
            run_dir / "post-processed" / "power" / "power-summary.json",
            power_payload,
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


class SloCompareFigureTest(unittest.TestCase):
    def test_plot_slo_compare_uses_distinct_colormaps_with_black_baseline(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        module_spec = importlib.util.spec_from_file_location(
            "plot_slo_compare",
            PLOT_SCRIPT,
        )
        assert module_spec is not None
        module = importlib.util.module_from_spec(module_spec)
        assert module_spec.loader is not None
        module_spec.loader.exec_module(module)

        throughput_colors = module._bar_colors(
            4,
            palette=module.THROUGHPUT_COLORS,
        )
        power_colors = module._bar_colors(
            4,
            palette=module.POWER_COLORS,
        )

        self.assertEqual(throughput_colors[0], module.NO_FREQ_CONTROL_COLOR)
        self.assertEqual(power_colors[0], module.NO_FREQ_CONTROL_COLOR)
        self.assertEqual(throughput_colors[1:], list(module.THROUGHPUT_COLORS))
        self.assertEqual(power_colors[1:], list(module.POWER_COLORS))

    def test_materialize_slo_compare_supports_direct_and_timestamped_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            uncontrolled_dir = tmp_path / "no-freq"
            no_slo_root = tmp_path / "freq-ctrl-no-slo"
            slo_35_root = tmp_path / "slo-35"
            slo_45_root = tmp_path / "slo-45"

            _make_run(
                uncontrolled_dir,
                avg_power_w=330.0,
                p5_throughput=44.0,
            )
            _make_run(
                no_slo_root / "20260411T000000Z",
                power_points=[200.0, 220.0, 240.0],
                agent_throughputs=[20.0, 40.0, 60.0, 80.0],
            )
            _make_run(
                no_slo_root / "20260412T000000Z",
                avg_power_w=210.0,
                p5_throughput=26.0,
            )
            _make_run(
                slo_35_root / "20260412T055419Z",
                avg_power_w=255.0,
                agent_throughputs=[30.0, 35.0, 40.0, 50.0],
            )
            _make_run(
                slo_45_root / "20260412T020056Z",
                p5_throughput=38.0,
                include_power=False,
            )

            output_path = tmp_path / "slo-compare.json"
            missing_log_path = tmp_path / "slo-compare.missing.log"
            module_spec = importlib.util.spec_from_file_location(
                "materialize_slo_compare",
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
                    "base_path": str(uncontrolled_dir),
                },
                {
                    "variant_key": "freq_control_no_slo",
                    "label": "Freq Control\nSLO 20",
                    "base_path": str(no_slo_root),
                },
                {
                    "variant_key": "freq_control_slo_35",
                    "label": "Freq Control\nSLO 35",
                    "base_path": str(slo_35_root),
                },
                {
                    "variant_key": "freq_control_slo_45",
                    "label": "Freq Control\nSLO 45",
                    "base_path": str(slo_45_root),
                },
            )
            try:
                module.main = module.main
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import importlib.util, pathlib, sys; "
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
            self.assertEqual(payload["figure_name"], "slo-compare")
            self.assertEqual(payload["variant_count"], 4)

            no_freq = payload["variants"][0]
            self.assertEqual(no_freq["selected_run_dir"], str(uncontrolled_dir.resolve()))
            self.assertAlmostEqual(
                no_freq["metrics"]["p5_output_throughput_tokens_per_s"],
                44.0,
            )
            self.assertAlmostEqual(no_freq["metrics"]["average_power_w"], 330.0)

            no_slo = payload["variants"][1]
            self.assertTrue(no_slo["selected_run_dir"].endswith("20260412T000000Z"))
            self.assertAlmostEqual(no_slo["metrics"]["average_power_w"], 210.0)
            self.assertAlmostEqual(
                no_slo["metrics"]["p5_output_throughput_tokens_per_s"],
                26.0,
            )

            slo_35 = payload["variants"][2]
            self.assertAlmostEqual(slo_35["metrics"]["average_power_w"], 255.0)
            self.assertAlmostEqual(
                slo_35["metrics"]["p5_output_throughput_tokens_per_s"],
                30.75,
            )

            slo_45 = payload["variants"][3]
            self.assertEqual(slo_45["metrics"]["average_power_w"], 0.0)
            self.assertAlmostEqual(
                slo_45["metrics"]["p5_output_throughput_tokens_per_s"],
                38.0,
            )

            missing_log = missing_log_path.read_text(encoding="utf-8")
            self.assertIn("[missing-file]", missing_log)
            self.assertIn("power-summary.json", missing_log)

    def test_plot_slo_compare_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            input_path = tmp_path / "slo-compare.json"
            output_path = tmp_path / "slo-compare.png"
            _write_json(
                input_path,
                {
                    "figure_name": "slo-compare",
                    "dataset": "dabstep",
                    "agent": "mini-swe-agent",
                    "qps": 0.03,
                    "variants": [
                        {
                            "label": "No Freq Control",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 45.0,
                                "average_power_w": 330.0,
                            },
                        },
                        {
                            "label": "Freq Control\nSLO 20",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 26.0,
                                "average_power_w": 224.0,
                            },
                        },
                        {
                            "label": "Freq Control\nSLO 35",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 35.0,
                                "average_power_w": 255.0,
                            },
                        },
                        {
                            "label": "Freq Control\nSLO 45",
                            "metrics": {
                                "p5_output_throughput_tokens_per_s": 38.0,
                                "average_power_w": 249.0,
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
