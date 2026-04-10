from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE_SCRIPT = REPO_ROOT / "figures" / "tmp-overlap" / "materialize_tmp_overlap.py"
PLOT_SCRIPT = REPO_ROOT / "figures" / "tmp-overlap" / "plot_tmp_overlap.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_source(
    run_dir: Path,
    *,
    throughputs: list[float],
    bin_size: float = 1.0,
) -> Path:
    _write_json(
        run_dir / "post-processed" / "agent-output-throughput" / "agent-output-throughput.json",
        {
            "source_run_dir": str(run_dir),
            "agent_count": len(throughputs),
            "output_throughput_tokens_per_s": round(sum(throughputs) / len(throughputs), 6),
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

    figure_path = (
        run_dir
        / "post-processed"
        / "visualization"
        / "agent-output-throughput"
        / "agent-output-throughput-histogram.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"fake-png")
    return figure_path


class TmpOverlapFigureTest(unittest.TestCase):
    def test_materialize_rebuilds_shared_bin_histograms_from_figure_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "replay"

            source_a = _make_source(
                root_dir / "sweep-qps-baseline" / "bench" / "agent" / "qps0_1" / "20260401T010101Z",
                throughputs=[1.1, 1.9, 3.2],
            )
            source_b = _make_source(
                root_dir / "sweep-qps-variant" / "bench" / "agent" / "qps0_1" / "20260402T020202Z",
                throughputs=[2.2, 2.7],
            )

            output_path = tmp_path / "tmp-overlap.json"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--source-a",
                    str(source_a),
                    "--source-b",
                    str(source_b),
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=str(REPO_ROOT),
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["figure_name"], "tmp-overlap")
            self.assertEqual(payload["dataset_count"], 2)
            self.assertEqual(payload["bin_size"], 1.0)

            datasets = payload["datasets"]
            self.assertEqual([dataset["label"] for dataset in datasets], ["baseline", "variant"])
            self.assertTrue(
                datasets[0]["resolved_input_path"].endswith("agent-output-throughput.json")
            )
            self.assertEqual(datasets[0]["summary"]["sample_count"], 3)
            self.assertEqual(datasets[1]["summary"]["sample_count"], 2)

            histogram_a = datasets[0]["histogram"]["bins"]
            histogram_b = datasets[1]["histogram"]["bins"]
            self.assertEqual(
                [(item["bin_start"], item["count"]) for item in histogram_a],
                [(1.0, 2), (2.0, 0), (3.0, 1)],
            )
            self.assertEqual(
                [(item["bin_start"], item["count"]) for item in histogram_b],
                [(1.0, 0), (2.0, 2), (3.0, 0)],
            )
            self.assertEqual(
                payload["shared_histogram"]["bins"][1]["counts_by_dataset"],
                {"dataset_1": 0, "dataset_2": 2},
            )

    def test_plot_tmp_overlap_writes_output(self) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root_dir = tmp_path / "results" / "replay"

            source_a = _make_source(
                root_dir / "sweep-qps-baseline" / "bench" / "agent" / "qps0_1" / "20260401T010101Z",
                throughputs=[1.1, 1.9, 3.2],
            )
            source_b = _make_source(
                root_dir / "sweep-qps-variant" / "bench" / "agent" / "qps0_1" / "20260402T020202Z",
                throughputs=[2.2, 2.7, 3.4],
            )

            input_path = tmp_path / "tmp-overlap.json"
            output_path = tmp_path / "tmp-overlap.png"
            subprocess.run(
                [
                    sys.executable,
                    str(MATERIALIZE_SCRIPT),
                    "--source-a",
                    str(source_a),
                    "--source-b",
                    str(source_b),
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
