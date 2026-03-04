from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = THIS_DIR / "plot_num_preemptions.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("plot_num_preemptions", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_num_preemptions_writes_both_pdfs(tmp_path: Path) -> None:
    module = _load_script_module()

    run_dir = tmp_path / "job"
    input_dir = run_dir / "post-processed" / "vllm-log"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "gauge-counter-timeseries.json"
    input_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "vllm:num_preemptions|engine=0": {
                        "name": "vllm:num_preemptions",
                        "sample_name": "vllm:num_preemptions_total",
                        "family": "vllm:num_preemptions",
                        "type": "counter",
                        "help": "Number of preemptions.",
                        "labels": {"engine": "0"},
                        "captured_at": [
                            "2026-03-01T00:00:00+00:00",
                            "2026-03-01T00:00:01+00:00",
                            "2026-03-01T00:00:02+00:00",
                        ],
                        "value": [0.0, 1.0, 3.0],
                        "time_from_start_s": [0.0, 1.0, 2.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_dir = run_dir / "visualization" / "vllm-log" / "num_preemptions"
    raw_output = output_dir / "num_preemptions.raw.pdf"
    avg_output = output_dir / "num_preemptions.avg5.pdf"
    raw_curated = output_dir / "num_preemptions.raw.json"
    avg_curated = output_dir / "num_preemptions.avg5.json"

    assert raw_output.is_file()
    assert avg_output.is_file()
    assert raw_curated.is_file()
    assert avg_curated.is_file()
    assert raw_output.read_bytes().startswith(b"%PDF-1.4")
    assert avg_output.read_bytes().startswith(b"%PDF-1.4")
    raw_payload = json.loads(raw_curated.read_text(encoding="utf-8"))
    avg_payload = json.loads(avg_curated.read_text(encoding="utf-8"))
    assert raw_payload["series"][0]["y"] == [0.0, 1.0, 3.0]
    assert avg_payload["series"][0]["y"] == [1.3333333333333333]
