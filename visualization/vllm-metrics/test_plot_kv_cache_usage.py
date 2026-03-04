from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = THIS_DIR / "plot_kv_cache_usage.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("plot_kv_cache_usage", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_kv_cache_usage_writes_pdf(tmp_path: Path) -> None:
    module = _load_script_module()

    run_dir = tmp_path / "job"
    input_dir = run_dir / "post-processed" / "vllm-log"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "gauge-counter-timeseries.json"
    input_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "vllm:kv_cache_usage_perc|engine=0": {
                        "name": "vllm:kv_cache_usage_perc",
                        "sample_name": "vllm:kv_cache_usage_perc",
                        "family": "vllm:kv_cache_usage_perc",
                        "type": "gauge",
                        "help": "KV-cache usage. 1 means 100 percent usage.",
                        "labels": {"engine": "0"},
                        "captured_at": [
                            "2026-03-01T00:00:00+00:00",
                            "2026-03-01T00:00:01+00:00",
                        ],
                        "value": [0.1, 0.3],
                        "time_from_start_s": [0.0, 1.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_path = (
        run_dir / "visualization" / "vllm-log" / "kv_cache_usage" / "kv_cache_usage.pdf"
    )
    curated_path = output_path.with_suffix(".json")
    assert output_path.is_file()
    assert curated_path.is_file()
    assert output_path.read_bytes().startswith(b"%PDF-1.4")
    curated_payload = json.loads(curated_path.read_text(encoding="utf-8"))
    assert curated_payload["series_count"] == 1
    assert curated_payload["series"][0]["y"] == [0.1, 0.3]
