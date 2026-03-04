from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = THIS_DIR / "plot_prefill_computed_tokens.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("plot_prefill_computed_tokens", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_prefill_computed_tokens_writes_both_pdfs(tmp_path: Path) -> None:
    module = _load_script_module()

    run_dir = tmp_path / "job"
    input_dir = run_dir / "post-processed" / "vllm-log"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "gauge-counter-timeseries.json"
    input_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "vllm:prompt_tokens|engine=0": {
                        "name": "vllm:prompt_tokens",
                        "sample_name": "vllm:prompt_tokens_total",
                        "family": "vllm:prompt_tokens",
                        "type": "counter",
                        "help": "Prompt tokens.",
                        "labels": {"engine": "0"},
                        "captured_at": [
                            "2026-03-01T00:00:00+00:00",
                            "2026-03-01T00:00:01+00:00",
                            "2026-03-01T00:00:02+00:00",
                        ],
                        "value": [12.0, 18.0, 30.0],
                        "time_from_start_s": [0.0, 1.0, 2.0],
                    },
                    "vllm:prefix_cache_hits|engine=0": {
                        "name": "vllm:prefix_cache_hits",
                        "sample_name": "vllm:prefix_cache_hits_total",
                        "family": "vllm:prefix_cache_hits",
                        "type": "counter",
                        "help": "Prefix cache hits.",
                        "labels": {"engine": "0"},
                        "captured_at": [
                            "2026-03-01T00:00:00+00:00",
                            "2026-03-01T00:00:01+00:00",
                            "2026-03-01T00:00:02+00:00",
                        ],
                        "value": [2.0, 3.0, 11.0],
                        "time_from_start_s": [0.0, 1.0, 2.0],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--run-dir", str(run_dir)])
    assert exit_code == 0

    output_dir = run_dir / "visualization" / "vllm-log" / "prefill_computed_tokens"
    raw_output = output_dir / "prefill_computed_tokens.raw.pdf"
    avg_output = output_dir / "prefill_computed_tokens.avg5.pdf"
    raw_curated = output_dir / "prefill_computed_tokens.raw.json"
    avg_curated = output_dir / "prefill_computed_tokens.avg5.json"

    assert raw_output.is_file()
    assert avg_output.is_file()
    assert raw_curated.is_file()
    assert avg_curated.is_file()
    assert raw_output.read_bytes().startswith(b"%PDF-1.4")
    assert avg_output.read_bytes().startswith(b"%PDF-1.4")
    raw_payload = json.loads(raw_curated.read_text(encoding="utf-8"))
    avg_payload = json.loads(avg_curated.read_text(encoding="utf-8"))
    assert raw_payload["series"][0]["y"] == [10.0, 15.0, 19.0]
    assert avg_payload["series"][0]["y"] == [14.666666666666666]
