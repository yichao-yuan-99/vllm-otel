from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = THIS_DIR / "generate_all_figures.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_all_figures", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_all_figures_writes_all_outputs(tmp_path: Path) -> None:
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
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00"],
                        "value": [0.1, 0.2],
                        "time_from_start_s": [0.0, 1.0],
                    },
                    "vllm:num_requests_running|engine=0": {
                        "name": "vllm:num_requests_running",
                        "sample_name": "vllm:num_requests_running",
                        "family": "vllm:num_requests_running",
                        "type": "gauge",
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00"],
                        "value": [1.0, 2.0],
                        "time_from_start_s": [0.0, 1.0],
                    },
                    "vllm:num_requests_waiting|engine=0": {
                        "name": "vllm:num_requests_waiting",
                        "sample_name": "vllm:num_requests_waiting",
                        "family": "vllm:num_requests_waiting",
                        "type": "gauge",
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00"],
                        "value": [0.0, 3.0],
                        "time_from_start_s": [0.0, 1.0],
                    },
                    "vllm:generation_tokens|engine=0": {
                        "name": "vllm:generation_tokens",
                        "sample_name": "vllm:generation_tokens_total",
                        "family": "vllm:generation_tokens",
                        "type": "counter",
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00", "2026-03-01T00:00:02+00:00"],
                        "value": [10.0, 20.0, 15.0],
                        "time_from_start_s": [0.0, 1.0, 2.0],
                    },
                    "vllm:num_preemptions|engine=0": {
                        "name": "vllm:num_preemptions",
                        "sample_name": "vllm:num_preemptions_total",
                        "family": "vllm:num_preemptions",
                        "type": "counter",
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00", "2026-03-01T00:00:02+00:00"],
                        "value": [0.0, 1.0, 3.0],
                        "time_from_start_s": [0.0, 1.0, 2.0],
                    },
                    "vllm:prompt_tokens|engine=0": {
                        "name": "vllm:prompt_tokens",
                        "sample_name": "vllm:prompt_tokens_total",
                        "family": "vllm:prompt_tokens",
                        "type": "counter",
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00", "2026-03-01T00:00:02+00:00"],
                        "value": [12.0, 18.0, 30.0],
                        "time_from_start_s": [0.0, 1.0, 2.0],
                    },
                    "vllm:prefix_cache_hits|engine=0": {
                        "name": "vllm:prefix_cache_hits",
                        "sample_name": "vllm:prefix_cache_hits_total",
                        "family": "vllm:prefix_cache_hits",
                        "type": "counter",
                        "help": "",
                        "labels": {"engine": "0"},
                        "captured_at": ["2026-03-01T00:00:00+00:00", "2026-03-01T00:00:01+00:00", "2026-03-01T00:00:02+00:00"],
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

    expected = [
        run_dir / "visualization" / "vllm-log" / "kv_cache_usage" / "kv_cache_usage.pdf",
        run_dir / "visualization" / "vllm-log" / "kv_cache_usage" / "kv_cache_usage.json",
        run_dir / "visualization" / "vllm-log" / "num_requests_running" / "num_requests_running.pdf",
        run_dir / "visualization" / "vllm-log" / "num_requests_running" / "num_requests_running.json",
        run_dir / "visualization" / "vllm-log" / "num_requests_waiting" / "num_requests_waiting.pdf",
        run_dir / "visualization" / "vllm-log" / "num_requests_waiting" / "num_requests_waiting.json",
        run_dir / "visualization" / "vllm-log" / "generation_tokens" / "generation_tokens.raw.pdf",
        run_dir / "visualization" / "vllm-log" / "generation_tokens" / "generation_tokens.raw.json",
        run_dir / "visualization" / "vllm-log" / "generation_tokens" / "generation_tokens.avg5.pdf",
        run_dir / "visualization" / "vllm-log" / "generation_tokens" / "generation_tokens.avg5.json",
        run_dir / "visualization" / "vllm-log" / "num_preemptions" / "num_preemptions.raw.pdf",
        run_dir / "visualization" / "vllm-log" / "num_preemptions" / "num_preemptions.raw.json",
        run_dir / "visualization" / "vllm-log" / "num_preemptions" / "num_preemptions.avg5.pdf",
        run_dir / "visualization" / "vllm-log" / "num_preemptions" / "num_preemptions.avg5.json",
        run_dir / "visualization" / "vllm-log" / "prefill_computed_tokens" / "prefill_computed_tokens.raw.pdf",
        run_dir / "visualization" / "vllm-log" / "prefill_computed_tokens" / "prefill_computed_tokens.raw.json",
        run_dir / "visualization" / "vllm-log" / "prefill_computed_tokens" / "prefill_computed_tokens.avg5.pdf",
        run_dir / "visualization" / "vllm-log" / "prefill_computed_tokens" / "prefill_computed_tokens.avg5.json",
    ]
    for path in expected:
        assert path.is_file()
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-1.4")
