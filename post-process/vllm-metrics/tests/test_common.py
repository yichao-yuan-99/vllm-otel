from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
COMMON_PATH = THIS_DIR.parent / "common" / "parse_metrics.py"


def _load_common_module():
    spec = importlib.util.spec_from_file_location("vllm_metrics_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_metric_content_to_json_parses_example_metrics() -> None:
    module = _load_common_module()
    example_path = THIS_DIR / "example.json"
    payload = json.loads(example_path.read_text(encoding="utf-8"))

    parsed = module.parse_metric_content_to_json(payload["content"])

    assert isinstance(parsed["timestamp"], int)
    families = parsed["families"]
    running = families["vllm:num_requests_running"]
    assert running["type"] == "gauge"
    assert running["help"] == "Number of requests in model execution batches."
    assert running["samples"][0]["name"] == "vllm:num_requests_running"
    assert running["samples"][0]["labels"]["engine"] == "0"
    assert running["samples"][0]["labels"]["model_name"] == "Qwen3-Coder-30B-A3B-Instruct"
    assert running["samples"][0]["value"] == 6.0

    request_success = families["vllm:request_success"]
    assert request_success["type"] == "counter"
    assert len(request_success["samples"]) >= 4
    assert request_success["samples"][0]["name"] == "vllm:request_success_total"

    response_duration = families["vllm:request_generation_tokens"]
    assert response_duration["type"] == "histogram"
    sample_names = {sample["name"] for sample in response_duration["samples"]}
    assert "vllm:request_generation_tokens_bucket" in sample_names
    assert "vllm:request_generation_tokens_count" in sample_names
    assert "vllm:request_generation_tokens_sum" in sample_names

    assert "http_request_duration_seconds" not in families
