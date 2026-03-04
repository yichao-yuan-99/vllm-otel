from __future__ import annotations

import importlib.util
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = THIS_DIR.parent / "parse_record.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("vllm_metrics_parse_record", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_metric_record_payload_uses_content_field() -> None:
    module = _load_script_module()
    example_path = THIS_DIR / "example.json"
    payload = json.loads(example_path.read_text(encoding="utf-8"))

    parsed = module.parse_metric_record_payload(payload)

    assert isinstance(parsed["timestamp"], int)
    assert "families" in parsed
    assert "vllm:num_requests_running" in parsed["families"]
