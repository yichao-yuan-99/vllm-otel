from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import amd_power_reader as power_reader


def test_resolve_gpu_indices_from_daemon(monkeypatch) -> None:
    def fake_request_gpu_list(*, socket_path: str, timeout: float) -> dict[str, object]:
        assert socket_path == "/tmp/test.sock"
        assert timeout == 2.0
        return {
            "ok": True,
            "gpus": [
                {"index": 0},
                {"index": 2},
            ],
        }

    monkeypatch.setattr(power_reader, "request_gpu_list", fake_request_gpu_list)

    indices = power_reader.resolve_gpu_indices(
        socket_path="/tmp/test.sock",
        gpu_indices=None,
        timeout=2.0,
    )

    assert indices == [0, 2]


def test_collect_power_sample(monkeypatch) -> None:
    def fake_request_power(
        *,
        gpu_index: int,
        socket_path: str,
        timeout: float,
    ) -> dict[str, object]:
        assert socket_path == "/tmp/test.sock"
        assert timeout == 3.0
        return {
            "ok": True,
            "gpu": {"index": gpu_index, "uuid": f"uuid-{gpu_index}"},
            "power_w": 40.0 + gpu_index,
            "power_info": {"socket_power": 40 + gpu_index},
        }

    monkeypatch.setattr(power_reader, "request_power", fake_request_power)

    sample = power_reader.collect_power_sample(
        socket_path="/tmp/test.sock",
        gpu_indices=[0, 1],
        timeout=3.0,
    )

    payload = sample["payload"]["/tmp/test.sock"]
    assert sample["timestamp"].endswith("Z")
    assert payload["gpu_power_w"] == {"0": 40.0, "1": 41.0}
    assert payload["gpu_payload"]["1"]["power_info"]["socket_power"] == 41


def test_run_power_reader_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        power_reader,
        "resolve_gpu_indices",
        lambda **kwargs: [0],
    )
    monkeypatch.setattr(
        power_reader,
        "collect_power_sample",
        lambda **kwargs: {
            "timestamp": "2026-04-07T18:30:00.123456Z",
            "payload": {
                "/tmp/test.sock": {
                    "timestamp_s": 1.0,
                    "gpu_power_w": {"0": 40.0},
                    "gpu_payload": {"0": {"gpu": {"index": 0}, "power_info": {}}},
                }
            },
        },
    )

    power_reader.run_power_reader(
        output_dir=tmp_path,
        gpu_indices=None,
        socket_path="/tmp/test.sock",
        interval_s=0.01,
        timeout=1.0,
        max_samples=1,
    )

    output_file = tmp_path / "power-log.jsonl"
    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["payload"]["/tmp/test.sock"]["gpu_power_w"]["0"] == 40.0
