"""Standalone vLLM Prometheus metrics monitor for con-driver runs."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_ENDPOINT = "http://localhost:12138/metrics"
DEFAULT_INTERVAL_S = 1.0
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_BLOCK_SIZE = 100
BLOCK_INDEX_FILE = "blocks.index.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_endpoint(endpoint: str) -> str:
    value = endpoint.strip()
    if not value:
        raise ValueError("Endpoint cannot be empty.")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value


def _fetch_metrics(endpoint: str, timeout_s: float) -> str:
    response = requests.get(endpoint, timeout=timeout_s)
    response.raise_for_status()
    return response.text


def _build_raw_record(raw_metrics: str) -> dict[str, Any]:
    return {
        "timestamp": int(time.time()),
        "captured_at": _utc_now_iso(),
        "content": raw_metrics,
    }


@dataclass
class _BlockWriter:
    output_dir: Path
    block_size: int
    endpoint: str
    interval_s: float
    timeout_s: float

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._next_block_id = 0
        self._block_index: list[dict[str, Any]] = []
        self._total_records = 0
        self._started_at = _utc_now_iso()

    def append(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        self._total_records += 1
        if len(self._records) >= self.block_size:
            self._flush_block()

    def finalize(self, *, status: str, error: str | None = None) -> None:
        if self._records:
            self._flush_block()

        payload: dict[str, Any] = {
            "version": 1,
            "status": status,
            "started_at": self._started_at,
            "finished_at": _utc_now_iso(),
            "endpoint": self.endpoint,
            "interval_s": self.interval_s,
            "timeout_s": self.timeout_s,
            "block_size": self.block_size,
            "total_records": self._total_records,
            "block_count": len(self._block_index),
            "blocks": self._block_index,
        }
        if error is not None:
            payload["error"] = error

        (self.output_dir / BLOCK_INDEX_FILE).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _flush_block(self) -> None:
        block_id = self._next_block_id
        self._next_block_id += 1

        first = self._records[0] if self._records else None
        last = self._records[-1] if self._records else None
        jsonl_name = f"block-{block_id:06d}.jsonl"
        tar_name = f"block-{block_id:06d}.tar.gz"
        tar_path = self.output_dir / tar_name

        with tempfile.TemporaryDirectory(prefix="con-driver-vllm-") as tmp_dir_raw:
            tmp_dir = Path(tmp_dir_raw)
            jsonl_path = tmp_dir / jsonl_name
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in self._records:
                    handle.write(json.dumps(record) + "\n")

            with tarfile.open(tar_path, mode="w:gz") as archive:
                archive.add(jsonl_path, arcname=jsonl_name)

        self._block_index.append(
            {
                "block_id": block_id,
                "file": tar_name,
                "member": jsonl_name,
                "record_count": len(self._records),
                "first_timestamp": first.get("timestamp") if isinstance(first, dict) else None,
                "last_timestamp": last.get("timestamp") if isinstance(last, dict) else None,
                "first_captured_at": (
                    first.get("captured_at") if isinstance(first, dict) else None
                ),
                "last_captured_at": (
                    last.get("captured_at") if isinstance(last, dict) else None
                ),
                "written_at": _utc_now_iso(),
                "tar_size_bytes": tar_path.stat().st_size,
            }
        )
        self._records = []


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor vLLM /metrics and store compressed JSONL blocks."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Prometheus metrics endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where vllm-log artifacts are written.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP timeout for each metrics request in seconds.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Number of records per compressed block.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print polling activity to stderr.",
    )
    return parser.parse_args(argv)


def run_monitor(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    endpoint = _normalize_endpoint(args.endpoint)
    output_dir = Path(args.output_dir).expanduser().resolve()
    interval_s = float(args.interval_s)
    timeout_s = float(args.timeout_s)
    block_size = int(args.block_size)

    if interval_s <= 0:
        raise ValueError("--interval-s must be > 0")
    if timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if block_size <= 0:
        raise ValueError("--block-size must be > 0")

    writer = _BlockWriter(
        output_dir=output_dir,
        block_size=block_size,
        endpoint=endpoint,
        interval_s=interval_s,
        timeout_s=timeout_s,
    )

    stopping = False

    def _stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    while not stopping:
        started = time.monotonic()
        try:
            if args.verbose:
                print(f"Polling {endpoint}", file=sys.stderr)
            raw = _fetch_metrics(endpoint, timeout_s)
            writer.append(_build_raw_record(raw))
        except Exception as exc:
            writer.finalize(status="failed", error=str(exc))
            raise RuntimeError(f"vLLM monitor failed: {exc}") from exc

        elapsed = time.monotonic() - started
        sleep_s = max(0.0, interval_s - elapsed)
        deadline = time.monotonic() + sleep_s
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(0.2, deadline - time.monotonic()))

    writer.finalize(status="stopped")
    return 0


def main() -> None:
    raise SystemExit(run_monitor())


if __name__ == "__main__":
    main()
