#!/usr/bin/env python3
"""Materialize the multi-policy power comparison figure dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "data"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "mutli.json"
DEFAULT_MISSING_LOG_PATH = DEFAULT_OUTPUT_DIR / "mutli.missing.log"
DEFAULT_VARIANTS = (
    {
        "variant_key": "no_freq_control",
        "label": "No Freq Control",
        "input_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-gateway_multi/"
            "swebench-verified/mini-swe-agent/split/exclude-unranked/"
            "qps0_15/20260414T183055Z"
        ),
    },
    {
        "variant_key": "round_robin",
        "label": "Round Robin",
        "input_path": (
            "/srv/local/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-gateway_multi-round_robin-"
            "freq-ctrl-linesapce-instance-ctx-aware/swebench-verified/"
            "mini-swe-agent/split/exclude-unranked/qps0_16/20260416T004134Z"
        ),
    },
    {
        "variant_key": "kairos",
        "label": "KAIROS",
        "input_path": (
            "/srv/scratch/yichaoy2/work/vllm-otel/results/replay/"
            "sweep-qps-docker-power-clean-gateway_multi-lowest_profile_leq_reloc_2-"
            "freq-ctrl-linesapce-instance-ctx-aware/swebench-verified/"
            "mini-swe-agent/split/exclude-unranked/qps0_16/20260415T054807Z"
        ),
    },
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the mutli figure dataset from three replay runs."
    )
    parser.add_argument(
        "--no-freq-control-path",
        default=DEFAULT_VARIANTS[0]["input_path"],
        help="Baseline run path or parent directory containing timestamped runs.",
    )
    parser.add_argument(
        "--round-robin-path",
        default=DEFAULT_VARIANTS[1]["input_path"],
        help="Round-robin run path or parent directory containing timestamped runs.",
    )
    parser.add_argument(
        "--kairos-path",
        default=DEFAULT_VARIANTS[2]["input_path"],
        help="KAIROS run path or parent directory containing timestamped runs.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path. Default: figures/mutli/data/mutli.json",
    )
    parser.add_argument(
        "--missing-log",
        default=str(DEFAULT_MISSING_LOG_PATH),
        help="Missing-data log path. Default: figures/mutli/data/mutli.missing.log",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _stable_round(value: float | None) -> float:
    return 0.0 if value is None else round(value, 6)


def _select_run_dir(base_path: Path, missing_log: list[str]) -> Path | None:
    if not base_path.exists():
        missing_log.append(f"[missing-path] {base_path}")
        return None

    if (base_path / "replay" / "summary.json").is_file():
        return base_path

    candidates = sorted(
        (child for child in base_path.iterdir() if child.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "replay" / "summary.json").is_file():
            return candidate

    missing_log.append(f"[missing-run-dir] {base_path}")
    return None


def _extract_average_power_w(
    power_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    missing_log: list[str],
) -> float | None:
    if power_payload is None:
        missing_log.append(
            f"[missing-file] {run_dir / 'post-processed/power/power-summary.json'}"
        )
        return None

    power_stats = power_payload.get("power_stats_w")
    if isinstance(power_stats, dict):
        avg_power_w = _float_or_none(power_stats.get("avg"))
        if avg_power_w is not None:
            return avg_power_w

    power_points = power_payload.get("power_points")
    if isinstance(power_points, list):
        values = [
            numeric
            for numeric in (
                _float_or_none(point.get("power_w"))
                for point in power_points
                if isinstance(point, dict)
            )
            if numeric is not None
        ]
        if values:
            return sum(values) / len(values)

    missing_log.append(
        f"[missing-metric] average_power_w {run_dir / 'post-processed/power/power-summary.json'}"
    )
    return None


def _extract_total_energy_j(power_payload: dict[str, Any] | None) -> float | None:
    if not isinstance(power_payload, dict):
        return None
    return _float_or_none(power_payload.get("total_energy_j"))


def _extract_per_gpu_average_power(
    power_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    missing_log: list[str],
) -> list[dict[str, Any]]:
    if power_payload is None:
        return []

    raw_per_gpu = power_payload.get("per_gpu_power")
    if not isinstance(raw_per_gpu, list):
        missing_log.append(
            f"[missing-metric] per_gpu_power {run_dir / 'post-processed/power/power-summary.json'}"
        )
        return []

    per_gpu: list[dict[str, Any]] = []
    for index, item in enumerate(raw_per_gpu):
        if not isinstance(item, dict):
            continue
        points = item.get("power_points")
        if not isinstance(points, list):
            continue
        values = [
            numeric
            for numeric in (
                _float_or_none(point.get("power_w"))
                for point in points
                if isinstance(point, dict)
            )
            if numeric is not None
        ]
        if not values:
            continue
        avg_power_w = sum(values) / len(values)
        display_label = item.get("display_label")
        if not isinstance(display_label, str) or not display_label:
            display_label = f"GPU {index}"
        per_gpu.append(
            {
                "gpu_index": index,
                "display_label": display_label,
                "average_power_w": _stable_round(avg_power_w),
            }
        )

    if not per_gpu:
        missing_log.append(
            f"[missing-metric] per_gpu_average_power_w {run_dir / 'post-processed/power/power-summary.json'}"
        )
        return []

    total = sum(item["average_power_w"] for item in per_gpu)
    for item in per_gpu:
        item["share_pct"] = _stable_round(
            100.0 * item["average_power_w"] / total if total > 0.0 else 0.0
        )
    return per_gpu


def _build_variant_record(
    variant_key: str,
    label: str,
    input_path: str,
    *,
    missing_log: list[str],
) -> dict[str, Any]:
    base_path = Path(input_path).expanduser().resolve()
    run_dir = _select_run_dir(base_path, missing_log)

    power_payload: dict[str, Any] | None = None
    if run_dir is not None:
        power_path = run_dir / "post-processed" / "power" / "power-summary.json"
        if power_path.is_file():
            loaded = _load_json(power_path)
            if isinstance(loaded, dict):
                power_payload = loaded
        else:
            missing_log.append(f"[missing-file] {power_path}")

    selected_run_dir = run_dir if run_dir is not None else base_path
    average_power_w = _extract_average_power_w(
        power_payload,
        run_dir=selected_run_dir,
        missing_log=missing_log,
    )
    total_energy_j = _extract_total_energy_j(power_payload)
    per_gpu_average_power = _extract_per_gpu_average_power(
        power_payload,
        run_dir=selected_run_dir,
        missing_log=missing_log,
    )

    return {
        "variant_key": variant_key,
        "label": label,
        "input_path": str(base_path),
        "resolved_run_dir": str(selected_run_dir),
        "metrics": {
            "average_power_w": _stable_round(average_power_w),
            "total_energy_j": _stable_round(total_energy_j),
        },
        "per_gpu_average_power": per_gpu_average_power,
    }


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    missing_log_path = Path(args.missing_log).expanduser().resolve()

    missing_log: list[str] = []
    variants = [
        _build_variant_record(
            "no_freq_control",
            "No Freq Control",
            args.no_freq_control_path,
            missing_log=missing_log,
        ),
        _build_variant_record(
            "round_robin",
            "Round Robin",
            args.round_robin_path,
            missing_log=missing_log,
        ),
        _build_variant_record(
            "kairos",
            "KAIROS",
            args.kairos_path,
            missing_log=missing_log,
        ),
    ]

    payload = {
        "figure_name": "mutli",
        "bar_metric_key": "average_power_w",
        "bar_metric_label": "Average Power (W)",
        "pie_metric_key": "per_gpu_average_power",
        "variant_count": len(variants),
        "variants": variants,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    missing_log_path.parent.mkdir(parents=True, exist_ok=True)
    missing_log_path.write_text(
        ("\n".join(missing_log) + "\n") if missing_log else "",
        encoding="utf-8",
    )

    print(f"[written] {output_path}")
    print(f"[written] {missing_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
