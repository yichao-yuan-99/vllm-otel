from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    build_grouped_average_series,
    load_timeseries_payload,
    render_line_chart_pdf,
    select_metric_series,
    write_curated_series_json,
)


RAW_OUTPUT_NAME = "prefill_computed_tokens.raw.pdf"
AVG5_OUTPUT_NAME = "prefill_computed_tokens.avg5.pdf"


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


def extract_prefill_computed_tokens_series(payload: dict) -> list[dict]:
    prompt_series = select_metric_series(payload, "vllm:prompt_tokens")
    prefix_series = select_metric_series(payload, "vllm:prefix_cache_hits")

    prompt_by_labels = {_label_key(item.get("labels") or {}): item for item in prompt_series}
    prefix_by_labels = {_label_key(item.get("labels") or {}): item for item in prefix_series}

    missing = sorted(set(prompt_by_labels) ^ set(prefix_by_labels))
    if missing:
        raise ValueError("prompt_tokens and prefix_cache_hits series do not align by labels")

    combined: list[dict] = []
    for key in sorted(prompt_by_labels):
        prompt = prompt_by_labels[key]
        prefix = prefix_by_labels[key]
        if prompt["x"] != prefix["x"]:
            raise ValueError("prompt_tokens and prefix_cache_hits series do not align in time")
        if len(prompt["y"]) != len(prefix["y"]):
            raise ValueError("prompt_tokens and prefix_cache_hits series do not align in length")

        combined.append(
            {
                "label": prompt["label"],
                "labels": dict(prompt.get("labels") or {}),
                "x": list(prompt["x"]),
                "y": [float(a) - float(b) for a, b in zip(prompt["y"], prefix["y"])],
            }
        )
    if not combined:
        raise ValueError("No prefill computed token series found in payload")
    return combined


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot prefill-computed token throughput from prompt_tokens - prefix_cache_hits."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run result root directory containing post-processed/vllm-log/gauge-counter-timeseries.json",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to gauge-counter-timeseries.json. Defaults to the run's post-processed location.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <run-dir>/visualization/vllm-log/prefill_computed_tokens/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    input_path = (
        Path(args.input).expanduser().resolve()
        if args.input
        else (run_dir / "post-processed" / "vllm-log" / "gauge-counter-timeseries.json").resolve()
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (run_dir / "visualization" / "vllm-log" / "prefill_computed_tokens").resolve()
    )

    payload = load_timeseries_payload(input_path)
    raw_series = extract_prefill_computed_tokens_series(payload)
    avg5_series = build_grouped_average_series(raw_series, group_size=5)

    raw_output = output_dir / RAW_OUTPUT_NAME
    avg5_output = output_dir / AVG5_OUTPUT_NAME

    render_line_chart_pdf(
        series=raw_series,
        output_path=raw_output,
        title=f"prefill computed tokens: {run_dir.name}",
        y_axis_label="tokens_per_sample",
    )
    render_line_chart_pdf(
        series=avg5_series,
        output_path=avg5_output,
        title=f"prefill computed tokens avg5: {run_dir.name}",
        y_axis_label="avg_tokens_per_sample",
    )
    raw_curated_output = write_curated_series_json(
        output_path=raw_output.with_suffix(".json"),
        title=f"prefill computed tokens: {run_dir.name}",
        y_axis_label="tokens_per_sample",
        series=raw_series,
    )
    avg5_curated_output = write_curated_series_json(
        output_path=avg5_output.with_suffix(".json"),
        title=f"prefill computed tokens avg5: {run_dir.name}",
        y_axis_label="avg_tokens_per_sample",
        series=avg5_series,
    )

    print(str(raw_output))
    print(str(avg5_output))
    print(str(raw_curated_output))
    print(str(avg5_curated_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
