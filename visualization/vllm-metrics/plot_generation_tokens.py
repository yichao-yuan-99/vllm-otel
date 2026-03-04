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


RAW_OUTPUT_NAME = "generation_tokens.raw.pdf"
AVG5_OUTPUT_NAME = "generation_tokens.avg5.pdf"


def extract_generation_tokens_series(payload: dict) -> list[dict]:
    return select_metric_series(payload, "vllm:generation_tokens")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generation token throughput from a post-processed run."
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
        help="Optional output directory. Defaults to <run-dir>/visualization/vllm-log/generation_tokens/",
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
        else (run_dir / "visualization" / "vllm-log" / "generation_tokens").resolve()
    )

    payload = load_timeseries_payload(input_path)
    raw_series = extract_generation_tokens_series(payload)
    avg5_series = build_grouped_average_series(raw_series, group_size=5)

    raw_output = output_dir / RAW_OUTPUT_NAME
    avg5_output = output_dir / AVG5_OUTPUT_NAME

    render_line_chart_pdf(
        series=raw_series,
        output_path=raw_output,
        title=f"generation_tokens throughput: {run_dir.name}",
        y_axis_label="tokens_per_sample",
    )
    render_line_chart_pdf(
        series=avg5_series,
        output_path=avg5_output,
        title=f"generation_tokens throughput avg5: {run_dir.name}",
        y_axis_label="avg_tokens_per_sample",
    )
    raw_curated_output = write_curated_series_json(
        output_path=raw_output.with_suffix(".json"),
        title=f"generation_tokens throughput: {run_dir.name}",
        y_axis_label="tokens_per_sample",
        series=raw_series,
    )
    avg5_curated_output = write_curated_series_json(
        output_path=avg5_output.with_suffix(".json"),
        title=f"generation_tokens throughput avg5: {run_dir.name}",
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
