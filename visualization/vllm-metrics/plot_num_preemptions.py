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


RAW_OUTPUT_NAME = "num_preemptions.raw.pdf"
AVG5_OUTPUT_NAME = "num_preemptions.avg5.pdf"


def extract_num_preemptions_series(payload: dict) -> list[dict]:
    return select_metric_series(payload, "vllm:num_preemptions")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vLLM preemption counts from a post-processed run."
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
        help="Optional output directory. Defaults to <run-dir>/visualization/vllm-log/num_preemptions/",
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
        else (run_dir / "visualization" / "vllm-log" / "num_preemptions").resolve()
    )

    payload = load_timeseries_payload(input_path)
    raw_series = extract_num_preemptions_series(payload)
    avg5_series = build_grouped_average_series(raw_series, group_size=5)

    raw_output = output_dir / RAW_OUTPUT_NAME
    avg5_output = output_dir / AVG5_OUTPUT_NAME

    render_line_chart_pdf(
        series=raw_series,
        output_path=raw_output,
        title=f"num_preemptions per sample: {run_dir.name}",
        y_axis_label="preemptions_per_sample",
    )
    render_line_chart_pdf(
        series=avg5_series,
        output_path=avg5_output,
        title=f"num_preemptions avg5: {run_dir.name}",
        y_axis_label="avg_preemptions_per_sample",
    )
    raw_curated_output = write_curated_series_json(
        output_path=raw_output.with_suffix(".json"),
        title=f"num_preemptions per sample: {run_dir.name}",
        y_axis_label="preemptions_per_sample",
        series=raw_series,
    )
    avg5_curated_output = write_curated_series_json(
        output_path=avg5_output.with_suffix(".json"),
        title=f"num_preemptions avg5: {run_dir.name}",
        y_axis_label="avg_preemptions_per_sample",
        series=avg5_series,
    )

    print(str(raw_output))
    print(str(avg5_output))
    print(str(raw_curated_output))
    print(str(avg5_curated_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
