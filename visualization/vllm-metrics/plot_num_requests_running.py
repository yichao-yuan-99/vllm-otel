from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    load_timeseries_payload,
    render_line_chart_pdf,
    select_metric_series,
    write_curated_series_json,
)


DEFAULT_OUTPUT_NAME = "num_requests_running.pdf"


def extract_num_requests_running_series(payload: dict) -> list[dict]:
    return select_metric_series(payload, "vllm:num_requests_running")


def render_num_requests_running_pdf(
    *,
    series: list[dict],
    output_path: Path,
    title: str,
) -> None:
    render_line_chart_pdf(
        series=series,
        output_path=output_path,
        title=title,
        y_axis_label="num_requests_running",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vllm:num_requests_running from a post-processed run into PDF."
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
        "--output",
        default=None,
        help="Optional output PDF path. Defaults to <run-dir>/visualization/vllm-log/num_requests_running/num_requests_running.pdf",
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
    payload = load_timeseries_payload(input_path)
    series = extract_num_requests_running_series(payload)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            run_dir
            / "visualization"
            / "vllm-log"
            / "num_requests_running"
            / DEFAULT_OUTPUT_NAME
        ).resolve()
    )
    render_num_requests_running_pdf(
        series=series,
        output_path=output_path,
        title=f"num_requests_running: {run_dir.name}",
    )
    curated_output = write_curated_series_json(
        output_path=output_path.with_suffix(".json"),
        title=f"num_requests_running: {run_dir.name}",
        y_axis_label="num_requests_running",
        series=series,
    )
    print(str(output_path))
    print(str(curated_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
