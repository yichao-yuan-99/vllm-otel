from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import plot_generation_tokens
import plot_kv_cache_usage
import plot_num_preemptions
import plot_num_requests_running
import plot_num_requests_waiting
import plot_prefill_computed_tokens


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all vLLM metric figures for a post-processed run."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base_args = ["--run-dir", str(Path(args.run_dir).expanduser().resolve())]
    if args.input:
        base_args.extend(["--input", str(Path(args.input).expanduser().resolve())])

    exit_codes = [
        plot_kv_cache_usage.main(list(base_args)),
        plot_num_requests_running.main(list(base_args)),
        plot_num_requests_waiting.main(list(base_args)),
        plot_generation_tokens.main(list(base_args)),
        plot_num_preemptions.main(list(base_args)),
        plot_prefill_computed_tokens.main(list(base_args)),
    ]
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
