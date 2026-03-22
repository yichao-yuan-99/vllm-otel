#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

CONFIG_MAP = {
    ("swebench-verified", "mini-swe-agent"): "config.swebench-verified.mswe.toml",
    ("swebench-verified", "terminus-2"): "config.swebench-verfied.terminus2.toml",
    ("livecodebench", "mini-swe-agent"): "config.livecodebench.mswe.toml",
    ("livecodebench", "terminus-2"): "config.livecodebench.terminus2.toml",
    ("dabstep", "mini-swe-agent"): "config.dabstep.mswe.toml",
    ("dabstep", "terminus-2"): "config.dabstep.terminus2.toml",
    ("terminal-bench@2.0", "mini-swe-agent"): "config.terminal-bench-2.0.mswe.toml",
    ("terminal-bench@2.0", "terminus-2"): "config.terminal-bench-2.0.terminus2.toml",
    ("terminalbench@2.0", "mini-swe-agent"): "config.terminal-bench-2.0.mswe.toml",
    ("terminalbench@2.0", "terminus-2"): "config.terminal-bench-2.0.terminus2.toml",
}

VALID_BENCHMARKS = sorted({benchmark for benchmark, _agent in CONFIG_MAP})
VALID_AGENTS = sorted({_agent for _benchmark, _agent in CONFIG_MAP})


def _build_max_list(per_profile_conc: int, port_profile_id_list: str) -> str:
    ids = [item.strip() for item in port_profile_id_list.split(",") if item.strip()]
    if not ids:
        raise ValueError("empty --port-profile-id-list")
    return ",".join(str(per_profile_conc) for _ in ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MiniMax-M2.5 experiment (single benchmark + single agent)."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=VALID_BENCHMARKS,
        help="Benchmark to run.",
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=VALID_AGENTS,
        help="Agent to run.",
    )
    parser.add_argument(
        "--per-profile-conc",
        type=int,
        help=(
            "Positive integer used to build --max-concurrent-list. "
            "Required when --max-concurrent-list is not provided."
        ),
    )
    parser.add_argument(
        "--max-concurrent-list",
        type=str,
        help="Explicit CSV override, e.g. 5,5,5,5,5.",
    )
    parser.add_argument(
        "--port-profile-id-list",
        type=str,
        default="0,1,2,3,4",
        help="CSV profile IDs (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each con-driver invocation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.max_concurrent_list:
        max_concurrent_list = args.max_concurrent_list
    else:
        if args.per_profile_conc is None:
            print(
                "missing required --per-profile-conc (or provide --max-concurrent-list)",
                file=sys.stderr,
            )
            return 1
        if args.per_profile_conc <= 0:
            print("--per-profile-conc must be a positive integer", file=sys.stderr)
            return 1
        try:
            max_concurrent_list = _build_max_list(
                args.per_profile_conc, args.port_profile_id_list
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    config_name = CONFIG_MAP[(args.benchmark, args.agent)]
    config_path = SCRIPT_DIR / "configs" / config_name
    if not config_path.is_file():
        print(f"missing config: {config_path}", file=sys.stderr)
        return 1

    cmd = [
        "bash",
        str(REPO_ROOT / "con-driver" / "run_con_driver.sh"),
        "--config",
        str(config_path),
        "--port-profile-id-list",
        args.port_profile_id_list,
        "--max-concurrent-list",
        max_concurrent_list,
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"=== minimax-m2.5: {args.benchmark} / {args.agent} ===", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"port_profile_id_list: {args.port_profile_id_list}", flush=True)
    print(f"max_concurrent_list: {max_concurrent_list}", flush=True)
    print(f"cmd: {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
