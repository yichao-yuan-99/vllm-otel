#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI for embedded TP=1 AMD HPC sbatch rendering/submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embedded_tp1 import (
    DEFAULT_CONFIG_PATH,
    EmbeddedTp1Launcher,
    parse_extra_env_list,
    parse_extra_vllm_args,
)
from control_plane import ControlPlaneError, error_payload  # type: ignore[import-not-found]


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_error(exc: ControlPlaneError) -> None:
    _print_json(error_payload(exc))
    raise SystemExit(1)


def _resolve_inputs(
    *,
    env: list[str],
    extra_vllm_args: str | None,
    lmcache: int | None,
) -> tuple[dict[str, str], list[str], str | None]:
    try:
        extra_env = parse_extra_env_list(env)
    except ValueError as exc:
        raise SystemExit(f"--env: {exc}") from exc

    try:
        parsed_extra_vllm_args = parse_extra_vllm_args(extra_vllm_args)
    except ValueError as exc:
        raise SystemExit(f"--extra-vllm-args: {exc}") from exc

    if lmcache is not None and lmcache <= 0:
        raise SystemExit("--lmcache must be a positive integer")
    if lmcache is not None and "LMCACHE_MAX_LOCAL_CPU_SIZE" in extra_env:
        raise SystemExit(
            "cannot combine --lmcache with --env LMCACHE_MAX_LOCAL_CPU_SIZE=..."
        )

    return extra_env, parsed_extra_vllm_args, str(lmcache) if lmcache is not None else None


def _handle_render(args: argparse.Namespace) -> None:
    extra_env, parsed_extra_vllm_args, lmcache_value = _resolve_inputs(
        env=args.env,
        extra_vllm_args=args.extra_vllm_args,
        lmcache=args.lmcache,
    )
    launcher = EmbeddedTp1Launcher(Path(args.config))
    try:
        payload = launcher.render(
            partition=args.partition,
            model=args.model,
            experiment_script=Path(args.experiment_script),
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_value,
            extra_vllm_args=parsed_extra_vllm_args,
            no_async_scheduling=bool(args.no_async_scheduling),
        )
    except ControlPlaneError as exc:
        _emit_error(exc)

    _print_json(
        {
            "ok": True,
            "code": 0,
            "message": f"rendered embedded TP1 sbatch for partition {args.partition}",
            "data": payload,
        }
    )


def _handle_submit(args: argparse.Namespace) -> None:
    extra_env, parsed_extra_vllm_args, lmcache_value = _resolve_inputs(
        env=args.env,
        extra_vllm_args=args.extra_vllm_args,
        lmcache=args.lmcache,
    )
    launcher = EmbeddedTp1Launcher(Path(args.config))
    try:
        payload = launcher.submit(
            partition=args.partition,
            model=args.model,
            experiment_script=Path(args.experiment_script),
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_value,
            extra_vllm_args=parsed_extra_vllm_args,
            no_async_scheduling=bool(args.no_async_scheduling),
        )
    except ControlPlaneError as exc:
        _emit_error(exc)

    _print_json(
        {
            "ok": True,
            "code": 0,
            "message": f"submitted embedded TP1 sbatch for partition {args.partition}",
            "data": payload,
        }
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to embedded TP1 config TOML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--partition",
        "-p",
        required=True,
        help="Partition key: mi3001x or mi3008x.",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model key from configs/model_config.toml.",
    )
    parser.add_argument(
        "--experiment-script",
        "-e",
        required=True,
        help="Experiment script path. It will be invoked once per port profile.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional vLLM environment variable in KEY=VALUE form. Repeat to pass multiple values.",
    )
    parser.add_argument(
        "--lmcache",
        type=int,
        default=None,
        help=(
            "Enable LMCache with a maximum local CPU size. "
            "Sets LMCACHE_MAX_LOCAL_CPU_SIZE and enables kv-transfer-config."
        ),
    )
    parser.add_argument(
        "--extra-vllm-args",
        default=None,
        help="Additional vLLM CLI args string appended to model defaults.",
    )
    parser.add_argument(
        "--no-async-scheduling",
        action="store_true",
        help="Append --no-async-scheduling to the rendered vLLM command.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render or submit embedded TP=1 Slurm jobs for mi3001x/mi3008x. "
            "The rendered sbatch starts the service stack directly on the node and "
            "runs the provided experiment script with the port profile as its only argument."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render the sbatch script without submitting it.")
    _add_common_arguments(render_parser)
    render_parser.set_defaults(handler=_handle_render)

    submit_parser = subparsers.add_parser("submit", help="Render the sbatch script and submit it with sbatch.")
    _add_common_arguments(submit_parser)
    submit_parser.set_defaults(handler=_handle_submit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
