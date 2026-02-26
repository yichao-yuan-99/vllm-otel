#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pull required Apptainer images (Jaeger + vLLM) into SIF files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Final

try:
    from .control_plane import ControlPlaneError, RuntimeConfig, load_runtime_config
except ImportError:  # pragma: no cover
    from control_plane import (  # type: ignore[no-redef]
        ControlPlaneError,
        RuntimeConfig,
        load_runtime_config,
    )


JAEGER_PULL_TIMEOUT_SECONDS: Final[int] = 60 * 60
VLLM_PULL_TIMEOUT_SECONDS: Final[int] = 2 * 60 * 60


def _run_pull_command(command: list[str], *, timeout_seconds: int) -> None:
    print(f"[pull] running: {shlex.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out after {timeout_seconds}s: {shlex.join(command)}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "\n".join(
                [
                    f"command failed (code={result.returncode}): {shlex.join(command)}",
                    f"stdout:\n{result.stdout.strip() or '(empty)'}",
                    f"stderr:\n{result.stderr.strip() or '(empty)'}",
                ]
            )
        )


def _pull_image(
    *,
    name: str,
    image: str,
    sif_path: Path,
    timeout_seconds: int,
    force: bool,
) -> str:
    if sif_path.exists():
        if not force:
            return f"skip existing {name} SIF: {sif_path}"
        print(f"[pull] removing existing {name} SIF: {sif_path}")
        sif_path.unlink()

    sif_path.parent.mkdir(parents=True, exist_ok=True)
    _run_pull_command(
        ["apptainer", "pull", str(sif_path), image],
        timeout_seconds=timeout_seconds,
    )
    return f"pulled {name}: {image} -> {sif_path}"


def _pull_all_images(config: RuntimeConfig, *, force: bool) -> list[str]:
    if shutil.which("apptainer") is None:
        raise RuntimeError("required command not found: apptainer")

    config.apptainer_imgs.mkdir(parents=True, exist_ok=True)

    actions: list[str] = []
    actions.append(
        _pull_image(
            name="jaeger",
            image=config.jaeger_image,
            sif_path=config.jaeger_sif,
            timeout_seconds=JAEGER_PULL_TIMEOUT_SECONDS,
            force=force,
        )
    )
    actions.append(
        _pull_image(
            name="vllm",
            image=config.vllm_image,
            sif_path=config.vllm_sif,
            timeout_seconds=VLLM_PULL_TIMEOUT_SECONDS,
            force=force,
        )
    )
    return actions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull required Apptainer images for control-plane startup")
    default_config = Path(__file__).resolve().parent / "server_config.toml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"Path to server config TOML (default: {default_config})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull images even if target SIF files already exist.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        config = load_runtime_config(args.config.resolve())
    except ControlPlaneError as exc:
        print(f"failed to load runtime config: code={exc.code} message={exc.message}")
        if exc.details:
            print(json.dumps(exc.details, indent=2, sort_keys=True))
        return 2

    try:
        actions = _pull_all_images(config, force=args.force)
    except RuntimeError as exc:
        print(f"pull failed: {exc}")
        return 1

    print("pull complete")
    print(
        json.dumps(
            {
                "actions": actions,
                "jaeger_sif": str(config.jaeger_sif),
                "vllm_sif": str(config.vllm_sif),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
