#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate, build, and push the vLLM OTEL Docker image from a selected base tag."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SCRIPT_PATH = Path(__file__).resolve()
DOCKER_DIR = SCRIPT_PATH.parent
REPO_ROOT = DOCKER_DIR.parents[1]
FORCESEQ_DIR = DOCKER_DIR / "forceSeq"
ENTRYPOINT_PATH = DOCKER_DIR / "vllm_entrypoint.sh"

DEFAULT_STANDARD_BASE_REPO = "vllm/vllm-openai"
DEFAULT_ROCM_BASE_REPO = "vllm/vllm-openai-rocm"
DEFAULT_STANDARD_TARGET_REPO = "yichaoyuan/vllm-openai-otel-lp"
DEFAULT_ROCM_TARGET_REPO = "yichaoyuan/vllm-openai-otel"

OTEL_REQUIREMENTS = [
    "'opentelemetry-sdk>=1.26.0,<1.27.0'",
    "'opentelemetry-api>=1.26.0,<1.27.0'",
    "'opentelemetry-exporter-otlp>=1.26.0,<1.27.0'",
    "'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0'",
]


@dataclass(frozen=True)
class BuildSpec:
    base_tag: str
    rocm: bool
    base_image: str
    target_image: str


def _default_target_tag(base_tag: str, *, rocm: bool) -> str:
    if rocm:
        return f"{base_tag}-otel-lp-rocm"
    return f"{base_tag}-otel-lp"


def _default_target_repo(*, rocm: bool) -> str:
    if rocm:
        return DEFAULT_ROCM_TARGET_REPO
    return DEFAULT_STANDARD_TARGET_REPO


def _render_dockerfile(base_image: str) -> str:
    requirements = " \\\n  ".join(OTEL_REQUIREMENTS)
    return (
        f"FROM {base_image}\n\n"
        "RUN python3 -m pip install --no-cache-dir \\\n"
        f"  {requirements}\n\n"
        "COPY servers/docker/forceSeq /opt/vllm-plugins/forceSeq\n"
        "COPY servers/docker/vllm_entrypoint.sh /opt/vllm-plugins/vllm_entrypoint.sh\n\n"
        "RUN chmod +x /opt/vllm-plugins/vllm_entrypoint.sh\n\n"
        "ENV PYTHONPATH=/opt/vllm-plugins\n"
    )


def _write_dockerfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_command(args: list[str], *, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _resolve_spec(args: argparse.Namespace) -> BuildSpec:
    base_repo = DEFAULT_ROCM_BASE_REPO if args.rocm else DEFAULT_STANDARD_BASE_REPO
    base_image = f"{base_repo}:{args.base_tag}"

    if args.image:
        target_image = args.image
    else:
        target_repo = args.target_repo or _default_target_repo(rocm=args.rocm)
        target_tag = args.target_tag or _default_target_tag(args.base_tag, rocm=args.rocm)
        target_image = f"{target_repo}:{target_tag}"

    return BuildSpec(
        base_tag=args.base_tag,
        rocm=bool(args.rocm),
        base_image=base_image,
        target_image=target_image,
    )


def _validate_repo_layout() -> None:
    missing: list[str] = []
    if not FORCESEQ_DIR.is_dir():
        missing.append(str(FORCESEQ_DIR))
    if not ENTRYPOINT_PATH.is_file():
        missing.append(str(ENTRYPOINT_PATH))
    if missing:
        raise SystemExit(f"error: required docker assets are missing: {', '.join(missing)}")


def _build_push(args: argparse.Namespace) -> int:
    _validate_repo_layout()
    _require_command("docker")

    spec = _resolve_spec(args)
    dockerfile_content = _render_dockerfile(spec.base_image)

    if args.dockerfile_output:
        dockerfile_path = Path(args.dockerfile_output).expanduser().resolve()
        _write_dockerfile(dockerfile_path, dockerfile_content)
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="vllm-otel-docker-")
        dockerfile_path = Path(temp_dir.name) / "Dockerfile"
        _write_dockerfile(dockerfile_path, dockerfile_content)

    print(f"base image: {spec.base_image}")
    print(f"target image: {spec.target_image}")
    print(f"dockerfile: {dockerfile_path}")

    try:
        _run_command(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                spec.target_image,
                ".",
            ],
            cwd=REPO_ROOT,
        )
        _run_command(["docker", "push", spec.target_image], cwd=REPO_ROOT)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"pushed {spec.target_image}")
    return 0


def _render(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args)
    dockerfile_content = _render_dockerfile(spec.base_image)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        _write_dockerfile(output_path, dockerfile_content)
        print(output_path)
        return 0

    sys.stdout.write(dockerfile_content)
    return 0


def _require_command(name: str) -> None:
    if shutil.which(name):
        return
    raise SystemExit(f"error: required command not found: {name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, build, and push the vLLM OTEL Docker image from a chosen base tag."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--base-tag", required=True, help="Base vLLM image tag, for example v0.14.0.")
        command_parser.add_argument(
            "--rocm",
            action="store_true",
            help="Use vllm/vllm-openai-rocm:<tag> as the base image instead of vllm/vllm-openai:<tag>.",
        )
        command_parser.add_argument(
            "--target-repo",
            help=(
                "Override the default output repository. "
                f"Defaults to {DEFAULT_STANDARD_TARGET_REPO} for CUDA and {DEFAULT_ROCM_TARGET_REPO} for ROCm."
            ),
        )
        command_parser.add_argument(
            "--target-tag",
            help="Override the default pushed image tag. Defaults to <base-tag>-otel-lp or <base-tag>-otel-lp-rocm.",
        )
        command_parser.add_argument(
            "--image",
            help="Override the full pushed image reference, for example repo/name:tag.",
        )

    render_parser = subparsers.add_parser("render", help="Render the generated Dockerfile.")
    add_common_flags(render_parser)
    render_parser.add_argument("--output", help="Write the rendered Dockerfile to this path instead of stdout.")
    render_parser.set_defaults(func=_render)

    build_push_parser = subparsers.add_parser(
        "build-push",
        help="Render the Dockerfile, then run docker build and docker push.",
    )
    add_common_flags(build_push_parser)
    build_push_parser.add_argument(
        "--dockerfile-output",
        help="Optional path to keep the rendered Dockerfile instead of using a temporary file.",
    )
    build_push_parser.set_defaults(func=_build_push)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as exc:
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
