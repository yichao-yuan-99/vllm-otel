#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate, build, and push the vLLM OTEL Docker image from a base image ref."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


SCRIPT_PATH = Path(__file__).resolve()
DOCKER_DIR = SCRIPT_PATH.parent
REPO_ROOT = DOCKER_DIR.parents[1]
FORCESEQ_DIR = DOCKER_DIR / "forceSeq"
ENTRYPOINT_PATH = DOCKER_DIR / "vllm_entrypoint.sh"

OTEL_REQUIREMENTS = [
    "'opentelemetry-sdk>=1.26.0,<1.27.0'",
    "'opentelemetry-api>=1.26.0,<1.27.0'",
    "'opentelemetry-exporter-otlp>=1.26.0,<1.27.0'",
    "'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0'",
]

DEFAULT_TARGET_NAMESPACE = "yichaoyuan"


@dataclass(frozen=True)
class BuildSpec:
    base_image: str
    gfx: str | None
    add_lmcache: bool
    target_image: str


def _default_target_tag(base_tag: str, *, rocm: bool, add_lmcache: bool) -> str:
    if rocm:
        tag = f"{base_tag}-otel-lp-rocm"
    else:
        tag = f"{base_tag}-otel-lp"
    if add_lmcache:
        return f"{tag}-lmcache"
    return tag


def _strip_image_tag_and_digest(image_ref: str) -> str:
    no_digest = image_ref.split("@", 1)[0]
    last_slash = no_digest.rfind("/")
    last_colon = no_digest.rfind(":")
    # Treat ":" as a tag separator only when it appears after the last "/".
    if last_colon > last_slash:
        return no_digest[:last_colon]
    return no_digest


def _extract_base_image_repo_path(base_image: str) -> str:
    normalized = base_image.strip()
    if not normalized:
        raise SystemExit("error: --base-image must be non-empty")

    path_without_tag = _strip_image_tag_and_digest(normalized)
    parts = [part for part in path_without_tag.split("/") if part]
    if not parts:
        raise SystemExit("error: could not derive repository path from --base-image")

    # Drop registry host component if present (e.g., ghcr.io, localhost:5000).
    if len(parts) >= 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        parts = parts[1:]
    if not parts:
        raise SystemExit("error: could not derive repository path from --base-image")

    return "/".join(parts)


def _repo_slug_from_base_image(base_image: str) -> str:
    repo_path = _extract_base_image_repo_path(base_image)
    slug = repo_path.lower().replace("/", "-")
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        raise SystemExit("error: derived target repository slug from --base-image is empty")
    return slug


def _default_target_repo(base_image: str, *, target_namespace: str) -> str:
    slug = _repo_slug_from_base_image(base_image)
    normalized_namespace = target_namespace.strip().strip("/")
    if not normalized_namespace:
        return slug
    return f"{normalized_namespace}/{slug}"


def _extract_base_image_tag(base_image: str) -> str:
    normalized = base_image.strip()
    if not normalized:
        raise SystemExit("error: --base-image must be non-empty")

    # Drop digest suffix first, then parse tag from the last path segment.
    no_digest = normalized.split("@", 1)[0]
    last_segment = no_digest.rsplit("/", 1)[-1]
    if ":" in last_segment:
        candidate = last_segment.rsplit(":", 1)[-1].strip()
        if candidate:
            return candidate
    return "latest"


def _is_rocm_base_image(base_image: str) -> bool:
    return "rocm" in base_image.lower()


def _render_dockerfile(base_image: str, *, add_lmcache: bool) -> str:
    requirements_list = list(OTEL_REQUIREMENTS)
    if add_lmcache:
        requirements_list.append("'lmcache'")
    requirements = " \\\n  ".join(requirements_list)

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
    base_image = str(args.base_image).strip()
    if not base_image:
        raise SystemExit("error: --base-image must be non-empty")
    base_image_tag = _extract_base_image_tag(base_image)
    rocm_mode = bool(args.rocm) or _is_rocm_base_image(base_image)
    add_lmcache = bool(args.add_lmcache)
    gfx = str(args.gfx).strip() if args.gfx else None
    if gfx is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", gfx):
        raise SystemExit("error: --gfx must match [A-Za-z0-9_.-]+ (for example gfx942)")

    if args.image:
        target_image = args.image
    else:
        target_repo = args.target_repo or _default_target_repo(
            base_image,
            target_namespace=args.target_namespace,
        )
        target_tag = args.target_tag or _default_target_tag(
            base_image_tag,
            rocm=rocm_mode,
            add_lmcache=add_lmcache,
        )
        target_image = f"{target_repo}:{target_tag}"

    return BuildSpec(
        base_image=base_image,
        gfx=gfx,
        add_lmcache=add_lmcache,
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
    dockerfile_content = _render_dockerfile(spec.base_image, add_lmcache=spec.add_lmcache)

    if args.dockerfile_output:
        dockerfile_path = Path(args.dockerfile_output).expanduser().resolve()
        _write_dockerfile(dockerfile_path, dockerfile_content)
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="vllm-otel-docker-")
        dockerfile_path = Path(temp_dir.name) / "Dockerfile"
        _write_dockerfile(dockerfile_path, dockerfile_content)

    print(f"base image: {spec.base_image}")
    if spec.gfx:
        print(
            "warning: --gfx is deprecated in servers/docker/build_image.py and no longer "
            "controls LMCache install in Docker images; use --add-lmcache to pip install LMCache "
            "or use servers/sif tooling to bake LMCache into SIF."
        )
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
    dockerfile_content = _render_dockerfile(spec.base_image, add_lmcache=spec.add_lmcache)

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
        description="Generate, build, and push the vLLM OTEL Docker image from a chosen base image."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--base-image",
            required=True,
            help="Full base image reference, for example vllm/vllm-openai:v0.14.0.",
        )
        command_parser.add_argument(
            "--rocm",
            action="store_true",
            help=(
                "Force ROCm-flavored default output naming when deriving target repo/tag. "
                "By default ROCm is auto-detected from --base-image."
            ),
        )
        command_parser.add_argument(
            "--target-repo",
            help=(
                "Override the default output repository. "
                "Defaults to <target-namespace>/<base-image-repo-slug> "
                "(for example yichaoyuan/rocm-vllm-dev)."
            ),
        )
        command_parser.add_argument(
            "--target-namespace",
            default=DEFAULT_TARGET_NAMESPACE,
            help=(
                "Namespace used by default target repo derivation "
                "(set empty string to disable namespace prefix). "
                f"Default: {DEFAULT_TARGET_NAMESPACE}."
            ),
        )
        command_parser.add_argument(
            "--target-tag",
            help=(
                "Override the default pushed image tag. "
                "Defaults to <base-image-tag>-otel-lp or <base-image-tag>-otel-lp-rocm, "
                "with -lmcache appended when --add-lmcache is set."
            ),
        )
        command_parser.add_argument(
            "--add-lmcache",
            action="store_true",
            help=(
                "Install the lmcache Python package in the generated image and append "
                "-lmcache to the default derived target tag."
            ),
        )
        command_parser.add_argument(
            "--gfx",
            help=(
                "Deprecated in docker image build flow. "
                "Use --add-lmcache for Docker builds, or servers/sif tooling to bake "
                "LMCache into the SIF."
            ),
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
