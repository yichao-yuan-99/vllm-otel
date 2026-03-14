#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a vLLM SIF with LMCache baked in at image-build time."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEF_TEMPLATE_PATH = SCRIPT_DIR / "lmcache-rocm.def.in"

DEFAULT_LMCACHE_REPO_URL = "https://github.com/LMCache/LMCache.git"
DEFAULT_LMCACHE_REPO_TAG = "v0.4.1"


@dataclass(frozen=True)
class BuildSpec:
    base_sif: Path | None
    base_image: str | None
    output_sif: Path
    rocm_arch: str
    lmcache_repo_url: str
    lmcache_repo_tag: str
    use_fakeroot: bool
    force: bool
    work_dir: Path | None
    keep_work_dir: bool


def _require_command(name: str) -> None:
    if shutil.which(name):
        return
    raise SystemExit(f"error: required command not found: {name}")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _normalize_base_image(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise SystemExit("error: --base-image must be non-empty")
    if "://" not in normalized:
        return f"docker://{normalized}"
    return normalized


def _resolve_spec(args: argparse.Namespace) -> BuildSpec:
    base_sif = Path(args.base_sif).expanduser().resolve() if args.base_sif else None
    base_image = _normalize_base_image(str(args.base_image)) if args.base_image else None
    if (base_sif is None and base_image is None) or (base_sif is not None and base_image is not None):
        raise SystemExit("error: provide exactly one of --base-sif or --base-image")
    if base_sif is not None and not base_sif.exists():
        raise SystemExit(f"error: --base-sif does not exist: {base_sif}")

    output_sif = Path(args.output_sif).expanduser().resolve()
    rocm_arch = str(args.gfx).strip()
    if not rocm_arch:
        raise SystemExit("error: --gfx must be non-empty")

    lmcache_repo_url = str(args.lmcache_repo_url).strip()
    lmcache_repo_tag = str(args.lmcache_repo_tag).strip()
    if not lmcache_repo_url:
        raise SystemExit("error: --lmcache-repo-url must be non-empty")
    if not lmcache_repo_tag:
        raise SystemExit("error: --lmcache-repo-tag must be non-empty")

    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else None
    return BuildSpec(
        base_sif=base_sif,
        base_image=base_image,
        output_sif=output_sif,
        rocm_arch=rocm_arch,
        lmcache_repo_url=lmcache_repo_url,
        lmcache_repo_tag=lmcache_repo_tag,
        use_fakeroot=not bool(args.no_fakeroot),
        force=bool(args.force),
        work_dir=work_dir,
        keep_work_dir=bool(args.keep_work_dir),
    )


def _clone_lmcache(*, repo_url: str, repo_tag: str, dest_dir: Path) -> None:
    _run(["git", "clone", repo_url, str(dest_dir)])
    _run(["git", "checkout", f"tags/{repo_tag}"], cwd=dest_dir)


def _render_def_file(
    *,
    base_sif: Path,
    lmcache_src_dir: Path,
    rocm_arch: str,
    lmcache_repo_url: str,
    lmcache_repo_tag: str,
    output_path: Path,
) -> None:
    template = DEF_TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = (
        template.replace("__BASE_SIF__", str(base_sif))
        .replace("__LMCACHE_SRC_DIR__", str(lmcache_src_dir))
        .replace("__ROCM_ARCH__", rocm_arch)
        .replace("__LMCACHE_REPO_URL__", lmcache_repo_url)
        .replace("__LMCACHE_REPO_TAG__", lmcache_repo_tag)
    )
    output_path.write_text(rendered, encoding="utf-8")


def _infer_pulled_base_name(base_image: str) -> str:
    normalized = base_image.split("@", 1)[0]
    tail = normalized.rsplit("/", 1)[-1]
    token = tail.replace(":", "-")
    if not token.endswith(".sif"):
        token += ".sif"
    return token


def _build(spec: BuildSpec) -> int:
    _require_command("apptainer")
    _require_command("git")
    if not DEF_TEMPLATE_PATH.exists():
        raise SystemExit(f"error: definition template not found: {DEF_TEMPLATE_PATH}")

    work_dir_cm: tempfile.TemporaryDirectory[str] | None = None
    if spec.work_dir is None:
        work_dir_cm = tempfile.TemporaryDirectory(prefix="vllm-sif-lmcache-")
        work_dir = Path(work_dir_cm.name).resolve()
    else:
        work_dir = spec.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if spec.base_sif is not None:
            base_sif = spec.base_sif
        else:
            assert spec.base_image is not None
            base_sif = work_dir / _infer_pulled_base_name(spec.base_image)
            _run(["apptainer", "pull", str(base_sif), spec.base_image])

        lmcache_src_dir = work_dir / "LMCache"
        _clone_lmcache(
            repo_url=spec.lmcache_repo_url,
            repo_tag=spec.lmcache_repo_tag,
            dest_dir=lmcache_src_dir,
        )

        def_file = work_dir / "build-lmcache.def"
        _render_def_file(
            base_sif=base_sif,
            lmcache_src_dir=lmcache_src_dir,
            rocm_arch=spec.rocm_arch,
            lmcache_repo_url=spec.lmcache_repo_url,
            lmcache_repo_tag=spec.lmcache_repo_tag,
            output_path=def_file,
        )

        spec.output_sif.parent.mkdir(parents=True, exist_ok=True)
        if spec.output_sif.exists():
            if not spec.force:
                raise SystemExit(
                    f"error: output already exists: {spec.output_sif} (use --force to overwrite)"
                )
            spec.output_sif.unlink()

        cmd = ["apptainer", "build"]
        if spec.use_fakeroot:
            cmd.append("--fakeroot")
        cmd.extend([str(spec.output_sif), str(def_file)])
        _run(cmd)

        print(f"base_sif: {base_sif}")
        print(f"output_sif: {spec.output_sif}")
        print(f"rocm_arch: {spec.rocm_arch}")
        print(f"lmcache_repo: {spec.lmcache_repo_url}")
        print(f"lmcache_tag: {spec.lmcache_repo_tag}")
        if spec.keep_work_dir or spec.work_dir is not None:
            print(f"work_dir: {work_dir}")
        return 0
    finally:
        if work_dir_cm is not None and not spec.keep_work_dir:
            work_dir_cm.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SIF from either --base-sif or --base-image and bake LMCache into it "
            "during apptainer build."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-sif", help="Existing base SIF path.")
    source.add_argument(
        "--base-image",
        help=(
            "Base OCI image ref used for apptainer pull, e.g. "
            "docker://yichaoyuan/vllm-vllm-openai-rocm:v0.17.1-otel-lp-rocm"
        ),
    )
    parser.add_argument("--output-sif", required=True, help="Output SIF path.")
    parser.add_argument("--gfx", required=True, help="ROCm arch, e.g. gfx942.")
    parser.add_argument(
        "--lmcache-repo-url",
        default=DEFAULT_LMCACHE_REPO_URL,
        help=f"LMCache git repo URL (default: {DEFAULT_LMCACHE_REPO_URL}).",
    )
    parser.add_argument(
        "--lmcache-repo-tag",
        default=DEFAULT_LMCACHE_REPO_TAG,
        help=f"LMCache git tag (default: {DEFAULT_LMCACHE_REPO_TAG}).",
    )
    parser.add_argument("--no-fakeroot", action="store_true", help="Do not pass --fakeroot to apptainer build.")
    parser.add_argument("--force", action="store_true", help="Overwrite --output-sif if it already exists.")
    parser.add_argument("--work-dir", help="Optional work directory to keep intermediate files.")
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep temporary work directory when --work-dir is not provided.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    spec = _resolve_spec(args)
    try:
        return _build(spec)
    except subprocess.CalledProcessError as exc:
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
