from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_VENV_DIR = REPO_ROOT / ".venv"
_BOOTSTRAP_ENV = "VLLM_OTEL_GATEWAY_MULTI_BOOTSTRAPPED"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _extract_option(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _bootstrap_if_needed(argv: list[str]) -> None:
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        return
    if importlib.util.find_spec("typer") is not None:
        return

    venv_arg = _extract_option(argv, "--venv-dir")
    venv_dir = Path(venv_arg).expanduser().resolve() if venv_arg else DEFAULT_VENV_DIR
    venv_python = _venv_python(venv_dir)
    skip_install = "--skip-install" in argv

    if not venv_python.exists():
        if skip_install:
            print(
                f"error: shared virtual environment not found: {venv_dir}",
                file=sys.stderr,
            )
            print(
                "hint: rerun without --skip-install so the gateway can bootstrap .venv",
                file=sys.stderr,
            )
            raise SystemExit(1)
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    if not skip_install:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(PACKAGE_ROOT)],
            check=True,
        )

    env = os.environ.copy()
    env[_BOOTSTRAP_ENV] = "1"
    os.execvpe(
        str(venv_python),
        [str(venv_python), "-m", "gateway_multi", *argv],
        env,
    )


def main() -> None:
    _bootstrap_if_needed(sys.argv[1:])
    from gateway_multi.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
