#!/usr/bin/env python3
"""Render or submit a dedicated mi2104x embedded TP=1 Slurm job."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import textwrap
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parents[1]
START_SCRIPT = MODULE_ROOT / "start-services.sh"
RUN_DIR = MODULE_ROOT / "run"
LOG_DIR = MODULE_ROOT / "logs"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
DEFAULT_AMD_SMI_POWER_DAEMON_BIN = REPO_ROOT / ".venv" / "bin" / "amd-smi-power-daemon"
DEFAULT_AMD_POWER_READER_BIN = REPO_ROOT / ".venv" / "bin" / "amd-power-reader"
DEFAULT_FREQ_CONTROLLER_BIN = REPO_ROOT / ".venv" / "bin" / "freq-controller-linespace-amd"
DEFAULT_RESET_GPU_CORE_FREQ_BIN = REPO_ROOT / ".venv" / "bin" / "amd-reset-gpu-core-freq"

PARTITION = "mi2104x"
MAX_TIME = "24:00:00"
JOB_NAME_PREFIX = "vllm_embedded_tp1_"
GPU_MEMORY_GB = 64.0
GPUS_PER_SERVICE = 1
MAX_ALLOWED_WEIGHT_VRAM_GB = GPU_MEMORY_GB * GPUS_PER_SERVICE * 0.75

PROFILE_LAYOUTS = [
    {
        "profile_id": 0,
        "gpu_ids": "0",
        "vllm_port": 11451,
        "gateway_port": 11457,
        "gateway_parse_port": 18171,
        "lmcache_port": 29411,
    },
    {
        "profile_id": 1,
        "gpu_ids": "1",
        "vllm_port": 24123,
        "gateway_port": 24157,
        "gateway_parse_port": 28171,
        "lmcache_port": 29437,
    },
    {
        "profile_id": 2,
        "gpu_ids": "2",
        "vllm_port": 31987,
        "gateway_port": 31955,
        "gateway_parse_port": 38171,
        "lmcache_port": 29459,
    },
    {
        "profile_id": 3,
        "gpu_ids": "3",
        "vllm_port": 40823,
        "gateway_port": 40857,
        "gateway_parse_port": 48171,
        "lmcache_port": 29483,
    },
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    vllm_model_name: str
    served_model_name: str
    weight_vram_gb: float
    extra_args: list[str]


class LaunchError(RuntimeError):
    def __init__(
        self,
        *,
        message: str,
        code: int,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_error(exc: LaunchError) -> None:
    _print_json(
        {
            "ok": False,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        }
    )
    raise SystemExit(1)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    token = token.strip(".-_")
    return token or "value"


def parse_extra_env_list(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_item in values:
        item = raw_item.strip()
        if not item:
            raise ValueError("env value cannot be empty; use KEY=VALUE")
        if "=" not in item:
            raise ValueError(f"env value '{item}' must be in KEY=VALUE format")
        key_raw, value = item.split("=", 1)
        key = key_raw.strip()
        if not key:
            raise ValueError(f"env value '{item}' has an empty key")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(
                f"env key '{key}' is invalid; expected [A-Za-z_][A-Za-z0-9_]*"
            )
        if key in parsed:
            raise ValueError(f"duplicate env key '{key}'")
        parsed[key] = value
    return dict(sorted(parsed.items()))


def parse_extra_vllm_args(value: str | None) -> list[str]:
    if value is None:
        return []
    text = value.strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"invalid extra-vllm-args: {exc}") from exc


def _normalize_tp1_extra_args(extra_args: list[str]) -> list[str]:
    normalized_args: list[str] = []
    skip_next = False

    for index, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue

        normalized = arg.strip().lower()
        if normalized == "--trust-remote-code":
            continue
        if normalized.startswith("--trust-remote-code="):
            continue
        if normalized == "--distributed_executor_backend":
            if index + 1 < len(extra_args):
                skip_next = True
            continue
        if normalized.startswith("--distributed_executor_backend="):
            continue

        normalized_args.append(arg)

    normalized_args.append("--trust-remote-code")
    return normalized_args


def _encode_b64_json(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def load_model_specs(config_path: Path) -> dict[str, ModelSpec]:
    if not config_path.exists():
        raise ValueError(f"missing model config file: {config_path}")

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError(f"model config must define [models.<name>] tables: {config_path}")

    models: dict[str, ModelSpec] = {}
    for key, raw_spec in raw_models.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"models.{key} must be a table in {config_path}")

        vllm_model_name = raw_spec.get("vllm_model_name")
        served_model_name = raw_spec.get("served_model_name")
        weight_vram_gb = raw_spec.get("weight_vram_gb")
        extra_args = raw_spec.get("extra_args", [])
        if not isinstance(vllm_model_name, str) or not vllm_model_name.strip():
            raise ValueError(f"models.{key}.vllm_model_name must be a non-empty string")
        if not isinstance(served_model_name, str) or not served_model_name.strip():
            raise ValueError(f"models.{key}.served_model_name must be a non-empty string")
        if isinstance(weight_vram_gb, bool) or not isinstance(weight_vram_gb, (int, float)):
            raise ValueError(f"models.{key}.weight_vram_gb must be numeric")
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError(f"models.{key}.extra_args must be a string array")

        models[key] = ModelSpec(
            key=key,
            vllm_model_name=vllm_model_name.strip(),
            served_model_name=served_model_name.strip(),
            weight_vram_gb=float(weight_vram_gb),
            extra_args=list(extra_args),
        )

    return models


def resolve_experiment_script(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"experiment script not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"experiment script must be a file: {resolved}")
    return resolved


def _resolve_extra_env(
    *,
    extra_env: dict[str, str],
    lmcache_max_local_cpu_size: str | None,
) -> tuple[dict[str, str], bool]:
    normalized = dict(extra_env)
    if lmcache_max_local_cpu_size is None:
        return normalized, False

    value = lmcache_max_local_cpu_size.strip()
    if not value:
        raise ValueError("lmcache size must be non-empty")
    existing = normalized.get("LMCACHE_MAX_LOCAL_CPU_SIZE")
    if existing is not None and existing != value:
        raise ValueError(
            "cannot combine --lmcache with --env LMCACHE_MAX_LOCAL_CPU_SIZE=..."
        )
    normalized["LMCACHE_MAX_LOCAL_CPU_SIZE"] = value
    return dict(sorted(normalized.items())), True


def _resolve_time_limit(time_limit: str | None, *, default: str) -> str:
    if time_limit is None:
        return default
    normalized = time_limit.strip()
    if not normalized:
        raise ValueError("time_limit must be non-empty")
    if "\x00" in normalized:
        raise ValueError("time_limit cannot contain NUL byte")
    return normalized


@dataclass
class Mi2104xEmbeddedTp1Launcher:
    model_config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    start_script: Path = START_SCRIPT
    run_dir: Path = RUN_DIR
    log_dir: Path = LOG_DIR

    def __post_init__(self) -> None:
        self.model_config_path = self.model_config_path.expanduser().resolve()
        self.start_script = self.start_script.expanduser().resolve()
        self.run_dir = self.run_dir.expanduser().resolve()
        self.log_dir = self.log_dir.expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        *,
        model: str,
        experiment_script: Path,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        no_async_scheduling: bool = False,
        time_limit: str | None = None,
    ) -> dict[str, Any]:
        if not self.start_script.exists():
            raise LaunchError(
                message=f"required start script not found: {self.start_script}",
                code=301,
            )

        try:
            resolved_experiment_script = resolve_experiment_script(experiment_script)
            model_specs = load_model_specs(self.model_config_path)
        except ValueError as exc:
            raise LaunchError(message=str(exc), code=302) from exc

        model_spec = model_specs.get(model)
        if model_spec is None:
            raise LaunchError(
                message=f"unknown model '{model}'",
                code=303,
                details={"allowed_models": sorted(model_specs)},
            )

        if model_spec.weight_vram_gb > MAX_ALLOWED_WEIGHT_VRAM_GB:
            raise LaunchError(
                message=(
                    f"model '{model}' requires {model_spec.weight_vram_gb:.1f} GB which exceeds "
                    f"75% of single-GPU VRAM on partition '{PARTITION}' "
                    f"({MAX_ALLOWED_WEIGHT_VRAM_GB:.1f} GB)"
                ),
                code=304,
                details={
                    "model_weight_vram_gb": model_spec.weight_vram_gb,
                    "max_allowed_weight_vram_gb": MAX_ALLOWED_WEIGHT_VRAM_GB,
                },
            )

        normalized_extra_env = dict(sorted((extra_env or {}).items()))
        try:
            normalized_extra_env, lmcache_enabled = _resolve_extra_env(
                extra_env=normalized_extra_env,
                lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            )
            effective_time_limit = _resolve_time_limit(time_limit, default=MAX_TIME)
        except ValueError as exc:
            raise LaunchError(message=str(exc), code=305) from exc

        normalized_extra_vllm_args = list(extra_vllm_args or [])
        effective_model_extra_args = _normalize_tp1_extra_args(
            [*model_spec.extra_args, *normalized_extra_vllm_args]
        )
        if lmcache_enabled:
            effective_model_extra_args.extend(
                [
                    "--kv-transfer-config",
                    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
                ]
            )
        if no_async_scheduling:
            effective_model_extra_args.append("--no-async-scheduling")

        encoded_model_extra_args = _encode_b64_json(effective_model_extra_args)
        encoded_extra_env = _encode_b64_json(normalized_extra_env) if normalized_extra_env else ""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        script_path = self.run_dir / (
            f"sbatch-{timestamp}-{PARTITION}-{_safe_token(model_spec.key)}-embedded-tp1.sh"
        )
        script_text = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(f"{JOB_NAME_PREFIX}{PARTITION}_tp1")}
            #SBATCH --output={self.log_dir / "slurm.%j.out"}
            #SBATCH --error={self.log_dir / "slurm.%j.err"}
            #SBATCH --nodes=1
            #SBATCH --time={effective_time_limit}
            #SBATCH --partition={PARTITION}

            set -euo pipefail

            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            cd "${{REPO_ROOT}}"

            export MODEL_CONFIG_PATH={shlex.quote(str(self.model_config_path))}
            export EXPERIMENT_SCRIPT={shlex.quote(str(resolved_experiment_script))}
            export VLLM_MODEL_KEY={shlex.quote(model_spec.key)}
            export VLLM_MODEL_EXTRA_ARGS_B64={shlex.quote(encoded_model_extra_args)}
            export VLLM_EXTRA_ENV_B64={shlex.quote(encoded_extra_env)}
            export AMD_SMI_POWER_DAEMON_BIN="${{AMD_SMI_POWER_DAEMON_BIN:-{shlex.quote(str(DEFAULT_AMD_SMI_POWER_DAEMON_BIN))}}}"
            export AMD_POWER_READER_BIN="${{AMD_POWER_READER_BIN:-{shlex.quote(str(DEFAULT_AMD_POWER_READER_BIN))}}}"
            export FREQ_CONTROLLER_BIN="${{FREQ_CONTROLLER_BIN:-{shlex.quote(str(DEFAULT_FREQ_CONTROLLER_BIN))}}}"
            export RESET_GPU_CORE_FREQ_BIN="${{RESET_GPU_CORE_FREQ_BIN:-{shlex.quote(str(DEFAULT_RESET_GPU_CORE_FREQ_BIN))}}}"
            export AMD_SMI_POWER_SOCKET_PATH="${{AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.${{SLURM_JOB_ID}}.sock}}"

            bash {shlex.quote(str(self.start_script))}
            """
        )
        script_path.write_text(script_text, encoding="utf-8")
        script_path.chmod(0o750)

        return {
            "partition": PARTITION,
            "model": model_spec.key,
            "vllm_model_name": model_spec.vllm_model_name,
            "served_model_name": model_spec.served_model_name,
            "profile_list": [layout["profile_id"] for layout in PROFILE_LAYOUTS],
            "service_count": len(PROFILE_LAYOUTS),
            "tensor_parallel_size": 1,
            "experiment_script": str(resolved_experiment_script),
            "sbatch_script": str(script_path),
            "extra_env": dict(normalized_extra_env),
            "extra_vllm_args": list(normalized_extra_vllm_args),
            "effective_model_extra_args": list(effective_model_extra_args),
            "lmcache_enabled": lmcache_enabled,
            "no_async_scheduling": bool(no_async_scheduling),
            "time_limit": effective_time_limit,
            "profiles": [dict(layout) for layout in PROFILE_LAYOUTS],
        }

    def submit(
        self,
        *,
        model: str,
        experiment_script: Path,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        no_async_scheduling: bool = False,
        time_limit: str | None = None,
    ) -> dict[str, Any]:
        rendered = self.render(
            model=model,
            experiment_script=experiment_script,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            extra_vllm_args=extra_vllm_args,
            no_async_scheduling=no_async_scheduling,
            time_limit=time_limit,
        )
        script_path = Path(rendered["sbatch_script"])
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise LaunchError(message="required command not found: sbatch", code=306) from exc
        except subprocess.TimeoutExpired as exc:
            raise LaunchError(
                message="command timed out: sbatch",
                code=307,
                details={"timeout_seconds": 120},
            ) from exc

        if result.returncode != 0:
            raise LaunchError(
                message="command failed: sbatch",
                code=308,
                details={
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                },
            )

        rendered["job_id"] = self._extract_sbatch_job_id(f"{result.stdout}\n{result.stderr}")
        rendered["sbatch_stdout"] = result.stdout.strip()
        rendered["sbatch_stderr"] = result.stderr.strip()
        return rendered

    def _extract_sbatch_job_id(self, text: str) -> str:
        match = re.search(r"\bSubmitted\s+batch\s+job\s+(\d+)\b", text)
        if match is None:
            raise LaunchError(
                message="unable to parse sbatch job id",
                code=309,
                details={"sbatch_output": text.strip()},
            )
        return match.group(1)


def _resolve_cli_inputs(
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
    return extra_env, parsed_extra_vllm_args, str(lmcache) if lmcache is not None else None


def _handle_render(args: argparse.Namespace) -> None:
    extra_env, parsed_extra_vllm_args, lmcache_value = _resolve_cli_inputs(
        env=args.env,
        extra_vllm_args=args.extra_vllm_args,
        lmcache=args.lmcache,
    )
    launcher = Mi2104xEmbeddedTp1Launcher(model_config_path=Path(args.model_config))
    try:
        payload = launcher.render(
            model=args.model,
            experiment_script=Path(args.experiment_script),
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_value,
            extra_vllm_args=parsed_extra_vllm_args,
            no_async_scheduling=bool(args.no_async_scheduling),
            time_limit=args.time_limit,
        )
    except LaunchError as exc:
        _emit_error(exc)

    _print_json(
        {
            "ok": True,
            "code": 0,
            "message": "rendered mi2104x embedded TP1 sbatch",
            "data": payload,
        }
    )


def _handle_submit(args: argparse.Namespace) -> None:
    extra_env, parsed_extra_vllm_args, lmcache_value = _resolve_cli_inputs(
        env=args.env,
        extra_vllm_args=args.extra_vllm_args,
        lmcache=args.lmcache,
    )
    launcher = Mi2104xEmbeddedTp1Launcher(model_config_path=Path(args.model_config))
    try:
        payload = launcher.submit(
            model=args.model,
            experiment_script=Path(args.experiment_script),
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_value,
            extra_vllm_args=parsed_extra_vllm_args,
            no_async_scheduling=bool(args.no_async_scheduling),
            time_limit=args.time_limit,
        )
    except LaunchError as exc:
        _emit_error(exc)

    _print_json(
        {
            "ok": True,
            "code": 0,
            "message": "submitted mi2104x embedded TP1 sbatch",
            "data": payload,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render or submit a dedicated mi2104x embedded TP1 Slurm job. "
            "The job launches the explicit four-profile TP1 service stack and then "
            "runs the supplied experiment script once for each profile id 0..3."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--model-config",
            default=str(DEFAULT_MODEL_CONFIG_PATH),
            help=f"Path to configs/model_config.toml (default: {DEFAULT_MODEL_CONFIG_PATH})",
        )
        command_parser.add_argument(
            "--model",
            "-m",
            required=True,
            help="Model key from configs/model_config.toml.",
        )
        command_parser.add_argument(
            "--experiment-script",
            "-e",
            required=True,
            help="Experiment script path. It will be invoked once per profile id 0..3.",
        )
        command_parser.add_argument(
            "--env",
            action="append",
            default=[],
            help="Additional vLLM environment variable in KEY=VALUE form. Repeat to pass multiple values.",
        )
        command_parser.add_argument(
            "--lmcache",
            type=int,
            default=None,
            help=(
                "Enable LMCache with a maximum local CPU size. "
                "Sets LMCACHE_MAX_LOCAL_CPU_SIZE and appends kv-transfer-config."
            ),
        )
        command_parser.add_argument(
            "--extra-vllm-args",
            default=None,
            help="Additional vLLM CLI args string appended to the model defaults.",
        )
        command_parser.add_argument(
            "--no-async-scheduling",
            action="store_true",
            help="Append --no-async-scheduling to the rendered vLLM launch.",
        )
        command_parser.add_argument(
            "--time-limit",
            default=None,
            help=(
                "Override the Slurm wall-clock time limit for the rendered job, "
                f"for example 08:00:00 or 1-00:00:00. Default: {MAX_TIME}."
            ),
        )

    render_parser = subparsers.add_parser("render", help="Render the sbatch script without submitting it.")
    add_common_arguments(render_parser)
    render_parser.set_defaults(handler=_handle_render)

    submit_parser = subparsers.add_parser("submit", help="Render the sbatch script and submit it with sbatch.")
    add_common_arguments(submit_parser)
    submit_parser.set_defaults(handler=_handle_submit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
