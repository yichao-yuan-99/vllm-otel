"""Typer CLI entrypoint for the concurrent trial driver."""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path
from random import Random
from typing import Any, Sequence

import typer

from con_driver.backends.harbor import HarborBackend, HarborBackendConfig
from con_driver.parsing import (
    assert_safe_harbor_args,
    parse_cli_kv,
    parse_comma_arg_string,
    parse_command_prefix,
    parse_pool_specs,
)
from con_driver.patterns import build_arrival_pattern
from con_driver.scheduler import (
    ConcurrentDriver,
    GatewayModeConfig,
    SchedulerConfig,
    VLLMLogConfig,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, invoke_without_command=True)

_OPTIONS_WITH_VALUE = {
    "--driver-backend",
    "--config",
    "--pool",
    "--pattern",
    "--pattern-args",
    "--max-concurrent",
    "--n-task",
    "--results-dir",
    "--harbor-bin",
    "--seed",
    "--vllm-log-endpoint",
    "--vllm-log-interval-s",
    "--vllm-log-timeout-s",
    "--gateway-url",
    "--gateway-job-output-root",
    "--gateway-timeout-s",
}
_FLAG_OPTIONS = {
    "--dry-run",
    "--no-dry-run",
    "--sample-without-replacement",
    "--sample-with-replacement",
    "--vllm-log",
    "--no-vllm-log",
    "--gateway",
    "--no-gateway",
    "--help",
}


def _run_driver(
    *,
    backend_name: str,
    config_path: Path | None,
    forwarded_args: list[str],
    pool_raw: str,
    pattern: str,
    pattern_args_tokens: list[str],
    max_concurrent: int,
    n_task: int,
    results_dir: Path,
    harbor_bin_tokens: list[str],
    seed: int | None,
    dry_run: bool,
    sample_without_replacement: bool,
    vllm_log_enabled: bool,
    vllm_log_endpoint: str,
    vllm_log_interval_s: float,
    vllm_log_timeout_s: float,
    gateway_enabled: bool,
    gateway_url: str,
    gateway_job_output_root: str,
    gateway_timeout_s: float,
) -> int:
    try:
        if max_concurrent <= 0:
            raise ValueError("--max-concurrent must be > 0")
        if n_task <= 0:
            raise ValueError("--n-task must be > 0")
        if vllm_log_interval_s <= 0:
            raise ValueError("--vllm-log-interval-s must be > 0")
        if vllm_log_timeout_s <= 0:
            raise ValueError("--vllm-log-timeout-s must be > 0")
        if gateway_timeout_s <= 0:
            raise ValueError("--gateway-timeout-s must be > 0")
        if gateway_enabled:
            if not gateway_url.strip():
                raise ValueError("--gateway-url cannot be empty")
            if not gateway_job_output_root.strip():
                raise ValueError("--gateway-job-output-root cannot be empty")
            normalized_subdir = gateway_job_output_root.strip()
            if normalized_subdir in {".", "./"}:
                raise ValueError("--gateway-job-output-root must be a subdirectory path")
            if Path(normalized_subdir).is_absolute():
                raise ValueError("--gateway-job-output-root must be a relative path")

        backend_name_normalized = backend_name.strip().lower()
        if backend_name_normalized != "harbor":
            raise ValueError(
                "Unsupported --driver-backend "
                f"'{backend_name}'. Supported: harbor."
            )
        assert_safe_harbor_args(forwarded_args)

        pool_specs = parse_pool_specs(pool_raw)
        pattern_args = parse_cli_kv(pattern_args_tokens)

        rng = Random(seed)
        arrival_pattern = build_arrival_pattern(
            name=pattern,
            pattern_args=pattern_args,
            rng=rng,
        )
        backend = HarborBackend(
            HarborBackendConfig(
                harbor_bin=harbor_bin_tokens,
                harbor_args=forwarded_args,
            )
        )
        vllm_log_config = (
            VLLMLogConfig(
                endpoint=vllm_log_endpoint,
                interval_s=vllm_log_interval_s,
                timeout_s=vllm_log_timeout_s,
            )
            if vllm_log_enabled
            else None
        )
        gateway_config = (
            GatewayModeConfig(
                base_url=gateway_url,
                job_output_root=gateway_job_output_root,
                timeout_s=gateway_timeout_s,
            )
            if gateway_enabled
            else None
        )
        effective_config: dict[str, object] = {
            "backend": backend_name_normalized,
            "pool_raw": pool_raw,
            "pool_specs": pool_specs,
            "pattern": pattern,
            "pattern_args_tokens": pattern_args_tokens,
            "max_concurrent": max_concurrent,
            "n_task": n_task,
            "dry_run": dry_run,
            "sample_without_replacement": sample_without_replacement,
            "seed": seed,
            "harbor_bin_tokens": harbor_bin_tokens,
            "forwarded_args": forwarded_args,
            "vllm_log_enabled": vllm_log_enabled,
            "vllm_log_endpoint": vllm_log_endpoint,
            "vllm_log_interval_s": vllm_log_interval_s,
            "vllm_log_timeout_s": vllm_log_timeout_s,
            "gateway_enabled": gateway_enabled,
            "gateway_url": gateway_url,
            "gateway_job_output_subdir": gateway_job_output_root,
            "gateway_job_output_root": gateway_job_output_root,
            "gateway_timeout_s": gateway_timeout_s,
        }
        if config_path is not None:
            effective_config["source_config"] = str(config_path.resolve())

        driver = ConcurrentDriver(
            backend=backend,
            arrival_pattern=arrival_pattern,
            rng=rng,
            config=SchedulerConfig(
                max_concurrent=max_concurrent,
                n_task=n_task,
                results_dir=results_dir,
                dry_run=dry_run,
                sample_with_replacement=not sample_without_replacement,
                effective_config=effective_config,
                vllm_log=vllm_log_config,
                gateway=gateway_config,
            ),
        )

        summary = asyncio.run(driver.run(pool_specs=pool_specs))
    except KeyboardInterrupt:
        typer.echo("Interrupted (Ctrl-C). Broadcasted interrupt to active subprocesses.", err=True)
        return 130
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        return 1

    config_toml_path = summary.results_dir / "meta" / "config.toml"
    summary_payload = {
        "run_id": summary.run_id,
        "total_requested": summary.total_requested,
        "launched": summary.launched,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "dry_run": summary.dry_run,
        "results_dir": str(summary.results_dir),
        "manifest_path": str(summary.manifest_path),
        "events_path": str(summary.events_path),
        "config_toml_path": str(config_toml_path),
    }
    typer.echo(json.dumps(summary_payload, indent=2))

    return 0 if summary.failed == 0 else 2


def _load_toml_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse TOML config {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a TOML table: {config_path}")

    if "driver" in data:
        section = data["driver"]
        if not isinstance(section, dict):
            raise ValueError("Config key 'driver' must be a table")
        return dict(section)

    return dict(data)


def _coerce_str(value: Any, *, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Config key '{key}' must be a string")


def _coerce_int(value: Any, *, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Config key '{key}' must be an integer") from exc
    raise ValueError(f"Config key '{key}' must be an integer")


def _coerce_bool(value: Any, *, key: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Config key '{key}' must be a boolean")


def _coerce_path(value: Any, *, key: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise ValueError(f"Config key '{key}' must be a path string")


def _coerce_float(value: Any, *, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Config key '{key}' must be a float") from exc
    raise ValueError(f"Config key '{key}' must be a float")


def _parse_token_list(value: Any, *, key: str) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return parse_comma_arg_string(value)

    if isinstance(value, list):
        tokens: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(
                    f"Config key '{key}' list item at index {index} must be a string"
                )
            tokens.append(item)
        return tokens

    raise ValueError(f"Config key '{key}' must be a string or list of strings")


def _resolve_required[T](value: T | None, *, option_name: str, config_key: str) -> T:
    if value is None:
        raise ValueError(
            f"Missing required option '{option_name}' (or set '{config_key}' in --config)."
        )
    return value


def _resolve_command_prefix_tokens(
    *,
    cli_value: str | None,
    config_value: Any,
    default_command: str,
    key_name: str,
) -> list[str]:
    source = cli_value if cli_value is not None else config_value

    if source is None:
        return parse_command_prefix(default_command)

    if isinstance(source, str):
        return parse_command_prefix(source)

    if isinstance(source, list):
        if not source:
            raise ValueError(f"Config key '{key_name}' list cannot be empty")
        tokens: list[str] = []
        for index, item in enumerate(source):
            if not isinstance(item, str):
                raise ValueError(
                    f"Config key '{key_name}' list item at index {index} must be a string"
                )
            tokens.append(item)
        return tokens

    raise ValueError(
        f"Config key '{key_name}' must be a string or list of strings"
    )

@app.callback()
def run(
    ctx: typer.Context,
    driver_backend: str | None = typer.Option(
        None,
        "--driver-backend",
        help="Driver backend to use. Only 'harbor' is supported.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to TOML config file. Values in CLI options override config values.",
    ),
    pool: str | None = typer.Option(
        None,
        "--pool",
        help=(
            "Comma-separated dataset specs, e.g. "
            "'swebench-verified,terminal-bench@2.0'."
        ),
    ),
    pattern: str | None = typer.Option(
        None,
        "--pattern",
        help="Arrival pattern for launch timing: eager or poisson.",
    ),
    pattern_args: str | None = typer.Option(
        None,
        "--pattern-args",
        help=(
            "Comma-separated args for arrival pattern. "
            "For poisson: '--rate=0.2' or '--mean-interval-s=5'."
        ),
    ),
    max_concurrent: int | None = typer.Option(
        None,
        "--max-concurrent",
        help="Maximum number of in-flight trial processes.",
    ),
    n_task: int | None = typer.Option(
        None,
        "--n-task",
        help="Total number of trial launches to attempt.",
    ),
    results_dir: Path | None = typer.Option(
        None,
        "--results-dir",
        help="Output directory containing downloads, logs, metadata, and trial dirs.",
    ),
    harbor_bin: str | None = typer.Option(
        None,
        "--harbor-bin",
        help="Command prefix used to call Harbor, e.g. 'harbor' or 'uv run harbor'.",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible sampling and poisson delays.",
    ),
    dry_run: bool | None = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="Build commands and sampling decisions without launching trials.",
    ),
    sample_without_replacement: bool | None = typer.Option(
        None,
        "--sample-without-replacement/--sample-with-replacement",
        help=(
            "Sample tasks without replacement. If enabled, each sampled task path "
            "appears at most once."
        ),
    ),
    vllm_log: bool | None = typer.Option(
        None,
        "--vllm-log/--no-vllm-log",
        help="Enable vLLM Prometheus metrics logging process.",
    ),
    vllm_log_endpoint: str | None = typer.Option(
        None,
        "--vllm-log-endpoint",
        help="Prometheus endpoint polled by the vLLM monitor.",
    ),
    vllm_log_interval_s: float | None = typer.Option(
        None,
        "--vllm-log-interval-s",
        help="Polling interval in seconds for the vLLM metrics monitor.",
    ),
    vllm_log_timeout_s: float | None = typer.Option(
        None,
        "--vllm-log-timeout-s",
        help="HTTP timeout in seconds for the vLLM metrics monitor.",
    ),
    gateway: bool | None = typer.Option(
        None,
        "--gateway/--no-gateway",
        help="Enable gateway mode: job lifecycle API + per-agent unique token wrapper.",
    ),
    gateway_url: str | None = typer.Option(
        None,
        "--gateway-url",
        help="Gateway base URL, e.g. http://127.0.0.1:11457",
    ),
    gateway_job_output_root: str | None = typer.Option(
        None,
        "--gateway-job-output-root",
        help=(
            "Run-local subdirectory under the con-driver run directory sent to "
            "gateway /job/start as output_location."
        ),
    ),
    gateway_timeout_s: float | None = typer.Option(
        None,
        "--gateway-timeout-s",
        help="HTTP timeout in seconds for gateway lifecycle calls.",
    ),
) -> None:
    """Launch concurrent Harbor trials; extra CLI args are forwarded to Harbor."""
    forwarded_args_from_cli = []
    if isinstance(ctx.obj, dict):
        raw = ctx.obj.get("forwarded_args", [])
        if isinstance(raw, list):
            forwarded_args_from_cli = [str(arg) for arg in raw if str(arg) != "--"]

    try:
        config_values = _load_toml_config(config)
        backend_value = _coerce_str(
            driver_backend
            if driver_backend is not None
            else config_values.get("driver_backend", config_values.get("backend", "harbor")),
            key="driver_backend",
        )
        backend_value = _resolve_required(
            backend_value,
            option_name="--driver-backend",
            config_key="driver_backend",
        )

        pool_value = _coerce_str(
            pool if pool is not None else config_values.get("pool"),
            key="pool",
        )
        pool_value = _resolve_required(
            pool_value,
            option_name="--pool",
            config_key="pool",
        )

        pattern_value = _coerce_str(
            pattern if pattern is not None else config_values.get("pattern", "eager"),
            key="pattern",
        )
        pattern_value = _resolve_required(
            pattern_value,
            option_name="--pattern",
            config_key="pattern",
        )

        max_concurrent_value = _coerce_int(
            max_concurrent
            if max_concurrent is not None
            else config_values.get("max_concurrent"),
            key="max_concurrent",
        )
        max_concurrent_value = _resolve_required(
            max_concurrent_value,
            option_name="--max-concurrent",
            config_key="max_concurrent",
        )

        n_task_value = _coerce_int(
            n_task if n_task is not None else config_values.get("n_task"),
            key="n_task",
        )
        n_task_value = _resolve_required(
            n_task_value,
            option_name="--n-task",
            config_key="n_task",
        )

        results_dir_value = _coerce_path(
            results_dir if results_dir is not None else config_values.get("results_dir"),
            key="results_dir",
        )
        results_dir_value = _resolve_required(
            results_dir_value,
            option_name="--results-dir",
            config_key="results_dir",
        )

        seed_value = _coerce_int(
            seed if seed is not None else config_values.get("seed"),
            key="seed",
        )

        dry_run_value = _coerce_bool(
            dry_run if dry_run is not None else config_values.get("dry_run", False),
            key="dry_run",
        )
        dry_run_value = _resolve_required(
            dry_run_value,
            option_name="--dry-run/--no-dry-run",
            config_key="dry_run",
        )

        sample_without_replacement_value = _coerce_bool(
            sample_without_replacement
            if sample_without_replacement is not None
            else config_values.get(
                "sample_without_replacement",
                config_values.get("no_repeat_sampling", False),
            ),
            key="sample_without_replacement",
        )
        sample_without_replacement_value = _resolve_required(
            sample_without_replacement_value,
            option_name="--sample-without-replacement/--sample-with-replacement",
            config_key="sample_without_replacement",
        )

        vllm_log_value = _coerce_bool(
            vllm_log if vllm_log is not None else config_values.get("vllm_log", False),
            key="vllm_log",
        )
        vllm_log_value = _resolve_required(
            vllm_log_value,
            option_name="--vllm-log/--no-vllm-log",
            config_key="vllm_log",
        )

        vllm_log_endpoint_value = _coerce_str(
            (
                vllm_log_endpoint
                if vllm_log_endpoint is not None
                else config_values.get("vllm_log_endpoint", "http://localhost:12138/metrics")
            ),
            key="vllm_log_endpoint",
        )
        vllm_log_endpoint_value = _resolve_required(
            vllm_log_endpoint_value,
            option_name="--vllm-log-endpoint",
            config_key="vllm_log_endpoint",
        )

        vllm_log_interval_s_value = _coerce_float(
            (
                vllm_log_interval_s
                if vllm_log_interval_s is not None
                else config_values.get("vllm_log_interval_s", 1.0)
            ),
            key="vllm_log_interval_s",
        )
        vllm_log_interval_s_value = _resolve_required(
            vllm_log_interval_s_value,
            option_name="--vllm-log-interval-s",
            config_key="vllm_log_interval_s",
        )

        vllm_log_timeout_s_value = _coerce_float(
            (
                vllm_log_timeout_s
                if vllm_log_timeout_s is not None
                else config_values.get("vllm_log_timeout_s", 5.0)
            ),
            key="vllm_log_timeout_s",
        )
        vllm_log_timeout_s_value = _resolve_required(
            vllm_log_timeout_s_value,
            option_name="--vllm-log-timeout-s",
            config_key="vllm_log_timeout_s",
        )

        gateway_value = _coerce_bool(
            gateway if gateway is not None else config_values.get("gateway", True),
            key="gateway",
        )
        gateway_value = _resolve_required(
            gateway_value,
            option_name="--gateway/--no-gateway",
            config_key="gateway",
        )

        gateway_url_value = _coerce_str(
            (
                gateway_url
                if gateway_url is not None
                else config_values.get("gateway_url", "http://127.0.0.1:11457")
            ),
            key="gateway_url",
        )
        gateway_url_value = _resolve_required(
            gateway_url_value,
            option_name="--gateway-url",
            config_key="gateway_url",
        )

        gateway_job_output_root_value = _coerce_str(
            (
                gateway_job_output_root
                if gateway_job_output_root is not None
                else config_values.get(
                    "gateway_job_output_root",
                    config_values.get("gateway_job_output_subdir", "gateway-output"),
                )
            ),
            key="gateway_job_output_root",
        )
        gateway_job_output_root_value = _resolve_required(
            gateway_job_output_root_value,
            option_name="--gateway-job-output-root",
            config_key="gateway_job_output_root",
        )

        gateway_timeout_s_value = _coerce_float(
            (
                gateway_timeout_s
                if gateway_timeout_s is not None
                else config_values.get("gateway_timeout_s", 30.0)
            ),
            key="gateway_timeout_s",
        )
        gateway_timeout_s_value = _resolve_required(
            gateway_timeout_s_value,
            option_name="--gateway-timeout-s",
            config_key="gateway_timeout_s",
        )

        pattern_args_source = (
            pattern_args if pattern_args is not None else config_values.get("pattern_args")
        )
        pattern_args_tokens = _parse_token_list(
            pattern_args_source,
            key="pattern_args",
        )

        harbor_bin_tokens = _resolve_command_prefix_tokens(
            cli_value=harbor_bin,
            config_value=config_values.get("harbor_bin"),
            default_command="harbor",
            key_name="harbor_bin",
        )

        forwarded_args_from_config = []
        forwarded_args_from_config.extend(
            _parse_token_list(config_values.get("forwarded_args"), key="forwarded_args")
        )
        # Backward-compatible alias.
        forwarded_args_from_config.extend(
            _parse_token_list(config_values.get("harbor_args"), key="harbor_args")
        )

    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)

    exit_code = _run_driver(
        backend_name=backend_value,
        config_path=config,
        forwarded_args=forwarded_args_from_config + forwarded_args_from_cli,
        pool_raw=pool_value,
        pattern=pattern_value,
        pattern_args_tokens=pattern_args_tokens,
        max_concurrent=max_concurrent_value,
        n_task=n_task_value,
        results_dir=results_dir_value,
        harbor_bin_tokens=harbor_bin_tokens,
        seed=seed_value,
        dry_run=dry_run_value,
        sample_without_replacement=sample_without_replacement_value,
        vllm_log_enabled=vllm_log_value,
        vllm_log_endpoint=vllm_log_endpoint_value,
        vllm_log_interval_s=vllm_log_interval_s_value,
        vllm_log_timeout_s=vllm_log_timeout_s_value,
        gateway_enabled=gateway_value,
        gateway_url=gateway_url_value,
        gateway_job_output_root=gateway_job_output_root_value,
        gateway_timeout_s=gateway_timeout_s_value,
    )
    raise typer.Exit(code=exit_code)


def _split_driver_and_forwarded_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    driver_args: list[str] = []
    forwarded_args: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]

        if token == "--":
            forwarded_args.extend(argv[i + 1 :])
            break

        if token.startswith("--"):
            option_name = token.split("=", 1)[0]
            if option_name in _OPTIONS_WITH_VALUE:
                driver_args.append(token)
                if "=" not in token:
                    if i + 1 >= len(argv):
                        # Let Typer produce the missing-value error.
                        i += 1
                        continue
                    driver_args.append(argv[i + 1])
                    i += 2
                else:
                    i += 1
                continue

            if option_name in _FLAG_OPTIONS:
                driver_args.append(token)
                i += 1
                continue

        forwarded_args.append(token)
        i += 1

    return driver_args, forwarded_args


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    driver_args, forwarded_args = _split_driver_and_forwarded_args(raw_args)

    try:
        app(args=driver_args, prog_name="con-driver", obj={"forwarded_args": forwarded_args})
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
