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

from con_driver.backends.harbor import (
    HarborBackend,
    HarborBackendConfig,
    resolve_harbor_runtime,
)
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
    LaunchProfileConfig,
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
    "--port-profile-id",
    "--port-profile-id-list",
    "--max-concurrent-list",
    "--agent-name",
    "--seed",
    "--vllm-log-interval-s",
    "--vllm-log-timeout-s",
    "--gateway-url",
    "--gateway-job-output-root",
    "--gateway-timeout-s",
    # Removed option; keep here so main parser routes it to Typer and returns
    # an explicit unknown-option error instead of forwarding to Harbor.
    "--gateway-api-key",
    "--task-subset-start",
    "--task-subset-end",
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
    trial_env: dict[str, str],
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
    task_subset_start: int | None,
    task_subset_end: int | None,
    port_profile_id: int | None,
    launch_profiles: list[LaunchProfileConfig] | None,
    resolved_agent_name: str | None,
    resolved_model_name: str | None,
    resolved_model_context_window: int | None,
    agent_base_url: str | None,
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
        task_subset_start_value = 0 if task_subset_start is None else task_subset_start
        task_subset_end_value = task_subset_end
        if task_subset_start_value < 0:
            raise ValueError("--task-subset-start must be >= 0")
        if task_subset_end_value is not None:
            if task_subset_end_value < 0:
                raise ValueError("--task-subset-end must be >= 0")
            if task_subset_end_value <= task_subset_start_value:
                raise ValueError("--task-subset-end must be greater than --task-subset-start")

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
                harbor_args=([] if launch_profiles is not None else forwarded_args),
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
            "task_subset_start": task_subset_start_value,
            "task_subset_end": task_subset_end_value,
            "harbor_bin_tokens": harbor_bin_tokens,
            "forwarded_args": forwarded_args,
            "port_profile_id": port_profile_id,
            "launch_profiles": (
                [
                    {
                        "port_profile_id": profile.port_profile_id,
                        "max_concurrent": profile.max_concurrent,
                        "gateway_base_url": profile.gateway_base_url,
                        "vllm_log_endpoint": profile.vllm_log_endpoint,
                    }
                    for profile in launch_profiles
                ]
                if launch_profiles is not None
                else None
            ),
            "resolved_agent_name": resolved_agent_name,
            "resolved_model_name": resolved_model_name,
            "resolved_model_context_window": resolved_model_context_window,
            "agent_base_url": agent_base_url,
            "vllm_log_enabled": vllm_log_enabled,
            "vllm_log_endpoint": vllm_log_endpoint,
            "vllm_log_endpoints": (
                [
                    profile.vllm_log_endpoint
                    for profile in launch_profiles
                    if profile.vllm_log_endpoint is not None
                ]
                if launch_profiles is not None
                else (
                    [vllm_log_endpoint]
                    if vllm_log_enabled and vllm_log_endpoint.strip()
                    else []
                )
            ),
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
                task_subset_start=task_subset_start_value,
                task_subset_end=task_subset_end_value,
                launch_profiles=launch_profiles,
                effective_config=effective_config,
                vllm_log=vllm_log_config,
                gateway=gateway_config,
                launch_env=dict(trial_env),
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


def _coerce_int_list(value: Any, *, key: str) -> list[int] | None:
    if value is None:
        return None

    raw_items: list[Any]
    if isinstance(value, str):
        raw_items = parse_comma_arg_string(value)
    elif isinstance(value, list):
        raw_items = list(value)
    else:
        raise ValueError(
            f"Config key '{key}' must be a comma-separated string or list of integers"
        )

    values: list[int] = []
    for index, item in enumerate(raw_items):
        parsed = _coerce_int(item, key=f"{key}[{index}]")
        if parsed is None:
            raise ValueError(f"Config key '{key}[{index}]' must be an integer")
        values.append(parsed)

    if not values:
        raise ValueError(f"Config key '{key}' cannot be empty")
    return values


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
        help=(
            "Maximum number of in-flight trial processes. Optional when "
            "--port-profile-id-list is set; defaults to sum(--max-concurrent-list)."
        ),
    ),
    n_task: int | None = typer.Option(
        None,
        "--n-task",
        help="Total number of trial launches to attempt.",
    ),
    task_subset_start: int | None = typer.Option(
        None,
        "--task-subset-start",
        help=(
            "0-based inclusive task index into the prepared task pool before sampling. "
            "Defaults to 0."
        ),
    ),
    task_subset_end: int | None = typer.Option(
        None,
        "--task-subset-end",
        help=(
            "0-based exclusive task index into the prepared task pool before sampling. "
            "Defaults to pool size."
        ),
    ),
    results_dir: Path | None = typer.Option(
        None,
        "--results-dir",
        help="Output directory containing run logs, metadata, and trial dirs.",
    ),
    harbor_bin: str | None = typer.Option(
        None,
        "--harbor-bin",
        help="Command prefix used to call Harbor, e.g. 'harbor' or 'uv run harbor'.",
    ),
    port_profile_id: int | None = typer.Option(
        None,
        "--port-profile-id",
        help="Port profile numeric ID from configs/port_profiles.toml.",
    ),
    port_profile_id_list: str | None = typer.Option(
        None,
        "--port-profile-id-list",
        help=(
            "Comma-separated port profile IDs for cluster mode, for example "
            "'0,1,2,3,4'. Cannot be combined with --port-profile-id."
        ),
    ),
    max_concurrent_list: str | None = typer.Option(
        None,
        "--max-concurrent-list",
        help=(
            "Comma-separated per-profile max concurrency values matching "
            "--port-profile-id-list order, for example '5,5,5,5,5'."
        ),
    ),
    agent_name: str | None = typer.Option(
        None,
        "--agent-name",
        help="Harbor agent name. Config keys 'agent' and 'agent_name' are also accepted.",
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

        port_profile_id_value = _coerce_int(
            (
                port_profile_id
                if port_profile_id is not None
                else config_values.get("port_profile_id")
            ),
            key="port_profile_id",
        )
        port_profile_id_list_value = _coerce_int_list(
            (
                port_profile_id_list
                if port_profile_id_list is not None
                else config_values.get("port_profile_id_list")
            ),
            key="port_profile_id_list",
        )
        max_concurrent_list_value = _coerce_int_list(
            (
                max_concurrent_list
                if max_concurrent_list is not None
                else config_values.get("max_concurrent_list")
            ),
            key="max_concurrent_list",
        )

        cluster_mode = port_profile_id_list_value is not None
        if cluster_mode and port_profile_id_value is not None:
            raise ValueError(
                "--port-profile-id cannot be used with --port-profile-id-list."
            )
        if cluster_mode and max_concurrent_list_value is None:
            raise ValueError(
                "--max-concurrent-list is required when --port-profile-id-list is set."
            )
        if not cluster_mode and max_concurrent_list_value is not None:
            raise ValueError(
                "--max-concurrent-list requires --port-profile-id-list."
            )
        if (
            cluster_mode
            and max_concurrent_list_value is not None
            and len(port_profile_id_list_value) != len(max_concurrent_list_value)
        ):
            raise ValueError(
                "--port-profile-id-list and --max-concurrent-list must have the same number of values."
            )
        if max_concurrent_list_value is not None:
            for index, value in enumerate(max_concurrent_list_value):
                if value <= 0:
                    raise ValueError(f"max_concurrent_list[{index}] must be > 0")
        if port_profile_id_list_value is not None:
            if len(set(port_profile_id_list_value)) != len(port_profile_id_list_value):
                raise ValueError("--port-profile-id-list cannot contain duplicate IDs.")

        max_concurrent_candidate = _coerce_int(
            max_concurrent
            if max_concurrent is not None
            else config_values.get("max_concurrent"),
            key="max_concurrent",
        )
        if cluster_mode and max_concurrent_candidate is None:
            max_concurrent_value = sum(max_concurrent_list_value or [])
        else:
            max_concurrent_value = _resolve_required(
                max_concurrent_candidate,
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
        task_subset_start_value = _coerce_int(
            (
                task_subset_start
                if task_subset_start is not None
                else config_values.get("task_subset_start", 0)
            ),
            key="task_subset_start",
        )
        task_subset_end_value = _coerce_int(
            (
                task_subset_end
                if task_subset_end is not None
                else config_values.get("task_subset_end")
            ),
            key="task_subset_end",
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

        agent_name_value = _coerce_str(
            (
                agent_name
                if agent_name is not None
                else config_values.get("agent", config_values.get("agent_name"))
            ),
            key="agent",
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

        gateway_value = _coerce_bool(
            gateway if gateway is not None else config_values.get("gateway", True),
            key="gateway",
        )
        gateway_value = _resolve_required(
            gateway_value,
            option_name="--gateway/--no-gateway",
            config_key="gateway",
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
                else config_values.get("gateway_timeout_s", 3600.0)
            ),
            key="gateway_timeout_s",
        )
        gateway_timeout_s_value = _resolve_required(
            gateway_timeout_s_value,
            option_name="--gateway-timeout-s",
            config_key="gateway_timeout_s",
        )
        if "gateway_api_key" in config_values:
            raise ValueError(
                "Config key 'gateway_api_key' has been removed. "
                "Configure upstream API key in gateway-lite instead."
            )

        configured_vllm_log_value = _coerce_bool(
            vllm_log if vllm_log is not None else config_values.get("vllm_log"),
            key="vllm_log",
        )

        if "vllm_log_endpoint" in config_values:
            raise ValueError(
                "Config key 'vllm_log_endpoint' is no longer supported. "
                "vLLM log endpoint is always resolved from 'port_profile_id'."
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

        configured_gateway_url_value = _coerce_str(
            (
                gateway_url
                if gateway_url is not None
                else config_values.get("gateway_url")
            ),
            key="gateway_url",
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

        forwarded_args_from_config: list[str] = []
        forwarded_args_from_config.extend(
            _parse_token_list(config_values.get("forwarded_args"), key="forwarded_args")
        )
        # Backward-compatible alias.
        forwarded_args_from_config.extend(
            _parse_token_list(config_values.get("harbor_args"), key="harbor_args")
        )
        launch_profiles_value: list[LaunchProfileConfig] | None = None
        if cluster_mode:
            if gateway_value and configured_gateway_url_value is not None:
                raise ValueError(
                    "Do not set 'gateway_url' when using --port-profile-id-list. "
                    "Gateway URL is derived from each profile."
                )

            profile_pairs = list(
                zip(
                    port_profile_id_list_value or [],
                    max_concurrent_list_value or [],
                )
            )
            profile_pairs.sort(key=lambda item: item[0])
            launch_profiles_value = []
            resolved_agent_name = None
            resolved_model_name = None
            resolved_model_context_window = None
            derived_agent_base_url = None
            first_gateway_url: str | None = None
            first_vllm_log_endpoint: str | None = None
            resolved_vllm_log_enabled: bool | None = None
            for profile_id, profile_max_concurrent in profile_pairs:
                profile_runtime = resolve_harbor_runtime(
                    forwarded_args_from_config=forwarded_args_from_config,
                    forwarded_args_from_cli=forwarded_args_from_cli,
                    port_profile_id=profile_id,
                    agent_name=agent_name_value,
                    gateway_enabled=gateway_value,
                    configured_gateway_url=None,
                    gateway_timeout_s=gateway_timeout_s_value,
                    configured_vllm_log=configured_vllm_log_value,
                )
                if resolved_agent_name is None:
                    resolved_agent_name = profile_runtime.resolved_agent_name
                elif profile_runtime.resolved_agent_name != resolved_agent_name:
                    raise ValueError(
                        "Resolved agent differs across port profiles in cluster mode."
                    )
                if resolved_model_name is None:
                    resolved_model_name = profile_runtime.resolved_model_name
                    resolved_model_context_window = profile_runtime.resolved_model_context_window
                elif profile_runtime.resolved_model_name != resolved_model_name:
                    raise ValueError(
                        "Resolved served model differs across port profiles in cluster mode. "
                        "All selected profiles must serve the same model."
                    )
                if resolved_vllm_log_enabled is None:
                    resolved_vllm_log_enabled = profile_runtime.vllm_log_enabled
                elif profile_runtime.vllm_log_enabled != resolved_vllm_log_enabled:
                    raise ValueError(
                        "Resolved vllm_log setting differs across port profiles in cluster mode."
                    )
                if first_gateway_url is None:
                    first_gateway_url = profile_runtime.gateway_url
                if (
                    profile_runtime.vllm_log_enabled
                    and first_vllm_log_endpoint is None
                    and profile_runtime.vllm_log_endpoint.strip()
                ):
                    first_vllm_log_endpoint = profile_runtime.vllm_log_endpoint
                launch_profiles_value.append(
                    LaunchProfileConfig(
                        port_profile_id=profile_id,
                        max_concurrent=profile_max_concurrent,
                        forwarded_args=list(profile_runtime.forwarded_args),
                        launch_env=dict(profile_runtime.trial_env),
                        gateway_base_url=(
                            profile_runtime.gateway_url if gateway_value else None
                        ),
                        vllm_log_endpoint=(
                            profile_runtime.vllm_log_endpoint
                            if profile_runtime.vllm_log_enabled
                            else None
                        ),
                    )
                )

            synthesized_forwarded_args = []
            launch_env = {}
            gateway_url_value = first_gateway_url or "http://127.0.0.1:11457"
            vllm_log_value = bool(resolved_vllm_log_enabled)
            vllm_log_endpoint_value = first_vllm_log_endpoint or ""
        else:
            harbor_runtime = resolve_harbor_runtime(
                forwarded_args_from_config=forwarded_args_from_config,
                forwarded_args_from_cli=forwarded_args_from_cli,
                port_profile_id=port_profile_id_value,
                agent_name=agent_name_value,
                gateway_enabled=gateway_value,
                configured_gateway_url=configured_gateway_url_value,
                gateway_timeout_s=gateway_timeout_s_value,
                configured_vllm_log=configured_vllm_log_value,
            )
            synthesized_forwarded_args = harbor_runtime.forwarded_args
            launch_env = harbor_runtime.trial_env
            gateway_url_value = harbor_runtime.gateway_url
            vllm_log_value = harbor_runtime.vllm_log_enabled
            vllm_log_endpoint_value = harbor_runtime.vllm_log_endpoint
            resolved_agent_name = harbor_runtime.resolved_agent_name
            resolved_model_name = harbor_runtime.resolved_model_name
            resolved_model_context_window = harbor_runtime.resolved_model_context_window
            derived_agent_base_url = harbor_runtime.agent_base_url

    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)

    exit_code = _run_driver(
        backend_name=backend_value,
        config_path=config,
        forwarded_args=synthesized_forwarded_args,
        trial_env=launch_env,
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
        task_subset_start=task_subset_start_value,
        task_subset_end=task_subset_end_value,
        port_profile_id=port_profile_id_value,
        launch_profiles=launch_profiles_value,
        resolved_agent_name=resolved_agent_name,
        resolved_model_name=resolved_model_name,
        resolved_model_context_window=resolved_model_context_window,
        agent_base_url=derived_agent_base_url,
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
