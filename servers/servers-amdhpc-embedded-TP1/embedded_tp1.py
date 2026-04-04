#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render or submit embedded TP=1 Slurm jobs for AMD HPC nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import textwrap
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODULE_ROOT.parent.parent
AMDHPC_ROOT = REPO_ROOT / "servers" / "servers-amdhpc"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(AMDHPC_ROOT) not in sys.path:
    sys.path.insert(0, str(AMDHPC_ROOT))

from gateway.port_profiles import load_port_profile as load_gateway_port_profile  # type: ignore[import-not-found]
from control_plane import (  # type: ignore[import-not-found]
    ControlPlaneError,
    _apply_lmcache_option,
    _effective_vllm_extra_args,
    _encode_model_extra_args,
    _normalize_service_extra_env,
    _normalize_user_extra_vllm_args,
    _render_apptainer_extra_env_flags,
    _safe_token,
    load_runtime_config,
)


DEFAULT_CONFIG_PATH = MODULE_ROOT / "server_config.toml"
SUPPORTED_PARTITION_PROFILE_IDS: dict[str, list[int]] = {
    "mi3001x": [0],
    "mi3008x": list(range(8)),
}


@dataclass(frozen=True)
class EmbeddedProfileLayout:
    profile_id: int
    vllm_port: int
    jaeger_api_port: int
    jaeger_otlp_port: int
    gateway_port: int
    gateway_parse_port: int
    lmcache_port: int
    rocr_visible_devices: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_profile": self.profile_id,
            "vllm_port": self.vllm_port,
            "jaeger_api_port": self.jaeger_api_port,
            "jaeger_otlp_port": self.jaeger_otlp_port,
            "gateway_port": self.gateway_port,
            "gateway_parse_port": self.gateway_parse_port,
            "lmcache_port": self.lmcache_port,
            "rocr_visible_devices": self.rocr_visible_devices,
        }


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
    return parsed


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


def resolve_experiment_script(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"experiment script not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"experiment script must be a file: {resolved}")
    return resolved


class EmbeddedTp1Launcher:
    """Render or submit Slurm jobs that run one TP=1 vLLM per GPU."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path.expanduser().resolve()
        self._cfg = load_runtime_config(self._config_path)
        self._cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.state_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> Any:
        return self._cfg

    def render(
        self,
        *,
        partition: str,
        model: str,
        experiment_script: Path,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        no_async_scheduling: bool = False,
    ) -> dict[str, Any]:
        try:
            resolved_experiment_script = resolve_experiment_script(experiment_script)
            normalized_extra_env = _normalize_service_extra_env(extra_env)
            normalized_extra_env, lmcache_enabled = _apply_lmcache_option(
                extra_env=normalized_extra_env,
                lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            )
            normalized_extra_vllm_args = _normalize_user_extra_vllm_args(extra_vllm_args)
        except ValueError as exc:
            raise ControlPlaneError(
                message=f"invalid embedded-tp1 render input: {exc}",
                code=301,
                http_status=400,
            ) from exc

        profile_ids = SUPPORTED_PARTITION_PROFILE_IDS.get(partition)
        if profile_ids is None:
            raise ControlPlaneError(
                message=(
                    "embedded TP1 supports only partitions "
                    f"{sorted(SUPPORTED_PARTITION_PROFILE_IDS.keys())}; got '{partition}'"
                ),
                code=302,
                http_status=400,
            )

        partition_spec = self._cfg.partitions.get(partition)
        if partition_spec is None:
            raise ControlPlaneError(
                message=f"unknown partition '{partition}'",
                code=303,
                http_status=400,
                details={"allowed_partitions": sorted(self._cfg.partitions.keys())},
            )

        model_spec = self._cfg.models.get(model)
        if model_spec is None:
            raise ControlPlaneError(
                message=f"unknown model '{model}'",
                code=304,
                http_status=400,
                details={"allowed_models": sorted(self._cfg.models.keys())},
            )

        max_weight = partition_spec.gpu_memory_gb * 0.75
        if model_spec.weight_vram_gb > max_weight:
            raise ControlPlaneError(
                message=(
                    f"model '{model}' requires {model_spec.weight_vram_gb:.1f} GB which exceeds "
                    f"75% of single-GPU VRAM on partition '{partition}' ({max_weight:.1f} GB)"
                ),
                code=305,
                http_status=422,
                details={
                    "model_weight_vram_gb": model_spec.weight_vram_gb,
                    "partition_gpu_memory_gb": partition_spec.gpu_memory_gb,
                    "max_allowed_weight_vram_gb": max_weight,
                },
            )

        profiles = self._build_profiles(profile_ids)
        script_path = self._write_sbatch_script(
            partition=partition_spec.name,
            model=model_spec,
            profiles=profiles,
            experiment_script=resolved_experiment_script,
            extra_env=normalized_extra_env,
            lmcache_enabled=lmcache_enabled,
            extra_vllm_args=normalized_extra_vllm_args,
            no_async_scheduling=no_async_scheduling,
        )
        effective_vllm_sif = partition_spec.vllm_sif or self._cfg.vllm_sif

        return {
            "partition": partition_spec.name,
            "model": model_spec.name,
            "profile_list": [profile.profile_id for profile in profiles],
            "service_count": len(profiles),
            "tensor_parallel_size": 1,
            "jaeger_sif": str(self._cfg.jaeger_sif),
            "vllm_sif": str(effective_vllm_sif),
            "experiment_script": str(resolved_experiment_script),
            "sbatch_script": str(script_path),
            "extra_env": dict(normalized_extra_env),
            "extra_vllm_args": list(normalized_extra_vllm_args),
            "lmcache_enabled": lmcache_enabled,
            "no_async_scheduling": bool(no_async_scheduling),
            "profiles": [profile.to_dict() for profile in profiles],
        }

    def submit(
        self,
        *,
        partition: str,
        model: str,
        experiment_script: Path,
        extra_env: dict[str, str] | None = None,
        lmcache_max_local_cpu_size: str | None = None,
        extra_vllm_args: list[str] | None = None,
        no_async_scheduling: bool = False,
    ) -> dict[str, Any]:
        rendered = self.render(
            partition=partition,
            model=model,
            experiment_script=experiment_script,
            extra_env=extra_env,
            lmcache_max_local_cpu_size=lmcache_max_local_cpu_size,
            extra_vllm_args=extra_vllm_args,
            no_async_scheduling=no_async_scheduling,
        )

        script_path = Path(rendered["sbatch_script"])
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                cwd=self._cfg.repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise ControlPlaneError(
                message="required command not found: sbatch",
                code=306,
                http_status=500,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ControlPlaneError(
                message="command timed out: sbatch",
                code=307,
                http_status=500,
                details={"timeout_seconds": 120},
            ) from exc

        if result.returncode != 0:
            raise ControlPlaneError(
                message="command failed: sbatch",
                code=308,
                http_status=500,
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

    def _build_profiles(self, profile_ids: list[int]) -> list[EmbeddedProfileLayout]:
        profiles: list[EmbeddedProfileLayout] = []
        for gpu_index, profile_id in enumerate(profile_ids):
            service_profile = self._cfg.port_profiles.get(profile_id)
            if service_profile is None:
                raise ControlPlaneError(
                    message=f"missing vLLM/Jaeger port profile '{profile_id}' in configs/port_profiles.toml",
                    code=309,
                    http_status=500,
                )
            gateway_profile = load_gateway_port_profile(profile_id)
            profiles.append(
                EmbeddedProfileLayout(
                    profile_id=profile_id,
                    vllm_port=service_profile.vllm_port,
                    jaeger_api_port=service_profile.jaeger_api_port,
                    jaeger_otlp_port=service_profile.jaeger_otlp_port,
                    gateway_port=gateway_profile.gateway_port,
                    gateway_parse_port=gateway_profile.gateway_parse_port,
                    lmcache_port=service_profile.lmcache_port,
                    rocr_visible_devices=str(gpu_index),
                )
            )
        return profiles

    def _write_sbatch_script(
        self,
        *,
        partition: str,
        model: Any,
        profiles: list[EmbeddedProfileLayout],
        experiment_script: Path,
        extra_env: dict[str, str],
        lmcache_enabled: bool,
        extra_vllm_args: list[str],
        no_async_scheduling: bool,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_partition = _safe_token(partition)
        safe_model = _safe_token(model.name)
        script_path = (
            self._cfg.run_dir
            / f"sbatch-{timestamp}-{safe_partition}-{safe_model}-embedded-tp1.sh"
        )

        tmp_root = os.environ.get("TMPDIR", "/tmp")
        user_name = os.environ.get("USER", "user")
        aiter_jit_dir = self._cfg.env.get("AITER_JIT_DIR", f"{tmp_root}/vllm-aiter-jit-{user_name}")
        runtime_root = self._cfg.env.get("VLLM_RUNTIME_ROOT", f"{tmp_root}/vllm-runtime-{user_name}")
        xdg_cache_home = self._cfg.env.get("XDG_CACHE_HOME", f"{runtime_root}/xdg-cache")
        vllm_cache_root = self._cfg.env.get("VLLM_CACHE_ROOT", f"{xdg_cache_home}/vllm")
        effective_vllm_sif = self._cfg.partitions[partition].vllm_sif or self._cfg.vllm_sif
        effective_extra_args = _effective_vllm_extra_args(
            extra_args=[*model.extra_args, *extra_vllm_args],
            gpus_per_node=1,
        )
        encoded_extra_args = _encode_model_extra_args(effective_extra_args)
        force_seq_trust_remote_code = self._cfg.env.get("VLLM_FORCE_SEQ_TRUST_REMOTE_CODE", "true")
        extra_apptainer_env_flags = _render_apptainer_extra_env_flags(
            extra_env=extra_env,
            indent="        ",
        )
        lmcache_kv_transfer_args = ""
        if lmcache_enabled:
            lmcache_kv_transfer_args = (
                "      --kv-transfer-config\n"
                "      '{\"kv_connector\":\"LMCacheConnectorV1\", \"kv_role\":\"kv_both\"}'\n"
            )
        no_async_scheduling_args = ""
        if no_async_scheduling:
            no_async_scheduling_args = "      --no-async-scheduling\n"

        gateway_config_default = self._cfg.repo_root / "gateway" / "config.toml"
        gateway_config_fallback = self._cfg.repo_root / "gateway" / "config.example.toml"
        gateway_venv_dir_default = self._cfg.repo_root / ".venv"
        service_ready_timeout_seconds = int(max(1, self._cfg.startup_timeout))
        service_ready_poll_interval_seconds = float(max(0.2, self._cfg.wait_up_poll_interval_seconds))

        launch_vllm_lines = "\n".join(
            f"launch_vllm {profile.profile_id} {profile.vllm_port} {profile.lmcache_port} {profile.rocr_visible_devices}"
            for profile in profiles
        )
        wait_vllm_lines = "\n".join(
            f"wait_for_vllm_ready {profile.profile_id} {profile.vllm_port}"
            for profile in profiles
        )
        launch_gateway_lines = "\n".join(
            f"launch_gateway {profile.profile_id} {profile.gateway_port} {profile.gateway_parse_port}"
            for profile in profiles
        )
        wait_gateway_lines = "\n".join(
            f"wait_for_gateway_ready {profile.profile_id} {profile.gateway_port} {profile.gateway_parse_port}"
            for profile in profiles
        )
        launch_experiment_lines = "\n".join(
            f"launch_experiment {profile.profile_id}"
            for profile in profiles
        )

        script = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name={_safe_token(f"{self._cfg.job_name_prefix}{safe_partition}_tp1")}
            #SBATCH --output={self._cfg.log_dir / "slurm.%j.out"}
            #SBATCH --error={self._cfg.log_dir / "slurm.%j.err"}
            #SBATCH --nodes=1
            #SBATCH --time={self._cfg.partitions[partition].max_time}
            #SBATCH --partition={partition}

            set -euo pipefail

            echo "Embedded TP1 job ${{SLURM_JOB_ID}} starting on $(hostname) at $(date)"

            REPO_ROOT={shlex.quote(str(self._cfg.repo_root))}
            cd "${{REPO_ROOT}}"

            EXPERIMENT_SCRIPT={shlex.quote(str(experiment_script))}
            EXPERIMENT_RUNNER="${{EXPERIMENT_RUNNER:-bash}}"
            SERVICE_READY_TIMEOUT_SECONDS={service_ready_timeout_seconds}
            SERVICE_READY_POLL_INTERVAL_SECONDS={service_ready_poll_interval_seconds}
            JAEGER_OTLP_LOCAL_PORT=4317
            JAEGER_UI_LOCAL_PORT=16686

            JAEGER_SIF={shlex.quote(str(self._cfg.jaeger_sif))}
            VLLM_SIF={shlex.quote(str(effective_vllm_sif))}
            VLLM_MODEL_NAME={shlex.quote(model.vllm_model_name)}
            VLLM_SERVED_MODEL_NAME={shlex.quote(model.served_model_name)}
            VLLM_TENSOR_PARALLEL_SIZE=1

            VLLM_APPTAINER_HOME={shlex.quote(self._cfg.env.get("VLLM_APPTAINER_HOME", ""))}
            HF_HOME={shlex.quote(self._cfg.env.get("HF_HOME", ""))}
            HF_HUB_CACHE={shlex.quote(self._cfg.env.get("HF_HUB_CACHE", ""))}
            HF_HUB_OFFLINE=1
            TRANSFORMERS_OFFLINE=1
            HF_TOKEN="${{HF_TOKEN:-}}"
            export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

            AITER_JIT_DIR={shlex.quote(aiter_jit_dir)}
            XDG_CACHE_HOME={shlex.quote(xdg_cache_home)}
            VLLM_CACHE_ROOT={shlex.quote(vllm_cache_root)}

            OTEL_SERVICE_NAME={shlex.quote(self._cfg.env.get("OTEL_SERVICE_NAME", "vllm-server"))}
            OTEL_EXPORTER_OTLP_TRACES_INSECURE={shlex.quote(self._cfg.env.get("OTEL_EXPORTER_OTLP_TRACES_INSECURE", "true"))}
            OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}"
            VLLM_COLLECT_DETAILED_TRACES={shlex.quote(self._cfg.env.get("VLLM_COLLECT_DETAILED_TRACES", "all"))}
            VLLM_LOGITS_PROCESSORS={shlex.quote(self._cfg.env.get("VLLM_LOGITS_PROCESSORS", "forceSeq.force_sequence_logits_processor:ForceSequenceAdapter"))}
            VLLM_MODEL_EXTRA_ARGS_B64={shlex.quote(encoded_extra_args)}
            VLLM_FORCE_SEQ_TRUST_REMOTE_CODE={shlex.quote(force_seq_trust_remote_code)}

            GATEWAY_CONFIG_DEFAULT={shlex.quote(str(gateway_config_default))}
            GATEWAY_CONFIG_FALLBACK={shlex.quote(str(gateway_config_fallback))}
            GATEWAY_VENV_DIR="${{GATEWAY_VENV_DIR:-{shlex.quote(str(gateway_venv_dir_default))}}}"
            GATEWAY_HOST="${{GATEWAY_HOST:-127.0.0.1}}"
            GATEWAY_SKIP_INSTALL="${{GATEWAY_SKIP_INSTALL:-1}}"

            JOB_LOG_DIR={shlex.quote(str(self._cfg.log_dir))}
            mkdir -p "${{JOB_LOG_DIR}}" "${{AITER_JIT_DIR}}" "${{XDG_CACHE_HOME}}" "${{VLLM_CACHE_ROOT}}"

            JAEGER_LOG_SHARED="${{JOB_LOG_DIR}}/jaeger.${{SLURM_JOB_ID}}.shared.log"

            declare -A PROFILE_VLLM_PID=()
            declare -A PROFILE_GATEWAY_PID=()
            declare -A PROFILE_EXPERIMENT_PID=()
            declare -A PROFILE_EXPERIMENT_EXIT_CODE=()
            declare -A PROFILE_VLLM_PORT=()
            declare -A PROFILE_GATEWAY_PORT=()
            declare -A PROFILE_GATEWAY_PARSE_PORT=()
            declare -A PROFILE_VLLM_LOG=()
            declare -A PROFILE_GATEWAY_LOG=()
            declare -A PROFILE_EXPERIMENT_LOG=()

            PROFILE_IDS=({" ".join(str(profile.profile_id) for profile in profiles)})

            APPTAINER_HOME_ARGS=()
            if [[ -n "${{VLLM_APPTAINER_HOME}}" ]]; then
              mkdir -p "${{VLLM_APPTAINER_HOME}}"
              APPTAINER_HOME_ARGS=(-H "${{VLLM_APPTAINER_HOME}}")
            fi

            BIND_ARGS=()
            if [[ -n "${{HF_HOME}}" ]]; then
              BIND_ARGS+=(--bind "${{HF_HOME}}:${{HF_HOME}}")
            fi
            if [[ -n "${{HF_HUB_CACHE}}" ]]; then
              BIND_ARGS+=(--bind "${{HF_HUB_CACHE}}:${{HF_HUB_CACHE}}")
            fi

            probe_http_url() {{
              local url="$1"
              python3 -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
            }}

            probe_tcp_port() {{
              local host="$1"
              local port="$2"
              python3 -c "import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()" "$host" "$port" >/dev/null 2>&1
            }}

            terminate_process() {{
              local name="$1"
              local pid="$2"
              if [[ -z "${{pid}}" ]]; then
                return 0
              fi
              if ! kill -0 "${{pid}}" >/dev/null 2>&1; then
                return 0
              fi
              echo "Stopping ${{name}} (pid=${{pid}})"
              kill "${{pid}}" >/dev/null 2>&1 || true
              local deadline=$((SECONDS + 20))
              while kill -0 "${{pid}}" >/dev/null 2>&1; do
                if (( SECONDS >= deadline )); then
                  kill -9 "${{pid}}" >/dev/null 2>&1 || true
                  break
                fi
                sleep 1
              done
              wait "${{pid}}" >/dev/null 2>&1 || true
            }}

            cleanup() {{
              set +e
              for profile_id in "${{PROFILE_IDS[@]}}"; do
                terminate_process "experiment profile=${{profile_id}}" "${{PROFILE_EXPERIMENT_PID[$profile_id]:-}}"
              done
              for profile_id in "${{PROFILE_IDS[@]}}"; do
                terminate_process "gateway profile=${{profile_id}}" "${{PROFILE_GATEWAY_PID[$profile_id]:-}}"
              done
              for profile_id in "${{PROFILE_IDS[@]}}"; do
                terminate_process "vllm profile=${{profile_id}}" "${{PROFILE_VLLM_PID[$profile_id]:-}}"
              done
              terminate_process "jaeger" "${{JAEGER_PID:-}}"
            }}
            trap cleanup EXIT INT TERM

            start_shared_jaeger() {{
              apptainer run \
                --cleanenv \
                "${{APPTAINER_HOME_ARGS[@]}}" \
                --env COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
                "${{JAEGER_SIF}}" \
                >"${{JAEGER_LOG_SHARED}}" 2>&1 &
              JAEGER_PID=$!
              sleep 1
              if ! kill -0 "${{JAEGER_PID}}" >/dev/null 2>&1; then
                wait "${{JAEGER_PID}}" >/dev/null 2>&1 || true
                echo "Shared Jaeger failed to start. See ${{JAEGER_LOG_SHARED}}" >&2
                exit 71
              fi
            }}

            launch_vllm() {{
              local profile_id="$1"
              local vllm_port="$2"
              local lmcache_port="$3"
              local rocr_visible_devices="$4"
              local vllm_log="${{JOB_LOG_DIR}}/vllm.${{SLURM_JOB_ID}}.p${{profile_id}}.log"
              local otel_service_name_worker="${{OTEL_SERVICE_NAME}}-p${{profile_id}}"
              local vllm_cmd=(
                /opt/vllm-plugins/vllm_entrypoint.sh
                --model "${{VLLM_MODEL_NAME}}"
                --served-model-name "${{VLLM_SERVED_MODEL_NAME}}"
                --port "${{vllm_port}}"
                --tensor-parallel-size "${{VLLM_TENSOR_PARALLEL_SIZE}}"
                --otlp-traces-endpoint "${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}"
                --collect-detailed-traces "${{VLLM_COLLECT_DETAILED_TRACES}}"
                --enable-prompt-tokens-details
                --logits-processors "${{VLLM_LOGITS_PROCESSORS}}"
{lmcache_kv_transfer_args}{no_async_scheduling_args}              )

              PROFILE_VLLM_PORT["${{profile_id}}"]="${{vllm_port}}"
              PROFILE_VLLM_LOG["${{profile_id}}"]="${{vllm_log}}"

              echo "Launching vLLM profile=${{profile_id}} port=${{vllm_port}} gpu=${{rocr_visible_devices}}"
              apptainer exec \
                --rocm \
                --cleanenv \
                "${{APPTAINER_HOME_ARGS[@]}}" \
                "${{BIND_ARGS[@]}}" \
                --env PYTHONNOUSERSITE=1 \
                --env AITER_JIT_DIR="${{AITER_JIT_DIR}}" \
                --env XDG_CACHE_HOME="${{XDG_CACHE_HOME}}" \
                --env VLLM_CACHE_ROOT="${{VLLM_CACHE_ROOT}}" \
                --env HF_HOME="${{HF_HOME}}" \
                --env HF_HUB_CACHE="${{HF_HUB_CACHE}}" \
                --env HF_HUB_OFFLINE="${{HF_HUB_OFFLINE}}" \
                --env TRANSFORMERS_OFFLINE="${{TRANSFORMERS_OFFLINE}}" \
                --env HF_TOKEN="${{HF_TOKEN}}" \
                --env OTEL_SERVICE_NAME="${{otel_service_name_worker}}" \
                --env OTEL_EXPORTER_OTLP_TRACES_INSECURE="${{OTEL_EXPORTER_OTLP_TRACES_INSECURE}}" \
                --env OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${{OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}}" \
                --env HIP_VISIBLE_DEVICE=0 \
                --env HIP_VISIBLE_DEVICES=0 \
                --env ROCR_VISIBLE_DEVICES="${{rocr_visible_devices}}" \
                --env VLLM_MODEL_NAME="${{VLLM_MODEL_NAME}}" \
                --env VLLM_MODEL_EXTRA_ARGS_B64="${{VLLM_MODEL_EXTRA_ARGS_B64}}" \
                --env VLLM_FORCE_SEQ_TRUST_REMOTE_CODE="${{VLLM_FORCE_SEQ_TRUST_REMOTE_CODE}}" \
{extra_apptainer_env_flags}                --env LMCACHE_INTERNAL_API_SERVER_ENABLED=1 \
                --env PYTHONHASHSEED=0 \
                --env LMCACHE_INTERNAL_API_SERVER_PORT_START="${{lmcache_port}}" \
                "${{VLLM_SIF}}" \
                "${{vllm_cmd[@]}}" \
                >"${{vllm_log}}" 2>&1 &
              PROFILE_VLLM_PID["${{profile_id}}"]=$!
            }}

            wait_for_vllm_ready() {{
              local profile_id="$1"
              local vllm_port="$2"
              local pid="${{PROFILE_VLLM_PID[$profile_id]}}"
              local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
              local url="http://127.0.0.1:${{vllm_port}}/v1/models"
              while (( SECONDS < deadline )); do
                if ! kill -0 "${{pid}}" >/dev/null 2>&1; then
                  echo "vLLM for profile ${{profile_id}} exited before readiness. See ${{PROFILE_VLLM_LOG[$profile_id]}}" >&2
                  return 1
                fi
                if probe_http_url "${{url}}"; then
                  return 0
                fi
                sleep "${{SERVICE_READY_POLL_INTERVAL_SECONDS}}"
              done
              echo "Timed out waiting for vLLM readiness for profile ${{profile_id}}." >&2
              return 1
            }}

            launch_gateway() {{
              local profile_id="$1"
              local gateway_port="$2"
              local gateway_parse_port="$3"
              local gateway_log="${{JOB_LOG_DIR}}/gateway.${{SLURM_JOB_ID}}.p${{profile_id}}.log"
              local gateway_config_path=""
              local gateway_python="python3"
              if [[ -n "${{GATEWAY_CONFIG:-}}" ]]; then
                gateway_config_path="${{GATEWAY_CONFIG}}"
              elif [[ -f "${{GATEWAY_CONFIG_DEFAULT}}" ]]; then
                gateway_config_path="${{GATEWAY_CONFIG_DEFAULT}}"
              else
                gateway_config_path="${{GATEWAY_CONFIG_FALLBACK}}"
              fi
              if [[ -x "${{GATEWAY_VENV_DIR}}/bin/python" ]]; then
                gateway_python="${{GATEWAY_VENV_DIR}}/bin/python"
              fi

              PROFILE_GATEWAY_PORT["${{profile_id}}"]="${{gateway_port}}"
              PROFILE_GATEWAY_PARSE_PORT["${{profile_id}}"]="${{gateway_parse_port}}"
              PROFILE_GATEWAY_LOG["${{profile_id}}"]="${{gateway_log}}"

              local gateway_cmd=(
                "${{gateway_python}}"
                -m gateway
                start
                --config "${{gateway_config_path}}"
                --host "${{GATEWAY_HOST}}"
                --venv-dir "${{GATEWAY_VENV_DIR}}"
                --port-profile-id "${{profile_id}}"
              )
              if [[ "${{GATEWAY_SKIP_INSTALL}}" == "1" ]]; then
                gateway_cmd+=(--skip-install)
              fi

              echo "Launching gateway profile=${{profile_id}} raw=${{gateway_port}} parsed=${{gateway_parse_port}}"
              GATEWAY_JAEGER_API_BASE_URL_OVERRIDE="http://127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}/api/traces" \
              GATEWAY_OTLP_TRACES_ENDPOINT_OVERRIDE="grpc://127.0.0.1:${{JAEGER_OTLP_LOCAL_PORT}}" \
                "${{gateway_cmd[@]}}" >"${{gateway_log}}" 2>&1 &
              PROFILE_GATEWAY_PID["${{profile_id}}"]=$!
            }}

            wait_for_gateway_ready() {{
              local profile_id="$1"
              local gateway_port="$2"
              local gateway_parse_port="$3"
              local pid="${{PROFILE_GATEWAY_PID[$profile_id]}}"
              local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
              while (( SECONDS < deadline )); do
                if ! kill -0 "${{pid}}" >/dev/null 2>&1; then
                  echo "Gateway for profile ${{profile_id}} exited before readiness. See ${{PROFILE_GATEWAY_LOG[$profile_id]}}" >&2
                  return 1
                fi
                if [[ "${{gateway_port}}" -eq "${{gateway_parse_port}}" ]]; then
                  if probe_tcp_port "127.0.0.1" "${{gateway_port}}"; then
                    return 0
                  fi
                else
                  if probe_tcp_port "127.0.0.1" "${{gateway_port}}" && probe_tcp_port "127.0.0.1" "${{gateway_parse_port}}"; then
                    return 0
                  fi
                fi
                sleep 1
              done
              echo "Timed out waiting for gateway readiness for profile ${{profile_id}}." >&2
              return 1
            }}

            launch_experiment() {{
              local profile_id="$1"
              local experiment_log="${{JOB_LOG_DIR}}/experiment.${{SLURM_JOB_ID}}.p${{profile_id}}.log"
              PROFILE_EXPERIMENT_LOG["${{profile_id}}"]="${{experiment_log}}"
              echo "Launching experiment profile=${{profile_id}} script=${{EXPERIMENT_SCRIPT}}"
              (
                export PORT_PROFILE_ID="${{profile_id}}"
                export VLLM_BASE_URL="http://127.0.0.1:${{PROFILE_VLLM_PORT[$profile_id]}}"
                export GATEWAY_BASE_URL="http://127.0.0.1:${{PROFILE_GATEWAY_PORT[$profile_id]}}"
                export GATEWAY_PARSE_BASE_URL="http://127.0.0.1:${{PROFILE_GATEWAY_PARSE_PORT[$profile_id]}}"
                export JAEGER_BASE_URL="http://127.0.0.1:${{JAEGER_UI_LOCAL_PORT}}"
                "${{EXPERIMENT_RUNNER}}" "${{EXPERIMENT_SCRIPT}}" "${{profile_id}}"
              ) >"${{experiment_log}}" 2>&1 &
              PROFILE_EXPERIMENT_PID["${{profile_id}}"]=$!
            }}

            wait_for_experiment_phase() {{
              local first_failure=0
              while true; do
                local active_experiments=0
                if ! kill -0 "${{JAEGER_PID}}" >/dev/null 2>&1; then
                  echo "Shared Jaeger exited during experiment phase." >&2
                  return 91
                fi
                for profile_id in "${{PROFILE_IDS[@]}}"; do
                  local vllm_pid="${{PROFILE_VLLM_PID[$profile_id]:-}}"
                  local gateway_pid="${{PROFILE_GATEWAY_PID[$profile_id]:-}}"
                  local experiment_pid="${{PROFILE_EXPERIMENT_PID[$profile_id]:-}}"
                  if [[ -n "${{vllm_pid}}" ]] && ! kill -0 "${{vllm_pid}}" >/dev/null 2>&1; then
                    echo "vLLM for profile ${{profile_id}} exited during experiment phase." >&2
                    return 92
                  fi
                  if [[ -n "${{gateway_pid}}" ]] && ! kill -0 "${{gateway_pid}}" >/dev/null 2>&1; then
                    echo "Gateway for profile ${{profile_id}} exited during experiment phase." >&2
                    return 93
                  fi
                  if [[ -n "${{experiment_pid}}" ]]; then
                    if kill -0 "${{experiment_pid}}" >/dev/null 2>&1; then
                      active_experiments=1
                    elif [[ -z "${{PROFILE_EXPERIMENT_EXIT_CODE[$profile_id]:-}}" ]]; then
                      if wait "${{experiment_pid}}"; then
                        PROFILE_EXPERIMENT_EXIT_CODE["${{profile_id}}"]=0
                      else
                        PROFILE_EXPERIMENT_EXIT_CODE["${{profile_id}}"]=$?
                      fi
                      if [[ "${{PROFILE_EXPERIMENT_EXIT_CODE[$profile_id]}}" -ne 0 && "${{first_failure}}" -eq 0 ]]; then
                        first_failure="${{PROFILE_EXPERIMENT_EXIT_CODE[$profile_id]}}"
                      fi
                    fi
                  fi
                done
                if [[ "${{active_experiments}}" -eq 0 ]]; then
                  return "${{first_failure}}"
                fi
                sleep 2
              done
            }}

            start_shared_jaeger
{textwrap.indent(launch_vllm_lines, "            ")}
{textwrap.indent(wait_vllm_lines, "            ")}
{textwrap.indent(launch_gateway_lines, "            ")}
{textwrap.indent(wait_gateway_lines, "            ")}
{textwrap.indent(launch_experiment_lines, "            ")}

            echo "All services are ready; waiting for experiment completion."
            EXPERIMENT_PHASE_EXIT_CODE=0
            if ! wait_for_experiment_phase; then
              EXPERIMENT_PHASE_EXIT_CODE=$?
            fi
            echo "Experiment phase finished with exit code ${{EXPERIMENT_PHASE_EXIT_CODE}} at $(date)"
            exit "${{EXPERIMENT_PHASE_EXIT_CODE}}"
            """
        )

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)
        return script_path

    def _extract_sbatch_job_id(self, text: str) -> str:
        match = re.search(r"Submitted\\s+batch\\s+job\\s+(\\d+)", text)
        if match is None:
            raise ControlPlaneError(
                message="unable to parse sbatch job id",
                code=310,
                http_status=500,
                details={"sbatch_output": text.strip()},
            )
        return match.group(1)

