#!/usr/bin/env python3
"""Generate one local sweep-QPS + power logging + gateway_multi balanced instance replay bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT
    / "experiments"
    / "sweep-qps-docker-power-clean-gateway_multi-balanced-freq-ctrl-linesapce-instance-ctx-aware"
    / "generated"
)
DEFAULT_REPLAY_OUTPUT_ROOT = (
    Path("results")
    / "replay"
    / "sweep-qps-docker-power-clean-gateway_multi-balanced-freq-ctrl-linesapce-instance-ctx-aware"
)
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
EXPERIMENT_DIR_NAME = (
    "sweep-qps-docker-power-clean-gateway_multi-balanced-freq-ctrl-linesapce-instance-ctx-aware"
)
EXPERIMENT_LOG_TAG = EXPERIMENT_DIR_NAME
DEFAULT_ASSIGNMENT_POLICY = "balanced"
DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS = 263_856
DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS = 501280
DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS = 474897
CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS = 3000
DEFAULT_FREQ_CONTROLLER_THRESHOLD = 395784.0
DEFAULT_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace-instance"
DEFAULT_FREQ_CONTROLLER_BIN = "freq-controller-linespace-instance"
_BASE_GENERATOR_MODULE_NAME = "generate_sweep_qps_docker_power_clean_experiment"


@dataclass(frozen=True)
class BackendFreqControllerTarget:
    port_profile_id: int
    gpu_index: int


@dataclass(frozen=True)
class QpsReplayJob:
    qps: float
    qps_display: str
    qps_slug: str
    replay_config_relpath: str
    replay_config_path: Path
    replay_output_dir: Path
    power_output_dir: Path
    freq_controller_log_dir: Path


def _load_base_generator_module() -> object:
    module_path = (
        REPO_ROOT
        / "experiments"
        / "sweep-qps-docker-power-clean"
        / "generate_experiment.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        _BASE_GENERATOR_MODULE_NAME,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base generator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_GENERATOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_generator_module()


def _sync_base_module_globals() -> None:
    BASE.REPO_ROOT = REPO_ROOT
    BASE.RESULTS_ROOT = RESULTS_ROOT
    BASE.MODEL_CONFIG_PATH = MODEL_CONFIG_PATH


def build_generation_command_raw(argv: list[str] | None) -> str:
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])
    return shlex.join(
        [str(sys.executable), str(Path(__file__).resolve()), *[str(token) for token in argv]]
    )


def parse_port_profile_ids(raw_values: Sequence[str]) -> list[int]:
    parsed: list[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            stripped = part.strip()
            if not stripped:
                continue
            try:
                profile_id = int(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid --port-profile id: {stripped}") from exc
            if profile_id < 0:
                raise ValueError(f"invalid --port-profile id: {stripped}")
            if profile_id in seen:
                raise ValueError(f"duplicate --port-profile id: {profile_id}")
            parsed.append(profile_id)
            seen.add(profile_id)
    if not parsed:
        raise ValueError("at least one --port-profile id is required")
    return parsed


def parse_positive_int(value: str, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def safe_name(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars) or "value"


def normalize_output_suffix(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        raise ValueError("--output-suffix cannot be empty")
    return safe_name(stripped)


def apply_output_suffix_to_replay_root(root: Path, output_suffix: str | None) -> Path:
    if output_suffix is None:
        return root
    if not root.name:
        raise ValueError("--output-suffix cannot be applied to an empty replay output root")
    return root.parent / f"{root.name}-{output_suffix}"


def build_backend_freq_controller_targets(
    *,
    port_profile_ids: Sequence[int],
    gpu_indices: Sequence[int],
) -> list[BackendFreqControllerTarget]:
    if len(port_profile_ids) != len(gpu_indices):
        raise ValueError(
            "--power-gpu-indices must contain one GPU index per selected "
            "--port-profile when using independent freq-controller-linespace-instance"
        )
    return [
        BackendFreqControllerTarget(port_profile_id=port_profile_id, gpu_index=gpu_index)
        for port_profile_id, gpu_index in zip(port_profile_ids, gpu_indices)
    ]


def freq_controller_backend_log_dir_name(port_profile_id: int) -> str:
    return f"profile-{port_profile_id}"


def path_for_config(path: Path) -> str:
    _sync_base_module_globals()
    return BASE.path_for_config(path)


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def write_run_script(
    path: Path,
    *,
    default_port_profile_ids: list[int],
    target_model: str,
    split: str,
    power_gpu_indices: list[int],
    qps_jobs: list[QpsReplayJob],
    balanced_usage_threshold_tokens: int,
    ctx_aware_usage_threshold_tokens: int,
    ctx_aware_scheduling_threshold_tokens: int,
    freq_controller_threshold: float,
) -> None:
    default_port_profile_ids_csv = ",".join(
        str(profile_id) for profile_id in default_port_profile_ids
    )
    power_gpu_indices_payload = " ".join(str(gpu_index) for gpu_index in power_gpu_indices)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"DEFAULT_PORT_PROFILE_IDS_CSV={_shell_quote(default_port_profile_ids_csv)}",
        "PORT_PROFILE_IDS_VALUE=\"${1:-${PORT_PROFILE_IDS:-${DEFAULT_PORT_PROFILE_IDS_CSV}}}\"",
        "PYTHON_BIN=\"${PYTHON_BIN:-python3}\"",
        "CURL_BIN=\"${CURL_BIN:-curl}\"",
        "ZEUS_POWER_READER_BIN=\"${ZEUS_POWER_READER_BIN:-zeus-power-reader}\"",
        f"FREQ_CONTROLLER_BIN=\"${{FREQ_CONTROLLER_BIN:-{DEFAULT_FREQ_CONTROLLER_BIN}}}\"",
        "RESET_GPU_CORE_FREQ_BIN=\"${RESET_GPU_CORE_FREQ_BIN:-reset-gpu-core-freq}\"",
        "ZEUSD_SOCKET_PATH_VALUE=\"${ZEUSD_SOCKET_PATH:-}\"",
        "GATEWAY_BASE_URL_VALUE=\"${GATEWAY_BASE_URL:-}\"",
        f"TARGET_ASSIGNMENT_POLICY={_shell_quote(DEFAULT_ASSIGNMENT_POLICY)}",
        f"DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS={balanced_usage_threshold_tokens}",
        (
            "BALANCED_USAGE_THRESHOLD_TOKENS_VALUE="
            "\"${BALANCED_USAGE_THRESHOLD_TOKENS:-${DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS}}\""
        ),
        "FREQ_CONTROLLER_CONFIG_VALUE=\"${FREQ_CONTROLLER_CONFIG:-}\"",
        f"DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS={ctx_aware_usage_threshold_tokens}",
        (
            "DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS="
            f"{ctx_aware_scheduling_threshold_tokens}"
        ),
        (
            "CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE="
            "\"${CTX_AWARE_USAGE_THRESHOLD_TOKENS:-${DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS}}\""
        ),
        (
            "CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE="
            "\"${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS:-${DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS}}\""
        ),
        f"DEFAULT_FREQ_CONTROLLER_THRESHOLD={freq_controller_threshold}",
        (
            "FREQ_CONTROLLER_THRESHOLD_VALUE="
            "\"${FREQ_CONTROLLER_THRESHOLD:-${DEFAULT_FREQ_CONTROLLER_THRESHOLD}}\""
        ),
        f"POWER_GPU_INDICES=({power_gpu_indices_payload})",
        "",
        "POWER_READER_PID=\"\"",
        "CONTROL_PORT_PROFILE_ID=\"\"",
        "PORT_PROFILE_IDS_CSV_NORMALIZED=\"\"",
        "PORT_PROFILE_IDS_DISPLAY=\"\"",
        "GATEWAY_BASE_URL_RESOLVED=\"\"",
        "CTX_AWARE_STARTED=0",
        "FREQ_CONTROLLER_STARTED=0",
        "BACKEND_PAIRINGS_DISPLAY=\"\"",
        "declare -a PORT_PROFILE_IDS_ARRAY=()",
        "declare -a FREQ_CONTROLLER_PIDS=()",
        "",
        "stop_power_reader() {",
        "  if [[ -n \"${POWER_READER_PID}\" ]]; then",
        "    kill \"${POWER_READER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${POWER_READER_PID}\" 2>/dev/null || true",
        "    POWER_READER_PID=\"\"",
        "  fi",
        "}",
        "",
        "stop_freq_controllers() {",
        "  local pid=\"\"",
        "  for pid in \"${FREQ_CONTROLLER_PIDS[@]}\"; do",
        "    [[ -n \"${pid}\" ]] || continue",
        "    kill \"${pid}\" >/dev/null 2>&1 || true",
        "  done",
        "  for pid in \"${FREQ_CONTROLLER_PIDS[@]}\"; do",
        "    [[ -n \"${pid}\" ]] || continue",
        "    wait \"${pid}\" 2>/dev/null || true",
        "  done",
        "  FREQ_CONTROLLER_PIDS=()",
        "}",
        "",
        "reset_gpu_core_if_needed() {",
        "  if [[ \"${FREQ_CONTROLLER_STARTED}\" -eq 1 ]]; then",
        "    local gpu_index=\"\"",
        "    for gpu_index in \"${POWER_GPU_INDICES[@]}\"; do",
        "      if ! \"${RESET_GPU_CORE_FREQ_BIN}\" --gpu-index \"${gpu_index}\"; then",
        (
            f"        echo \"[{EXPERIMENT_LOG_TAG}] warning: failed to reset GPU "
            "${gpu_index} core clocks\" >&2"
        ),
        "      fi",
        "    done",
        "    FREQ_CONTROLLER_STARTED=0",
        "  fi",
        "}",
        "",
        "parse_port_profile_ids() {",
        "  local raw_value=\"$1\"",
        "  local normalized_value=\"${raw_value//[[:space:]]/}\"",
        "  local -a raw_parts=()",
        "  local part=\"\"",
        "  local seen_csv=\",\"",
        "",
        "  if [[ -z \"${normalized_value}\" ]]; then",
        f"    echo \"[{EXPERIMENT_LOG_TAG}] error: port profile list cannot be empty\" >&2",
        "    exit 1",
        "  fi",
        "",
        "  PORT_PROFILE_IDS_ARRAY=()",
        "  IFS=',' read -r -a raw_parts <<< \"${normalized_value}\"",
        "  for part in \"${raw_parts[@]}\"; do",
        "    [[ -n \"${part}\" ]] || continue",
        "    if [[ ! \"${part}\" =~ ^[0-9]+$ ]]; then",
        f"      echo \"[{EXPERIMENT_LOG_TAG}] error: invalid port profile id: ${{part}}\" >&2",
        "      exit 1",
        "    fi",
        "    if [[ \"${seen_csv}\" == *\",${part},\"* ]]; then",
        f"      echo \"[{EXPERIMENT_LOG_TAG}] error: duplicate port profile id: ${{part}}\" >&2",
        "      exit 1",
        "    fi",
        "    seen_csv+=\"${part},\"",
        "    PORT_PROFILE_IDS_ARRAY+=(\"${part}\")",
        "  done",
        "",
        "  if [[ \"${#PORT_PROFILE_IDS_ARRAY[@]}\" -eq 0 ]]; then",
        f"    echo \"[{EXPERIMENT_LOG_TAG}] error: port profile list cannot be empty\" >&2",
        "    exit 1",
        "  fi",
        "",
        "  CONTROL_PORT_PROFILE_ID=\"${PORT_PROFILE_IDS_ARRAY[0]}\"",
        "  PORT_PROFILE_IDS_DISPLAY=\"${PORT_PROFILE_IDS_ARRAY[*]}\"",
        "  PORT_PROFILE_IDS_CSV_NORMALIZED=\"$(IFS=,; printf '%s' \"${PORT_PROFILE_IDS_ARRAY[*]}\")\"",
        "}",
        "",
        "validate_backend_pairing() {",
        "  local idx=\"\"",
        "  local -a pairings=()",
        "  if [[ \"${#PORT_PROFILE_IDS_ARRAY[@]}\" -ne \"${#POWER_GPU_INDICES[@]}\" ]]; then",
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] error: port profile count must match "
            "power GPU count for independent freq-controller-linespace-instance\" >&2"
        ),
        "    exit 1",
        "  fi",
        "  for idx in \"${!PORT_PROFILE_IDS_ARRAY[@]}\"; do",
        "    pairings+=(\"profile-${PORT_PROFILE_IDS_ARRAY[$idx]}:gpu-${POWER_GPU_INDICES[$idx]}\")",
        "  done",
        "  BACKEND_PAIRINGS_DISPLAY=\"${pairings[*]}\"",
        "}",
        "",
        "resolve_gateway_base_url() {",
        "  if [[ -n \"${GATEWAY_BASE_URL_VALUE}\" ]]; then",
        "    printf '%s\\n' \"${GATEWAY_BASE_URL_VALUE}\"",
        "    return 0",
        "  fi",
        "  \"${PYTHON_BIN}\" - \"${REPO_ROOT}\" \"${CONTROL_PORT_PROFILE_ID}\" <<'PY'",
        "from pathlib import Path",
        "import sys",
        "try:",
        "    import tomllib",
        "except ModuleNotFoundError:  # pragma: no cover",
        "    import tomli as tomllib",
        "repo_root = Path(sys.argv[1]).resolve()",
        "port_profile_id = str(sys.argv[2])",
        "payload = tomllib.loads((repo_root / 'configs' / 'port_profiles.toml').read_text(encoding='utf-8'))",
        "profiles = payload.get('profiles')",
        "if not isinstance(profiles, dict):",
        "    raise SystemExit('configs/port_profiles.toml must include [profiles]')",
        "profile = profiles.get(port_profile_id)",
        "if not isinstance(profile, dict):",
        "    raise SystemExit(f'unknown port profile id: {port_profile_id}')",
        "gateway_port = profile.get('gateway_port')",
        "if isinstance(gateway_port, bool) or not isinstance(gateway_port, int):",
        "    raise SystemExit(f'invalid gateway_port for profile {port_profile_id}')",
        "print(f'http://127.0.0.1:{gateway_port}')",
        "PY",
        "}",
        "",
        "ensure_gateway_base_url() {",
        "  if [[ -z \"${GATEWAY_BASE_URL_RESOLVED}\" ]]; then",
        "    GATEWAY_BASE_URL_RESOLVED=\"$(resolve_gateway_base_url)\"",
        "  fi",
        "}",
        "",
        "set_gateway_assignment_policy() {",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] setting assignment_policy=${{TARGET_ASSIGNMENT_POLICY}} "
            "balanced_usage_threshold_tokens=${BALANCED_USAGE_THRESHOLD_TOKENS_VALUE} "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "control_port_profile=${CONTROL_PORT_PROFILE_ID}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    -H \"content-type: application/json\" \\",
        (
            "    --data "
            "\"{\\\"assignment_policy\\\": \\\"${TARGET_ASSIGNMENT_POLICY}\\\", "
            "\\\"balanced_usage_threshold_tokens\\\": ${BALANCED_USAGE_THRESHOLD_TOKENS_VALUE}}\" \\"
        ),
        "    \"${GATEWAY_BASE_URL_RESOLVED}/policy\"",
        "}",
        "",
        "start_ctx_aware_mode() {",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] enabling ctx-aware "
            "usage=${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE} "
            "scheduling=${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE} "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "control_port_profile=${CONTROL_PORT_PROFILE_ID}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    -H \"content-type: application/json\" \\",
        (
            "    --data "
            "\"{\\\"usage_threshold_tokens\\\": ${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE}, "
            "\\\"scheduling_threshold_tokens\\\": ${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE}}\" \\"
        ),
        "    \"${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/start\"",
        "  CTX_AWARE_STARTED=1",
        "}",
        "",
        "end_ctx_aware_mode() {",
        "  if [[ \"${CTX_AWARE_STARTED}\" -eq 0 ]]; then",
        "    return 0",
        "  fi",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] disabling ctx-aware "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "control_port_profile=${CONTROL_PORT_PROFILE_ID}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    \"${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/end\"",
        "  CTX_AWARE_STARTED=0",
        "}",
        "",
        "cleanup() {",
        "  local exit_code=\"$1\"",
        "  stop_freq_controllers",
        "  end_ctx_aware_mode || true",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "  return \"${exit_code}\"",
        "}",
        "",
        "trap '__exit_code=$?; trap - EXIT INT TERM; cleanup \"${__exit_code}\"; exit \"${__exit_code}\"' EXIT",
        f"trap 'echo \"[{EXPERIMENT_LOG_TAG}] interrupted\" >&2; exit 130' INT TERM",
        "",
        "parse_port_profile_ids \"${PORT_PROFILE_IDS_VALUE}\"",
        "validate_backend_pairing",
        "set_gateway_assignment_policy",
        "",
        "run_one_qps() {",
        "  local qps_value=\"$1\"",
        "  local qps_slug=\"$2\"",
        "  local replay_config_ref=\"$3\"",
        "  local replay_output_ref=\"$4\"",
        "  local replay_config_path=\"\"",
        "  local replay_output_dir=\"\"",
        "  local power_output_dir=\"\"",
        "  local freq_controller_log_root_dir=\"\"",
        "  local idx=\"\"",
        "  local profile_id=\"\"",
        "  local gpu_index=\"\"",
        "  local freq_controller_log_dir=\"\"",
        "  local freq_controller_pid=\"\"",
        "  local -a FREQ_CONTROLLER_CMD=()",
        "",
        "  if [[ \"${replay_config_ref}\" = /* ]]; then",
        "    replay_config_path=\"${replay_config_ref}\"",
        "  else",
        "    replay_config_path=\"${SCRIPT_DIR}/${replay_config_ref}\"",
        "  fi",
        "",
        "  if [[ \"${replay_output_ref}\" = /* ]]; then",
        "    replay_output_dir=\"${replay_output_ref}\"",
        "  else",
        "    replay_output_dir=\"${REPO_ROOT}/${replay_output_ref}\"",
        "  fi",
        "  power_output_dir=\"${replay_output_dir}/power\"",
        (
            "  freq_controller_log_root_dir="
            f"\"${{replay_output_dir}}/{DEFAULT_FREQ_CONTROL_LOG_DIR_NAME}\""
        ),
        "",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] qps=${{qps_value}} slug=${{qps_slug}} "
            "assignment_policy=${TARGET_ASSIGNMENT_POLICY} "
            "usage=${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE} "
            "scheduling=${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE} "
            "threshold=${FREQ_CONTROLLER_THRESHOLD_VALUE} "
            "output=${replay_output_ref} "
            "backend_pairings=${BACKEND_PAIRINGS_DISPLAY} "
            "control_port_profile=${CONTROL_PORT_PROFILE_ID}\""
        ),
        "  mkdir -p \"${power_output_dir}\" \"${freq_controller_log_root_dir}\"",
        "  if [[ -n \"${ZEUSD_SOCKET_PATH_VALUE}\" ]]; then",
        "    \"${ZEUS_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${POWER_GPU_INDICES[@]}\" --socket-path \"${ZEUSD_SOCKET_PATH_VALUE}\" &",
        "  else",
        "    \"${ZEUS_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${POWER_GPU_INDICES[@]}\" &",
        "  fi",
        "  POWER_READER_PID=\"$!\"",
        "",
        "  start_ctx_aware_mode",
        "",
        "  FREQ_CONTROLLER_PIDS=()",
        "  for idx in \"${!PORT_PROFILE_IDS_ARRAY[@]}\"; do",
        "    profile_id=\"${PORT_PROFILE_IDS_ARRAY[$idx]}\"",
        "    gpu_index=\"${POWER_GPU_INDICES[$idx]}\"",
        "    freq_controller_log_dir=\"${freq_controller_log_root_dir}/profile-${profile_id}\"",
        "    mkdir -p \"${freq_controller_log_dir}\"",
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] starting freq-controller-linespace-instance "
            "port_profile=${profile_id} gpu_index=${gpu_index} log_dir=${freq_controller_log_dir}\""
        ),
        "    FREQ_CONTROLLER_CMD=(",
        "      \"${FREQ_CONTROLLER_BIN}\"",
        "      \"--log-dir\" \"${freq_controller_log_dir}\"",
        "      \"--threshold\" \"${FREQ_CONTROLLER_THRESHOLD_VALUE}\"",
        "      \"--port-profile-id\" \"${profile_id}\"",
        "      \"--gpu-index\" \"${gpu_index}\"",
        "    )",
        "    if [[ -n \"${FREQ_CONTROLLER_CONFIG_VALUE}\" ]]; then",
        "      FREQ_CONTROLLER_CMD+=(\"--config\" \"${FREQ_CONTROLLER_CONFIG_VALUE}\")",
        "    fi",
        "    \"${FREQ_CONTROLLER_CMD[@]}\" &",
        "    freq_controller_pid=\"$!\"",
        "    FREQ_CONTROLLER_PIDS+=(\"${freq_controller_pid}\")",
        "    FREQ_CONTROLLER_STARTED=1",
        "    sleep 1",
        "    if ! kill -0 \"${freq_controller_pid}\" >/dev/null 2>&1; then",
        (
            f"      echo \"[{EXPERIMENT_LOG_TAG}] "
            "freq-controller-linespace-instance failed to stay alive after startup "
            "for port_profile=${profile_id} gpu_index=${gpu_index}\" >&2"
        ),
        "      wait \"${freq_controller_pid}\" || true",
        "      exit 1",
        "    fi",
        "  done",
        "",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --port-profile-id \"${CONTROL_PORT_PROFILE_ID}\" \\",
        "    --port-profile-id-list \"${PORT_PROFILE_IDS_CSV_NORMALIZED}\"",
        "",
        "  stop_freq_controllers",
        "  end_ctx_aware_mode",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "}",
        "",
        (
            f"echo \"[{EXPERIMENT_LOG_TAG}] target_model={target_model} split={split} "
            f"qps_points={len(qps_jobs)} power_gpu_indices={power_gpu_indices_payload} "
            "assignment_policy=${TARGET_ASSIGNMENT_POLICY} "
            "usage=${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE} "
            "scheduling=${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE} "
            "threshold=${FREQ_CONTROLLER_THRESHOLD_VALUE} "
            "backend_pairings=${BACKEND_PAIRINGS_DISPLAY} "
            "control_port_profile=${CONTROL_PORT_PROFILE_ID}\""
        ),
        "",
    ]

    for job in qps_jobs:
        lines.append(
            "run_one_qps "
            f"{_shell_quote(job.qps_display)} "
            f"{_shell_quote(job.qps_slug)} "
            f"{_shell_quote(job.replay_config_relpath)} "
            f"{_shell_quote(path_for_config(job.replay_output_dir))}"
        )
    lines.extend(
        [
            "",
            f"echo \"[{EXPERIMENT_LOG_TAG}] completed {len(qps_jobs)} qps points\"",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o750)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_experiment.py",
        description=(
            "Generate a sweep-QPS replay bundle with power logging for a "
            "gateway_multi deployment by looking up an existing compiled clean "
            "split plan, forcing assignment policy balanced with a configurable "
            "balanced usage threshold, enabling ctx-aware mode, and starting one "
            "freq-controller-linespace-instance controller per backend GPU."
        ),
    )
    parser.add_argument("--source-run-dir", required=True, help="Profiled source run directory under results/.")
    parser.add_argument("--poisson-seed", required=True, type=BASE.parse_non_negative_int, help="Poisson launch seed.")
    parser.add_argument("--randomize-seed", required=True, type=BASE.parse_non_negative_int, help="Replay randomization seed.")
    parser.add_argument("--qps-list", required=True, help="Comma-separated Poisson rates, e.g. 0.1,0.2,0.3")
    parser.add_argument("--time-constraint-s", required=True, type=float, help="Replay time limit in seconds.")
    parser.add_argument("--target-model", required=True, help="Target model key from configs/model_config.toml.")
    parser.add_argument(
        "--port-profile",
        "-P",
        required=True,
        action="append",
        help=(
            "Port profile IDs for gateway_multi. Repeat or pass a comma-separated "
            "list like 2,3. The first selected profile is used as the public "
            "control profile for replay."
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        help=(
            "Clean split mode: exclude-unranked | top | rest. "
            "Compatibility alias: full -> exclude-unranked."
        ),
    )
    parser.add_argument(
        "--split-two-group-metric",
        choices=sorted(BASE.SPLIT_METRIC_ALIASES),
        default="token_usage",
        help="Split grouping metric used for top/rest plan lookup (default: token_usage).",
    )
    parser.add_argument(
        "--power-gpu-indices",
        default="0",
        help=(
            "Comma-separated GPU indices passed to zeus-power-reader and mapped "
            "positionally to the selected gateway_multi backend port profiles "
            "for independent freq-controller-linespace-instance, for example: 2,3."
        ),
    )
    parser.add_argument(
        "--balanced-usage-threshold-tokens",
        default=DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS,
        type=int,
        help=(
            "Balanced assignment threshold sent to gateway_multi policy update. "
            f"Default: {DEFAULT_BALANCED_USAGE_THRESHOLD_TOKENS}."
        ),
    )
    parser.add_argument(
        "--ctx-aware-usage-threshold-tokens",
        default=DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS,
        type=int,
        help=(
            "Usage threshold sent to gateway ctx-aware start. "
            f"Default: {DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS}."
        ),
    )
    parser.add_argument(
        "--ctx-aware-scheduling-threshold-tokens",
        default=DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS,
        type=int,
        help=(
            "Scheduling threshold sent to gateway ctx-aware start. "
            f"Default: {DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS}."
        ),
    )
    parser.add_argument(
        "--freq-controller-threshold",
        type=float,
        default=DEFAULT_FREQ_CONTROLLER_THRESHOLD,
        help=(
            "Default freq-controller-linespace-instance context threshold. "
            f"Default: {int(DEFAULT_FREQ_CONTROLLER_THRESHOLD)}."
        ),
    )
    parser.add_argument(
        "--output-config-dir",
        default=str(DEFAULT_OUTPUT_CONFIG_DIR),
        help=f"Generated bundle root (default: {DEFAULT_OUTPUT_CONFIG_DIR}).",
    )
    parser.add_argument(
        "--additional-suffix",
        default=None,
        help=(
            "Optional replay plan suffix for lookup. Example: with suffix fp8 and split rest, "
            "looks for replay-plan.clean.token.rest.fp8.json."
        ),
    )
    parser.add_argument(
        "--replay-output-root",
        default=str(DEFAULT_REPLAY_OUTPUT_ROOT),
        help=(
            "Replay output root. Default appends "
            "<dataset-lineage>/split/<split>/<qps>/<timestamp> under "
            "results/replay/"
            "sweep-qps-docker-power-clean-gateway_multi-balanced-freq-ctrl-linesapce-instance-ctx-aware/. "
            "dataset-lineage is inferred from --source-run-dir by dropping "
            "the first (<model>) and last (<run-dir>) path segments."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Optional suffix appended to the replay output root directory name. "
            "For example, --output-suffix lmcache writes under "
            "results/replay/"
            "sweep-qps-docker-power-clean-gateway_multi-balanced-freq-ctrl-linesapce-instance-ctx-aware-lmcache/."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    generation_command_raw = build_generation_command_raw(argv)

    try:
        _sync_base_module_globals()
        source_run_dir = Path(args.source_run_dir).expanduser().resolve()
        if not source_run_dir.exists() or not source_run_dir.is_dir():
            raise ValueError(f"invalid --source-run-dir: {source_run_dir}")
        try:
            source_run_dir.relative_to(RESULTS_ROOT.resolve())
        except ValueError:
            raise ValueError("--source-run-dir must live under top-level results/")

        split = BASE.canonical_split(args.split)
        qps_values = BASE.parse_qps_list(str(args.qps_list))
        time_constraint_s = BASE.parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )
        port_profile_ids = parse_port_profile_ids(args.port_profile)
        control_port_profile = port_profile_ids[0]
        power_gpu_indices = BASE.parse_gpu_index_list(str(args.power_gpu_indices))
        backend_targets = build_backend_freq_controller_targets(
            port_profile_ids=port_profile_ids,
            gpu_indices=power_gpu_indices,
        )
        balanced_usage_threshold_tokens = parse_positive_int(
            str(args.balanced_usage_threshold_tokens),
            field_name="--balanced-usage-threshold-tokens",
        )
        ctx_aware_usage_threshold_tokens = parse_positive_int(
            str(args.ctx_aware_usage_threshold_tokens),
            field_name="--ctx-aware-usage-threshold-tokens",
        )
        ctx_aware_scheduling_threshold_tokens = parse_positive_int(
            str(args.ctx_aware_scheduling_threshold_tokens),
            field_name="--ctx-aware-scheduling-threshold-tokens",
        )
        if ctx_aware_scheduling_threshold_tokens >= ctx_aware_usage_threshold_tokens:
            raise ValueError(
                "--ctx-aware-scheduling-threshold-tokens must be smaller than "
                "--ctx-aware-usage-threshold-tokens"
            )
        if ctx_aware_scheduling_threshold_tokens < CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS:
            raise ValueError(
                "--ctx-aware-scheduling-threshold-tokens must be >= "
                f"{CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS}"
            )
        freq_controller_threshold = BASE.parse_positive_float(
            str(args.freq_controller_threshold),
            field_name="--freq-controller-threshold",
        )

        target_model = str(args.target_model).strip()
        if not target_model:
            raise ValueError("--target-model cannot be empty")
        target_model_specs = BASE._load_target_model_specs()
        target_model_spec = target_model_specs.get(target_model)
        if target_model_spec is None:
            available = ", ".join(sorted(target_model_specs.keys()))
            raise ValueError(
                f"unknown --target-model {target_model!r}; available keys: {available}"
            )
        additional_suffix = (
            str(args.additional_suffix).strip()
            if args.additional_suffix is not None and str(args.additional_suffix).strip()
            else None
        )

        output_config_root = Path(args.output_config_dir).expanduser().resolve()
        replay_output_root = Path(args.replay_output_root).expanduser()
        if not replay_output_root.is_absolute():
            replay_output_root = (REPO_ROOT / replay_output_root).resolve()
        else:
            replay_output_root = replay_output_root.resolve()
        output_suffix = normalize_output_suffix(getattr(args, "output_suffix", None))
        replay_output_root = apply_output_suffix_to_replay_root(
            replay_output_root,
            output_suffix,
        )
        source_dataset_lineage = BASE.derive_dataset_lineage_from_source_run_dir(source_run_dir)

        batch_timestamp = BASE.build_utc_timestamp_slug()
        batch_dir = (output_config_root / batch_timestamp).resolve()

        lookup_result = BASE._resolve_lookup_plan_paths(
            source_run_dir=source_run_dir,
            split=split,
            split_two_group_metric=str(args.split_two_group_metric),
            additional_suffix=additional_suffix,
        )
        related_plan_models: dict[str, str] = {}
        for plan_key, plan_path_text in lookup_result.related_plan_paths.items():
            plan_model = BASE._validate_plan_model_matches_target(
                plan_path=Path(plan_path_text),
                target_model_spec=target_model_spec,
            )
            related_plan_models[plan_key] = plan_model

        plan_copy_dir = (batch_dir / "plan").resolve()
        plan_copy_dir.mkdir(parents=True, exist_ok=True)
        selected_plan_copy_path = (plan_copy_dir / lookup_result.selected_plan_path.name).resolve()
        shutil.copy2(lookup_result.selected_plan_path, selected_plan_copy_path)

        qps_jobs: list[QpsReplayJob] = []
        for qps in qps_values:
            qps_slug = BASE.format_qps_slug(qps)
            replay_output_dir = (
                replay_output_root
                / source_dataset_lineage
                / "split"
                / split
                / qps_slug
                / batch_timestamp
            ).resolve()
            replay_payload = {
                "plan": BASE.path_for_config(selected_plan_copy_path),
                "output_dir": BASE.path_for_config(replay_output_dir),
                "randomize_seed": int(args.randomize_seed),
                "time_constraint_s": time_constraint_s,
                "port_profile_id": control_port_profile,
                "port_profile_id_list": list(port_profile_ids),
                "launch_policy_override": {
                    "seed": int(args.poisson_seed),
                    "pattern": {"name": "poisson"},
                    "pattern_args": {"rate": qps},
                },
            }
            replay_config_path = (batch_dir / qps_slug / "replay.toml").resolve()
            BASE.write_replay_config(replay_config_path, replay_payload=replay_payload)
            replay_config_relpath = str(replay_config_path.relative_to(batch_dir))

            qps_jobs.append(
                QpsReplayJob(
                    qps=qps,
                    qps_display=BASE.qps_display_text(qps),
                    qps_slug=qps_slug,
                    replay_config_relpath=replay_config_relpath,
                    replay_config_path=replay_config_path,
                    replay_output_dir=replay_output_dir,
                    power_output_dir=(replay_output_dir / "power").resolve(),
                    freq_controller_log_dir=(
                        replay_output_dir / DEFAULT_FREQ_CONTROL_LOG_DIR_NAME
                    ).resolve(),
                )
            )

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile_ids=port_profile_ids,
            target_model=target_model,
            split=split,
            power_gpu_indices=power_gpu_indices,
            qps_jobs=qps_jobs,
            balanced_usage_threshold_tokens=balanced_usage_threshold_tokens,
            ctx_aware_usage_threshold_tokens=ctx_aware_usage_threshold_tokens,
            ctx_aware_scheduling_threshold_tokens=ctx_aware_scheduling_threshold_tokens,
            freq_controller_threshold=freq_controller_threshold,
        )

        qps_summary = []
        for job in qps_jobs:
            qps_summary.append(
                {
                    "qps": job.qps,
                    "qps_slug": job.qps_slug,
                    "replay_config": str(job.replay_config_path),
                    "replay_output_dir": BASE.path_for_config(job.replay_output_dir),
                    "power_output_dir": BASE.path_for_config(job.power_output_dir),
                    "freq_controller_log_dir": BASE.path_for_config(job.freq_controller_log_dir),
                    "freq_controller_backend_log_dirs": [
                        {
                            "port_profile_id": target.port_profile_id,
                            "gpu_index": target.gpu_index,
                            "freq_controller_log_dir": BASE.path_for_config(
                                (
                                    job.freq_controller_log_dir
                                    / freq_controller_backend_log_dir_name(target.port_profile_id)
                                ).resolve()
                            ),
                        }
                        for target in backend_targets
                    ],
                }
            )
        summary = {
            "status": "ok",
            "batch_timestamp": batch_timestamp,
            "source_run_dir": str(source_run_dir),
            "source_dataset_lineage": str(source_dataset_lineage),
            "target_model": target_model,
            "split": split,
            "split_two_group_metric": args.split_two_group_metric,
            "launch_pattern": "poisson",
            "poisson_seed": int(args.poisson_seed),
            "randomize_seed": int(args.randomize_seed),
            "qps_list": qps_values,
            "time_constraint_s": time_constraint_s,
            "port_profile": control_port_profile,
            "control_port_profile": control_port_profile,
            "port_profile_ids": port_profile_ids,
            "assignment_policy": DEFAULT_ASSIGNMENT_POLICY,
            "balanced_usage_threshold_tokens": balanced_usage_threshold_tokens,
            "power_gpu_indices": power_gpu_indices,
            "ctx_aware_usage_threshold_tokens": ctx_aware_usage_threshold_tokens,
            "ctx_aware_scheduling_threshold_tokens": ctx_aware_scheduling_threshold_tokens,
            "freq_controller_threshold": freq_controller_threshold,
            "freq_controller_backend_pairings": [
                {
                    "port_profile_id": target.port_profile_id,
                    "gpu_index": target.gpu_index,
                }
                for target in backend_targets
            ],
            "additional_suffix": additional_suffix,
            "output_suffix": output_suffix,
            "replay_output_root_dir": BASE.path_for_config(replay_output_root),
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "plan_lookup_only": True,
            "selected_source_plan": str(lookup_result.selected_plan_path),
            "related_source_plans": lookup_result.related_plan_paths,
            "selected_plan_model": related_plan_models.get("selected"),
            "related_plan_models": related_plan_models,
            "selected_plan_copy": str(selected_plan_copy_path),
            "qps_points": qps_summary,
            "run_script": str(run_script_path),
            "run_command_default": f"bash {BASE.path_for_config(run_script_path)}",
            "run_command_with_port_profiles": (
                f"bash {BASE.path_for_config(run_script_path)} "
                f"{','.join(str(profile_id) for profile_id in port_profile_ids)}"
            ),
            "generation_command_raw": generation_command_raw,
        }
        manifest_path = (batch_dir / "manifest.json").resolve()
        manifest_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        summary["manifest_path"] = str(manifest_path)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
