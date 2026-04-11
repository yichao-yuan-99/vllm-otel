#!/usr/bin/env python3
"""Generate one dedicated mi3008x embedded TP1 AMD sweep-QPS + sweep-SLO bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_DIR_NAME = "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive"
MI3008X_MIN_PORT_PROFILE_ID = 0
MI3008X_MAX_PORT_PROFILE_ID = 7
MAX_QPS_POINTS = MI3008X_MAX_PORT_PROFILE_ID - MI3008X_MIN_PORT_PROFILE_ID + 1
DEFAULT_PORT_PROFILE_ID = MI3008X_MIN_PORT_PROFILE_ID
GPU_INDEX_RUNTIME_DEFAULT = "match-port-profile"
DEFAULT_OUTPUT_CONFIG_DIR = (
    REPO_ROOT
    / "experiments"
    / "amd-embeded"
    / "servers-amdhpc-mi3008x-embedded-TP1"
    / EXPERIMENT_DIR_NAME
    / "generated"
)
DEFAULT_REPLAY_OUTPUT_ROOT = (
    Path("results")
    / "replay"
    / "amd-embeded"
    / "servers-amdhpc-mi3008x-embedded-TP1"
    / EXPERIMENT_DIR_NAME
)
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.toml"
EXPERIMENT_PATH = (
    "experiments/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
    f"{EXPERIMENT_DIR_NAME}"
)
EXPERIMENT_LOG_TAG = (
    "amd-embeded-servers-amdhpc-mi3008x-embedded-TP1-"
    "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive"
)
PROFILE_OUTPUT_PLACEHOLDER = "profile-<port_profile_id>"
DEFAULT_CTX_AWARE_USAGE_THRESHOLD_TOKENS = 1445793
DEFAULT_CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS = 1369699
CTX_AWARE_NEW_AGENT_PSEUDO_TOKENS = 3000
CTX_AWARE_POLICY_MODE = "throughput"
SLO_AWARE_POLICY_MODE = "push-back-half-slack"
DEFAULT_FREQ_CONTROLLER_THRESHOLD = 1141416.0
DEFAULT_FREQ_CONTROL_LOG_DIR_NAME = "freq-control-linespace"
DEFAULT_FREQ_CONTROLLER_BIN_NAME = "freq-controller-linespace-slo-amd"
FREQ_CONTROLLER_AGGRESSIVE_SLO_CONTROL = True
DEFAULT_GATEWAY_CTX_IPC_SOCKET_DIR = "/tmp"
_AMD_BASE_GENERATOR_MODULE_NAME = (
    "generate_sweep_qps_docker_power_clean_experiment_for_mi3008x_embedded_tp1_freq_ctrl_linespace_amd_base"
)


@dataclass(frozen=True)
class QpsSloReplayJob:
    qps: float
    qps_display: str
    qps_slug: str
    assigned_port_profile: int
    assigned_gpu_index: int
    target_slo: float
    target_slo_display: str
    target_slo_slug: str
    replay_config_relpath: str
    replay_config_path: Path
    replay_output_dir: Path


def _load_amd_base_generator_module() -> object:
    module_path = (
        REPO_ROOT
        / "experiments"
        / "amd-embeded"
        / "servers-amdhpc-mi3008x-embedded-TP1"
        / "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd"
        / "generate_experiment.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        _AMD_BASE_GENERATOR_MODULE_NAME,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load AMD base generator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_AMD_BASE_GENERATOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


AMD_BASE = _load_amd_base_generator_module()
BASE = AMD_BASE.BASE


def _sync_base_module_globals() -> None:
    AMD_BASE.REPO_ROOT = REPO_ROOT
    AMD_BASE.RESULTS_ROOT = RESULTS_ROOT
    AMD_BASE.MODEL_CONFIG_PATH = MODEL_CONFIG_PATH
    AMD_BASE._sync_base_module_globals()


def build_generation_command_raw(argv: list[str] | None) -> str:
    if argv is None:
        original = getattr(sys, "orig_argv", None)
        if isinstance(original, list) and original:
            return shlex.join([str(token) for token in original])
        return shlex.join([str(sys.executable), *[str(token) for token in sys.argv]])
    return shlex.join(
        [str(sys.executable), str(Path(__file__).resolve()), *[str(token) for token in argv]]
    )


def path_for_config(path: Path) -> str:
    _sync_base_module_globals()
    return BASE.path_for_config(path)


def parse_positive_int(value: str, *, field_name: str) -> int:
    return AMD_BASE.parse_positive_int(value, field_name=field_name)


def parse_optional_gpu_index(raw: str | None) -> int | None:
    return AMD_BASE.parse_optional_gpu_index(raw)


def validate_mi3008x_port_profile(value: int, *, field_name: str) -> int:
    return AMD_BASE.validate_mi3008x_port_profile(value, field_name=field_name)


def ensure_supported_qps_point_count(qps_values: list[float]) -> None:
    AMD_BASE.ensure_supported_qps_point_count(qps_values)


def normalize_output_suffix(raw: str | None) -> str | None:
    return AMD_BASE.normalize_output_suffix(raw)


def apply_output_suffix_to_replay_root(root: Path, output_suffix: str | None) -> Path:
    return AMD_BASE.apply_output_suffix_to_replay_root(root, output_suffix)


def build_assigned_profile_output_dir(path: Path, *, assigned_port_profile: int) -> str:
    return AMD_BASE.build_assigned_profile_output_dir(
        path,
        assigned_port_profile=assigned_port_profile,
    )


def parse_positive_float_list(raw: str, *, field_name: str) -> list[float]:
    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{field_name} must be a non-empty comma-separated list")

    values: list[float] = []
    seen: set[float] = set()
    for token in tokens:
        try:
            parsed = float(token)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} contains non-numeric value: {token!r}"
            ) from exc
        if parsed <= 0:
            raise ValueError(f"{field_name} values must be > 0")
        if parsed in seen:
            raise ValueError(f"{field_name} contains duplicate value: {token}")
        seen.add(parsed)
        values.append(parsed)
    return values


def parse_target_slo_list(raw: str) -> list[float]:
    return parse_positive_float_list(raw, field_name="--target-slo-list")


def display_text(value: float) -> str:
    return format(value, ".12g")


def _format_positive_value_slug(*, prefix: str, value: float) -> str:
    text = format(value, ".12g")
    text = text.replace("-", "m").replace(".", "_").replace("+", "")
    return f"{prefix}{text}"


def format_target_slo_slug(target_slo: float) -> str:
    return _format_positive_value_slug(prefix="slo", value=target_slo)


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _shell_array_literal(values: list[str]) -> str:
    return "(" + " ".join(_shell_quote(value) for value in values) + ")"


def _resolve_default_python_bin() -> Path:
    python_bin = Path(sys.executable).expanduser().resolve()
    if not python_bin.exists():
        raise RuntimeError(f"Unable to resolve absolute python executable: {python_bin}")
    return python_bin


def _resolve_default_curl_bin() -> Path:
    discovered = shutil.which("curl")
    candidates = [discovered] if discovered is not None else []
    candidates.extend(["/usr/bin/curl", "/bin/curl", "/usr/local/bin/curl"])
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    raise RuntimeError("Unable to resolve absolute curl executable path")


def write_replay_config(path: Path, *, replay_payload: dict[str, Any]) -> None:
    _sync_base_module_globals()
    lines: list[str] = [
        f"# Auto-generated by {EXPERIMENT_PATH}/generate_experiment.py",
        "",
    ]
    BASE._append_toml_table(lines, "replay", replay_payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_script(
    path: Path,
    *,
    default_port_profile: int,
    target_model: str,
    split: str,
    qps_slo_jobs: list[QpsSloReplayJob],
    ctx_aware_usage_threshold_tokens: int,
    ctx_aware_scheduling_threshold_tokens: int,
    freq_controller_threshold: float,
) -> None:
    qps_groups: list[tuple[int, list[QpsSloReplayJob]]] = []
    jobs_by_port_profile: dict[int, list[QpsSloReplayJob]] = {}
    for job in qps_slo_jobs:
        if job.assigned_port_profile not in jobs_by_port_profile:
            jobs_by_port_profile[job.assigned_port_profile] = []
            qps_groups.append(
                (job.assigned_port_profile, jobs_by_port_profile[job.assigned_port_profile])
            )
        jobs_by_port_profile[job.assigned_port_profile].append(job)

    total_slo_rounds = max((len(group_jobs) for _, group_jobs in qps_groups), default=0)
    default_amd_power_reader_bin = (REPO_ROOT / ".venv" / "bin" / "amd-power-reader").resolve()
    default_freq_controller_bin = (
        REPO_ROOT / ".venv" / "bin" / DEFAULT_FREQ_CONTROLLER_BIN_NAME
    ).resolve()
    default_reset_gpu_core_freq_bin = (
        REPO_ROOT / ".venv" / "bin" / "amd-reset-gpu-core-freq"
    ).resolve()
    default_python_bin = _resolve_default_python_bin()
    default_curl_bin = _resolve_default_curl_bin()

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"DEFAULT_PORT_PROFILE_ID={default_port_profile}",
        "PORT_PROFILE_ID_VALUE=\"${1:-${PORT_PROFILE_ID:-${DEFAULT_PORT_PROFILE_ID}}}\"",
        "GPU_INDEX_VALUE=\"${GPU_INDEX:-}\"",
        (
            "PYTHON_BIN="
            f"\"${{PYTHON_BIN:-{_shell_quote(str(default_python_bin))}}}\""
        ),
        (
            "CURL_BIN="
            f"\"${{CURL_BIN:-{_shell_quote(str(default_curl_bin))}}}\""
        ),
        (
            "AMD_POWER_READER_BIN="
            f"\"${{AMD_POWER_READER_BIN:-{_shell_quote(str(default_amd_power_reader_bin))}}}\""
        ),
        (
            "FREQ_CONTROLLER_BIN="
            f"\"${{FREQ_CONTROLLER_BIN:-{_shell_quote(str(default_freq_controller_bin))}}}\""
        ),
        (
            "RESET_GPU_CORE_FREQ_BIN="
            f"\"${{RESET_GPU_CORE_FREQ_BIN:-{_shell_quote(str(default_reset_gpu_core_freq_bin))}}}\""
        ),
        "AMD_SMI_POWER_SOCKET_PATH_VALUE=\"${AMD_SMI_POWER_SOCKET_PATH:-/tmp/amdsmi-power-reader.sock}\"",
        "GATEWAY_BASE_URL_VALUE=\"${GATEWAY_BASE_URL:-}\"",
        (
            "DEFAULT_FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH="
            f"\"{DEFAULT_GATEWAY_CTX_IPC_SOCKET_DIR}/vllm-gateway-ctx-profile-${{PORT_PROFILE_ID_VALUE}}.sock\""
        ),
        (
            "FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH_VALUE="
            "\"${FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH:-${DEFAULT_FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH}}\""
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
        f"ASSIGNED_QPS_POINTS={len(qps_groups)}",
        f"TOTAL_SLO_ROUNDS={total_slo_rounds}",
        "SELECTED_QPS_VALUE=\"\"",
        "SELECTED_QPS_SLUG=\"\"",
        "SELECTED_ASSIGNED_GPU_INDEX=\"\"",
        "SELECTED_SLO_VALUES=()",
        "SELECTED_SLO_SLUGS=()",
        "SELECTED_REPLAY_CONFIG_REFS=()",
        "SELECTED_REPLAY_OUTPUT_REFS=()",
        "",
        "POWER_READER_PID=\"\"",
        "FREQ_CONTROLLER_PID=\"\"",
        "CTX_AWARE_STARTED=0",
        "SLO_AWARE_STARTED=0",
        "FREQ_CONTROLLER_STARTED=0",
        "GATEWAY_BASE_URL_RESOLVED=\"\"",
        "",
        "stop_power_reader() {",
        "  if [[ -n \"${POWER_READER_PID}\" ]]; then",
        "    kill \"${POWER_READER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${POWER_READER_PID}\" 2>/dev/null || true",
        "    POWER_READER_PID=\"\"",
        "  fi",
        "}",
        "",
        "stop_freq_controller() {",
        "  if [[ -n \"${FREQ_CONTROLLER_PID}\" ]]; then",
        "    kill \"${FREQ_CONTROLLER_PID}\" >/dev/null 2>&1 || true",
        "    wait \"${FREQ_CONTROLLER_PID}\" 2>/dev/null || true",
        "    FREQ_CONTROLLER_PID=\"\"",
        "  fi",
        "}",
        "",
        "reset_gpu_core_if_needed() {",
        "  if [[ \"${FREQ_CONTROLLER_STARTED}\" -eq 1 ]]; then",
        "    if ! \"${RESET_GPU_CORE_FREQ_BIN}\" --gpu-index \"${GPU_INDEX_VALUE}\"; then",
        (
            f"      echo \"[{EXPERIMENT_LOG_TAG}] warning: failed to reset GPU "
            "${GPU_INDEX_VALUE} core clocks\" >&2"
        ),
        "    fi",
        "    FREQ_CONTROLLER_STARTED=0",
        "  fi",
        "}",
        "",
        "normalize_port_profile_id() {",
        "  local raw_value=\"$1\"",
        "  local normalized_value=\"${raw_value//[[:space:]]/}\"",
        "  if [[ -z \"${normalized_value}\" || ! \"${normalized_value}\" =~ ^[0-9]+$ ]]; then",
        f"    echo \"[{EXPERIMENT_LOG_TAG}] error: invalid port profile id: ${{raw_value}}\" >&2",
        "    exit 1",
        "  fi",
        (
            f"  if (( normalized_value < {MI3008X_MIN_PORT_PROFILE_ID} "
            f"|| normalized_value > {MI3008X_MAX_PORT_PROFILE_ID} )); then"
        ),
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] error: supported mi3008x port "
            f"profile ids are {MI3008X_MIN_PORT_PROFILE_ID}..{MI3008X_MAX_PORT_PROFILE_ID}\" >&2"
        ),
        "    exit 1",
        "  fi",
        "  PORT_PROFILE_ID_VALUE=\"${normalized_value}\"",
        "}",
        "",
        "normalize_gpu_index() {",
        "  local raw_value=\"$1\"",
        "  local expected_value=\"$2\"",
        "  local normalized_value=\"${raw_value//[[:space:]]/}\"",
        "  if [[ -z \"${normalized_value}\" ]]; then",
        "    GPU_INDEX_VALUE=\"${expected_value}\"",
        "    return 0",
        "  fi",
        "  if [[ ! \"${normalized_value}\" =~ ^[0-9]+$ ]]; then",
        f"    echo \"[{EXPERIMENT_LOG_TAG}] error: invalid gpu index: ${{raw_value}}\" >&2",
        "    exit 1",
        "  fi",
        (
            f"  if (( normalized_value < {MI3008X_MIN_PORT_PROFILE_ID} "
            f"|| normalized_value > {MI3008X_MAX_PORT_PROFILE_ID} )); then"
        ),
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] error: supported mi3008x gpu "
            f"indices are {MI3008X_MIN_PORT_PROFILE_ID}..{MI3008X_MAX_PORT_PROFILE_ID}\" >&2"
        ),
        "    exit 1",
        "  fi",
        "  if [[ \"${normalized_value}\" != \"${expected_value}\" ]]; then",
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] error: this mi3008x workflow "
            "expects gpu index ${expected_value} for port profile "
            "${PORT_PROFILE_ID_VALUE}\" >&2"
        ),
        "    exit 1",
        "  fi",
        "  GPU_INDEX_VALUE=\"${normalized_value}\"",
        "}",
        "",
        "resolve_gateway_base_url() {",
        "  if [[ -n \"${GATEWAY_BASE_URL_VALUE}\" ]]; then",
        "    printf '%s\\n' \"${GATEWAY_BASE_URL_VALUE}\"",
        "    return 0",
        "  fi",
        "  \"${PYTHON_BIN}\" - \"${REPO_ROOT}\" \"${PORT_PROFILE_ID_VALUE}\" <<'PY'",
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
        "start_ctx_aware_mode() {",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] enabling ctx-aware "
            f"mode={CTX_AWARE_POLICY_MODE} "
            "usage=${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE} "
            "scheduling=${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE} "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    -H \"content-type: application/json\" \\",
        (
            "    --data "
            "\"{\\\"usage_threshold_tokens\\\": ${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE}, "
            "\\\"scheduling_threshold_tokens\\\": ${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE}, "
            f"\\\"policy_mode\\\": \\\"{CTX_AWARE_POLICY_MODE}\\\"}}\" \\"
        ),
        "    \"${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/start\"",
        "  CTX_AWARE_STARTED=1",
        "}",
        "",
        "start_slo_aware_mode() {",
        "  local target_slo_value=\"$1\"",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] enabling slo-aware "
            f"mode={SLO_AWARE_POLICY_MODE} "
            "target_slo=${target_slo_value} "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    -H \"content-type: application/json\" \\",
        (
            "    --data "
            "\"{\\\"target_tokens_per_s\\\": ${target_slo_value}, "
            f"\\\"policy_mode\\\": \\\"{SLO_AWARE_POLICY_MODE}\\\"}}\" \\"
        ),
        "    \"${GATEWAY_BASE_URL_RESOLVED}/slo-aware/start\"",
        "  SLO_AWARE_STARTED=1",
        "}",
        "",
        "end_slo_aware_mode() {",
        "  if [[ \"${SLO_AWARE_STARTED}\" -eq 0 ]]; then",
        "    return 0",
        "  fi",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] disabling slo-aware "
            f"mode={SLO_AWARE_POLICY_MODE} "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    \"${GATEWAY_BASE_URL_RESOLVED}/slo-aware/end\"",
        "  SLO_AWARE_STARTED=0",
        "}",
        "",
        "end_ctx_aware_mode() {",
        "  if [[ \"${CTX_AWARE_STARTED}\" -eq 0 ]]; then",
        "    return 0",
        "  fi",
        "  ensure_gateway_base_url",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] disabling ctx-aware "
            f"mode={CTX_AWARE_POLICY_MODE} "
            "gateway=${GATEWAY_BASE_URL_RESOLVED} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "  \"${CURL_BIN}\" --fail --silent --show-error -o /dev/null \\",
        "    -X POST \\",
        "    \"${GATEWAY_BASE_URL_RESOLVED}/ctx-aware/end\"",
        "  CTX_AWARE_STARTED=0",
        "  SLO_AWARE_STARTED=0",
        "}",
        "",
        "cleanup() {",
        "  local exit_code=\"$1\"",
        "  stop_freq_controller",
        "  end_slo_aware_mode || true",
        "  end_ctx_aware_mode || true",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "  return \"${exit_code}\"",
        "}",
        "",
        "trap '__exit_code=$?; trap - EXIT INT TERM; cleanup \"${__exit_code}\"; exit \"${__exit_code}\"' EXIT",
        f"trap 'echo \"[{EXPERIMENT_LOG_TAG}] interrupted\" >&2; exit 130' INT TERM",
        "",
        "run_one_qps_slo() {",
        "  local qps_value=\"$1\"",
        "  local qps_slug=\"$2\"",
        "  local target_slo_value=\"$3\"",
        "  local target_slo_slug=\"$4\"",
        "  local replay_config_ref=\"$5\"",
        "  local replay_output_ref=\"$6\"",
        "  local round_number=\"$7\"",
        "  local total_rounds=\"$8\"",
        "  local replay_config_path=\"\"",
        "  local replay_output_base_dir=\"\"",
        "  local replay_output_dir=\"\"",
        "  local power_output_dir=\"\"",
        "  local freq_controller_log_dir=\"\"",
        "  local -a FREQ_CONTROLLER_CMD=()",
        "",
        "  if [[ \"${replay_config_ref}\" = /* ]]; then",
        "    replay_config_path=\"${replay_config_ref}\"",
        "  else",
        "    replay_config_path=\"${SCRIPT_DIR}/${replay_config_ref}\"",
        "  fi",
        "",
        "  if [[ \"${replay_output_ref}\" = /* ]]; then",
        "    replay_output_base_dir=\"${replay_output_ref}\"",
        "  else",
        "    replay_output_base_dir=\"${REPO_ROOT}/${replay_output_ref}\"",
        "  fi",
        "  replay_output_dir=\"${replay_output_base_dir}/profile-${PORT_PROFILE_ID_VALUE}\"",
        "  power_output_dir=\"${replay_output_dir}/power\"",
        (
            "  freq_controller_log_dir="
            f"\"${{replay_output_dir}}/{DEFAULT_FREQ_CONTROL_LOG_DIR_NAME}\""
        ),
        "",
        (
            f"  echo \"[{EXPERIMENT_LOG_TAG}] split={split} "
            "qps=${qps_value} qps_slug=${qps_slug} "
            "target_slo=${target_slo_value} target_slo_slug=${target_slo_slug} "
            "round=${round_number}/${total_rounds} "
            f"ctx_mode={CTX_AWARE_POLICY_MODE} slo_mode={SLO_AWARE_POLICY_MODE} "
            "usage=${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE} "
            "scheduling=${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE} "
            "threshold=${FREQ_CONTROLLER_THRESHOLD_VALUE} "
            "gateway_ipc_socket=${FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH_VALUE} "
            "output=${replay_output_dir} "
            "gpu_index=${GPU_INDEX_VALUE} "
            "port_profile=${PORT_PROFILE_ID_VALUE}\""
        ),
        "  mkdir -p \"${power_output_dir}\" \"${freq_controller_log_dir}\"",
        "  \"${AMD_POWER_READER_BIN}\" --output-dir \"${power_output_dir}\" --gpu-indices \"${GPU_INDEX_VALUE}\" --socket-path \"${AMD_SMI_POWER_SOCKET_PATH_VALUE}\" &",
        "  POWER_READER_PID=\"$!\"",
        "",
        "  start_ctx_aware_mode",
        "  start_slo_aware_mode \"${target_slo_value}\"",
        "",
        "  FREQ_CONTROLLER_CMD=(",
        "    \"${FREQ_CONTROLLER_BIN}\"",
        "    \"--log-dir\" \"${freq_controller_log_dir}\"",
        "    \"--threshold\" \"${FREQ_CONTROLLER_THRESHOLD_VALUE}\"",
        "    \"--throughput-target\" \"${target_slo_value}\"",
        "    \"--aggresive\"",
        (
            "    \"--gateway-ipc-socket-path\" "
            "\"${FREQ_CONTROLLER_GATEWAY_IPC_SOCKET_PATH_VALUE}\""
        ),
        "    \"--port-profile-id\" \"${PORT_PROFILE_ID_VALUE}\"",
        "    \"--gpu-index\" \"${GPU_INDEX_VALUE}\"",
        "  )",
        "  if [[ -n \"${FREQ_CONTROLLER_CONFIG_VALUE}\" ]]; then",
        "    FREQ_CONTROLLER_CMD+=(\"--config\" \"${FREQ_CONTROLLER_CONFIG_VALUE}\")",
        "  fi",
        "  \"${FREQ_CONTROLLER_CMD[@]}\" &",
        "  FREQ_CONTROLLER_PID=\"$!\"",
        "  FREQ_CONTROLLER_STARTED=1",
        "  sleep 1",
        "  if ! kill -0 \"${FREQ_CONTROLLER_PID}\" >/dev/null 2>&1; then",
        (
            f"    echo \"[{EXPERIMENT_LOG_TAG}] "
            "freq-controller-linespace-slo-amd failed to stay alive after startup "
            "for port_profile=${PORT_PROFILE_ID_VALUE} gpu_index=${GPU_INDEX_VALUE}\" >&2"
        ),
        "    wait \"${FREQ_CONTROLLER_PID}\" || true",
        "    exit 1",
        "  fi",
        "",
        "  \"${PYTHON_BIN}\" -m replayer replay \\",
        "    --config \"${replay_config_path}\" \\",
        "    --output-dir \"${replay_output_dir}\" \\",
        "    --port-profile-id \"${PORT_PROFILE_ID_VALUE}\"",
        "",
        "  stop_freq_controller",
        "  end_slo_aware_mode",
        "  end_ctx_aware_mode",
        "  stop_power_reader",
        "  reset_gpu_core_if_needed",
        "}",
        "",
        "select_qps_job() {",
        "  case \"${PORT_PROFILE_ID_VALUE}\" in",
    ]

    for assigned_port_profile, group_jobs in qps_groups:
        first_job = group_jobs[0]
        lines.extend(
            [
                f"    {assigned_port_profile})",
                f"      SELECTED_QPS_VALUE={_shell_quote(first_job.qps_display)}",
                f"      SELECTED_QPS_SLUG={_shell_quote(first_job.qps_slug)}",
                f"      SELECTED_ASSIGNED_GPU_INDEX={first_job.assigned_gpu_index}",
                f"      SELECTED_SLO_VALUES={_shell_array_literal([job.target_slo_display for job in group_jobs])}",
                f"      SELECTED_SLO_SLUGS={_shell_array_literal([job.target_slo_slug for job in group_jobs])}",
                f"      SELECTED_REPLAY_CONFIG_REFS={_shell_array_literal([job.replay_config_relpath for job in group_jobs])}",
                f"      SELECTED_REPLAY_OUTPUT_REFS={_shell_array_literal([path_for_config(job.replay_output_dir) for job in group_jobs])}",
                "      ;;",
            ]
        )

    lines.extend(
        [
            "    *)",
            (
                f"      echo \"[{EXPERIMENT_LOG_TAG}] no qps assigned for "
                "port_profile=${PORT_PROFILE_ID_VALUE} "
                "assigned_qps_points=${ASSIGNED_QPS_POINTS}; skipping\""
            ),
            "      exit 0",
            "      ;;",
            "  esac",
            "}",
            "",
            "normalize_port_profile_id \"${PORT_PROFILE_ID_VALUE}\"",
            "select_qps_job",
            "normalize_gpu_index \"${GPU_INDEX_VALUE}\" \"${SELECTED_ASSIGNED_GPU_INDEX}\"",
            "",
            (
                f"echo \"[{EXPERIMENT_LOG_TAG}] target_model={target_model} split={split} "
                f"assigned_qps_points={len(qps_groups)} "
                "selected_qps=${SELECTED_QPS_VALUE} "
                "slo_rounds=${#SELECTED_SLO_VALUES[@]} "
                "gpu_index=${GPU_INDEX_VALUE} "
                f"ctx_mode={CTX_AWARE_POLICY_MODE} slo_mode={SLO_AWARE_POLICY_MODE} "
                "usage=${CTX_AWARE_USAGE_THRESHOLD_TOKENS_VALUE} "
                "scheduling=${CTX_AWARE_SCHEDULING_THRESHOLD_TOKENS_VALUE} "
                "threshold=${FREQ_CONTROLLER_THRESHOLD_VALUE} "
                "port_profile=${PORT_PROFILE_ID_VALUE}\""
            ),
            "",
            "for slo_index in \"${!SELECTED_SLO_VALUES[@]}\"; do",
            "  round_number=$((slo_index + 1))",
            "  run_one_qps_slo \\",
            "    \"${SELECTED_QPS_VALUE}\" \\",
            "    \"${SELECTED_QPS_SLUG}\" \\",
            "    \"${SELECTED_SLO_VALUES[$slo_index]}\" \\",
            "    \"${SELECTED_SLO_SLUGS[$slo_index]}\" \\",
            "    \"${SELECTED_REPLAY_CONFIG_REFS[$slo_index]}\" \\",
            "    \"${SELECTED_REPLAY_OUTPUT_REFS[$slo_index]}\" \\",
            "    \"${round_number}\" \\",
            "    \"${#SELECTED_SLO_VALUES[@]}\"",
            "done",
            "",
            (
                f"echo \"[{EXPERIMENT_LOG_TAG}] completed qps=${{SELECTED_QPS_VALUE}} "
                "slo_rounds=${#SELECTED_SLO_VALUES[@]} "
                "port_profile=${PORT_PROFILE_ID_VALUE}\""
            ),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o750)


def build_embedded_tp1_submit_command(*, run_script_path: Path, target_model: str) -> str:
    return shlex.join(
        [
            str(_resolve_default_python_bin()),
            "servers/servers-amdhpc-mi3008x-embedded-TP1/launch.py",
            "submit",
            "-m",
            target_model,
            "-e",
            path_for_config(run_script_path),
        ]
    )


def write_submit_script(
    path: Path,
    *,
    run_script_path: Path,
    target_model: str,
) -> None:
    default_python_bin = _resolve_default_python_bin()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
        f"REPO_ROOT={_shell_quote(str(REPO_ROOT.resolve()))}",
        f"PYTHON_BIN=\"${{PYTHON_BIN:-{_shell_quote(str(default_python_bin))}}}\"",
        (
            "EMBEDDED_TP1_LAUNCH_SCRIPT="
            "\"${EMBEDDED_TP1_LAUNCH_SCRIPT:-servers/servers-amdhpc-mi3008x-embedded-TP1/launch.py}\""
        ),
        f"TARGET_MODEL={_shell_quote(target_model)}",
        "RUN_SCRIPT_PATH=\"${SCRIPT_DIR}/run_replay.sh\"",
        "",
        "cd \"${REPO_ROOT}\"",
        "exec \"${PYTHON_BIN}\" \"${EMBEDDED_TP1_LAUNCH_SCRIPT}\" submit \\",
        "  -m \"${TARGET_MODEL}\" \\",
        "  -e \"${RUN_SCRIPT_PATH}\"",
        "",
        (
            "# Equivalent raw command: "
            f"{build_embedded_tp1_submit_command(run_script_path=run_script_path, target_model=target_model)}"
        ),
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o750)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_experiment.py",
        description=(
            "Generate a dedicated mi3008x embedded TP1 AMD sweep-QPS + sweep-SLO "
            "replay bundle with amd-power-reader, gateway ctx-aware control, "
            "gateway slo-aware control, and freq-controller-linespace-slo-amd "
            "running in aggressive SLO mode. QPS points are assigned to port "
            "profiles 0..7 in ascending order; SLO values run as sequential "
            "rounds inside the selected QPS slot."
        ),
    )
    parser.add_argument("--source-run-dir", required=True, help="Profiled source run directory under results/.")
    parser.add_argument("--poisson-seed", required=True, type=BASE.parse_non_negative_int, help="Poisson launch seed.")
    parser.add_argument("--randomize-seed", required=True, type=BASE.parse_non_negative_int, help="Replay randomization seed.")
    parser.add_argument(
        "--qps-list",
        required=True,
        help=(
            "Comma-separated Poisson rates, e.g. 0.1,0.2,0.3. "
            f"Supports at most {MAX_QPS_POINTS} values."
        ),
    )
    parser.add_argument(
        "--target-slo-list",
        required=True,
        help=(
            "Comma-separated target throughput SLO values in token/s, "
            "e.g. 8,10,12. The same QPS slot is replayed once per SLO round."
        ),
    )
    parser.add_argument("--time-constraint-s", required=True, type=float, help="Replay time limit in seconds.")
    parser.add_argument("--target-model", required=True, help="Target model key from configs/model_config.toml.")
    parser.add_argument(
        "--port-profile",
        "-P",
        default=DEFAULT_PORT_PROFILE_ID,
        type=int,
        help=(
            "Default port profile used when the generated run script is started "
            "without an explicit profile id. Runtime still accepts profile ids 0..7."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        default=None,
        help=(
            "Optional compatibility check for manual default runs. If provided, "
            "it must match --port-profile. Runtime otherwise defaults to "
            "gpu_index=port_profile_id."
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
            "Default freq-controller-linespace-slo-amd context threshold. "
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
            "<dataset-lineage>/split/<split>/<qps>/<slo>/<timestamp>/profile-<assigned-port-profile> "
            "under results/replay/amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive/. "
            "dataset-lineage is inferred from --source-run-dir by dropping "
            "the first (<model>) and last (<run-dir>) path segments."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Optional suffix appended to the replay output root directory name. "
            "For example, --output-suffix lmcache writes under results/replay/"
            "amd-embeded/servers-amdhpc-mi3008x-embedded-TP1/"
            "sweep-qps-docker-power-clean-freq-ctrl-linespace-amd-sweep-slo-aggressive-lmcache/."
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
        ensure_supported_qps_point_count(qps_values)
        target_slo_values = parse_target_slo_list(str(args.target_slo_list))
        time_constraint_s = BASE.parse_positive_float(
            str(args.time_constraint_s),
            field_name="--time-constraint-s",
        )
        default_port_profile = validate_mi3008x_port_profile(
            int(args.port_profile),
            field_name="--port-profile",
        )
        gpu_index = parse_optional_gpu_index(args.gpu_index)
        if gpu_index is not None and gpu_index != default_port_profile:
            raise ValueError(
                "--gpu-index must match --port-profile when provided for mi3008x embedded TP1"
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

        qps_slo_jobs: list[QpsSloReplayJob] = []
        for assigned_port_profile, qps in enumerate(qps_values):
            assigned_gpu_index = assigned_port_profile
            qps_slug = BASE.format_qps_slug(qps)
            qps_display = BASE.qps_display_text(qps)
            for target_slo in target_slo_values:
                target_slo_slug = format_target_slo_slug(target_slo)
                target_slo_display = display_text(target_slo)
                replay_output_dir = (
                    replay_output_root
                    / source_dataset_lineage
                    / "split"
                    / split
                    / qps_slug
                    / target_slo_slug
                    / batch_timestamp
                ).resolve()
                replay_payload = {
                    "plan": BASE.path_for_config(selected_plan_copy_path),
                    "output_dir": BASE.path_for_config(replay_output_dir),
                    "randomize_seed": int(args.randomize_seed),
                    "time_constraint_s": time_constraint_s,
                    "port_profile_id": assigned_port_profile,
                    "launch_policy_override": {
                        "seed": int(args.poisson_seed),
                        "pattern": {"name": "poisson"},
                        "pattern_args": {"rate": qps},
                    },
                }
                replay_config_path = (
                    batch_dir / qps_slug / target_slo_slug / "replay.toml"
                ).resolve()
                write_replay_config(replay_config_path, replay_payload=replay_payload)
                replay_config_relpath = str(replay_config_path.relative_to(batch_dir))

                qps_slo_jobs.append(
                    QpsSloReplayJob(
                        qps=qps,
                        qps_display=qps_display,
                        qps_slug=qps_slug,
                        assigned_port_profile=assigned_port_profile,
                        assigned_gpu_index=assigned_gpu_index,
                        target_slo=target_slo,
                        target_slo_display=target_slo_display,
                        target_slo_slug=target_slo_slug,
                        replay_config_relpath=replay_config_relpath,
                        replay_config_path=replay_config_path,
                        replay_output_dir=replay_output_dir,
                    )
                )

        run_script_path = (batch_dir / "run_replay.sh").resolve()
        write_run_script(
            run_script_path,
            default_port_profile=default_port_profile,
            target_model=target_model,
            split=split,
            qps_slo_jobs=qps_slo_jobs,
            ctx_aware_usage_threshold_tokens=ctx_aware_usage_threshold_tokens,
            ctx_aware_scheduling_threshold_tokens=ctx_aware_scheduling_threshold_tokens,
            freq_controller_threshold=freq_controller_threshold,
        )
        submit_script_path = (batch_dir / "submit_embedded_tp1.sh").resolve()
        write_submit_script(
            submit_script_path,
            run_script_path=run_script_path,
            target_model=target_model,
        )
        embedded_tp1_submit_command = build_embedded_tp1_submit_command(
            run_script_path=run_script_path,
            target_model=target_model,
        )

        qps_points: list[dict[str, Any]] = []
        for assigned_port_profile, qps in enumerate(qps_values):
            qps_slug = BASE.format_qps_slug(qps)
            qps_jobs = [
                job
                for job in qps_slo_jobs
                if job.qps_slug == qps_slug and job.assigned_port_profile == assigned_port_profile
            ]
            qps_points.append(
                {
                    "qps": qps,
                    "qps_slug": qps_slug,
                    "assigned_port_profile": assigned_port_profile,
                    "assigned_gpu_index": assigned_port_profile,
                    "slo_points": [
                        {
                            "target_slo": job.target_slo,
                            "target_slo_slug": job.target_slo_slug,
                            "slo_aware_target_tokens_per_s": job.target_slo,
                            "replay_config": str(job.replay_config_path),
                            "replay_output_dir_base": path_for_config(job.replay_output_dir),
                            "replay_output_dir": build_assigned_profile_output_dir(
                                job.replay_output_dir,
                                assigned_port_profile=job.assigned_port_profile,
                            ),
                            "power_output_dir": (
                                f"{build_assigned_profile_output_dir(job.replay_output_dir, assigned_port_profile=job.assigned_port_profile)}/power"
                            ),
                            "freq_controller_log_dir": (
                                f"{build_assigned_profile_output_dir(job.replay_output_dir, assigned_port_profile=job.assigned_port_profile)}/"
                                f"{DEFAULT_FREQ_CONTROL_LOG_DIR_NAME}"
                            ),
                        }
                        for job in qps_jobs
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
            "target_slo_list": target_slo_values,
            "max_qps_points": MAX_QPS_POINTS,
            "assigned_qps_points": len(qps_values),
            "total_slo_rounds": len(target_slo_values),
            "total_qps_slo_runs": len(qps_slo_jobs),
            "time_constraint_s": time_constraint_s,
            "port_profile": default_port_profile,
            "default_port_profile": default_port_profile,
            "gpu_index": gpu_index,
            "gpu_index_runtime_default": GPU_INDEX_RUNTIME_DEFAULT,
            "profile_output_suffix": PROFILE_OUTPUT_PLACEHOLDER,
            "ctx_aware_enabled": True,
            "ctx_aware_policy_mode": CTX_AWARE_POLICY_MODE,
            "ctx_aware_usage_threshold_tokens": ctx_aware_usage_threshold_tokens,
            "ctx_aware_scheduling_threshold_tokens": ctx_aware_scheduling_threshold_tokens,
            "slo_aware_enabled": True,
            "slo_aware_policy_mode": SLO_AWARE_POLICY_MODE,
            "slo_targets_drive_gateway_and_freq_controller": True,
            "freq_controller_aggressive_slo_control": (
                FREQ_CONTROLLER_AGGRESSIVE_SLO_CONTROL
            ),
            "freq_controller_threshold": freq_controller_threshold,
            "freq_controller_bin_default": DEFAULT_FREQ_CONTROLLER_BIN_NAME,
            "freq_controller_gateway_ipc_socket_default": (
                f"{DEFAULT_GATEWAY_CTX_IPC_SOCKET_DIR}/vllm-gateway-ctx-profile-<port_profile_id>.sock"
            ),
            "additional_suffix": additional_suffix,
            "output_suffix": output_suffix,
            "replay_output_root_dir": path_for_config(replay_output_root),
            "output_config_root_dir": str(output_config_root),
            "output_batch_dir": str(batch_dir),
            "plan_lookup_only": True,
            "selected_source_plan": str(lookup_result.selected_plan_path),
            "related_source_plans": lookup_result.related_plan_paths,
            "selected_plan_model": related_plan_models.get("selected"),
            "related_plan_models": related_plan_models,
            "selected_plan_copy": str(selected_plan_copy_path),
            "qps_points": qps_points,
            "run_script": str(run_script_path),
            "run_command_default": f"bash {path_for_config(run_script_path)}",
            "run_command_with_port_profile": (
                f"bash {path_for_config(run_script_path)} {default_port_profile}"
            ),
            "submit_script": str(submit_script_path),
            "submit_command_default": f"bash {path_for_config(submit_script_path)}",
            "embedded_tp1_submit_command": embedded_tp1_submit_command,
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
