#!/usr/bin/env bash

# Shared helper for interactive mi3001x experiment scripts.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This file is meant to be sourced from another bash script." >&2
  exit 64
fi

_INTERACTIVE_EMBEDDED_TP1_HELPER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_INTERACTIVE_EMBEDDED_TP1_REPO_ROOT_DEFAULT=$(
  cd -- "${_INTERACTIVE_EMBEDDED_TP1_HELPER_DIR}/../.." && pwd
)

INTERACTIVE_EMBEDDED_TP1_REPO_ROOT="${REPO_ROOT:-${_INTERACTIVE_EMBEDDED_TP1_REPO_ROOT_DEFAULT}}"
INTERACTIVE_CLIENT_SCRIPT="${INTERACTIVE_CLIENT_SCRIPT:-servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/client.py}"
INTERACTIVE_START_SERVICES_SCRIPT="${INTERACTIVE_START_SERVICES_SCRIPT:-servers/servers-amdhpc-interactive-mi3001x-embedded-TP1/start-services.sh}"
INTERACTIVE_START_SERVICES_COMMAND="${INTERACTIVE_START_SERVICES_COMMAND:-python3 ${INTERACTIVE_CLIENT_SCRIPT} start}"
SERVICE_READY_TIMEOUT_SECONDS="${SERVICE_READY_TIMEOUT_SECONDS:-120}"
SERVICE_READY_POLL_INTERVAL_SECONDS="${SERVICE_READY_POLL_INTERVAL_SECONDS:-2}"

INTERACTIVE_EMBEDDED_TP1_PROFILE_PORTS_RESOLVED=0
INTERACTIVE_EMBEDDED_TP1_PROFILE_PORTS_PROFILE_ID=""
INTERACTIVE_EMBEDDED_TP1_PROFILE_VLLM_PORT=""
INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT=""
INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PARSE_PORT=""

interactive_embedded_tp1_probe_http_url() {
  local url="$1"
  local python_bin="${PYTHON_BIN:-python3}"
  "${python_bin}" -c "import sys, urllib.request; req = urllib.request.Request(sys.argv[1], method='GET'); resp = urllib.request.urlopen(req, timeout=3); sys.exit(0 if int(resp.status) == 200 else 1)" "$url" >/dev/null 2>&1
}

interactive_embedded_tp1_probe_tcp_port() {
  local host="$1"
  local port="$2"
  local python_bin="${PYTHON_BIN:-python3}"
  "${python_bin}" -c "import socket, sys; sock = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2); sock.close()" "$host" "$port" >/dev/null 2>&1
}

interactive_embedded_tp1_normalize_port_profile_id() {
  local raw_value="$1"
  local expected_value="${2:-0}"
  local log_tag="${3:-interactive-embedded-tp1}"
  local normalized_value="${raw_value//[[:space:]]/}"

  if [[ -z "${normalized_value}" || ! "${normalized_value}" =~ ^[0-9]+$ ]]; then
    echo "[${log_tag}] error: invalid port profile id: ${raw_value}" >&2
    return 1
  fi
  if [[ "${normalized_value}" != "${expected_value}" ]]; then
    echo "[${log_tag}] error: this mi3001x workflow only supports port profile ${expected_value}" >&2
    return 1
  fi
  printf '%s\n' "${normalized_value}"
}

interactive_embedded_tp1_normalize_gpu_index() {
  local raw_value="$1"
  local expected_value="${2:-0}"
  local log_tag="${3:-interactive-embedded-tp1}"
  local normalized_value="${raw_value//[[:space:]]/}"

  if [[ -z "${normalized_value}" || ! "${normalized_value}" =~ ^[0-9]+$ ]]; then
    echo "[${log_tag}] error: invalid gpu index: ${raw_value}" >&2
    return 1
  fi
  if [[ "${normalized_value}" != "${expected_value}" ]]; then
    echo "[${log_tag}] error: this mi3001x workflow only supports gpu index ${expected_value}" >&2
    return 1
  fi
  printf '%s\n' "${normalized_value}"
}

interactive_embedded_tp1_resolve_profile_ports() {
  local port_profile_id="$1"
  local log_tag="${2:-interactive-embedded-tp1}"
  local python_bin="${PYTHON_BIN:-python3}"

  if [[ "${INTERACTIVE_EMBEDDED_TP1_PROFILE_PORTS_RESOLVED}" -eq 1 && "${INTERACTIVE_EMBEDDED_TP1_PROFILE_PORTS_PROFILE_ID}" == "${port_profile_id}" ]]; then
    return 0
  fi

  local resolved_output=""
  if ! resolved_output="$("${python_bin}" - "${INTERACTIVE_EMBEDDED_TP1_REPO_ROOT}" "${port_profile_id}" <<'PY'
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

repo_root = Path(sys.argv[1]).resolve()
port_profile_id = str(sys.argv[2])
payload = tomllib.loads((repo_root / "configs" / "port_profiles.toml").read_text(encoding="utf-8"))
profiles = payload.get("profiles")
if not isinstance(profiles, dict):
    raise SystemExit("configs/port_profiles.toml must include [profiles]")
profile = profiles.get(port_profile_id)
if not isinstance(profile, dict):
    raise SystemExit(f"unknown port profile id: {port_profile_id}")
for field in ("vllm_port", "gateway_port", "gateway_parse_port"):
    value = profile.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"invalid {field} for profile {port_profile_id}")
    print(value)
PY
  )"; then
    echo "[${log_tag}] error: failed to resolve ports for ${port_profile_id}" >&2
    return 1
  fi

  local resolved_fields=()
  mapfile -t resolved_fields <<<"${resolved_output}"
  if [[ "${#resolved_fields[@]}" -ne 3 ]]; then
    echo "[${log_tag}] error: unexpected port resolution output for profile ${port_profile_id}" >&2
    return 1
  fi

  INTERACTIVE_EMBEDDED_TP1_PROFILE_VLLM_PORT="${resolved_fields[0]}"
  INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT="${resolved_fields[1]}"
  INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PARSE_PORT="${resolved_fields[2]}"
  INTERACTIVE_EMBEDDED_TP1_PROFILE_PORTS_PROFILE_ID="${port_profile_id}"
  INTERACTIVE_EMBEDDED_TP1_PROFILE_PORTS_RESOLVED=1
  return 0
}

interactive_embedded_tp1_wait_for_services() {
  local port_profile_id="$1"
  local log_tag="${2:-interactive-embedded-tp1}"

  interactive_embedded_tp1_resolve_profile_ports "${port_profile_id}" "${log_tag}" || return 1

  local deadline=$((SECONDS + SERVICE_READY_TIMEOUT_SECONDS))
  local vllm_url="http://127.0.0.1:${INTERACTIVE_EMBEDDED_TP1_PROFILE_VLLM_PORT}/v1/models"
  echo "[${log_tag}] waiting for interactive services profile=${port_profile_id} vllm=${INTERACTIVE_EMBEDDED_TP1_PROFILE_VLLM_PORT} gateway=${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT} parse=${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PARSE_PORT}"
  while (( SECONDS < deadline )); do
    if interactive_embedded_tp1_probe_http_url "${vllm_url}"; then
      if [[ "${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT}" -eq "${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PARSE_PORT}" ]]; then
        if interactive_embedded_tp1_probe_tcp_port "127.0.0.1" "${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT}"; then
          return 0
        fi
      else
        if interactive_embedded_tp1_probe_tcp_port "127.0.0.1" "${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT}" && interactive_embedded_tp1_probe_tcp_port "127.0.0.1" "${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PARSE_PORT}"; then
          return 0
        fi
      fi
    fi
    sleep "${SERVICE_READY_POLL_INTERVAL_SECONDS}"
  done

  echo "[${log_tag}] error: interactive services on port profile ${port_profile_id} were not ready. Start them with: ${INTERACTIVE_START_SERVICES_COMMAND}" >&2
  return 1
}

interactive_embedded_tp1_resolve_gateway_base_url() {
  local port_profile_id="$1"
  local log_tag="${2:-interactive-embedded-tp1}"

  if [[ -n "${GATEWAY_BASE_URL:-}" ]]; then
    printf '%s\n' "${GATEWAY_BASE_URL}"
    return 0
  fi

  interactive_embedded_tp1_resolve_profile_ports "${port_profile_id}" "${log_tag}" || return 1
  printf 'http://127.0.0.1:%s\n' "${INTERACTIVE_EMBEDDED_TP1_PROFILE_GATEWAY_PORT}"
}
