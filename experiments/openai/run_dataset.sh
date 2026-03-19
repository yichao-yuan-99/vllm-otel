#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_PORT_PROFILE_ID="0"
DEFAULT_AGENT="mini-swe-agent"
DEFAULT_MODEL="openai/gpt-4.1-mini"
DEFAULT_MAX_CONCURRENT="1"
DEFAULT_RESULTS_ROOT="experiments/results/openai"
DEFAULT_GATEWAY_HOST="127.0.0.1"

PORT_PROFILE_ID="${DEFAULT_PORT_PROFILE_ID}"
DATASET=""
AGENT="${DEFAULT_AGENT}"
MODEL="${DEFAULT_MODEL}"
MAX_CONCURRENT="${DEFAULT_MAX_CONCURRENT}"
RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
GATEWAY_HOST="${DEFAULT_GATEWAY_HOST}"
GATEWAY_URL=""
DRY_RUN="false"

list_datasets() {
  cat <<'EOF'
terminal-bench@2.0
livecodebench
dabstep
swebench-verified
EOF
}

dataset_slug() {
  case "$1" in
    terminal-bench@2.0) printf '%s\n' "terminal-bench-2.0" ;;
    livecodebench) printf '%s\n' "livecodebench" ;;
    dabstep) printf '%s\n' "dabstep" ;;
    swebench-verified) printf '%s\n' "swebench-verified" ;;
    *) return 1 ;;
  esac
}

dataset_n_task() {
  case "$1" in
    terminal-bench@2.0) printf '%s\n' "89" ;;
    livecodebench) printf '%s\n' "100" ;;
    dabstep) printf '%s\n' "450" ;;
    swebench-verified) printf '%s\n' "500" ;;
    *) return 1 ;;
  esac
}

resolve_gateway_parse_port() {
  local profile_id="$1"
  local python_bin="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="python3"
  fi

  "${python_bin}" - "${REPO_ROOT}/configs/port_profiles.toml" "${profile_id}" <<'PY'
import pathlib
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

config_path = pathlib.Path(sys.argv[1])
profile_id = str(sys.argv[2])
payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
profiles = payload.get("profiles")
if not isinstance(profiles, dict):
    raise SystemExit("configs/port_profiles.toml is missing [profiles]")
profile = profiles.get(profile_id)
if not isinstance(profile, dict):
    raise SystemExit(f"unknown port profile id: {profile_id}")
port = profile.get("gateway_parse_port")
if isinstance(port, bool) or not isinstance(port, int):
    raise SystemExit(f"invalid gateway_parse_port for profile {profile_id}")
print(port)
PY
}

resolve_agent_api_base() {
  local gateway_url="$1"
  local container_host="${CON_DRIVER_CONTAINER_HOST:-192.168.5.1}"
  local python_bin="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="python3"
  fi

  "${python_bin}" - "${gateway_url}" "${container_host}" <<'PY'
import sys
from urllib.parse import urlsplit, urlunsplit

gateway_url = sys.argv[1].strip()
container_host = sys.argv[2].strip() or "192.168.5.1"
parsed = urlsplit(gateway_url)

if parsed.scheme not in {"http", "https"} or not parsed.netloc:
    raise SystemExit(f"invalid gateway URL: {gateway_url}")

host = parsed.hostname
if host is None:
    raise SystemExit(f"gateway URL is missing host: {gateway_url}")

if host in {"127.0.0.1", "localhost", "::1"}:
    resolved_host = container_host
else:
    resolved_host = host

if ":" in resolved_host and not resolved_host.startswith("["):
    resolved_host = f"[{resolved_host}]"

netloc = resolved_host if parsed.port is None else f"{resolved_host}:{parsed.port}"

base_path = parsed.path.rstrip("/")
if base_path.endswith("/v1"):
    api_path = base_path
elif base_path:
    api_path = f"{base_path}/v1"
else:
    api_path = "/v1"

print(urlunsplit((parsed.scheme, netloc, api_path, "", "")))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --dataset" >&2
        exit 1
      fi
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --port-profile-id)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --port-profile-id" >&2
        exit 1
      fi
      PORT_PROFILE_ID="$2"
      shift 2
      ;;
    --port-profile-id=*)
      PORT_PROFILE_ID="${1#*=}"
      shift
      ;;
    --agent)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --agent" >&2
        exit 1
      fi
      AGENT="$2"
      shift 2
      ;;
    --agent=*)
      AGENT="${1#*=}"
      shift
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --model" >&2
        exit 1
      fi
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --max-concurrent)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --max-concurrent" >&2
        exit 1
      fi
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --max-concurrent=*)
      MAX_CONCURRENT="${1#*=}"
      shift
      ;;
    --results-root)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --results-root" >&2
        exit 1
      fi
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --results-root=*)
      RESULTS_ROOT="${1#*=}"
      shift
      ;;
    --gateway-host)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --gateway-host" >&2
        exit 1
      fi
      GATEWAY_HOST="$2"
      shift 2
      ;;
    --gateway-host=*)
      GATEWAY_HOST="${1#*=}"
      shift
      ;;
    --gateway-url)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --gateway-url" >&2
        exit 1
      fi
      GATEWAY_URL="$2"
      shift 2
      ;;
    --gateway-url=*)
      GATEWAY_URL="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --list-datasets)
      list_datasets
      exit 0
      ;;
    -h|--help)
      cat <<'EOF'
usage: bash experiments/openai/run_dataset.sh --dataset <name> [options]

Run one benchmark dataset through gateway-lite -> OpenAI using con-driver.

Required:
  --dataset <name>            terminal-bench@2.0 | livecodebench | dabstep | swebench-verified

Options:
  --port-profile-id <id>      Profile used to resolve gateway_parse_port (default: 0)
  --agent <name>              Harbor agent name (default: mini-swe-agent)
  --model <name>              Harbor model string (default: openai/gpt-4.1-mini)
  --max-concurrent <n>        Max concurrent trials (default: 1)
  --results-root <path>       Root results directory (default: experiments/results/openai)
  --gateway-host <host>       Host for computed gateway control URL (default: 127.0.0.1)
  --gateway-url <url>         Explicit gateway control URL override (default: http://<gateway-host>:<gateway_parse_port>)
  --dry-run                   Pass --dry-run to con-driver
  --list-datasets             Print supported datasets

Notes:
  Agent API base is auto-derived from the gateway URL and exported via
  OPENAI_API_BASE / HOSTED_VLLM_API_BASE / OPENAI_BASE_URL.
  Loopback hosts are rewritten for container reachability using
  CON_DRIVER_CONTAINER_HOST (default: 192.168.5.1).
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  echo "missing required --dataset" >&2
  exit 1
fi

if ! DATASET_SLUG="$(dataset_slug "${DATASET}")"; then
  echo "unknown dataset: ${DATASET}" >&2
  echo "supported datasets:" >&2
  list_datasets >&2
  exit 1
fi

if ! N_TASK="$(dataset_n_task "${DATASET}")"; then
  echo "failed to resolve n_task for dataset: ${DATASET}" >&2
  exit 1
fi

if ! [[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrent must be a positive integer" >&2
  exit 1
fi

if [[ -z "${GATEWAY_URL}" ]]; then
  GATEWAY_PARSE_PORT="$(resolve_gateway_parse_port "${PORT_PROFILE_ID}")"
  GATEWAY_URL="http://${GATEWAY_HOST}:${GATEWAY_PARSE_PORT}"
fi
AGENT_API_BASE="$(resolve_agent_api_base "${GATEWAY_URL}")"

RESULTS_DIR="${RESULTS_ROOT}/${DATASET_SLUG}/${AGENT}"
mkdir -p "${REPO_ROOT}/${RESULTS_DIR}"

CMD=(
  env
  "OPENAI_API_BASE=${AGENT_API_BASE}"
  "HOSTED_VLLM_API_BASE=${AGENT_API_BASE}"
  "OPENAI_BASE_URL=${AGENT_API_BASE}"
  bash "${REPO_ROOT}/con-driver/run_con_driver.sh"
  --driver-backend harbor
  --pool "${DATASET}"
  --pattern eager
  --max-concurrent "${MAX_CONCURRENT}"
  --n-task "${N_TASK}"
  --results-dir "${RESULTS_DIR}"
  --sample-without-replacement
  --gateway
  --gateway-url "${GATEWAY_URL}"
)
if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi
CMD+=(-- --agent "${AGENT}" --model "${MODEL}")

echo "=== openai dataset run ==="
echo "dataset: ${DATASET}"
echo "n_task: ${N_TASK}"
echo "agent: ${AGENT}"
echo "model: ${MODEL}"
echo "port_profile_id: ${PORT_PROFILE_ID}"
echo "gateway_url: ${GATEWAY_URL}"
echo "agent_api_base: ${AGENT_API_BASE}"
echo "max_concurrent: ${MAX_CONCURRENT}"
echo "results_dir: ${RESULTS_DIR}"
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "dry_run: true"
fi
printf 'cmd:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
