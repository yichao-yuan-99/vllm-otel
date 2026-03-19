#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PORT_PROFILE_ID="0"
HOST="0.0.0.0"
UPSTREAM_BASE_URL="${GATEWAY_LITE_OPENAI_UPSTREAM_BASE_URL:-https://api.openai.com/v1}"
UPSTREAM_API_KEY="${GATEWAY_LITE_UPSTREAM_API_KEY:-${OPENAI_API_KEY:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --host)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --host" >&2
        exit 1
      fi
      HOST="$2"
      shift 2
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --upstream-base-url)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --upstream-base-url" >&2
        exit 1
      fi
      UPSTREAM_BASE_URL="$2"
      shift 2
      ;;
    --upstream-base-url=*)
      UPSTREAM_BASE_URL="${1#*=}"
      shift
      ;;
    --upstream-api-key)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --upstream-api-key" >&2
        exit 1
      fi
      UPSTREAM_API_KEY="$2"
      shift 2
      ;;
    --upstream-api-key=*)
      UPSTREAM_API_KEY="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
usage: bash experiments/openai/start_gateway.sh [options]

Start gateway-lite and forward traffic to an OpenAI-compatible upstream.

Options:
  --port-profile-id <id>      Port profile ID from configs/port_profiles.toml (default: 0)
  --host <host>               Bind host for gateway-lite (default: 0.0.0.0)
  --upstream-base-url <url>   Upstream OpenAI-compatible base URL (default: GATEWAY_LITE_OPENAI_UPSTREAM_BASE_URL or https://api.openai.com/v1)
  --upstream-api-key <key>    Real upstream key (default: GATEWAY_LITE_UPSTREAM_API_KEY or OPENAI_API_KEY)
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${UPSTREAM_BASE_URL}" ]]; then
  echo "--upstream-base-url cannot be empty" >&2
  exit 1
fi
if [[ -z "${UPSTREAM_API_KEY}" ]]; then
  echo "missing upstream API key: set OPENAI_API_KEY or GATEWAY_LITE_UPSTREAM_API_KEY, or pass --upstream-api-key" >&2
  exit 1
fi

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

CMD=(
  "${PYTHON_BIN}" -m gateway_lite run
  --host "${HOST}"
  --port-profile-id "${PORT_PROFILE_ID}"
  --upstream-base-url "${UPSTREAM_BASE_URL}"
  --upstream-api-key "${UPSTREAM_API_KEY}"
)

echo "=== gateway-lite openai forwarder ==="
echo "port_profile_id: ${PORT_PROFILE_ID}"
echo "host: ${HOST}"
echo "upstream_base_url: ${UPSTREAM_BASE_URL}"
echo "upstream_api_key: configured"
printf 'cmd:'
printf ' %q' "${CMD[@]}"
printf '\n'

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" exec "${CMD[@]}"
