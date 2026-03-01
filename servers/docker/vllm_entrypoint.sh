#!/usr/bin/env bash
set -euo pipefail

VLLM_MODEL_EXTRA_ARGS=()

load_model_extra_args() {
  local encoded="${VLLM_MODEL_EXTRA_ARGS_B64:-}"
  local tmp_file=""
  if [[ -z "${encoded}" ]]; then
    return 0
  fi

  tmp_file="$(mktemp)"

  VLLM_MODEL_EXTRA_ARGS_B64="${encoded}" python3 - "${tmp_file}" <<'PY'
import base64
import json
import os
import sys

encoded = os.environ["VLLM_MODEL_EXTRA_ARGS_B64"]
try:
    decoded = base64.b64decode(encoded.encode("ascii"), validate=True)
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"error: invalid VLLM_MODEL_EXTRA_ARGS_B64 payload: {exc}")

try:
    extra_args = json.loads(decoded)
except json.JSONDecodeError as exc:
    raise SystemExit(f"error: VLLM_MODEL_EXTRA_ARGS_B64 did not decode to valid JSON: {exc}")

if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
    raise SystemExit("error: decoded VLLM_MODEL_EXTRA_ARGS_B64 payload must be a JSON string array")

with open(sys.argv[1], "wb") as handle:
    for arg in extra_args:
        handle.write(arg.encode("utf-8"))
        handle.write(b"\0")
PY

  mapfile -d '' -t VLLM_MODEL_EXTRA_ARGS < "${tmp_file}"
  rm -f "${tmp_file}"
}

parse_model_name() {
  local args=("$@")
  local model_name=""
  local idx=0
  local argc="${#args[@]}"

  while [[ "${idx}" -lt "${argc}" ]]; do
    local arg="${args[$idx]}"
    case "${arg}" in
      --model)
        idx=$((idx + 1))
        if [[ "${idx}" -ge "${argc}" ]]; then
          echo "error: --model requires a value" >&2
          return 1
        fi
        model_name="${args[$idx]}"
        ;;
      --model=*)
        model_name="${arg#--model=}"
        ;;
    esac
    idx=$((idx + 1))
  done

  if [[ -z "${model_name}" && -n "${VLLM_MODEL_NAME:-}" ]]; then
    model_name="${VLLM_MODEL_NAME}"
  fi
  if [[ -z "${model_name}" ]]; then
    echo "error: could not infer model name from args; pass --model <name>" >&2
    return 1
  fi

  printf '%s' "${model_name}"
}

detect_trust_remote_code() {
  local args=("$@")
  local idx=0
  local argc="${#args[@]}"

  while [[ "${idx}" -lt "${argc}" ]]; do
    local arg="${args[$idx]}"
    case "${arg}" in
      --trust-remote-code)
        printf 'true'
        return 0
        ;;
      --trust-remote-code=1|--trust-remote-code=true|--trust-remote-code=yes|--trust-remote-code=on)
        printf 'true'
        return 0
        ;;
    esac
    idx=$((idx + 1))
  done

  if [[ "${VLLM_FORCE_SEQ_TRUST_REMOTE_CODE:-}" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Oo][Nn])$ ]]; then
    printf 'true'
    return 0
  fi

  printf 'false'
}

load_model_extra_args
ALL_ARGS=("$@" "${VLLM_MODEL_EXTRA_ARGS[@]}")

MODEL_NAME="$(parse_model_name "${ALL_ARGS[@]}")"
TRUST_REMOTE_CODE="$(detect_trust_remote_code "${ALL_ARGS[@]}")"
EOS_ID="$(
  MODEL_NAME="${MODEL_NAME}" TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" python3 - <<'PY'
import os
from transformers import AutoTokenizer

model_name = os.environ["MODEL_NAME"]
trust_remote_code = os.environ["TRUST_REMOTE_CODE"].strip().lower() == "true"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=trust_remote_code,
)
eos_token_id = tokenizer.eos_token_id
if eos_token_id is None:
    raise RuntimeError(f"Tokenizer for model {model_name!r} has no eos_token_id")
print(eos_token_id)
PY
)"

export VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID="${EOS_ID}"
echo "bootstrap: model=${MODEL_NAME} trust_remote_code=${TRUST_REMOTE_CODE} eos_token_id=${VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID} extra_args_count=${#VLLM_MODEL_EXTRA_ARGS[@]}" >&2

exec python3 -m vllm.entrypoints.openai.api_server "${ALL_ARGS[@]}"
