#!/usr/bin/env bash
set -euo pipefail

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

MODEL_NAME="$(parse_model_name "$@")"
EOS_ID="$(
  MODEL_NAME="${MODEL_NAME}" python3 - <<'PY'
import os
from transformers import AutoTokenizer

model_name = os.environ["MODEL_NAME"]
trust_remote_code = os.getenv("VLLM_FORCE_SEQ_TRUST_REMOTE_CODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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
echo "bootstrap: model=${MODEL_NAME} eos_token_id=${VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID}" >&2

exec python3 -m vllm.entrypoints.openai.api_server "$@"
