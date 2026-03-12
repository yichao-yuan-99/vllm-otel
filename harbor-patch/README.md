# Harbor Mini-SWE Hosted-vLLM Patch

This document records the local Harbor library patch that enables `mini-swe-agent`
to run with `--model hosted_vllm/<served_model_name>` directly.

## Why this patch is needed

Without this patch:

- Harbor mini-swe adapter raises `Unknown provider: hosted_vllm` during API-key env resolution.
- Harbor mini-swe adapter only forwards `OPENAI_API_BASE` to the in-container `mini` process,
  so `HOSTED_VLLM_API_BASE` can be dropped even if it is set upstream.

Result: `hosted_vllm/...` model configs fail unless con-driver rewrites model prefixes.

## Patched files

1. `/home/yichaoy2/.local/share/uv/tools/harbor/lib/python3.12/site-packages/harbor/agents/utils.py`
2. `/home/yichaoy2/.local/share/uv/tools/harbor/lib/python3.12/site-packages/harbor/agents/installed/mini_swe_agent.py`

## Changes made

### 1) Provider recognition for hosted_vllm

In `harbor/agents/utils.py`:

- Added provider key mapping:
  - `"hosted_vllm": "OPENAI_API_KEY"`

This prevents `Unknown provider: hosted_vllm` and keeps API key handling aligned
with OpenAI-compatible local endpoints.

### 2) Pass hosted_vllm API base/key into mini process

In `harbor/agents/installed/mini_swe_agent.py` (`create_run_agent_commands`):

- Forward both API-base env vars when present:
  - `OPENAI_API_BASE`
  - `HOSTED_VLLM_API_BASE`
- If only `HOSTED_VLLM_API_BASE` is present, also set:
  - `OPENAI_API_BASE = HOSTED_VLLM_API_BASE`
- Forward `HOSTED_VLLM_API_KEY` if set.

This ensures the in-container `mini` process can see hosted-vLLM-specific routing
inputs and still works with stacks that only read `OPENAI_API_BASE`.

### 3) Runtime LiteLLM cost-map injection for hosted_vllm models

In `harbor/agents/installed/mini_swe_agent.py` (`create_run_agent_commands`):

- Set:
  - `MSWEA_MODEL_NAME=<selected model>`
  - `LITELLM_LOCAL_MODEL_COST_MAP=True`
- Prepend a Python bootstrap step before `mini ...` that updates:
  - `/root/.local/share/uv/tools/mini-swe-agent/lib/python*/site-packages/litellm/model_prices_and_context_window_backup.json`
- Inject zero-cost entries for aliases of the selected model (including
  `hosted_vllm/<name>` and `<name>`), with:
  - `input_cost_per_token=0.0`
  - `output_cost_per_token=0.0`
  - `mode="chat"`
  - `max_tokens/max_input_tokens/max_output_tokens=262144`

This prevents mini-swe-agent from crashing on:

- `This model isn't mapped yet. model=<...>, custom_llm_provider=hosted_vllm`

## Runtime behavior after patch

With the patch, this flow is supported end-to-end:

1. Harbor receives `-m hosted_vllm/<model>`.
2. Harbor mini-swe adapter resolves provider/env vars without error.
3. `mini` process receives hosted-vLLM base URL env and API key env.
4. Runtime bootstrap ensures LiteLLM has cost metadata for the selected hosted-vLLM model.
5. LiteLLM routes calls using `hosted_vllm` provider inputs directly.

## Related con-driver change

Con-driver no longer needs to rewrite `hosted_vllm/...` to `openai/...` for
`mini-swe-agent` compatibility after this Harbor patch.
