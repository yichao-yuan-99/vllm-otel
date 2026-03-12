# Harbor Backend Resolution

This directory contains Harbor-specific logic for `con-driver`.

The generic scheduler and CLI stay responsible for:

- parsing backend-agnostic driver options
- sampling tasks
- launching subprocesses
- gateway job lifecycle handling

The Harbor backend is responsible for Harbor-specific command construction and Harbor-specific runtime resolution.

Harbor dataset downloads are cached under:

- `con-driver/.cache/harbor-datasets/`

That cache is shared across runs and should stay gitignored.

## Files

- `backend.py`: builds `harbor datasets download` and `harbor trials start` commands, and defines the shared Harbor dataset cache root.
- `runtime.py`: resolves Harbor forwarded args and Harbor launch environment from top-level driver inputs.

## What `runtime.py` Resolves

When `port_profile_id` is provided, `resolve_harbor_runtime(...)` resolves:

- `gateway_url`
  - defaults to `http://127.0.0.1:<gateway_parse_port>`
- `vllm_log_enabled`
  - defaults to `true`
- `vllm_log_endpoint`
  - always resolved as `http://127.0.0.1:<vllm_port>/metrics`
  - not user-configurable
- `resolved_agent_name`
  - from top-level `agent` / `agent_name`, or forwarded `--agent`
- `resolved_model_name`
  - probed from `http://127.0.0.1:<vllm_port>/v1/models`
  - exactly one served model is expected
- `resolved_model_context_window`
  - loaded from `configs/model_config.toml`
- `agent_base_url`
  - derived from local ports
  - for `mini-swe-agent` only: loopback hosts are rewritten to a container-reachable host
  - default container host is `192.168.5.1` (override via `CON_DRIVER_CONTAINER_HOST`)
  - `<gateway_parse_port>/v1` when gateway is enabled
  - `<vllm_port>/v1` when gateway is disabled
- synthesized Harbor forwarded args
  - `--agent <agent>`
  - `--model hosted_vllm/<served_model_name>`
  - `--agent-kwarg api_base=<agent_base_url>`
  - `--agent-kwarg base_url=<agent_base_url>`
  - `--agent-kwarg model_info=<json>`
  - extra agent defaults from `con-driver/configs/agent_defaults.toml`
- synthesized trial environment variables
  - `ANTHROPIC_BASE_URL`
  - `OPENAI_BASE_URL`
  - `LLM_BASE_URL`
  - `BASE_URL`
  - `OPENAI_API_BASE`
  - `HOSTED_VLLM_API_BASE`
  - `VLLM_API_BASE`

## `model_info` Resolution

`model_info` is always synthesized for Harbor in simplified mode.

It is built as:

- `max_input_tokens = context_window`
- `max_output_tokens = context_window`
- `input_cost_per_token = 0.0`
- `output_cost_per_token = 0.0`

The `context_window` must exist in `configs/model_config.toml` for the probed served model.

## Agent Defaults

Agent-specific Harbor defaults are stored in:

- `con-driver/configs/agent_defaults.toml`

These are applied after the auto-managed Harbor args are synthesized. Current usage is for Harbor agent kwargs such as Terminus trajectory settings.

## Validation Rules

When `port_profile_id` is set, Harbor resolution rejects:

- forwarded `--model`
- forwarded `--agent-kwarg api_base=...`
- forwarded `--agent-kwarg base_url=...`
- forwarded `--agent-kwarg model_info=...`
- top-level `agent` together with forwarded `--agent`

Con-driver keeps Harbor-specific auto-resolution authoritative when simplified mode is active.

## Inputs Still Required From The User

In the simplified Harbor path, the user still supplies:

- `pool`
- `pattern`
- `max_concurrent`
- `n_task`
- `results_dir`
- `port_profile_id`
- `agent`

Optional user overrides still allowed:

- `gateway`
- `gateway_url`
- `gateway_job_output_root`
- `gateway_timeout_s`
- `vllm_log`
- `vllm_log_interval_s`
- `vllm_log_timeout_s`
- additional Harbor forwarded args, as long as they do not override Harbor-managed fields
