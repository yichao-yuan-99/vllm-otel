# Server Smoke Test Runner

This package runs smoke tests against a running local `vllm + jaeger` stack.

Input selection is by port profile ID from `configs/port_profiles.toml`.

Structure:

- `servers/test/client.py`: entrypoint/orchestrator
- `servers/test/test_jaeger_otlp.py`: Jaeger OTLP reachability test
- `servers/test/test_inference.py`: basic vLLM inference test
- `servers/test/test_force_sequence.py`: force-sequence behavior test
- `servers/test/test_jaeger_service.py`: Jaeger service-index verification

## Usage

From repo root:

```bash
python3 servers/test/client.py --port-profile 0
```

Example with custom timeouts:

```bash
python3 servers/test/client.py \
  --port-profile 0 \
  --startup-timeout-seconds 900 \
  --request-timeout-seconds 30 \
  --trace-wait-seconds 45
```

## What it checks

- Jaeger OTLP port is reachable.
- vLLM `/v1/models` is up and returns a model.
- Basic chat-completions inference succeeds.
- Force-sequence logits processor behavior works.
- Jaeger service index contains `vllm-server`.
