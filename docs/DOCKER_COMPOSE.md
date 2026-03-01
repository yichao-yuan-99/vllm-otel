# Docker Runtime Operation (CLI-Only)

Runtime launch and operations are managed through `servers/servers-docker/client.py`.
Manual `docker compose up/down/logs` workflows are intentionally not part of this runbook.

## Setup

From repo root:

`servers/servers-docker/client.py` materializes compose env vars internally. There is no manual docker `.env` step.

Export Hugging Face env vars in your current shell:

```bash
export HF_HOME=/path/to/hf-home
export HF_HUB_CACHE=/path/to/hf-hub-cache
export HF_TOKEN=your_hf_token
```

## Profiles

```bash
python3 servers/servers-docker/client.py profiles models
python3 servers/servers-docker/client.py profiles ports
python3 servers/servers-docker/client.py profiles launches
```

## Start

```bash
python3 servers/servers-docker/client.py start -m qwen3_coder_30b -p 0 -l h100_nvl_gpu23 -b
```

## Operate

```bash
python3 servers/servers-docker/client.py status
python3 servers/servers-docker/client.py up
python3 servers/servers-docker/client.py wait-up --timeout-seconds 900
python3 servers/servers-docker/client.py logs -n 200
python3 servers/test/client.py --port-profile 0
```

## Stop

```bash
python3 servers/servers-docker/client.py stop -b
python3 servers/servers-docker/client.py daemon-stop
```
