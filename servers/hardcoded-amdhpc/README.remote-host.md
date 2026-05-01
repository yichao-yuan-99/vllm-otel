# Hardcoded Remote Port Forwarding

This note documents the minimal hardcoded port-forward daemon in
`servers/hardcoded-amdhpc/port_forward_d.py`.

It is the hardcoded equivalent of the remote/local forwarding behavior used by
`servers/servers-amdhpc`, but trimmed down to one port profile and one vLLM
port per daemon.

## What It Does

Each hardcoded `sbatch` job reverse-tunnels its vLLM port from the compute node
back to the login node.

`port_forward_d.py` then opens one local SSH forward from your current machine
to that login-node port:

- local `127.0.0.1:<profile.vllm_port>`
- forwarded through SSH to `127.0.0.1:<profile.vllm_port>` on the remote host

One daemon handles one port profile. `start` refuses to launch if a live daemon
already exists for that profile.

The hardcoded jobs also reverse-tunnel Jaeger ports back to the login node, but
this helper only forwards the vLLM API port.

## Start

Default remote host is `amd-hpc`:

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py start \
  -P 0
```

If your SSH target should be something else, use the same alias style as
`servers-amdhpc`:

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py start \
  -r some-other-host \
  -P 0
```

Equivalent long option:

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py start \
  --remote-host some-other-host \
  --port-profile 0
```

The daemon returns JSON including:

- `vllm_url`
- `pid_file`
- `log_file`
- `ssh_command`

## Status

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py status \
  -P 0
```

## Check Health

This probes the local forwarded vLLM endpoint at
`http://127.0.0.1:<profile.vllm_port>/v1/models` and returns JSON. A healthy
result includes the first served model name.

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py check-health \
  -P 0
```

## Stop

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py stop \
  -P 0
```

## Example Flow

1. Start a hardcoded service job, for example:

```bash
sbatch servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3008x.sbatch
```

2. Start the local forwarding daemon for the same port profile:

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py start -P 0
```

3. Check that the forwarded vLLM endpoint is healthy:

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py check-health -P 0
```

4. Use the local forwarded endpoint:

```bash
curl http://127.0.0.1:11451/v1/models
```

5. Stop the local daemon when finished:

```bash
python3 servers/hardcoded-amdhpc/port_forward_d.py stop -P 0
```

## Notes

- This daemon only forwards the vLLM API port.
- It does not start or stop the Slurm job.
- It does not start if a live daemon already exists for the selected profile.
- Runtime pid/log files live under `~/.cache/vllm-otel-apptainer/hardcoded-port-forward-d/` by default.
