# Hardcoded AMD HPC

This directory contains simple hardcoded `sbatch` scripts for single vLLM
services with a local Jaeger sidecar.

Common behavior:

- tensor parallel size: `8`
- GPUs: `0,1,2,3,4,5,6,7`
- login host reverse tunnel: `login`
- Jaeger UI local port: `16686`
- Jaeger OTLP local port: `4317`

Related note:

- `servers/hardcoded-amdhpc/README.remote-host.md`: local single-profile
  forwarding daemon for the hardcoded service endpoint.

The only intended submit-time knob is `PORT_PROFILE_ID`, which is resolved from
`configs/port_profiles.toml` for:

- `vllm_port`
- `jaeger_api_port`
- `jaeger_otlp_port`

## Available Scripts

`start-single-service-kimi-k2.6-mi3008x.sbatch`

- partition: `mi3008x`
- model: `moonshotai/Kimi-K2.6`

```bash
sbatch --chdir=servers/hardcoded-amdhpc \
  servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3008x.sbatch
```

`start-single-service-kimi-k2.6-mi3258x.sbatch`

- partition: `mi3258x`
- model: `moonshotai/Kimi-K2.6`

```bash
sbatch --chdir=servers/hardcoded-amdhpc \
  servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3258x.sbatch
```

`start-single-service-glm-5.1-fp8-mi3008x.sbatch`

- partition: `mi3008x`
- model: `zai-org/GLM-5.1-FP8`

```bash
sbatch --chdir=servers/hardcoded-amdhpc \
  servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3008x.sbatch
```

`start-single-service-glm-5.1-fp8-mi3258x.sbatch`

- partition: `mi3258x`
- model: `zai-org/GLM-5.1-FP8`

```bash
sbatch --chdir=servers/hardcoded-amdhpc \
  servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3258x.sbatch
```

`start-single-service-glm-5.1-mi3258x.sbatch`

- partition: `mi3258x`
- model: `zai-org/GLM-5.1`

```bash
sbatch --chdir=servers/hardcoded-amdhpc \
  servers/hardcoded-amdhpc/start-single-service-glm-5.1-mi3258x.sbatch
```

Use a different port profile if needed:

```bash
sbatch --chdir=servers/hardcoded-amdhpc \
  --export=ALL,PORT_PROFILE_ID=0 \
  servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3008x.sbatch
sbatch --chdir=servers/hardcoded-amdhpc \
  --export=ALL,PORT_PROFILE_ID=1 \
  servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3258x.sbatch
sbatch --chdir=servers/hardcoded-amdhpc \
  --export=ALL,PORT_PROFILE_ID=2 \
  servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3008x.sbatch
sbatch --chdir=servers/hardcoded-amdhpc \
  --export=ALL,PORT_PROFILE_ID=3 \
  servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3258x.sbatch
sbatch --chdir=servers/hardcoded-amdhpc \
  --export=ALL,PORT_PROFILE_ID=4 \
  servers/hardcoded-amdhpc/start-single-service-glm-5.1-mi3258x.sbatch
```

## Direct Run on an Allocated Node

If you already have a shell on an allocated node, you can run the same script
directly with `bash` instead of submitting it with `sbatch`.

- Run the script that matches the node partition you were allocated.
- The launcher stays in the foreground until you stop it with `Ctrl-C`.
- The script still opens reverse tunnels back to `login`.
- When not started by Slurm, `RUN_ID` defaults to
  `hardcoded-<utc timestamp>` unless you override it.

If you still need an interactive shell on a node, one example is:

```bash
salloc -p mi3008x -N 1 -t 12:00:00
srun --pty bash
```

From an allocated `mi3008x` node:

```bash
cd /home1/yichaoy/vllm-otel
bash servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3008x.sbatch
bash servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3008x.sbatch
```

From an allocated `mi3258x` node:

```bash
cd /home1/yichaoy/vllm-otel
bash servers/hardcoded-amdhpc/start-single-service-kimi-k2.6-mi3258x.sbatch
bash servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3258x.sbatch
bash servers/hardcoded-amdhpc/start-single-service-glm-5.1-mi3258x.sbatch
```

You can still override the port profile when launching directly:

```bash
cd /home1/yichaoy/vllm-otel
PORT_PROFILE_ID=3 \
bash servers/hardcoded-amdhpc/start-single-service-glm-5.1-fp8-mi3258x.sbatch
```

## Logs

- Jaeger log: `logs/jaeger.<run_id>.log`
- Slurm stdout: `logs/slurm.<job_id>.out`
- Slurm stderr: `logs/slurm.<job_id>.err`
- vLLM log: `logs/vllm.<run_id>.log`
