# Standalone AMD HPC

This directory contains minimal `sbatch` files for one standalone vLLM
service on a single AMD HPC node, plus two tiny Python helpers for port and
model resolution.

Related note:

- `servers/standalone-amdhpc/README.remote-host.md`: local single-profile
  forwarding daemon for remote usage.

The job launches only one vLLM process. By default it uses the full node:

- `VLLM_TENSOR_PARALLEL_SIZE=8`
- `ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

There is no multi-profile fan-out, gateway, Jaeger, or experiment runner in
this directory. Each standalone job reverse-tunnels its vLLM port back to the
login node, following the same basic compute-to-login pattern used by
`servers/servers-amdhpc`.
Because these standalone jobs only launch one service, the tunneled port set is
the single vLLM API port.

## Submit

Submit from the repo root and use `--chdir=servers/standalone-amdhpc` so the
relative Slurm log paths still land under `servers/standalone-amdhpc/logs/`.

```bash
sbatch --chdir=servers/standalone-amdhpc \
  servers/standalone-amdhpc/start-single-service-mi2508x.sbatch
```

MI3008X uses the sibling script:

```bash
sbatch --chdir=servers/standalone-amdhpc \
  servers/standalone-amdhpc/start-single-service-mi3008x.sbatch
```

By default the script resolves `PORT_PROFILE_ID=0` from
`configs/port_profiles.toml`, which gives vLLM port `11451`.

To switch to another existing port profile during submit:

```bash
sbatch --chdir=servers/standalone-amdhpc \
  --export=ALL,PORT_PROFILE_ID=2 \
  servers/standalone-amdhpc/start-single-service-mi2508x.sbatch
sbatch --chdir=servers/standalone-amdhpc \
  --export=ALL,PORT_PROFILE_ID=2 \
  servers/standalone-amdhpc/start-single-service-mi3008x.sbatch
```

If you want to bypass the port-profile lookup entirely, override `VLLM_PORT`
directly:

```bash
sbatch --chdir=servers/standalone-amdhpc \
  --export=ALL,VLLM_PORT=31000 \
  servers/standalone-amdhpc/start-single-service-mi2508x.sbatch
sbatch --chdir=servers/standalone-amdhpc \
  --export=ALL,VLLM_PORT=31000 \
  servers/standalone-amdhpc/start-single-service-mi3008x.sbatch
```

To stop the service later:

```bash
scancel <job_id>
```

After the job is ready, the login-node endpoint is available on the same port.
For example, profile `0` uses `11451`, so from another shell you can check:

```bash
ssh login 'curl http://127.0.0.1:11451/v1/models'
```

## Common Overrides

Model selection:

- `VLLM_MODEL_KEY`
- `VLLM_MODEL_NAME`
- `VLLM_SERVED_MODEL_NAME`
- `VLLM_MODEL_EXTRA_ARGS_B64`

Runtime selection:

- `PORT_PROFILE_ID`
- `VLLM_PORT`
- `VLLM_TENSOR_PARALLEL_SIZE`
- `ROCR_VISIBLE_DEVICES`
- `LOGIN_HOST`
- `SSH_EXTRA_OPTIONS`
- `VLLM_SIF`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_TOKEN`
- `RUN_ID`

`VLLM_MODEL_KEY` is the normal way to pick a configured model. The other model
selection variables are lower-level overrides for raw model name, served model
name, or extra vLLM launch arguments when you need to bypass the default model
config entry.

The runtime variables control port selection, GPU/TP layout, reverse-tunnel
target and SSH options, container and Hugging Face paths, authentication, and
the run id used in logs and runtime artifacts.

Example using a different model and a different TP setting:

```bash
sbatch --chdir=servers/standalone-amdhpc \
  --export=ALL,VLLM_MODEL_KEY=qwen3_coder_30b_fp8,VLLM_TENSOR_PARALLEL_SIZE=4,ROCR_VISIBLE_DEVICES=0,1,2,3 \
  servers/standalone-amdhpc/start-single-service-mi2508x.sbatch
sbatch --chdir=servers/standalone-amdhpc \
  --export=ALL,VLLM_MODEL_KEY=qwen3_coder_30b_fp8,VLLM_TENSOR_PARALLEL_SIZE=4,ROCR_VISIBLE_DEVICES=0,1,2,3 \
  servers/standalone-amdhpc/start-single-service-mi3008x.sbatch
```

`VLLM_MODEL_KEY` is resolved from `configs/model_config.toml`. If you need a
raw Hugging Face model path that is not in that config, set both
`VLLM_MODEL_NAME` and `VLLM_SERVED_MODEL_NAME`.

## Ready Output

When the service is ready the job prints:

- `LOGIN_HOST=<login host>`
- `PORT_PROFILE_ID=<profile id>`
- `VLLM_TENSOR_PARALLEL_SIZE=<tp>`
- `ROCR_VISIBLE_DEVICES=<gpu list>`
- `VLLM_BASE_URL=http://127.0.0.1:<port>`

## Logs

The job writes:

- Slurm stdout: `logs/slurm.<job_id>.out`
- Slurm stderr: `logs/slurm.<job_id>.err`
- vLLM log: `logs/vllm.<run_id>.log`

## Direct Node Run

If you already have an allocation on the matching node type, you can run the
same script directly:

```bash
bash servers/standalone-amdhpc/start-single-service-mi2508x.sbatch
bash servers/standalone-amdhpc/start-single-service-mi3008x.sbatch
```

## Notes

- The default wall-clock limit in the static script is `12:00:00`.
- `start-single-service-mi2508x.sbatch` and
  `start-single-service-mi3008x.sbatch` are intentionally small hand-authored
  submission scripts.
