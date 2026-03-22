# start-many (Vectorized start)

`servers/servers-amdhpc/client.py start-many` is a vectorized version of `start`.
It is now server-backed and submits one multi-node sbatch job where each selected profile is assigned to one node:

1. Client starts local `client-d` tunnels for all selected profiles.
2. Control server submits one Slurm job with `N` nodes and `N` tasks (`N = len(profile_list)`).
3. Each task binds to one profile and launches one full-node vLLM+Jaeger stack.
4. Client launches local gateway daemons per profile after readiness.

Use this when you want one full node per profile inside a single grouped Slurm allocation, instead of `start-group` single-node GPU splitting.

## Basic usage

```bash
python3 servers/servers-amdhpc/client.py start-many \
  -r amd-hpc \
  -g bench_many_a \
  -L 0,1,2,3 \
  -p mi3008x \
  -m qwen3_coder_30b
```

## Useful options

- `--group-name` / `-g`: optional; recommended so `stop-group` can target this run directly.
- `--block` / `-b`: accepted for compatibility; server-backed `start-many` currently always runs in blocking mode.
- `--env KEY=VALUE` (repeatable): extra environment variables for vLLM.
- `--lmcache <size>`: enables LMCache settings per profile start.
- `--extra-vllm-args "<args>"`: appends extra vLLM flags, for example:
  - `--extra-vllm-args "--enable-expert-parallel --max-model-len 32768"`
- `--continue-on-error` / `--fail-fast`: compatibility flags retained for CLI stability (ignored by server-backed start-many).

Notes:

- If `--group-name` is omitted, client auto-generates one and prints it in output.
- Stop with `python3 servers/servers-amdhpc/client.py stop-group -r amd-hpc -g <group_name>`.

## Output

The command prints a single JSON summary with:

- `group_name`
- `profile_list`
- `server` payload from `many/start`
- per-profile `clientd`, `wait_up` (reserved field), and `gateway` payloads

Exit code is `0` only when all selected profiles succeed.

## Difference vs start-group

- `start-many`: one grouped multi-node job; one full node per profile.
- `start-group`: one grouped single-node job; GPUs are split across profiles on the same node.
