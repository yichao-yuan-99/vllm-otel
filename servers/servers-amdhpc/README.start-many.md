# start-many (Vectorized start)

`servers/servers-amdhpc/client.py start-many` is a vectorized version of `start`.
It runs the same single-profile start flow across multiple port profiles, one profile at a time.

Use this when you want one independent Slurm job per profile (effectively one node per profile for single-node partitions), instead of `start-group` single-node GPU splitting.

## Basic usage

```bash
python3 servers/servers-amdhpc/client.py start-many \
  -r amd-hpc \
  -L 0,1,2,3 \
  -p mi3008x \
  -m qwen3_coder_30b
```

## Useful options

- `--block` / `-b`: per-profile blocking submit (same semantics as `start --block`).
- `--env KEY=VALUE` (repeatable): extra environment variables for vLLM.
- `--lmcache <size>`: enables LMCache settings per profile start.
- `--extra-vllm-args "<args>"`: appends extra vLLM flags, for example:
  - `--extra-vllm-args "--enable-expert-parallel --max-model-len 32768"`
- `--continue-on-error` (default): continue attempting later profiles after a failure.
- `--fail-fast`: stop at first failed profile.

## Output

The command prints a single JSON summary with:

- `profile_list`
- `attempted_profile_ids`
- `started_profile_ids`
- `failed_profile_ids`
- `pending_profile_ids` (when `--fail-fast` stops early)
- per-profile `results`

Exit code is `0` only when all attempted profiles succeed.

## Difference vs start-group

- `start-many`: N independent starts/jobs, one per profile.
- `start-group`: one grouped job that colocates all listed profiles on a single node and splits GPUs across profiles.
