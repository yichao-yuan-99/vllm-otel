# Remote SSH Target Override (`-r` / `--remote-host`)

This note documents how to override the default SSH target used by
`servers/servers-amdhpc/client.py`.

## What Changed

All client commands now accept:

- `-r <host>`
- `--remote-host <host>`

These are aliases for the existing `--ssh-target` option.

Default behavior is unchanged:

- if you do not provide `-r`/`--remote-host`/`--ssh-target`, the target is `amd-hpc`

## Why Use It

Use this when your control server runs on a different SSH host than the default,
for example `login`, and you launch the client from a compute node.

The SSH target controls where `client-d` opens the local tunnel endpoints.

## Examples

Start single profile, targeting `login`:

```bash
python3 servers/servers-amdhpc/client.py start \
  -r login \
  -P 0 \
  -p mi3001x \
  -m qwen3_coder_30b
```

Equivalent using long option:

```bash
python3 servers/servers-amdhpc/client.py start \
  --remote-host login \
  -P 0 \
  -p mi3001x \
  -m qwen3_coder_30b
```

Grouped start with explicit remote host:

```bash
python3 servers/servers-amdhpc/client.py start-group \
  -r login \
  -g bench_a \
  -L 0,1,2,3 \
  -p mi3008x \
  -m qwen3_coder_30b
```

Status and stop using the same alias:

```bash
python3 servers/servers-amdhpc/client.py status -r login -P 0
python3 servers/servers-amdhpc/client.py stop -r login -P 0
```

You can still use `--ssh-target` exactly as before.
