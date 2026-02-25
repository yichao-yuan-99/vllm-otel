# Force Sequence Logits Processor

This folder contains a custom vLLM V1 logits processor that forces generation
to follow a token-id sequence exactly, one decoding step at a time.

## What It Does

For a request with:

- `vllm_xargs.forced_token_ids = [42, 53, 99]`

the decoded output tokens are constrained as:

1. first generated token must be `42`
2. second generated token must be `53`
3. third generated token must be `99`

After the provided sequence is consumed:

- it can optionally force EOS using server env
  `VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID`.
- enable per request with `vllm_xargs.force_eos_after_sequence = true`.

## Files

- `forceSeq/force_sequence_logits_processor.py`
- `forceSeq/__init__.py`

## Server Startup

From the repo root, start `vllm serve` with this custom processor:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --logits-processors forceSeq.force_sequence_logits_processor:ForceSequenceAdapter
```

If you run `vllm serve` from outside this repo, make the repo importable:

```bash
export PYTHONPATH=/scratch/yichaoy2/read/vllm:$PYTHONPATH
```

## Server-Side Enablement Details

Use this flow to ensure the server actually enables the processor.

1. Make the module importable in the serving environment.
   - Option A: run `vllm serve` from repo root `/scratch/yichaoy2/read/vllm`.
   - Option B: set `PYTHONPATH=/scratch/yichaoy2/read/vllm:$PYTHONPATH`.
   - Quick check:
     ```bash
     python3 -c "from forceSeq.force_sequence_logits_processor import ForceSequenceAdapter; print(ForceSequenceAdapter.__name__)"
     ```
2. Add the processor at server startup via engine arg:
   - `--logits-processors forceSeq.force_sequence_logits_processor:ForceSequenceAdapter`
   - set env var `VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID=<model_eos_id>` before startup.
   - in this repo's docker flow, `docker/vllm_entrypoint.sh` auto-resolves this from `--model`.
3. Keep the server in a supported mode:
   - Do not enable speculative decoding with custom logits processors.
   - Do not use pooling models with custom logits processors.
4. Send requests with `vllm_xargs.forced_token_ids` to activate it per request.
   - If `forced_token_ids` is absent, this processor is a no-op for that request.

Example with additional server args:

```bash
PYTHONPATH=/scratch/yichaoy2/read/vllm:$PYTHONPATH \
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --tensor-parallel-size 2 \
  --enable-prefix-caching \
  --logits-processors forceSeq.force_sequence_logits_processor:ForceSequenceAdapter
```

## Request Usage (OpenAI-compatible API)

Pass forced token ids via `vllm_xargs`.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [{"role":"user","content":"ignore this text"}],
    "temperature": 0,
    "max_tokens": 16,
    "vllm_xargs": {
      "forced_token_ids": [42, 53, 99],
      "force_eos_after_sequence": true
    }
  }'
```

## Request Arguments

- `forced_token_ids` (required to enable this processor)
  - type: non-empty `list[int]`
  - must be non-negative token ids
- `force_eos_after_sequence` (optional)
  - type: `bool`
  - default: `true`

Server-side env requirement:

- `VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID`
  - required at processor load time
  - type: non-negative integer token id

If `forced_token_ids` is missing, this processor does nothing for that request.

## Notes and Caveats

- This forces token ids, not text strings.
- The step index is based on generated output length, not prompt length.
- `min_tokens` can delay EOS stopping if EOS is forced before minimum length is
  reached.
- Custom logits processors are not supported with speculative decoding.
- Custom logits processors are not supported for pooling models.

## Getting Token IDs for a Target String

Use the model tokenizer to convert text into ids:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
forced_ids = tok.encode("hello world", add_special_tokens=False)
print(forced_ids)
```

Resolve EOS token id (server env) with:

```bash
python scripts/get_model_eos_id.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --format shell
```
