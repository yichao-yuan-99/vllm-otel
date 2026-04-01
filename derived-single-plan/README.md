This directory contains a small helper that takes a single-trail replay plan and
scales it into a synthetic multi-trail replay plan.

The derived plan keeps the same per-request structure and timing as the source
trail, but replaces every deterministic `forced_token_ids` sequence with new
contiguous token windows sampled from a user-provided text corpus. It also
retokens the initial prompt messages from the first request and rewrites that
carried-forward prefix in later request histories with equal-length replacement
text from the same corpus.

## Sampling Rule

The replacement token corpus comes from a required text file:

1. read the full contents of `--replacement-text-file`
2. tokenize that text with the replay model tokenizer
3. for each request, choose a random start offset into that token array
4. take a contiguous token window of the same length as the source request

The generator is seeded, so the same `--seed` plus the same input plan will
produce the same derived output.

## Replay Compatibility

Changing only `forced_token_ids` would make replay fail strict response-text
validation, so the helper also refreshes each request's
`expected_response_text` by calling vLLM `/detokenize`.

The helper validates each sampled token window through vLLM `/detokenize` and
`/tokenize` before writing the final `forced_token_ids` and
`expected_response_text` into the plan. The same replay-stable sampling rule is
used for carried-forward initial prompt messages, with replacement text sized to
the original message token count. If a sampled window is not replay-safe, it
resamples a new offset.

Because of that, this helper currently supports replay plans that contain only
deterministic requests with `forced_token_ids`.

## CLI

```bash
python3 derived-single-plan/generate_derived_single_plan.py \
  --plan /path/to/replay-plan.trail-foo.json \
  --replacement-text-file /path/to/corpus.txt \
  --seed 7 \
  --size 500 \
  --port-profile 0
```

Batch mode scans a directory recursively for non-derived single-trail
`replay-plan*.json` files and derives all detected source plans:

```bash
python3 derived-single-plan/generate_derived_single_plan.py \
  --plan /path/to/results/qwen3-coder-30b/swebench-verified/mini-swe-agent \
  --replacement-text-file /path/to/corpus.txt \
  --seed 7 \
  --size 500 \
  --port-profile 0
```

Arguments:

- `--plan`: source replay plan with exactly one worker, or a directory to scan
  recursively in batch mode for plans that include
  `compile_options.single_trail`
- `--replacement-text-file`: required UTF-8 text file used as the replacement
  token corpus
- `--seed`: sampling seed
- `--size`: number of workers in the derived plan, default `500`
- `--port-profile`: port profile numeric ID from `configs/port_profiles.toml`;
  the helper resolves both `http://127.0.0.1:<vllm_port>/detokenize` and
  `http://127.0.0.1:<vllm_port>/tokenize` from it
- `--request-timeout-s`: per-request timeout for `/detokenize` and `/tokenize`,
  default `60`
- `--output`: optional explicit output path in single-plan mode only

## Output

By default the output is written next to the input plan with a prefixed name:

```text
seed-7-size-500.replay-plan.trail-foo.json
```

The generated plan also includes:

- top-level `is_derived: true`
- top-level `derived_single_plan` metadata with the seed, size,
  replacement-text path, replacement token counts, and source-plan path

In batch mode, the helper writes one derived plan next to each detected source
plan and prints a JSON summary with candidate, success, and failure counts.
When run in an interactive terminal, it also shows a progress bar on stderr
while requests are being derived, including the current plan, worker index, and
request index. The final JSON summary for each derived plan includes a
verification block that checks request-count preservation, prompt/generation
length preservation, and cross-worker history uniqueness.
