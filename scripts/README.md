# scripts

Utility scripts for local setup and validation.

## get_model_eos_id.py

Resolve `eos_token_id` from a Hugging Face tokenizer.
Defaults to the Qwen model used by docker force-sequence tests.

### Usage

```bash
python scripts/get_model_eos_id.py
python scripts/get_model_eos_id.py --format shell
python scripts/get_model_eos_id.py --format json
python scripts/get_model_eos_id.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --format shell
```

### Optional: inspect EOS token id

```bash
python scripts/get_model_eos_id.py --format shell
```

Docker vLLM startup now infers EOS automatically via `docker/vllm_entrypoint.sh`,
so this script is mainly for debugging/verification.

### Requirements

`transformers` must be installed in your local Python environment:

```bash
pip install transformers
```
