#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve eos_token_id from a Hugging Face tokenizer. "
            "Defaults to the Qwen model used by docker tests."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="Hugging Face model id (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        choices=("plain", "shell", "json"),
        default="plain",
        help=(
            "Output format: plain prints only the id, shell prints "
            "VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID=<id>, json prints structured output."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        print(
            "transformers is required. Install with: pip install transformers",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        print(
            f"Tokenizer for model {args.model!r} has no eos_token_id.",
            file=sys.stderr,
        )
        return 1

    eos_token = tokenizer.eos_token

    if args.format == "plain":
        print(eos_token_id)
    elif args.format == "shell":
        print(f"VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID={eos_token_id}")
    else:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "eos_token_id": eos_token_id,
                    "eos_token": eos_token,
                },
                ensure_ascii=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
