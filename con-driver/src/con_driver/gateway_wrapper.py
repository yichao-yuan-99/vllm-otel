"""Gateway-aware wrapper for trial subprocess execution."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

import requests


def _post_json(*, endpoint: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed POST {endpoint}: {exc}") from exc
    if response.status_code >= 300:
        raise RuntimeError(
            f"Non-success response from {endpoint}: "
            f"{response.status_code} {response.text}"
        )
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid JSON response from {endpoint}: expected object")
    return parsed


def run_with_gateway(
    *,
    gateway_url: str,
    api_token: str,
    timeout_s: float,
    command: list[str],
) -> int:
    base = gateway_url.rstrip("/")
    start_endpoint = f"{base}/agent/start"
    end_endpoint = f"{base}/agent/end"

    started = False
    return_code = 1

    try:
        _post_json(
            endpoint=start_endpoint,
            payload={"api_token": api_token},
            timeout_s=timeout_s,
        )
        started = True
        wrapped_env = os.environ.copy()
        # Terminus 2 + LiteLLM resolve hosted_vllm auth from OPENAI_API_KEY.
        # Set per-agent token here so gateway /agent/start token and /v1 auth match.
        wrapped_env["OPENAI_API_KEY"] = api_token
        wrapped_env["LITELLM_API_KEY"] = api_token
        # Hosted vLLM adapter uses HOSTED_VLLM_API_KEY fallback when no explicit api_key
        # is passed through the Harbor Terminus 2 stack.
        wrapped_env["HOSTED_VLLM_API_KEY"] = api_token
        completed = subprocess.run(command, check=False, env=wrapped_env)
        return_code = int(completed.returncode)
    except Exception as exc:
        print(f"gateway wrapper error: {exc}", file=sys.stderr, flush=True)
    finally:
        if started:
            try:
                _post_json(
                    endpoint=end_endpoint,
                    payload={"api_token": api_token, "return_code": return_code},
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                print(
                    f"gateway wrapper error during /agent/end: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if return_code == 0:
                    return_code = 1

    return return_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m con_driver.gateway_wrapper",
        description="Wrap one trial subprocess with gateway /agent/start and /agent/end.",
    )
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--api-token", required=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _normalize_command(raw_command: list[str]) -> list[str]:
    command = list(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Missing wrapped command. Use '-- <command> <args...>'")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    if parsed.timeout_s <= 0:
        print("--timeout-s must be > 0", file=sys.stderr)
        return 2
    try:
        command = _normalize_command(parsed.command)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return run_with_gateway(
        gateway_url=parsed.gateway_url,
        api_token=parsed.api_token,
        timeout_s=float(parsed.timeout_s),
        command=command,
    )


if __name__ == "__main__":
    raise SystemExit(main())
