"""Parsing helpers for CLI string arguments."""

from __future__ import annotations

import shlex
from typing import Sequence


_BRACKET_CLOSE = {"{": "}", "[": "]", "(": ")"}


def split_top_level_commas(raw: str) -> list[str]:
    """Split a string on top-level commas.

    Commas inside quotes or bracket pairs are ignored.
    """
    text = raw.strip()
    if not text:
        return []

    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    closers: list[str] = []

    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue

        if char == "\\":
            current.append(char)
            escaped = True
            continue

        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue

        if char in ("'", '"'):
            current.append(char)
            quote = char
            continue

        if char in _BRACKET_CLOSE:
            current.append(char)
            closers.append(_BRACKET_CLOSE[char])
            continue

        if closers and char == closers[-1]:
            current.append(char)
            closers.pop()
            continue

        if char == "," and not closers:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
            continue

        current.append(char)

    if quote is not None:
        raise ValueError(f"Unclosed quote in value: {raw}")
    if closers:
        raise ValueError(f"Unclosed bracket in value: {raw}")

    piece = "".join(current).strip()
    if piece:
        parts.append(piece)
    return parts


def parse_comma_arg_string(raw: str) -> list[str]:
    """Parse a comma-separated argument list into shell tokens.

    Example:
      --foo=1,--bar='a b' -> ["--foo=1", "--bar=a b"]
    """
    tokens: list[str] = []
    for part in split_top_level_commas(raw):
        tokens.extend(shlex.split(part))
    return tokens


def parse_pool_specs(raw: str) -> list[str]:
    """Parse dataset pool list like 'a@1,b@2'."""
    specs = split_top_level_commas(raw)
    if not specs:
        raise ValueError("--pool cannot be empty")
    return specs


def parse_cli_kv(tokens: Sequence[str]) -> dict[str, str]:
    """Parse CLI-style key/value options into a dict.

    Supports:
      --k=v
      --k v
      --flag    (treated as "true")
    """
    result: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:]
        elif token.startswith("-"):
            key = token[1:]
        else:
            raise ValueError(f"Expected option token, got: {token}")

        if not key:
            raise ValueError(f"Invalid option token: {token}")

        if "=" in key:
            key_name, value = key.split("=", 1)
            if not key_name:
                raise ValueError(f"Invalid option token: {token}")
            result[key_name] = value
            i += 1
            continue

        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            value = tokens[i + 1]
            i += 2
        else:
            value = "true"
            i += 1
        result[key] = value

    return result


def parse_command_prefix(raw: str) -> list[str]:
    """Parse a shell-like command prefix, for example 'uv run harbor'."""
    parts = shlex.split(raw)
    if not parts:
        raise ValueError("Command prefix cannot be empty")
    return parts


def assert_safe_harbor_args(harbor_args: Sequence[str]) -> None:
    """Reject args managed internally by this driver."""
    blocked = {"-p", "--path", "--trial-name", "--trials-dir"}

    for token in harbor_args:
        flag = token.split("=", 1)[0]
        if flag in blocked:
            raise ValueError(
                "Do not include "
                f"'{flag}' in --harbor-args. The driver sets task path/trial id/trials-dir."
            )
