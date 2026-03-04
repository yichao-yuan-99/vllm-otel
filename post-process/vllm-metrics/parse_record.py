from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import parse_metric_content_to_json


def parse_metric_record_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object")

    content = payload.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("Input JSON must contain a non-empty 'content' string")

    return parse_metric_content_to_json(content)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a raw vLLM metrics record JSON into structured JSON."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a raw metrics record JSON file with a top-level 'content' field.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. If omitted, prints parsed JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    parsed = parse_metric_record_payload(payload)
    rendered = json.dumps(parsed, ensure_ascii=True, indent=2)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
