"""Script entrypoint for the concurrent Harbor driver."""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from con_driver.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
