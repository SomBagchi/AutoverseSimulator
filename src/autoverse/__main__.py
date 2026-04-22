"""Allow `python -m autoverse …` to invoke the CLI."""

from __future__ import annotations

import sys

from autoverse.cli import main

if __name__ == "__main__":
    sys.exit(main())
