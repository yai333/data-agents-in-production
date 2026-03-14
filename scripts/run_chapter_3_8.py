#!/usr/bin/env python3
"""Chapter 3.8 — run the full SQL agent CLI."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.cli import main


if __name__ == "__main__":
    main()
