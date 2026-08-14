#!/usr/bin/env python3
"""Compatibility entry point for complete trajectory analysis."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from mof_heat_capacity.analysis.results import *  # noqa: F403,E402
from mof_heat_capacity.analysis.results import main  # noqa: E402


if __name__ == "__main__":
    main()
