#!/usr/bin/env python3
"""Prepare the loaded classical-MD campaign."""

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from mof_heat_capacity.protocols.loaded import *  # noqa: F403,E402
from mof_heat_capacity.protocols.loaded import main  # noqa: E402


if __name__ == "__main__":
    main()
