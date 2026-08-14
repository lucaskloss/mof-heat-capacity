#!/usr/bin/env python3
"""Compatibility entry point for the molecular-dynamics workflow."""

from mof_heat_capacity.simulation.md import *  # noqa: F403
from mof_heat_capacity.simulation.md import main


if __name__ == "__main__":
    main()
