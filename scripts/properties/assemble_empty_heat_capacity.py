#!/usr/bin/env python3
"""Assemble an empty-MOF harmonic heat-capacity curve from a merged Hessian."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mof_heat_capacity.campaign import GUEST_SYMBOLS, structure_directory, system_directory, validate_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof", default="mof5")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--model-label", required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/post-processing/harmonic-correction"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_selection(args.mof, args.guest)
    archive = args.input_root / system_directory(args.model_label, 0, args.mof, args.guest) / "hessians/hessian.npz"
    output = args.output or (
        args.input_root
        / system_directory(args.model_label, 0, args.mof, args.guest)
        / "heat-capacity.exploratory-discard-imaginary.csv"
    )
    if not archive.is_file():
        raise FileNotFoundError(f"empty-MOF Hessian archive not found: {archive}")
    with np.load(archive, allow_pickle=False) as data:
        required = {"temperatures_K", "cv_J_per_gK", "llpr_cv_J_per_gK"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                "empty-MOF Hessian archive lacks LLPR heat capacities: "
                + ", ".join(missing)
            )
        temperatures = np.asarray(data["temperatures_K"], dtype=float)
        central = np.asarray(data["cv_J_per_gK"], dtype=float)
        members = np.asarray(data["llpr_cv_J_per_gK"], dtype=float)
    if central.shape != (1, len(temperatures)):
        raise ValueError("expected one central empty-MOF Hessian curve")
    if members.ndim != 3 or members.shape[0] != 1 or members.shape[2] != len(temperatures):
        raise ValueError("invalid member-resolved empty-MOF heat capacities")
    if members.shape[1] < 2:
        raise ValueError("at least two LLPR member curves are required")
    values = central[0]
    uncertainty = members[0].std(axis=0, ddof=1)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(uncertainty)):
        raise ValueError("empty-MOF heat-capacity curve is non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "temperature_K",
                "approximate_cp_J_per_gK",
                "approximate_cp_combined_standard_uncertainty_J_per_gK",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "temperature_K": temperature,
                "approximate_cp_J_per_gK": value,
                "approximate_cp_combined_standard_uncertainty_J_per_gK": error,
            }
            for temperature, value, error in zip(temperatures, values, uncertainty)
        )
    print(f"Saved empty-MOF heat capacity: {output}")


if __name__ == "__main__":
    main()
