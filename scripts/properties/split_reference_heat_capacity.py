#!/usr/bin/env python3
"""Split a consecutive-loading reference heat-capacity CSV into per-loading files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mof_heat_capacity.campaign import GUEST_SYMBOLS, structure_directory, system_directory, validate_selection


LOADINGS = (0, 50, 100, 150)
ROWS_PER_LOADING = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof", default="mof5")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_selection(args.mof, args.guest)
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    expected_rows = len(LOADINGS) * ROWS_PER_LOADING
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} reference rows for {len(LOADINGS)} loadings; "
            f"found {len(rows)}"
        )
    if set(rows[0]) != {"x", "y"}:
        raise ValueError("reference CSV must have x and y columns")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, loading in enumerate(LOADINGS):
        block = rows[index * ROWS_PER_LOADING : (index + 1) * ROWS_PER_LOADING]
        output = args.output_dir / structure_directory(loading, args.mof, args.guest).with_suffix(".csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("temperature_K", "reference_heat_capacity_J_per_gK"),
            )
            writer.writeheader()
            writer.writerows(
                {
                    "temperature_K": int(round(float(row["x"]))),
                    "reference_heat_capacity_J_per_gK": float(row["y"]),
                }
                for row in block
            )
        print(f"Saved loading {loading}: {output}")


if __name__ == "__main__":
    main()
