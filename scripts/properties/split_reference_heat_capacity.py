#!/usr/bin/env python3
"""Split a consecutive-loading reference heat-capacity CSV into per-loading files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LOADINGS = (0, 50, 100, 150)
ROWS_PER_LOADING = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        output = args.output_dir / f"{loading}ch4.csv"
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
