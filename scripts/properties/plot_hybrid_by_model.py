#!/usr/bin/env python3
"""Compare central-only hybrid heat-capacity curves from multiple MLIPs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_TITLES = {
    "pet-mad-1.5-s-40nn": "PET-MAD",
    "pet-sol-s-best": "PET-SOL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loading", type=int, required=True)
    parser.add_argument(
        "--models", default="pet-mad-1.5-s-40nn,pet-sol-s-best"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/post-processing/harmonic-correction"),
    )
    parser.add_argument(
        "--csv-name",
        default="heat-capacity.exploratory-discard-imaginary.central-only.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"hybrid CSV has no data rows: {path}")
    return tuple(
        np.asarray([float(row[column]) for row in rows])
        for column in (
            "temperature_K",
            "approximate_cp_J_per_gK",
            "approximate_cp_standard_error_J_per_gK",
        )
    )


def main() -> None:
    args = parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not models or len(set(models)) != len(models):
        raise ValueError("--models must contain unique model labels")

    figure, axis = plt.subplots(figsize=(10, 6))
    for model in models:
        path = args.input_root / model / f"{args.loading}ch4" / args.csv_name
        if not path.is_file():
            raise FileNotFoundError(f"hybrid CSV not found: {path}")
        temperatures, heat_capacity, standard_error = read_curve(path)
        line = axis.plot(
            temperatures,
            heat_capacity,
            marker="o",
            linewidth=2.5,
            label=MODEL_TITLES.get(model, model),
        )[0]
        axis.fill_between(
            temperatures,
            heat_capacity - standard_error,
            heat_capacity + standard_error,
            color=line.get_color(),
            alpha=0.18,
        )

    axis.set(
        title=f"MOF-5 + {args.loading} methane: central-only hybrid heat capacity",
        xlabel="Temperature (K)",
        ylabel=r"Hybrid heat capacity (J g$^{-1}$ K$^{-1}$)",
    )
    axis.grid(alpha=0.25)
    axis.legend(title="MLIP")
    figure.text(
        0.5,
        0.01,
        "Exploratory: unstable/near-zero modes discarded; shading is sampling ±1 SE (LLPR uncertainty omitted).",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved model comparison: {args.output}")


if __name__ == "__main__":
    main()
