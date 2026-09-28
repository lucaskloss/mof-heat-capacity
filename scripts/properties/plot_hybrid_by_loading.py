#!/usr/bin/env python3
"""Plot one hybrid heat-capacity curve per methane loading from assembled CSVs."""

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
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--loadings", default="50,100")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/post-processing/harmonic-correction"),
    )
    parser.add_argument(
        "--csv-name",
        default="heat-capacity.exploratory-discard-imaginary.csv",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("output/reference_heat_capacity"),
        help="Directory containing optional <loading>ch4.csv reference curves",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"hybrid CSV not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"hybrid CSV has no data rows: {path}")
    return tuple(
        np.asarray([float(row[column]) for row in rows])
        for column in (
            "temperature_K",
            "approximate_cp_J_per_gK",
            "approximate_cp_combined_standard_uncertainty_J_per_gK",
        )
    )


def load_reference_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"reference CSV not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"reference CSV has no data rows: {path}")
    return (
        np.asarray([float(row["temperature_K"]) for row in rows]),
        np.asarray([float(row["reference_heat_capacity_J_per_gK"]) for row in rows]),
    )


def main() -> None:
    args = parse_args()
    loadings = [int(value.strip()) for value in args.loadings.split(",") if value.strip()]
    if not loadings or len(set(loadings)) != len(loadings) or any(value < 0 for value in loadings):
        raise ValueError("--loadings must contain unique non-negative integers")

    figure, axis = plt.subplots(figsize=(10, 6))
    for loading in loadings:
        path = args.input_root / args.model_label / f"{loading}ch4" / args.csv_name
        temperatures, heat_capacity, uncertainty = load_curve(path)
        line = axis.plot(
            temperatures,
            heat_capacity,
            marker="o",
            linewidth=2.5,
            label=f"{loading} CH4 — hybrid",
        )[0]
        axis.fill_between(
            temperatures,
            heat_capacity - uncertainty,
            heat_capacity + uncertainty,
            color=line.get_color(),
            alpha=0.18,
        )
        reference_path = args.reference_root / f"{loading}ch4.csv"
        if reference_path.is_file():
            reference_temperatures, reference_heat_capacity = load_reference_curve(
                reference_path
            )
            axis.plot(
                reference_temperatures,
                reference_heat_capacity,
                color=line.get_color(),
                linestyle="--",
                linewidth=2.2,
                label=f"{loading} CH4 — reference",
            )

    model_title = MODEL_TITLES.get(args.model_label, args.model_label)
    axis.set(
        title=f"MOF-5 + methane: {model_title} hybrid heat capacity",
        xlabel="Temperature (K)",
        ylabel=r"Hybrid heat capacity (J g$^{-1}$ K$^{-1}$)",
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Methane loading / curve", ncol=2)
    figure.text(
        0.5,
        0.01,
        "Exploratory: unstable/near-zero modes discarded; loaded uncertainty includes sampling and LLPR spread.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved loading comparison: {args.output}")


if __name__ == "__main__":
    main()
