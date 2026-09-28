#!/usr/bin/env python3
"""Plot loaded hybrid heat-capacity uncertainty components by model and loading."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("pet-mad-1.5-s-40nn", "pet-sol-s-best")
MODEL_TITLES = {MODELS[0]: "PET-MAD", MODELS[1]: "PET-SOL"}
COMPONENTS = (
    ("MD finite-difference sampling", "approximate_cp_standard_error_J_per_gK"),
    ("Classical CEA/LLPR", "classical_cp_model_standard_deviation_J_per_gK"),
    ("Harmonic LLPR Hessian", "harmonic_correction_model_standard_deviation_J_per_gK"),
    ("Final correlated combination", "approximate_cp_combined_standard_uncertainty_J_per_gK"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loadings", default="50,100,150")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/post-processing/harmonic-correction"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"hybrid CSV not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"hybrid CSV has no data rows: {path}")
    return (
        np.asarray([float(row["temperature_K"]) for row in rows]),
        {
            column: np.asarray([float(row[column]) for row in rows])
            for _, column in COMPONENTS
        },
    )


def main() -> None:
    args = parse_args()
    loadings = [int(value) for value in args.loadings.split(",") if value]
    if not loadings or any(value < 1 for value in loadings):
        raise ValueError("--loadings must contain positive integers")
    figure, axes = plt.subplots(
        len(MODELS), len(loadings), figsize=(4.7 * len(loadings), 6.5),
        sharex=True, sharey=True, layout="constrained",
    )
    axes = np.atleast_2d(axes)
    for row, model in enumerate(MODELS):
        for column, loading in enumerate(loadings):
            axis = axes[row, column]
            path = args.input_root / model / f"{loading}ch4" / "heat-capacity.exploratory-discard-imaginary.csv"
            temperatures, values = read_csv(path)
            for label, field in COMPONENTS:
                axis.plot(temperatures, values[field], marker="o", linewidth=2, label=label)
            axis.set_title(f"{MODEL_TITLES[model]}, {loading} CH4")
            axis.grid(alpha=0.25)
            if row == len(MODELS) - 1:
                axis.set_xlabel("Temperature (K)")
            if column == 0:
                axis.set_ylabel(r"Standard uncertainty (J g$^{-1}$ K$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved uncertainty-component comparison: {args.output}")


if __name__ == "__main__":
    main()
