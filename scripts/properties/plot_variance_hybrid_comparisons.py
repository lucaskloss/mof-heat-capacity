#!/usr/bin/env python3
"""Plot hybrid loading curves and uncertainty components using NPT variances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_TITLES = {
    "pet-mad-1.5-s-40nn": "PET-MAD",
    "pet-sol-s-best": "PET-SOL",
}
LOADINGS = (50, 100, 150)
COMPONENTS = (
    ("MD NPT variance sampling", "gaussian_npt_central_sampling_standard_error_J_per_gK"),
    ("Classical Gaussian LLPR", "classical_variance_model_sd_J_per_gK"),
    ("Harmonic LLPR Hessian", "harmonic_model_sd_J_per_gK"),
    ("Final correlated combination", "combined_standard_uncertainty_J_per_gK"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="pet-mad-1.5-s-40nn,pet-sol-s-best")
    parser.add_argument("--loadings", default="50,100,150")
    parser.add_argument(
        "--analysis-root", type=Path, default=PROJECT_DIR / "output/post-processing/trajectory-analysis"
    )
    parser.add_argument(
        "--hybrid-root", type=Path, default=PROJECT_DIR / "output/post-processing/harmonic-correction"
    )
    parser.add_argument(
        "--reference-root", type=Path, default=PROJECT_DIR / "output/reference_heat_capacity"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_DIR / "output/post-processing/harmonic-correction/loading-comparisons",
    )
    return parser.parse_args()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"required estimator archive not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _assemble(model: str, loading: int, args: argparse.Namespace) -> dict[str, np.ndarray]:
    variance_path = (
        args.analysis_root / model / f"{loading}ch4" / "variance_heat_capacity_comparison.npz"
    )
    hybrid_path = (
        args.hybrid_root / model / f"{loading}ch4"
        / "heat-capacity.exploratory-discard-imaginary.npz"
    )
    variance = _load_arrays(variance_path)
    hybrid = _load_arrays(hybrid_path)

    if str(variance["model_sha256"].item()) != str(hybrid["llpr_checkpoint_sha256"].item()):
        raise ValueError(f"variance and Hessian archives use different LLPR checkpoints for {model}/{loading}")
    temperature = np.asarray(hybrid["temperatures_K"], dtype=float)
    full_grid = np.asarray(variance["temperatures_K"], dtype=float)
    grid_indices = []
    for value in temperature:
        matches = np.flatnonzero(np.isclose(full_grid, value, rtol=0.0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"hybrid temperature {value:g} K is missing from variance archive {variance_path}")
        grid_indices.append(int(matches[0]))

    classical_members = np.asarray(
        variance["gaussian_npt_cp_by_member_J_per_gK"], dtype=float
    )[:, grid_indices]
    harmonic_members = np.asarray(
        hybrid["harmonic_quantum_correction_by_member_J_per_gK"], dtype=float
    )
    if classical_members.shape != harmonic_members.shape:
        raise ValueError(f"classical and harmonic committee shapes differ for {model}/{loading}")

    correlated_member_deviations = (
        classical_members - classical_members.mean(axis=0)
        + harmonic_members - harmonic_members.mean(axis=0)
    )
    correlated_model_sd = correlated_member_deviations.std(axis=0, ddof=1)
    md_sampling_se = np.asarray(
        variance["gaussian_npt_central_sampling_standard_error_J_per_gK"], dtype=float
    )[grid_indices]
    combined = np.sqrt(md_sampling_se**2 + correlated_model_sd**2)
    central_hybrid = (
        np.asarray(variance["gaussian_npt_central_cp_J_per_gK"], dtype=float)[grid_indices]
        + np.asarray(hybrid["harmonic_quantum_correction_J_per_gK"], dtype=float)
    )
    result = {
        "temperature_K": temperature,
        "gaussian_npt_central_cp_J_per_gK": np.asarray(
            variance["gaussian_npt_central_cp_J_per_gK"], dtype=float
        )[grid_indices],
        "classical_cp_J_per_gK": np.asarray(
            hybrid["classical_anharmonic_cp_J_per_gK"], dtype=float
        ),
        "central_hybrid_cp_J_per_gK": central_hybrid,
        "gaussian_npt_central_sampling_standard_error_J_per_gK": md_sampling_se,
        "classical_variance_model_sd_J_per_gK": classical_members.std(axis=0, ddof=1),
        "harmonic_model_sd_J_per_gK": np.asarray(
            hybrid["harmonic_quantum_correction_model_standard_deviation_J_per_gK"], dtype=float
        ),
        "correlated_hybrid_model_sd_J_per_gK": correlated_model_sd,
        "combined_standard_uncertainty_J_per_gK": combined,
    }
    return result


def _write_csv(path: Path, result: dict[str, np.ndarray]) -> None:
    fields = ["temperature_K", *[name for _, name in COMPONENTS],
              "gaussian_npt_central_cp_J_per_gK", "classical_cp_J_per_gK",
              "central_hybrid_cp_J_per_gK", "correlated_hybrid_model_sd_J_per_gK",
              ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, temperature in enumerate(result["temperature_K"]):
            writer.writerow({name: result[name][index] for name in fields})


def _read_reference(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["temperature_K"]) for row in rows]),
        np.asarray([float(row["reference_heat_capacity_J_per_gK"]) for row in rows]),
    )


def _plot_hybrid_curves(
    models: list[str], loadings: list[int], results: dict[tuple[str, int], dict[str, np.ndarray]], args: argparse.Namespace
) -> None:
    for model in models:
        figure, axis = plt.subplots(figsize=(10, 6))
        for loading in loadings:
            result = results[(model, loading)]
            line = axis.plot(
                result["temperature_K"], result["central_hybrid_cp_J_per_gK"],
                marker="o", linewidth=2.5, label=f"{loading} CH4 — variance hybrid",
            )[0]
            uncertainty = result["combined_standard_uncertainty_J_per_gK"]
            axis.fill_between(
                result["temperature_K"],
                result["central_hybrid_cp_J_per_gK"] - uncertainty,
                result["central_hybrid_cp_J_per_gK"] + uncertainty,
                color=line.get_color(), alpha=0.18,
            )
            reference = _read_reference(args.reference_root / f"{loading}ch4.csv")
            if reference is not None:
                axis.plot(*reference, color=line.get_color(), linestyle="--", linewidth=2,
                          label=f"{loading} CH4 — reference")
        axis.set(
            title=f"MOF-5 + methane: {MODEL_TITLES.get(model, model)} Gaussian-variance hybrid heat capacity",
            xlabel="Temperature (K)", ylabel=r"Hybrid heat capacity (J g$^{-1}$ K$^{-1}$)",
        )
        axis.grid(alpha=0.25)
        axis.legend(title="Methane loading / curve", ncol=2)
        figure.text(
            0.5, 0.01,
            "Exploratory Hessian policy; Gaussian NPT fluctuation closure; shading combines MD sampling error and LLPR spread.",
            ha="center", fontsize=9,
        )
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        path = args.output_dir / f"{model}-50-vs-100-vs-150ch4-variance-hybrid-heat-capacity.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        print(f"Saved variance hybrid loading comparison: {path}")


def _plot_components(
    models: list[str], loadings: list[int], results: dict[tuple[str, int], dict[str, np.ndarray]], output: Path
) -> None:
    figure, axes = plt.subplots(
        len(models), len(loadings), figsize=(4.7 * len(loadings), 6.5),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    for row, model in enumerate(models):
        for column, loading in enumerate(loadings):
            axis = axes[row, column]
            result = results[(model, loading)]
            for label, field in COMPONENTS:
                axis.plot(result["temperature_K"], result[field], marker="o", linewidth=2, label=label)
            axis.set_title(f"{MODEL_TITLES.get(model, model)}, {loading} CH4")
            axis.grid(alpha=0.25)
            if row == len(models) - 1:
                axis.set_xlabel("Temperature (K)")
            if column == 0:
                axis.set_ylabel(r"Standard uncertainty (J g$^{-1}$ K$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, frameon=False,
    )
    figure.subplots_adjust(top=0.84, bottom=0.10, left=0.075, right=0.99,
                           hspace=0.20, wspace=0.04)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved variance uncertainty-component comparison: {output}")


def main() -> None:
    args = parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    loadings = [int(value.strip()) for value in args.loadings.split(",") if value.strip()]
    if not models or not loadings or any(value <= 0 for value in loadings):
        raise ValueError("--models and --loadings must be non-empty; loadings must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in models:
        for loading in loadings:
            result = _assemble(model, loading, args)
            results[(model, loading)] = result
            _write_csv(args.output_dir / f"{model}-{loading}ch4-variance-hybrid-heat-capacity.csv", result)
    _plot_hybrid_curves(models, loadings, results, args)
    _plot_components(
        models, loadings, results,
        args.output_dir / "hybrid-heat-capacity-variance-uncertainty-components.png",
    )


if __name__ == "__main__":
    main()
