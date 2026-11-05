#!/usr/bin/env python3
"""Export hybrid loading curves and uncertainty components using NPT variances."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mof_heat_capacity.campaign import GUEST_SYMBOLS, system_directory, validate_selection


PROJECT_DIR = Path(__file__).resolve().parents[2]
COMPONENTS = (
    ("MD NPT variance sampling", "gaussian_npt_central_sampling_standard_error_J_per_gK"),
    ("Classical Gaussian LLPR", "classical_variance_model_sd_J_per_gK"),
    ("Harmonic LLPR Hessian", "harmonic_model_sd_J_per_gK"),
    ("Final correlated combination", "combined_standard_uncertainty_J_per_gK"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof", default="mof5")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--models", default="pet-mad-1.5-s-40nn,pet-sol-s-best")
    parser.add_argument("--loadings", default="50,100,150")
    parser.add_argument(
        "--analysis-root", type=Path, default=PROJECT_DIR / "output/post-processing/trajectory-analysis"
    )
    parser.add_argument(
        "--hybrid-root", type=Path, default=PROJECT_DIR / "output/post-processing/harmonic-correction"
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
        args.analysis_root / system_directory(model, loading, args.mof, args.guest) / "variance_heat_capacity_comparison.npz"
    )
    hybrid_path = (
        args.hybrid_root / system_directory(model, loading, args.mof, args.guest)
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


def main() -> None:
    args = parse_args()
    validate_selection(args.mof, args.guest)
    if (args.mof, args.guest) != ("mof5", "ch4"):
        args.output_dir = args.output_dir / args.mof / args.guest
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    loadings = [int(value.strip()) for value in args.loadings.split(",") if value.strip()]
    if not models or not loadings or any(value <= 0 for value in loadings):
        raise ValueError("--models and --loadings must be non-empty; loadings must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for loading in loadings:
            result = _assemble(model, loading, args)
            output = args.output_dir / f"{model}-{loading}{args.guest}-variance-hybrid-heat-capacity.csv"
            _write_csv(output, result)
            print(f"Saved variance hybrid data: {output}")


if __name__ == "__main__":
    main()
