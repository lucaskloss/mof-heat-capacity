#!/usr/bin/env python3
"""Plot CEA and DPOSE hybrid uncertainty components with the same MD sampling error."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mof_heat_capacity.campaign import GUEST_SYMBOLS, structure_directory, system_directory, validate_selection

from mof_heat_capacity.analysis.uncertainty_comparison import hybrid_uncertainty_components


MODELS = ("pet-mad-1.5-s-40nn", "pet-sol-s-best")
MODEL_TITLES = {MODELS[0]: "PET-MAD", MODELS[1]: "PET-SOL"}
METHOD_TITLES = {"cea": "CEA", "dpose": "DPOSE (Gaussian NPT)"}
COMMON_FIELDS = (
    "temperature_K",
    "central_hybrid_cp_J_per_gK",
    "md_sampling_standard_error_J_per_gK",
    "harmonic_model_standard_deviation_J_per_gK",
    "selected_frame_count",
    "production_frame_count",
    "frame_stride",
    "replica_counts",
)


def components(method: str) -> tuple[tuple[str, str, str, str], ...]:
    classical_label = "Classical CEA / LLPR" if method == "cea" else "Classical DPOSE / LLPR (Gaussian NPT)"
    return (
        ("MD sampling", "md_sampling_standard_error_J_per_gK", "tab:blue", "-"),
        (classical_label, "classical_model_standard_deviation_J_per_gK", "tab:orange", "-"),
        ("Hessian LLPR spread", "harmonic_model_standard_deviation_J_per_gK", "tab:green", "-"),
        ("Combined uncertainty (paired members)", "combined_standard_uncertainty_J_per_gK", "tab:red", "-"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof", default="mof5")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--loadings", default="50,100,150")
    parser.add_argument(
        "--input-root", type=Path,
        default=Path("output/post-processing/harmonic-correction"),
    )
    parser.add_argument(
        "--analysis-root", type=Path,
        default=Path("output/post-processing/trajectory-analysis"),
    )
    parser.add_argument(
        "--uncertainty-method", choices=("cea", "dpose", "both"), default="cea",
        help="'both' writes matching plots with -cea and -dpose appended to --output",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-all-frames", action="store_true",
        help="Reject sparse, stale, or unverified LLPR archives before plotting",
    )
    return parser.parse_args()


def write_csv(
    path: Path, method: str,
    results: dict[tuple[str, int], dict[str, np.ndarray]],
    mof: str = "mof5", guest: str = "ch4",
) -> None:
    fields = list(next(iter(results.values())))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "mof", "guest", "model", "loading", *fields])
        writer.writeheader()
        for (model, loading), values in results.items():
            for index in range(len(values["temperature_K"])):
                writer.writerow({
                    "method": method, "mof": mof, "guest": guest, "model": model, "loading": loading,
                    **{field: values[field][index] for field in fields},
                })


def plot_components(
    path: Path, method: str, loadings: list[int],
    results: dict[tuple[str, int], dict[str, np.ndarray]], ymax: float,
    mof: str = "mof5", guest: str = "ch4",
) -> None:
    figure, axes = plt.subplots(
        len(MODELS), len(loadings), figsize=(4.7 * len(loadings), 7.6),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, model in enumerate(MODELS):
        for column, loading in enumerate(loadings):
            axis = axes[row, column]
            values = results[(model, loading)]
            for label, field, color, linestyle in components(method):
                axis.plot(
                    values["temperature_K"], values[field], marker="o",
                    markersize=4, linewidth=1.8, color=color,
                    linestyle=linestyle, label=label,
                )
            axis.set_title(f"{MODEL_TITLES[model]}, {loading} {guest.upper()}")
            axis.set_ylim(0, ymax)
            axis.grid(alpha=0.25)
            if row == len(MODELS) - 1:
                axis.set_xlabel("Temperature (K)")
            if column == 0:
                axis.set_ylabel(r"Standard uncertainty (J g$^{-1}$ K$^{-1}$)")

    figure.suptitle(f"{mof} + {guest.upper()}: hybrid uncertainty — {METHOD_TITLES[method]}", y=0.985, fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95),
        ncol=2, frameon=False, fontsize=10,
    )
    figure.text(
        0.5, 0.012,
        "Exploratory mode handling. Uncertainty includes MD sampling and paired classical/Hessian LLPR spread.",
        ha="center", fontsize=9,
    )
    counts = np.concatenate([values["selected_frame_count"] for values in results.values()])
    strides = np.concatenate([values["frame_stride"] for values in results.values()])
    if np.all(counts > 0) and np.all(strides > 0):
        count_label = str(int(counts[0])) if np.all(counts == counts[0]) else f"{counts.min()}–{counts.max()}"
        stride_label = ", ".join(str(int(value)) for value in np.unique(strides))
        figure.text(
            0.5, 0.038, f"LLPR: {count_label} selected frames per temperature (summed over replicas); frame stride {stride_label}.",
            ha="center", fontsize=9,
        )
    figure.subplots_adjust(
        top=0.78, bottom=0.10, left=0.07, right=0.985,
        hspace=0.30, wspace=0.05,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Saved {method.upper()} uncertainty components: {path}")


def main() -> None:
    args = parse_args()
    validate_selection(args.mof, args.guest)
    loadings = [int(value.strip()) for value in args.loadings.split(",") if value.strip()]
    if not loadings or any(value < 1 for value in loadings) or len(set(loadings)) != len(loadings):
        raise ValueError("--loadings must contain unique positive integers")
    methods = ("cea", "dpose") if args.uncertainty_method == "both" else (args.uncertainty_method,)
    results = {}
    for method in methods:
        method_results = {}
        for model in MODELS:
            for loading in loadings:
                hybrid_path = args.input_root / system_directory(model, loading, args.mof, args.guest) / "heat-capacity.exploratory-discard-imaginary.npz"
                variance_path = args.analysis_root / system_directory(model, loading, args.mof, args.guest) / "variance_heat_capacity_comparison.npz"
                ensemble_path = args.analysis_root / system_directory(model, loading, args.mof, args.guest) / "model_uncertainty_heat_capacity.npz"
                method_results[(model, loading)] = hybrid_uncertainty_components(
                    hybrid_path, method=method, variance_path=variance_path,
                    ensemble_path=ensemble_path, require_all_frames=args.require_all_frames,
                )
        results[method] = method_results

    if len(methods) == 2:
        for key in results["cea"]:
            for field in COMMON_FIELDS:
                if not np.array_equal(results["cea"][key][field], results["dpose"][key][field]):
                    raise ValueError(f"CEA and DPOSE must share {field} for {key}")
        print("Verified identical central values, MD sampling, and Hessian LLPR spread.")

    # Share limits across all panels and methods for a direct visual comparison.
    ymax = max(
        float(np.max(values[field]))
        for method, method_results in results.items()
        for values in method_results.values()
        for _, field, _, _ in components(method)
    )
    ymax = 1.08 * ymax if ymax > 0 else 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        path = args.output
        if len(methods) == 2:
            path = path.with_name(f"{path.stem}-{method}{path.suffix}")
        plot_components(path, method, loadings, results[method], ymax, args.mof, args.guest)
        write_csv(path.with_suffix(".csv"), method, results[method], args.mof, args.guest)


if __name__ == "__main__":
    main()
