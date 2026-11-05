#!/usr/bin/env python3
"""Export CEA and DPOSE hybrid uncertainty tables with the same MD sampling error."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mof_heat_capacity.campaign import GUEST_SYMBOLS, system_directory, validate_selection

from mof_heat_capacity.analysis.uncertainty_comparison import hybrid_uncertainty_components


MODELS = ("pet-mad-1.5-s-40nn", "pet-sol-s-best")
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
        help="'both' writes matching CSV tables with -cea and -dpose appended to --output",
    )
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument(
        "--require-all-frames", action="store_true",
        help="Reject sparse, stale, or unverified LLPR archives before exporting",
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        path = args.output.with_suffix(".csv")
        if len(methods) == 2:
            path = path.with_name(f"{path.stem}-{method}{path.suffix}")
        write_csv(path, method, results[method], args.mof, args.guest)
        print(f"Saved {method.upper()} uncertainty data: {path}")


if __name__ == "__main__":
    main()
