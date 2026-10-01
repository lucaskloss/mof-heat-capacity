"""Compare LLPR CEA finite-difference and Gaussian NPT fluctuation heat capacities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mof_heat_capacity.analysis.statistics import AMU_TO_G, EV_TO_J, KB_EV_PER_K
from mof_heat_capacity.analysis.statistics import integrated_autocorrelation_time
from mof_heat_capacity.analysis.uncertainty import BAR_A3_TO_EV
from mof_heat_capacity.analysis.uncertainty_comparison import archive_sha256


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DIR = PROJECT_DIR / "output/post-processing/trajectory-analysis"


def _positive_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected comma-separated unique positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", default="pet-mad-1.5-s-40nn")
    parser.add_argument("--loading", type=int, required=True)
    parser.add_argument("--replicas", type=_positive_int_list, default=[1])
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--require-all-frames", action="store_true",
        help="Reject archives without stride-1 coverage of all saved production frames",
    )
    return parser.parse_args()


def _load_configuration(summary_path: Path) -> tuple[dict, float]:
    summary = json.loads(summary_path.read_text())
    config_path = Path(summary["config"])
    config = tomllib.loads(config_path.read_text())
    return summary, float(config["md"]["pressure_bar"])


def _run_directory(
    analysis_dir: Path, model_label: str, loading: int, temperature: float, replica: int
) -> Path:
    return (
        analysis_dir
        / model_label
        / f"{loading}ch4"
        / f"{temperature:g}K"
        / f"rep{replica:02d}"
    )


def compare(args: argparse.Namespace) -> tuple[Path, Path]:
    analysis_dir = args.analysis_dir.expanduser().resolve()
    ensemble_path = analysis_dir / args.model_label / f"{args.loading}ch4" / "model_uncertainty_heat_capacity.npz"
    if not ensemble_path.is_file():
        raise FileNotFoundError(
            f"CEA heat-capacity archive not found: {ensemble_path}; run trajectory analysis with --model-uncertainty"
        )

    with np.load(ensemble_path) as ensemble:
        temperatures = np.asarray(ensemble["temperatures_K"], dtype=float)
        cea_by_member = np.asarray(ensemble["cea_cp_by_member_J_per_gK"], dtype=float)
        direct_by_member = np.asarray(ensemble["direct_cp_by_member_J_per_gK"], dtype=float)
        expected_hash = str(ensemble["model_sha256"])
        expected_frame_counts = np.asarray(ensemble["selected_frame_count"], dtype=int) if "selected_frame_count" in ensemble else None
        expected_stride = int(ensemble["frame_stride"].item()) if "frame_stride" in ensemble else None
    cea_source_sha256 = archive_sha256(ensemble_path)
    if args.require_all_frames and (expected_frame_counts is None or expected_stride != 1):
        raise ValueError("rerun stride-1 trajectory analysis and aggregation before an all-frame comparison")

    member_count = cea_by_member.shape[0]
    if cea_by_member.shape != direct_by_member.shape or cea_by_member.shape[1] != len(temperatures):
        raise ValueError(f"invalid member curves in {ensemble_path}")

    variance_by_member = np.empty_like(cea_by_member)
    central_variance = np.empty(len(temperatures), dtype=float)
    central_variance_sampling_se = np.empty(len(temperatures), dtype=float)
    frame_counts = np.zeros(len(temperatures), dtype=int)
    production_counts = np.zeros(len(temperatures), dtype=int)
    frame_strides: set[int] = set()
    atom_counts: set[int] = set()
    masses: set[float] = set()
    pressures: set[float] = set()
    for temp_index, temperature in enumerate(temperatures):
        replica_member_curves = []
        replica_central_curves = []
        replica_central_standard_errors = []
        for replica in args.replicas:
            run_dir = _run_directory(
                analysis_dir, args.model_label, args.loading, temperature, replica
            )
            summary, pressure_bar = _load_configuration(run_dir / "summary.json")
            archive_path = run_dir / "model_uncertainty.npz"
            with np.load(archive_path) as archive:
                model_hash = str(archive["model_sha256"])
                if model_hash != expected_hash:
                    raise ValueError(f"LLPR checkpoint mismatch in {archive_path}")
                central_potential = np.asarray(archive["central_potential_eV"], dtype=float)
                member_potential = np.asarray(archive["member_potential_eV"], dtype=float)
                volume = np.asarray(archive["volume_A3"], dtype=float)
                selected_count = len(central_potential)
                stride = int(archive["frame_stride"].item()) if "frame_stride" in archive else int(summary["model_uncertainty"]["frame_stride"])
                available_count = int(archive["production_frame_count"].item()) if "production_frame_count" in archive else -1
                if selected_count != int(summary["model_uncertainty"]["sampled_frames"]):
                    raise ValueError(f"summary and LLPR archive frame counts differ: {run_dir}; rerun analysis")
                if expected_stride is not None and stride != expected_stride:
                    raise ValueError(f"CEA aggregate and LLPR archive strides differ: {run_dir}; rerun aggregation")
                if args.require_all_frames and (stride != 1 or selected_count != available_count):
                    raise ValueError(f"LLPR archive does not cover all saved production frames: {archive_path}")
                frame_strides.add(stride)
                if available_count < 0 or production_counts[temp_index] < 0:
                    production_counts[temp_index] = -1
                else:
                    production_counts[temp_index] += available_count
                if member_potential.shape != (len(central_potential), member_count):
                    raise ValueError(f"member energy shape mismatch in {archive_path}")
                if len(central_potential) < 2:
                    raise ValueError(f"too few frames in {archive_path}")

            mass_amu = float(summary["metadata"]["total_mass_amu"])
            atom_count = int(summary["metadata"]["atom_count"])
            atom_counts.add(atom_count)
            masses.add(mass_amu)
            pressures.add(pressure_bar)
            frame_counts[temp_index] += len(central_potential)

            # NPT fluctuations use configurational enthalpy Q = U + P V.
            # Initial mom yes removes three translational degrees of freedom.
            degrees_of_freedom = 3 * atom_count - 3
            q_members = member_potential + pressure_bar * volume[:, None] * BAR_A3_TO_EV
            q_central = central_potential + pressure_bar * volume * BAR_A3_TO_EV
            kinetic_cp_eV_per_K = 0.5 * degrees_of_freedom * KB_EV_PER_K
            conversion = EV_TO_J / (mass_amu * AMU_TO_G)
            member_cp = (
                np.var(q_members, axis=0, ddof=1)
                / (KB_EV_PER_K * temperature**2)
                + kinetic_cp_eV_per_K
            ) * conversion
            central_cp = (
                np.var(q_central, ddof=1)
                / (KB_EV_PER_K * temperature**2)
                + kinetic_cp_eV_per_K
            ) * conversion
            replica_member_curves.append(member_cp)
            replica_central_curves.append(central_cp)
            centered_q = q_central - q_central.mean()
            variance_influence = centered_q**2 - np.var(q_central, ddof=1)
            tau_variance = integrated_autocorrelation_time(variance_influence)
            variance_se_eV2 = np.sqrt(
                np.var(variance_influence, ddof=1)
                * 2.0
                * tau_variance
                / len(q_central)
            )
            replica_central_standard_errors.append(
                variance_se_eV2
                / (KB_EV_PER_K * temperature**2)
                * conversion
            )

        variance_by_member[:, temp_index] = np.mean(replica_member_curves, axis=0)
        central_variance[temp_index] = float(np.mean(replica_central_curves))
        within_replica_variance = np.mean(
            np.square(replica_central_standard_errors)
        ) / len(args.replicas)
        between_replica_variance = (
            np.var(replica_central_curves, ddof=1) / len(args.replicas)
            if len(args.replicas) > 1
            else 0.0
        )
        central_variance_sampling_se[temp_index] = np.sqrt(
            within_replica_variance + between_replica_variance
        )

    if expected_frame_counts is not None and not np.array_equal(frame_counts, expected_frame_counts):
        raise ValueError("CEA aggregate and Gaussian comparison select different frame counts or replicas; rerun aggregation")
    if len(frame_strides) != 1:
        raise ValueError("selected LLPR runs do not share a frame stride")
    if len(atom_counts) != 1 or len(masses) != 1 or len(pressures) != 1:
        raise ValueError("selected replicas do not share atom count, mass, and pressure")

    output_dir = (args.output_dir or ensemble_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "variance_heat_capacity_comparison"
    archive_output = output_dir / f"{stem}.npz"
    csv_output = output_dir / f"{stem}.csv"
    np.savez(
        archive_output,
        temperatures_K=temperatures,
        cea_cp_by_member_J_per_gK=cea_by_member,
        direct_cp_by_member_J_per_gK=direct_by_member,
        gaussian_npt_cp_by_member_J_per_gK=variance_by_member,
        cea_cp_mean_J_per_gK=cea_by_member.mean(axis=0),
        cea_cp_model_standard_deviation_J_per_gK=cea_by_member.std(axis=0, ddof=1),
        direct_cp_mean_J_per_gK=direct_by_member.mean(axis=0),
        direct_cp_model_standard_deviation_J_per_gK=direct_by_member.std(axis=0, ddof=1),
        gaussian_npt_cp_mean_J_per_gK=variance_by_member.mean(axis=0),
        gaussian_npt_cp_model_standard_deviation_J_per_gK=variance_by_member.std(axis=0, ddof=1),
        gaussian_npt_central_cp_J_per_gK=central_variance,
        gaussian_npt_central_sampling_standard_error_J_per_gK=central_variance_sampling_se,
        selected_frame_count=frame_counts,
        production_frame_count=production_counts,
        frame_stride=next(iter(frame_strides)),
        cea_source_sha256=cea_source_sha256,
        atom_count=next(iter(atom_counts)),
        pressure_bar=next(iter(pressures)),
        replica_ids=np.asarray(args.replicas, dtype=int),
        model_sha256=expected_hash,
        method="Gaussian NPT centered-moment approximation applied to U_member + P*V; kinetic term f*kB/2 with f=3N-3",
    )

    columns = [
        "temperature_K",
        "cea_finite_difference_mean_J_per_gK",
        "cea_finite_difference_model_sd_J_per_gK",
        "direct_reweighting_finite_difference_mean_J_per_gK",
        "direct_reweighting_finite_difference_model_sd_J_per_gK",
        "gaussian_npt_variance_mean_J_per_gK",
        "gaussian_npt_variance_model_sd_J_per_gK",
        "gaussian_npt_central_cp_J_per_gK",
        "gaussian_npt_central_sampling_standard_error_J_per_gK",
        "selected_frames",
        "production_frames",
        "frame_stride",
    ]
    with csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, temperature in enumerate(temperatures):
            writer.writerow(
                [
                    temperature,
                    cea_by_member[:, index].mean(),
                    cea_by_member[:, index].std(ddof=1),
                    direct_by_member[:, index].mean(),
                    direct_by_member[:, index].std(ddof=1),
                    variance_by_member[:, index].mean(),
                    variance_by_member[:, index].std(ddof=1),
                    central_variance[index],
                    central_variance_sampling_se[index],
                    frame_counts[index],
                    production_counts[index],
                    next(iter(frame_strides)),
                ]
            )

    _plot_comparison(temperatures, cea_by_member, variance_by_member, central_variance, output_dir / f"{stem}.png")
    return csv_output, archive_output


def _plot_comparison(
    temperatures: np.ndarray,
    cea_by_member: np.ndarray,
    variance_by_member: np.ndarray,
    central_variance: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    cea_mean = cea_by_member.mean(axis=0)
    cea_sd = cea_by_member.std(axis=0, ddof=1)
    variance_mean = variance_by_member.mean(axis=0)
    variance_sd = variance_by_member.std(axis=0, ddof=1)
    axes[0].plot(temperatures, cea_mean, "o-", label="CEA + finite difference")
    axes[0].fill_between(temperatures, cea_mean - cea_sd, cea_mean + cea_sd, alpha=0.18)
    axes[0].plot(temperatures, variance_mean, "s-", label="Gaussian NPT variance")
    axes[0].fill_between(
        temperatures, variance_mean - variance_sd, variance_mean + variance_sd, alpha=0.18
    )
    axes[0].plot(temperatures, central_variance, "k--", label="Central-model variance")
    axes[0].set_ylabel(r"Classical $C_P$ (J g$^{-1}$ K$^{-1}$)")
    axes[0].legend(fontsize=8)
    axes[1].plot(temperatures, cea_sd, "o-", label="CEA committee SD")
    axes[1].plot(temperatures, variance_sd, "s-", label="Variance committee SD")
    axes[1].set(xlabel="Temperature (K)", ylabel=r"LLPR model SD (J g$^{-1}$ K$^{-1}$)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.loading <= 0:
        raise SystemExit("--loading must be positive")
    csv_path, archive_path = compare(args)
    print(f"Saved estimator comparison: {csv_path}")
    print(f"Saved member curves: {archive_path}")


if __name__ == "__main__":
    main()
