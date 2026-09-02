"""Run the complete thermodynamic and structural analysis for MD trajectories."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
from pathlib import Path
import sys

import numpy as np


from ..config import find_classical_output_file, load_run_config, run_output_parts
from .lammps import read_lammps_thermo
from .statistics import (
    autocorrelation,
    classical_fluctuation_heat_capacity,
    running_mean,
    summarize_series,
)
from .trajectory import (
    methane_mean_squared_displacement,
    read_trajectory_observables,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


CORE_SERIES = (
    "temperature_K",
    "total_energy_eV",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "volume_A3",
    "density_g_cm3",
    "pressure_bar",
    "max_force_eV_A",
    "rms_force_eV_A",
    "cell_a_A",
    "cell_b_A",
    "cell_c_A",
    "cell_alpha_deg",
    "cell_beta_deg",
    "cell_gamma_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=PROJECT_DIR / "configs"
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=PROJECT_DIR / "output" / "analysis"
    )
    parser.add_argument(
        "--runs",
        default="*",
        help="Comma-separated run-name glob patterns (default: every completed run)",
    )
    equilibration = parser.add_mutually_exclusive_group()
    equilibration.add_argument(
        "--equilibration-fraction",
        type=float,
        default=0.5,
        help="Discard this initial fraction of each trajectory (default: 0.5)",
    )
    equilibration.add_argument(
        "--discard-ps", type=float, help="Discard this fixed initial time in ps"
    )
    parser.add_argument("--host-atoms", type=int, default=424)
    parser.add_argument("--structural-stride", type=int, default=20)
    parser.add_argument("--rdf-stride", type=int, default=100)
    parser.add_argument("--rdf-bins", type=int, default=80)
    parser.add_argument("--rdf-cutoff", type=float)
    parser.add_argument(
        "--block-size",
        type=int,
        help="Block length in saved frames; defaults to twice each series' tau_int",
    )
    parser.add_argument("--no-plots", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--run-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    execution.add_argument(
        "--aggregate-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        return
    names = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_from_summary(summary: dict) -> dict:
    statistics = summary["statistics"]
    provenance = summary["simulation_provenance"]
    model_path = provenance["checkpoint"] or provenance["exported_model"]
    classical_cv = summary["classical_fluctuation_cv"]["cv_J_per_gK"]
    if classical_cv is None:
        classical_cv = float("nan")
    return {
        "run": summary["run"],
        "model": Path(model_path).stem,
        "methane_loading": summary["metadata"]["methane_molecules"],
        "temperature_K": summary["target_temperature_K"],
        "production_mean_temperature_K": statistics["temperature_K"]["mean"],
        "temperature_tau_ps": statistics["temperature_K"][
            "integrated_autocorrelation_time_ps"
        ],
        "potential_energy_eV": statistics["potential_energy_eV"]["mean"],
        "energy_tau_ps": statistics["total_energy_eV"][
            "integrated_autocorrelation_time_ps"
        ],
        "density_g_cm3": statistics["density_g_cm3"]["mean"],
        "fixed_volume": summary["fixed_volume"],
        "classical_cv_J_per_gK": classical_cv,
        "warning_count": len(summary["warnings"]),
    }


def discover_runs(config_dir: Path, patterns: list[str]):
    selected = []
    skipped = []
    for path in sorted(config_dir.rglob("*.toml")):
        try:
            config = load_run_config(path)
        except (KeyError, ValueError) as error:
            skipped.append({"config": str(path), "reason": str(error)})
            continue
        if not any(fnmatch.fnmatch(config.name, pattern) for pattern in patterns):
            continue
        trajectory = find_classical_output_file(
            config, "trajectory.lammpstrj", config.md_trajectory_file
        )
        if trajectory is not None:
            selected.append((path, config, trajectory))
        else:
            skipped.append(
                {
                    "config": str(path),
                    "run": config.name,
                    "reason": "production trajectory absent",
                }
            )
    return selected, skipped


def production_start(args: argparse.Namespace, configured_duration_ps: float) -> float:
    if args.discard_ps is not None:
        if args.discard_ps < 0.0:
            raise ValueError("--discard-ps must be non-negative")
        return args.discard_ps
    if not 0.0 <= args.equilibration_fraction < 1.0:
        raise ValueError("--equilibration-fraction must be in [0, 1)")
    return configured_duration_ps * args.equilibration_fraction


def running_production_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    result[mask] = running_mean(values[mask])
    return result


def plot_run(
    output_dir: Path,
    series: dict[str, np.ndarray],
    structural: dict[str, np.ndarray],
    production_mask: np.ndarray,
    correlations: dict[str, np.ndarray],
    correlation_time_ps: np.ndarray,
    msd_time_ps: np.ndarray,
    msd_A2: np.ndarray,
) -> None:
    import os

    matplotlib_cache = output_dir.parent / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = series["time_ps"]
    start = float(time[production_mask][0])
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    panels = (
        ("temperature_K", "Temperature (K)"),
        ("potential_energy_eV", "Potential energy (eV)"),
        ("total_energy_eV", "Total energy (eV)"),
        ("density_g_cm3", r"Density (g cm$^{-3}$)"),
        ("pressure_bar", "Diagnostic pressure (bar)"),
        ("max_force_eV_A", r"Maximum force (eV $\AA^{-1}$)"),
    )
    for axis, (name, label) in zip(axes.flat, panels):
        axis.plot(time, series[name], linewidth=0.7, alpha=0.65)
        axis.plot(
            time,
            running_production_mean(series[name], production_mask),
            linewidth=1.7,
            label="production running mean",
        )
        axis.axvline(start, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
        if np.ptp(series[name]) <= max(1.0, abs(float(np.mean(series[name])))) * 1e-10:
            center = float(np.mean(series[name]))
            half_width = max(abs(center) * 1e-5, 1e-6)
            axis.set_ylim(center - half_width, center + half_width)
            axis.text(
                0.02, 0.08, "constant within numerical precision",
                transform=axis.transAxes, fontsize=8,
            )
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time (ps)")
    axes[-1, 1].set_xlabel("Time (ps)")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "timeseries.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    cell_panels = (
        (("volume_A3",), r"Volume ($\AA^3$)"),
        (("density_g_cm3",), r"Density (g cm$^{-3}$)"),
        (("cell_a_A", "cell_b_A", "cell_c_A"), r"Cell length ($\AA$)"),
        (
            ("cell_alpha_deg", "cell_beta_deg", "cell_gamma_deg"),
            "Cell angle (degree)",
        ),
        (("cell_xy_A", "cell_xz_A", "cell_yz_A"), r"Cell tilt ($\AA$)"),
        (
            ("pressure_xx_bar", "pressure_yy_bar", "pressure_zz_bar"),
            "Normal pressure component (bar)",
        ),
    )
    for axis, (names, label) in zip(axes.flat, cell_panels):
        for name in names:
            axis.plot(time, series[name], linewidth=0.7, alpha=0.7, label=name)
        axis.axvline(start, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        if len(names) > 1:
            axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel("Time (ps)")
    axes[-1, 1].set_xlabel("Time (ps)")
    figure.tight_layout()
    figure.savefig(output_dir / "cell.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for name, values in correlations.items():
        axis.plot(correlation_time_ps[: len(values)], values, label=name)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xlabel="Lag (ps)", ylabel="Normalized autocorrelation")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "autocorrelation.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    structural_time = structural["time_ps"]
    axes[0, 0].plot(structural_time, structural["framework_rmsd_A"])
    axes[0, 0].set(xlabel="Time (ps)", ylabel=r"Framework RMSD ($\AA$)")
    has_methane = structural["methane_com_unwrapped_A"].shape[1] > 0
    if has_methane:
        axes[0, 1].plot(
            structural_time, structural["minimum_host_methane_com_distance_A"]
        )
        axes[0, 1].set(
            xlabel="Time (ps)",
            ylabel=r"Minimum host--CH$_4$ COM distance ($\AA$)",
        )
        axes[1, 0].plot(
            structural["rdf_distance_A"], structural["host_methane_com_rdf"],
            label=r"host atom--CH$_4$ COM",
        )
        axes[1, 0].plot(
            structural["rdf_distance_A"], structural["methane_com_rdf"],
            label=r"CH$_4$ COM--COM",
        )
        axes[1, 0].set(xlabel=r"Distance ($\AA$)", ylabel="g(r)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].plot(msd_time_ps, msd_A2)
        axes[1, 1].set(xlabel="Lag (ps)", ylabel=r"Methane COM MSD ($\AA^2$)")
    else:
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.set_axis_off()
            axis.text(
                0.5,
                0.5,
                "Not applicable: pristine MOF-5",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "structure.png", dpi=180)
    plt.close(figure)

def analyze_run(path, config, trajectory, args) -> tuple[dict, dict]:
    parts = run_output_parts(config)
    run_output = args.analysis_dir.joinpath(*parts) if parts else args.analysis_dir / config.name
    run_output.mkdir(parents=True, exist_ok=True)
    duration_ps = config.md_steps * config.timestep_fs / 1000.0
    start_ps = production_start(args, duration_ps)
    print(f"Analyzing {config.name}: {trajectory}", flush=True)
    thermodynamic_series = None
    thermo_log = None
    if config.md_driver == "lammps":
        thermo_log = find_classical_output_file(
            config, "md.lammps.log", f"{config.md_prefix}.lammps.log"
        )
        if thermo_log is None:
            raise FileNotFoundError(f"LAMMPS thermo log is missing for {config.name}")
        thermodynamic_series = read_lammps_thermo(thermo_log)
    extracted = read_trajectory_observables(
        trajectory,
        frame_spacing_fs=config.timestep_fs * config.output_stride,
        production_start_ps=start_ps,
        host_atoms=args.host_atoms,
        structural_stride=args.structural_stride,
        rdf_stride=args.rdf_stride,
        rdf_bins=args.rdf_bins,
        rdf_cutoff_A=args.rdf_cutoff,
        thermodynamic_series=thermodynamic_series,
    )
    series = extracted["series"]
    structural = extracted["structural"]
    production_mask = series["time_ps"] >= start_ps
    if np.count_nonzero(production_mask) < 4:
        raise ValueError(
            f"{config.name} has fewer than four production frames after {start_ps:g} ps"
        )
    production_time = series["time_ps"][production_mask]

    statistics = {
        name: summarize_series(
            series[name][production_mask],
            production_time,
            block_size=args.block_size,
        )
        for name in CORE_SERIES
    }
    structural_mask = structural["time_ps"] >= start_ps
    structural_statistics = {}
    for name in (
        "framework_rmsd_A",
        "framework_bond_rms_change_A",
        "framework_bond_max_change_A",
        "minimum_host_guest_atom_distance_A",
        "minimum_host_methane_com_distance_A",
        "minimum_methane_com_distance_A",
    ):
        values = structural[name][structural_mask]
        finite = np.isfinite(values)
        if np.count_nonzero(finite) >= 2:
            structural_statistics[name] = summarize_series(
                values[finite], structural["time_ps"][structural_mask][finite]
            )

    if config.md_ensemble == "nvt":
        energy_block_size = int(
            statistics["total_energy_eV"]["block_size_frames"]
        )
        classical_cv = {
            "available": True,
            **classical_fluctuation_heat_capacity(
                series["total_energy_eV"][production_mask],
                float(statistics["temperature_K"]["mean"]),
                float(extracted["metadata"]["total_mass_amu"]),
                block_size=energy_block_size,
            ),
        }
    else:
        classical_cv = {
            "available": False,
            "cv_J_per_gK": float("nan"),
            "reason": (
                "Total-energy fluctuations estimate C_V only in NVT; this "
                f"trajectory uses {config.md_ensemble}."
            ),
        }
    msd_time, msd = methane_mean_squared_displacement(
        structural["methane_com_unwrapped_A"],
        structural["time_ps"],
        production_start_ps=start_ps,
    )

    maximum_lag = min(500, np.count_nonzero(production_mask) - 1)
    correlation_names = (
        "temperature_K", "total_energy_eV", "potential_energy_eV",
        "density_g_cm3", "pressure_bar",
    )
    correlations = {
        name: autocorrelation(series[name][production_mask], maximum_lag)
        for name in correlation_names
    }
    saved_spacing_ps = float(np.median(np.diff(production_time)))
    correlation_time = np.arange(maximum_lag + 1) * saved_spacing_ps

    fixed_volume = bool(
        np.ptp(series["volume_A3"])
        <= max(1.0, abs(float(series["volume_A3"].mean()))) * 1e-10
    )
    stress_validated = config.stress_validated
    target_temperature_std = config.temperature_K * math.sqrt(
        2.0 / (3.0 * extracted["metadata"]["atom_count"])
    )
    warnings = []
    if duration_ps <= 2.0:
        warnings.append("The trajectory is only 2 ps or shorter; convergence evidence is weak.")
    if fixed_volume:
        warnings.append(
            "Volume and density are fixed by NVT; this cannot establish equilibrium density "
            "or thermal expansion. Use validated-stress NPT trajectories."
        )
    if not stress_validated:
        warnings.append(
            "The configuration does not record stress validation; pressure must not be "
            "used for NPT conclusions."
        )
    for name, item in statistics.items():
        if abs(float(item["split_stationarity_z"])) > 2.0:
            warnings.append(f"{name} differs between production halves by more than 2 SE.")
        if float(item["effective_samples"]) < 20.0:
            warnings.append(f"{name} has fewer than 20 effective production samples.")
    summary = {
        "run": config.name,
        "config": str(path),
        "trajectory": str(trajectory),
        "target_temperature_K": config.temperature_K,
        "configured_duration_ps": duration_ps,
        "actual_duration_ps": float(series["time_ps"][-1]),
        "production_start_ps": start_ps,
        "ensemble": config.md_ensemble,
        "fixed_volume": fixed_volume,
        "stress_model_validated": stress_validated,
        "simulation_provenance": {
            "md_backend": config.md_backend,
            "ad_backend": config.ad_backend,
            "checkpoint": str(config.checkpoint) if config.checkpoint else None,
            "exported_model": str(config.exported_model),
            "jax_checkpoint": str(config.jax_checkpoint) if config.jax_checkpoint else None,
            "timestep_fs": config.timestep_fs,
            "configured_steps": config.md_steps,
            "saved_frame_spacing_fs": config.timestep_fs * config.output_stride,
            "thermodynamic_log": str(thermo_log) if thermo_log else None,
        },
        "metadata": extracted["metadata"],
        "statistics": statistics,
        "structural_statistics": structural_statistics,
        "temperature_canonical_prediction": {
            "mean_K": config.temperature_K,
            "standard_deviation_K": target_temperature_std,
        },
        "classical_fluctuation_cv": classical_cv,
        "warnings": warnings,
    }
    write_json(run_output / "summary.json", summary)

    time_rows = []
    running_names = ("temperature_K", "total_energy_eV", "density_g_cm3")
    running = {
        name: running_production_mean(series[name], production_mask)
        for name in running_names
    }
    for index in range(len(series["frame"])):
        row = {name: values[index] for name, values in series.items()}
        row["production"] = bool(production_mask[index])
        for name in running_names:
            row[f"running_mean_{name}"] = running[name][index]
        time_rows.append(row)
    write_rows(run_output / "timeseries.csv", time_rows)

    structural_rows = []
    plain_structural_names = [
        name for name, values in structural.items() if np.asarray(values).ndim == 1
        and name not in {"rdf_distance_A", "host_methane_com_rdf", "methane_com_rdf"}
    ]
    for index in range(len(structural["frame"])):
        row = {name: structural[name][index] for name in plain_structural_names}
        row["production"] = bool(structural_mask[index])
        structural_rows.append(row)
    write_rows(run_output / "structural_timeseries.csv", structural_rows)
    np.savez(
        run_output / "autocorrelation.npz",
        lag_time_ps=correlation_time,
        **correlations,
    )
    np.savez(
        run_output / "structural_distributions.npz",
        rdf_distance_A=structural["rdf_distance_A"],
        host_methane_com_rdf=structural["host_methane_com_rdf"],
        methane_com_rdf=structural["methane_com_rdf"],
        msd_lag_time_ps=msd_time,
        methane_com_msd_A2=msd,
    )
    if not args.no_plots:
        plot_run(
            run_output, series, structural, production_mask, correlations,
            correlation_time, msd_time, msd,
        )

    return summary, aggregate_from_summary(summary)


def workflow_requirements(summaries: list[dict]) -> dict:
    all_fixed = bool(summaries) and all(item["fixed_volume"] for item in summaries)
    stress_valid = bool(summaries) and all(
        item["stress_model_validated"] for item in summaries
    )
    requirements = {
        "production_targets": [
            "Initial classical campaign: one 500 ps flexible-cell NPT run at "
            "each selected temperature and 1 bar, discarding the first 100 ps.",
            "Add independent replicas later to quantify placement and velocity-seed "
            "uncertainty.",
            "Hybrid C_P: differentiate converged replica-averaged loaded-system "
            "enthalpies on the temperature grid.",
            "Quantum correction: independently optimize representative loaded "
            "configurations and check Hessian/numerical convergence.",
            "Empty MOF-5: relax the supplied equilibrated structure and compute "
            "one reference Hessian without an MD campaign.",
        ],
        "available_from_current_trajectories": [
            "temperature, kinetic/potential/total energy, forces, cell, volume, density",
            "running and block statistics, drift, split-half stationarity",
            "autocorrelation times and effective sample counts",
            "framework RMSD and reference-bond distortion",
            "host--methane and methane--methane distances/RDFs, methane COM MSD",
            "classical NVT energy-fluctuation C_V diagnostic",
        ],
        "not_established_by_current_data": [
            "equilibrium density, thermal expansion, or NPT response" if all_fixed else None,
            "validated pressure/stress" if not stress_valid else None,
            "final hybrid C_p until the temperature grid and Hessians are complete",
            "uncertainty across independent initial conditions/seeds",
            "host--guest interaction-energy decomposition",
            "complete cluster provenance such as Slurm job ID and resource usage",
        ],
        "recommended_next_simulations": [
            "extend production until slow observables contain many autocorrelation times",
            "run multiple independent seeds",
            "quench representative independent loaded structures before Hessian analysis",
            "use a stress-trained/validated potential and NPT runs for density and expansion",
            "run converged neighboring temperatures for the enthalpy derivative",
            "quench representative loaded configurations and compute AD Hessians",
        ],
    }
    requirements["not_established_by_current_data"] = [
        item for item in requirements["not_established_by_current_data"]
        if item is not None
    ]
    return requirements


def plot_sweep(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    systems = sorted({
        (str(row["model"]), int(row["methane_loading"])) for row in rows
    })
    for model, methane_loading in systems:
        selected = sorted(
            (
                row for row in rows
                if row["model"] == model
                and int(row["methane_loading"]) == methane_loading
            ),
            key=lambda item: item["temperature_K"],
        )
        temperature = [row["temperature_K"] for row in selected]
        axes[0].plot(
            temperature,
            [row["density_g_cm3"] for row in selected],
            marker="o",
            label=f"{model}, {methane_loading} CH4",
        )
        axes[1].plot(
            temperature,
            [row["potential_energy_eV"] for row in selected],
            marker="o",
            label=f"{model}, {methane_loading} CH4",
        )
    axes[0].set(xlabel="MD temperature (K)", ylabel=r"Density (g cm$^{-3}$)")
    axes[1].set(xlabel="MD temperature (K)", ylabel="Potential energy (eV)")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_aggregate_outputs(
    args: argparse.Namespace,
    summaries: list[dict],
    aggregate_rows: list[dict],
    skipped: list[dict],
) -> None:
    write_rows(args.analysis_dir / "runs.csv", aggregate_rows)
    write_json(
        args.analysis_dir / "workflow_requirements.json",
        workflow_requirements(summaries),
    )
    if not args.no_plots:
        plot_sweep(args.analysis_dir / "temperature_sweep.png", aggregate_rows)
    manifest = {
        "analysis_directory": str(args.analysis_dir),
        "completed_runs": [item["run"] for item in summaries],
        "failed_runs": [],
        "skipped_configs": skipped,
        "settings": vars(args),
    }
    write_json(args.analysis_dir / "manifest.json", manifest)


def aggregate_existing_runs(
    args: argparse.Namespace,
    runs: list[tuple],
    skipped: list[dict],
) -> None:
    summaries = []
    aggregate_rows = []
    missing = []
    for _, config, _ in runs:
        parts = run_output_parts(config)
        run_output = args.analysis_dir.joinpath(*parts) if parts else args.analysis_dir / config.name
        summary_path = run_output / "summary.json"
        if not summary_path.is_file():
            missing.append(config.name)
            continue
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        aggregate_rows.append(aggregate_from_summary(summary))
    if missing:
        raise FileNotFoundError(
            "per-trajectory analysis output is missing for: " + ", ".join(missing)
        )
    write_aggregate_outputs(args, summaries, aggregate_rows, skipped)


def main() -> None:
    args = parse_args()
    args.config_dir = args.config_dir.expanduser().resolve()
    args.analysis_dir = args.analysis_dir.expanduser().resolve()
    if args.host_atoms < 1 or args.structural_stride < 1 or args.rdf_stride < 1:
        raise ValueError("host atom count and strides must be positive")
    if args.rdf_stride % args.structural_stride:
        raise ValueError("--rdf-stride must be an integer multiple of --structural-stride")
    if args.block_size is not None and args.block_size < 2:
        raise ValueError("--block-size must be at least two")
    patterns = [item.strip() for item in args.runs.split(",") if item.strip()]
    if not patterns:
        raise ValueError("--runs must contain at least one glob pattern")
    runs, skipped = discover_runs(args.config_dir, patterns)
    if not runs:
        raise FileNotFoundError("no completed trajectories match --runs")
    if args.run_only and len(runs) != 1:
        raise ValueError("--run-only requires exactly one completed trajectory")

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_existing_runs(args, runs, skipped)
        print(f"Aggregate analysis written to {args.analysis_dir}")
        return
    if args.run_only:
        path, config, trajectory = runs[0]
        analyze_run(path, config, trajectory, args)
        print(f"Finished {config.name}", flush=True)
        print(f"Analysis written to {args.analysis_dir}")
        return

    summaries = []
    aggregate_rows = []
    failures = []
    for path, config, trajectory in runs:
        try:
            summary, aggregate = analyze_run(path, config, trajectory, args)
        except Exception as error:
            failures.append({"run": config.name, "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {config.name}: {error}", file=sys.stderr, flush=True)
            continue
        summaries.append(summary)
        aggregate_rows.append(aggregate)
        print(f"Finished {config.name}", flush=True)

    if aggregate_rows:
        write_aggregate_outputs(args, summaries, aggregate_rows, skipped)
    manifest = {
        "analysis_directory": str(args.analysis_dir),
        "completed_runs": [item["run"] for item in summaries],
        "failed_runs": failures,
        "skipped_configs": skipped,
        "settings": vars(args),
    }
    write_json(args.analysis_dir / "manifest.json", manifest)
    print(f"Analysis written to {args.analysis_dir}")
    if failures:
        raise RuntimeError(f"analysis failed for {len(failures)} run(s); see manifest.json")


if __name__ == "__main__":
    main()
