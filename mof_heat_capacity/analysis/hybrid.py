"""Assemble Hessian-corrected classical heat capacity for loaded MOF-5."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read

from ..config import find_classical_output_file, find_loaded_config, load_run_config
from .lammps import read_lammps_thermo
from .statistics import AMU_TO_G, EV_TO_J, KB_EV_PER_K, summarize_series


BAR_A3_TO_EV = 6.241509074e-7
HC_OVER_K_CM_K = 1.438776877
ANGSTROM3_TO_CM3 = 1.0e-24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-label",
        default="pet-mad-1.5-s-40nn",
        help="Model label embedded in generated run names",
    )
    parser.add_argument("--loading", type=int, default=100)
    parser.add_argument("--replicas", default="1")
    parser.add_argument(
        "--temperatures",
        default="200,225,250,275,300,325,350,375,400",
        help="Classical-MD grid (default: 200 to 400 K in 25 K steps)",
    )
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--hybrid-dir", type=Path, default=Path("output/post-processing/harmonic-correction"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--zero-threshold-cm1", type=float, default=1.0)
    parser.add_argument(
        "--max-near-zero-modes",
        type=int,
        default=3,
        help="Maximum modes allowed within the zero-frequency threshold",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def _replicas(specification: str) -> list[int]:
    values = [int(item.strip()) for item in specification.split(",") if item.strip()]
    if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
        raise ValueError("--replicas must contain unique positive integers")
    return values


def _temperatures(specification: str) -> list[int]:
    values = [int(item.strip()) for item in specification.split(",") if item.strip()]
    if (
        len(values) < 3
        or any(value < 1 for value in values)
        or values != sorted(set(values))
    ):
        raise ValueError(
            "--temperatures must contain at least three unique, increasing "
            "positive integers"
        )
    return values


def _enthalpy_records(
    configs_dir: Path,
    *,
    model_label: str,
    loading: int,
    replicas: list[int],
    temperatures: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    list[dict],
]:
    records = []
    mass_amu = None
    for replica in replicas:
        replica_tag = f"{replica:02d}"
        for requested_temperature in temperatures:
            path = find_loaded_config(
                configs_dir,
                model_label,
                loading,
                requested_temperature,
                replica,
            )
            if not path.is_file():
                raise FileNotFoundError(f"configuration not found: {path}")
            config = load_run_config(path)
            if not math.isclose(config.temperature_K, requested_temperature):
                raise ValueError(
                    f"configuration temperature does not match its selection: {path}"
                )
            if config.md_driver != "lammps" or config.md_ensemble != "npt-flexible":
                raise ValueError(f"hybrid C_P requires classical NPT: {path}")
            log_path = find_classical_output_file(
                config, "md.lammps.log", f"{config.md_prefix}.lammps.log"
            )
            if log_path is None:
                raise FileNotFoundError(
                    f"completed LAMMPS thermo log not found for {config.name}"
                )
            thermo = read_lammps_thermo(log_path)
            start_ps = config.equilibration_steps * config.timestep_fs / 1000.0
            mask = thermo["time_ps"] >= start_ps
            if np.count_nonzero(mask) < 4:
                raise ValueError(f"insufficient production enthalpy samples: {log_path}")
            enthalpy = (
                thermo["total_energy_eV"][mask]
                + config.pressure_bar
                * thermo["volume_A3"][mask]
                * BAR_A3_TO_EV
            )
            summary = summarize_series(enthalpy, thermo["time_ps"][mask])
            volume_summary = summarize_series(
                thermo["volume_A3"][mask], thermo["time_ps"][mask]
            )
            current_mass = float(read(config.structure).get_masses().sum())
            if mass_amu is None:
                mass_amu = current_mass
            elif not math.isclose(mass_amu, current_mass, rel_tol=1e-10):
                raise ValueError("loaded MD configurations do not have a common mass")
            records.append(
                {
                    "temperature_K": config.temperature_K,
                    "replica": replica,
                    "mean_enthalpy_eV": summary["mean"],
                    "enthalpy_standard_error_eV": summary["standard_error"],
                    "mean_volume_A3": volume_summary["mean"],
                    "volume_standard_error_A3": volume_summary["standard_error"],
                    "effective_samples": summary["effective_samples"],
                    "volume_effective_samples": volume_summary["effective_samples"],
                    "source": str(log_path),
                }
            )

    temperature_array = np.asarray(temperatures, dtype=float)
    means = []
    errors = []
    volume_means = []
    volume_errors = []
    for temperature in temperature_array:
        selected = [row for row in records if row["temperature_K"] == temperature]
        present = {row["replica"] for row in selected}
        if present != set(replicas):
            raise ValueError(f"temperature {temperature:g} K lacks requested replicas")
        values = np.asarray([row["mean_enthalpy_eV"] for row in selected])
        within = np.asarray([row["enthalpy_standard_error_eV"] for row in selected])
        between_sem = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        within_sem = math.sqrt(float(np.sum(within**2))) / len(within)
        means.append(float(values.mean()))
        errors.append(math.hypot(between_sem, within_sem))
        volumes = np.asarray([row["mean_volume_A3"] for row in selected])
        volume_within = np.asarray(
            [row["volume_standard_error_A3"] for row in selected]
        )
        volume_between_sem = (
            volumes.std(ddof=1) / math.sqrt(len(volumes))
            if len(volumes) > 1
            else 0.0
        )
        volume_within_sem = math.sqrt(float(np.sum(volume_within**2))) / len(
            volume_within
        )
        volume_means.append(float(volumes.mean()))
        volume_errors.append(math.hypot(volume_between_sem, volume_within_sem))
    return (
        temperature_array,
        np.asarray(means),
        np.asarray(errors),
        np.asarray(volume_means),
        np.asarray(volume_errors),
        float(mass_amu),
        records,
    )


def _volumetric_heat_capacity(
    gravimetric: np.ndarray,
    gravimetric_error: np.ndarray,
    density: np.ndarray,
    density_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = gravimetric * density
    errors = np.hypot(gravimetric_error * density, gravimetric * density_error)
    return values, errors


def plot_hybrid_heat_capacity(
    path: Path,
    *,
    model_label: str,
    loading: int,
    temperatures: np.ndarray,
    classical: np.ndarray,
    classical_error: np.ndarray,
    correction: np.ndarray,
    correction_error: np.ndarray,
    approximate: np.ndarray,
    approximate_error: np.ndarray,
    classical_vol: np.ndarray,
    classical_error_vol: np.ndarray,
    correction_vol: np.ndarray,
    correction_error_vol: np.ndarray,
    approximate_vol: np.ndarray,
    approximate_error_vol: np.ndarray,
) -> None:
    """Plot classical, harmonic-correction, and final hybrid curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    series = (
        (
            axes[0],
            classical,
            classical_error,
            correction,
            correction_error,
            approximate,
            approximate_error,
            r"Heat capacity ($J\,g^{-1}\,K^{-1}$)",
        ),
        (
            axes[1],
            classical_vol,
            classical_error_vol,
            correction_vol,
            correction_error_vol,
            approximate_vol,
            approximate_error_vol,
            r"Heat capacity ($J\,cm^{-3}\,K^{-1}$)",
        ),
    )
    for (
        axis,
        classical_values,
        classical_errors,
        correction_values,
        correction_errors,
        approximate_values,
        approximate_errors,
        ylabel,
    ) in series:
        axis.plot(
            temperatures,
            classical_values,
            marker="o",
            linestyle="--",
            label=r"Classical $C_P$",
        )
        axis.fill_between(
            temperatures,
            classical_values - classical_errors,
            classical_values + classical_errors,
            alpha=0.12,
        )
        axis.plot(
            temperatures,
            correction_values,
            marker="o",
            linestyle=":",
            label="Harmonic correction",
        )
        axis.fill_between(
            temperatures,
            correction_values - correction_errors,
            correction_values + correction_errors,
            alpha=0.12,
        )
        axis.plot(
            temperatures,
            approximate_values,
            marker="o",
            linewidth=2.0,
            label=r"Hybrid $C_P$",
        )
        axis.fill_between(
            temperatures,
            approximate_values - approximate_errors,
            approximate_values + approximate_errors,
            alpha=0.2,
        )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
        axis.set(xlabel="Temperature (K)", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"{model_label}, {loading} CH4")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _harmonic_corrections(
    directory: Path,
    *,
    replicas: list[int],
    temperatures: np.ndarray,
    expected_mass_amu: float,
    zero_threshold_cm1: float,
    max_near_zero_modes: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    paths = {
        (int(temperature), replica): directory
        / "hessians"
        / f"{temperature:g}K"
        / f"rep{replica:02d}"
        / "hessian.npz"
        for temperature in temperatures
        for replica in replicas
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing temperature-specific loaded Hessian archive(s): "
            + ", ".join(str(path) for path in missing)
        )
    corrections = np.empty((len(replicas), len(temperatures)), dtype=float)
    records = []
    conversion = EV_TO_J / (expected_mass_amu * AMU_TO_G)
    for temperature_index, temperature in enumerate(temperatures):
        for replica_index, replica in enumerate(replicas):
            path = paths[(int(temperature), replica)]
            with np.load(path, allow_pickle=False) as data:
                frequencies = np.asarray(data["frequencies_cm1"], dtype=float)
                trajectory = Path(str(np.asarray(data["trajectory"]).item()))
                metadata = json.loads(str(np.asarray(data["metadata"]).item()))
            if metadata.get("frequency_convention") != "signed":
                raise ValueError(
                    f"Hessian archive lacks signed frequencies and cannot prove that "
                    f"its structure is a minimum; recompute it: {path}"
                )
            if frequencies.ndim == 2:
                if frequencies.shape[0] != 1:
                    raise ValueError(f"expected one optimized frame in {path}")
                frequencies = frequencies[0]
            minimum_directory = (
                directory
                / "minima"
                / f"{temperature:g}K"
                / f"rep{replica:02d}"
            )
            optimized_path = minimum_directory / "optimized.extxyz"
            relaxation_path = minimum_directory / "optimized.relax.json"
            if not optimized_path.is_file():
                raise FileNotFoundError(f"optimized Hessian input is missing: {optimized_path}")
            if not relaxation_path.is_file():
                raise FileNotFoundError(
                    "relaxation provenance is missing for Hessian input: "
                    f"{relaxation_path}"
                )
            relaxation = json.loads(relaxation_path.read_text())
            if not relaxation.get("fixed_cell", False):
                raise ValueError(
                    f"Hessian input was not relaxed at fixed cell: {trajectory}"
                )
            if (
                relaxation["final_max_force_eV_per_A"]
                > relaxation["fmax_target_eV_per_A"]
            ):
                raise ValueError(f"Hessian input relaxation is unconverged: {trajectory}")
            current_mass = float(read(optimized_path).get_masses().sum())
            if not math.isclose(current_mass, expected_mass_amu, rel_tol=1e-10):
                raise ValueError(f"Hessian and classical-MD masses differ: {path}")
            imaginary = frequencies < -zero_threshold_cm1
            if np.any(imaginary):
                raise ValueError(
                    f"optimized Hessian contains {np.count_nonzero(imaginary)} imaginary "
                    f"mode(s) below {-zero_threshold_cm1:g} cm^-1: {path}"
                )
            near_zero = np.abs(frequencies) <= zero_threshold_cm1
            if np.count_nonzero(near_zero) > max_near_zero_modes:
                raise ValueError(
                    f"optimized Hessian contains {np.count_nonzero(near_zero)} near-zero "
                    f"mode(s), exceeding the allowed {max_near_zero_modes}: {path}"
                )
            positive = frequencies[frequencies > zero_threshold_cm1]
            x = HC_OVER_K_CM_K * positive / temperature
            decay = np.exp(-x)
            quantum_eV_per_K = KB_EV_PER_K * np.sum(
                x**2 * decay / (1.0 - decay) ** 2
            )
            classical_eV_per_K = KB_EV_PER_K * len(positive)
            corrections[replica_index, temperature_index] = (
                quantum_eV_per_K - classical_eV_per_K
            ) * conversion
            records.append(
                {
                    "temperature_K": float(temperature),
                    "replica": replica,
                    "source": str(path),
                    "optimized_structure": str(optimized_path),
                    "archived_trajectory": str(trajectory),
                    "relaxation": str(relaxation_path),
                    "relaxation_steps": int(relaxation["steps"]),
                    "final_max_force_eV_per_A": float(
                        relaxation["final_max_force_eV_per_A"]
                    ),
                    "retained_modes": int(len(positive)),
                    "imaginary_modes": int(np.count_nonzero(imaginary)),
                    "near_zero_modes": int(np.count_nonzero(near_zero)),
                    "minimum_frequency_cm1": float(frequencies.min()),
                    "maximum_frequency_cm1": float(frequencies.max()),
                    "hessian_metadata": metadata,
                }
            )
    mean = corrections.mean(axis=0)
    error = (
        corrections.std(axis=0, ddof=1) / math.sqrt(len(replicas))
        if len(replicas) > 1
        else np.zeros_like(mean)
    )
    return mean, error, records


def run(args: argparse.Namespace) -> Path:
    if (
        args.loading < 1
        or args.zero_threshold_cm1 < 0.0
        or args.max_near_zero_modes < 0
    ):
        raise ValueError(
            "loading must be positive and spectral thresholds must be non-negative"
        )
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    replicas = _replicas(args.replicas)
    selected_temperatures = _temperatures(args.temperatures)
    (
        temperatures,
        enthalpy,
        enthalpy_error,
        volume_A3,
        volume_error_A3,
        mass_amu,
        md_records,
    ) = _enthalpy_records(
        args.configs_dir,
        model_label=args.model_label,
        loading=args.loading,
        replicas=replicas,
        temperatures=selected_temperatures,
    )
    rng = np.random.default_rng(2025)
    sampled_enthalpy = rng.normal(
        enthalpy,
        enthalpy_error,
        size=(args.bootstrap_samples, len(temperatures)),
    )
    cp_eV_per_K = np.gradient(enthalpy, temperatures, edge_order=2)
    sampled_cp = np.gradient(sampled_enthalpy, temperatures, axis=1, edge_order=2)
    conversion = EV_TO_J / (mass_amu * AMU_TO_G)
    classical_cp = cp_eV_per_K * conversion
    classical_error = sampled_cp.std(axis=0, ddof=1) * conversion

    loaded_dir = args.hybrid_dir / args.model_label / f"{args.loading}ch4"
    correction, correction_error, hessian_records = _harmonic_corrections(
        loaded_dir,
        replicas=replicas,
        temperatures=temperatures,
        expected_mass_amu=mass_amu,
        zero_threshold_cm1=args.zero_threshold_cm1,
        max_near_zero_modes=args.max_near_zero_modes,
    )
    approximate = classical_cp + correction
    approximate_error = np.hypot(classical_error, correction_error)
    mass_g = mass_amu * AMU_TO_G
    density = mass_g / (volume_A3 * ANGSTROM3_TO_CM3)
    density_error = density * volume_error_A3 / volume_A3
    classical_cp_vol, classical_error_vol = _volumetric_heat_capacity(
        classical_cp, classical_error, density, density_error
    )
    correction_vol, correction_error_vol = _volumetric_heat_capacity(
        correction, correction_error, density, density_error
    )
    approximate_vol, approximate_error_vol = _volumetric_heat_capacity(
        approximate, approximate_error, density, density_error
    )

    output = args.output or loaded_dir / "heat-capacity.npz"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        temperatures_K=temperatures,
        classical_anharmonic_cp_J_per_gK=classical_cp,
        classical_anharmonic_cp_standard_error_J_per_gK=classical_error,
        harmonic_quantum_correction_J_per_gK=correction,
        harmonic_quantum_correction_standard_error_J_per_gK=correction_error,
        approximate_cp_J_per_gK=approximate,
        approximate_cp_standard_error_J_per_gK=approximate_error,
        mean_volume_A3=volume_A3,
        mean_volume_standard_error_A3=volume_error_A3,
        density_g_per_cm3=density,
        density_standard_error_g_per_cm3=density_error,
        classical_anharmonic_cp_J_per_cm3K=classical_cp_vol,
        classical_anharmonic_cp_standard_error_J_per_cm3K=classical_error_vol,
        harmonic_quantum_correction_J_per_cm3K=correction_vol,
        harmonic_quantum_correction_standard_error_J_per_cm3K=correction_error_vol,
        approximate_cp_J_per_cm3K=approximate_vol,
        approximate_cp_standard_error_J_per_cm3K=approximate_error_vol,
        total_mass_amu=mass_amu,
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "temperature_K",
                "classical_anharmonic_cp_J_per_gK",
                "classical_cp_standard_error_J_per_gK",
                "harmonic_quantum_correction_J_per_gK",
                "harmonic_correction_standard_error_J_per_gK",
                "approximate_cp_J_per_gK",
                "approximate_cp_standard_error_J_per_gK",
                "mean_volume_A3",
                "mean_volume_standard_error_A3",
                "density_g_per_cm3",
                "density_standard_error_g_per_cm3",
                "classical_anharmonic_cp_J_per_cm3K",
                "classical_cp_standard_error_J_per_cm3K",
                "harmonic_quantum_correction_J_per_cm3K",
                "harmonic_correction_standard_error_J_per_cm3K",
                "approximate_cp_J_per_cm3K",
                "approximate_cp_standard_error_J_per_cm3K",
            ]
        )
        writer.writerows(
            zip(
                temperatures,
                classical_cp,
                classical_error,
                correction,
                correction_error,
                approximate,
                approximate_error,
                volume_A3,
                volume_error_A3,
                density,
                density_error,
                classical_cp_vol,
                classical_error_vol,
                correction_vol,
                correction_error_vol,
                approximate_vol,
                approximate_error_vol,
                strict=True,
            )
        )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_label": args.model_label,
                "loading": args.loading,
                "replicas": replicas,
                "temperatures_K": temperatures.tolist(),
                "zero_threshold_cm1": args.zero_threshold_cm1,
                "max_near_zero_modes": args.max_near_zero_modes,
                "classical_md_records": md_records,
                "hessian_records": hessian_records,
                "notes": [
                    "Classical term is d<Etot + Pext*V>/dT from loaded NPT MD.",
                    "Harmonic correction is C_qn_har - C_cl_har for identical retained modes.",
                    "Endpoint derivatives are second-order one-sided estimates.",
                    "Temperature spacing is a heat-capacity convergence parameter.",
                    "Volumetric values use the production NPT mean volume at each temperature.",
                    "Volumetric uncertainty propagation neglects covariance between heat capacity and volume.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    plot_path = output.with_suffix(".png")
    plot_hybrid_heat_capacity(
        plot_path,
        model_label=args.model_label,
        loading=args.loading,
        temperatures=temperatures,
        classical=classical_cp,
        classical_error=classical_error,
        correction=correction,
        correction_error=correction_error,
        approximate=approximate,
        approximate_error=approximate_error,
        classical_vol=classical_cp_vol,
        classical_error_vol=classical_error_vol,
        correction_vol=correction_vol,
        correction_error_vol=correction_error_vol,
        approximate_vol=approximate_vol,
        approximate_error_vol=approximate_error_vol,
    )
    print(f"Saved hybrid heat capacity: {output}")
    print(f"Saved hybrid heat-capacity plot: {plot_path}")
    return output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
