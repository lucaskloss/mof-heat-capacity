"""Assemble Hessian-corrected classical heat capacity for loaded MOF-5."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read

from ..config import load_run_config
from .lammps import read_lammps_thermo
from .statistics import AMU_TO_G, EV_TO_J, KB_EV_PER_K, summarize_series


BAR_A3_TO_EV = 6.241509074e-7
HC_OVER_K_CM_K = 1.438776877


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
        default="100,200,300,400,500",
        help="Comma-separated classical-MD temperatures",
    )
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--hybrid-dir", type=Path, default=Path("output/hybrid"))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict]]:
    records = []
    mass_amu = None
    for replica in replicas:
        replica_tag = f"{replica:02d}"
        for requested_temperature in temperatures:
            name = (
                f"mof5-{loading}ch4-{model_label}-npt-"
                f"{requested_temperature}K-rep{replica_tag}.toml"
            )
            path = configs_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"configuration not found: {path}")
            config = load_run_config(path)
            if not math.isclose(config.temperature_K, requested_temperature):
                raise ValueError(
                    f"configuration temperature does not match its selection: {path}"
                )
            if config.md_driver != "lammps" or config.md_ensemble != "npt-flexible":
                raise ValueError(f"hybrid C_P requires classical NPT: {path}")
            log_path = config.output_dir / f"{config.md_prefix}.lammps.log"
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
                    "effective_samples": summary["effective_samples"],
                    "source": str(log_path),
                }
            )

    temperature_array = np.asarray(temperatures, dtype=float)
    means = []
    errors = []
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
    return (
        temperature_array,
        np.asarray(means),
        np.asarray(errors),
        float(mass_amu),
        records,
    )


def _harmonic_corrections(
    directory: Path,
    *,
    replicas: list[int],
    temperatures: np.ndarray,
    expected_mass_amu: float,
    zero_threshold_cm1: float,
    max_near_zero_modes: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    paths = [directory / "hessians" / f"rep{replica:02d}.npz" for replica in replicas]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing requested loaded Hessian archive(s): "
            + ", ".join(str(path) for path in missing)
        )
    corrections = []
    records = []
    conversion = EV_TO_J / (expected_mass_amu * AMU_TO_G)
    for path in paths:
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
        relaxation_path = trajectory.with_suffix(".relax.json")
        if not relaxation_path.is_file():
            raise FileNotFoundError(
                f"relaxation provenance is missing for Hessian input: {relaxation_path}"
            )
        relaxation = json.loads(relaxation_path.read_text())
        if not relaxation.get("fixed_cell", False):
            raise ValueError(f"Hessian input was not relaxed at fixed cell: {trajectory}")
        if relaxation["final_max_force_eV_per_A"] > relaxation["fmax_target_eV_per_A"]:
            raise ValueError(f"Hessian input relaxation is unconverged: {trajectory}")
        current_mass = float(read(trajectory).get_masses().sum())
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
        active = frequencies > zero_threshold_cm1
        positive = frequencies[active]
        x = HC_OVER_K_CM_K * positive[:, None] / temperatures[None, :]
        decay = np.exp(-x)
        quantum_eV_per_K = KB_EV_PER_K * np.sum(
            x**2 * decay / (1.0 - decay) ** 2,
            axis=0,
        )
        classical_eV_per_K = KB_EV_PER_K * len(positive)
        correction = (quantum_eV_per_K - classical_eV_per_K) * conversion
        corrections.append(correction)
        records.append(
            {
                "source": str(path),
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
    array = np.asarray(corrections)
    mean = array.mean(axis=0)
    error = (
        array.std(axis=0, ddof=1) / math.sqrt(len(array))
        if len(array) > 1
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
    temperatures, enthalpy, enthalpy_error, mass_amu, md_records = _enthalpy_records(
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

    output = args.output or loaded_dir / "hybrid-heat-capacity.npz"
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
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved hybrid heat capacity: {output}")
    return output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
