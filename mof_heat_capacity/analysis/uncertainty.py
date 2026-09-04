"""Propagate model-ensemble energies into thermodynamic uncertainty."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .statistics import KB_EV_PER_K


BAR_A3_TO_EV = 6.241509074e-7


def committee_enthalpy_estimates(
    central_potential_eV: np.ndarray,
    member_potential_eV: np.ndarray,
    kinetic_energy_eV: np.ndarray,
    volume_A3: np.ndarray,
    *,
    temperature_K: float,
    pressure_bar: float,
) -> dict[str, np.ndarray]:
    """Return direct and first-order CEA enthalpy estimates for each member."""
    central = np.asarray(central_potential_eV, dtype=float)
    members = np.asarray(member_potential_eV, dtype=float)
    kinetic = np.asarray(kinetic_energy_eV, dtype=float)
    volume = np.asarray(volume_A3, dtype=float)
    if members.ndim != 2:
        raise ValueError("member energies must have shape (frames, members)")
    if central.ndim != 1 or kinetic.ndim != 1 or volume.ndim != 1:
        raise ValueError("central energy, kinetic energy, and volume must be 1D")
    if not (len(central) == len(kinetic) == len(volume) == members.shape[0]):
        raise ValueError("committee reweighting arrays must contain the same frames")
    if len(central) < 2 or members.shape[1] < 2:
        raise ValueError("committee reweighting requires at least two frames and members")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    arrays = (central, members, kinetic, volume)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("committee reweighting arrays must be finite")

    beta = 1.0 / (KB_EV_PER_K * temperature_K)
    delta = members - central[:, None]
    dimensionless_delta = beta * delta
    member_enthalpy = (
        kinetic[:, None]
        + members
        + pressure_bar * volume[:, None] * BAR_A3_TO_EV
    )

    shifted_log_weights = -dimensionless_delta
    shifted_log_weights -= shifted_log_weights.max(axis=0, keepdims=True)
    weights = np.exp(shifted_log_weights)
    weights /= weights.sum(axis=0, keepdims=True)
    direct = np.sum(weights * member_enthalpy, axis=0)
    effective_samples = 1.0 / np.sum(weights**2, axis=0)

    centered_enthalpy = member_enthalpy - member_enthalpy.mean(
        axis=0, keepdims=True
    )
    centered_delta = delta - delta.mean(axis=0, keepdims=True)
    covariance = np.mean(centered_enthalpy * centered_delta, axis=0)
    cea = member_enthalpy.mean(axis=0) - beta * covariance

    return {
        "direct_mean_enthalpy_eV": direct,
        "cea_mean_enthalpy_eV": cea,
        "effective_samples": effective_samples,
        "dimensionless_delta_variance": np.var(
            dimensionless_delta, axis=0, ddof=0
        ),
        "delta_potential_eV": delta,
    }


def _system_values(output, batch_size: int, name: str) -> np.ndarray:
    values = output[0].values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float).reshape(batch_size, -1)
    if values.shape[1] != 1:
        raise ValueError(f"{name} must provide one system value per frame")
    return values[:, 0]


def _ensemble_values(output, batch_size: int) -> np.ndarray:
    values = output[0].values.detach().cpu().numpy()
    values = np.asarray(values, dtype=float).reshape(batch_size, -1)
    if values.shape[1] < 2:
        raise ValueError("energy_ensemble must provide at least two members")
    return values


def _model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_trajectory_committee(
    trajectory_path: Path,
    thermodynamic_series: dict[str, np.ndarray],
    production_mask: np.ndarray,
    *,
    model_path: Path,
    device: str,
    temperature_K: float,
    pressure_bar: float,
    stride: int,
    batch_size: int,
    central_tolerance_eV: float,
) -> dict[str, np.ndarray | str | float]:
    """Evaluate model-ensemble energies on selected production frames."""
    from ase.io import iread
    import metatomic.torch as metatomic_torch
    from metatomic.torch import ModelOutput
    from metatomic_ase import MetatomicCalculator

    if stride < 1 or batch_size < 1 or central_tolerance_eV <= 0.0:
        raise ValueError(
            "uncertainty stride, batch size, and central tolerance must be positive"
        )
    model = model_path.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"uncertainty model not found: {model}")

    raw_steps = np.asarray(thermodynamic_series["md_step"])
    keep_raw = np.ones(len(raw_steps), dtype=bool)
    keep_raw[1:] = raw_steps[1:] != raw_steps[:-1]
    unique_raw_indices = np.flatnonzero(keep_raw)
    mask = np.asarray(production_mask, dtype=bool)
    if len(unique_raw_indices) != len(mask):
        raise ValueError("unique thermo frames do not align with trajectory analysis")
    selected_unique = np.flatnonzero(mask)[::stride]
    selected_raw = unique_raw_indices[selected_unique]
    selected_raw_set = set(int(index) for index in selected_raw)

    loaded_model = metatomic_torch.load_atomistic_model(str(model))
    capabilities = set(loaded_model.capabilities().outputs)
    missing = sorted({"energy", "energy_ensemble"}.difference(capabilities))
    if missing:
        raise ValueError(
            "uncertainty model is missing required output(s): " + ", ".join(missing)
        )
    has_analytical_uncertainty = "energy_uncertainty" in capabilities
    del loaded_model
    calculator = MetatomicCalculator(str(model), device=device)
    requested = {
        "energy": ModelOutput(sample_kind="system"),
        "energy_ensemble": ModelOutput(sample_kind="system"),
    }
    if has_analytical_uncertainty:
        requested["energy_uncertainty"] = ModelOutput(sample_kind="system")
    central_batches = []
    member_batches = []
    uncertainty_batches = []
    structures = []

    def evaluate_batch() -> None:
        if not structures:
            return
        try:
            outputs = calculator.run_model(structures, requested)
        except Exception as error:
            raise RuntimeError(
                "the configured exported model could not evaluate energy, "
                "and energy_ensemble; use an exported, calibrated ensemble model"
            ) from error
        count = len(structures)
        central_batches.append(_system_values(outputs["energy"], count, "energy"))
        member_batches.append(_ensemble_values(outputs["energy_ensemble"], count))
        if has_analytical_uncertainty:
            uncertainty_batches.append(
                _system_values(
                    outputs["energy_uncertainty"], count, "energy_uncertainty"
                )
            )
        else:
            uncertainty_batches.append(np.full(count, np.nan))
        structures.clear()

    for raw_index, atoms in enumerate(iread(str(trajectory_path), index=":")):
        if raw_index not in selected_raw_set:
            continue
        structures.append(atoms)
        if len(structures) == batch_size:
            evaluate_batch()
    evaluate_batch()
    if not central_batches:
        raise ValueError("no production frames were selected for model uncertainty")

    central = np.concatenate(central_batches)
    members = np.concatenate(member_batches)
    analytical_uncertainty = np.concatenate(uncertainty_batches)
    if len(central) != len(selected_unique):
        raise ValueError("trajectory ended before all uncertainty frames were evaluated")
    logged_central = np.asarray(
        thermodynamic_series["potential_energy_eV"], dtype=float
    )[selected_raw]
    kinetic = np.asarray(
        thermodynamic_series["kinetic_energy_eV"], dtype=float
    )[selected_raw]
    volume = np.asarray(thermodynamic_series["volume_A3"], dtype=float)[selected_raw]
    estimates = committee_enthalpy_estimates(
        logged_central,
        members,
        kinetic,
        volume,
        temperature_K=temperature_K,
        pressure_bar=pressure_bar,
    )

    ensemble_mean_residual = members.mean(axis=1) - central
    logged_energy_residual = central - logged_central
    logged_energy_residual -= logged_energy_residual.mean()
    sampling_model_residual = members.mean(axis=1) - logged_central
    sampling_model_residual -= sampling_model_residual.mean()
    sampling_model_max = float(np.max(np.abs(sampling_model_residual)))
    if sampling_model_max > central_tolerance_eV:
        raise ValueError(
            "ensemble mean does not reproduce the trajectory-driving potential "
            "after removal of a constant energy offset: maximum residual "
            f"{sampling_model_max:.6g} eV exceeds {central_tolerance_eV:.6g} eV"
        )
    return {
        "frame": selected_unique.astype(np.int64),
        "md_step": np.asarray(thermodynamic_series["md_step"])[selected_raw],
        "time_ps": np.asarray(thermodynamic_series["time_ps"])[selected_raw],
        "central_potential_eV": central,
        "logged_central_potential_eV": logged_central,
        "member_potential_eV": members,
        "analytical_energy_uncertainty_eV": analytical_uncertainty,
        **estimates,
        "model_path": str(model),
        "model_sha256": _model_sha256(model),
        "ensemble_mean_residual_rms_eV": float(
            np.sqrt(np.mean(ensemble_mean_residual**2))
        ),
        "logged_energy_residual_rms_eV": float(
            np.sqrt(np.mean(logged_energy_residual**2))
        ),
        "logged_energy_residual_max_abs_eV": float(
            np.max(np.abs(logged_energy_residual))
        ),
        "sampling_model_residual_rms_eV": float(
            np.sqrt(np.mean(sampling_model_residual**2))
        ),
        "sampling_model_residual_max_abs_eV": sampling_model_max,
    }


def committee_standard_deviation(values: np.ndarray) -> np.ndarray:
    """Return the sample standard deviation over committee members."""
    array = np.asarray(values, dtype=float)
    if array.shape[0] < 2:
        raise ValueError("at least two committee members are required")
    return array.std(axis=0, ddof=1)
