"""Propagate model-ensemble energies into thermodynamic uncertainty."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .statistics import KB_EV_PER_K


BAR_A3_TO_EV = 6.241509074e-7
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LLPR_CHECKPOINTS = {
    "pet-mad-1.5-s-40nn": PROJECT_DIR
    / "models"
    / "pet-mad-1.5-s_40nn_nostress-llpr.ckpt",
    "pet-sol-s-best": PROJECT_DIR
    / "models"
    / "pet_sol-s-best_nostress-llpr.ckpt",
}


def default_llpr_checkpoint(run_name: str) -> Path:
    """Return the calibrated LLPR checkpoint associated with a run name."""
    matches = [
        path for label, path in DEFAULT_LLPR_CHECKPOINTS.items() if label in run_name
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot determine the LLPR checkpoint for {run_name!r}")
    return matches[0]


def load_llpr_energy_parameters(path: Path) -> dict[str, np.ndarray | float | str]:
    """Load persistent energy members and calibration data from an LLPR checkpoint."""
    import metatomic.torch  # noqa: F401  (registers metadata for torch.load)
    import torch

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LLPR checkpoint not found: {checkpoint}")
    loaded = torch.load(str(checkpoint), weights_only=False, map_location="cpu")
    if loaded.get("architecture_name") != "llpr":
        raise ValueError(f"uncertainty checkpoint is not an LLPR model: {checkpoint}")
    state = loaded.get("model_state_dict")
    wrapped = loaded.get("wrapped_model_checkpoint")
    if not isinstance(state, dict) or not isinstance(wrapped, dict):
        raise ValueError(f"invalid LLPR checkpoint structure: {checkpoint}")
    try:
        weights = state["llpr_ensemble_layers.energy.weight"]
        cholesky = state["cholesky_energy_uncertainty"]
        multiplier = state["multiplier_energy_uncertainty"]
        wrapped_state = wrapped["model_state_dict"]
        node = wrapped_state[
            "backend.node_last_layers.energy.0.energy___0.weight"
        ]
        edge = wrapped_state[
            "backend.edge_last_layers.energy.0.energy___0.weight"
        ]
    except KeyError as error:
        raise ValueError(
            f"LLPR checkpoint lacks PET energy ensemble data: {checkpoint}"
        ) from error
    weights = weights.detach().cpu().numpy().astype(float, copy=False)
    central_weights = np.concatenate(
        [
            node.detach().cpu().numpy().reshape(-1),
            edge.detach().cpu().numpy().reshape(-1),
        ]
    )
    if weights.ndim != 2 or weights.shape[0] < 2:
        raise ValueError("LLPR energy ensemble must contain at least two members")
    if weights.shape[1] != central_weights.size:
        raise ValueError("LLPR and wrapped PET last-layer sizes do not match")
    effective_weights = weights - weights.mean(axis=0) + central_weights
    return {
        "weights": effective_weights,
        "central_weights": central_weights,
        "cholesky": cholesky.detach().cpu().numpy().astype(float, copy=False),
        "multiplier": float(multiplier.detach().cpu().numpy().reshape(-1)[0]),
        "sha256": _model_sha256(checkpoint),
        "path": str(checkpoint),
    }


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


def _use_vesin_neighbor_lists() -> None:
    """Avoid the incompatible CUDA nvalchemi neighbor-list implementation.

    The installed metatomic-ase/nvalchemiops combination computes a float
    ``max_neighbors`` value for PET's 24-angstrom cutoff.  PyTorch rejects that
    value, while using it after an integer cast allocates an impractically large
    dense neighbor matrix.  Vesin is metatomic-ase's supported CUDA fallback and
    keeps the model evaluation itself on the requested GPU.
    """
    import metatomic_ase._neighbors as neighbors

    neighbors.HAS_NVALCHEMIOPS = False


def _selected_thermodynamic_indices(
    thermodynamic_series: dict[str, np.ndarray],
    trajectory_steps: np.ndarray,
    production_mask: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match selected trajectory frames to thermo records by LAMMPS timestep."""
    raw_steps = np.asarray(thermodynamic_series["md_step"])
    dump_steps = np.asarray(trajectory_steps)
    mask = np.asarray(production_mask, dtype=bool)
    if dump_steps.ndim != 1 or len(dump_steps) != len(mask):
        raise ValueError("trajectory steps and production mask must have the same length")

    thermo_index_by_step: dict[int, int] = {}
    for index, raw_step in enumerate(raw_steps):
        step = int(raw_step)
        if step not in thermo_index_by_step:
            thermo_index_by_step[step] = index
    selected_trajectory = np.flatnonzero(mask)[::stride]
    try:
        selected_raw = np.asarray(
            [thermo_index_by_step[int(dump_steps[index])] for index in selected_trajectory],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError(
            "a trajectory timestep is absent from the LAMMPS thermo log"
        ) from error
    return selected_trajectory, selected_raw


def write_committee_archive(path: Path, values: dict) -> None:
    """Replace an NPZ atomically, preserving the previous file on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _cached_committee_rows(
    path: Path, *, context: dict, trajectory_steps: np.ndarray,
    selected_steps: np.ndarray, thermodynamic_rows: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate and align arbitrary cached rows by timestep, independent of stride."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        data = {key: np.asarray(saved[key]) for key in saved.files}

    def incompatible(reason: str) -> None:
        raise ValueError(f"cannot reuse LLPR cache {path}: {reason}; use a separate analysis directory for changed inputs")

    if "model_sha256" in data:
        if str(data["model_sha256"].item()) != context["model_sha256"]:
            incompatible("model checkpoint hash changed")
    elif str(data.get("model_path", "")) != context["model_path"]:
        incompatible("legacy progress model path differs")
    if "inference_model_sha256" in data:
        if str(data["inference_model_sha256"].item()) != context["inference_model_sha256"]:
            incompatible("central export hash changed")
    else:
        # Old archives lack export fingerprints. Check their known source paths
        # and require unchanged source files before promoting them to the new format.
        if context["inference_model_path"] != context["model_path"]:
            summary_path = path.parent / "summary.json"
            if not summary_path.is_file():
                incompatible("legacy LLPR cache needs summary.json to identify its central export")
            summary = json.loads(summary_path.read_text())
            export = summary.get("simulation_provenance", {}).get("exported_model")
            if export is None or str(Path(export).expanduser().resolve()) != context["inference_model_path"]:
                incompatible("legacy central export path differs")
        for source in (context["model_path"], context["inference_model_path"]):
            if Path(source).stat().st_mtime_ns > path.stat().st_mtime_ns:
                incompatible("model or central export was modified after the legacy cache")

    if "trajectory_path" in data and str(data["trajectory_path"].item()) != context["trajectory_path"]:
        incompatible("trajectory path changed")
    if "trajectory_size_bytes" in data and (
        int(data["trajectory_size_bytes"].item()) != context["trajectory_size_bytes"]
        or int(data["trajectory_mtime_ns"].item()) != context["trajectory_mtime_ns"]
    ):
        incompatible("trajectory was modified")
    if "trajectory_size_bytes" not in data:
        if context["trajectory_mtime_ns"] > path.stat().st_mtime_ns:
            incompatible("trajectory was modified after the legacy cache")
        summary_path = path.parent / "summary.json"
        if summary_path.is_file():
            source = json.loads(summary_path.read_text()).get("trajectory")
            if source is not None and str(Path(source).expanduser().resolve()) != context["trajectory_path"]:
                incompatible("legacy trajectory path differs")

    if "md_step" in data:
        steps = np.asarray(data["md_step"], dtype=np.int64)
    else:
        frames = np.asarray(data["frame"], dtype=np.int64)
        if frames.ndim != 1 or np.any(frames < 0) or np.any(frames >= len(trajectory_steps)):
            incompatible("legacy progress frame indices are invalid")
        steps = np.asarray(trajectory_steps, dtype=np.int64)[frames]
    if steps.ndim != 1 or len(np.unique(steps)) != len(steps):
        incompatible("cached timesteps are invalid or duplicated")
    central = np.asarray(data["central_potential_eV"], dtype=float)
    members = np.asarray(data["member_potential_eV"], dtype=float)
    analytical = np.asarray(data["analytical_energy_uncertainty_eV"], dtype=float)
    if (
        central.shape != steps.shape or analytical.shape != steps.shape
        or members.ndim != 2 or members.shape[0] != len(steps) or members.shape[1] < 2
        or not np.all(np.isfinite(central)) or not np.all(np.isfinite(members))
    ):
        incompatible("cached energy arrays are invalid")

    position_by_step = {int(step): index for index, step in enumerate(selected_steps)}
    rows = np.asarray([index for index, step in enumerate(steps) if int(step) in position_by_step], dtype=int)
    positions = np.asarray([position_by_step[int(steps[index])] for index in rows], dtype=int)
    for field, current in thermodynamic_rows.items():
        if field in data:
            values = np.asarray(data[field], dtype=float)
            if values.shape != steps.shape or not np.array_equal(values[rows], current[positions]):
                incompatible(f"saved {field} does not match the current thermo log")
    return positions, central[rows], members[rows], analytical[rows]


def evaluate_trajectory_committee(
    trajectory_path: Path,
    thermodynamic_series: dict[str, np.ndarray],
    trajectory_steps: np.ndarray,
    production_mask: np.ndarray,
    *,
    model_path: Path,
    central_model_path: Path | None = None,
    device: str,
    temperature_K: float,
    pressure_bar: float,
    stride: int,
    batch_size: int,
    checkpoint_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict[str, np.ndarray | str | float]:
    """Reuse completed and interrupted rows, evaluating only missing timesteps."""
    if stride < 1 or batch_size < 1:
        raise ValueError("uncertainty stride and batch size must be positive")
    model = model_path.expanduser().resolve()
    trajectory = trajectory_path.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"uncertainty model not found: {model}")
    selected_trajectory, selected_raw = _selected_thermodynamic_indices(
        thermodynamic_series, trajectory_steps, production_mask, stride,
    )
    if not len(selected_trajectory):
        raise ValueError("no production frames were selected for model uncertainty")
    selected_steps = np.asarray(thermodynamic_series["md_step"])[selected_raw]
    thermodynamic_rows = {
        "time_ps": np.asarray(thermodynamic_series["time_ps"], dtype=float)[selected_raw],
        "logged_central_potential_eV": np.asarray(thermodynamic_series["potential_energy_eV"], dtype=float)[selected_raw],
        "kinetic_energy_eV": np.asarray(thermodynamic_series["kinetic_energy_eV"], dtype=float)[selected_raw],
        "volume_A3": np.asarray(thermodynamic_series["volume_A3"], dtype=float)[selected_raw],
    }
    checkpoint_mode = model.suffix == ".ckpt"
    llpr = load_llpr_energy_parameters(model) if checkpoint_mode else None
    inference_model = (
        central_model_path.expanduser().resolve()
        if checkpoint_mode and central_model_path is not None else model
    )
    if not inference_model.is_file():
        raise FileNotFoundError(f"central exported model not found: {inference_model}")
    trajectory_stat = trajectory.stat()
    context = {
        "model_path": str(model),
        "model_sha256": str(llpr["sha256"]) if checkpoint_mode else _model_sha256(model),
        "inference_model_path": str(inference_model),
        "inference_model_sha256": _model_sha256(inference_model),
        "trajectory_path": str(trajectory),
        "trajectory_size_bytes": trajectory_stat.st_size,
        "trajectory_mtime_ns": trajectory_stat.st_mtime_ns,
    }
    count = len(selected_trajectory)
    central = np.full(count, np.nan)
    analytical = np.full(count, np.nan)
    members = None
    completed = np.zeros(count, dtype=bool)
    for path in (cache_path, checkpoint_path):
        if path is None:
            continue
        cached = _cached_committee_rows(
            path, context=context, trajectory_steps=trajectory_steps,
            selected_steps=selected_steps, thermodynamic_rows=thermodynamic_rows,
        )
        if cached is None:
            continue
        positions, cached_central, cached_members, cached_analytical = cached
        if members is None:
            members = np.full((count, cached_members.shape[1]), np.nan)
        if members.shape[1] != cached_members.shape[1] or (
            checkpoint_mode and cached_members.shape[1] != len(llpr["weights"])
        ):
            raise ValueError(f"cached LLPR member count differs: {path}")
        overlap = completed[positions]
        if np.any(overlap) and (
            not np.array_equal(central[positions[overlap]], cached_central[overlap])
            or not np.array_equal(members[positions[overlap]], cached_members[overlap])
            or not np.array_equal(analytical[positions[overlap]], cached_analytical[overlap], equal_nan=True)
        ):
            raise ValueError(f"completed and progress LLPR caches disagree: {path}")
        central[positions] = cached_central
        members[positions] = cached_members
        analytical[positions] = cached_analytical
        completed[positions] = True
    reused_count = int(completed.sum())
    print(f"LLPR cache: reusing {reused_count} frames; evaluating {count - reused_count} missing frames", flush=True)

    def persist_progress() -> None:
        if checkpoint_path is None or not np.any(completed):
            return
        rows = np.flatnonzero(completed)
        write_committee_archive(checkpoint_path, {
            **context,
            "frame": selected_trajectory[rows], "md_step": selected_steps[rows],
            "central_potential_eV": central[rows], "member_potential_eV": members[rows],
            "analytical_energy_uncertainty_eV": analytical[rows],
            **{field: values[rows] for field, values in thermodynamic_rows.items()},
            "stride": stride,
        })

    if not np.all(completed):
        from ase.io import iread
        import metatomic.torch as metatomic_torch
        from metatomic.torch import ModelOutput
        from metatomic_ase import MetatomicCalculator

        _use_vesin_neighbor_lists()
        loaded_model = metatomic_torch.load_atomistic_model(str(inference_model))
        capabilities = set(loaded_model.capabilities().outputs)
        required = {"energy", "mtt::aux::energy_last_layer_features"} if checkpoint_mode else {"energy", "energy_ensemble"}
        missing = sorted(required.difference(capabilities))
        if missing:
            raise ValueError("uncertainty inference model is missing output(s): " + ", ".join(missing))
        has_analytical_uncertainty = checkpoint_mode or "energy_uncertainty" in capabilities
        del loaded_model
        calculator = MetatomicCalculator(str(inference_model), device=device)
        requested = {"energy": ModelOutput(sample_kind="system")}
        if checkpoint_mode:
            requested["mtt::aux::energy_last_layer_features"] = ModelOutput(sample_kind="system")
        else:
            requested["energy_ensemble"] = ModelOutput(sample_kind="system")
            if has_analytical_uncertainty:
                requested["energy_uncertainty"] = ModelOutput(sample_kind="system")
        positions_by_frame = {int(frame): position for position, frame in enumerate(selected_trajectory)}
        structures = []
        batch_positions: list[int] = []

        def evaluate_batch() -> None:
            nonlocal members
            if not structures:
                return
            try:
                outputs = calculator.run_model(structures, requested)
            except Exception as error:
                raise RuntimeError("the configured model could not evaluate the LLPR energy ensemble") from error
            batch_count = len(structures)
            values = _system_values(outputs["energy"], batch_count, "energy")
            if checkpoint_mode:
                features = outputs["mtt::aux::energy_last_layer_features"][0].values
                features = features.detach().cpu().numpy().reshape(batch_count, -1)
                predicted_members = values[:, None] + features @ (np.asarray(llpr["weights"]) - np.asarray(llpr["central_weights"])).T
                solved = np.linalg.solve(np.asarray(llpr["cholesky"]), features.T)
                predicted_uncertainty = np.sqrt(np.sum(solved**2, axis=0)) * float(llpr["multiplier"])
            else:
                predicted_members = _ensemble_values(outputs["energy_ensemble"], batch_count)
                predicted_uncertainty = (
                    _system_values(outputs["energy_uncertainty"], batch_count, "energy_uncertainty")
                    if has_analytical_uncertainty else np.full(batch_count, np.nan)
                )
            if members is None:
                members = np.full((count, predicted_members.shape[1]), np.nan)
            if members.shape[1] != predicted_members.shape[1]:
                raise ValueError("cached and evaluated LLPR member counts differ")
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(predicted_members)):
                raise ValueError("non-finite LLPR predictions; completed cache remains available")
            central[batch_positions] = values
            members[batch_positions] = predicted_members
            analytical[batch_positions] = predicted_uncertainty
            completed[batch_positions] = True
            structures.clear()
            batch_positions.clear()
            persist_progress()

        for raw_index, atoms in enumerate(iread(str(trajectory), index=":")):
            position = positions_by_frame.get(raw_index)
            if position is None or completed[position]:
                continue
            structures.append(atoms)
            batch_positions.append(position)
            if len(structures) == batch_size:
                evaluate_batch()
        evaluate_batch()
    if not np.all(completed):
        raise ValueError("trajectory ended before all missing uncertainty frames were evaluated")
    if trajectory.stat().st_size != trajectory_stat.st_size or trajectory.stat().st_mtime_ns != trajectory_stat.st_mtime_ns:
        raise ValueError("trajectory changed during LLPR evaluation")

    estimates = committee_enthalpy_estimates(
        thermodynamic_rows["logged_central_potential_eV"], members,
        thermodynamic_rows["kinetic_energy_eV"], thermodynamic_rows["volume_A3"],
        temperature_K=temperature_K, pressure_bar=pressure_bar,
    )
    ensemble_mean_residual = members.mean(axis=1) - central
    logged_energy_residual = central - thermodynamic_rows["logged_central_potential_eV"]
    logged_energy_residual -= logged_energy_residual.mean()
    sampling_model_residual = members.mean(axis=1) - thermodynamic_rows["logged_central_potential_eV"]
    sampling_model_residual -= sampling_model_residual.mean()
    return {
        **context, **thermodynamic_rows, **estimates,
        "frame": selected_trajectory.astype(np.int64), "md_step": selected_steps,
        "frame_stride": stride,
        "production_frame_count": int(np.count_nonzero(production_mask)),
        "central_potential_eV": central, "member_potential_eV": members,
        "analytical_energy_uncertainty_eV": analytical,
        "reused_frame_count": reused_count, "evaluated_frame_count": count - reused_count,
        "ensemble_mean_residual_rms_eV": float(np.sqrt(np.mean(ensemble_mean_residual**2))),
        "logged_energy_residual_rms_eV": float(np.sqrt(np.mean(logged_energy_residual**2))),
        "logged_energy_residual_max_abs_eV": float(np.max(np.abs(logged_energy_residual))),
        "sampling_model_residual_rms_eV": float(np.sqrt(np.mean(sampling_model_residual**2))),
        "sampling_model_residual_max_abs_eV": float(np.max(np.abs(sampling_model_residual))),
    }


def committee_standard_deviation(values: np.ndarray) -> np.ndarray:
    """Return the sample standard deviation over committee members."""
    array = np.asarray(values, dtype=float)
    if array.shape[0] < 2:
        raise ValueError("at least two committee members are required")
    return array.std(axis=0, ddof=1)
