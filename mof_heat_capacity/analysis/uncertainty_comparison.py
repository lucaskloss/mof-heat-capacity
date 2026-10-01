"""Compare refreshed CEA and Gaussian NPT spread with common sampling errors."""

import hashlib
from pathlib import Path

import numpy as np


def archive_sha256(path: Path) -> str:
    """Fingerprint a saved estimator to detect stale downstream comparisons."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_indices(grid: np.ndarray, temperatures: np.ndarray, path: Path) -> list[int]:
    if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError(f"invalid temperature grid in {path}")
    indices = []
    for temperature in temperatures:
        matches = np.flatnonzero(np.isclose(grid, temperature, rtol=0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"temperature {temperature:g} K missing or duplicated in {path}")
        indices.append(int(matches[0]))
    return indices


def _paired_spread(classical: np.ndarray, harmonic: np.ndarray, path: Path) -> np.ndarray:
    if classical.ndim != 2 or classical.shape != harmonic.shape or classical.shape[0] < 2:
        raise ValueError(f"classical and harmonic member arrays do not align: {path}")
    if not np.all(np.isfinite(classical)) or not np.all(np.isfinite(harmonic)):
        raise ValueError(f"non-finite member curves: {path}")
    return (
        classical - classical.mean(axis=0) + harmonic - harmonic.mean(axis=0)
    ).std(axis=0, ddof=1)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"uncertainty archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def hybrid_uncertainty_components(
    hybrid_path: Path, *, method: str = "cea", variance_path: Path | None = None,
    ensemble_path: Path | None = None, require_all_frames: bool = False,
) -> dict[str, np.ndarray]:
    """Keep the central curve and MD sampling error; select model spread.

    DPOSE here means the paper's Gaussian NPT centered-moment approximation.
    Classical and harmonic member deviations are paired before computing the
    hybrid model spread. The MD sampling error is combined in quadrature with
    the paired model spread.
    """
    if method not in {"cea", "dpose"}:
        raise ValueError("uncertainty method must be 'cea' or 'dpose'")

    hybrid = _load_arrays(hybrid_path)
    temperatures = np.asarray(hybrid["temperatures_K"], dtype=float)
    if (
        temperatures.ndim != 1
        or not temperatures.size
        or not np.all(np.isfinite(temperatures))
        or np.any(np.diff(temperatures) <= 0)
    ):
        raise ValueError(f"invalid temperature grid in {hybrid_path}")

    fields = {
        "central_hybrid_cp_J_per_gK": "approximate_cp_J_per_gK",
        "md_sampling_standard_error_J_per_gK": "classical_anharmonic_cp_standard_error_J_per_gK",
        "classical_model_standard_deviation_J_per_gK": "classical_anharmonic_cp_model_standard_deviation_J_per_gK",
        "harmonic_model_standard_deviation_J_per_gK": "harmonic_quantum_correction_model_standard_deviation_J_per_gK",
        "hybrid_model_standard_deviation_J_per_gK": "approximate_cp_model_standard_deviation_J_per_gK",
    }
    result = {"temperature_K": temperatures}
    for name, source in fields.items():
        value = np.asarray(hybrid[source], dtype=float)
        if value.shape != temperatures.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid {source} in {hybrid_path}")
        if name != "central_hybrid_cp_J_per_gK" and np.any(value < 0):
            raise ValueError(f"negative uncertainty in {source}: {hybrid_path}")
        result[name] = value

    ensemble = _load_arrays(ensemble_path) if ensemble_path is not None else None
    if require_all_frames and ensemble is None:
        raise ValueError("an all-frame plot requires the refreshed CEA ensemble archive")
    if ensemble is not None:
        if str(hybrid["llpr_checkpoint_sha256"].item()) != str(ensemble["model_sha256"].item()):
            raise ValueError(f"CEA and Hessian LLPR checkpoint mismatch: {ensemble_path}")
        cea_grid = np.asarray(ensemble["temperatures_K"], dtype=float)
        cea_indices = _grid_indices(cea_grid, temperatures, ensemble_path)
        cea_members = np.asarray(ensemble["cea_cp_by_member_J_per_gK"], dtype=float)
        if cea_members.ndim != 2 or cea_members.shape[1] != cea_grid.size:
            raise ValueError(f"invalid CEA member curves in {ensemble_path}")
        cea_members = cea_members[:, cea_indices]
        harmonic = np.asarray(hybrid["harmonic_quantum_correction_by_member_J_per_gK"], dtype=float)
        cea_hybrid_sd = _paired_spread(cea_members, harmonic, ensemble_path)
        if method == "cea":
            result["classical_model_standard_deviation_J_per_gK"] = cea_members.std(axis=0, ddof=1)
            result["hybrid_model_standard_deviation_J_per_gK"] = cea_hybrid_sd

        for field in ("selected_frame_count", "production_frame_count", "replica_counts"):
            values = np.asarray(ensemble.get(field, np.full(cea_grid.size, -1)), dtype=int)
            if values.shape != cea_grid.shape:
                raise ValueError(f"invalid {field} in {ensemble_path}")
            result[field] = values[cea_indices]
        stride = int(ensemble["frame_stride"].item()) if "frame_stride" in ensemble else -1
        result["frame_stride"] = np.full(temperatures.size, stride, dtype=int)
        if require_all_frames and (
            stride != 1 or np.any(result["selected_frame_count"] < 1)
            or not np.array_equal(result["selected_frame_count"], result["production_frame_count"])
        ):
            raise ValueError("CEA archive does not cover all saved production frames; rerun stride-1 analysis and aggregation")

    if method == "dpose":
        if variance_path is None:
            raise ValueError("DPOSE comparison requires a variance archive")
        variance = _load_arrays(variance_path)
        if str(hybrid["llpr_checkpoint_sha256"].item()) != str(variance["model_sha256"].item()):
            raise ValueError(f"LLPR checkpoint mismatch: {hybrid_path} and {variance_path}")

        full_grid = np.asarray(variance["temperatures_K"], dtype=float)
        indices = _grid_indices(full_grid, temperatures, variance_path)
        if ensemble is not None:
            if "cea_source_sha256" in variance:
                if str(variance["cea_source_sha256"].item()) != archive_sha256(ensemble_path):
                    raise ValueError("DPOSE archive is stale; rerun compare_heat_capacity_estimators.py after CEA aggregation")
            elif require_all_frames:
                raise ValueError("DPOSE archive lacks all-frame provenance; rerun compare_heat_capacity_estimators.py")
            if require_all_frames and not {"frame_stride", "production_frame_count", "selected_frame_count"}.issubset(variance):
                raise ValueError("DPOSE archive lacks frame coverage metadata; rerun compare_heat_capacity_estimators.py")
            copied_cea = np.asarray(variance["cea_cp_by_member_J_per_gK"], dtype=float)[:, indices]
            if not np.array_equal(copied_cea, cea_members):
                raise ValueError("DPOSE archive uses different CEA data; rerun compare_heat_capacity_estimators.py")
            if "selected_frame_count" in ensemble and not np.array_equal(
                np.asarray(variance["selected_frame_count"], dtype=int)[indices], result["selected_frame_count"],
            ):
                raise ValueError("CEA and DPOSE archives use different frame counts or replicas")
            if require_all_frames and (
                int(variance["frame_stride"].item()) != 1
                or not np.array_equal(
                    np.asarray(variance["production_frame_count"], dtype=int)[indices], result["production_frame_count"],
                )
            ):
                raise ValueError("DPOSE archive does not cover the same saved production frames as CEA")

        classical = np.asarray(variance["gaussian_npt_cp_by_member_J_per_gK"], dtype=float)
        harmonic = np.asarray(hybrid["harmonic_quantum_correction_by_member_J_per_gK"], dtype=float)
        if classical.ndim != 2 or classical.shape[1] != full_grid.size:
            raise ValueError(f"invalid classical member curves in {variance_path}")
        classical = classical[:, indices]
        result["classical_model_standard_deviation_J_per_gK"] = classical.std(axis=0, ddof=1)
        result["hybrid_model_standard_deviation_J_per_gK"] = _paired_spread(classical, harmonic, variance_path)

    result["combined_standard_uncertainty_J_per_gK"] = np.hypot(
        result["md_sampling_standard_error_J_per_gK"],
        result["hybrid_model_standard_deviation_J_per_gK"],
    )
    return result
