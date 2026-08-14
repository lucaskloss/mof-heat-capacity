"""Collect harmonic heat-capacity results and quantify frame convergence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "frame_indices",
    "temperatures_K",
    "frequencies_cm1",
    "cv_J_per_gK",
}


def _load_archive(path: Path) -> list[dict]:
    """Load frame-level records from one harmonic heat-capacity archive."""
    with np.load(path, allow_pickle=False) as data:
        if not REQUIRED_KEYS.issubset(data.files):
            return []
        frames = np.asarray(data["frame_indices"], dtype=int).reshape(-1)
        temperatures = np.asarray(data["temperatures_K"], dtype=float).reshape(-1)
        frequencies = np.asarray(data["frequencies_cm1"], dtype=float)
        heat_capacity = np.asarray(data["cv_J_per_gK"], dtype=float)
        archive_metadata = {}
        if "metadata" in data.files:
            archive_metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        archive_identity = {
            key: str(np.asarray(data[key]).item())
            for key in ("run_name", "structure", "trajectory", "checkpoint")
            if key in data.files
        }

    if frequencies.ndim == 1:
        frequencies = frequencies[None, :]
    if heat_capacity.ndim == 1:
        heat_capacity = heat_capacity[None, :]
    expected = (len(frames), len(temperatures))
    if heat_capacity.shape != expected or frequencies.shape[0] != len(frames):
        raise ValueError(
            f"inconsistent harmonic arrays in {path}: frames={len(frames)}, "
            f"temperatures={len(temperatures)}, Cv={heat_capacity.shape}, "
            f"frequencies={frequencies.shape}"
        )

    return [
        {
            "frame": int(frame),
            "source": str(path),
            "temperatures_K": temperatures.copy(),
            "cv_J_per_gK": heat_capacity[index].copy(),
            "frequencies_cm1": frequencies[index].copy(),
            "archive_metadata": archive_metadata,
            "archive_identity": archive_identity,
        }
        for index, frame in enumerate(frames)
    ]


def collect_heat_capacity_results(
    run_directory: Path,
    *,
    target_temperature_K: float,
    imaginary_threshold_cm1: float = -1.0,
    zero_mode_threshold_cm1: float = 1.0,
) -> dict:
    """Collect all harmonic archives in a run and summarize frame convergence."""
    records: list[dict] = []
    ignored: list[str] = []
    for path in sorted(run_directory.glob("*.npz")):
        loaded = _load_archive(path)
        if loaded:
            records.extend(loaded)
        else:
            ignored.append(str(path))

    # One physical frame should contribute once even if archives overlap. Keep the
    # first result, but fail when duplicate values disagree instead of averaging them.
    unique: dict[int, dict] = {}
    duplicate_sources: dict[int, list[str]] = {}
    for record in records:
        frame = record["frame"]
        if frame not in unique:
            unique[frame] = record
            continue
        previous = unique[frame]
        same_grid = np.array_equal(
            previous["temperatures_K"], record["temperatures_K"]
        )
        same_values = same_grid and np.allclose(
            previous["cv_J_per_gK"], record["cv_J_per_gK"], equal_nan=True
        )
        if not same_values:
            raise ValueError(
                f"conflicting harmonic heat capacities for frame {frame}: "
                f"{previous['source']} and {record['source']}"
            )
        duplicate_sources.setdefault(frame, []).append(record["source"])

    ordered = [unique[frame] for frame in sorted(unique)]
    frame_rows = []
    target_values = []
    for record in ordered:
        temperatures = record["temperatures_K"]
        if not temperatures[0] <= target_temperature_K <= temperatures[-1]:
            raise ValueError(
                f"target temperature {target_temperature_K:g} K lies outside "
                f"{temperatures[0]:g}..{temperatures[-1]:g} K in {record['source']}"
            )
        target_cv = float(
            np.interp(target_temperature_K, temperatures, record["cv_J_per_gK"])
        )
        frequencies = record["frequencies_cm1"]
        target_values.append(target_cv)
        frame_rows.append(
            {
                "frame": record["frame"],
                "target_temperature_K": target_temperature_K,
                "cv_J_per_gK": target_cv,
                "running_mean_cv_J_per_gK": float(np.mean(target_values)),
                "imaginary_modes_below_threshold": int(
                    np.count_nonzero(frequencies < imaginary_threshold_cm1)
                ),
                "near_zero_modes": int(
                    np.count_nonzero(np.abs(frequencies) <= zero_mode_threshold_cm1)
                ),
                "minimum_frequency_cm1": float(np.min(frequencies)),
                "maximum_frequency_cm1": float(np.max(frequencies)),
                "source": record["source"],
            }
        )

    if ordered:
        reference_grid = ordered[0]["temperatures_K"]
        common_grid = all(
            np.array_equal(item["temperatures_K"], reference_grid)
            for item in ordered[1:]
        )
        if common_grid:
            curves = np.stack([item["cv_J_per_gK"] for item in ordered])
            mean_curve = curves.mean(axis=0)
            curve_standard_error = (
                curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
                if len(curves) > 1
                else np.full(len(reference_grid), np.nan)
            )
        else:
            reference_grid = np.empty(0)
            curves = np.empty((0, 0))
            mean_curve = np.empty(0)
            curve_standard_error = np.empty(0)
    else:
        reference_grid = np.empty(0)
        curves = np.empty((0, 0))
        mean_curve = np.empty(0)
        curve_standard_error = np.empty(0)

    values = np.asarray(target_values)
    archive_summaries = []
    seen_sources = set()
    for item in ordered:
        if item["source"] in seen_sources:
            continue
        seen_sources.add(item["source"])
        archive_summaries.append(
            {
                "source": item["source"],
                **item["archive_identity"],
                **item["archive_metadata"],
            }
        )
    summary = {
        "available": bool(ordered),
        "frames_analyzed": len(ordered),
        "frame_indices": [item["frame"] for item in ordered],
        "target_temperature_K": target_temperature_K,
        "mean_cv_J_per_gK": float(values.mean()) if len(values) else None,
        "frame_standard_deviation_J_per_gK": (
            float(values.std(ddof=1)) if len(values) > 1 else None
        ),
        "frame_standard_error_J_per_gK": (
            float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else None
        ),
        "frame_convergence_assessable": len(ordered) >= 3,
        "ignored_npz_files": ignored,
        "duplicate_sources": duplicate_sources,
        "archives": archive_summaries,
    }
    return {
        "summary": summary,
        "frame_rows": frame_rows,
        "temperatures_K": reference_grid,
        "frame_curves_J_per_gK": curves,
        "mean_curve_J_per_gK": mean_curve,
        "curve_standard_error_J_per_gK": curve_standard_error,
    }
