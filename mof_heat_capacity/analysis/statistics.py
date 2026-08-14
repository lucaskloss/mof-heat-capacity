"""Statistical convergence tools for correlated molecular-dynamics series."""

from __future__ import annotations

import math

import numpy as np


KB_EV_PER_K = 8.617333262145e-5
EV_TO_J = 1.602176634e-19
AMU_TO_G = 1.66053906660e-24


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Return the normalized autocorrelation using an FFT estimator."""
    data = np.asarray(values, dtype=float).reshape(-1)
    if len(data) == 0:
        raise ValueError("cannot correlate an empty series")
    centered = data - data.mean()
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= np.finfo(float).eps * max(1.0, float(data.mean() ** 2)):
        size = len(data) if max_lag is None else min(len(data), max_lag + 1)
        result = np.zeros(size, dtype=float)
        result[0] = 1.0
        return result

    fft_size = 1 << (2 * len(centered) - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    covariance = np.fft.irfft(transform * transform.conjugate(), n=fft_size)[
        : len(centered)
    ]
    covariance /= np.arange(len(centered), 0, -1)
    result = covariance / covariance[0]
    if max_lag is not None:
        result = result[: max_lag + 1]
    return result


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate tau_int in sampled frames using Geyer's positive sequence."""
    correlation = autocorrelation(values)
    tau = 0.5
    lag = 1
    while lag < len(correlation):
        pair_sum = correlation[lag]
        if lag + 1 < len(correlation):
            pair_sum += correlation[lag + 1]
        if not np.isfinite(pair_sum) or pair_sum <= 0.0:
            break
        tau += float(pair_sum)
        lag += 2
    return max(0.5, tau)


def running_mean(values: np.ndarray) -> np.ndarray:
    """Return the cumulative mean of a one-dimensional series."""
    data = np.asarray(values, dtype=float).reshape(-1)
    return np.cumsum(data) / np.arange(1, len(data) + 1)


def block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    """Return means of complete, non-overlapping blocks."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    data = np.asarray(values, dtype=float).reshape(-1)
    count = len(data) // block_size
    if count == 0:
        return np.empty(0, dtype=float)
    return data[: count * block_size].reshape(count, block_size).mean(axis=1)


def _correlated_standard_error(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float).reshape(-1)
    if len(data) < 2:
        return 0.0, 0.5, float(len(data))
    standard_deviation = float(data.std(ddof=1))
    if standard_deviation == 0.0:
        return 0.0, 0.5, float(len(data))
    tau = integrated_autocorrelation_time(data)
    effective_samples = min(float(len(data)), len(data) / (2.0 * tau))
    return standard_deviation / math.sqrt(effective_samples), tau, effective_samples


def summarize_series(
    values: np.ndarray,
    times_ps: np.ndarray,
    *,
    block_size: int | None = None,
) -> dict[str, float | int]:
    """Summarize stationarity and uncertainty for one production series."""
    data = np.asarray(values, dtype=float).reshape(-1)
    times = np.asarray(times_ps, dtype=float).reshape(-1)
    if len(data) != len(times) or len(data) < 2:
        raise ValueError("values and times must have the same length of at least two")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(times)):
        raise ValueError("series contains non-finite values")

    standard_error, tau, effective_samples = _correlated_standard_error(data)
    spacing_ps = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    active_block_size = block_size or max(1, int(math.ceil(2.0 * tau)))
    blocks = block_means(data, active_block_size)
    block_error = (
        float(blocks.std(ddof=1) / math.sqrt(len(blocks)))
        if len(blocks) > 1
        else float("nan")
    )

    midpoint = len(data) // 2
    first = data[:midpoint]
    second = data[midpoint:]
    first_error, _, _ = _correlated_standard_error(first)
    second_error, _, _ = _correlated_standard_error(second)
    split_difference = float(second.mean() - first.mean())
    split_error = math.sqrt(first_error**2 + second_error**2)
    split_z = split_difference / split_error if split_error > 0.0 else 0.0
    drift = (
        float(np.polyfit(times, data, 1)[0])
        if np.ptp(times) > 0.0
        else float("nan")
    )
    mean = float(data.mean())
    return {
        "samples": len(data),
        "mean": mean,
        "standard_deviation": float(data.std(ddof=1)),
        "standard_error": standard_error,
        "relative_standard_error": (
            abs(standard_error / mean) if mean != 0.0 else float("nan")
        ),
        "integrated_autocorrelation_time_frames": tau,
        "integrated_autocorrelation_time_ps": tau * spacing_ps,
        "effective_samples": effective_samples,
        "block_size_frames": active_block_size,
        "complete_blocks": len(blocks),
        "block_standard_error": block_error,
        "first_half_mean": float(first.mean()),
        "second_half_mean": float(second.mean()),
        "second_minus_first_half": split_difference,
        "split_stationarity_z": split_z,
        "linear_drift_per_ps": drift,
    }


def classical_fluctuation_heat_capacity(
    energies_eV: np.ndarray,
    temperature_K: float,
    total_mass_amu: float,
    *,
    block_size: int,
) -> dict[str, float | int]:
    """Estimate classical NVT C_V from total-energy fluctuations."""
    energies = np.asarray(energies_eV, dtype=float).reshape(-1)
    if len(energies) < 2 or temperature_K <= 0.0 or total_mass_amu <= 0.0:
        raise ValueError("heat-capacity inputs must be positive and non-empty")

    conversion = EV_TO_J / (total_mass_amu * AMU_TO_G)

    def estimate(block: np.ndarray) -> float:
        variance = float(np.var(block, ddof=1))
        return variance / (KB_EV_PER_K * temperature_K**2) * conversion

    value = estimate(energies)
    count = len(energies) // block_size
    block_values = np.asarray(
        [
            estimate(block)
            for block in energies[: count * block_size].reshape(count, block_size)
            if len(block) > 1
        ],
        dtype=float,
    )
    block_error = (
        float(block_values.std(ddof=1) / math.sqrt(len(block_values)))
        if len(block_values) > 1
        else float("nan")
    )
    return {
        "cv_J_per_gK": value,
        "block_standard_error_J_per_gK": block_error,
        "block_size_frames": block_size,
        "complete_blocks": len(block_values),
    }
