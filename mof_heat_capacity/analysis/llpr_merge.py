"""Merge independently computed LLPR member batches into a harmonic archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read_archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def merge_archives(central_path: Path, batch_paths: list[Path], output: Path,
                   *, overwrite: bool = False) -> Path:
    """Validate member coverage and pool matrix moments without retaining all Hessians."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"Merge output exists: {output}; use --overwrite")
    central = read_archive(central_path)
    central_hash = hashlib.sha256(central_path.read_bytes()).hexdigest()
    batches = [read_archive(path) for path in batch_paths]
    if not batches:
        raise ValueError("No LLPR batches supplied")
    expected = int(batches[0]["llpr_total_member_count"])
    if expected != 64:
        raise ValueError(f"Expected 64 LLPR members, got {expected}")
    identity_keys = (
        "run_name", "trajectory", "trajectory_sha256", "checkpoint", "metadata",
        "frame_indices", "frame_times_ps", "temperatures_K", "frequencies_cm1", "cv_J_per_gK",
        "central_hessians_eV_per_A2",
    )
    llpr_keys = ("llpr_checkpoint_sha256", "llpr_checkpoint", "llpr_total_member_count")
    ids = []
    for path, batch in zip(batch_paths, batches, strict=True):
        if str(batch["central_archive_sha256"].item()) != central_hash:
            raise ValueError(f"Batch uses a different central archive: {path}")
        # The central job has no LLPR metadata, so compare numerical settings.
        for key in identity_keys:
            if key == "metadata":
                a, b = json.loads(str(batch[key].item())), json.loads(str(central[key].item()))
                a.pop("llpr_method", None)
                b.pop("llpr_method", None)
                equal = a == b
            else:
                equal = np.array_equal(batch[key], central[key])
            if not equal:
                raise ValueError(f"Batch {key} differs from central archive: {path}")
        for key in llpr_keys:
            if not np.array_equal(batch[key], batches[0][key]):
                raise ValueError(f"Batch {key} differs: {path}")
        member_ids = batch["llpr_member_indices"]
        count = int(batch["llpr_member_count"])
        if member_ids.ndim != 1 or len(member_ids) != count or count != 8:
            raise ValueError(f"Expected eight indexed members in batch: {path}")
        for key in ("llpr_frequencies_cm1", "llpr_cv_J_per_gK", "llpr_hessian_member_rms_deviation_eV_per_A2"):
            if batch[key].shape[1] != count or not np.isfinite(batch[key]).all():
                raise ValueError(f"Invalid batch {key}: {path}")
        ids.extend(member_ids.tolist())
    if sorted(ids) != list(range(expected)):
        raise ValueError("LLPR batches must cover members 0..63 exactly once")
    order = np.argsort(ids)
    result = dict(central)
    for key in ("llpr_frequencies_cm1", "llpr_cv_J_per_gK", "llpr_hessian_member_rms_deviation_eV_per_A2"):
        result[key] = np.concatenate([batch[key] for batch in batches], axis=1)[:, order]
    result["llpr_frequency_standard_deviation_cm1"] = result["llpr_frequencies_cm1"].std(axis=1, ddof=1)
    result["llpr_cv_standard_deviation_J_per_gK"] = result["llpr_cv_J_per_gK"].std(axis=1, ddof=1)
    count = 0
    mean = np.zeros_like(batches[0]["llpr_hessian_member_mean_eV_per_A2"])
    m2 = np.zeros_like(mean)
    for batch in batches:
        n = int(batch["llpr_member_count"])
        batch_mean = batch["llpr_hessian_member_mean_eV_per_A2"]
        batch_m2 = batch["llpr_hessian_member_m2_eV2_per_A4"]
        if batch_mean.shape != mean.shape or batch_m2.shape != mean.shape:
            raise ValueError("LLPR matrix moment shapes differ")
        if not np.isfinite(batch_mean).all() or not np.isfinite(batch_m2).all():
            raise ValueError("Nonfinite LLPR matrix moments")
        delta = batch_mean - mean
        m2 += batch_m2 + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    result["llpr_hessian_member_mean_eV_per_A2"] = mean
    result["llpr_hessian_member_m2_eV2_per_A4"] = m2
    result["llpr_hessian_rms_standard_deviation_eV_per_A2"] = np.sqrt(np.mean(m2 / (count - 1), axis=(1, 2)))
    for key in llpr_keys:
        result[key] = batches[0][key]
    result["llpr_member_count"] = expected
    result["llpr_member_indices"] = np.arange(expected)
    result["central_archive_sha256"] = central_hash
    result["metadata"] = batches[0]["metadata"]
    result["llpr_batch_archives"] = np.array([str(path.resolve()) for path in batch_paths])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(temporary, **result)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    batches = [args.output.with_name(f"{args.output.stem}.llpr-{i}.npz") for i in range(8)]
    print(f"Saved merged LLPR archive: {merge_archives(args.central_archive, batches, args.output, overwrite=args.overwrite)}")


if __name__ == "__main__":
    main()
