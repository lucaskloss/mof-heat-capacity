"""Compute harmonic heat capacity from selected ASE trajectory frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.io import read
from sadmof.models.pet import load_pet
from sadmof.observables import cv_from_hessian
from sadmof.models.pet import atoms_to_inputs, get_energy_fn
from sadmof.sparse import get_hessian_fn, sparsity_pattern
from petjax.convert import convert_checkpoint
import asdex
import jax
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = SCRIPT_DIR / "output" / "long-run" / "mof5-ase-100.traj"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.ckpt"
DEFAULT_JAX_CHECKPOINT = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_jax"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--jax-checkpoint", type=Path, default=DEFAULT_JAX_CHECKPOINT)
    parser.add_argument("--frames", default="1", help="count, 'all', or comma-separated indices")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--temperatures", default="100:500:10", help="start:stop:step in K")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "output" / "heat-capacity.npz")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--hops", type=int, default=7)
    parser.add_argument("--remat", action="store_true")
    parser.add_argument("--shadow", action="store_true", help="retain PET adaptive-cutoff shadow derivatives")
    return parser.parse_args()


def select_indices(spec: str, n_frames: int, stride: int | None) -> list[int]:
    """Select trajectory indices from a count, list, or ``all``."""
    if stride is not None:
        if stride < 1:
            raise ValueError("stride must be positive")
        return list(range(0, n_frames, stride))
    if spec == "all":
        return list(range(n_frames))
    if "," in spec:
        indices = [int(item.strip()) for item in spec.split(",")]
    else:
        count = int(spec)
        if count < 1:
            raise ValueError("frames must be positive")
        indices = np.linspace(0, n_frames - 1, min(count, n_frames), dtype=int).tolist()
    if any(index < 0 or index >= n_frames for index in indices):
        raise IndexError(f"frame index outside 0..{n_frames - 1}")
    return list(dict.fromkeys(indices))


def parse_temperatures(spec: str) -> np.ndarray:
    """Parse an inclusive ``start:stop:step`` temperature range."""
    start, stop, step = (float(value) for value in spec.split(":"))
    if step <= 0.0 or stop < start:
        raise ValueError("temperatures must be start:stop:positive_step")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def ensure_jax_checkpoint(checkpoint: Path, jax_checkpoint: Path) -> Path:
    """Convert a local PET checkpoint when a PET-JAX directory is absent."""
    if (checkpoint / "model.msgpack").is_file():
        return checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if (jax_checkpoint / "model.msgpack").is_file():
        return jax_checkpoint
    jax_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting PET checkpoint to PET-JAX: {jax_checkpoint}")
    convert_checkpoint(str(checkpoint), str(jax_checkpoint))
    return jax_checkpoint


def compute_frame_hessian(atoms, model, params, metadata, *, args, jax, np):
    """Compute one sparse PET Hessian and return its real-atom dense block."""
    positions, cell, graph = atoms_to_inputs(
        atoms,
        model,
        metadata,
        float_dtype=np.float64 if args.dtype == "float64" else np.float32,
    )
    pattern_graph = {
        "centers": graph["sel_centers"],
        "others": graph["sel_others"],
        "atomic_numbers": graph["atomic_numbers"],
    }
    pattern = sparsity_pattern(pattern_graph, hops=args.hops)
    coloring = asdex.hessian_coloring_from_sparsity(pattern, mode="fwd_over_rev")
    energy_fn = get_energy_fn(model, metadata, no_shadow=not args.shadow)
    hessian_fn = jax.jit(
        get_hessian_fn(energy_fn, coloring, chunk_size=args.chunk_size, remat=args.remat)
    )
    hessian = jax.block_until_ready(hessian_fn(params, positions, cell, graph))
    hessian = np.asarray(hessian.todense())
    n_atoms = len(atoms)
    return hessian[:n_atoms, :, :n_atoms, :].reshape(3 * n_atoms, 3 * n_atoms)


def main() -> None:
    """Load trajectory frames, calculate Hessians, and save harmonic C_v."""
    args = parse_args()
    trajectory_path = args.trajectory.expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    frames = read(str(trajectory_path), index=":")
    indices = select_indices(args.frames, len(frames), args.stride)
    temperatures = parse_temperatures(args.temperatures)
    jax_checkpoint = ensure_jax_checkpoint(
        args.checkpoint.expanduser().resolve(), args.jax_checkpoint.expanduser().resolve()
    )
    model, params, metadata = load_pet(jax_checkpoint, dtype=args.dtype)

    frequencies = []
    heat_capacities = []
    for index in indices:
        atoms = frames[index].copy()
        atoms.calc = None
        print(f"Computing Hessian for frame {index} ({len(atoms)} atoms)")
        hessian = compute_frame_hessian(atoms, model, params, metadata, args=args, jax=jax, np=np)
        freqs, cv = cv_from_hessian(hessian, atoms.get_masses(), temperatures, enforce_asr=True)
        frequencies.append(freqs)
        heat_capacities.append([cv[temperature] for temperature in temperatures])
        print(f"  C_v(300 K) = {cv.get(300.0, float('nan')):.6f} J/(g K)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        trajectory=str(trajectory_path),
        frame_indices=np.asarray(indices, dtype=int),
        temperatures_K=temperatures,
        frequencies_cm1=np.asarray(frequencies),
        cv_J_per_gK=np.asarray(heat_capacities),
        metadata=json.dumps({"dtype": args.dtype, "hops": args.hops, "shadow": args.shadow}),
    )
    print(f"Saved heat-capacity results: {args.output}")


if __name__ == "__main__":
    main()
