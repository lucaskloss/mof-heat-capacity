"""Compute harmonic heat capacity from selected trajectory frames and a PET-JAX model."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from ..config import RunConfig, find_classical_output_file, load_run_config
from .lammps import read_lammps_thermo


IMAGINARY_THRESHOLD_CM1 = 1.0


def parse_args() -> argparse.Namespace:
    """Parse a run specification and heat-capacity overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--frames", help="Count, 'all', or comma-separated indices")
    parser.add_argument(
        "--frame-indices",
        help="Explicit comma-separated trajectory indices (a single index is allowed)",
    )
    parser.add_argument(
        "--frame-times-ps",
        help=(
            "Comma-separated physical trajectory times in ps. For resumed LAMMPS "
            "runs, these are mapped through the thermo log to the correct raw frames."
        ),
    )
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--start-frame",
        type=int,
        help="First frame eligible for count/all/stride selection",
    )
    parser.add_argument("--temperatures", help="Inclusive start:stop:step range in K")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--hops", type=int)
    parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output archive instead of failing safely",
    )
    return parser.parse_args()


def select_indices(
    spec: str, n_frames: int, stride: int | None, start_frame: int = 0
) -> list[int]:
    """Select trajectory indices from a count, list, or ``all``."""
    if start_frame < 0 or start_frame >= n_frames:
        raise IndexError(f"start frame outside 0..{n_frames - 1}")
    if stride is not None:
        if stride < 1:
            raise ValueError("stride must be positive")
        return list(range(start_frame, n_frames, stride))
    if spec == "all":
        return list(range(start_frame, n_frames))
    if "," in spec:
        indices = [int(item.strip()) for item in spec.split(",")]
    else:
        count = int(spec)
        if count < 1:
            raise ValueError("frames must be positive")
        available = n_frames - start_frame
        indices = np.linspace(
            start_frame, n_frames - 1, min(count, available), dtype=int
        ).tolist()
    if any(index < 0 or index >= n_frames for index in indices):
        raise IndexError(f"frame index outside 0..{n_frames - 1}")
    return list(dict.fromkeys(indices))


def select_explicit_indices(spec: str, n_frames: int) -> list[int]:
    """Parse one or more explicit trajectory indices."""
    indices = [int(item.strip()) for item in spec.split(",")]
    if any(index < 0 or index >= n_frames for index in indices):
        raise IndexError(f"frame index outside 0..{n_frames - 1}")
    return list(dict.fromkeys(indices))


def select_time_indices(
    spec: str,
    times_ps: np.ndarray,
    *,
    production_start_ps: float,
) -> tuple[list[int], np.ndarray]:
    """Map requested physical times to unique saved-frame indices."""
    requested = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not requested:
        raise ValueError("--frame-times-ps must contain at least one time")
    if len(set(requested)) != len(requested):
        raise ValueError("--frame-times-ps must not contain duplicate times")
    if any(time < production_start_ps for time in requested):
        raise ValueError(
            "--frame-times-ps contains a time before the configured production "
            f"start ({production_start_ps:g} ps)"
        )

    times = np.asarray(times_ps, dtype=float)
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) < 0.0):
        raise ValueError("saved trajectory times must be a non-empty ordered series")
    positive_spacing = np.diff(np.unique(times))
    tolerance = (
        max(1e-9, 0.1 * float(np.median(positive_spacing)))
        if len(positive_spacing)
        else 1e-9
    )
    indices = []
    selected_times = []
    for requested_time in requested:
        difference = np.abs(times - requested_time)
        minimum = float(difference.min())
        if minimum > tolerance:
            raise ValueError(
                f"requested time {requested_time:g} ps is not a saved trajectory "
                f"time (nearest difference {minimum:g} ps)"
            )
        # At a resume boundary, prefer the later copy of an otherwise identical time.
        matches = np.flatnonzero(np.isclose(difference, minimum, rtol=0.0, atol=1e-12))
        index = int(matches[-1])
        indices.append(index)
        selected_times.append(float(times[index]))
    if len(set(indices)) != len(indices):
        raise ValueError("requested times map to duplicate trajectory frames")
    return indices, np.asarray(selected_times, dtype=float)


def parse_temperatures(spec: str) -> np.ndarray:
    """Parse an inclusive ``start:stop:step`` temperature range."""
    start, stop, step = (float(value) for value in spec.split(":"))
    if step <= 0.0 or stop < start:
        raise ValueError("temperatures must be start:stop:positive_step")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def read_selected_frames(trajectory_path: Path, indices: list[int]):
    """Stream a trajectory once and retain only requested frame indices."""
    from ase.io import iread

    requested = set(indices)
    selected = {}
    for index, atoms in enumerate(iread(str(trajectory_path), index=":")):
        if index in requested:
            selected[index] = atoms
            if len(selected) == len(requested):
                break
    missing = [index for index in indices if index not in selected]
    if missing:
        raise IndexError(
            "trajectory does not contain requested frame index/indices: "
            + ", ".join(str(index) for index in missing)
        )
    return [selected[index] for index in indices]


def convert_pet_checkpoint(checkpoint: Path, output_dir: Path) -> None:
    """Convert wrapped or bare PET checkpoints with the pinned PET-JAX release."""
    import metatomic.torch  # noqa: F401  (registers ModelMetadata for torch.load)
    import torch

    import petjax.convert as converter

    loaded = torch.load(str(checkpoint), weights_only=False, map_location="cpu")
    if loaded.get("architecture_name") != "pet":
        converter.convert_checkpoint(str(checkpoint), str(output_dir))
        return

    version = loaded.get("model_ckpt_version")
    if version != 11:
        raise ValueError(
            f"unsupported bare PET checkpoint version {version!r}; expected version 11"
        )

    metadata = converter._extract_metadata(loaded)
    params = converter._unflatten(
        converter._convert_state_dict(loaded["best_model_state_dict"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    converter.write_msgpack(output_dir / "model.msgpack", params)
    converter.write_yaml(
        output_dir / "metadata.yaml",
        {
            "config": metadata["config"],
            "energy_scale": metadata["energy_scale"],
            "shifts": metadata["shifts"],
            "species_to_index": metadata["species_to_index"],
        },
    )


def ensure_jax_checkpoint(checkpoint: Path | None, jax_checkpoint: Path | None) -> Path:
    """Convert a PET checkpoint when no PET-JAX directory is available."""
    if checkpoint is not None and checkpoint.is_dir() and (checkpoint / "model.msgpack").is_file():
        return checkpoint
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"PET checkpoint not found: {checkpoint}")
    if jax_checkpoint is None:
        raise ValueError("model.jax_checkpoint is required for a PET checkpoint conversion")

    jax_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock_path = jax_checkpoint.with_name(f".{jax_checkpoint.name}.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata_path = jax_checkpoint / "metadata.yaml"
        if (jax_checkpoint / "model.msgpack").is_file() and metadata_path.is_file() and (
            "energy_scale:" in metadata_path.read_text()
            and "species_to_index:" in metadata_path.read_text()
        ):
            return jax_checkpoint

        if jax_checkpoint.exists():
            print(f"Regenerating incomplete PET-JAX conversion: {jax_checkpoint}")

        print(f"Converting PET checkpoint to PET-JAX: {jax_checkpoint}")
        convert_pet_checkpoint(checkpoint, jax_checkpoint)
    return jax_checkpoint


def compute_frame_hessian(
    atoms,
    model,
    params,
    metadata,
    *,
    dtype: str,
    hops: int,
    chunk_size: int,
    remat: bool,
):
    """Compute one sparse PET Hessian and return its real-atom dense block."""
    import asdex
    import jax

    from sadmof.models.pet import atoms_to_inputs, get_energy_fn
    from sadmof.sparse import get_hessian_fn, sparsity_pattern

    positions, cell, graph = atoms_to_inputs(
        atoms,
        model,
        metadata,
        float_dtype=np.float64 if dtype == "float64" else np.float32,
    )
    pattern_graph = {
        "centers": graph["sel_centers"],
        "others": graph["sel_others"],
        "atomic_numbers": graph["atomic_numbers"],
    }
    pattern = sparsity_pattern(pattern_graph, hops=hops)
    coloring = asdex.hessian_coloring_from_sparsity(pattern, mode="fwd_over_rev")
    energy_fn = get_energy_fn(model, metadata, no_shadow=True)
    hessian_fn = jax.jit(
        get_hessian_fn(energy_fn, coloring, chunk_size=chunk_size, remat=remat)
    )
    hessian = jax.block_until_ready(hessian_fn(params, positions, cell, graph))
    hessian = np.asarray(hessian.todense())
    n_atoms = len(atoms)
    return hessian[:n_atoms, :, :n_atoms, :].reshape(3 * n_atoms, 3 * n_atoms)


def signed_frequencies_from_hessian(
    hessian: np.ndarray,
    masses: np.ndarray,
    *,
    enforce_asr: bool = True,
) -> np.ndarray:
    """Return signed normal-mode frequencies while retaining unstable modes.

    Negative mass-weighted Hessian eigenvalues are represented as negative
    frequencies. SADMOF's compatibility convention maps them to zero, which is
    useful for reproducing published heat capacities but cannot distinguish a
    minimum from a saddle point.
    """
    from ase import units

    masses = np.asarray(masses, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    expected_shape = (3 * len(masses), 3 * len(masses))
    if hessian.shape != expected_shape:
        raise ValueError(
            f"Hessian shape {hessian.shape} does not match {len(masses)} masses"
        )
    hessian = (hessian + hessian.T) / 2.0
    if enforce_asr:
        force_constants = hessian.reshape(len(masses), 3, len(masses), 3)
        force_constants = force_constants - force_constants.mean(
            axis=0, keepdims=True
        )
        force_constants = force_constants - force_constants.mean(
            axis=2, keepdims=True
        )
        hessian = force_constants.reshape(expected_shape)

    inverse_sqrt_mass = np.repeat(masses**-0.5, 3)
    dynamical_matrix = (
        inverse_sqrt_mass[:, None] * hessian * inverse_sqrt_mass[None, :]
    )
    eigenvalues = np.linalg.eigvalsh(dynamical_matrix)
    conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
    return (
        np.sign(eigenvalues)
        * conversion
        * np.sqrt(np.abs(eigenvalues))
        / units.invcm
    )


def run_heat_capacity(config: RunConfig, args: argparse.Namespace) -> Path:
    """Evaluate SADMOF's PET sparse-Hessian path for selected trajectory frames."""
    if config.ad_backend != "pet-jax":
        raise NotImplementedError(
            f"AD backend {config.ad_backend!r} is not implemented; "
            "only PET-JAX/SADMOF currently supports harmonic heat capacity"
        )
    if args.shadow or config.heat_shadow:
        raise ValueError(
            "shadow derivatives require a dense PET Hessian path, not this sparse workflow"
        )

    trajectory_path = args.trajectory
    if trajectory_path is None:
        trajectory_path = find_classical_output_file(
            config, "trajectory.lammpstrj", config.md_trajectory_file
        )
        if trajectory_path is None:
            raise FileNotFoundError(f"trajectory is missing for {config.name}")
    trajectory_path = trajectory_path.expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")

    dtype = args.dtype or config.heat_dtype
    hops = config.heat_hops if args.hops is None else args.hops
    chunk_size = config.heat_chunk_size if args.chunk_size is None else args.chunk_size
    remat = config.heat_remat if args.remat is None else args.remat
    frame_spec = args.frames or config.heat_frames
    temperature_spec = args.temperatures or config.heat_temperatures
    if args.output is not None:
        output_path = args.output
    elif args.frame_times_ps is not None:
        time_tag = "-".join(
            f"{float(item.strip()):g}ps"
            for item in args.frame_times_ps.split(",")
            if item.strip()
        )
        output_path = config.output_dir / f"heat-capacity-times-{time_tag}.npz"
    else:
        output_path = config.output_dir / "heat-capacity.npz"
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"heat-capacity output already exists: {output_path}; "
            "use --overwrite only for an intentional replacement"
        )
    if hops < 0 or chunk_size < 1:
        raise ValueError("hops must be non-negative and chunk-size must be positive")

    import jax
    from ase.io import read
    from sadmof.models.pet import load_pet
    from sadmof.observables import cv_curve

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    selected_times_ps = np.empty(0, dtype=float)
    if args.frame_times_ps is not None:
        if any(
            value is not None
            for value in (
                args.frames,
                args.frame_indices,
                args.stride,
                args.start_frame,
            )
        ):
            raise ValueError(
                "--frame-times-ps cannot be combined with frame index/count options"
            )
        if config.md_driver != "lammps":
            raise ValueError("--frame-times-ps currently requires a LAMMPS trajectory")
        thermo_log = find_classical_output_file(
            config, "md.lammps.log", f"{config.md_prefix}.lammps.log"
        )
        if thermo_log is None:
            raise FileNotFoundError(f"LAMMPS thermo log is missing for {config.name}")
        thermo = read_lammps_thermo(thermo_log)
        indices, selected_times_ps = select_time_indices(
            args.frame_times_ps,
            thermo["time_ps"],
            production_start_ps=(
                config.equilibration_steps * config.timestep_fs / 1000.0
            ),
        )
        selected_frames = read_selected_frames(trajectory_path, indices)
    else:
        frames = read(str(trajectory_path), index=":")
        if args.frame_indices is not None:
            if args.frames is not None or args.stride is not None:
                raise ValueError(
                    "--frame-indices cannot be combined with --frames or --stride"
                )
            indices = select_explicit_indices(args.frame_indices, len(frames))
        else:
            start_frame = (
                config.heat_start_frame if args.start_frame is None else args.start_frame
            )
            indices = select_indices(frame_spec, len(frames), args.stride, start_frame)
        selected_frames = [frames[index] for index in indices]
    temperatures = parse_temperatures(temperature_spec)
    jax_checkpoint = ensure_jax_checkpoint(config.checkpoint, config.jax_checkpoint)
    model, params, metadata = load_pet(jax_checkpoint, dtype=dtype)

    frequencies = []
    heat_capacities = []
    for index, selected_frame in zip(indices, selected_frames, strict=True):
        atoms = selected_frame.copy()
        atoms.calc = None
        print(f"Computing Hessian for frame {index} ({len(atoms)} atoms)")
        hessian = compute_frame_hessian(
            atoms,
            model,
            params,
            metadata,
            dtype=dtype,
            hops=hops,
            chunk_size=chunk_size,
            remat=remat,
        )
        freqs = signed_frequencies_from_hessian(
            hessian, atoms.get_masses(), enforce_asr=True
        )
        cv = cv_curve(
            freqs, temperatures, float(atoms.get_masses().sum())
        )
        frequencies.append(freqs)
        heat_capacities.append([cv[temperature] for temperature in temperatures])
        imaginary_count = int(np.count_nonzero(freqs < -IMAGINARY_THRESHOLD_CM1))
        near_zero_count = int(
            np.count_nonzero(np.abs(freqs) <= IMAGINARY_THRESHOLD_CM1)
        )
        print(
            f"  modes: {imaginary_count} imaginary below "
            f"-{IMAGINARY_THRESHOLD_CM1:g} cm^-1; "
            f"{near_zero_count} within +/-{IMAGINARY_THRESHOLD_CM1:g} cm^-1"
        )
        print(f"  C_v(300 K) = {cv.get(300.0, float('nan')):.6f} J/(g K)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        run_name=config.name,
        structure=str(config.structure),
        trajectory=str(trajectory_path),
        checkpoint=str(jax_checkpoint),
        frame_indices=np.asarray(indices, dtype=int),
        frame_times_ps=selected_times_ps,
        temperatures_K=temperatures,
        frequencies_cm1=np.asarray(frequencies),
        cv_J_per_gK=np.asarray(heat_capacities),
        metadata=json.dumps(
            {
                "ad_backend": config.ad_backend,
                "dtype": dtype,
                "hops": hops,
                "chunk_size": chunk_size,
                "remat": remat,
                "shadow": False,
                "frequency_convention": "signed",
                "acoustic_sum_rule": "force-constant-row-sum",
                "imaginary_threshold_cm1": IMAGINARY_THRESHOLD_CM1,
            }
        ),
    )
    print(f"Saved heat-capacity results: {output_path}")
    return output_path


def main() -> None:
    """Load a TOML specification and calculate harmonic heat capacity."""
    args = parse_args()
    run_heat_capacity(load_run_config(args.config), args)


if __name__ == "__main__":
    main()
