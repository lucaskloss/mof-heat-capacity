"""Relax one periodic structure at fixed cell with the configured MLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE, LBFGSLineSearch

from ..config import load_run_config
from ..models import configure_metatomic_neighbors, ensure_exported_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        help="Structure or trajectory; defaults to the configured input structure",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Frame index for a trajectory input (default: final frame)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--optimizer",
        choices=("fire", "lbfgs-linesearch"),
        default="fire",
        help="Geometry optimizer (default: fire)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-element-subset",
        action="store_true",
        help="Allow an input whose elements are a subset of the configured loaded system",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_path(config) -> Path:
    if config.exported_model.is_file():
        return config.exported_model
    if config.checkpoint is None:
        raise FileNotFoundError(
            f"exported metatomic model not found: {config.exported_model}"
        )
    return ensure_exported_model(config.checkpoint, config.exported_model)


def run_relaxation(args: argparse.Namespace) -> Path:
    from metatomic_ase import MetatomicCalculator
    import metatomic_ase._neighbors as metatomic_neighbors

    config = load_run_config(args.config)
    input_path = (args.input or config.structure).expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"relaxation input not found: {input_path}")
    if args.fmax <= 0.0 or args.steps < 1:
        raise ValueError("--fmax and --steps must be positive")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"relaxed structure exists; use --overwrite: {output_path}"
        )

    read_format = "lammps-data" if input_path.suffix == ".data" else None
    atoms = read(input_path, index=args.index, format=read_format)
    elements = frozenset(atoms.get_chemical_symbols())
    elements_match = config.required_elements is None or elements == config.required_elements
    subset_allowed = (
        args.allow_element_subset
        and config.required_elements is not None
        and elements < config.required_elements
    )
    if not elements_match and not subset_allowed:
        raise ValueError(
            f"expected elements {sorted(config.required_elements)}, "
            f"found {sorted(elements)} in {input_path}"
        )
    configure_metatomic_neighbors(metatomic_neighbors)
    atoms.calc = MetatomicCalculator(
        model_path(config), device=args.device, check_consistency=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_path = output_path.with_suffix(".optimizer.traj")
    log_path = output_path.with_suffix(".relax.log")
    metadata_path = output_path.with_suffix(".relax.json")
    if args.overwrite:
        optimizer_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    initial_energy = float(atoms.get_potential_energy())
    initial_max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    optimizer_class = {
        "fire": FIRE,
        "lbfgs-linesearch": LBFGSLineSearch,
    }[args.optimizer]
    optimizer = optimizer_class(
        atoms, trajectory=str(optimizer_path), logfile=str(log_path)
    )
    converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
    final_energy = float(atoms.get_potential_energy())
    final_forces = np.asarray(atoms.get_forces(), dtype=float)
    finite_result = (
        np.isfinite(atoms.positions).all()
        and np.isfinite(atoms.cell.array).all()
        and np.isfinite(final_energy)
        and np.isfinite(final_forces).all()
    )
    if not finite_result:
        raise RuntimeError(
            "fixed-cell relaxation produced non-finite coordinates, energy, or forces"
        )
    final_max_force = float(np.linalg.norm(final_forces, axis=1).max())
    if not converged or final_max_force > args.fmax:
        raise RuntimeError(
            "fixed-cell relaxation did not converge: "
            f"max force {final_max_force:.6g} eV/A after {optimizer.nsteps} steps"
        )

    # Keep the Hessian input to one unambiguous, optimized frame. The complete
    # optimizer history remains in the adjacent .optimizer.traj file.
    write(output_path, atoms)
    metadata_path.write_text(
        json.dumps(
            {
                "config": str(args.config.expanduser().resolve()),
                "input": str(input_path),
                "input_index": args.index,
                "output": str(output_path),
                "fixed_cell": True,
                "optimizer": f"ASE {optimizer_class.__name__}",
                "steps": optimizer.nsteps,
                "fmax_target_eV_per_A": args.fmax,
                "initial_energy_eV": initial_energy,
                "final_energy_eV": final_energy,
                "initial_max_force_eV_per_A": initial_max_force,
                "final_max_force_eV_per_A": final_max_force,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Relaxed {len(atoms)} atoms in {optimizer.nsteps} steps; "
        f"max force={final_max_force:.6g} eV/A; output={output_path}"
    )
    return output_path


def main() -> None:
    run_relaxation(parse_args())


if __name__ == "__main__":
    main()
