"""Run a short PET-MAD/ASE CUDA molecular-dynamics smoke test for MOF-5."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import units
from ase.io import Trajectory
from ase.md import Langevin, MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from metatomic_ase import MetatomicCalculator
import metatomic_ase._neighbors as metatomic_neighbors

from model_utils import ensure_exported_model
from workflow_io import load_mof5_structure

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STRUCTURE = SCRIPT_DIR / "data" / "mof5.cif"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.ckpt"
DEFAULT_EXPORTED_MODEL = SCRIPT_DIR / "models" / "pet-mad-1.5-s_40nn_nostress.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, default=DEFAULT_EXPORTED_MODEL)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--prefix", default="mof5-ase-smoke")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run short NVT dynamics with ASE and the metatomic calculator."""
    args = parse_args()
    if args.steps < 1 or args.temperature <= 0.0 or args.timestep_fs <= 0.0:
        raise ValueError("steps, temperature, and timestep-fs must be positive")

    structure_path = args.structure.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.prefix).name
    output_file = output_dir / f"{prefix}.traj"
    log_file = output_dir / f"{prefix}.log"
    if output_file.exists() and not args.rerun:
        print(f"Reusing existing trajectory: {output_file}")
        return
    if not structure_path.is_file():
        raise FileNotFoundError(f"MOF-5 structure not found: {structure_path}")

    atoms = load_mof5_structure(structure_path)
    model_path = ensure_exported_model(args.checkpoint, args.model)
    # The optional nvalchemi neighbor backend currently fails for this
    # environment; metatomic-ase's vesin fallback supports CPU and CUDA.
    metatomic_neighbors.HAS_NVALCHEMIOPS = False
    atoms.calc = MetatomicCalculator(
        model_path,
        device=args.device,
        check_consistency=True,
    )

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(
        f"Loaded {len(atoms)} atoms; energy={energy:.6f} eV; "
        f"max force={abs(forces).max():.6f} eV/A"
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.01 / units.fs,
    )
    trajectory = Trajectory(output_file, "w", atoms)
    dynamics.attach(trajectory.write, interval=1)
    dynamics.attach(
        MDLogger(dynamics, atoms, log_file, header=True, stress=False, peratom=False), interval=1
    )
    dynamics.run(args.steps)
    trajectory.close()
    print(f"Completed ASE MD. Trajectory: {output_file}")


if __name__ == "__main__":
    main()
