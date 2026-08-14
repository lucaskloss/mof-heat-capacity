"""Insert methane molecules into a periodic MOF structure with ASE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms, io
from ase.geometry import get_distances

from ..io import write_lammps_data, write_structure_pdb


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HOST = PROJECT_DIR / "input" / "mof5.pdb"
DEFAULT_METHANE = PROJECT_DIR / "input" / "ch4.gro"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "mof5-pet-mad" / "mof5-md.pdb"
DEFAULT_DATA_OUTPUT = PROJECT_DIR / "output" / "mof5-pet-mad" / "mof5-md.data"


def parse_args() -> argparse.Namespace:
    """Parse structure and insertion options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=DEFAULT_HOST,
                        help="Periodic host structure (PDB, CIF, or GRO).")
    parser.add_argument("--molecule", type=Path, default=DEFAULT_METHANE,
                        help="One-molecule methane structure (GRO or PDB).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Combined PDB output structure (default: matching MD output path).")
    parser.add_argument("--data-output", type=Path, default=DEFAULT_DATA_OUTPUT,
                        help="LAMMPS data output (default: matching MD output path).")
    parser.add_argument("--nmol", type=int, default=1,
                        help="Number of methane molecules to insert (default: 1).")
    parser.add_argument("--try", dest="tries", type=int, default=1000,
                        help="Placement attempts per molecule (default: 1000).")
    parser.add_argument("--min-distance", type=float, default=1.5,
                        help="Minimum periodic atom distance in Angstrom (default: 1.5).")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed (default: 2025).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and prepare without writing the output.")
    return parser.parse_args()


def random_rotation(random: np.random.Generator) -> np.ndarray:
    """Return a uniformly distributed three-dimensional rotation matrix."""
    quaternion = random.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def prepare_molecule(molecule: Atoms, rotation: np.ndarray) -> Atoms:
    """Center and randomly orient one molecule without changing its geometry."""
    prepared = molecule.copy()
    prepared.set_pbc(False)
    prepared.positions -= prepared.get_center_of_mass()
    prepared.positions = prepared.positions @ rotation.T
    return prepared


def has_no_overlap(candidate: Atoms, existing: Atoms, minimum: float) -> bool:
    """Check candidate distances against an existing periodic structure."""
    distances = get_distances(candidate.positions, existing.positions,
                              cell=existing.cell, pbc=existing.pbc)[1]
    return bool(np.min(distances) >= minimum)


def insert_molecules(host: Atoms, molecule: Atoms, count: int, tries: int,
                     minimum: float, seed: int) -> Atoms:
    """Insert randomly oriented molecules into the host periodic cell."""
    random = np.random.default_rng(seed)
    result = host.copy()
    result.set_pbc(host.pbc)

    for molecule_number in range(count):
        for _ in range(tries):
            candidate = prepare_molecule(molecule, random_rotation(random))
            fractional_position = random.random(3)
            candidate.positions += result.cell.cartesian_positions(fractional_position)
            if has_no_overlap(candidate, result, minimum):
                result += candidate
                break
        else:
            raise RuntimeError(
                f"could not place methane molecule {molecule_number + 1} after {tries} attempts; "
                "reduce --nmol or --min-distance"
            )

    return result


def main() -> None:
    """Read structures, insert methane, and write the combined structure."""
    args = parse_args()
    if not args.host.is_file() or not args.molecule.is_file():
        raise FileNotFoundError("host and molecule structure files must exist")
    if args.nmol < 1 or args.tries < 1 or args.min_distance <= 0.0:
        raise ValueError("--nmol, --try, and --min-distance must be positive")
    input_paths = {args.host.resolve(), args.molecule.resolve()}
    if args.output.resolve() in input_paths or args.data_output.resolve() in input_paths:
        raise ValueError("output files must differ from both input structures")

    host = io.read(args.host)
    molecule = io.read(args.molecule)
    if host.cell.volume <= 0.0 or not all(host.pbc):
        raise ValueError("host structure must have a non-zero periodic cell")
    if len(molecule) == 0:
        raise ValueError("molecule structure contains no atoms")

    combined = insert_molecules(host, molecule, args.nmol, args.tries,
                                args.min_distance, args.seed)
    print(f"Prepared {len(combined)} atoms ({args.nmol} methane molecules) in "
          f"{combined.cell.volume:.3f} A^3")
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_structure_pdb(args.output, combined)
        write_lammps_data(args.data_output, combined)
        print(f"Wrote host-plus-CH4 PDB structure: {args.output}")
        print(f"Wrote host-plus-CH4 LAMMPS data: {args.data_output}")


if __name__ == "__main__":
    main()
