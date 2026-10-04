"""Insert CH4, CO2, or H2O molecules into a periodic MOF structure with ASE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms, io
from ase.geometry import get_distances

from ..campaign import GUEST_SYMBOLS, host_path, structure_directory, validate_guest, validate_selection
from ..config import output_root
from ..io import write_lammps_data, write_structure_pdb


PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse structure and insertion options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mof", default="mof5")
    parser.add_argument("--guest", type=str.lower, choices=tuple(GUEST_SYMBOLS), default="ch4")
    parser.add_argument("--host", type=Path,
                        help="Periodic host structure (PDB, CIF, or GRO).")
    parser.add_argument("--molecule", type=Path,
                        help="Single guest molecule structure (GRO or PDB).")
    parser.add_argument("--output", type=Path,
                        help="Combined PDB output structure (default: md/structures/[<mof>/]<N><guest>/manual/seed<seed>).")
    parser.add_argument("--data-output", type=Path,
                        help="LAMMPS data output (default: md/structures/[<mof>/]<N><guest>/manual/seed<seed>).")
    parser.add_argument("--nmol", type=int, default=1,
                        help="Number of guest molecules to insert (default: 1).")
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
                f"could not place guest molecule {molecule_number + 1} after {tries} attempts; "
                "reduce --nmol or --min-distance"
            )

    return result


def main() -> None:
    """Read structures, insert guest molecules, and write the combined structure."""
    args = parse_args()
    validate_selection(args.mof, args.guest)
    args.host = host_path(args.mof, args.host)
    args.molecule = args.molecule or PROJECT_DIR / "input" / f"{args.guest}.gro"
    directory = output_root() / "md" / "structures" / structure_directory(args.nmol, args.mof, args.guest) / "manual" / f"seed{args.seed}"
    args.output = args.output or directory / "structure.pdb"
    args.data_output = args.data_output or directory / "structure.data"
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
    validate_guest(molecule, args.guest)

    combined = insert_molecules(host, molecule, args.nmol, args.tries,
                                args.min_distance, args.seed)
    print(f"Prepared {len(combined)} atoms ({args.nmol} {args.guest.upper()} molecules) in "
          f"{combined.cell.volume:.3f} A^3")
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_structure_pdb(args.output, combined)
        write_lammps_data(args.data_output, combined)
        print(f"Wrote host-plus-{args.guest.upper()} PDB structure: {args.output}")
        print(f"Wrote host-plus-{args.guest.upper()} LAMMPS data: {args.data_output}")


if __name__ == "__main__":
    main()
